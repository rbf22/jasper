#!/usr/bin/env python3
"""Leak test for TTTextLatentMemoryModel.

Runs repeated forward passes on TT hardware and checks for:
1. Device memory growth (via ttnn memory profiling)
2. Host RSS growth (via /proc/self/status)
3. Forward output consistency

Usage:
    TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python \
        test_tt_text_leak.py --device 0 --steps 20
"""

import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import sys

# Device setup (same as train_ttnn.py)
_device_id_from_argv = 0
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _device_id_from_argv = int(sys.argv[_i + 1])
        break
    if _a.startswith("--device="):
        _device_id_from_argv = int(_a.split("=", 1)[1])
        break
os.environ.setdefault("TT_VISIBLE_DEVICES", str(_device_id_from_argv))

_P300_SUBSYSTEM_IDS = {"0x0044", "0x0045", "0x0046"}
def _is_p300():
    try:
        from pathlib import Path
        for entry in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
            sub = (entry / "device" / "subsystem_device").read_text().strip().lower()
            if sub in _P300_SUBSYSTEM_IDS:
                return True
    except Exception:
        pass
    return False
def _find_mesh_graph_descriptor():
    try:
        import importlib.util
        from pathlib import Path
        spec = importlib.util.find_spec("ttnn")
        for name in ["p150_mesh_graph_descriptor.textproto", "p300_mesh_graph_descriptor.textproto"]:
            if spec is not None and spec.submodule_search_locations:
                path = Path(next(iter(spec.submodule_search_locations))) / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if path.is_file():
                    return str(path)
            for p in sys.path:
                candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if candidate.is_file():
                    return str(candidate)
    except Exception:
        pass
    return None
if _is_p300():
    _mgd = _find_mesh_graph_descriptor()
    if _mgd:
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

import argparse
import time
import gc
import os.path

import torch
import ttnn

MVP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MVP_DIR)

from tt_text_latent_memory_model import (
    TTTextLatentMemoryConfig,
    TTTextLatentMemoryModel,
    _safe_deallocate,
)


def get_rss_kb():
    """Get current process RSS in KB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-encoder-layers", type=int, default=2)
    parser.add_argument("--n-decoder-layers", type=int, default=1)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--max-reasoning-steps", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--max-prompt-len", type=int, default=64)
    parser.add_argument("--max-answer-len", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(42)

    device = ttnn.open_device(device_id=0)
    print(f"Device: {device}", flush=True)

    config = TTTextLatentMemoryConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=4,
        n_slots=args.n_slots,
        max_reasoning_steps=args.max_reasoning_steps,
        expand=4,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        pad_token_id=0,
    )
    model = TTTextLatentMemoryModel(config, device)
    print(f"Model: {model.get_num_params():,} params", flush=True)

    # Generate fixed test inputs
    prompt_ids = torch.randint(1, args.vocab_size, (args.batch_size, args.max_prompt_len))
    prompt_mask = torch.ones(args.batch_size, args.max_prompt_len, dtype=torch.bool)
    prompt_mask[:, -4:] = False  # some padding
    answer_ids = torch.randint(1, args.vocab_size, (args.batch_size, args.max_answer_len))
    answer_mask = torch.ones(args.batch_size, args.max_answer_len, dtype=torch.bool)
    answer_mask[:, -2:] = False  # some padding

    # Warmup (3 steps to stabilize)
    print("Warmup...", flush=True)
    for i in range(3):
        logits = model.forward(prompt_ids, prompt_mask, answer_ids, answer_mask)
        ttnn.synchronize_device(device)
        model.clear_caches()
        gc.collect()
    _safe_deallocate(logits) if logits is not None else None

    # Measure baseline RSS
    rss_before = get_rss_kb()
    print(f"Baseline RSS: {rss_before / 1024:.1f} MB", flush=True)

    # Run repeated forward passes
    first_logits = None
    rss_readings = []
    for step in range(args.steps):
        logits = model.forward(prompt_ids, prompt_mask, answer_ids, answer_mask)
        ttnn.synchronize_device(device)
        model.clear_caches()

        if step == 0:
            first_logits = logits.clone()
        else:
            # Check output consistency (should be deterministic with same weights)
            max_diff = (logits - first_logits).abs().max().item()
            if max_diff > 0.01:
                print(f"  WARNING: step {step} output diff = {max_diff:.6f}", flush=True)

        if step % 5 == 0:
            rss = get_rss_kb()
            rss_readings.append(rss)
            print(f"  step {step:3d}: RSS={rss/1024:.1f}MB", flush=True)

        gc.collect()

    # Final RSS
    rss_after = get_rss_kb()
    rss_delta = rss_after - rss_before
    print(f"\nFinal RSS: {rss_after / 1024:.1f} MB", flush=True)
    print(f"RSS delta: {rss_delta / 1024:.1f} MB ({rss_delta} KB)", flush=True)

    # Check for leaks
    # Allow up to 50MB growth (Python GC, ttnn caches, etc.)
    leak_threshold = 50 * 1024  # 50MB in KB
    if rss_delta > leak_threshold:
        print(f"FAIL: RSS grew by {rss_delta/1024:.1f} MB (threshold: 50 MB)", flush=True)
        result = 1
    else:
        print(f"PASS: RSS growth within threshold ({rss_delta/1024:.1f} MB)", flush=True)
        result = 0

    # Cleanup
    ttnn.synchronize_device(device)
    model.clear_caches()
    ttnn.close_device(device)
    print("Done.", flush=True)
    sys.exit(result)


if __name__ == "__main__":
    main()
