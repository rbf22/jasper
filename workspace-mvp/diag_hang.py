#!/usr/bin/env python
"""Diagnostic script to investigate device hang in TT-NN native training.

Hypothesis: The hang is caused by variable sequence lengths. Each new
(prompt_len, answer_len) combination triggers new kernel compilation in
TT-NN's program cache. After ~100 unique shapes, the device hangs during
compilation.

This script runs training with FIXED sequence lengths (all batches padded
to max_prompt_len=64, max_answer_len=16). If training completes 150 steps
without hanging, the root cause is variable sequence lengths.

Usage:
    TT_VISIBLE_DEVICES=0 TT_METAL_HOME=/home/rfenwick/Documents/tt-metal-src \
    TT_METAL_LOGGER_LEVEL=ERROR /home/rfenwick/Documents/jasper/.tt-venv/bin/python -u diag_hang.py
"""

import argparse
import ctypes
import math
import os
import random
import resource
import time
import sys

# Force glibc to return freed memory to the OS
_libc = ctypes.CDLL("libc.so.6")
def malloc_trim():
    _libc.malloc_trim(0)

# Set env before importing ttnn
os.environ.setdefault("TT_METAL_HOME", "/home/rfenwick/Documents/tt-metal-src")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

# P300 fabric mesh graph descriptor setup (same as train_native_tt.py)
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

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tt_text_latent_memory_model import (
    TTTextLatentMemoryConfig,
    TTTextLatentMemoryModel,
    to_device,
    from_device,
    _safe_deallocate,
)
from challenge_data import ChallengeDataset


# ---------------------------------------------------------------------------
# Fixed-length batch sampler
# ---------------------------------------------------------------------------

def sample_fixed_batch(dataset, batch_size, max_prompt_len, max_answer_len, rng):
    """Sample a batch with FIXED sequence lengths — always pad to max_prompt_len
    and max_answer_len, regardless of actual content length.

    This ensures every batch has the same tensor shapes, so no new kernel
    compilations are triggered after the first step.
    """
    examples = dataset.train_examples
    batch = rng.sample(examples, min(batch_size, len(examples)))

    pad_id = dataset.pad_id

    # Always pad to fixed max lengths
    prompt_ids = torch.full((len(batch), max_prompt_len), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros(len(batch), max_prompt_len, dtype=torch.bool)
    answer_full = torch.full((len(batch), max_answer_len + 1), pad_id, dtype=torch.long)
    answer_mask = torch.zeros(len(batch), max_answer_len + 1, dtype=torch.bool)

    for i, (p_ids, a_ids) in enumerate(batch):
        # Truncate to max lengths
        p_trim = p_ids[:max_prompt_len]
        a_trim = a_ids[:max_answer_len + 1]

        prompt_ids[i, :len(p_trim)] = torch.tensor(p_trim)
        prompt_mask[i, :len(p_trim)] = True
        answer_full[i, :len(a_trim)] = torch.tensor(a_trim)
        answer_mask[i, :len(a_trim)] = True

    # decoder_input = answer_full[:, :-1]  (BOS + tokens except last)
    # answer_targets = answer_full[:, 1:]  (tokens except BOS)
    decoder_input = answer_full[:, :-1]
    answer_targets = answer_full[:, 1:]
    ans_mask = answer_mask[:, 1:]

    return prompt_ids, prompt_mask, decoder_input, answer_targets, ans_mask


# ---------------------------------------------------------------------------
# Variable-length batch sampler (for comparison)
# ---------------------------------------------------------------------------

def sample_variable_batch(dataset, batch_size, rng):
    """Sample a batch with VARIABLE sequence lengths — pads to max length
    within the batch only (same as ChallengeDataset.sample_batch).

    This is the original behavior that causes the hang.
    """
    return dataset.sample_batch(batch_size, "train", rng)


# ---------------------------------------------------------------------------
# Device memory tracking
# ---------------------------------------------------------------------------

def get_rss_mb():
    """Get current RSS in MB."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0


def get_device_memory(device):
    """Get device memory view. Returns dict with bank info or None."""
    try:
        mem_view = ttnn.get_memory_view(device)
        # mem_view is a dict-like with per-bank info
        # Try to extract total/free per bank
        result = {}
        if hasattr(mem_view, 'items'):
            for bank_id, bank_info in mem_view.items():
                if hasattr(bank_info, 'total_free_bytes'):
                    result[bank_id] = {
                        'total': bank_info.total_bytes_per_bank,
                        'free': bank_info.total_free_bytes,
                        'used': bank_info.total_bytes_per_bank - bank_info.total_free_bytes,
                    }
                elif isinstance(bank_info, dict):
                    result[bank_id] = {
                        'total': bank_info.get('total_bytes_per_bank', 0),
                        'free': bank_info.get('total_free_bytes', 0),
                    }
        return result
    except Exception as e:
        return {"error": str(e)}


def format_device_mem(mem):
    """Format device memory dict for printing."""
    if not mem:
        return "N/A"
    if "error" in mem:
        return f"error: {mem['error']}"
    total_all = 0
    free_all = 0
    for bank_id, info in mem.items():
        total_all += info.get('total', 0)
        free_all += info.get('free', 0)
    used_all = total_all - free_all
    return f"used={used_all / 1e6:.1f}MB free={free_all / 1e6:.1f}MB total={total_all / 1e6:.1f}MB"


# ---------------------------------------------------------------------------
# Cross-entropy loss (same as train_native_tt.py)
# ---------------------------------------------------------------------------

def cross_entropy_loss_and_grad(logits, targets, mask, device, dtype=ttnn.bfloat16):
    """Compute cross-entropy loss and gradient on host, then move grad to device."""
    logits_torch = ttnn.to_torch(logits)
    logits_f = logits_torch.float()
    del logits_torch

    log_probs = torch.log_softmax(logits_f, dim=-1)
    del logits_f

    targets_long = targets.long()
    gathered = log_probs.gather(2, targets_long.unsqueeze(-1)).squeeze(-1)
    per_token_loss = -gathered
    del gathered

    mask_f = mask.float()
    n_valid = mask_f.sum().clamp_min(1)
    loss_val = (per_token_loss * mask_f).sum().item() / n_valid.item()
    del per_token_loss

    probs = torch.exp(log_probs)
    del log_probs

    probs.scatter_(2, targets_long.unsqueeze(-1),
                   probs.gather(2, targets_long.unsqueeze(-1)) - 1.0)
    grad = probs
    grad = grad * mask_f.unsqueeze(-1) / n_valid
    del mask_f, n_valid, probs

    grad_tt = to_device(grad, device, dtype=dtype)
    del grad

    return loss_val, grad_tt


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnostic for TT-NN device hang")
    parser.add_argument("--steps", type=int, default=150,
                        help="Number of training steps (default 150)")
    parser.add_argument("--fixed_length", action="store_true", default=True,
                        help="Use fixed-length batches (default True)")
    parser.add_argument("--variable_length", action="store_true", default=False,
                        help="Use variable-length batches (for comparison)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_slots", type=int, default=8)
    parser.add_argument("--max_reasoning_steps", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--max_prompt_len", type=int, default=64)
    parser.add_argument("--max_answer_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    # Determine mode
    use_fixed = args.fixed_length and not args.variable_length
    mode_str = "FIXED" if use_fixed else "VARIABLE"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = ttnn.bfloat16 if args.precision == "bf16" else ttnn.float32

    print(f"=" * 70, flush=True)
    print(f"DIAGNOSTIC: Device hang investigation", flush=True)
    print(f"Mode: {mode_str} sequence lengths", flush=True)
    print(f"Steps: {args.steps}, batch_size: {args.batch_size}", flush=True)
    print(f"d_model={args.d_model}, enc_layers={args.n_encoder_layers}, "
          f"dec_layers={args.n_decoder_layers}", flush=True)
    print(f"max_prompt_len={args.max_prompt_len}, max_answer_len={args.max_answer_len}", flush=True)
    print(f"=" * 70, flush=True)

    # Create device
    print(f"\nOpening device {args.device}...", flush=True)
    device = ttnn.open_device(device_id=args.device)

    # Create config
    config = TTTextLatentMemoryConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        n_slots=args.n_slots,
        max_reasoning_steps=args.max_reasoning_steps,
        expand=args.expand,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # Create model
    print("Creating TT model...", flush=True)
    model = TTTextLatentMemoryModel(config, device, dtype=dtype)
    n_params = model.get_num_params()
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M) precision={args.precision}", flush=True)

    # Create dataset
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, "tiny_challenges_train.txt")
    valid_path = os.path.join(data_dir, "tiny_challenges_valid.txt")
    dataset = ChallengeDataset(
        train_path=train_path,
        valid_path=valid_path,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # Create optimizer (inline simplified AdamW from train_native_tt.py)
    # We import the full TTAdamW to avoid reimplementing
    from train_native_tt import TTAdamW, clip_grad_norm
    params = model.get_params()
    optimizer = TTAdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW lr={args.lr} wd={args.weight_decay}", flush=True)

    # LR schedule
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Track unique shapes seen
    unique_shapes = set()
    initial_rss = get_rss_mb()
    initial_dev_mem = get_device_memory(device)

    print(f"\nInitial RSS: {initial_rss:.1f} MB", flush=True)
    print(f"Initial device memory: {format_device_mem(initial_dev_mem)}", flush=True)
    print(f"\nStarting training for {args.steps} steps...", flush=True)
    print(f"{'step':>6} {'loss':>8} {'grad':>8} {'lr':>10} {'time':>7} "
          f"{'rss_mb':>10} {'dev_mem':>30} {'shapes':>7}", flush=True)
    print("-" * 100, flush=True)

    rng = random.Random(args.seed)
    start_time = time.time()
    hang_detected = False

    for step in range(args.steps):
        step_start = time.time()
        lr = get_lr(step)
        optimizer.set_lr(lr)

        # Sample batch
        if use_fixed:
            prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
                sample_fixed_batch(dataset, args.batch_size,
                                   args.max_prompt_len, args.max_answer_len, rng)
        else:
            prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
                sample_variable_batch(dataset, args.batch_size, rng)

        # Track unique shapes
        shape_key = (prompt_ids.shape[1], dec_input.shape[1])
        is_new_shape = shape_key not in unique_shapes
        unique_shapes.add(shape_key)

        if is_new_shape:
            print(f"  [NEW SHAPE] step={step} prompt_len={shape_key[0]} "
                  f"answer_len={shape_key[1]} total_unique={len(unique_shapes)}",
                  flush=True)

        # Forward pass
        logits_tt = model.forward_train(prompt_ids, prompt_mask, dec_input, ans_mask)

        # Loss + gradient
        loss_val, grad_logits = cross_entropy_loss_and_grad(
            logits_tt, ans_targets, ans_mask, device, dtype=dtype
        )
        _safe_deallocate(logits_tt)

        # Backward pass
        grads = model.backward(grad_logits)
        _safe_deallocate(grad_logits)

        # Gradient clipping
        grad_norm = clip_grad_norm(grads, args.grad_clip)

        # Optimizer step
        optimizer.step(grads, model)

        # Deallocate gradients
        for name, g in grads.items():
            _safe_deallocate(g)

        # Clear caches
        model.clear_caches()

        # malloc trim every 10 steps
        if step % 10 == 0:
            malloc_trim()

        # Check for hang via step time
        step_time = time.time() - step_start
        if step_time > 60:
            print(f"\n*** WARNING: Step {step} took {step_time:.1f}s — possible hang ***",
                  flush=True)

        # Log every 10 steps
        if step % 10 == 0 or step < 5 or is_new_shape:
            elapsed = time.time() - start_time
            rss = get_rss_mb()
            dev_mem = get_device_memory(device)
            print(
                f"{step:6d} {loss_val:8.4f} {grad_norm:8.3f} {lr:10.2e} "
                f"{elapsed:7.0f}s {rss:10.1f} {format_device_mem(dev_mem):>30} "
                f"{len(unique_shapes):7d}",
                flush=True,
            )

        # Check for NaN/Inf
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"ERROR: loss became {loss_val} at step {step}", flush=True)
            break

        if grad_norm > 1e4:
            print(f"WARNING: gradient norm {grad_norm} is very large at step {step}",
                  flush=True)

    # Final report
    elapsed = time.time() - start_time
    final_rss = get_rss_mb()
    final_dev_mem = get_device_memory(device)

    print("\n" + "=" * 70, flush=True)
    print(f"DIAGNOSTIC COMPLETE", flush=True)
    print(f"Mode: {mode_str} sequence lengths", flush=True)
    print(f"Steps completed: {step + 1}/{args.steps}", flush=True)
    print(f"Total time: {elapsed:.0f}s ({elapsed / max(step + 1, 1):.2f}s/step)", flush=True)
    print(f"Unique shapes seen: {len(unique_shapes)}", flush=True)
    print(f"RSS: {initial_rss:.1f} MB -> {final_rss:.1f} MB (delta: {final_rss - initial_rss:.1f} MB)",
          flush=True)
    print(f"Device memory: {format_device_mem(initial_dev_mem)} -> {format_device_mem(final_dev_mem)}",
          flush=True)

    if step + 1 >= args.steps:
        print(f"\nRESULT: Training completed all {args.steps} steps WITHOUT hanging.", flush=True)
        if use_fixed:
            print("  => Fixed-length sequences prevent the hang.", flush=True)
            print("  => Root cause is likely variable sequence lengths triggering", flush=True)
            print("     excessive kernel compilations in the TT-NN program cache.", flush=True)
        else:
            print("  => Variable-length sequences also completed without hanging.", flush=True)
            print("  => The hang may require more steps or different conditions.", flush=True)
    else:
        print(f"\nRESULT: Training stopped early at step {step + 1}.", flush=True)
        if hang_detected:
            print("  => Hang detected during training.", flush=True)

    print("=" * 70, flush=True)

    # Close device
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
