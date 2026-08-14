#!/usr/bin/env python
"""Test mitigations for RSS growth: malloc_trim, clear_program_cache, MALLOC_ARENA_MAX."""
import gc
import os
import sys
import time
import ctypes
import resource

import torch
import ttnn

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import importlib.util
from pathlib import Path
def _find_mgd():
    spec = importlib.util.find_spec("ttnn")
    for name in ["p150_mesh_graph_descriptor.textproto", "p300_mesh_graph_descriptor.textproto"]:
        if spec is not None and spec.submodule_search_locations:
            path = (Path(next(iter(spec.submodule_search_locations)))
                    / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name)
            if path.is_file():
                return str(path)
        for p in sys.path:
            candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
            if candidate.is_file():
                return str(candidate)
        candidate = Path("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages") / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
        if candidate.is_file():
                return str(candidate)
    return None

_mgd = _find_mgd()
if _mgd:
    os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import build_model_config, cross_entropy_loss
from data import Vocab, sample_batch
import yaml

libc = ctypes.CDLL("libc.so.6")

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

def get_heap_kb():
    try:
        class mallinfo2_entry(ctypes.Structure):
            _fields_ = [("arena", ctypes.c_int64), ("ordblks", ctypes.c_int64),
                        ("smblks", ctypes.c_int64), ("hblks", ctypes.c_int64),
                        ("hblkhd", ctypes.c_int64), ("usmblks", ctypes.c_int64),
                        ("fsmblks", ctypes.c_int64), ("uordblks", ctypes.c_int64),
                        ("fordblks", ctypes.c_int64), ("keepcost", ctypes.c_int64)]
        class mallinfo2_struct(ctypes.Structure):
            _fields_ = [("total", mallinfo2_entry)]
        info = mallinfo2_struct()
        libc.mallinfo2(ctypes.byref(info))
        return info.total.uordblks // 1024
    except Exception:
        return 0


def run_test(config_path, steps, micro_batch=8, mitigation="none"):
    """Run model with optional mitigations."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    model = TTWRAPModel(model_config, device)
    vocab = Vocab()
    seq_len = cfg.get("seq_len", 128)
    import random
    rng = random.Random(42)

    # Warmup
    for _ in range(3):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        try: ttnn.deallocate(grad_logits)
        except: pass
        for g in grads.values():
            try: ttnn.deallocate(g)
            except: pass
        try: ttnn.deallocate(logits)
        except: pass
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
    gc.collect()
    ttnn.synchronize_device(device)

    rss0 = get_rss_kb()
    heap0 = get_heap_kb()
    cache0 = device.num_program_cache_entries()
    prev = rss0

    print(f"\n=== {mitigation} ===  (RSS={rss0//1024} MB, heap={heap0//1024} MB, cache={cache0})")

    for step in range(steps):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        try: ttnn.deallocate(grad_logits)
        except: pass
        for g in grads.values():
            try: ttnn.deallocate(g)
            except: pass
        try: ttnn.deallocate(logits)
        except: pass
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
        gc.collect()

        # Apply mitigation
        if mitigation == "malloc_trim":
            libc.malloc_trim(0)
        elif mitigation == "clear_cache+trim":
            if step > 0 and step % 20 == 0:
                device.clear_program_cache()
                libc.malloc_trim(0)
        elif mitigation == "sync+trim":
            ttnn.synchronize_device(device)
            libc.malloc_trim(0)

        if step % 20 == 0 or step == steps - 1:
            rss = get_rss_kb()
            heap = get_heap_kb()
            cache = device.num_program_cache_entries()
            print(f"  step {step:>4}: RSS={rss//1024:>5} MB, delta={rss-prev:>6} KB, "
                  f"heap={heap//1024:>4} MB, cache={cache}")
            prev = rss

    final = get_rss_kb()
    growth = final - rss0
    rate = growth / steps
    print(f"  Total: {growth//1024} MB / {steps} steps = {rate:.1f} KB/step")
    ttnn.close_device(device)
    return rate


if __name__ == "__main__":
    STEPS = 60

    print("=== Testing mitigations for RSS growth ===")

    rate_none = run_test("configs/cell_b_tt.yaml", STEPS, mitigation="none")
    rate_trim = run_test("configs/cell_b_tt.yaml", STEPS, mitigation="malloc_trim")
    rate_clear = run_test("configs/cell_b_tt.yaml", STEPS, mitigation="clear_cache+trim")

    print(f"\n=== SUMMARY ===")
    print(f"  No mitigation:      {rate_none:.1f} KB/step")
    print(f"  malloc_trim:        {rate_trim:.1f} KB/step")
    print(f"  clear_cache+trim:   {rate_clear:.1f} KB/step")
