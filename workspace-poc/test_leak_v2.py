#!/usr/bin/env python
"""Targeted memory leak test for the actual training loop.

Runs N forward+backward passes with the real model and tracks:
  - Python heap growth (gc.get_objects())
  - RSS growth (resource module)
  - Device DRAM (if available)

Usage:
  TT_VISIBLE_DEVICES=0 .tt-venv/bin/python test_leak_v2.py --config configs/cell_b_tt.yaml --steps 100
"""
import argparse
import gc
import os
import resource
import sys
import time
import weakref

import torch
import ttnn

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import build_model_config, cross_entropy_loss
from data import Vocab, sample_batch

def get_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def count_ttnn_tensors():
    """Count live ttnn.Tensor wrapper objects via gc."""
    count = 0
    for obj in gc.get_objects():
        if type(obj).__name__ == 'Tensor' and 'ttnn' in type(obj).__module__:
            count += 1
    return count

def get_python_heap_kb():
    """Estimate Python heap size via gc."""
    total = 0
    for obj in gc.get_objects():
        total += sys.getsizeof(obj, 0)
    return total // 1024

def run_leak_test(config_path, steps, micro_batch=4, accum_steps=1):
    with open(config_path) as f:
        import yaml
        cfg = yaml.safe_load(f)

    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    print(f"Config: {config_path}")
    print(f"micro_batch={micro_batch}, accum_steps={accum_steps}")
    print(f"Model: {model_config.n_layers} layers, d_model={model_config.d_model}")
    print()

    model = TTWRAPModel(model_config, device)
    vocab = Vocab()
    seq_len = cfg.get("seq_len", 128)

    import random
    rng = random.Random(42)

    # Warmup - first call compiles kernels
    print("Warmup (3 steps)...")
    for i in range(3):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        model.clear_caches()
        del logits, grad_logits, input_ids, labels
    gc.collect()
    ttnn.synchronize_device(device)

    rss_after_warmup = get_rss_kb()
    heap_after_warmup = get_python_heap_kb()
    tensor_count_after_warmup = count_ttnn_tensors()
    print(f"After warmup: RSS={rss_after_warmup} KB, heap={heap_after_warmup} KB, tensors={tensor_count_after_warmup}")
    print()

    # Main test loop
    print(f"Running {steps} forward+backward passes...")
    print(f"{'Step':>6} {'RSS(KB)':>12} {'dRSS':>10} {'Heap(KB)':>12} {'dHeap':>10} {'Tensors':>10} {'dTens':>8} {'Time':>8}")

    prev_rss = rss_after_warmup
    prev_heap = heap_after_warmup
    prev_tensors = tensor_count_after_warmup

    for step in range(steps):
        t0 = time.time()

        # Forward
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)

        # Loss
        loss_val, grad_logits = cross_entropy_loss(logits, labels)

        # Backward
        grads = model.backward(grad_logits)

        # Cleanup (same as training loop)
        ttnn.synchronize_device(device)
        # Deallocate logits and grad_logits
        try: ttnn.deallocate(logits)
        except: pass
        try: ttnn.deallocate(grad_logits)
        except: pass

        # Deallocate grads
        for g in grads.values():
            try: ttnn.deallocate(g)
            except: pass

        # Clear caches
        ttnn.synchronize_device(device)
        model.clear_caches()

        del logits, grad_logits, grads, input_ids, labels
        gc.collect()

        t1 = time.time()

        if step % 10 == 0 or step == steps - 1:
            rss = get_rss_kb()
            heap = get_python_heap_kb()
            tensors = count_ttnn_tensors()
            drss = rss - prev_rss
            dheap = heap - prev_heap
            dtens = tensors - prev_tensors
            print(f"{step:>6} {rss:>12} {drss:>10} {heap:>12} {dheap:>10} {tensors:>10} {dtens:>8} {t1-t0:>7.2f}s")
            prev_rss = rss
            prev_heap = heap
            prev_tensors = tensors

    # Final summary
    final_rss = get_rss_kb()
    total_growth = final_rss - rss_after_warmup
    per_step = total_growth / steps
    print()
    print(f"Total RSS growth: {total_growth} KB over {steps} steps")
    print(f"Per-step leak: {per_step:.1f} KB/step ({per_step*1024:.0f} B/step)")
    print(f"Per-step leak rate: {per_step/micro_batch:.1f} KB/micro-batch")

    ttnn.close_device(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cell_b_tt.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--micro_batch", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=1)
    args = parser.parse_args()

    run_leak_test(args.config, args.steps, args.micro_batch, args.accum_steps)
