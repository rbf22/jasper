#!/usr/bin/env python
"""Test if force=True deallocation fixes the leak."""
import gc
import os
import resource
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import build_model_config, cross_entropy_loss
from data import Vocab, sample_batch
import yaml

def get_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def force_dealloc(tensor):
    if tensor is None:
        return
    try:
        ttnn.deallocate(tensor, force=True)
    except Exception:
        pass

def run_test(config_path, steps, micro_batch=8, use_force=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    model = TTWRAPModel(model_config, device)
    vocab = Vocab()
    seq_len = cfg.get("seq_len", 128)
    import random
    rng = random.Random(42)

    dealloc = force_dealloc if use_force else lambda t: None  # we'll inline the logic

    # Warmup
    for i in range(3):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        if use_force:
            force_dealloc(logits)
            force_dealloc(grad_logits)
            for g in grads.values():
                force_dealloc(g)
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
    gc.collect()
    ttnn.synchronize_device(device)

    rss_base = get_rss_kb()
    print(f"{'Mode':>8} {'Step':>6} {'RSS(KB)':>12} {'dRSS':>10}")
    prev = rss_base

    for step in range(steps):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)

        if use_force:
            force_dealloc(logits)
            force_dealloc(grad_logits)
            for g in grads.values():
                force_dealloc(g)

        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
        gc.collect()

        if step % 20 == 0 or step == steps - 1:
            rss = get_rss_kb()
            print(f"{'FORCE' if use_force else 'NORMAL':>8} {step:>6} {rss:>12} {rss - prev:>10}")
            prev = rss

    final = get_rss_kb()
    print(f"\n{'FORCE' if use_force else 'NORMAL'}: total growth {final - rss_base} KB over {steps} steps ({(final - rss_base)/steps:.1f} KB/step)")
    ttnn.close_device(device)

if __name__ == "__main__":
    print("=== NORMAL (force=False) ===")
    run_test("configs/cell_b_tt.yaml", 60, micro_batch=8, use_force=False)
    print("\n=== FORCE=True ===")
    run_test("configs/cell_b_tt.yaml", 60, micro_batch=8, use_force=True)
