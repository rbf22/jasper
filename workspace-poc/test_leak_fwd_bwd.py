#!/usr/bin/env python
"""Test forward-only vs forward+backward leak."""
import gc
import os
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
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

def run_test(config_path, steps, micro_batch=8, do_backward=True):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    model = TTWRAPModel(model_config, device)
    vocab = Vocab()
    seq_len = cfg.get("seq_len", 128)
    import random
    rng = random.Random(42)

    mode = "fwd+bwd" if do_backward else "fwd-only"

    # Warmup
    for i in range(3):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        if do_backward:
            loss_val, grad_logits = cross_entropy_loss(logits, labels)
            grads = model.backward(grad_logits)
            ttnn.synchronize_device(device)
            try: ttnn.deallocate(grad_logits)
            except: pass
            for g in grads.values():
                try: ttnn.deallocate(g)
                except: pass
            del grad_logits, grads
        ttnn.synchronize_device(device)
        try: ttnn.deallocate(logits)
        except: pass
        model.clear_caches()
        del logits, input_ids, labels
    gc.collect()
    ttnn.synchronize_device(device)

    base = get_rss_kb()
    print(f"\n{mode}: base RSS = {base} KB")
    prev = base

    for step in range(steps):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        if do_backward:
            loss_val, grad_logits = cross_entropy_loss(logits, labels)
            grads = model.backward(grad_logits)
            ttnn.synchronize_device(device)
            try: ttnn.deallocate(grad_logits)
            except: pass
            for g in grads.values():
                try: ttnn.deallocate(g)
                except: pass
            del grad_logits, grads
        ttnn.synchronize_device(device)
        try: ttnn.deallocate(logits)
        except: pass
        model.clear_caches()
        del logits, input_ids, labels
        gc.collect()

        if step % 20 == 0 or step == steps - 1:
            rss = get_rss_kb()
            print(f"  {mode:>10} step {step:>4}: RSS={rss:>8} KB, delta={rss - prev:>6} KB")
            prev = rss

    final = get_rss_kb()
    growth = final - base
    print(f"  {mode}: total growth {growth} KB / {steps} steps = {growth/steps:.1f} KB/step")
    ttnn.close_device(device)
    return growth / steps

if __name__ == "__main__":
    print("=== Cell B: Forward-only vs Forward+Backward ===")
    fwd_rate = run_test("configs/cell_b_tt.yaml", 60, micro_batch=8, do_backward=False)
    bwd_rate = run_test("configs/cell_b_tt.yaml", 60, micro_batch=8, do_backward=True)
    print(f"\nSummary: fwd-only={fwd_rate:.1f} KB/step, fwd+bwd={bwd_rate:.1f} KB/step, backward leak={bwd_rate - fwd_rate:.1f} KB/step")
