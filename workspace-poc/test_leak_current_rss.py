#!/usr/bin/env python
"""Test leak with accurate current RSS (not peak)."""
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

def get_current_rss_kb():
    """Get current RSS from /proc/self/status (not peak)."""
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

def get_vm_size_kb():
    """Get current virtual memory size."""
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmSize:'):
                return int(line.split()[1])
    return 0

def run_test(config_path, steps, micro_batch=8):
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
    for i in range(3):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
    gc.collect()
    ttnn.synchronize_device(device)

    rss_base = get_current_rss_kb()
    vms_base = get_vm_size_kb()
    print(f"{'Step':>6} {'RSS(KB)':>12} {'dRSS':>10} {'VMS(KB)':>12} {'dVMS':>10}")
    prev_rss = rss_base
    prev_vms = vms_base

    for step in range(steps):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
        gc.collect()

        if step % 10 == 0 or step == steps - 1:
            rss = get_current_rss_kb()
            vms = get_vm_size_kb()
            print(f"{step:>6} {rss:>12} {rss - prev_rss:>10} {vms:>12} {vms - prev_vms:>10}")
            prev_rss = rss
            prev_vms = vms

    final_rss = get_current_rss_kb()
    print(f"\nCurrent RSS growth: {final_rss - rss_base} KB over {steps} steps ({(final_rss - rss_base)/steps:.1f} KB/step)")
    ttnn.close_device(device)

if __name__ == "__main__":
    run_test("configs/cell_b_tt.yaml", 100, micro_batch=8)
