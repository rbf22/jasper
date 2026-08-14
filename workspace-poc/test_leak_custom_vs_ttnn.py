#!/usr/bin/env python
"""Test if custom kernels (generic_op) are the source of the mmap leak.
Compares: custom kernels vs equivalent ttnn ops."""
import gc, os, sys, time
import torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import build_model_config, cross_entropy_loss
from data import Vocab, sample_batch
import yaml

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

def run_test(config_path, steps, micro_batch=8, use_custom_kernels=True):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    # Monkey-patch the custom kernels if needed
    if not use_custom_kernels:
        # Replace _fused_scale_decay with regular ttnn ops
        def replacement_scale_decay(scores_raw, D_decay, scale, B, H, T, device):
            # scores = scores_raw * scale * D_decay
            scaled = ttnn.mul(scores_raw, scale)
            result = ttnn.mul(scaled, D_decay)
            _safe_deallocate(scaled)
            return result
        
        # Replace _fused_gate_backward with regular ttnn ops
        def replacement_gate_backward(grad_out_gated, gate, out_flat, B, T, D, device):
            # grad_out_flat = grad_out_gated * gate
            # grad_g = sum(grad_out_gated * out_flat)
            grad_out_flat = ttnn.mul(grad_out_gated, gate)
            # For grad_g, we need sum over all elements of grad_out_gated * out_flat
            product = ttnn.mul(grad_out_gated, out_flat)
            grad_g = ttnn.sum(product, dim=[-2, -1])  # sum all
            _safe_deallocate(product)
            return grad_out_flat, grad_g

        TTRetentionLayer = None
        # Find the retention layer class and patch it
        import model_ttnn
        for name in dir(model_ttnn):
            obj = getattr(model_ttnn, name)
            if hasattr(obj, '_fused_scale_decay'):
                obj._fused_scale_decay = staticmethod(replacement_scale_decay)
                print(f"  Patched {name}._fused_scale_decay")
            if hasattr(obj, '_fused_gate_backward'):
                obj._fused_gate_backward = staticmethod(replacement_gate_backward)
                print(f"  Patched {name}._fused_gate_backward")

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
        _safe_deallocate(grad_logits)
        for g in grads.values(): _safe_deallocate(g)
        _safe_deallocate(logits)
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
    gc.collect()

    rss0 = get_rss_kb()
    mode = "custom kernels" if use_custom_kernels else "ttnn ops only"
    print(f"\n{mode}: baseline RSS = {rss0//1024} MB")
    prev = rss0

    for step in range(steps):
        input_ids, labels, _ = sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values(): _safe_deallocate(g)
        _safe_deallocate(logits)
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
        gc.collect()

        if step % 20 == 0 or step == steps - 1:
            rss = get_rss_kb()
            print(f"  {mode:>16} step {step:>4}: RSS={rss//1024:>5} MB, delta={rss-prev:>6} KB")
            prev = rss

    final = get_rss_kb()
    growth = final - rss0
    print(f"  {mode}: total growth {growth//1024} MB / {steps} steps = {growth/steps:.1f} KB/step")
    ttnn.close_device(device)
    return growth / steps

if __name__ == "__main__":
    print("=== Test 1: Custom kernels (baseline) ===")
    custom_rate = run_test("configs/cell_b_tt.yaml", 100, micro_batch=8, use_custom_kernels=True)
    
    print("\n=== Test 2: Regular ttnn ops (no custom kernels) ===")
    ttnn_rate = run_test("configs/cell_b_tt.yaml", 100, micro_batch=8, use_custom_kernels=False)
    
    print(f"\nSummary: custom={custom_rate:.1f} KB/step, ttnn={ttnn_rate:.1f} KB/step")
    print(f"Custom kernel leak: {custom_rate - ttnn_rate:.1f} KB/step")
