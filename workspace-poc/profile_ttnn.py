"""
Detailed profiling script for the tt-nn Mamba2 training.

Breaks down the forward and backward passes into sub-components
to identify the exact bottlenecks.

Usage:
    .tt-venv/bin/python profile_ttnn.py --micro_batch 32
"""

import os
# Suppress Metal C++ warnings (e.g. ROW MAJOR tile extraction in ttnn.embedding)
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import argparse
import sys
import time
import torch
import ttnn

from loguru import logger
logger.remove()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_ttnn import TTMambaWorkspaceModel, ModelConfig, TTMamba2Layer, TTGatedResidualLayer
from data import Vocab, sample_batch
import random


def profile_full_step(model, optimizer, input_ids, labels, device, n_warmup=2, n_runs=5):
    """Profile a full training step with sub-component timing."""
    config = model.config
    vocab_size = config.vocab_size

    # Warmup
    for _ in range(n_warmup):
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss_simple(logits, labels, device)
        grads = model.backward(grad_logits)
        optimizer.step(grads, model)

    # Profile
    times = {}
    for run in range(n_runs):
        t0 = time.time()
        logits = model.forward(input_ids)
        t1 = time.time()
        loss_val, grad_logits = cross_entropy_loss_simple(logits, labels, device)
        t2 = time.time()
        grads = model.backward(grad_logits)
        t3 = time.time()
        optimizer.step(grads, model)
        t4 = time.time()

        times.setdefault("forward", []).append(t1 - t0)
        times.setdefault("loss", []).append(t2 - t1)
        times.setdefault("backward", []).append(t3 - t2)
        times.setdefault("optimizer", []).append(t4 - t3)
        times.setdefault("total", []).append(t4 - t0)

    return times, loss_val


def profile_forward_breakdown(model, input_ids, device, n_warmup=2, n_runs=5):
    """Break down the forward pass into per-layer timing."""
    config = model.config

    # Warmup
    for _ in range(n_warmup):
        model.forward(input_ids)

    # Profile per-layer
    layer_times = []
    other_times = {"embedding": [], "final_norm": [], "lm_head": [], "total": []}

    for run in range(n_runs):
        t0 = time.time()

        # Embedding
        t_emb0 = time.time()
        indices = ttnn.from_torch(input_ids.to(torch.int32), dtype=ttnn.uint32,
                                  layout=ttnn.TILE_LAYOUT, device=device)
        x = ttnn.embedding(indices, model.token_emb_weight)
        t_emb1 = time.time()
        other_times["embedding"].append(t_emb1 - t_emb0)

        # Layers
        for i, layer in enumerate(model.layers):
            t_layer0 = time.time()
            x = layer.forward(x)
            t_layer1 = time.time()
            if len(layer_times) <= i:
                layer_times.append([])
            layer_times[i].append(t_layer1 - t_layer0)

        # Final norm
        t_fn0 = time.time()
        x = model.norm.forward(x)
        t_fn1 = time.time()
        other_times["final_norm"].append(t_fn1 - t_fn0)

        # LM head
        t_lh0 = time.time()
        lm_head_w = ttnn.transpose(model.lm_head_weight, 0, 1)
        logits = ttnn.linear(x, lm_head_w)
        t_lh1 = time.time()
        other_times["lm_head"].append(t_lh1 - t_lh0)

        other_times["total"].append(t_lh1 - t0)

    return layer_times, other_times


def profile_backward_breakdown(model, input_ids, device, n_warmup=2, n_runs=5):
    """Break down the backward pass into per-layer and sub-component timing."""
    config = model.config

    # We need to run forward first to populate caches
    for _ in range(n_warmup):
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss_simple(logits, None, device, labels_placeholder=True)
        grads = model.backward(grad_logits)

    # Profile
    layer_times = []
    other_times = {"embedding_bw": [], "final_norm_bw": [], "lm_head_bw": [], "total": []}

    for run in range(n_runs):
        # Forward (not timed - just to populate caches)
        logits = model.forward(input_ids)
        B, T = input_ids.shape
        V = config.vocab_size
        # Create a simple gradient
        grad_logits_host = torch.randn(B, T, V, dtype=torch.bfloat16) * 0.01
        grad_logits = ttnn.from_torch(grad_logits_host, dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=device)

        t0 = time.time()

        # LM head backward
        t_lh0 = time.time()
        x_pre_lm = model._cached_x_pre_lm_head  # need to cache this
        grad_lm_head = ttnn.matmul(
            ttnn.reshape(grad_logits, [B * T, V]).T,
            ttnn.reshape(x_pre_lm, [B * T, config.d_model])
        )
        # Actually, let's just use the model's backward and time the whole thing
        # The model backward does everything
        t_lh1 = time.time()

        # Full backward
        t_bw0 = time.time()
        grads = model.backward(grad_logits)
        t_bw1 = time.time()

        other_times["total"].append(t_bw1 - t_bw0)

    return other_times


def profile_ssd_backward(model, input_ids, device, n_warmup=2, n_runs=5):
    """Profile just the SSD backward (the suspected bottleneck)."""
    config = model.config
    B, T = input_ids.shape
    H = config.n_heads
    d_head = config.d_head
    d_inner = config.d_inner

    # Run forward to populate caches
    for _ in range(n_warmup):
        model.forward(input_ids)

    # Get the first layer's cache
    layer = model.layers[0].layer  # TTMamba2Layer inside TTGatedResidualLayer
    c = layer._cache

    # Simulate the SSD backward
    times = []
    for run in range(n_runs):
        # Re-run forward to populate caches
        model.forward(input_ids)
        c = layer._cache

        t0 = time.time()

        # Reproduce the SSD backward from model_ttnn.py
        x_conv_h = ttnn.to_torch(c["x_conv"]).float().detach().requires_grad_(True)
        dt_proj_w_h = ttnn.to_torch(layer.dt_proj_weight).float().detach().requires_grad_(True)
        dt_proj_b_h = ttnn.to_torch(layer.dt_proj_bias).float().detach().requires_grad_(True)
        B_proj_w_h = ttnn.to_torch(layer.B_proj_weight).float().detach().requires_grad_(True)
        C_proj_w_h = ttnn.to_torch(layer.C_proj_weight).float().detach().requires_grad_(True)
        A_log_h = ttnn.to_torch(layer.A_log).float().detach().requires_grad_(True)
        D_h = ttnn.to_torch(layer.D).float().detach().requires_grad_(True)

        dt_pre = torch.nn.functional.linear(x_conv_h, dt_proj_w_h.T, dt_proj_b_h)
        dt = torch.nn.functional.softplus(dt_pre)
        B_mat = torch.nn.functional.linear(x_conv_h, B_proj_w_h.T)
        C_mat = torch.nn.functional.linear(x_conv_h, C_proj_w_h.T)
        V = x_conv_h.view(B, T, H, d_head)
        V_h = V.permute(0, 2, 1, 3)

        A = torch.exp(A_log_h)
        decay = torch.exp(-dt * A.unsqueeze(0).unsqueeze(0))
        CB = torch.matmul(C_mat, B_mat.transpose(1, 2))
        log_decay = torch.log(decay.clamp(min=1e-8))
        log_decay_h = log_decay.permute(0, 2, 1)
        log_decay_exp = log_decay_h.unsqueeze(-1).expand(B, H, T, T)
        mask_low = torch.tril(torch.ones(T, T), diagonal=-1).bool()
        log_decay_exp = log_decay_exp.masked_fill(~mask_low, 0.0)
        log_L = torch.cumsum(log_decay_exp, dim=-2)
        mask_causal = torch.tril(torch.ones(T, T), diagonal=0).bool()
        log_L = log_L.masked_fill(~mask_causal, float("-inf"))
        L = torch.exp(log_L)
        scores = CB.unsqueeze(1) * L
        Y_ssd = torch.matmul(scores.float(), V_h.float()).to(x_conv_h.dtype)
        Y_ssd = Y_ssd + D_h.view(1, H, 1, 1) * V_h

        # Backward
        grad_Y = torch.randn_like(Y_ssd)
        Y_ssd.backward(grad_Y)

        t1 = time.time()
        times.append(t1 - t0)

    return times


def cross_entropy_loss_simple(logits_tt, labels, device, labels_placeholder=False):
    """Simplified loss for profiling — just returns random gradient."""
    if labels_placeholder:
        B, T, V = ttnn.to_torch(logits_tt).shape
        grad_host = torch.randn(B, T, V, dtype=torch.bfloat16) * 0.01
        grad_tt = ttnn.from_torch(grad_host, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device)
        return 0.0, grad_tt

    logits = ttnn.to_torch(logits_tt).float()
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, :-1].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100
    )
    grad_host = torch.randn(B, T, V, dtype=torch.bfloat16) * 0.01
    grad_tt = ttnn.from_torch(grad_host, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=device)
    return loss.item(), grad_tt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro_batch", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_runs", type=int, default=5)
    args = parser.parse_args()

    B = args.micro_batch
    T = args.seq_len

    device = ttnn.open_device(device_id=0)
    print(f"Device: {device}", flush=True)

    # Full Cell A config
    config = ModelConfig(
        d_model=384, n_layers=14, vocab_size=128,
        d_state=64, d_conv=4, expand=4, n_heads=4,
    )

    model = TTMambaWorkspaceModel(config, device)
    n_params = model.get_num_params()
    print(f"Model: {config.n_layers} layers, d_model={config.d_model}, "
          f"expand={config.expand}, n_heads={config.n_heads}", flush=True)
    print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)
    print(f"Micro-batch: {B}, Seq len: {T}", flush=True)
    print(flush=True)

    vocab = Vocab()
    rng = random.Random(42)
    input_ids, labels, _ = sample_batch(B, T, vocab, depth_range=(2, 8), rng=rng)

    # --- 1. Full step profiling ---
    print("=" * 60, flush=True)
    print("1. Full training step breakdown", flush=True)
    print("=" * 60, flush=True)

    from train_ttnn import TTAdamW
    optimizer = TTAdamW(model.get_params(), lr=6e-4, weight_decay=0.1)

    times, loss_val = profile_full_step(model, optimizer, input_ids, labels, device,
                                        n_warmup=2, n_runs=args.n_runs)
    total_avg = sum(times["total"]) / len(times["total"])
    print(f"  Loss: {loss_val:.4f}", flush=True)
    print(f"  {'Component':>15s}  {'Avg (s)':>10s}  {'%':>6s}  {'Min (s)':>10s}  {'Max (s)':>10s}", flush=True)
    for comp in ["forward", "loss", "backward", "optimizer"]:
        vals = times[comp]
        avg = sum(vals) / len(vals)
        pct = 100 * avg / total_avg
        print(f"  {comp:>15s}  {avg:>10.4f}  {pct:>5.1f}%  {min(vals):>10.4f}  {max(vals):>10.4f}", flush=True)
    print(f"  {'total':>15s}  {total_avg:>10.4f}  {'100.0':>5s}%  {min(times['total']):>10.4f}  {max(times['total']):>10.4f}", flush=True)
    tokens_per_sec = B * T / total_avg
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec", flush=True)
    print(flush=True)

    # --- 2. Forward per-layer breakdown ---
    print("=" * 60, flush=True)
    print("2. Forward pass: per-layer breakdown", flush=True)
    print("=" * 60, flush=True)

    layer_times, other_times = profile_forward_breakdown(model, input_ids, device,
                                                         n_warmup=2, n_runs=args.n_runs)
    total_fwd = sum(other_times["total"]) / len(other_times["total"])
    print(f"  {'Component':>15s}  {'Avg (s)':>10s}  {'%':>6s}", flush=True)
    for comp in ["embedding", "final_norm", "lm_head"]:
        vals = other_times[comp]
        avg = sum(vals) / len(vals)
        pct = 100 * avg / total_fwd
        print(f"  {comp:>15s}  {avg:>10.4f}  {pct:>5.1f}%", flush=True)
    for i, vals in enumerate(layer_times):
        avg = sum(vals) / len(vals)
        pct = 100 * avg / total_fwd
        print(f"  {'layer_'+str(i):>15s}  {avg:>10.4f}  {pct:>5.1f}%", flush=True)
    print(f"  {'total':>15s}  {total_fwd:>10.4f}  {'100.0':>5s}%", flush=True)
    print(flush=True)

    # --- 3. SSD backward profiling (single layer) ---
    print("=" * 60, flush=True)
    print("3. SSD backward: single layer (host-side autograd)", flush=True)
    print("=" * 60, flush=True)

    ssd_times = profile_ssd_backward(model, input_ids, device,
                                     n_warmup=2, n_runs=args.n_runs)
    ssd_avg = sum(ssd_times) / len(ssd_times)
    print(f"  SSD backward (1 layer): {ssd_avg:.4f}s avg", flush=True)
    print(f"  SSD backward (14 layers est): {ssd_avg * 14:.4f}s", flush=True)
    bw_avg = sum(times["backward"]) / len(times["backward"])
    print(f"  Actual total backward: {bw_avg:.4f}s", flush=True)
    print(f"  SSD as % of backward: {100 * ssd_avg * 14 / bw_avg:.1f}%", flush=True)
    print(flush=True)

    # --- 4. Host-device transfer profiling ---
    print("=" * 60, flush=True)
    print("4. Host-device transfer costs", flush=True)
    print("=" * 60, flush=True)

    # Measure transfer times for typical tensor sizes
    d_model = config.d_model
    d_inner = config.d_inner
    d_state = config.d_state
    V = config.vocab_size

    transfer_sizes = {
        "logits (B,T,V)": (B, T, V),
        "x (B,T,d_model)": (B, T, d_model),
        "x_conv (B,T,d_inner)": (B, T, d_inner),
        "decay_matrix (B,H,T,T)": (B, config.n_heads, T, T),
    }

    print(f"  {'Tensor':>30s}  {'Size':>12s}  {'H->D (ms)':>10s}  {'D->H (ms)':>10s}", flush=True)
    for name, shape in transfer_sizes.items():
        n_elements = 1
        for s in shape:
            n_elements *= s
        size_mb = n_elements * 2 / 1e6  # bf16

        # Create tensor on host
        t_host = torch.randn(*shape, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            t_device = ttnn.from_torch(t_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            t_back = ttnn.to_torch(t_device)

        # Measure H->D
        n_transfers = 10
        t0 = time.time()
        for _ in range(n_transfers):
            t_device = ttnn.from_torch(t_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t1 = time.time()
        hd_ms = (t1 - t0) / n_transfers * 1000

        # Measure D->H
        t0 = time.time()
        for _ in range(n_transfers):
            t_back = ttnn.to_torch(t_device)
        t1 = time.time()
        dh_ms = (t1 - t0) / n_transfers * 1000

        print(f"  {name:>30s}  {size_mb:>8.1f} MB  {hd_ms:>10.2f}  {dh_ms:>10.2f}", flush=True)

    print(flush=True)

    # --- 5. Summary ---
    print("=" * 60, flush=True)
    print("5. Summary & bottleneck analysis", flush=True)
    print("=" * 60, flush=True)

    fwd_avg = sum(times["forward"]) / len(times["forward"])
    bw_avg = sum(times["backward"]) / len(times["backward"])
    opt_avg = sum(times["optimizer"]) / len(times["optimizer"])

    print(f"  Forward:  {fwd_avg:.3f}s ({100*fwd_avg/total_avg:.1f}%)", flush=True)
    print(f"  Backward: {bw_avg:.3f}s ({100*bw_avg/total_avg:.1f}%)", flush=True)
    print(f"    - SSD backward (est): {ssd_avg*14:.3f}s ({100*ssd_avg*14/total_avg:.1f}% of total)", flush=True)
    print(f"    - Other backward:     {bw_avg - ssd_avg*14:.3f}s ({100*(bw_avg-ssd_avg*14)/total_avg:.1f}% of total)", flush=True)
    print(f"  Optimizer: {opt_avg:.3f}s ({100*opt_avg/total_avg:.1f}%)", flush=True)
    print(f"  Total:    {total_avg:.3f}s", flush=True)
    print(f"  Throughput: {B*T/total_avg:.0f} tokens/sec", flush=True)
    print(flush=True)

    # Estimate training time
    tokens_per_batch = 250000
    steps = 10000
    time_per_step = tokens_per_batch / (B * T) * total_avg
    total_hours = time_per_step * steps / 3600
    print(f"  Estimated training (250K tokens/step, 10K steps):", flush=True)
    print(f"    Time per step: {time_per_step:.1f}s (with accum={tokens_per_batch/(B*T):.0f})", flush=True)
    print(f"    Total time: {total_hours:.1f} hours", flush=True)

    ttnn.close_device(device)
    print("\nProfiling complete.", flush=True)


if __name__ == "__main__":
    main()
