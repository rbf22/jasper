"""Standalone test for fused gate backward kernel."""
import os, sys
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import ModelConfig, TTRetentionLayer

print("Opening device...", flush=True)
device = ttnn.open_device(device_id=0)
print("Device opened!", flush=True)

try:
    config = ModelConfig(d_model=64, n_heads=2)
    layer = TTRetentionLayer(config, device, use_fused_rope=True)

    B, T, D = 2, 32, 64

    # Create inputs
    grad_out_gated_t = torch.randn(B, T, D, dtype=torch.bfloat16) * 0.3
    g_t = torch.randn(B, T, D, dtype=torch.bfloat16) * 0.3
    out_flat_t = torch.randn(B, T, D, dtype=torch.bfloat16) * 0.3

    grad_out_gated = ttnn.from_torch(grad_out_gated_t, dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.from_torch(g_t, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    out_flat = ttnn.from_torch(out_flat_t, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)

    # Run fused kernel
    print("Running fused gate backward...", flush=True)
    grad_out_flat, grad_g = TTRetentionLayer._fused_gate_backward(
        grad_out_gated, gate, out_flat, B, T, D, device)
    ttnn.synchronize_device(device)

    gof_host = ttnn.to_torch(grad_out_flat).float()
    gg_host = ttnn.to_torch(grad_g).float()

    # Reference computation
    gog = grad_out_gated_t.float()
    g = g_t.float()
    of = out_flat_t.float()
    ref_gof = gog * g
    sig_prime = g * (1 - g)
    ref_gg = gog * of * sig_prime

    d_gof = (gof_host - ref_gof).abs().max().item()
    d_gg = (gg_host - ref_gg).abs().max().item()
    rel_gof = d_gof / ref_gof.abs().max().item()
    rel_gg = d_gg / ref_gg.abs().max().item()
    print(f"  grad_out_flat: max diff={d_gof:.6f}, rel err={rel_gof:.6f}", flush=True)
    print(f"  grad_g:        max diff={d_gg:.6f}, rel err={rel_gg:.6f}", flush=True)
    print("  PASS" if rel_gof < 0.02 and rel_gg < 0.02 else "  FAIL", flush=True)

finally:
    ttnn.close_device(device)
