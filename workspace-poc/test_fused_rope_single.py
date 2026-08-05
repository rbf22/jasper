"""Single-pass full RoPE kernel test."""
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
    layer._init_rope(32, device)

    B, H, T, d_h = 2, 2, 32, 32
    d_half = d_h // 2

    x_torch = torch.randn(B, H, T, d_h, dtype=torch.bfloat16) * 0.3
    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x1 = ttnn.slice(x, [0, 0, 0, 0], [B, H, T, d_half])
    x2 = ttnn.slice(x, [0, 0, 0, d_half], [B, H, T, d_h])

    # Single-pass fused kernel: rot1 = x1*cos - x2*sin, rot2 = x1*sin + x2*cos
    print("Running fused RoPE kernel...", flush=True)
    rot1, rot2 = TTRetentionLayer._fused_rope_4d(
        x1, x2, layer._rope_cos_2d, layer._rope_sin_2d,
        B, H, T, d_half, device)
    ttnn.synchronize_device(device)
    r1_host = ttnn.to_torch(rot1).reshape(-1, d_half)
    r2_host = ttnn.to_torch(rot2).reshape(-1, d_half)
    print(f"  rot1: std={r1_host.std().item():.4f}", flush=True)
    print(f"  rot2: std={r2_host.std().item():.4f}", flush=True)

    # References via ttnn ops
    ref_rot1 = ttnn.to_torch(ttnn.sub(ttnn.mul(x1, layer._rope_cos), ttnn.mul(x2, layer._rope_sin)))
    ref_rot2 = ttnn.to_torch(ttnn.add(ttnn.mul(x1, layer._rope_sin), ttnn.mul(x2, layer._rope_cos)))
    ref_rot1_flat = ref_rot1.reshape(-1, d_half)
    ref_rot2_flat = ref_rot2.reshape(-1, d_half)

    d1 = (r1_host - ref_rot1_flat).abs().max().item()
    d2 = (r2_host - ref_rot2_flat).abs().max().item()
    rel1 = d1 / ref_rot1_flat.abs().max().item()
    rel2 = d2 / ref_rot2_flat.abs().max().item()
    print(f"  rot1 max diff: {d1:.6f}, rel err: {rel1:.6f}", flush=True)
    print(f"  rot2 max diff: {d2:.6f}, rel err: {rel2:.6f}", flush=True)
    print("  PASS" if rel1 < 0.02 and rel2 < 0.02 else "  FAIL", flush=True)

    # Also test backward (neg sin)
    print("\nTesting backward RoPE (neg sin)...", flush=True)
    grad_r1, grad_r2 = TTRetentionLayer._fused_rope_4d(
        x1, x2, layer._rope_cos_2d, layer._rope_neg_sin_2d,
        B, H, T, d_half, device)
    ttnn.synchronize_device(device)
    g1_host = ttnn.to_torch(grad_r1).reshape(-1, d_half)
    g2_host = ttnn.to_torch(grad_r2).reshape(-1, d_half)

    ref_g1 = ttnn.to_torch(ttnn.add(ttnn.mul(x1, layer._rope_cos), ttnn.mul(x2, layer._rope_sin)))
    ref_g2 = ttnn.to_torch(ttnn.add(ttnn.mul(ttnn.neg(x1), layer._rope_sin), ttnn.mul(x2, layer._rope_cos)))
    ref_g1_flat = ref_g1.reshape(-1, d_half)
    ref_g2_flat = ref_g2.reshape(-1, d_half)

    dg1 = (g1_host - ref_g1_flat).abs().max().item()
    dg2 = (g2_host - ref_g2_flat).abs().max().item()
    relg1 = dg1 / ref_g1_flat.abs().max().item()
    relg2 = dg2 / ref_g2_flat.abs().max().item()
    print(f"  grad_x1 max diff: {dg1:.6f}, rel err: {relg1:.6f}", flush=True)
    print(f"  grad_x2 max diff: {dg2:.6f}, rel err: {relg2:.6f}", flush=True)
    print("  PASS" if relg1 < 0.02 and relg2 < 0.02 else "  FAIL", flush=True)

finally:
    ttnn.close_device(device)
