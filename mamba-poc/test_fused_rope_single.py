"""2-pass test with FPU warm-up fix."""
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

    # Pass 1: x1*cos, x2*sin
    print("Pass 1...", flush=True)
    tc, ts = TTRetentionLayer._fused_mul_kernel(x1, x2, layer._rope_cos_2d, layer._rope_sin_2d,
                                                 B, H, T, d_half, device)
    ttnn.synchronize_device(device)
    rot1_2d = ttnn.sub(tc, ts)
    r1_host = ttnn.to_torch(rot1_2d)
    print(f"  rot1: std={r1_host.std().item():.4f}", flush=True)

    # Pass 2: x1*sin, x2*cos
    print("Pass 2...", flush=True)
    ts2, tc2 = TTRetentionLayer._fused_mul_kernel(x1, x2, layer._rope_sin_2d, layer._rope_cos_2d,
                                                 B, H, T, d_half, device)
    ttnn.synchronize_device(device)
    rot2_2d = ttnn.add(ts2, tc2)
    r2_host = ttnn.to_torch(rot2_2d)
    print(f"  rot2: std={r2_host.std().item():.4f}", flush=True)

    # References
    ref_rot1 = ttnn.to_torch(ttnn.sub(ttnn.mul(x1, layer._rope_cos), ttnn.mul(x2, layer._rope_sin)))
    ref_rot2 = ttnn.to_torch(ttnn.add(ttnn.mul(x1, layer._rope_sin), ttnn.mul(x2, layer._rope_cos)))
    ref_rot1_flat = ref_rot1.reshape(-1, d_half)
    ref_rot2_flat = ref_rot2.reshape(-1, d_half)

    d1 = (r1_host - ref_rot1_flat).abs().max().item()
    d2 = (r2_host - ref_rot2_flat).abs().max().item()
    print(f"  rot1 diff: {d1:.6f}", flush=True)
    print(f"  rot2 diff: {d2:.6f}", flush=True)
    print("  PASS" if d1 < 0.01 and d2 < 0.01 else "  FAIL", flush=True)

finally:
    ttnn.close_device(device)
