"""Test fused scale + decay kernel."""
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
    config = ModelConfig(d_model=384, n_heads=4)
    layer = TTRetentionLayer(config, device, use_fused_rope=True)
    layer._init_rope(128, device)

    B, H, T = 8, 4, 128
    scale = layer.scale  # 1/sqrt(d_head) = 1/sqrt(96)

    # Create random scores_raw (B, H, T, T)
    scores_torch = torch.randn(B, H, T, T, dtype=torch.bfloat16) * 0.5
    scores_tt = ttnn.from_torch(scores_torch, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)

    # Get decay matrix
    D_decay = layer._get_decay_matrix(T, device)  # (1, H, T, T)

    # Reference: scores = scores_raw * scale * D_decay
    ref = ttnn.to_torch(ttnn.mul(ttnn.mul(scores_tt, scale), D_decay))

    # Fused
    fused_out = TTRetentionLayer._fused_scale_decay(scores_tt, D_decay, scale, B, H, T, device)
    ttnn.synchronize_device(device)
    fused_host = ttnn.to_torch(fused_out)

    # Reshape both to 4D for comparison
    ref_4d = ref.reshape(B, H, T, T)
    fused_4d = fused_host.reshape(B, H, T, T)

    max_diff = (fused_4d - ref_4d).abs().max().item()
    rel_err = max_diff / ref_4d.abs().max().item()
    print(f"max abs diff: {max_diff:.6f}", flush=True)
    print(f"max rel err: {rel_err:.6f}", flush=True)
    print(f"ref std: {ref_4d.std().item():.4f}", flush=True)
    print(f"fused std: {fused_4d.std().item():.4f}", flush=True)

    if rel_err < 0.02:
        print("  PASS", flush=True)
    else:
        print("  FAIL", flush=True)
        print(f"  fused[0,0,0,:4]: {fused_4d[0,0,0,:4]}", flush=True)
        print(f"  ref  [0,0,0,:4]: {ref_4d[0,0,0,:4]}", flush=True)

finally:
    ttnn.close_device(device)
