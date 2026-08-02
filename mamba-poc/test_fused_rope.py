"""Quick test of the fused RoPE 4D kernel — tries multiple devices."""
import os, sys
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

# Mesh graph descriptor — use p150 (p300 causes topology mismatches)
_MGD_BASE = ("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/"
             "pjrt_plugin_tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/")
os.environ["TT_MESH_GRAPH_DESC_PATH"] = _MGD_BASE + "p150_mesh_graph_descriptor.textproto"
os.environ["TT_VISIBLE_DEVICES"] = "0"

import torch
import ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/mamba-poc")
from model_ttnn import ModelConfig, TTRetentionLayer

print("Opening device 0...", flush=True)
device = ttnn.open_device(device_id=0)
print("Device opened!", flush=True)

try:
    config = ModelConfig(d_model=64, n_heads=2)
    layer = TTRetentionLayer(config, device, use_fused_rope=True)
    layer._init_rope(32, device)  # T=32 (tile-aligned)

    B, H, T, d_h = 2, 2, 32, 32
    d_half = d_h // 2

    # Create test input
    x_torch = torch.randn(B, H, T, d_h, dtype=torch.bfloat16) * 0.3
    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print("Testing fused RoPE forward...", flush=True)
    # Fused version
    x1 = ttnn.slice(x, [0, 0, 0, 0], [B, H, T, d_half])
    x2 = ttnn.slice(x, [0, 0, 0, d_half], [B, H, T, d_h])
    rot1, rot2 = layer._fused_rope_4d(x1, x2, layer._rope_cos_2d, layer._rope_sin_2d,
                                       B, H, T, d_half, device)
    fused_out = ttnn.concat([rot1, rot2], dim=-1)
    fused_host = ttnn.to_torch(fused_out)
    print(f"  fused output shape: {fused_host.shape}", flush=True)
    print(f"  fused output std: {fused_host.std().item():.4f}", flush=True)

    # Reference (old ttnn ops version)
    x1_ref = ttnn.slice(x, [0, 0, 0, 0], [B, H, T, d_half])
    x2_ref = ttnn.slice(x, [0, 0, 0, d_half], [B, H, T, d_h])
    x1_cos = ttnn.mul(x1_ref, layer._rope_cos)
    x2_sin = ttnn.mul(x2_ref, layer._rope_sin)
    x1_sin = ttnn.mul(x1_ref, layer._rope_sin)
    x2_cos = ttnn.mul(x2_ref, layer._rope_cos)
    ref_out = ttnn.concat([ttnn.sub(x1_cos, x2_sin), ttnn.add(x1_sin, x2_cos)], dim=-1)
    ref_host = ttnn.to_torch(ref_out)
    print(f"  ref output shape: {ref_host.shape}", flush=True)
    print(f"  ref output std: {ref_host.std().item():.4f}", flush=True)

    # Compare
    max_abs_diff = (fused_host - ref_host).abs().max().item()
    rel_err = max_abs_diff / ref_host.abs().max().item()
    print(f"  max abs diff: {max_abs_diff:.6f}", flush=True)
    print(f"  max rel err: {rel_err:.6f}", flush=True)
    if rel_err < 0.01:
        print("  PASS", flush=True)
    else:
        print("  FAIL", flush=True)
        print(f"  fused[0,0,0,:4]: {fused_host[0,0,0,:4]}", flush=True)
        print(f"  ref  [0,0,0,:4]: {ref_host[0,0,0,:4]}", flush=True)

finally:
    ttnn.close_device(device)
