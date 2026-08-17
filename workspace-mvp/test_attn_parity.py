#!/usr/bin/env python3
"""Test single attention layer parity between PyTorch and TT-NN."""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# P300 setup (same as train_tt_text_latent_memory.py)
_P300_SUBSYSTEM_IDS = {"0x0044", "0x0045", "0x0046"}
def _is_p300():
    from pathlib import Path
    for entry in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        sub = (entry / "device" / "subsystem_device").read_text().strip().lower()
        if sub in _P300_SUBSYSTEM_IDS:
            return True
    return False
def _find_mgd():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.find_spec("ttnn")
    for name in ["p150_mesh_graph_descriptor.textproto", "p300_mesh_graph_descriptor.textproto"]:
        if spec is not None and spec.submodule_search_locations:
            path = Path(next(iter(spec.submodule_search_locations))) / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
            if path.is_file():
                return str(path)
        for p in sys.path:
            candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
            if candidate.is_file():
                return str(candidate)
    return None
if _is_p300():
    _mgd = _find_mgd()
    if _mgd:
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

import torch
import ttnn
import torch.nn as nn
from tt_text_latent_memory_model import TTMultiHeadAttention, to_device, from_device, _safe_deallocate

device = ttnn.open_device(device_id=0)

d_model = 128
n_heads = 4
B, L = 2, 8

# PyTorch attention
pt_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
pt_attn.eval()

x = torch.randn(B, L, d_model)

with torch.no_grad():
    pt_out, _ = pt_attn(x, x, x, need_weights=False)
    print(f"PT out[0,0,:5]: {pt_out[0,0,:5]}")

# TT attention
tt_attn = TTMultiHeadAttention(d_model, n_heads, device, dtype=ttnn.float32)

# Transfer weights (transpose for TT layout)
_safe_deallocate(tt_attn.in_proj_weight)
_safe_deallocate(tt_attn.out_proj_weight)
_safe_deallocate(tt_attn.in_proj_bias)
_safe_deallocate(tt_attn.out_proj_bias)
tt_attn.in_proj_weight = to_device(pt_attn.in_proj_weight.t().contiguous(), device, dtype=ttnn.float32)
tt_attn.out_proj_weight = to_device(pt_attn.out_proj.weight.t().contiguous(), device, dtype=ttnn.float32)
tt_attn.in_proj_bias = to_device(pt_attn.in_proj_bias, device, dtype=ttnn.float32)
tt_attn.out_proj_bias = to_device(pt_attn.out_proj.bias, device, dtype=ttnn.float32)

x_tt = to_device(x, device, dtype=ttnn.float32)
tt_out = tt_attn.forward(x_tt, x_tt, x_tt)
tt_out_torch = from_device(tt_out)
_safe_deallocate(tt_out)
_safe_deallocate(x_tt)

print(f"TT out[0,0,:5]: {tt_out_torch[0,0,:5]}")
diff = (pt_out - tt_out_torch).abs()
print(f"Diff: mean={diff.mean():.6f} max={diff.max():.6f}")
cos = torch.nn.functional.cosine_similarity(pt_out.flatten(), tt_out_torch.flatten(), dim=0)
print(f"Cosine sim: {cos:.6f}")

ttnn.close_device(device)
