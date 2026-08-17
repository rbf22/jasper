#!/usr/bin/env python3
"""Test single encoder layer parity between PyTorch and TT-NN."""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from tt_text_latent_memory_model import (
    TTEncoderLayer, to_device, from_device, _safe_deallocate,
)

device = ttnn.open_device(device_id=0)

d_model = 128
n_heads = 4
d_ff = 512
B, L = 2, 8

# PyTorch encoder layer
pt_layer = nn.TransformerEncoderLayer(
    d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
    dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
)
pt_layer.eval()

x = torch.randn(B, L, d_model)
# Create a key padding mask (True = padding)
kpm = torch.zeros(B, L, dtype=torch.bool)
kpm[0, 5:] = True  # first example: positions 5-7 are padding
kpm[1, 6:] = True  # second example: positions 6-7 are padding

with torch.no_grad():
    pt_out = pt_layer(x, src_key_padding_mask=kpm)
    print(f"PT out[0,0,:5]: {pt_out[0,0,:5]}")

# TT encoder layer
tt_layer = TTEncoderLayer(d_model, n_heads, d_ff, device, dtype=ttnn.float32)

# Transfer weights
# in_proj_weight: (3*d, d) -> (d, 3*d) for TT
_safe_deallocate(tt_layer.self_attn.in_proj_weight)
_safe_deallocate(tt_layer.self_attn.out_proj_weight)
_safe_deallocate(tt_layer.self_attn.in_proj_bias)
_safe_deallocate(tt_layer.self_attn.out_proj_bias)
tt_layer.self_attn.in_proj_weight = to_device(
    pt_layer.self_attn.in_proj_weight.t().contiguous(), device, dtype=ttnn.float32)
tt_layer.self_attn.out_proj_weight = to_device(
    pt_layer.self_attn.out_proj.weight.t().contiguous(), device, dtype=ttnn.float32)
tt_layer.self_attn.in_proj_bias = to_device(
    pt_layer.self_attn.in_proj_bias, device, dtype=ttnn.float32)
tt_layer.self_attn.out_proj_bias = to_device(
    pt_layer.self_attn.out_proj.bias, device, dtype=ttnn.float32)

# Norms
_safe_deallocate(tt_layer.norm1.weight)
_safe_deallocate(tt_layer.norm1.bias)
_safe_deallocate(tt_layer.norm2.weight)
_safe_deallocate(tt_layer.norm2.bias)
tt_layer.norm1.weight = to_device(pt_layer.norm1.weight, device, dtype=ttnn.float32)
tt_layer.norm1.bias = to_device(pt_layer.norm1.bias, device, dtype=ttnn.float32)
tt_layer.norm2.weight = to_device(pt_layer.norm2.weight, device, dtype=ttnn.float32)
tt_layer.norm2.bias = to_device(pt_layer.norm2.bias, device, dtype=ttnn.float32)

# FFN
_safe_deallocate(tt_layer.linear1.weight)
_safe_deallocate(tt_layer.linear1.bias)
_safe_deallocate(tt_layer.linear2.weight)
_safe_deallocate(tt_layer.linear2.bias)
tt_layer.linear1.weight = to_device(pt_layer.linear1.weight.t().contiguous(), device, dtype=ttnn.float32)
tt_layer.linear1.bias = to_device(pt_layer.linear1.bias, device, dtype=ttnn.float32)
tt_layer.linear2.weight = to_device(pt_layer.linear2.weight.t().contiguous(), device, dtype=ttnn.float32)
tt_layer.linear2.bias = to_device(pt_layer.linear2.bias, device, dtype=ttnn.float32)

# TT forward — key padding mask is (B, L) with True=valid, False=padding
# PyTorch uses True=padding, so we invert
attention_mask = ~kpm  # True=valid
x_tt = to_device(x, device, dtype=ttnn.float32)
tt_out = tt_layer.forward(x_tt, key_padding_mask=tt_layer.self_attn._get_key_padding_mask(attention_mask) if hasattr(tt_layer.self_attn, '_get_key_padding_mask') else None)

# Actually, the TT layer expects key_padding_mask to be the additive mask
# Let me use the model's mask conversion
from tt_text_latent_memory_model import TTTextLatentMemoryConfig, TTTextLatentMemoryModel

# Create a minimal model just for the mask conversion method
# Actually, the _get_key_padding_mask is on the main model, not on the layer
# The layer.forward expects key_padding_mask as additive (0=valid, -inf=padding)
torch_dtype = torch.float32
additive = torch.zeros(B, L, dtype=torch_dtype)
additive[kpm] = float('-inf')
kpm_tt = to_device(additive, device, dtype=ttnn.float32)

tt_out = tt_layer.forward(x_tt, key_padding_mask=kpm_tt)
tt_out_torch = from_device(tt_out)
_safe_deallocate(tt_out)
_safe_deallocate(x_tt)
_safe_deallocate(kpm_tt)

print(f"TT out[0,0,:5]: {tt_out_torch[0,0,:5]}")
diff = (pt_out - tt_out_torch).abs()
print(f"Diff: mean={diff.mean():.6f} max={diff.max():.6f}")
cos = torch.nn.functional.cosine_similarity(pt_out.flatten(), tt_out_torch.flatten(), dim=0)
print(f"Cosine sim: {cos:.6f}")

# Also test without key padding mask
with torch.no_grad():
    pt_out_no_mask = pt_layer(x)
tt_out_no_mask = tt_layer.forward(to_device(x, device, dtype=ttnn.float32))
tt_out_no_mask_torch = from_device(tt_out_no_mask)
_safe_deallocate(tt_out_no_mask)
diff_no_mask = (pt_out_no_mask - tt_out_no_mask_torch).abs()
cos_no_mask = torch.nn.functional.cosine_similarity(pt_out_no_mask.flatten(), tt_out_no_mask_torch.flatten(), dim=0)
print(f"No mask - Diff: mean={diff_no_mask.mean():.6f} max={diff_no_mask.max():.6f} cos={cos_no_mask:.6f}")

ttnn.close_device(device)
