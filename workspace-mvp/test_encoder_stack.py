#!/usr/bin/env python3
"""Test full encoder stack parity."""
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
from text_latent_memory_model import TextLatentMemoryConfig, TextLatentMemoryModel
from tt_text_latent_memory_model import (
    TTTextLatentMemoryConfig, TTTextLatentMemoryModel,
    to_device, from_device, _safe_deallocate,
)
from train_tt_text_latent_memory import transfer_pytorch_to_tt
from challenge_data import ChallengeDataset
import random

device = ttnn.open_device(device_id=0)

ckpt = torch.load("checkpoints/tt_text_lm/text_lm_final.pt", weights_only=False)
pt_model = TextLatentMemoryModel(ckpt["config"])
pt_model.load_state_dict(ckpt["model"])
pt_model.eval()

tt_config = TTTextLatentMemoryConfig(
    vocab_size=ckpt["config"].vocab_size, d_model=ckpt["config"].d_model,
    n_encoder_layers=ckpt["config"].n_encoder_layers, n_decoder_layers=ckpt["config"].n_decoder_layers,
    n_heads=ckpt["config"].n_heads, n_slots=ckpt["config"].n_slots,
    max_reasoning_steps=ckpt["config"].max_reasoning_steps, expand=ckpt["config"].expand,
    max_prompt_len=256, max_answer_len=ckpt["config"].max_answer_len,
    pad_token_id=ckpt["config"].pad_token_id,
)
tt_model = TTTextLatentMemoryModel(tt_config, device, dtype=ttnn.float32)
transfer_pytorch_to_tt(pt_model, tt_model, device)

dataset = ChallengeDataset("data/tiny_challenges_train.txt", "data/tiny_challenges_valid.txt", 256, 32)
rng = random.Random(42)
prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = dataset.sample_batch(2, "valid", rng)

# Compare after each encoder layer
with torch.no_grad():
    # PyTorch: manually step through encoder
    pos = torch.arange(prompt_ids.shape[1]).unsqueeze(0)
    pt_hidden = pt_model.token_embedding(prompt_ids) + pt_model.prompt_pos_embedding(pos)
    print(f"PT input to encoder[0,0,:5]: {pt_hidden[0,0,:5]}")

    kpm_pt = ~prompt_mask.bool()
    pt_states = [pt_hidden.clone()]
    for i, layer in enumerate(pt_model.encoder.layers):
        pt_hidden = layer(pt_hidden, src_key_padding_mask=kpm_pt)
        pt_states.append(pt_hidden.clone())
        print(f"PT after enc layer {i}[0,0,:5]: {pt_hidden[0,0,:5]}")
    pt_encoded = pt_model.encoder_norm(pt_hidden)
    print(f"PT after encoder_norm[0,0,:5]: {pt_encoded[0,0,:5]}")

# TT: manually step through encoder
indices = ttnn.from_torch(prompt_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
tt_emb = tt_model._embedding_lookup(indices, tt_model.token_emb_weight_bf16)
_safe_deallocate(indices)
pos_ids = ttnn.from_torch(torch.arange(prompt_ids.shape[1], dtype=torch.int32).unsqueeze(0),
                          dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
tt_pos = tt_model._embedding_lookup(pos_ids, tt_model.prompt_pos_emb_bf16)
_safe_deallocate(pos_ids)
tt_hidden = ttnn.add(tt_emb, tt_pos)
_safe_deallocate(tt_emb)
_safe_deallocate(tt_pos)

tt_hidden_torch = from_device(tt_hidden)
print(f"\nTT input to encoder[0,0,:5]: {tt_hidden_torch[0,0,:5]}")

kpm_tt = tt_model._get_key_padding_mask(prompt_mask)
for i, layer in enumerate(tt_model.encoder_layers):
    tt_hidden = layer.forward(tt_hidden, key_padding_mask=kpm_tt)
    tt_hidden_torch = from_device(tt_hidden)
    diff = (pt_states[i+1] - tt_hidden_torch).abs()
    cos = torch.nn.functional.cosine_similarity(pt_states[i+1].flatten(), tt_hidden_torch.flatten(), dim=0)
    print(f"TT after enc layer {i}[0,0,:5]: {tt_hidden_torch[0,0,:5]}  diff={diff.mean():.6f} cos={cos:.6f}")

tt_encoded = tt_model.encoder_norm.forward(tt_hidden)
tt_encoded_torch = from_device(tt_encoded)
_safe_deallocate(tt_encoded)
_safe_deallocate(tt_hidden)
_safe_deallocate(kpm_tt)
print(f"TT after encoder_norm[0,0,:5]: {tt_encoded_torch[0,0,:5]}")

# Compare
diff = (pt_encoded - tt_encoded_torch).abs()
cos = torch.nn.functional.cosine_similarity(pt_encoded.flatten(), tt_encoded_torch.flatten(), dim=0)
print(f"\nFinal encoded diff: mean={diff.mean():.6f} max={diff.max():.6f} cos={cos:.6f}")

ttnn.close_device(device)
