#!/usr/bin/env python3
"""Test embedding + position embedding + encoder parity."""
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

# Test with a small batch
import random
from challenge_data import ChallengeDataset
dataset = ChallengeDataset("data/tiny_challenges_train.txt", "data/tiny_challenges_valid.txt", 256, 32)
rng = random.Random(42)
prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = dataset.sample_batch(2, "valid", rng)

# Compare embeddings
with torch.no_grad():
    pos = torch.arange(prompt_ids.shape[1]).unsqueeze(0)
    pt_emb = pt_model.token_embedding(prompt_ids)
    pt_pos = pt_model.prompt_pos_embedding(pos)
    pt_hidden = pt_emb + pt_pos
    print(f"PT emb[0,0,:5]: {pt_emb[0,0,:5]}")
    print(f"PT pos[0,0,:5]: {pt_pos[0,0,:5]}")
    print(f"PT hidden[0,0,:5]: {pt_hidden[0,0,:5]}")

# TT embedding
indices = ttnn.from_torch(prompt_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
tt_emb = tt_model._embedding_lookup(indices, tt_model.token_emb_weight_bf16)
_safe_deallocate(indices)
tt_emb_torch = from_device(tt_emb)
_safe_deallocate(tt_emb)
print(f"TT emb[0,0,:5]: {tt_emb_torch[0,0,:5]}")
print(f"Emb diff: mean={(pt_emb - tt_emb_torch).abs().mean():.6f} max={(pt_emb - tt_emb_torch).abs().max():.6f}")

# TT position embedding
pos_ids = ttnn.from_torch(torch.arange(prompt_ids.shape[1], dtype=torch.int32).unsqueeze(0),
                          dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
tt_pos = tt_model._embedding_lookup(pos_ids, tt_model.prompt_pos_emb_bf16)
_safe_deallocate(pos_ids)
tt_pos_torch = from_device(tt_pos)
_safe_deallocate(tt_pos)
print(f"TT pos[0,0,:5]: {tt_pos_torch[0,0,:5]}")
print(f"Pos diff: mean={(pt_pos - tt_pos_torch).abs().mean():.6f} max={(pt_pos - tt_pos_torch).abs().max():.6f}")

# TT hidden
tt_hidden = ttnn.add(to_device(pt_emb, device, dtype=ttnn.float32), to_device(pt_pos, device, dtype=ttnn.float32))
# Actually compare the full encoder output
pt_encoded = pt_model.encode_prompt(prompt_ids, prompt_mask)
tt_encoded = tt_model.encode_prompt(prompt_ids, prompt_mask)
tt_encoded_torch = from_device(tt_encoded)
_safe_deallocate(tt_encoded)

print(f"\nPT encoded[0,0,:5]: {pt_encoded[0,0,:5]}")
print(f"TT encoded[0,0,:5]: {tt_encoded_torch[0,0,:5]}")
print(f"Encoded diff: mean={(pt_encoded - tt_encoded_torch).abs().mean():.6f} max={(pt_encoded - tt_encoded_torch).abs().max():.6f}")
cos = torch.nn.functional.cosine_similarity(pt_encoded.flatten(), tt_encoded_torch.flatten(), dim=0)
print(f"Encoded cos_sim: {cos:.6f}")

# Check if the issue is the embedding precision loss
# Compare with fp32 embedding (bypass the bf16 requirement)
pt_emb_fp32 = pt_model.token_embedding.weight[prompt_ids]
print(f"\nDirect lookup (fp32) emb[0,0,:5]: {pt_emb_fp32[0,0,:5]}")
print(f"Direct vs TT emb diff: mean={(pt_emb_fp32 - tt_emb_torch).abs().mean():.6f}")

ttnn.close_device(device)
