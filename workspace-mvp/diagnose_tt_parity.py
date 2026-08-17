#!/usr/bin/env python3
"""Diagnose TT-NN vs PyTorch precision gap at each forward stage.

Compares intermediate outputs (encoded prompt, memory init, after each
reasoning step, decoder output, final logits) to identify where bfloat16
precision loss accumulates most.
"""

import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import ttnn
import random
import numpy as np
from text_latent_memory_model import TextLatentMemoryConfig, TextLatentMemoryModel
from tt_text_latent_memory_model import (
    TTTextLatentMemoryConfig, TTTextLatentMemoryModel,
    _safe_deallocate, to_device, from_device,
)
from challenge_data import ChallengeDataset
from train_tt_text_latent_memory import transfer_pytorch_to_tt


def compare_stage(name, pt_tensor, tt_tensor):
    """Compare a PyTorch tensor with a TT-NN tensor (converted to torch)."""
    pt = pt_tensor.detach().to(torch.float32)
    tt = tt_tensor.to(torch.float32) if not isinstance(tt_tensor, torch.Tensor) else tt_tensor.to(torch.float32)

    if pt.shape != tt.shape:
        print(f"  {name}: SHAPE MISMATCH pt={pt.shape} tt={tt.shape}")
        return

    abs_diff = (pt - tt).abs()
    rel_diff = abs_diff / (pt.abs().clamp_min(1e-6))
    cos_sim = torch.nn.functional.cosine_similarity(
        pt.flatten().unsqueeze(0), tt.flatten().unsqueeze(0)
    ).item()

    print(f"  {name:30s}  mean_abs={abs_diff.mean():.6f}  max_abs={abs_diff.max():.6f}  "
          f"mean_rel={rel_diff.mean():.4f}  cos_sim={cos_sim:.6f}")


def main():
    ckpt_path = "checkpoints/tt_text_lm/text_lm_final.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)

    dataset = ChallengeDataset(
        train_path="data/tiny_challenges_train.txt",
        valid_path="data/tiny_challenges_valid.txt",
        max_prompt_len=256, max_answer_len=32,
    )

    pt_model = TextLatentMemoryModel(ckpt["config"])
    pt_model.load_state_dict(ckpt["model"])
    pt_model.eval()

    device = ttnn.open_device(device_id=0)

    tt_config = TTTextLatentMemoryConfig(
        vocab_size=ckpt["config"].vocab_size,
        d_model=ckpt["config"].d_model,
        n_encoder_layers=ckpt["config"].n_encoder_layers,
        n_decoder_layers=ckpt["config"].n_decoder_layers,
        n_heads=ckpt["config"].n_heads,
        n_slots=ckpt["config"].n_slots,
        max_reasoning_steps=ckpt["config"].max_reasoning_steps,
        expand=ckpt["config"].expand,
        max_prompt_len=256,
        max_answer_len=ckpt["config"].max_answer_len,
        pad_token_id=ckpt["config"].pad_token_id,
    )
    tt_model = TTTextLatentMemoryModel(tt_config, device, dtype=ttnn.float32)
    transfer_pytorch_to_tt(pt_model, tt_model, device)

    # Get a fixed batch
    rng = random.Random(42)
    prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
        dataset.sample_batch(4, "valid", rng)

    print("=" * 80)
    print("Stage-by-stage comparison (PT fp32 vs TT bf16)")
    print("=" * 80)

    # --- PyTorch forward with intermediates ---
    with torch.no_grad():
        pt_encoded = pt_model.encode_prompt(prompt_ids, prompt_mask)
        pt_memory_init = pt_model.initialize_memory(pt_encoded, prompt_mask)
        pt_memory_states = [pt_memory_init]
        pt_memory = pt_memory_init
        for k in range(ckpt["config"].max_reasoning_steps):
            pt_memory, pt_gate = pt_model.transition(pt_memory)
            pt_memory_states.append(pt_memory)
        pt_logits = pt_model.decode_answer(pt_memory, dec_input, ans_mask)

    # --- TT-NN forward with intermediates ---
    # Stage 1: Encode prompt
    tt_encoded = tt_model.encode_prompt(prompt_ids, prompt_mask)
    compare_stage("encoded_prompt", pt_encoded, from_device(tt_encoded))

    # Stage 2: Initialize memory
    tt_memory = tt_model.initialize_memory(tt_encoded, prompt_mask)
    _safe_deallocate(tt_encoded)
    compare_stage("memory_init", pt_memory_init, from_device(tt_memory))

    # Stage 3: Reasoning steps
    for k in range(tt_config.max_reasoning_steps):
        tt_memory, gate = tt_model.transition.forward(tt_memory)
        _safe_deallocate(gate)
        compare_stage(f"memory_step_{k+1}", pt_memory_states[k+1], from_device(tt_memory))

    # Stage 4: Decode
    tt_logits_tt = tt_model.decode_answer(tt_memory, dec_input, ans_mask)
    _safe_deallocate(tt_memory)
    tt_logits = from_device(tt_logits_tt)
    _safe_deallocate(tt_logits_tt)

    compare_stage("final_logits", pt_logits, tt_logits)

    # Token-level comparison
    pt_preds = pt_logits.argmax(-1)
    tt_preds = tt_logits.argmax(-1)
    agreement = (pt_preds == tt_preds).float()
    masked_agreement = (agreement * ans_mask.float()).sum() / ans_mask.float().sum().clamp_min(1)
    print(f"\n  Token prediction agreement: {masked_agreement.item():.4f}")

    # Loss comparison
    V = pt_logits.size(-1)
    pt_loss = torch.nn.functional.cross_entropy(
        pt_logits.reshape(-1, V), ans_targets.reshape(-1), reduction='none'
    ).reshape(ans_targets.shape)
    pt_loss = (pt_loss * ans_mask.float()).sum() / ans_mask.float().sum().clamp_min(1)

    tt_loss = torch.nn.functional.cross_entropy(
        tt_logits.reshape(-1, V), ans_targets.reshape(-1), reduction='none'
    ).reshape(ans_targets.shape)
    tt_loss = (tt_loss * ans_mask.float()).sum() / ans_mask.float().sum().clamp_min(1)

    print(f"  PT loss: {pt_loss.item():.4f}  TT loss: {tt_loss.item():.4f}  diff: {abs(pt_loss.item()-tt_loss.item()):.4f}")

    ttnn.synchronize_device(device)
    tt_model.clear_caches()
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
