#!/usr/bin/env python3
"""CPU-vs-TT parity test for the text latent-memory model.

Creates a CPU PyTorch model and a TT model with the same config,
copies weights from CPU to TT, runs forward on the same input,
and compares logits and gradients.

Usage:
    TT_VISIBLE_DEVICES=0 python test_cpu_tt_parity.py
"""
import os
import sys
import torch
import numpy as np

# P300 mesh graph descriptor setup (must happen before ttnn import)
def _setup_mesh_descriptor():
    import importlib.util
    from pathlib import Path
    try:
        spec = importlib.util.find_spec("ttnn")
        for name in ["p150_mesh_graph_descriptor.textproto", "p300_mesh_graph_descriptor.textproto"]:
            if spec is not None and spec.submodule_search_locations:
                path = Path(next(iter(spec.submodule_search_locations))) / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if path.is_file():
                    os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(path)
                    return
            for p in sys.path:
                candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if candidate.is_file():
                    os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(candidate)
                    return
    except Exception:
        pass

_setup_mesh_descriptor()

import ttnn

from text_latent_memory_model import TextLatentMemoryConfig, TextLatentMemoryModel
from tt_text_latent_memory_model import TTTextLatentMemoryConfig, TTTextLatentMemoryModel
from challenge_data import ChallengeDataset


def to_device(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def copy_cpu_to_tt(cpu_model: TextLatentMemoryModel, tt_model: TTTextLatentMemoryModel, device, dtype=ttnn.bfloat16):
    """Copy weights from CPU PyTorch model to TT model.

    Key layout differences:
    - PyTorch nn.Linear: weight is (out, in), bias is (out,)
    - TT TTLinear: weight is (in, out), bias is (out,)
    - PyTorch nn.MultiheadAttention: in_proj_weight is (3*d, d), out_proj_weight is (d, d)
    - TT TTMultiHeadAttention: in_proj_weight is (d, 3d), out_proj_weight is (d, d)
    """
    cfg = cpu_model.config

    def transpose_linear(weight):
        """PyTorch (out, in) -> TT (in, out)"""
        return weight.t().contiguous()

    def transpose_in_proj(weight):
        """PyTorch (3*d, d) -> TT (d, 3*d)"""
        return weight.t().contiguous()

    def to_tt(tensor):
        return to_device(tensor.float(), device, dtype=dtype)

    # Token embedding (shared with lm_head via weight tying)
    tt_model._set_param("token_emb_weight", to_tt(cpu_model.token_embedding.weight))

    # Position embeddings
    # CPU has prompt_pos_embedding of size (512, d), TT has (max_prompt_len, d)
    pp = cpu_model.prompt_pos_embedding.weight[:cfg.d_model]  # truncate if needed
    # Actually TT model stores prompt_pos_emb with size max_prompt_len
    # Let's just copy the first max_prompt_len rows
    tt_model._set_param("prompt_pos_emb", to_tt(cpu_model.prompt_pos_embedding.weight[:tt_model.config.max_prompt_len]))
    tt_model._set_param("answer_pos_emb", to_tt(cpu_model.answer_pos_embedding.weight))

    # Slot queries
    tt_model._set_param("slot_queries", to_tt(cpu_model.slot_queries))

    # Encoder norm
    tt_model._set_param("encoder_norm_weight", to_tt(cpu_model.encoder_norm.weight))

    # Memory norm
    tt_model._set_param("memory_norm_weight", to_tt(cpu_model.memory_norm.weight))

    # Decoder norm
    tt_model._set_param("decoder_norm_weight", to_tt(cpu_model.decoder_norm.weight))

    # Encoder layers
    for i, enc_layer in enumerate(cpu_model.encoder.layers):
        # Self-attention
        tt_model._set_param(f"enc_{i}_in_proj_w", to_tt(transpose_in_proj(enc_layer.self_attn.in_proj_weight)))
        tt_model._set_param(f"enc_{i}_in_proj_b", to_tt(enc_layer.self_attn.in_proj_bias))
        tt_model._set_param(f"enc_{i}_out_proj_w", to_tt(transpose_linear(enc_layer.self_attn.out_proj.weight)))
        tt_model._set_param(f"enc_{i}_out_proj_b", to_tt(enc_layer.self_attn.out_proj.bias))
        # Norms
        tt_model._set_param(f"enc_{i}_norm1_w", to_tt(enc_layer.norm1.weight))
        tt_model._set_param(f"enc_{i}_norm1_b", to_tt(enc_layer.norm1.bias))
        tt_model._set_param(f"enc_{i}_norm2_w", to_tt(enc_layer.norm2.weight))
        tt_model._set_param(f"enc_{i}_norm2_b", to_tt(enc_layer.norm2.bias))
        # FFN
        tt_model._set_param(f"enc_{i}_linear1_w", to_tt(transpose_linear(enc_layer.linear1.weight)))
        tt_model._set_param(f"enc_{i}_linear1_b", to_tt(enc_layer.linear1.bias))
        tt_model._set_param(f"enc_{i}_linear2_w", to_tt(transpose_linear(enc_layer.linear2.weight)))
        tt_model._set_param(f"enc_{i}_linear2_b", to_tt(enc_layer.linear2.bias))

    # Memory init attention
    mem_init = cpu_model.memory_init
    tt_model._set_param("mem_init_in_proj_w", to_tt(transpose_in_proj(mem_init.in_proj_weight)))
    tt_model._set_param("mem_init_in_proj_b", to_tt(mem_init.in_proj_bias))
    tt_model._set_param("mem_init_out_proj_w", to_tt(transpose_linear(mem_init.out_proj.weight)))
    tt_model._set_param("mem_init_out_proj_b", to_tt(mem_init.out_proj.bias))

    # Transition
    trans = cpu_model.transition
    tt_model._set_param("trans_in_proj_w", to_tt(transpose_in_proj(trans.self_attn.in_proj_weight)))
    tt_model._set_param("trans_in_proj_b", to_tt(trans.self_attn.in_proj_bias))
    tt_model._set_param("trans_out_proj_w", to_tt(transpose_linear(trans.self_attn.out_proj.weight)))
    tt_model._set_param("trans_out_proj_b", to_tt(trans.self_attn.out_proj.bias))
    tt_model._set_param("trans_attn_norm_w", to_tt(trans.attn_norm.weight))
    # TT transition has ffn (d -> d*expand) and ffn_out (d*expand -> d), no bias
    # CPU transition has mlp: Linear(d, d*expand, bias=False), SiLU, Linear(d*expand, d, bias=False)
    tt_model._set_param("trans_ffn_w", to_tt(transpose_linear(trans.mlp[0].weight)))
    tt_model._set_param("trans_ffn_out_w", to_tt(transpose_linear(trans.mlp[2].weight)))
    tt_model._set_param("trans_gate_w", to_tt(transpose_linear(trans.gate.weight)))
    tt_model._set_param("trans_gate_b", to_tt(trans.gate.bias))
    tt_model._set_param("trans_output_norm_w", to_tt(trans.output_norm.weight))

    # Decoder layers
    for i, dec_layer in enumerate(cpu_model.decoder.layers):
        # Self-attention
        tt_model._set_param(f"dec_{i}_sa_in_proj_w", to_tt(transpose_in_proj(dec_layer.self_attn.in_proj_weight)))
        tt_model._set_param(f"dec_{i}_sa_in_proj_b", to_tt(dec_layer.self_attn.in_proj_bias))
        tt_model._set_param(f"dec_{i}_sa_out_proj_w", to_tt(transpose_linear(dec_layer.self_attn.out_proj.weight)))
        tt_model._set_param(f"dec_{i}_sa_out_proj_b", to_tt(dec_layer.self_attn.out_proj.bias))
        tt_model._set_param(f"dec_{i}_norm1_w", to_tt(dec_layer.norm1.weight))
        tt_model._set_param(f"dec_{i}_norm1_b", to_tt(dec_layer.norm1.bias))
        # Cross-attention (PyTorch uses multihead_attn, not cross_attn)
        tt_model._set_param(f"dec_{i}_ca_in_proj_w", to_tt(transpose_in_proj(dec_layer.multihead_attn.in_proj_weight)))
        tt_model._set_param(f"dec_{i}_ca_in_proj_b", to_tt(dec_layer.multihead_attn.in_proj_bias))
        tt_model._set_param(f"dec_{i}_ca_out_proj_w", to_tt(transpose_linear(dec_layer.multihead_attn.out_proj.weight)))
        tt_model._set_param(f"dec_{i}_ca_out_proj_b", to_tt(dec_layer.multihead_attn.out_proj.bias))
        tt_model._set_param(f"dec_{i}_norm2_w", to_tt(dec_layer.norm2.weight))
        tt_model._set_param(f"dec_{i}_norm2_b", to_tt(dec_layer.norm2.bias))
        # FFN
        tt_model._set_param(f"dec_{i}_linear1_w", to_tt(transpose_linear(dec_layer.linear1.weight)))
        tt_model._set_param(f"dec_{i}_linear1_b", to_tt(dec_layer.linear1.bias))
        tt_model._set_param(f"dec_{i}_linear2_w", to_tt(transpose_linear(dec_layer.linear2.weight)))
        tt_model._set_param(f"dec_{i}_linear2_b", to_tt(dec_layer.linear2.bias))
        tt_model._set_param(f"dec_{i}_norm3_w", to_tt(dec_layer.norm3.weight))
        tt_model._set_param(f"dec_{i}_norm3_b", to_tt(dec_layer.norm3.bias))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Small config for parity test
    d_model = 64
    n_heads = 2
    n_encoder_layers = 1
    n_decoder_layers = 1
    n_slots = 4
    max_reasoning_steps = 2
    expand = 2
    max_prompt_len = 32
    max_answer_len = 8
    vocab_size = 50257

    print("Creating CPU model...", flush=True)
    cpu_config = TextLatentMemoryConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        n_slots=n_slots,
        max_reasoning_steps=max_reasoning_steps,
        expand=expand,
        max_answer_len=max_answer_len,
        dropout=0.0,  # disable dropout for parity
    )
    cpu_model = TextLatentMemoryModel(cpu_config)
    cpu_model.eval()  # disable dropout
    cpu_params = sum(p.numel() for p in cpu_model.parameters())
    print(f"CPU model: {cpu_params:,} params", flush=True)

    print("Opening TT device...", flush=True)
    device = ttnn.open_device(device_id=0)

    print("Creating TT model...", flush=True)
    tt_config = TTTextLatentMemoryConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        n_slots=n_slots,
        max_reasoning_steps=max_reasoning_steps,
        expand=expand,
        max_prompt_len=max_prompt_len,
        max_answer_len=max_answer_len,
    )
    tt_model = TTTextLatentMemoryModel(tt_config, device, dtype=ttnn.float32)
    tt_params = tt_model.get_num_params()
    print(f"TT model: {tt_params:,} params (fp32)", flush=True)

    print("Copying weights from CPU to TT...", flush=True)
    copy_cpu_to_tt(cpu_model, tt_model, device, dtype=ttnn.float32)

    # Load dataset and get a fixed batch
    print("Loading dataset...", flush=True)
    dataset = ChallengeDataset(
        train_path="data/tiny_challenges_train.txt",
        valid_path="data/tiny_challenges_valid.txt",
        max_prompt_len=max_prompt_len,
        max_answer_len=max_answer_len,
    )

    torch.manual_seed(123)  # fixed seed for batch sampling
    prompt_ids, prompt_mask, decoder_input, answer_targets, answer_mask = dataset.sample_batch(batch_size=4)

    print(f"Batch: prompt_ids {prompt_ids.shape}, answer_ids {decoder_input.shape}", flush=True)

    # --- CPU forward (stage by stage) ---
    print("Running CPU forward...", flush=True)
    with torch.no_grad():
        cpu_encoded = cpu_model.encode_prompt(prompt_ids, prompt_mask)
        print(f"  CPU encoded: range [{cpu_encoded.min():.4f}, {cpu_encoded.max():.4f}]", flush=True)

        cpu_memory = cpu_model.initialize_memory(cpu_encoded, prompt_mask)
        print(f"  CPU memory_init: range [{cpu_memory.min():.4f}, {cpu_memory.max():.4f}]", flush=True)

        cpu_memory_final, cpu_mem_states, _ = cpu_model.reason(cpu_memory)
        for k, ms in enumerate(cpu_mem_states):
            print(f"    CPU mem_state[{k}]: range [{ms.min():.4f}, {ms.max():.4f}], mean={ms.mean():.4f}", flush=True)

        cpu_logits = cpu_model.decode_answer(cpu_memory_final, decoder_input, answer_mask)
        print(f"  CPU logits: {cpu_logits.shape}, range [{cpu_logits.min():.4f}, {cpu_logits.max():.4f}]", flush=True)

    # --- TT forward ---
    print("Running TT forward...", flush=True)
    tt_logits_tt = tt_model.forward_train(prompt_ids, prompt_mask, decoder_input, answer_mask)
    tt_logits = ttnn.to_torch(tt_logits_tt).float()
    print(f"  TT logits: {tt_logits.shape}, range [{tt_logits.min():.4f}, {tt_logits.max():.4f}]", flush=True)

    # Compare intermediate TT values from train cache
    tc = tt_model._train_cache
    tt_encoded = ttnn.to_torch(tc["enc_pre_norm"]).float()
    print(f"  TT encoded (pre_norm): range [{tt_encoded.min():.4f}, {tt_encoded.max():.4f}]", flush=True)
    enc_diff = (cpu_encoded - tt_encoded).abs().max().item()
    print(f"  ENCODER max diff: {enc_diff:.6f}", flush=True)

    tt_mem_states = [ttnn.to_torch(ms).float() for ms in tc["memory_states"]]
    for k, ms in enumerate(tt_mem_states):
        print(f"    TT mem_state[{k}]: range [{ms.min():.4f}, {ms.max():.4f}], mean={ms.mean():.4f}", flush=True)
    if len(tt_mem_states) > 0 and len(cpu_mem_states) > 0:
        mem_diff = (cpu_mem_states[0] - tt_mem_states[0]).abs().max().item()
        print(f"  MEMORY_INIT max diff: {mem_diff:.6f}", flush=True)
        for k in range(min(len(cpu_mem_states), len(tt_mem_states))):
            diff = (cpu_mem_states[k] - tt_mem_states[k]).abs().max().item()
            print(f"  mem_state[{k}] max diff: {diff:.6f}", flush=True)

    # --- Compare logits ---
    max_abs_diff = (cpu_logits - tt_logits).abs().max().item()
    mean_abs_diff = (cpu_logits - tt_logits).abs().mean().item()
    rel_diff = max_abs_diff / (cpu_logits.abs().max().item() + 1e-8)
    print(f"\n=== Forward Parity ===", flush=True)
    print(f"Max abs diff:  {max_abs_diff:.6f}", flush=True)
    print(f"Mean abs diff: {mean_abs_diff:.6f}", flush=True)
    print(f"Relative diff: {rel_diff:.6f}", flush=True)

    # Check token predictions match
    cpu_preds = cpu_logits.argmax(-1)
    tt_preds = tt_logits.argmax(-1)
    pred_match = (cpu_preds == tt_preds).float().mean().item()
    print(f"Prediction match: {pred_match:.4f}", flush=True)

    # --- Backward parity ---
    print("\nRunning CPU backward...", flush=True)
    cpu_model.zero_grad()
    cpu_output = cpu_model(prompt_ids, prompt_mask, decoder_input, answer_mask)
    cpu_loss = torch.nn.functional.cross_entropy(
        cpu_output["logits"].reshape(-1, vocab_size),
        answer_targets.reshape(-1),
        reduction="mean",
    )
    cpu_loss.backward()
    cpu_grad_norm = sum(p.grad.norm().item() ** 2 for p in cpu_model.parameters() if p.grad is not None) ** 0.5
    print(f"CPU loss: {cpu_loss.item():.6f}, grad norm: {cpu_grad_norm:.6f}", flush=True)

    print("Running TT backward...", flush=True)
    # Compute loss gradient on host (same as training script)
    tt_logits_for_loss = ttnn.to_torch(tt_logits_tt).float()
    tt_loss = torch.nn.functional.cross_entropy(
        tt_logits_for_loss.reshape(-1, vocab_size),
        answer_targets.reshape(-1),
        reduction="mean",
    )
    print(f"TT loss: {tt_loss.item():.6f}", flush=True)

    # Compute grad_logits
    grad_logits_torch = torch.nn.functional.cross_entropy(
        tt_logits_for_loss.reshape(-1, vocab_size),
        answer_targets.reshape(-1),
        reduction="mean",
    )
    # Manual gradient: (softmax - onehot) / N
    probs = torch.softmax(tt_logits_for_loss.reshape(-1, vocab_size), dim=-1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, answer_targets.reshape(-1, 1), 1.0)
    grad_logits_torch = (probs - onehot) / (answer_mask.sum().item())
    grad_logits_torch = grad_logits_torch.reshape(*tt_logits_for_loss.shape)

    grad_logits_tt = to_device(grad_logits_torch, device, dtype=ttnn.float32)
    _safe_deallocate_fn = getattr(ttnn, "deallocate", None)

    tt_grads = tt_model.backward(grad_logits_tt)
    tt_grad_norm = 0.0
    for name, g in tt_grads.items():
        g_host = ttnn.to_torch(g).float()
        tt_grad_norm += (g_host.norm().item() ** 2)
    tt_grad_norm = tt_grad_norm ** 0.5
    print(f"TT grad norm: {tt_grad_norm:.6f}", flush=True)

    # --- Compare key gradients ---
    print(f"\n=== Backward Parity ===", flush=True)
    print(f"CPU grad norm: {cpu_grad_norm:.6f}", flush=True)
    print(f"TT grad norm:  {tt_grad_norm:.6f}", flush=True)
    print(f"Grad norm ratio: {tt_grad_norm / (cpu_grad_norm + 1e-8):.4f}", flush=True)

    # Compare specific parameter gradients
    comparisons = [
        ("token_emb", cpu_model.token_embedding.weight.grad, tt_grads.get("token_emb_weight")),
        ("encoder_norm", cpu_model.encoder_norm.weight.grad, tt_grads.get("encoder_norm_weight")),
        ("slot_queries", cpu_model.slot_queries.grad.sum(0), tt_grads.get("slot_queries")),
        ("memory_norm", cpu_model.memory_norm.weight.grad, tt_grads.get("memory_norm_weight")),
        ("trans_gate_w", cpu_model.transition.gate.weight.grad, tt_grads.get("trans_gate_weight")),
        ("trans_gate_b", cpu_model.transition.gate.bias.grad, tt_grads.get("trans_gate_bias")),
        ("trans_ffn_w", cpu_model.transition.mlp[0].weight.grad, tt_grads.get("trans_ffn_weight")),
        ("trans_ffn_out_w", cpu_model.transition.mlp[2].weight.grad, tt_grads.get("trans_ffn_out_weight")),
    ]

    # Add encoder layer gradients
    for i in range(n_encoder_layers):
        enc = cpu_model.encoder.layers[i]
        comparisons.append((
            f"enc_{i}_in_proj_w",
            enc.self_attn.in_proj_weight.grad,
            tt_grads.get(f"enc_{i}_in_proj_weight"),
        ))
        comparisons.append((
            f"enc_{i}_linear1_w",
            enc.linear1.weight.grad,
            tt_grads.get(f"enc_{i}_linear1_weight"),
        ))

    # Add decoder layer gradients
    for i in range(n_decoder_layers):
        dec = cpu_model.decoder.layers[i]
        comparisons.append((
            f"dec_{i}_sa_in_proj_w",
            dec.self_attn.in_proj_weight.grad,
            tt_grads.get(f"dec_{i}_sa_in_proj_weight"),
        ))
        comparisons.append((
            f"dec_{i}_ca_in_proj_w",
            dec.multihead_attn.in_proj_weight.grad,
            tt_grads.get(f"dec_{i}_ca_in_proj_weight"),
        ))

    print(f"\n{'Parameter':<25} {'CPU norm':>10} {'TT norm':>10} {'Max abs diff':>14} {'Mean abs diff':>14}", flush=True)
    print("-" * 80, flush=True)

    total_max_diff = 0.0
    total_mean_diff = 0.0
    n_compared = 0

    for name, cpu_grad, tt_grad in comparisons:
        if cpu_grad is None or tt_grad is None:
            print(f"{name:<25} {'N/A':>10} {'N/A':>10} {'SKIPPED':>14}", flush=True)
            continue

        tt_grad_host = ttnn.to_torch(tt_grad).float()

        # Handle layout differences (TT weights are transposed)
        if "in_proj" in name and "w" in name:
            # TT grad is (d, 3d), CPU grad is (3d, d) — transpose TT
            tt_grad_host = tt_grad_host.t().contiguous()
        elif "linear" in name and "w" in name:
            # TT grad is (in, out), CPU grad is (out, in) — transpose TT
            tt_grad_host = tt_grad_host.t().contiguous()
        elif "ffn" in name and "w" in name:
            tt_grad_host = tt_grad_host.t().contiguous()
        elif "gate" in name and "w" in name:
            tt_grad_host = tt_grad_host.t().contiguous()
        elif "out_proj" in name and "w" in name:
            tt_grad_host = tt_grad_host.t().contiguous()

        if cpu_grad.shape != tt_grad_host.shape:
            print(f"{name:<25} {cpu_grad.norm().item():>10.4f} {tt_grad_host.norm().item():>10.4f} "
                  f"SHAPE MISMATCH: CPU {tuple(cpu_grad.shape)} vs TT {tuple(tt_grad_host.shape)}", flush=True)
            continue

        max_diff = (cpu_grad - tt_grad_host).abs().max().item()
        mean_diff = (cpu_grad - tt_grad_host).abs().mean().item()
        cpu_norm = cpu_grad.norm().item()
        tt_norm = tt_grad_host.norm().item()
        print(f"{name:<25} {cpu_norm:>10.4f} {tt_norm:>10.4f} {max_diff:>14.6f} {mean_diff:>14.6f}", flush=True)
        total_max_diff = max(total_max_diff, max_diff)
        total_mean_diff += mean_diff
        n_compared += 1

    print(f"\n=== Summary ===", flush=True)
    print(f"Parameters compared: {n_compared}", flush=True)
    print(f"Max abs diff across all: {total_max_diff:.6f}", flush=True)
    print(f"Mean abs diff (avg):     {total_mean_diff / max(n_compared, 1):.6f}", flush=True)

    if total_max_diff < 0.1:
        print("\nPASS: Gradients match within tolerance", flush=True)
    elif total_max_diff < 1.0:
        print("\nCLOSE: Gradients are approximately matching (bf16 noise)", flush=True)
    else:
        print("\nFAIL: Gradients differ significantly — investigate backward implementation", flush=True)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
