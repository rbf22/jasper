#!/usr/bin/env python3
"""Profile a single TT training step to identify host/device bottlenecks."""
import os, sys, time, torch

# Mesh descriptor setup (same as train_native_tt.py)
def _find_mesh_graph_descriptor():
    import importlib.util
    from pathlib import Path
    try:
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
    except Exception:
        pass
    return None

def _is_p300():
    try:
        import subprocess
        r = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=5)
        return "p300" in r.stdout.lower()
    except Exception:
        return False

# Always try to set mesh descriptor for P300
_mgd = _find_mesh_graph_descriptor()
if _mgd:
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = _mgd

import ttnn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tt_text_latent_memory_model import TTTextLatentMemoryConfig, TTTextLatentMemoryModel
from challenge_data import ChallengeDataset

def to_device(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

def main():
    device = ttnn.open_device(device_id=0)

    config = TTTextLatentMemoryConfig(
        d_model=384, n_encoder_layers=4, n_decoder_layers=2,
        n_heads=4, n_slots=16, max_reasoning_steps=6, expand=4,
        max_prompt_len=256, max_answer_len=32,
    )
    model = TTTextLatentMemoryModel(config, device, dtype=ttnn.bfloat16)
    print(f"Model: {model.get_num_params():,} params (bf16)")

    dataset = ChallengeDataset(
        train_path="data/tiny_challenges_train.txt",
        valid_path="data/tiny_challenges_valid.txt",
        max_prompt_len=256, max_answer_len=32,
    )

    # Warmup (compile kernels)
    prompt_ids, prompt_mask, dec_in, ans_tgt, ans_mask = dataset.sample_batch(batch_size=4)
    print("Warmup forward...")
    t0 = time.time()
    logits_tt = model.forward_train(prompt_ids, prompt_mask, dec_in, ans_mask)
    ttnn.synchronize_device(device)
    print(f"  Warmup forward: {time.time()-t0:.2f}s")

    # Compute loss + grad on host
    logits_torch = ttnn.to_torch(logits_tt).float()
    B, T, V = logits_torch.shape
    loss = torch.nn.functional.cross_entropy(logits_torch.reshape(-1, V), ans_tgt.reshape(-1), reduction="mean")
    probs = torch.softmax(logits_torch.reshape(-1, V), dim=-1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, ans_tgt.reshape(-1, 1), 1.0)
    grad_logits = ((probs - onehot) / ans_mask.sum().item()).reshape(B, T, V)
    grad_logits_tt = to_device(grad_logits, device, dtype=ttnn.bfloat16)
    print("Warmup backward...")
    t0 = time.time()
    grads = model.backward(grad_logits_tt)
    ttnn.synchronize_device(device)
    print(f"  Warmup backward: {time.time()-t0:.2f}s")

    # Clear caches
    model._train_cache = {}

    # Now profile a real step
    print("\n=== Profiling real step ===")

    # 1. Data sampling
    t0 = time.time()
    prompt_ids, prompt_mask, dec_in, ans_tgt, ans_mask = dataset.sample_batch(batch_size=4)
    t_data = time.time() - t0

    # 2. Forward
    t0 = time.time()
    logits_tt = model.forward_train(prompt_ids, prompt_mask, dec_in, ans_mask)
    ttnn.synchronize_device(device)
    t_fwd = time.time() - t0

    # 3. Loss + grad_logits on host
    t0 = time.time()
    logits_torch = ttnn.to_torch(logits_tt).float()
    B, T, V = logits_torch.shape
    loss = torch.nn.functional.cross_entropy(logits_torch.reshape(-1, V), ans_tgt.reshape(-1), reduction="mean")
    probs = torch.softmax(logits_torch.reshape(-1, V), dim=-1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, ans_tgt.reshape(-1, 1), 1.0)
    grad_logits = ((probs - onehot) / ans_mask.sum().item()).reshape(B, T, V)
    grad_logits_tt = to_device(grad_logits, device, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)
    t_loss = time.time() - t0

    # 4. Backward
    t0 = time.time()
    grads = model.backward(grad_logits_tt)
    ttnn.synchronize_device(device)
    t_bwd = time.time() - t0

    # 5. Optimizer step (host-side AdamW)
    t0 = time.time()
    # Just measure gradient transfer to host
    grad_norm = 0.0
    for name, g in grads.items():
        g_host = ttnn.to_torch(g).float()
        grad_norm += g_host.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    t_opt = time.time() - t0

    total = t_data + t_fwd + t_loss + t_bwd + t_opt

    print(f"\n{'Stage':<25} {'Time (s)':>10} {'% Total':>10}")
    print("-" * 47)
    print(f"{'Data sampling':<25} {t_data:>10.3f} {t_data/total*100:>9.1f}%")
    print(f"{'Forward (device)':<25} {t_fwd:>10.3f} {t_fwd/total*100:>9.1f}%")
    print(f"{'Loss + grad_logits (host)':<25} {t_loss:>10.3f} {t_loss/total*100:>9.1f}%")
    print(f"{'Backward (mixed host/dev)':<25} {t_bwd:>10.3f} {t_bwd/total*100:>9.1f}%")
    print(f"{'Grad transfer + opt (host)':<25} {t_opt:>10.3f} {t_opt/total*100:>9.1f}%")
    print("-" * 47)
    print(f"{'Total':<25} {total:>10.3f} {'100.0%':>10}")
    print(f"\nLoss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")

    # Count host transfers in backward
    # The attention backward does ttnn.to_torch for all cached tensors
    # Let's estimate the data transfer volume
    print(f"\n=== Host transfer estimate ===")
    print(f"Logits transfer (fwd->host): {B*T*V*4 / 1e6:.1f} MB")
    print(f"Grad_logits transfer (host->dev): {B*T*V*4 / 1e6:.1f} MB")

    # Attention backward transfers: q, k, v, attn, out_pre, query, key, grad_out
    # For each attention layer
    d = config.d_model
    H = config.n_heads
    n_enc = config.n_encoder_layers
    n_dec = config.n_decoder_layers
    K = config.max_reasoning_steps
    S = config.n_slots
    L_p = config.max_prompt_len
    L_a = config.max_answer_len

    # Encoder self-attn: q,k,v = (B, H, L_p, d/H), attn = (B, H, L_p, L_p)
    enc_attn_transfer = B * (3 * H * L_p * d // H + H * L_p * L_p + L_p * d) * 4  # qkv + attn + out_pre
    enc_attn_transfer += B * L_p * d * 4  # query input
    enc_attn_transfer *= n_enc

    # Memory init cross-attn: q=(B,H,S,d/H), k,v=(B,H,L_p,d/H), attn=(B,H,S,L_p)
    mem_init_transfer = B * (H * S * d // H + 2 * H * L_p * d // H + H * S * L_p + S * d) * 4
    mem_init_transfer += B * (S * d + L_p * d) * 4  # query + key inputs

    # Transition self-attn (K steps): q,k,v=(B,H,S,d/H), attn=(B,H,S,S)
    trans_attn_transfer = B * (3 * H * S * d // H + H * S * S + S * d) * 4
    trans_attn_transfer += B * S * d * 4  # query input
    trans_attn_transfer *= K

    # Decoder self-attn: q,k,v=(B,H,L_a,d/H), attn=(B,H,L_a,L_a)
    dec_sa_transfer = B * (3 * H * L_a * d // H + H * L_a * L_a + L_a * d) * 4
    dec_sa_transfer += B * L_a * d * 4
    dec_sa_transfer *= n_dec

    # Decoder cross-attn: q=(B,H,L_a,d/H), k,v=(B,H,S,d/H), attn=(B,H,L_a,S)
    dec_ca_transfer = B * (H * L_a * d // H + 2 * H * S * d // H + H * L_a * S + L_a * d) * 4
    dec_ca_transfer += B * (L_a * d + S * d) * 4
    dec_ca_transfer *= n_dec

    # Norm backward transfers (input + grad_out -> host, grad_x -> device)
    # Each norm: 2 * B * L * d * 4 bytes (input + grad_out in, grad_x out)
    norm_transfer = 0
    # Encoder: norm1, norm2 per layer + encoder_norm
    norm_transfer += n_enc * 2 * 2 * B * L_p * d * 4  # 2 norms, in+out
    norm_transfer += 2 * B * L_p * d * 4  # encoder_norm
    # Memory: memory_norm
    norm_transfer += 2 * B * S * d * 4
    # Transition: attn_norm (2x), output_norm per step
    norm_transfer += K * (2 * 2 * B * S * d * 4 + 2 * B * S * d * 4)
    # Decoder: norm1, norm2, norm3 per layer + decoder_norm
    norm_transfer += n_dec * 3 * 2 * B * L_a * d * 4
    norm_transfer += 2 * B * L_a * d * 4  # decoder_norm

    total_attn_transfer = enc_attn_transfer + mem_init_transfer + trans_attn_transfer + dec_sa_transfer + dec_ca_transfer
    total_transfer = total_attn_transfer + norm_transfer

    print(f"Attention backward transfers: {total_attn_transfer / 1e6:.1f} MB")
    print(f"  Encoder attn:     {enc_attn_transfer / 1e6:.1f} MB")
    print(f"  Memory init attn: {mem_init_transfer / 1e6:.1f} MB")
    print(f"  Transition attn:  {trans_attn_transfer / 1e6:.1f} MB")
    print(f"  Decoder self attn:{dec_sa_transfer / 1e6:.1f} MB")
    print(f"  Decoder cross attn:{dec_ca_transfer / 1e6:.1f} MB")
    print(f"Norm backward transfers: {norm_transfer / 1e6:.1f} MB")
    print(f"Total backward host transfer: {total_transfer / 1e6:.1f} MB")
    print(f"Total forward+loss transfer: {(B*T*V*4 + B*T*V*4) / 1e6:.1f} MB")

    ttnn.close_device(device)

if __name__ == "__main__":
    main()
