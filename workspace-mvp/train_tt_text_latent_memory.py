#!/usr/bin/env python3
"""Train the text latent-memory model and evaluate on Tenstorrent hardware.

Strategy:
  1. Train using the PyTorch TextLatentMemoryModel (CPU, with autograd)
  2. Transfer weights to the TT-NN model for hardware-accelerated evaluation
  3. This validates the TT-NN forward pass produces correct results
  4. The TT-NN model can later be used for inference on hardware

This approach:
  - Gets correct training (PyTorch autograd handles backward)
  - Validates the TT-NN implementation (forward parity check)
  - No memory leaks in training (pure PyTorch)
  - TT-NN memory management tested during eval

Usage:
    # Train + evaluate on hardware
    /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_tt_text_latent_memory.py \
        --steps 2000 --batch-size 64 --d-model 128 --device 0

    # Smoke test (3 steps)
    /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_tt_text_latent_memory.py \
        --smoke-test --device 0
"""

import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import sys

# ── TT_VISIBLE_DEVICES must be set BEFORE any ttnn/torch import ──────────────
_device_id_from_argv = 0
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _device_id_from_argv = int(sys.argv[_i + 1])
        break
    if _a.startswith("--device="):
        _device_id_from_argv = int(_a.split("=", 1)[1])
        break
os.environ.setdefault("TT_VISIBLE_DEVICES", str(_device_id_from_argv))

# P300 fabric mesh graph descriptor setup (same as train_ttnn.py)
_P300_SUBSYSTEM_IDS = {"0x0044", "0x0045", "0x0046"}
def _is_p300():
    try:
        from pathlib import Path
        for entry in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
            sub = (entry / "device" / "subsystem_device").read_text().strip().lower()
            if sub in _P300_SUBSYSTEM_IDS:
                return True
    except Exception:
        pass
    return False
def _find_mesh_graph_descriptor():
    try:
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
    except Exception:
        pass
    return None
if _is_p300():
    _mgd = _find_mesh_graph_descriptor()
    if _mgd:
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

import argparse
import random
import time
import gc
from pathlib import Path

import torch
import ttnn

MVP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MVP_DIR)

from text_latent_memory_model import TextLatentMemoryConfig, TextLatentMemoryModel
from tt_text_latent_memory_model import (
    TTTextLatentMemoryConfig,
    TTTextLatentMemoryModel,
    _safe_deallocate,
    to_device,
    from_device,
)
from challenge_data import ChallengeDataset, exact_match_accuracy


def transfer_pytorch_to_tt(pt_model: TextLatentMemoryModel, tt_model: TTTextLatentMemoryModel, device):
    """Transfer PyTorch model weights to TT-NN model.

    PyTorch Linear weights are (out, in). TT-NN linear weights are (in, out).
    So we need to transpose all linear weights during transfer.
    RMSNorm weights and embeddings are 1D, no transpose needed.
    """
    pt_state = pt_model.state_dict()

    # Determine which params need transposition (2D linear weights)
    # All params ending in _w that are 2D need transpose, except embeddings
    transpose_params = set()
    for tt_name in tt_model.get_params():
        pt_name = _map_tt_to_pt_name(tt_name)
        if pt_name and pt_name in pt_state:
            pt_w = pt_state[pt_name]
            if pt_w.dim() == 2 and "emb" not in tt_name and "queries" not in tt_name:
                transpose_params.add(tt_name)

    transfers = 0
    for tt_name in tt_model.get_params():
        pt_name = _map_tt_to_pt_name(tt_name)
        if pt_name and pt_name in pt_state:
            pt_w = pt_state[pt_name].to(torch.bfloat16)
            if tt_name in transpose_params:
                pt_w = pt_w.t().contiguous()  # (out, in) -> (in, out)
            new_tt = to_device(pt_w, device)
            tt_model._set_param(tt_name, new_tt)
            transfers += 1

    return transfers


def _map_tt_to_pt_name(tt_name: str) -> str:
    """Map TT-NN parameter name to PyTorch parameter name."""
    mapping = {
        "token_emb_weight": "token_embedding.weight",
        "prompt_pos_emb": "prompt_pos_embedding.weight",
        "answer_pos_emb": "answer_pos_embedding.weight",
        "slot_queries": "slot_queries",
        "encoder_norm_weight": "encoder_norm.weight",
        "memory_norm_weight": "memory_norm.weight",
        "decoder_norm_weight": "decoder_norm.weight",
    }

    if tt_name in mapping:
        return mapping[tt_name]

    # Encoder layers: enc_{i}_{suffix}
    if tt_name.startswith("enc_"):
        parts = tt_name.split("_", 2)
        idx = int(parts[1])
        suffix = parts[2]
        prefix = f"encoder.layers.{idx}"
        return _map_enc_dec_param(suffix, prefix)

    # Memory init attention
    if tt_name.startswith("mem_init_"):
        suffix = tt_name[len("mem_init_"):]
        return _map_attn_param(suffix, "memory_init")

    # Transition
    if tt_name.startswith("trans_"):
        suffix = tt_name[len("trans_"):]
        if suffix == "gate_w": return "transition.gate.weight"
        if suffix == "gate_b": return "transition.gate.bias"
        if suffix == "attn_norm_w": return "transition.attn_norm.weight"
        if suffix == "output_norm_w": return "transition.output_norm.weight"
        if suffix == "ffn_w": return "transition.mlp.0.weight"
        if suffix == "ffn_out_w": return "transition.mlp.2.weight"
        # Attention params
        return _map_attn_param(suffix, "transition.self_attn")

    # Decoder layers: dec_{i}_{suffix}
    if tt_name.startswith("dec_"):
        parts = tt_name.split("_", 2)
        idx = int(parts[1])
        suffix = parts[2]
        prefix = f"decoder.layers.{idx}"
        return _map_enc_dec_param(suffix, prefix)

    return None


def _map_attn_param(suffix, prefix):
    """Map attention parameter suffix to PyTorch name."""
    mapping = {
        "in_proj_w": f"{prefix}.in_proj_weight",
        "in_proj_b": f"{prefix}.in_proj_bias",
        "out_proj_w": f"{prefix}.out_proj.weight",
        "out_proj_b": f"{prefix}.out_proj.bias",
    }
    return mapping.get(suffix)


def _map_enc_dec_param(suffix, prefix):
    """Map encoder/decoder layer parameter suffix to PyTorch name."""
    # Self-attention
    if suffix.startswith("sa_"):
        return _map_attn_param(suffix[3:], f"{prefix}.self_attn")
    # Cross-attention
    if suffix.startswith("ca_"):
        return _map_attn_param(suffix[3:], f"{prefix}.multihead_attn")
    # Norms
    norm_map = {
        "norm1_w": f"{prefix}.norm1.weight", "norm1_b": f"{prefix}.norm1.bias",
        "norm2_w": f"{prefix}.norm2.weight", "norm2_b": f"{prefix}.norm2.bias",
        "norm3_w": f"{prefix}.norm3.weight", "norm3_b": f"{prefix}.norm3.bias",
    }
    if suffix in norm_map:
        return norm_map[suffix]
    # FFN
    ffn_map = {
        "linear1_w": f"{prefix}.linear1.weight", "linear1_b": f"{prefix}.linear1.bias",
        "linear2_w": f"{prefix}.linear2.weight", "linear2_b": f"{prefix}.linear2.bias",
    }
    if suffix in ffn_map:
        return ffn_map[suffix]
    return None


def evaluate_tt(tt_model, dataset, batch_size, n_batches, device, seed):
    """Evaluate TT-NN model on validation set."""
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    eval_rng = random.Random(seed)

    with torch.no_grad():
        for _ in range(n_batches):
            prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
                dataset.sample_batch(batch_size, "valid", eval_rng)

            # Forward on TT device
            logits = tt_model.forward(prompt_ids, prompt_mask, dec_input, ans_mask)

            # Loss on host
            V = logits.size(-1)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, V), ans_targets.reshape(-1),
                reduction='none',
            ).reshape(ans_targets.shape)
            loss = (loss * ans_mask.float()).sum() / ans_mask.float().sum().clamp_min(1)
            total_loss += loss.item() * ans_mask.float().sum().item()

            # Token accuracy
            preds = logits.argmax(-1)
            acc = (preds == ans_targets).float() * ans_mask.float()
            total_correct += acc.sum().item()
            total_examples += ans_mask.float().sum().item()

            # Clear device caches
            ttnn.synchronize_device(device)
            tt_model.clear_caches()

    avg_loss = total_loss / max(total_examples, 1)
    token_acc = total_correct / max(total_examples, 1)
    return avg_loss, token_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-encoder-layers", type=int, default=3)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--max-reasoning-steps", type=int, default=6)
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-answer-len", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints/tt_text_lm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tt-eval-only", action="store_true",
                        help="Skip training, just load checkpoint and eval on TT")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from PyTorch checkpoint")
    args = parser.parse_args()

    if args.smoke_test:
        args.steps = 3
        args.eval_interval = 999999

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # Load data
    data_dir = Path(MVP_DIR) / "data"
    dataset = ChallengeDataset(
        train_path=str(data_dir / "tiny_challenges_train.txt"),
        valid_path=str(data_dir / "tiny_challenges_valid.txt"),
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # PyTorch model for training (CPU, with autograd)
    pt_config = TextLatentMemoryConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        n_slots=args.n_slots,
        max_reasoning_steps=args.max_reasoning_steps,
        expand=args.expand,
        max_answer_len=args.max_answer_len,
        pad_token_id=dataset.pad_id,
    )
    pt_model = TextLatentMemoryModel(pt_config)
    n_params = sum(p.numel() for p in pt_model.parameters())
    print(f"PyTorch model: {n_params:,} params ({n_params/1e6:.1f}M)", flush=True)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, weights_only=False)
        pt_model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}", flush=True)

    optimizer = torch.optim.AdamW(
        pt_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / args.warmup_steps
        return args.lr * (1.0 - (step - start_step) / (args.steps - start_step))

    # Open TT device for evaluation
    device = ttnn.open_device(device_id=0)
    print(f"TT Device: {device}", flush=True)

    # Build TT model for hardware evaluation
    tt_config = TTTextLatentMemoryConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        n_slots=args.n_slots,
        max_reasoning_steps=args.max_reasoning_steps,
        expand=args.expand,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        pad_token_id=dataset.pad_id,
    )
    tt_model = TTTextLatentMemoryModel(tt_config, device)
    tt_n_params = tt_model.get_num_params()
    print(f"TT model: {tt_n_params:,} params ({tt_n_params/1e6:.1f}M)", flush=True)

    # Transfer initial weights to TT model
    n_transferred = transfer_pytorch_to_tt(pt_model, tt_model, device)
    print(f"Transferred {n_transferred} params to TT model", flush=True)

    def evaluate_pt():
        """Evaluate PyTorch model on validation set."""
        pt_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        eval_rng = random.Random(args.seed + 99999)
        with torch.no_grad():
            for _ in range(args.eval_batches):
                prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
                    dataset.sample_batch(args.eval_batch_size, "valid", eval_rng)
                outputs = pt_model(prompt_ids, prompt_mask, dec_input, ans_mask)
                losses = pt_model.loss(outputs, ans_targets, ans_mask)
                total_loss += losses["answer_loss"].item() * ans_mask.float().sum().item()
                preds = outputs["logits"].argmax(-1)
                acc = (preds == ans_targets).float() * ans_mask.float()
                total_correct += acc.sum().item()
                total_examples += ans_mask.float().sum().item()
        pt_model.train()
        return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)

    # Training loop (PyTorch on CPU)
    start_time = time.time()
    for step in range(start_step, args.steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample batch
        prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
            dataset.sample_batch(args.batch_size, "train", rng)

        # Forward + backward (PyTorch)
        outputs = pt_model(prompt_ids, prompt_mask, dec_input, ans_mask)
        losses = pt_model.loss(outputs, ans_targets, ans_mask)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(pt_model.parameters(), args.grad_clip)
        optimizer.step()

        # Log
        if step % 100 == 0 or args.smoke_test:
            elapsed = time.time() - start_time
            preds = outputs["logits"].argmax(-1)
            tok_acc = ((preds == ans_targets).float() * ans_mask.float()).sum() / \
                       ans_mask.float().sum().clamp_min(1)
            print(
                f"{step:6d} loss={losses['loss'].item():.4f} "
                f"tok_acc={tok_acc.item():.3f} "
                f"grad={float(grad_norm):.3f} "
                f"lr={lr:.2e} "
                f"t={elapsed:.0f}s",
                flush=True,
            )

        # Evaluate
        if step > 0 and step % args.eval_interval == 0:
            # PyTorch eval
            pt_loss, pt_acc = evaluate_pt()
            print(f"pt_eval step={step} val_loss={pt_loss:.4f} val_tok_acc={pt_acc:.3f}", flush=True)

            # Transfer weights to TT model
            transfer_pytorch_to_tt(pt_model, tt_model, device)

            # TT eval (validates forward parity on hardware)
            tt_loss, tt_acc = evaluate_tt(tt_model, dataset, args.eval_batch_size,
                                          args.eval_batches, device, args.seed + step)
            print(f"tt_eval step={step} val_loss={tt_loss:.4f} val_tok_acc={tt_acc:.3f}", flush=True)

            # Check parity
            loss_diff = abs(pt_loss - tt_loss)
            acc_diff = abs(pt_acc - tt_acc)
            if loss_diff < 0.1 and acc_diff < 0.05:
                print(f"  TT parity: OK (loss_diff={loss_diff:.4f}, acc_diff={acc_diff:.4f})", flush=True)
            else:
                print(f"  TT parity: MISMATCH (loss_diff={loss_diff:.4f}, acc_diff={acc_diff:.4f})", flush=True)

            # Save checkpoint
            torch.save(
                {"step": step, "config": pt_config, "model": pt_model.state_dict()},
                checkpoint_dir / f"text_lm_step{step}.pt",
            )
            tt_model.save_checkpoint(str(checkpoint_dir / f"tt_text_lm_step{step}.pt"), step=step)

        # Periodic GC
        if step % 50 == 0:
            gc.collect()

    # Final eval
    pt_loss, pt_acc = evaluate_pt()
    print(f"final pt_eval val_loss={pt_loss:.4f} val_tok_acc={pt_acc:.3f}", flush=True)

    transfer_pytorch_to_tt(pt_model, tt_model, device)
    tt_loss, tt_acc = evaluate_tt(tt_model, dataset, args.eval_batch_size,
                                  args.eval_batches, device, args.seed + 99999)
    print(f"final tt_eval val_loss={tt_loss:.4f} val_tok_acc={tt_acc:.3f}", flush=True)

    # Save final checkpoints
    torch.save(
        {"step": args.steps, "config": pt_config, "model": pt_model.state_dict()},
        checkpoint_dir / "text_lm_final.pt",
    )
    tt_model.save_checkpoint(str(checkpoint_dir / "tt_text_lm_final.pt"), step=args.steps)

    # Cleanup
    ttnn.synchronize_device(device)
    tt_model.clear_caches()
    ttnn.close_device(device)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
