#!/usr/bin/env python3
"""Train the Mamba Workspace model on text data (TinyStories).

Adapted from mamba-poc/train_ttnn.py — same training loop, optimizer,
gradient clipping, and checkpoint logic, but uses a real text dataset
(TinyStories, GPT-2 BPE) instead of synthetic arithmetic tasks.

The model architecture (model_ttnn.py) is unchanged — it's task-agnostic.
Only the data pipeline differs.

Usage:
    # Smoke test (3 steps)
    TT_VISIBLE_DEVICES=1 python train_text.py \
        --config configs/text_cell_c.yaml --steps 3 \
        --micro_batch 4 --accum_steps 1 \
        --checkpoint_dir /tmp/text_test --device 0

    # Full training
    TT_VISIBLE_DEVICES=2 nohup python train_text.py \
        --config configs/text_cell_c.yaml --device 0 \
        --checkpoint_dir checkpoints \
        > logs/text_cell_c.log 2>&1 &
"""

import os
import sys
import argparse
import time
import random
import gc
import yaml
import torch
import ttnn

# Set up paths — workspace-mvp symlinks to workspace-poc for model code
# We need to import from both directories
MVP_DIR = os.path.dirname(os.path.abspath(__file__))
# Find the POC directory: try sibling "workspace-poc" (current) or "mamba-poc" (legacy)
POC_DIR = os.path.realpath(os.path.join(MVP_DIR, "..", "workspace-poc"))
if not os.path.isdir(POC_DIR):
    POC_DIR = os.path.realpath(os.path.join(MVP_DIR, "..", "mamba-poc"))

# Insert POC dir first so train_ttnn's internal imports work
sys.path.insert(0, POC_DIR)
sys.path.insert(0, MVP_DIR)

# Import shared infrastructure from train_ttnn.py
from train_ttnn import (
    load_config,
    build_model_config,
    cross_entropy_loss,
    clip_grad_norm,
    accumulate_grads,
    host_grads_to_tt,
    get_lr,
    Profiler,
    _safe_deallocate,
)
from model_ttnn import TTMambaWorkspaceModel, ModelConfig

# Import text data pipeline
from text_data import BPETokenizer, TextDataset, sample_text_batch, make_eval_batches


# ---------------------------------------------------------------------------
# Cross-entropy loss for large vocab sizes
# ---------------------------------------------------------------------------
# The on-device cross_entropy_loss in train_ttnn.py uses a V×V identity matrix
# for one-hot encoding, which is fine for V=128 (32KB) but impractical for
# V=50257 (~5GB in bfloat16). Two alternatives are provided:
#
# 1. cross_entropy_loss_scatter (on-device, preferred): Uses ttnn.gather to
#    extract target probs and ttnn.scatter_add to build the gradient — no
#    V×V identity matrix, no host transfer of the full logits tensor.
#
# 2. cross_entropy_loss_host (host-side, fallback): Computes loss and gradient
#    in float32 on the host. ~200MB logits transfer per micro-batch.

_HOST_LOSS_VOCAB_THRESHOLD = 2048  # use scatter/host loss above this vocab size


def cross_entropy_loss_scatter(logits_tt, labels, ignore_index=-100):
    """On-device cross-entropy loss using gather/scatter (no V×V identity matrix).

    Avoids the V×V identity matrix by using:
    - ttnn.gather to extract target probs (for loss value)
    - ttnn.scatter_add to subtract 1/n_valid at target positions (for gradient)

    The only host transfer is the gathered target probs (B×T×32 ≈ 65K elements)
    and the final loss scalar — vs ~200MB for the host-side loss.

    logits_tt: (B, T, V) tt-nn tensor on device
    labels: (B, T) PyTorch tensor (already shifted by 1, -100 at invalid positions)
    Returns: (loss_value, grad_logits_tt)
    """
    device = logits_tt.device()
    B, T = labels.shape
    V = logits_tt.shape[-1]  # already tile-padded on device

    # --- Host prep (small tensors) ---
    valid_mask = (labels != ignore_index).float()  # (B, T)
    n_valid = int(valid_mask.sum().item())
    n_valid = max(n_valid, 1)
    inv_n = 1.0 / n_valid

    safe_labels = labels.clamp(min=0).to(torch.int32)  # (B, T)

    # Prepare index tensor for gather/scatter: (B, T, 32) in TILE_LAYOUT
    # First channel = label, rest = 0 (padding, adds 0 via scatter_add)
    TILE = 32
    labels_padded = torch.zeros(B, T, TILE, dtype=torch.int32)
    labels_padded[:, :, 0] = safe_labels

    # Source for scatter_add: (B, T, 32) — first channel = -1/n_valid, rest = 0
    src_padded = torch.zeros(B, T, TILE, dtype=torch.bfloat16)
    src_padded[:, :, 0] = -inv_n

    # Transfer small tensors to device
    labels_tt = ttnn.from_torch(
        labels_padded, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
    )
    src_tt = ttnn.from_torch(
        src_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    mask_tt = ttnn.from_torch(
        valid_mask.unsqueeze(-1).to(torch.bfloat16),  # (B, T, 1)
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # --- Loss value (gather target probs, compute on host) ---
    probs = ttnn.softmax(logits_tt, dim=-1)  # (B, T, V)

    # Gather target probs: (B, T, 32) — only first channel matters
    target_probs_tt = ttnn.gather(probs, dim=-1, index=labels_tt)  # (B, T, 32)

    # Transfer to host (tiny: B*T*32 elements), compute loss
    target_probs = ttnn.to_torch(target_probs_tt).float()  # (B, T, 32)
    target_prob = target_probs[:, :, 0]  # (B, T) — prob at target position
    target_log_prob = torch.log(target_prob.clamp(min=1e-8))  # (B, T)
    loss_value = -(target_log_prob * valid_mask).sum().item() / n_valid

    # --- Gradient (fully on device) ---
    # grad = (probs - one_hot) / n_valid
    #      = probs/n_valid + scatter_add(-1/n_valid at target positions)
    inv_n_tt = ttnn.from_torch(
        torch.tensor([inv_n], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # REVIEWED: reassignment leak — each ttnn op creates a new tensor;
    # old values must be explicitly deallocated.
    grad = ttnn.mul(probs, inv_n_tt)  # (B, T, V) = probs / n_valid
    _safe_deallocate(probs)
    _safe_deallocate(inv_n_tt)
    grad_scaled = ttnn.scatter_add(grad, dim=-1, index=labels_tt, src=src_tt)  # subtract 1/n_valid at targets
    _safe_deallocate(grad)
    _safe_deallocate(labels_tt)
    _safe_deallocate(src_tt)
    grad_final = ttnn.mul(grad_scaled, mask_tt)  # zero out invalid positions (broadcasts (B,T,1) → (B,T,V))
    _safe_deallocate(grad_scaled)
    _safe_deallocate(mask_tt)
    # REVIEWED: target_probs_tt no longer needed (loss already computed on host)
    _safe_deallocate(target_probs_tt)

    return loss_value, grad_final


def cross_entropy_loss_host(logits_tt, labels, ignore_index=-100):
    """Compute cross-entropy loss and gradient on the host (fallback for large vocab).

    logits_tt: (B, T, V) tt-nn tensor on device
    labels: (B, T) PyTorch tensor
    Returns: (loss_value, grad_logits_tt)
    """
    device = logits_tt.device()
    B, T, V = labels.shape[0], labels.shape[1], logits_tt.shape[-1]

    # Move logits to host in float32
    logits = ttnn.to_torch(logits_tt).float()  # (B, T, V)

    # Shift: predict token t+1 from token t
    shift_logits = logits[:, :-1, :]  # (B, T-1, V)
    shift_labels = labels[:, :-1]  # (B, T-1)

    # Valid mask
    valid_mask = (shift_labels != ignore_index).float()  # (B, T-1)
    n_valid = int(valid_mask.sum().item())
    n_valid = max(n_valid, 1)

    # Safe labels
    safe_labels = shift_labels.clamp(min=0)  # (B, T-1)

    # Compute log-softmax
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # Gather log-prob at target positions
    flat_log_probs = log_probs.reshape(-1, V)  # (B*(T-1), V)
    flat_labels = safe_labels.reshape(-1)  # (B*(T-1),)
    target_log_probs = flat_log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)  # (B*(T-1),)

    # Loss = -mean of target_log_probs at valid positions
    target_log_probs = target_log_probs.reshape(B, T - 1)
    loss_value = -(target_log_probs * valid_mask).sum().item() / n_valid

    # Gradient: (softmax - one_hot) / n_valid, masked
    probs = torch.exp(log_probs)  # (B, T-1, V)
    # Subtract 1 at target positions
    grad = probs.clone()
    grad.reshape(-1, V).scatter_(1, flat_labels.unsqueeze(1), -1.0)  # probs - one_hot
    grad = grad.reshape(B, T - 1, V)
    grad = grad * valid_mask.unsqueeze(-1)  # zero out invalid positions
    grad = grad / n_valid

    # Pad to (B, T, V) — last position has zero gradient
    grad_padded = torch.zeros(B, T, V, dtype=torch.float32)
    grad_padded[:, :-1, :] = grad

    # Move gradient back to device in bfloat16
    grad_logits_tt = ttnn.from_torch(
        grad_padded.to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    return loss_value, grad_logits_tt


def compute_loss(logits_tt, labels, vocab_size, ignore_index=-100, use_scatter=True):
    """Dispatch to on-device scatter, host-side, or identity-matrix loss.

    For large vocab (V > threshold):
      - use_scatter=True (default): on-device scatter-based loss
      - use_scatter=False: host-side loss (fallback)
    For small vocab (V <= threshold): on-device identity-matrix loss from train_ttnn.py
    """
    if vocab_size > _HOST_LOSS_VOCAB_THRESHOLD:
        if use_scatter:
            return cross_entropy_loss_scatter(logits_tt, labels, ignore_index)
        else:
            return cross_entropy_loss_host(logits_tt, labels, ignore_index)
    else:
        return cross_entropy_loss(logits_tt, labels, ignore_index)


# ---------------------------------------------------------------------------
# Mesh graph descriptor setup (for P300 boards)
# ---------------------------------------------------------------------------

def setup_mesh_graph():
    """Set TT_MESH_GRAPH_DESC_PATH for P300 boards if not already set."""
    if "TT_MESH_GRAPH_DESC_PATH" in os.environ:
        return
    from pathlib import Path
    candidates = [
        "/home/rfenwick/tt-boltz/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto",
        "/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto",
    ]
    for c in candidates:
        if Path(c).is_file():
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = c
            return


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def evaluate_perplexity(model, eval_batches, device, k_value=None):
    """Compute average perplexity on eval batches.

    Returns (avg_loss, avg_perplexity).
    """
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in eval_batches:
        with torch.no_grad():
            logits_tt = model.forward(input_ids, k_value=k_value)
            # Compute loss manually (no gradient needed)
            logits = ttnn.to_torch(logits_tt)  # (B, T, V)
            # REVIEWED: Deallocate device logits after host transfer
            _safe_deallocate(logits_tt)
            shift_logits = logits[:, :-1, :]  # (B, T-1, V)
            shift_labels = labels[:, :-1]  # (B, T-1)
            valid_mask = (shift_labels != -100)
            n_valid = valid_mask.sum().item()
            if n_valid > 0:
                # Cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
                    shift_labels.clamp(min=0).reshape(-1),
                    reduction="none",
                )
                loss = (loss * valid_mask.reshape(-1)).sum().item()
                total_loss += loss
                total_tokens += n_valid
            # REVIEWED: Clear model caches after eval forward (no backward ran)
            model.clear_caches()
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str, steps_override=None, micro_batch_override=None,
          accum_steps_override=None, profile=False,
          checkpoint_dir_override=None, resume=None, device_id=0,
          loss_method="host"):
    """Train the model on text data using a YAML config."""
    setup_mesh_graph()

    cfg = load_config(config_path)
    cell = cfg.get("cell", "text")

    # Training hyperparams
    max_steps = steps_override if steps_override else cfg.get("max_steps", 10000)
    tokens_per_batch = cfg.get("tokens_per_batch", 50000)
    seq_len = cfg.get("seq_len", 512)
    base_lr = cfg.get("lr", 2e-4)
    warmup_steps = cfg.get("warmup_steps", 200)
    weight_decay = cfg.get("weight_decay", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    ws_grad_clip = cfg.get("ws_grad_clip", 0.5)
    grad_norm_spike_threshold = cfg.get("grad_norm_spike_threshold", 5000.0)
    seed = cfg.get("seed", 42)
    eval_interval = cfg.get("eval_interval", 500)
    log_interval = cfg.get("log_interval", 50)
    checkpoint_interval = cfg.get("checkpoint_interval", 500)
    ckpt_dir = checkpoint_dir_override or cfg.get("ckpt_dir", "checkpoints")

    # Data config
    train_data_path = cfg.get("train_data", "data/tinystories_train.txt")
    valid_data_path = cfg.get("valid_data", "data/tinystories_valid.txt")
    max_train_tokens = cfg.get("max_train_tokens", None)  # None = use all

    # Micro-batch size
    if micro_batch_override:
        micro_batch = micro_batch_override
    else:
        micro_batch = cfg.get("micro_batch_size", 0)

    # Accumulation steps
    if accum_steps_override:
        accum_steps = accum_steps_override
    else:
        accum_steps = cfg.get("accum_steps", 0)

    # Auto-compute batch size if not specified
    if micro_batch == 0 or accum_steps == 0:
        target_batch = cfg.get("effective_batch_size", 384)
        if micro_batch == 0:
            # Auto-detect: start with 8, reduce if OOM
            micro_batch = 8
        if accum_steps == 0:
            accum_steps = max(1, target_batch // micro_batch)

    effective_batch = micro_batch * accum_steps
    effective_tokens = effective_batch * seq_len

    # Open device
    device = ttnn.open_device(device_id=device_id)
    print(f"Device: {device}", flush=True)

    # Build model
    model_config = build_model_config(cfg)
    model_config.slot_permutation = False  # deterministic for text
    model = TTMambaWorkspaceModel(model_config, device)
    print(f"Cell: {cell}, Params: {model.get_num_params() / 1e6:.2f}M", flush=True)

    # Optimizer
    from train_ttnn import TTAdamW
    lr_groups = cfg.get("lr_groups", {})
    optimizer = TTAdamW(model.get_params(), lr=base_lr, weight_decay=weight_decay,
                        lr_groups=lr_groups if lr_groups else None)
    if lr_groups:
        ws_mult = lr_groups.get("ws_", 1.0)
        ws_lr = base_lr * ws_mult
        gate_mult = lr_groups.get("ws_read_gate", None)
        if gate_mult is not None:
            gate_lr = base_lr * gate_mult
            print(f"LR groups: workspace={ws_lr:.2e} ({ws_mult}x), gates={gate_lr:.2e} ({gate_mult}x), backbone={base_lr:.2e}", flush=True)
        else:
            print(f"LR groups: workspace={ws_lr:.2e} ({ws_mult}x), backbone={base_lr:.2e}", flush=True)

    # Resume from checkpoint
    start_step = 0
    if resume and os.path.exists(resume):
        opt_state = model.load_checkpoint(resume, device=device)
        if opt_state:
            optimizer.load_state(opt_state, model)
            start_step = optimizer.step_count
        print(f"Resumed from step {start_step}", flush=True)

    # Load text data
    tokenizer = BPETokenizer()
    print(f"Loading training data from {train_data_path}...", flush=True)
    train_dataset = TextDataset(train_data_path, tokenizer, max_tokens=max_train_tokens)
    print(f"Loading validation data from {valid_data_path}...", flush=True)
    valid_dataset = TextDataset(valid_data_path, tokenizer)

    # Create eval batches
    eval_batches = make_eval_batches(
        valid_dataset, seq_len=seq_len,
        n_batches=cfg.get("eval_batches", 10),
        batch_size=micro_batch,
    )

    # RNG
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Checkpoint directory
    os.makedirs(ckpt_dir, exist_ok=True)

    # Print training config
    print(f"\nTraining config:", flush=True)
    print(f"  micro_batch={micro_batch}, accum_steps={accum_steps}, "
          f"effective_batch={effective_batch}", flush=True)
    print(f"  seq_len={seq_len}, tokens_per_batch={tokens_per_batch}", flush=True)
    print(f"  effective_tokens/step={effective_tokens}", flush=True)
    print(f"  lr={base_lr}, warmup={warmup_steps}, weight_decay={weight_decay}, "
          f"grad_clip={grad_clip}, ws_grad_clip={ws_grad_clip}", flush=True)
    print(f"  max_steps={max_steps}", flush=True)
    print(f"  train_tokens={len(train_dataset):,}, valid_tokens={len(valid_dataset):,}",
          flush=True)
    print(f"  vocab_size={tokenizer.VOCAB_SIZE}", flush=True)
    print(f"  loss_method={loss_method}", flush=True)

    print(f"\n{'Step':>6} {'Loss':>10} {'PPL':>8} {'LR':>10} {'Time':>8} "
          f"{'tokens/s':>10} {'GradNorm':>10} {'gz_gr':>8} {'gz_gw':>8}",
          flush=True)

    # Early stopping
    plateau_patience = cfg.get("plateau_patience", 1000)
    plateau_min_delta = cfg.get("plateau_min_delta", 1e-3)
    plateau_ema_beta = 0.99
    best_loss_ema = float("inf")
    loss_ema = None
    steps_since_best = 0
    if plateau_patience > 0:
        print(f"  Early stopping: patience={plateau_patience}, min_delta={plateau_min_delta}",
              flush=True)

    profiler = Profiler()
    total_time = 0
    total_tokens_trained = 0
    skipped_steps = 0
    best_ppl = float("inf")

    for step in range(start_step, max_steps):
        t_step_start = time.time()

        # Update LR (warmup)
        current_lr = get_lr(step, base_lr, warmup_steps)
        optimizer.set_lr(current_lr)

        # Gradient accumulation
        accum_grads = {}
        step_loss = 0.0

        for accum_idx in range(accum_steps):
            # Sample text batch
            input_ids, labels, _ = sample_text_batch(
                micro_batch, seq_len, train_dataset, rng=rng
            )

            # Forward — sample K for recurrent core
            k_value = None
            if model_config.recurrent_core:
                k_value = random.randint(1, model_config.k_train_max)

            logits = model.forward(input_ids, k_value=k_value)

            # Loss + gradient (host-side for large vocab)
            use_scatter = (loss_method == "scatter")
            loss_val, grad_logits = compute_loss(logits, labels, model_config.vocab_size, use_scatter=use_scatter)
            step_loss += loss_val

            # REVIEWED: Deallocate logits (no longer needed after loss computation)
            _safe_deallocate(logits)

            # Backward
            grads = model.backward(grad_logits)

            # REVIEWED: Deallocate grad_logits (consumed by backward)
            _safe_deallocate(grad_logits)

            # Accumulate gradients (host-side)
            accumulate_grads(accum_grads, grads, accum_steps)

            # REVIEWED: Deallocate device-side gradient tensors (already copied to host)
            for g in grads.values():
                _safe_deallocate(g)

            # REVIEWED: Clear model's forward/backward caches to free intermediate tensors
            model.clear_caches()

        # Average loss over accumulation steps
        step_loss /= accum_steps

        # Convert to device tensors
        tt_grads = host_grads_to_tt(accum_grads, device)

        # Gradient clipping (component-wise)
        grad_norm = clip_grad_norm(
            tt_grads, grad_clip,
            ws_max_norm=ws_grad_clip if model_config.use_workspace else None
        )

        # Skip step if gradient spike detected
        if grad_norm > grad_norm_spike_threshold:
            print(f"  *** SKIP step {step}: grad_norm {grad_norm:.1f} > threshold {grad_norm_spike_threshold} ***",
                  flush=True)
            skipped_steps += 1
            # REVIEWED: Deallocate tt_grads before skipping (device tensors)
            for g in tt_grads.values():
                _safe_deallocate(g)
            del tt_grads
            del accum_grads
            continue

        # Optimizer step
        optimizer.step(tt_grads, model)

        # Post-step normalization
        if model_config.use_workspace:
            model.normalize_workspace_slots()
        model.spectral_normalize_backbone_weights()
        if not model_config.freeze_gamma:
            model.clamp_retention_gammas()
            optimizer.sync_master_from_model(model)

        # REVIEWED: Deallocate tt_grads (no longer needed after optimizer step)
        for g in tt_grads.values():
            _safe_deallocate(g)
        del tt_grads
        # Free host-side accumulated gradients
        del accum_grads
        # REVIEWED: Force garbage collection to reclaim any remaining orphaned
        # ttnn wrapper objects. Without this, device DRAM grows until OOM.
        gc.collect()

        t_step_end = time.time()
        step_time = t_step_end - t_step_start
        total_time += step_time
        total_tokens_trained += effective_tokens
        tokens_per_sec = effective_tokens / step_time

        # Periodic evaluation
        ppl = 0.0
        if step > 0 and (step % eval_interval == 0 or step == max_steps - 1):
            avg_loss, ppl = evaluate_perplexity(
                model, eval_batches, device,
                k_value=model_config.k_inference if model_config.recurrent_core else None
            )
            if ppl < best_ppl:
                best_ppl = ppl

        # Logging
        if step < 50 or step % log_interval == 0 or step == max_steps - 1:
            ws_stats = model.get_workspace_stats()
            if ws_stats is not None:
                gate_str = f"{ws_stats['read_gate']:>8.4f} {ws_stats['write_gate']:>8.4f}"
            else:
                gate_str = f"{'-':>8} {'-':>8}"
            ppl_str = f"{ppl:>8.2f}" if ppl > 0 else f"{'':>8}"
            print(f"{step:>6} {step_loss:>10.4f} {ppl_str} {current_lr:>10.6f} "
                  f"{step_time:>7.2f}s {tokens_per_sec:>10.0f} {grad_norm:>10.4f} "
                  f"{gate_str}", flush=True)

        # Early stopping
        if plateau_patience > 0 and step >= warmup_steps:
            if loss_ema is None:
                loss_ema = step_loss
            else:
                loss_ema = plateau_ema_beta * loss_ema + (1 - plateau_ema_beta) * step_loss
            if loss_ema < best_loss_ema * (1 - plateau_min_delta):
                best_loss_ema = loss_ema
                steps_since_best = 0
            else:
                steps_since_best += 1
            if steps_since_best >= plateau_patience:
                print(f"\n*** Early stopping at step {step}: loss EMA plateaued "
                      f"(best={best_loss_ema:.4f}, current={loss_ema:.4f}) ***",
                      flush=True)
                break

        # Checkpoint
        if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"cell_{cell}_step{step+1}.pt")
            model.save_checkpoint(ckpt_path, optimizer_state=optimizer.get_state(), step=step+1)
            print(f"Checkpoint saved to {ckpt_path} (step {step+1})", flush=True)

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, f"cell_{cell}_final.pt")
    model.save_checkpoint(final_path, optimizer_state=optimizer.get_state(), step=max_steps)

    # Final perplexity
    avg_loss, ppl = evaluate_perplexity(
        model, eval_batches, device,
        k_value=model_config.k_inference if model_config.recurrent_core else None
    )
    print(f"\nFinal perplexity: {ppl:.2f} (best: {best_ppl:.2f})", flush=True)

    avg_tokens_per_sec = total_tokens_trained / total_time if total_time > 0 else 0
    print(f"\nTotal time: {total_time:.1f}s", flush=True)
    print(f"Avg step: {total_time/(max_steps-start_step):.2f}s", flush=True)
    print(f"Avg throughput: {avg_tokens_per_sec:.0f} tokens/sec", flush=True)
    print(f"Total tokens: {total_tokens_trained:,}", flush=True)
    if skipped_steps > 0:
        print(f"Skipped steps (grad spike): {skipped_steps}", flush=True)
    ttnn.close_device(device)
    print("Training complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text training (TinyStories)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--micro_batch", type=int, default=None,
                        help="Override micro-batch size")
    parser.add_argument("--accum_steps", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-section profiling")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=int, default=0,
                        help="Tenstorrent device ID")
    parser.add_argument("--loss_method", type=str, default="host",
                        choices=["host", "scatter"],
                        help="Loss computation method for large vocab: "
                             "'host' (float32 on CPU, faster for small models) "
                             "or 'scatter' (on-device, for production scale)")
    args = parser.parse_args()

    train(
        config_path=args.config,
        steps_override=args.steps,
        micro_batch_override=args.micro_batch,
        accum_steps_override=args.accum_steps,
        profile=args.profile,
        checkpoint_dir_override=args.checkpoint_dir,
        resume=args.resume,
        device_id=args.device,
        loss_method=args.loss_method,
    )
