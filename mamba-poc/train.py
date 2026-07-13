"""
Training loop for the Mamba + Workspace POC.

Features:
  - Fresh data every batch (no overfitting risk)
  - Checkpoint every 15 minutes with auto-resume
  - Wandb logging
  - CLI args for cell selection and hyperparameters
  - MPS/CUDA/CPU device auto-detection
"""

import os
import time
import math
import yaml
import json
import random
import shutil
import argparse
import torch
import torch.nn.functional as F
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Optional, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
GDRIVE_DIR = "/content/drive/MyDrive"

from data import Vocab, sample_batch, generate_eval_set, TASK_VERIFIERS
from model import MambaWorkspaceModel, get_cell_config, ModelConfig


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_use_amp(cfg: dict, device: torch.device) -> bool:
    """Auto-enable AMP on CUDA unless explicitly disabled."""
    if device.type != "cuda":
        return False
    return cfg.get("use_amp", True)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model_config(cfg: dict) -> ModelConfig:
    cell = cfg.get("cell", "D")
    config = get_cell_config(cell)
    # Override with any config file values
    for key in ["d_model", "n_layers", "vocab_size", "d_state", "d_conv", "expand",
                "n_heads", "n_workspace_slots", "k_train_max", "k_inference", "dropout"]:
        if key in cfg:
            setattr(config, key, cfg[key])
    if "attention_positions" in cfg:
        config.attention_positions = cfg["attention_positions"]
    if "core_start" in cfg:
        config.core_start = cfg["core_start"]
    if "core_end" in cfg:
        config.core_end = cfg["core_end"]
    return config


def save_checkpoint(model, optimizer, scheduler, step, loss, path, config):
    """Save checkpoint atomically (write to temp, then rename)."""
    path = str(path)
    tmp_path = path + ".tmp"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "loss": loss,
        "config": config,
    }, tmp_path)
    os.replace(tmp_path, path)

    # Backup to Google Drive if mounted and configured
    gdrive_ckpt_dir = config.get("gdrive_ckpt_dir")
    if gdrive_ckpt_dir and os.path.isdir(GDRIVE_DIR):
        gdrive_path = os.path.join(gdrive_ckpt_dir, os.path.basename(path))
        try:
            shutil.copy2(path, gdrive_path)
            print(f"Checkpoint backed up to Google Drive: {gdrive_path}")
        except Exception as e:
            print(f"Warning: Failed to backup checkpoint to Google Drive: {e}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and ckpt["optimizer_state_dict"]:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt["scheduler_state_dict"]:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"], ckpt["loss"]


@torch.no_grad()
def evaluate(model, eval_set, vocab, device, max_new=10, use_amp=False):
    """Evaluate accuracy on the eval set by generating answers and verifying."""
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else _nullcontext()
    correct = {1: 0, 2: 0, 3: 0}
    total = {1: 0, 2: 0, 3: 0}
    correct_by_depth = {}
    total_by_depth = {}

    for ex in eval_set:
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        task_id = ex["task_id"]
        depth = ex["depth"]
        prompt = ex["prompt"]
        answer_str = ex["answer_str"]

        # Find prompt length (where answer starts)
        prompt_ids = vocab.encode(prompt)
        prompt_len = len(prompt_ids) - 1  # exclude EOS so model predicts first answer char

        # Generate answer tokens
        generated = input_ids[:, :prompt_len]
        for _ in range(max_new):
            with amp_ctx:
                out = model(generated)
            next_logits = out["logits"][:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == vocab.EOS:
                break

        # Decode the generated answer
        response_ids = generated[0, prompt_len:].tolist()
        response = vocab.decode(response_ids)

        is_correct = TASK_VERIFIERS[task_id](prompt, response)
        total[task_id] += 1
        correct[task_id] += int(is_correct)

        if depth not in correct_by_depth:
            correct_by_depth[depth] = 0
            total_by_depth[depth] = 0
        total_by_depth[depth] += 1
        correct_by_depth[depth] += int(is_correct)

    model.train()

    results = {}
    for tid in [1, 2, 3]:
        results[f"task{tid}_acc"] = correct[tid] / max(total[tid], 1)
    for depth in sorted(total_by_depth.keys()):
        results[f"depth{depth}_acc"] = correct_by_depth[depth] / max(total_by_depth[depth], 1)
    results["overall_acc"] = sum(correct.values()) / max(sum(total.values()), 1)
    return results


def train(cfg: dict, config_path: str):
    device = get_device()
    print(f"Using device: {device}")

    # Build model config
    model_config = build_model_config(cfg)
    cell = cfg.get("cell", "D")
    print(f"Cell: {cell}, Params: ", end="")

    model = MambaWorkspaceModel(model_config).to(device)
    n_params = model.get_num_params()
    print(f"{n_params / 1e6:.1f}M")

    # AMP (auto-enabled on CUDA)
    use_amp = should_use_amp(cfg, device)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=use_amp) if use_amp else None
    print(f"AMP (fp16): {use_amp}")

    # Training hyperparameters
    lr = cfg.get("lr", 6e-4)
    max_steps = cfg.get("max_steps", 10000)
    seq_len = cfg.get("seq_len", 128)
    tokens_per_batch = float(cfg.get("tokens_per_batch", 250000))
    micro_batch_size = cfg.get("micro_batch_size", 0)  # 0 = no grad accumulation

    # Compute effective batch size and gradient accumulation steps
    if tokens_per_batch > 0:
        effective_batch_size = max(1, int(tokens_per_batch / seq_len))
    else:
        effective_batch_size = cfg.get("batch_size", 32)

    if micro_batch_size > 0 and micro_batch_size < effective_batch_size:
        grad_accum_steps = max(1, effective_batch_size // micro_batch_size)
        batch_size = micro_batch_size
    else:
        grad_accum_steps = 1
        batch_size = effective_batch_size
    warmup_steps = cfg.get("warmup_steps", 200)
    weight_decay = cfg.get("weight_decay", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    z_loss_coef = cfg.get("z_loss_coef", 1e-4)
    eval_interval = cfg.get("eval_interval", 500)
    log_interval = cfg.get("log_interval", 50)
    checkpoint_interval = cfg.get("checkpoint_interval", 900)  # 15 min in seconds
    depth_range = tuple(cfg.get("depth_range", [2, 8]))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Wandb
    use_wandb = cfg.get("wandb", False) and not cfg.get("no_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "mamba-workspace-poc"),
            name=cfg.get("run_name", f"cell{cell}"),
            config=cfg,
        )

    # Checkpointing — resolve to absolute path relative to script dir
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = SCRIPT_DIR / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"cell{cell}_latest.pt"

    # Resume
    start_step = 0
    resume_ckpt = None
    if cfg.get("resume", True):
        # Check local checkpoint first
        if ckpt_path.exists():
            resume_ckpt = ckpt_path
            print(f"Found local checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
        else:
            # Check Google Drive backup
            gdrive_ckpt_dir = cfg.get("gdrive_ckpt_dir")
            if gdrive_ckpt_dir and os.path.isdir(GDRIVE_DIR):
                gdrive_path = os.path.join(gdrive_ckpt_dir, f"cell{cell}_latest.pt")
                if os.path.exists(gdrive_path):
                    print(f"Local checkpoint missing, restoring from Google Drive: {gdrive_path}")
                    shutil.copy2(gdrive_path, str(ckpt_path))
                    resume_ckpt = ckpt_path
                else:
                    print(f"No checkpoint found locally or in Google Drive ({gdrive_path})")
            else:
                print(f"No checkpoint found at {ckpt_path}")

        if resume_ckpt is not None:
            try:
                print(f"Resuming from {resume_ckpt}")
                start_step, _ = load_checkpoint(str(resume_ckpt), model, optimizer, scheduler)
                print(f"Resumed at step {start_step}")
            except Exception as e:
                print(f"Checkpoint load failed: {type(e).__name__}: {e}")
                print("Starting fresh from step 0.")
                start_step = 0

    # Data
    vocab = Vocab()
    rng = random.Random(cfg.get("seed", 42))

    # Eval set
    eval_depths = list(range(2, 17))
    eval_set = generate_eval_set(
        n_per_task_per_depth=cfg.get("eval_samples", 20),
        depths=eval_depths,
        vocab=vocab,
        seq_len=seq_len,
        rng=random.Random(123),  # Fixed seed for eval
    )
    print(f"Eval set: {len(eval_set)} examples")

    # Training loop
    model.train()
    step = start_step
    last_ckpt_time = time.time()
    last_log_time = time.time()

    print(f"Starting training from step {step} to {max_steps}")
    print(f"Micro-batch: {batch_size}, Grad accum: {grad_accum_steps}, "
          f"Effective batch: {batch_size * grad_accum_steps}, "
          f"Seq len: {seq_len}, Tokens/step: {batch_size * grad_accum_steps * seq_len}")

    while step < max_steps:
        optimizer.zero_grad()
        total_loss = 0.0

        for accum_idx in range(grad_accum_steps):
            input_ids, labels, task_ids = sample_batch(
                batch_size, seq_len, vocab, depth_range=depth_range, rng=rng
            )
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward (with AMP autocast on CUDA)
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else _nullcontext()
            with amp_ctx:
                out = model(input_ids)
                logits = out["logits"]

                # Compute loss only on answer positions
                # Shift: predict next token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, :-1].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                # z-loss: penalize log-partition drift to keep logit scale stable (PaLM-style)
                z = torch.logsumexp(shift_logits.float(), dim=-1)  # (B, T-1)
                z_mask = shift_labels != -100
                z_loss = z_loss_coef * (z[z_mask] ** 2).mean()
                loss = loss + z_loss
                loss = loss / grad_accum_steps

            # Backward (accumulate)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item() * grad_accum_steps  # un-scale to get raw loss per micro-batch

        # Check for NaN before optimizer step
        if math.isnan(total_loss) or math.isinf(total_loss):
            print(f"Step {step+1}: NaN/Inf loss detected, skipping optimizer step")
            if scaler is not None:
                scaler.unscale_(optimizer)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.update()
            scheduler.step()
            step += 1
            continue

        # Optimizer step (after all accumulation steps)
        if scaler is not None:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        step += 1
        avg_loss = total_loss / grad_accum_steps

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - last_log_time
            tokens_sec = (batch_size * grad_accum_steps * seq_len * log_interval) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            msg = f"Step {step}/{max_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | {tokens_sec:.0f} tok/s"
            print(msg)
            if use_wandb:
                wandb.log({
                    "loss": avg_loss,
                    "lr": current_lr,
                    "tokens/sec": tokens_sec,
                    "step": step,
                })
            last_log_time = time.time()

        # Evaluation
        if step % eval_interval == 0:
            print(f"\n--- Evaluation at step {step} ---")
            eval_results = evaluate(model, eval_set, vocab, device, max_new=5, use_amp=use_amp)
            for k, v in sorted(eval_results.items()):
                print(f"  {k}: {v:.3f}")
            if use_wandb:
                wandb.log({f"eval/{k}": v for k, v in eval_results.items()})
                wandb.log({"step": step})
            print("---\n")

        # Checkpoint
        if time.time() - last_ckpt_time > checkpoint_interval:
            save_checkpoint(model, optimizer, scheduler, step, avg_loss, str(ckpt_path), cfg)
            print(f"Checkpoint saved at step {step}")
            last_ckpt_time = time.time()

    # Final checkpoint and eval
    save_checkpoint(model, optimizer, scheduler, step, avg_loss, str(ckpt_path), cfg)
    print(f"\nFinal checkpoint saved at step {step}")
    eval_results = evaluate(model, eval_set, vocab, device, max_new=5, use_amp=use_amp)
    print("Final evaluation:")
    for k, v in sorted(eval_results.items()):
        print(f"  {k}: {v:.3f}")

    if use_wandb:
        wandb.log({f"final/{k}": v for k, v in eval_results.items()})
        wandb.finish()

    return model, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train Mamba + Workspace POC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--cell", type=str, default=None, help="Override cell (A/B/C/D)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cell:
        cfg["cell"] = args.cell
    if args.no_wandb:
        cfg["no_wandb"] = True
    if args.smoke_test:
        cfg["max_steps"] = 200
        cfg["batch_size"] = 16
        cfg["seq_len"] = 64
        cfg["tokens_per_batch"] = 0  # use batch_size directly
        cfg["d_model"] = 128
        cfg["n_layers"] = 6
        cfg["warmup_steps"] = 20
        cfg["eval_interval"] = 100
        cfg["log_interval"] = 20
        cfg["no_wandb"] = True
        cfg["resume"] = False
        cfg["depth_range"] = [4, 4]  # Task 1 depth-4 only
        # Keep cell from config — don't override to D
        cfg["core_start"] = 3
        cfg["core_end"] = 5
        cfg["attention_positions"] = [2, 5]
        cfg["k_train_max"] = 3
        cfg["k_inference"] = 3
        print("=== SMOKE TEST MODE ===")

    train(cfg, args.config)


if __name__ == "__main__":
    main()
