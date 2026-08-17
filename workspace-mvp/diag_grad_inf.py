#!/usr/bin/env python
"""Diagnostic script to find which parameter gradients produce inf/nan.

Creates a small model, runs 60 training steps, and after each step checks
every gradient tensor for inf/nan by moving it to host and inspecting with
torch.isinf() / torch.isnan(). Prints which parameter has inf/nan gradient
and its max/min values.

Usage:
    TT_VISIBLE_DEVICES=0 TT_METAL_HOME=/home/rfenwick/Documents/tt-metal-src \
    TT_METAL_LOGGER_LEVEL=ERROR /home/rfenwick/Documents/jasper/.tt-venv/bin/python -u diag_grad_inf.py
"""

import argparse
import math
import os
import random
import sys
import time

# Set env before importing ttnn
os.environ.setdefault("TT_METAL_HOME", "/home/rfenwick/Documents/tt-metal-src")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

# P300 fabric mesh graph descriptor setup (same as train_native_tt.py)
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

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tt_text_latent_memory_model import (
    TTTextLatentMemoryConfig,
    TTTextLatentMemoryModel,
    to_device,
    from_device,
    _safe_deallocate,
)
from challenge_data import ChallengeDataset

# Reuse loss and optimizer from train_native_tt
from train_native_tt import (
    cross_entropy_loss_and_grad,
    TTAdamW,
    clip_grad_norm,
    create_device,
    get_lr,
)


def check_grads_for_inf_nan(grads: dict, step: int) -> dict:
    """Check each gradient tensor for inf/nan.

    Moves each gradient to host (fp32) and checks with torch.isinf / torch.isnan.
    Returns a dict of {param_name: (has_inf, has_nan, max_val, min_val)} for
    parameters that have inf or nan, plus max/min for all params.
    """
    results = {}
    bad_params = {}

    for name, g in grads.items():
        try:
            g_torch = ttnn.to_torch(g).float()
        except Exception as e:
            print(f"  [step {step}] ERROR reading grad '{name}': {e}", flush=True)
            continue

        has_inf = torch.isinf(g_torch).any().item()
        has_nan = torch.isnan(g_torch).any().item()

        # For inf/nan tensors, get finite-only stats
        if has_inf or has_nan:
            finite_mask = torch.isfinite(g_torch)
            if finite_mask.any():
                finite_vals = g_torch[finite_mask]
                max_val = finite_vals.max().item()
                min_val = finite_vals.min().item()
            else:
                max_val = float('nan')
                min_val = float('nan')
            n_inf = torch.isinf(g_torch).sum().item()
            n_nan = torch.isnan(g_torch).sum().item()
            bad_params[name] = {
                'has_inf': has_inf,
                'has_nan': has_nan,
                'n_inf': n_inf,
                'n_nan': n_nan,
                'max': max_val,
                'min': min_val,
                'shape': tuple(g_torch.shape),
            }
        else:
            max_val = g_torch.max().item()
            min_val = g_torch.min().item()

        results[name] = (has_inf, has_nan, max_val, min_val)

        del g_torch

    return results, bad_params


def print_all_grad_stats(results: dict, step: int):
    """Print max abs gradient for each parameter (for steps without inf/nan)."""
    print(f"\n--- Step {step}: gradient stats (max abs per param) ---", flush=True)
    sorted_names = sorted(results.keys())
    for name in sorted_names:
        has_inf, has_nan, max_val, min_val = results[name]
        max_abs = max(abs(max_val), abs(min_val))
        flag = ""
        if has_inf:
            flag = " *** INF ***"
        elif has_nan:
            flag = " *** NAN ***"
        print(f"  {name:40s} max_abs={max_abs:.6e}{flag}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Diagnose inf/nan gradients")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_slots", type=int, default=8)
    parser.add_argument("--max_reasoning_steps", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--max_prompt_len", type=int, default=64)
    parser.add_argument("--max_answer_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_all_steps", action="store_true",
                        help="Print gradient stats for every step, not just inf/nan steps")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = ttnn.bfloat16 if args.precision == "bf16" else ttnn.float32

    # Create device
    print(f"Opening device {args.device}...", flush=True)
    device = create_device(args.device)

    # Create config
    config = TTTextLatentMemoryConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        n_slots=args.n_slots,
        max_reasoning_steps=args.max_reasoning_steps,
        expand=args.expand,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # Create model
    print("Creating TT model...", flush=True)
    model = TTTextLatentMemoryModel(config, device, dtype=dtype)
    n_params = model.get_num_params()
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M) precision={args.precision}", flush=True)

    # Create dataset
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, "tiny_challenges_train.txt")
    valid_path = os.path.join(data_dir, "tiny_challenges_valid.txt")
    dataset = ChallengeDataset(
        train_path=train_path,
        valid_path=valid_path,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # Create optimizer
    params = model.get_params()
    optimizer = TTAdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW lr={args.lr} wd={args.weight_decay}", flush=True)

    # Training loop
    rng = random.Random(args.seed)
    start_time = time.time()

    print(f"\nStarting diagnostic training for {args.steps} steps...", flush=True)
    print(f"Checking each gradient tensor for inf/nan after every step.\n", flush=True)

    inf_steps = []  # track which steps had inf/nan
    prev_grad_stats = {}  # track max abs from previous step for comparison

    for step in range(args.steps):
        lr = get_lr(step, args)
        optimizer.set_lr(lr)

        # Sample batch
        prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
            dataset.sample_batch(args.batch_size, "train", rng)

        # Forward pass (training mode — caches intermediates)
        logits_tt = model.forward_train(prompt_ids, prompt_mask, dec_input, ans_mask)

        # Loss + gradient (host-side loss for large vocab, gradient back to device)
        loss_val, grad_logits = cross_entropy_loss_and_grad(
            logits_tt, ans_targets, ans_mask, device, dtype=dtype
        )
        _safe_deallocate(logits_tt)

        # Backward pass (on device)
        grads = model.backward(grad_logits)
        _safe_deallocate(grad_logits)

        # === DIAGNOSTIC: Check gradients for inf/nan BEFORE clipping ===
        results, bad_params = check_grads_for_inf_nan(grads, step)

        if bad_params:
            inf_steps.append(step)
            print(f"\n{'='*70}", flush=True)
            print(f"*** STEP {step}: inf/nan gradients detected! loss={loss_val:.4f} ***", flush=True)
            print(f"{'='*70}", flush=True)
            for name in sorted(bad_params.keys()):
                info = bad_params[name]
                print(f"  PARAM: {name}", flush=True)
                print(f"    shape: {info['shape']}", flush=True)
                print(f"    has_inf: {info['has_inf']}, has_nan: {info['has_nan']}", flush=True)
                print(f"    n_inf: {info['n_inf']}, n_nan: {info['n_nan']}", flush=True)
                print(f"    max (finite): {info['max']:.6e}", flush=True)
                print(f"    min (finite): {info['min']:.6e}", flush=True)
                # Show previous step's max abs for this param
                if name in prev_grad_stats:
                    print(f"    prev step max_abs: {prev_grad_stats[name]:.6e}", flush=True)
            print(f"{'='*70}\n", flush=True)

            # Also print all param stats for this step to see which are large
            print_all_grad_stats(results, step)
        elif args.print_all_steps or step % 10 == 0:
            elapsed = time.time() - start_time
            # Print summary: max abs across all params
            all_max_abs = []
            for name in sorted(results.keys()):
                has_inf, has_nan, max_val, min_val = results[name]
                max_abs = max(abs(max_val), abs(min_val))
                all_max_abs.append((name, max_abs))
            all_max_abs.sort(key=lambda x: -x[1])
            top5 = all_max_abs[:5]
            print(f"step {step:3d} loss={loss_val:.4f} lr={lr:.2e} t={elapsed:.0f}s", flush=True)
            print(f"  top-5 grad max_abs: " + ", ".join(
                f"{n.split('_')[-1]}={v:.2e}" for n, v in top5
            ), flush=True)

        # Save prev stats
        prev_grad_stats = {}
        for name, (has_inf, has_nan, max_val, min_val) in results.items():
            prev_grad_stats[name] = max(abs(max_val), abs(min_val))

        # Gradient clipping
        grad_norm = clip_grad_norm(grads, args.grad_clip)

        # Optimizer step
        optimizer.step(grads, model)

        # Deallocate gradients
        for name, g in grads.items():
            _safe_deallocate(g)

        # Synchronize and clear caches
        model.clear_caches()

        # Check for NaN/Inf loss
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"ERROR: loss became {loss_val} at step {step}", flush=True)
            break

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"DIAGNOSTIC SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Total steps: {args.steps}", flush=True)
    print(f"Steps with inf/nan gradients: {len(inf_steps)}", flush=True)
    if inf_steps:
        print(f"Inf/nan at steps: {inf_steps}", flush=True)
    else:
        print("No inf/nan gradients detected in any step.", flush=True)
    print(f"Total time: {elapsed:.0f}s", flush=True)
    print(f"{'='*70}", flush=True)

    # Close device
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
