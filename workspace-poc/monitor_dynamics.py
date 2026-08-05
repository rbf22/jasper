#!/usr/bin/env python3
"""Training dynamics monitor for architecture v2 runs.

Parses training logs and checkpoint files to track:
  - Gate values (ReZero) over time — critical for divergence detection
  - QK scale values over time — should stay near init (1/sqrt(d_head))
  - Grad norm trends — should be stable, not growing
  - Loss trajectory
  - Workspace vs backbone grad norms (from checkpoints)

Previous runs diverged when gates reached ~0.20-0.25. This monitor
flags if gates approach that range, or if grad norms show instability.

Usage:
    python monitor_dynamics.py                    # check all running cells
    python monitor_dynamics.py --cell b           # check specific cell
    python monitor_dynamics.py --history 50       # show last 50 steps
    python monitor_dynamics.py --checkpoint       # also inspect latest checkpoint
"""

import argparse
import os
import re
import sys
import math
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = SCRIPT_DIR / "logs"
CKPT_DIR = SCRIPT_DIR / "checkpoints"

# Previous divergence thresholds (for reference)
DIVERGENCE_GATE_THRESHOLD = 0.20  # gates diverged here in runs 1-4
WARNING_GATE_THRESHOLD = 0.15     # start watching
GRAD_NORM_SPIKE = 10.0            # grad norm > this is a spike
GRAD_NORM_CATASTROPHIC = 100.0    # grad norm > this is divergence

# Log column indices (0-based)
COL_STEP = 0
COL_LOSS = 1
COL_LR = 2
COL_TIME = 3
COL_TPS = 4
COL_GRAD = 5
COL_ENTROPY = 6
COL_DIVERSITY = 7
COL_GATE_READ = 8
COL_GATE_WRITE = 9
COL_SLOT_DECAY = 10


def parse_log_line(line):
    """Parse a training log line into floats. Returns None for non-data lines."""
    parts = line.split()
    if len(parts) < 6:
        return None
    try:
        # Strip 's' suffix from time column (e.g. "5.92s" -> 5.92)
        vals = []
        for i, p in enumerate(parts):
            if p == "-":
                vals.append(float("nan"))  # no workspace (Cell A)
            elif p.endswith("s") and i == COL_TIME:
                vals.append(float(p[:-1]))
            else:
                vals.append(float(p))
        return vals
    except ValueError:
        return None


def parse_log(filepath):
    """Parse a training log file, returning list of step records."""
    records = []
    if not filepath.exists():
        return records
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Step"):
                continue
            vals = parse_log_line(line)
            if vals is not None:
                records.append(vals)
    return records


def analyze_gate_trajectory(records, cell_name):
    """Analyze gate value trajectory and flag warnings."""
    if not records:
        print(f"  No training records found for cell {cell_name}")
        return []

    latest = records[-1]
    step = int(latest[COL_STEP])
    gate_r = latest[COL_GATE_READ] if len(latest) > COL_GATE_READ else float("nan")
    gate_w = latest[COL_GATE_WRITE] if len(latest) > COL_GATE_WRITE else float("nan")

    # Cell A has no workspace — gates are NaN
    if math.isnan(gate_r) and math.isnan(gate_w):
        print(f"\n  Gates: N/A (Cell A has no workspace)")
        return []

    print(f"\n  Gate values at step {step}:")
    print(f"    read_gate:  {gate_r:+.4f}")
    print(f"    write_gate: {gate_w:+.4f}")

    # Track gate growth rate
    if len(records) >= 20:
        recent = records[-20:]
        first_step = int(recent[0][COL_STEP])
        last_step = int(recent[-1][COL_STEP])
        step_delta = last_step - first_step
        if step_delta > 0:
            gr_growth = (recent[-1][COL_GATE_READ] - recent[0][COL_GATE_READ]) / step_delta
            gw_growth = (recent[-1][COL_GATE_WRITE] - recent[0][COL_GATE_WRITE]) / step_delta
            print(f"    growth rate (last {step_delta} steps):")
            print(f"      read_gate:  {gr_growth:+.6f}/step")
            print(f"      write_gate: {gw_growth:+.6f}/step")

    # Flag warnings
    # NOTE: The 0.20-0.25 threshold was for OLD sigmoid gates. ReZero gates
    # behave differently (no sigmoid saturation, can go negative). The threshold
    # is kept as a reference point, not a hard divergence predictor.
    warnings = []
    max_gate = max(abs(gate_r), abs(gate_w))
    if max_gate > DIVERGENCE_GATE_THRESHOLD:
        # This is NOT necessarily a problem with ReZero — just a milestone to note
        print(f"    ℹ Gate magnitude {max_gate:.4f} exceeds old sigmoid divergence threshold {DIVERGENCE_GATE_THRESHOLD}")
        print(f"      (With ReZero, this is expected — the old threshold was for sigmoid gates)")
    elif max_gate > WARNING_GATE_THRESHOLD:
        print(f"    ℹ Gate magnitude {max_gate:.4f} approaching old divergence threshold")

    return warnings  # Don't return gate threshold as warnings — just informational


def analyze_grad_norm(records, cell_name):
    """Analyze gradient norm stability."""
    if not records:
        return []

    latest = records[-1]
    step = int(latest[COL_STEP])
    grad_norm = latest[COL_GRAD]

    print(f"\n  Gradient norm at step {step}: {grad_norm:.4f}")

    # Compute statistics over recent window
    if len(records) >= 20:
        recent_grads = [r[COL_GRAD] for r in records[-20:]]
        mean_grad = sum(recent_grads) / len(recent_grads)
        max_grad = max(recent_grads)
        min_grad = min(recent_grads)
        ratio = max_grad / (min_grad + 1e-10)
        print(f"    Recent (20 steps): mean={mean_grad:.2f}, min={min_grad:.2f}, max={max_grad:.2f}, ratio={ratio:.2f}")

        # Check for spikes
        if max_grad > GRAD_NORM_SPIKE:
            print(f"    ⚠ Spike detected: max grad norm {max_grad:.2f} > {GRAD_NORM_SPIKE}")
        if max_grad > GRAD_NORM_CATASTROPHIC:
            print(f"    🚨 CATASTROPHIC: grad norm {max_grad:.2f} > {GRAD_NORM_CATASTROPHIC}")

    # Check for growth trend
    if len(records) >= 50:
        early = [r[COL_GRAD] for r in records[-50:-30]]
        late = [r[COL_GRAD] for r in records[-20:]]
        early_mean = sum(early) / len(early)
        late_mean = sum(late) / len(late)
        if early_mean > 0:
            growth = late_mean / early_mean
            if growth > 2.0:
                print(f"    ⚠ Grad norm growing: {early_mean:.2f} → {late_mean:.2f} ({growth:.1f}x)")
            elif growth < 0.5:
                print(f"    ✓ Grad norm shrinking: {early_mean:.2f} → {late_mean:.2f} ({growth:.1f}x)")

    warnings = []
    if grad_norm > GRAD_NORM_SPIKE:
        warnings.append(f"⚠ Grad norm {grad_norm:.2f} > {GRAD_NORM_SPIKE}")
    if grad_norm > GRAD_NORM_CATASTROPHIC:
        warnings.append(f"🚨 Grad norm {grad_norm:.2f} > {GRAD_NORM_CATASTROPHIC}")

    return warnings


def analyze_loss(records, cell_name):
    """Analyze loss trajectory."""
    if not records:
        return

    latest = records[-1]
    step = int(latest[COL_STEP])
    loss = latest[COL_LOSS]

    print(f"\n  Loss at step {step}: {loss:.4f}")

    if len(records) >= 50:
        # EMA-smoothed loss
        ema = records[0][COL_LOSS]
        beta = 0.99
        for r in records:
            ema = beta * ema + (1 - beta) * r[COL_LOSS]
        print(f"    EMA loss: {ema:.4f}")

        # Check for loss increase (bad sign)
        recent = [r[COL_LOSS] for r in records[-20:]]
        earlier = [r[COL_LOSS] for r in records[-40:-20]]
        if earlier:
            recent_mean = sum(recent) / len(recent)
            earlier_mean = sum(earlier) / len(earlier)
            if recent_mean > earlier_mean * 1.1:
                print(f"    ⚠ Loss increasing: {earlier_mean:.4f} → {recent_mean:.4f}")
            elif recent_mean < earlier_mean * 0.9:
                print(f"    ✓ Loss decreasing: {earlier_mean:.4f} → {recent_mean:.4f}")


def analyze_checkpoint(cell_name):
    """Inspect the latest checkpoint for QK scale and gate values."""
    # Find latest checkpoint
    pattern = f"cell_{cell_name.upper()}_step*.pt"
    ckpts = sorted(CKPT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print(f"\n  No checkpoints found for cell {cell_name}")
        return

    latest_ckpt = ckpts[-1]
    print(f"\n  Latest checkpoint: {latest_ckpt.name}")

    try:
        import torch
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        params = ckpt.get("model_state", ckpt)

        # Extract workspace-specific params
        d_head = 96  # d_model=384, n_heads=4
        init_qk = 1.0 / math.sqrt(d_head)

        ws_params = {}
        for k, v in params.items():
            if "qk_scale" in k or ("gate" in k and "ws_" in k) or ("slot_decay" in k and "ws_" in k):
                if hasattr(v, "item"):
                    ws_params[k] = v.item()
                elif isinstance(v, (int, float)):
                    ws_params[k] = float(v)

        if ws_params:
            print(f"    Workspace parameters:")
            for k, v in sorted(ws_params.items()):
                if "qk_scale" in k:
                    drift = abs(v - init_qk) / init_qk * 100
                    print(f"      {k:30s}: {v:.6f} (init: {init_qk:.6f}, drift: {drift:.1f}%)")
                else:
                    print(f"      {k:30s}: {v:+.6f}")

        # Check optimizer state for grad norm history
        opt_state = ckpt.get("optimizer_state", None)
        if opt_state and "state" in opt_state:
            # Look at exp_avg_sq for qk_scale to see gradient magnitude history
            for k, v in params.items():
                if "qk_scale" in k:
                    # Find matching optimizer state
                    param_idx = None
                    for idx, p in ckpt.get("param_groups", [{}])[0].get("params", []).items() if isinstance(ckpt.get("param_groups"), dict) else []:
                        if p is v:
                            param_idx = idx
                            break
                    break

    except Exception as e:
        print(f"    Error reading checkpoint: {e}")


def print_history(records, n=20):
    """Print a table of recent training steps."""
    if not records:
        return

    print(f"\n  {'Step':>6} {'Loss':>8} {'LR':>10} {'GradNorm':>10} {'gz_gr':>8} {'gz_gw':>8} {'gam_slot':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for r in records[-n:]:
        step = int(r[COL_STEP])
        loss = r[COL_LOSS]
        lr = r[COL_LR]
        grad = r[COL_GRAD]
        gr = r[COL_GATE_READ] if len(r) > COL_GATE_READ and not math.isnan(r[COL_GATE_READ]) else float("nan")
        gw = r[COL_GATE_WRITE] if len(r) > COL_GATE_WRITE and not math.isnan(r[COL_GATE_WRITE]) else float("nan")
        gs = r[COL_SLOT_DECAY] if len(r) > COL_SLOT_DECAY and not math.isnan(r[COL_SLOT_DECAY]) else float("nan")
        gr_str = f"{gr:>+8.4f}" if not math.isnan(gr) else f"{'—':>8}"
        gw_str = f"{gw:>+8.4f}" if not math.isnan(gw) else f"{'—':>8}"
        gs_str = f"{gs:>8.4f}" if not math.isnan(gs) else f"{'—':>8}"
        print(f"  {step:>6} {loss:>8.4f} {lr:>10.6f} {grad:>10.4f} {gr_str} {gw_str} {gs_str}")


def main():
    parser = argparse.ArgumentParser(description="Monitor training dynamics for architecture v2")
    parser.add_argument("--cell", choices=["a", "b", "c", "all"], default="all", help="Which cell to monitor")
    parser.add_argument("--history", type=int, default=20, help="Number of recent steps to show")
    parser.add_argument("--checkpoint", action="store_true", help="Also inspect latest checkpoint")
    args = parser.parse_args()

    cells = ["a", "b", "c"] if args.cell == "all" else [args.cell]

    all_warnings = []

    for cell in cells:
        cell_upper = cell.upper()
        log_file = LOG_DIR / f"cell_{cell}_qknorm_20260802.log"

        print(f"\n{'='*70}")
        print(f"Cell {cell_upper}")
        print(f"{'='*70}")

        records = parse_log(log_file)
        if not records:
            print(f"  No log found at {log_file}")
            continue

        print(f"  Total steps logged: {len(records)}")
        print(f"  Latest step: {int(records[-1][COL_STEP])}")

        # History table
        print_history(records, args.history)

        # Analysis
        analyze_loss(records, cell)
        warnings = analyze_grad_norm(records, cell)
        warnings += analyze_gate_trajectory(records, cell)

        if args.checkpoint:
            analyze_checkpoint(cell)

        if warnings:
            all_warnings.extend([(cell_upper, w) for w in warnings])
            print(f"\n  Warnings:")
            for w in warnings:
                print(f"    {w}")
        else:
            print(f"\n  ✓ No warnings")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if all_warnings:
        for cell, w in all_warnings:
            print(f"  Cell {cell}: {w}")
    else:
        print("  All cells stable — no warnings")

    # Comparison with previous divergence points
    print(f"\n  Previous divergence thresholds (for reference):")
    print(f"    Cell B: diverged at step ~1400-2300 (gate ~0.20-0.25)")
    print(f"    Cell C: diverged at step ~700-900 (gate ~0.20-0.25)")
    print(f"    Current runs need to pass these thresholds to confirm the fix works.")


if __name__ == "__main__":
    main()
