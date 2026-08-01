#!/usr/bin/env python3
"""Automated eval loop — monitors checkpoints and runs eval_ttnn.py on each new one.

For each cell (B, C, D, and later A, E):
  - Scans run_X/checkpoints/ for new checkpoint files
  - Runs eval_ttnn.py on the specified device
  - Saves JSON results to run_X/eval_results/
  - Tracks plateau: if Task 1 accuracy hasn't improved by >=1% over 3 consecutive
    evals, marks the cell as plateaued
  - Prints a summary table after each eval

Usage:
    python eval_loop.py --device 0 --poll-interval 300

    # One-shot: eval all existing checkpoints, then exit
    python eval_loop.py --device 0 --once

    # Custom cells and eval size
    python eval_loop.py --device 0 --cells B C D --n-per-task 5
"""

import os
import sys
import json
import time
import argparse
import subprocess
import glob
import re
from pathlib import Path
from datetime import datetime


# Cell -> config file mapping
CELL_CONFIGS = {
    "A": "configs/cell_a_tt.yaml",
    "B": "configs/cell_b_tt.yaml",
    "C": "configs/cell_c_tt.yaml",
}

# Plateau detection: no improvement >= threshold over N consecutive evals
PLATEAU_EVALS = 3
PLATEAU_THRESHOLD = 0.01  # 1% improvement required to count as "improved"


def find_checkpoints(cell, base_dir="."):
    """Find all checkpoint files for a cell, sorted by step number."""
    pattern = os.path.join(base_dir, f"run_{cell}", "checkpoints", f"cell_{cell}_step*.pt")
    files = glob.glob(pattern)
    # Extract step number and sort
    def step_of(path):
        m = re.search(r"step(\d+)\.pt$", path)
        return int(m.group(1)) if m else 0
    return sorted(files, key=step_of)


def already_evaluated(cell, checkpoint_path, base_dir="."):
    """Check if a checkpoint has already been evaluated."""
    step_match = re.search(r"step(\d+)\.pt$", checkpoint_path)
    if not step_match:
        return True  # skip unparseable
    step = step_match.group(1)
    json_path = os.path.join(base_dir, f"run_{cell}", "eval_results", f"step_{step}.json")
    return os.path.exists(json_path)


def run_eval(cell, checkpoint_path, device, base_dir, n_per_task, depths, timeout=600):
    """Run eval_ttnn.py on a checkpoint and return the JSON results path."""
    config = CELL_CONFIGS.get(cell)
    if not config:
        print(f"  ERROR: No config for cell {cell}")
        return None

    step_match = re.search(r"step(\d+)\.pt$", checkpoint_path)
    step = step_match.group(1) if step_match else "unknown"

    json_path = os.path.join(base_dir, f"run_{cell}", "eval_results", f"step_{step}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Use the cell's own tt_cache to avoid recompiling kernels
    tt_cache = os.path.join(base_dir, f"run_{cell}", "tt_cache")

    cmd = [
        "/home/ttuser/Documents/jasper/.tt-venv/bin/python",
        "eval_ttnn.py",
        "--config", config,
        "--device", str(device),
        "--checkpoint", checkpoint_path,
        "--n-per-task", str(n_per_task),
        "--depths", *[str(d) for d in depths],
        "--show-samples", "0",
        "--max-new", "10",
        "--json-output", json_path,
    ]

    env = os.environ.copy()
    env["TT_METAL_CACHE"] = tt_cache
    env["TT_METAL_LOGGER_LEVEL"] = "ERROR"

    print(f"  Running: cell {cell} step {step} on device {device}...")
    print(f"  Checkpoint: {checkpoint_path}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=base_dir,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            # Print last few lines of stderr for debugging
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-5:]:
                print(f"    {line}")
            return None

        # Print key results from stdout
        for line in result.stdout.split("\n"):
            if "Overall:" in line or "Task1" in line or "task1" in line.lower():
                print(f"  {line.strip()}")

        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            print(f"  Done: overall={data['overall_accuracy']:.1%}, "
                  f"task1={data['task_accuracy']['task1']:.1%}, "
                  f"task2={data['task_accuracy']['task2']:.1%}, "
                  f"task3={data['task_accuracy']['task3']:.1%}")
            return data
        else:
            print(f"  WARNING: JSON output not found at {json_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def load_eval_history(cell, base_dir="."):
    """Load all eval results for a cell, sorted by step."""
    pattern = os.path.join(base_dir, f"run_{cell}", "eval_results", "step_*.json")
    files = glob.glob(pattern)
    results = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            results.append(data)
        except (json.JSONDecodeError, IOError):
            pass
    results.sort(key=lambda x: x.get("step", 0))
    return results


def check_plateau(history, metric="task1", n=PLATEAU_EVALS, threshold=PLATEAU_THRESHOLD):
    """Check if a cell has plateaued based on recent eval history.

    Returns True if the last n evals show no improvement >= threshold.
    """
    if len(history) < n + 1:
        return False

    # Get the metric values for the last n+1 evals
    # The "baseline" is the best metric value before the last n evals
    recent = history[-(n + 1):]
    baseline = recent[0]["task_accuracy"][metric] if metric.startswith("task") else recent[0][metric]
    best_recent = baseline
    for r in recent[1:]:
        val = r["task_accuracy"][metric] if metric.startswith("task") else r[metric]
        best_recent = max(best_recent, val)

    improvement = best_recent - baseline
    return improvement < threshold


def print_summary(all_cells, base_dir="."):
    """Print a summary table of all eval results."""
    print(f"\n{'='*80}")
    print(f"Eval Summary — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"{'Cell':>5} {'Step':>6} {'Overall':>8} {'Task1':>8} {'Task2':>8} {'Task3':>8} {'Plateaued':>10}")
    print(f"{'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for cell in sorted(all_cells):
        history = load_eval_history(cell, base_dir)
        if not history:
            print(f"{cell:>5} {'—':>6} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>10}")
            continue

        latest = history[-1]
        plateaued = check_plateau(history, "task1")
        plateau_str = "YES ✓" if plateaued else f"({len(history)}/{PLATEAU_EVALS+1})"

        print(f"{cell:>5} {latest['step']:>6} "
              f"{latest['overall_accuracy']:>7.1%} "
              f"{latest['task_accuracy']['task1']:>7.1%} "
              f"{latest['task_accuracy']['task2']:>7.1%} "
              f"{latest['task_accuracy']['task3']:>7.1%} "
              f"{plateau_str:>10}")

    # Print trend for cells with multiple evals
    has_trends = False
    for cell in sorted(all_cells):
        history = load_eval_history(cell, base_dir)
        if len(history) >= 2:
            if not has_trends:
                print(f"\nTask 1 accuracy trend (key metric for workspace benefit):")
                has_trends = True
            trend = " → ".join(f"{h['step']}:{h['task_accuracy']['task1']:.0%}" for h in history[-6:])
            print(f"  Cell {cell}: {trend}")

    # Alert on plateaued cells
    plateaued_cells = []
    for cell in sorted(all_cells):
        history = load_eval_history(cell, base_dir)
        if check_plateau(history, "task1"):
            plateaued_cells.append(cell)

    if plateaued_cells:
        print(f"\n{'='*80}")
        print(f"PLATEAUED (no Task 1 improvement in last {PLATEAU_EVALS} evals): {', '.join(plateaued_cells)}")
        print(f"These cells can be stopped.")
        print(f"{'='*80}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Automated eval loop for TT training")
    parser.add_argument("--device", type=int, default=0, help="Device to run evals on")
    parser.add_argument("--cells", nargs="+", default=["B", "C", "D"], help="Cells to monitor")
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between polls")
    parser.add_argument("--n-per-task", type=int, default=5, help="Eval examples per task per depth")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--once", action="store_true", help="Eval all existing checkpoints once, then exit")
    parser.add_argument("--base-dir", default="/home/ttuser/Documents/jasper/mamba-poc")
    parser.add_argument("--eval-timeout", type=int, default=900, help="Timeout per eval in seconds")
    args = parser.parse_args()

    base = args.base_dir
    os.chdir(base)

    print(f"Eval loop started — device {args.device}, cells {args.cells}")
    print(f"Poll interval: {args.poll_interval}s, eval size: {args.n_per_task}/task/depth")
    print(f"Plateau detection: no Task 1 improvement >= {PLATEAU_THRESHOLD:.0%} over {PLATEAU_EVALS} evals")
    print(f"Base dir: {base}")
    print()

    # Track which checkpoints we've already evaluated
    evaluated = set()

    while True:
        found_new = False

        for cell in args.cells:
            checkpoints = find_checkpoints(cell, base)

            for ckpt in checkpoints:
                if ckpt in evaluated:
                    continue
                if already_evaluated(cell, ckpt, base):
                    evaluated.add(ckpt)
                    continue

                # New checkpoint found!
                step_match = re.search(r"step(\d+)\.pt$", ckpt)
                step = step_match.group(1) if step_match else "?"
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New checkpoint: Cell {cell} step {step}")

                result = run_eval(
                    cell, ckpt, args.device, base,
                    args.n_per_task, args.depths, args.eval_timeout
                )

                evaluated.add(ckpt)

                if result:
                    found_new = True

                # Check plateau after each eval
                history = load_eval_history(cell, base)
                if check_plateau(history, "task1"):
                    print(f"\n  *** Cell {cell} has PLATEAUED on Task 1 ***")
                    print(f"  *** No improvement in last {PLATEAU_EVALS} evals. Safe to stop. ***")

        # Print summary after each scan
        print_summary(args.cells, base)

        if args.once:
            print("One-shot mode complete. Exiting.")
            break

        if not found_new:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No new checkpoints. "
                  f"Sleeping {args.poll_interval}s...")

        # Sleep before next poll
        # But if we found new checkpoints, only sleep briefly
        sleep_time = 10 if found_new else args.poll_interval
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
