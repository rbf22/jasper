"""
Tenstorrent Quietbox 2 runner — trains all four cells sequentially on a single Blackhole chip.

Usage:
    # Activate the TT venv first:
    source ../../.tt-venv/bin/activate

    python tt_runner.py                    # train all 4 cells sequentially (A→B→C→D)
    python tt_runner.py --cell D           # train only Cell D
    python tt_runner.py --cell B D         # train Cell B then Cell D
    python tt_runner.py --status           # check status of training runs
    python tt_runner.py --clean            # delete checkpoints and start fresh
    python tt_runner.py --smoke            # quick 50-step smoke test on Cell A

Environment:
    TT_VISIBLE_DEVICES  — optional, comma-separated device IDs to use (default: all)
    TT_METAL_CACHE      — optional, kernel cache directory (default: ~/.cache/tt_metal)

The TT venv at ../../.tt-venv must be activated before running.
All 4 cells train on a single Blackhole chip (data-parallel multi-chip is a future extension).
"""

import os
import sys
import time
import glob
import shutil
import subprocess
import argparse

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(REPO_DIR, "configs")
CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")

# Cell → config mapping
CELL_CONFIGS = {
    "A": os.path.join(CONFIG_DIR, "cell_a_tt.yaml"),
    "B": os.path.join(CONFIG_DIR, "cell_b_tt.yaml"),
    "C": os.path.join(CONFIG_DIR, "cell_c_tt.yaml"),
    "D": os.path.join(CONFIG_DIR, "cell_d_tt.yaml"),
}


def clean_checkpoints(cells=None):
    """Delete checkpoints for specified cells (or all if None)."""
    cells = cells or list(CELL_CONFIGS.keys())
    if not os.path.isdir(CKPT_DIR):
        return
    removed = 0
    for f in os.listdir(CKPT_DIR):
        if f.endswith(".pt") and any(f.startswith(f"cell{c}") for c in cells):
            os.remove(os.path.join(CKPT_DIR, f))
            removed += 1
    print(f"Removed {removed} checkpoint(s) for cells: {', '.join(cells)}")


def run_cell(cell, smoke=False):
    """Train a single cell on the TT device."""
    config = CELL_CONFIGS[cell]
    if not os.path.exists(config):
        print(f"ERROR: Config not found: {config}")
        return False

    log_file = os.path.join(REPO_DIR, f"train_cell{cell}_tt.log")
    print(f"\n{'='*60}")
    print(f"Training Cell {cell} on Tenstorrent Quietbox 2")
    print(f"Config: {config}")
    print(f"Log: {log_file}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, os.path.join(REPO_DIR, "train.py"), "--config", config]
    if smoke:
        # Override max_steps for smoke test via a temp config
        import yaml
        with open(config) as f:
            cfg = yaml.safe_load(f)
        cfg["max_steps"] = 50
        cfg["eval_interval"] = 50
        cfg["log_interval"] = 10
        cfg["checkpoint_interval"] = 1
        cfg["run_name"] = f"cell{cell}-smoke"
        smoke_config = os.path.join(CONFIG_DIR, f"cell_{cell.lower()}_smoke.yaml")
        with open(smoke_config, "w") as f:
            yaml.dump(cfg, f)
        cmd = [sys.executable, os.path.join(REPO_DIR, "train.py"), "--config", smoke_config]
        print(f"SMOKE TEST: 50 steps only")

    print(f"Command: {' '.join(cmd)}")
    print()

    with open(log_file, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=REPO_DIR,
            env=os.environ.copy(),
        )
        for line in proc.stdout:
            line = line.decode()
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"\nCell {cell} training FAILED (exit code {proc.returncode})")
        return False
    print(f"\nCell {cell} training completed successfully")
    return True


def status():
    """Show status of training runs."""
    print("Training status:\n")
    for cell in sorted(CELL_CONFIGS.keys()):
        ckpt = os.path.join(CKPT_DIR, f"cell{cell}_latest.pt")
        log = os.path.join(REPO_DIR, f"train_cell{cell}_tt.log")
        ckpt_exists = os.path.exists(ckpt)
        log_exists = os.path.exists(log)
        ckpt_size = f"{os.path.getsize(ckpt) / 1e6:.1f} MB" if ckpt_exists else "—"

        # Get last step from log
        last_step = "—"
        if log_exists:
            with open(log) as f:
                for line in f:
                    if "Step " in line and "/" in line:
                        last_step = line.strip().split("|")[0].strip()

        print(f"  Cell {cell}: ckpt={ckpt_size:>10s}  last_log={last_step}")

    # Check if any training is running
    try:
        result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        if result.stdout.strip():
            print(f"\n  Running: {len(result.stdout.strip().split())} train.py process(es)")
        else:
            print("\n  No training processes running")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Tenstorrent Quietbox 2 training runner")
    parser.add_argument("--cell", nargs="+", default=None,
                        help="Cell(s) to train (A B C D). Default: all 4 sequentially")
    parser.add_argument("--status", action="store_true", help="Show training status")
    parser.add_argument("--clean", action="store_true", help="Delete checkpoints and start fresh")
    parser.add_argument("--smoke", action="store_true", help="Quick 50-step smoke test")
    args = parser.parse_args()

    if args.status:
        status()
        return

    if args.clean:
        cells = args.cell or list(CELL_CONFIGS.keys())
        clean_checkpoints(cells)
        return

    cells = args.cell or ["A", "B", "C", "D"]
    for cell in cells:
        if cell not in CELL_CONFIGS:
            print(f"ERROR: Unknown cell '{cell}'. Must be one of: A B C D")
            sys.exit(1)

    print(f"Tenstorrent Quietbox 2 — training cells: {', '.join(cells)}")
    print(f"Device: single Blackhole chip (bf16-native)")
    print(f"TT_VISIBLE_DEVICES: {os.environ.get('TT_VISIBLE_DEVICES', '(all)')}")
    print()

    results = {}
    for cell in cells:
        success = run_cell(cell, smoke=args.smoke)
        results[cell] = "PASS" if success else "FAIL"
        if not success and not args.smoke:
            print(f"\nCell {cell} failed — stopping. Use --cell to skip failed cells.")
            break

    print(f"\n{'='*60}")
    print("Results:")
    for cell, status in results.items():
        print(f"  Cell {cell}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
