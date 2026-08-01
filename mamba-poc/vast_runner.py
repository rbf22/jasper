"""
DEPRECATED — this runner was for the pytorch-on-GPU workflow (Vast.ai).
The project has moved to tt-nn on Tenstorrent Blackhole. Use run_all_cells.sh
or tt_runner.py instead. Cell references below updated to the new naming
(old B→deprecated, old D→C) but config files are no longer present.

Vast.ai runner — launches Cell A and Cell C in parallel on dual GPUs,
or sequentially on a single GPU.

Usage:
    python vast_runner.py                    # train both cells (auto-detects GPUs)
    python vast_runner.py --status           # check status and recent logs
    python vast_runner.py --wait             # wait for running jobs to finish
    python vast_runner.py --clean            # delete checkpoints and start fresh
    python vast_runner.py --save-outputs     # copy checkpoints + probes to output dir
    python vast_runner.py --sequential       # run A then C on single GPU

Setup on Vast.ai:
    git clone https://github.com/rbf22/jasper.git
    cd jasper/mamba-poc
    pip install einops pyyaml wandb numpy
    python vast_runner.py --clean            # start training
"""

import os
import sys
import time
import glob
import shutil
import subprocess
import argparse
import torch


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(REPO_DIR, "outputs"))
LOG_A = os.path.join(REPO_DIR, "train_cellA.log")
LOG_C = os.path.join(REPO_DIR, "train_cellC.log")
CONFIG_A = os.path.join(REPO_DIR, "configs", "cell_a_tt.yaml")
CONFIG_C = os.path.join(REPO_DIR, "configs", "cell_c_tt.yaml")


def clean_checkpoints():
    """Delete all checkpoints for a fresh start."""
    ckpt_dir = os.path.join(REPO_DIR, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith(".pt"):
                path = os.path.join(ckpt_dir, f)
                os.remove(path)
                print(f"Deleted: {path}")


def launch_training_parallel(clean=False):
    """Launch Cell A on GPU 0 and Cell C on GPU 1 in parallel."""
    if clean:
        clean_checkpoints()

    env_a = os.environ.copy()
    env_a["CUDA_VISIBLE_DEVICES"] = "0"
    env_a["PYTHONUNBUFFERED"] = "1"
    env_c = os.environ.copy()
    env_c["CUDA_VISIBLE_DEVICES"] = "1"
    env_c["PYTHONUNBUFFERED"] = "1"

    log_a = open(LOG_A, "w", buffering=1)
    log_c = open(LOG_C, "w", buffering=1)

    print("Launching Cell A on GPU 0 and Cell C on GPU 1...")
    proc_a = subprocess.Popen(
        ["python", "-u", "train.py", "--config", CONFIG_A],
        env=env_a, stdout=log_a, stderr=subprocess.STDOUT, cwd=REPO_DIR,
    )
    proc_c = subprocess.Popen(
        ["python", "-u", "train.py", "--config", CONFIG_C],
        env=env_c, stdout=log_c, stderr=subprocess.STDOUT, cwd=REPO_DIR,
    )

    print(f"Cell A PID: {proc_a.pid} (GPU 0)")
    print(f"Cell C PID: {proc_c.pid} (GPU 1)")
    print("Training in background. Press Ctrl+C to stop (processes keep running).\n")

    _wait_and_report(proc_a, proc_c, log_a, log_c)


def launch_training_sequential(clean=False):
    """Run Cell A then Cell C on a single GPU."""
    if clean:
        clean_checkpoints()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for name, config, logfile in [("Cell A", CONFIG_A, LOG_A), ("Cell C", CONFIG_C, LOG_C)]:
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}\n")

        log_f = open(logfile, "w", buffering=1)
        proc = subprocess.Popen(
            ["python", "-u", "train.py", "--config", config],
            env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO_DIR,
        )

        while proc.poll() is None:
            time.sleep(300)
            if os.path.exists(logfile):
                with open(logfile) as f:
                    lines = f.readlines()
                if lines:
                    print(f"[{time.strftime('%H:%M:%S')}] {name}: {lines[-1].rstrip()}")

        log_f.close()
        print(f"\n{name} exit code: {proc.returncode}")
        if proc.returncode != 0:
            print(f"{name} FAILED — check {os.path.basename(logfile)}")
            return proc.returncode

    print("\nBoth training runs completed successfully!")
    save_outputs()
    return 0


def _wait_and_report(proc_a, proc_c, log_a, log_c):
    """Wait for both processes and report status periodically."""
    while proc_a.poll() is None or proc_c.poll() is None:
        time.sleep(300)  # 5 minutes
        a_status = "RUNNING" if proc_a.poll() is None else f"DONE (exit {proc_a.returncode})"
        c_status = "RUNNING" if proc_c.poll() is None else f"DONE (exit {proc_c.returncode})"
        print(f"[{time.strftime('%H:%M:%S')}] Cell A: {a_status} | Cell C: {c_status}")
        for name, logfile in [("A", LOG_A), ("C", LOG_C)]:
            if os.path.exists(logfile):
                with open(logfile) as f:
                    lines = f.readlines()
                if lines:
                    for line in lines[-3:]:
                        print(f"  [{name}] {line.rstrip()}")

    log_a.close()
    log_c.close()

    print(f"\nCell A exit code: {proc_a.returncode}")
    print(f"Cell C exit code: {proc_c.returncode}")
    if proc_a.returncode != 0:
        print("Cell A FAILED — check train_cellA.log")
    if proc_c.returncode != 0:
        print("Cell C FAILED — check train_cellC.log")
    if proc_a.returncode == 0 and proc_c.returncode == 0:
        print("Both training runs completed successfully!")

    save_outputs()


def status():
    """Show status and recent logs from running training jobs."""
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    train_procs = [l for l in result.stdout.split("\n") if "train.py" in l and "grep" not in l]
    print(f"Running train.py processes: {len(train_procs)}")
    for p in train_procs:
        print(f"  {p[:120]}")

    for name, logfile in [("Cell A", LOG_A), ("Cell C", LOG_C)]:
        print(f"\n=== {name} ===")
        if os.path.exists(logfile):
            with open(logfile) as f:
                lines = f.readlines()
            if lines:
                print(f"  ({len(lines)} lines total, showing last 30)")
                for line in lines[-30:]:
                    print(line, end="")
            else:
                print("  Log file is empty — process may still be starting up")
        else:
            print("  Log not found")


def wait():
    """Block until no train.py processes are running."""
    print("Waiting for training to finish...")
    while True:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        train_procs = [l for l in result.stdout.split("\n") if "train.py" in l and "grep" not in l]
        if len(train_procs) == 0:
            print("No training processes running.")
            break
        time.sleep(300)
        print(f"[{time.strftime('%H:%M:%S')}] Still running: {len(train_procs)} processes")
    status()


def save_outputs():
    """Copy checkpoints, probe outputs, and logs to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ckpt_dir = os.path.join(REPO_DIR, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            src = os.path.join(ckpt_dir, f)
            dst = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"Saved: {dst} ({os.path.getsize(src) / 1e6:.1f} MB)")

    for fname in os.listdir(REPO_DIR):
        if fname.startswith("probe_") and (fname.endswith(".json") or fname.endswith(".csv") or fname.endswith(".png")):
            shutil.copy2(os.path.join(REPO_DIR, fname), os.path.join(OUTPUT_DIR, fname))
            print(f"Saved: {fname}")

    for logfile in [LOG_A, LOG_C]:
        if os.path.exists(logfile):
            dst = os.path.join(OUTPUT_DIR, os.path.basename(logfile))
            shutil.copy2(logfile, dst)
            print(f"Saved: {os.path.basename(logfile)}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Vast.ai parallel training runner")
    parser.add_argument("--status", action="store_true", help="Check status and recent logs")
    parser.add_argument("--wait", action="store_true", help="Wait for training to finish")
    parser.add_argument("--clean", action="store_true", help="Delete checkpoints before training")
    parser.add_argument("--save-outputs", action="store_true", help="Copy outputs to output dir")
    parser.add_argument("--sequential", action="store_true", help="Run A then C on single GPU")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.wait:
        wait()
    elif args.save_outputs:
        save_outputs()
    else:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if args.sequential or n_gpus < 2:
            print(f"Running sequential (single GPU mode, {n_gpus} GPU(s) detected)")
            sys.exit(launch_training_sequential(clean=args.clean))
        else:
            print(f"Running parallel ({n_gpus} GPUs detected)")
            launch_training_parallel(clean=args.clean)


if __name__ == "__main__":
    main()
