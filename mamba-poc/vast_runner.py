"""
Vast.ai runner — launches Cell B and Cell D in parallel on dual GPUs,
or sequentially on a single GPU.

Usage:
    python vast_runner.py                    # train both cells (auto-detects GPUs)
    python vast_runner.py --status           # check status and recent logs
    python vast_runner.py --wait             # wait for running jobs to finish
    python vast_runner.py --clean            # delete checkpoints and start fresh
    python vast_runner.py --save-outputs     # copy checkpoints + probes to output dir
    python vast_runner.py --smoke-test       # run smoke test on both cells
    python vast_runner.py --sequential       # run B then D on single GPU

Setup on Vast.ai:
    git clone https://github.com/rbf22/jasper.git
    cd jasper/mamba-poc
    pip install einops pyyaml wandb numpy
    python vast_runner.py --smoke-test       # verify everything works
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
LOG_B = os.path.join(REPO_DIR, "train_cellB.log")
LOG_D = os.path.join(REPO_DIR, "train_cellD.log")
CONFIG_B = os.path.join(REPO_DIR, "configs", "cell_b_vast.yaml")
CONFIG_D = os.path.join(REPO_DIR, "configs", "cell_d_vast.yaml")


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
    """Launch Cell B on GPU 0 and Cell D on GPU 1 in parallel."""
    if clean:
        clean_checkpoints()

    env_b = os.environ.copy()
    env_b["CUDA_VISIBLE_DEVICES"] = "0"
    env_b["PYTHONUNBUFFERED"] = "1"
    env_d = os.environ.copy()
    env_d["CUDA_VISIBLE_DEVICES"] = "1"
    env_d["PYTHONUNBUFFERED"] = "1"

    log_b = open(LOG_B, "w", buffering=1)
    log_d = open(LOG_D, "w", buffering=1)

    print("Launching Cell B on GPU 0 and Cell D on GPU 1...")
    proc_b = subprocess.Popen(
        ["python", "-u", "train.py", "--config", CONFIG_B],
        env=env_b, stdout=log_b, stderr=subprocess.STDOUT, cwd=REPO_DIR,
    )
    proc_d = subprocess.Popen(
        ["python", "-u", "train.py", "--config", CONFIG_D],
        env=env_d, stdout=log_d, stderr=subprocess.STDOUT, cwd=REPO_DIR,
    )

    print(f"Cell B PID: {proc_b.pid} (GPU 0)")
    print(f"Cell D PID: {proc_d.pid} (GPU 1)")
    print("Training in background. Press Ctrl+C to stop (processes keep running).\n")

    _wait_and_report(proc_b, proc_d, log_b, log_d)


def launch_training_sequential(clean=False):
    """Run Cell B then Cell D on a single GPU."""
    if clean:
        clean_checkpoints()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for name, config, logfile in [("Cell B", CONFIG_B, LOG_B), ("Cell D", CONFIG_D, LOG_D)]:
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


def _wait_and_report(proc_b, proc_d, log_b, log_d):
    """Wait for both processes and report status periodically."""
    while proc_b.poll() is None or proc_d.poll() is None:
        time.sleep(300)  # 5 minutes
        b_status = "RUNNING" if proc_b.poll() is None else f"DONE (exit {proc_b.returncode})"
        d_status = "RUNNING" if proc_d.poll() is None else f"DONE (exit {proc_d.returncode})"
        print(f"[{time.strftime('%H:%M:%S')}] Cell B: {b_status} | Cell D: {d_status}")
        for name, logfile in [("B", LOG_B), ("D", LOG_D)]:
            if os.path.exists(logfile):
                with open(logfile) as f:
                    lines = f.readlines()
                if lines:
                    for line in lines[-3:]:
                        print(f"  [{name}] {line.rstrip()}")

    log_b.close()
    log_d.close()

    print(f"\nCell B exit code: {proc_b.returncode}")
    print(f"Cell D exit code: {proc_d.returncode}")
    if proc_b.returncode != 0:
        print("Cell B FAILED — check train_cellB.log")
    if proc_d.returncode != 0:
        print("Cell D FAILED — check train_cellD.log")
    if proc_b.returncode == 0 and proc_d.returncode == 0:
        print("Both training runs completed successfully!")

    save_outputs()


def status():
    """Show status and recent logs from running training jobs."""
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    train_procs = [l for l in result.stdout.split("\n") if "train.py" in l and "grep" not in l]
    print(f"Running train.py processes: {len(train_procs)}")
    for p in train_procs:
        print(f"  {p[:120]}")

    for name, logfile in [("Cell B", LOG_B), ("Cell D", LOG_D)]:
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

    for logfile in [LOG_B, LOG_D]:
        if os.path.exists(logfile):
            dst = os.path.join(OUTPUT_DIR, os.path.basename(logfile))
            shutil.copy2(logfile, dst)
            print(f"Saved: {os.path.basename(logfile)}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


def smoke_test():
    """Run smoke test on both cells sequentially."""
    for name, config in [("Cell B", CONFIG_B), ("Cell D", CONFIG_D)]:
        print(f"\n=== Smoke test: {name} ===")
        result = subprocess.run(
            ["python", "train.py", "--config", config, "--smoke-test"],
            cwd=REPO_DIR,
        )
        if result.returncode != 0:
            print(f"{name} smoke test FAILED")
            return result.returncode
    print("\nBoth smoke tests passed!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Vast.ai parallel training runner")
    parser.add_argument("--status", action="store_true", help="Check status and recent logs")
    parser.add_argument("--wait", action="store_true", help="Wait for training to finish")
    parser.add_argument("--clean", action="store_true", help="Delete checkpoints before training")
    parser.add_argument("--save-outputs", action="store_true", help="Copy outputs to output dir")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests")
    parser.add_argument("--sequential", action="store_true", help="Run B then D on single GPU")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.wait:
        wait()
    elif args.save_outputs:
        save_outputs()
    elif args.smoke_test:
        sys.exit(smoke_test())
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
