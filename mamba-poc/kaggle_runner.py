"""
Kaggle runner — launches Cell B and Cell D in parallel on dual T4 GPUs.

Usage:
    python kaggle_runner.py                    # train both cells
    python kaggle_runner.py --status           # check status and recent logs
    python kaggle_runner.py --wait             # wait for running jobs to finish
    python kaggle_runner.py --clean            # delete checkpoints and start fresh
    python kaggle_runner.py --save-outputs     # copy checkpoints + probes to /kaggle/working
    python kaggle_runner.py --smoke-test       # run smoke test on both cells

The notebook just needs to call this script. When you git pull, this file
updates automatically — no need to copy/paste code into notebook cells.
"""

import os
import sys
import time
import glob
import shutil
import subprocess
import argparse


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_OUTPUT = "/kaggle/working"
LOG_B = os.path.join(REPO_DIR, "train_cellB.log")
LOG_D = os.path.join(REPO_DIR, "train_cellD.log")
CONFIG_B = os.path.join(REPO_DIR, "configs", "cell_b_kaggle.yaml")
CONFIG_D = os.path.join(REPO_DIR, "configs", "cell_d_kaggle.yaml")


def clean_checkpoints():
    """Delete all checkpoints for a fresh start."""
    ckpt_dir = os.path.join(REPO_DIR, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith(".pt"):
                path = os.path.join(ckpt_dir, f)
                os.remove(path)
                print(f"Deleted: {path}")
    # Also clean any in /kaggle/working
    if os.path.isdir(KAGGLE_OUTPUT):
        for f in os.listdir(KAGGLE_OUTPUT):
            if f.startswith("cell") and f.endswith("_latest.pt"):
                path = os.path.join(KAGGLE_OUTPUT, f)
                os.remove(path)
                print(f"Deleted: {path}")


def launch_training(clean=False):
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
    print("Training in background. This cell stays alive with periodic updates.\n")

    # Keep alive with periodic status until both finish
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

    # Auto-save outputs after training
    save_outputs()

    return proc_b.returncode, proc_d.returncode


def status():
    """Show status and recent logs from running training jobs."""
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    train_procs = [l for l in result.stdout.split("\n") if "train.py" in l and "grep" not in l]
    print(f"Running train.py processes: {len(train_procs)}")
    for p in train_procs:
        print(f"  {p[:120]}")

    for name, logfile in [("Cell B (GPU 0)", LOG_B), ("Cell D (GPU 1)", LOG_D)]:
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
    """Copy checkpoints and probe outputs to /kaggle/working for persistence."""
    if not os.path.isdir(KAGGLE_OUTPUT):
        print(f"{KAGGLE_OUTPUT} not found — not on Kaggle, skipping.")
        return

    # Copy checkpoints
    ckpt_dir = os.path.join(REPO_DIR, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            src = os.path.join(ckpt_dir, f)
            dst = os.path.join(KAGGLE_OUTPUT, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"Saved: {dst} ({os.path.getsize(src) / 1e6:.1f} MB)")

    # Copy probe outputs
    for fname in os.listdir(REPO_DIR):
        if fname.startswith("probe_") and (fname.endswith(".json") or fname.endswith(".csv") or fname.endswith(".png")):
            shutil.copy2(os.path.join(REPO_DIR, fname), os.path.join(KAGGLE_OUTPUT, fname))
            print(f"Saved: {fname}")

    # Copy training logs
    for logfile in [LOG_B, LOG_D]:
        if os.path.exists(logfile):
            dst = os.path.join(KAGGLE_OUTPUT, os.path.basename(logfile))
            shutil.copy2(logfile, dst)
            print(f"Saved: {os.path.basename(logfile)}")

    print(f"\nAll outputs saved to {KAGGLE_OUTPUT}/ — download from the Output tab after session ends.")


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
    parser = argparse.ArgumentParser(description="Kaggle parallel training runner")
    parser.add_argument("--status", action="store_true", help="Check status and recent logs")
    parser.add_argument("--wait", action="store_true", help="Wait for training to finish")
    parser.add_argument("--clean", action="store_true", help="Delete checkpoints before training")
    parser.add_argument("--save-outputs", action="store_true", help="Copy outputs to /kaggle/working")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests")
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
        sys.exit(launch_training(clean=args.clean)[0] or 0)


if __name__ == "__main__":
    main()
