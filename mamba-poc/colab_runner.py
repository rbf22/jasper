"""
Colab runner — trains Cell B and Cell D sequentially on a single T4 GPU.

Usage:
    python colab_runner.py                    # train both cells sequentially
    python colab_runner.py --status           # check status and recent logs
    python colab_runner.py --wait             # wait for running jobs to finish
    python colab_runner.py --clean            # delete checkpoints and start fresh
    python colab_runner.py --save-outputs     # copy checkpoints + probes to Google Drive
    python colab_runner.py --smoke-test       # run smoke test on both cells
    python colab_runner.py --cell B           # train only Cell B
    python colab_runner.py --cell D           # train only Cell D

Setup on Colab:
    git clone https://github.com/rbf22/jasper.git
    cd jasper/mamba-poc
    pip install einops pyyaml wandb numpy
    python colab_runner.py --smoke-test       # verify everything works
    python colab_runner.py --clean            # start training
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
CONFIG_B = os.path.join(REPO_DIR, "configs", "cell_b_colab.yaml")
CONFIG_D = os.path.join(REPO_DIR, "configs", "cell_d_colab.yaml")

# Google Drive mount point (Colab standard)
GDRIVE_DIR = "/content/drive/MyDrive"


def clean_checkpoints():
    """Delete all checkpoints for a fresh start."""
    ckpt_dir = os.path.join(REPO_DIR, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith(".pt"):
                path = os.path.join(ckpt_dir, f)
                os.remove(path)
                print(f"Deleted: {path}")


def launch_training_sequential(clean=False, cell_filter=None):
    """Run Cell B then Cell D on a single GPU."""
    if clean:
        clean_checkpoints()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cells = []
    if cell_filter is None or cell_filter.upper() == "B":
        cells.append(("Cell B", CONFIG_B, LOG_B))
    if cell_filter is None or cell_filter.upper() == "D":
        cells.append(("Cell D", CONFIG_D, LOG_D))

    for name, config, logfile in cells:
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

    print("\nAll training runs completed successfully!")
    save_outputs()
    return 0


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
    """Copy checkpoints, probe outputs, and logs to output directory and Google Drive."""
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

    # Also copy to Google Drive if mounted
    gdrive_output = os.path.join(GDRIVE_DIR, "jasper-outputs")
    if os.path.isdir(GDRIVE_DIR):
        os.makedirs(gdrive_output, exist_ok=True)
        for f in os.listdir(OUTPUT_DIR):
            src = os.path.join(OUTPUT_DIR, f)
            dst = os.path.join(gdrive_output, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"Also saved to Google Drive: {gdrive_output}/")
    else:
        print("Google Drive not mounted — skipping Drive backup.")
        print("Run: from google.colab import drive; drive.mount('/content/drive')")


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
    parser = argparse.ArgumentParser(description="Colab sequential training runner")
    parser.add_argument("--status", action="store_true", help="Check status and recent logs")
    parser.add_argument("--wait", action="store_true", help="Wait for training to finish")
    parser.add_argument("--clean", action="store_true", help="Delete checkpoints before training")
    parser.add_argument("--save-outputs", action="store_true", help="Copy outputs to Google Drive")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests")
    parser.add_argument("--cell", type=str, default=None, help="Train only one cell (B or D)")
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
        print(f"GPUs detected: {n_gpus} (Colab uses single GPU — running sequentially)")
        sys.exit(launch_training_sequential(clean=args.clean, cell_filter=args.cell))


if __name__ == "__main__":
    main()
