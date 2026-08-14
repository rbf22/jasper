#!/usr/bin/env python3
"""Monitor loss comparison across cells A, B, C.

Parses the training logs and prints a comparison table showing
the latest loss for each cell and the gap vs Cell A.

Run: python monitor_loss.py [--watch SECONDS]
"""
import re
import sys
import time
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOGS = {
    "A": "cell_a_wrap_20260812.log",
    "B": "cell_b_wrap_20260812.log",
    "C": "cell_c_wrap_20260812.log",
}

# Matches lines like:
#   8300     0.8607   0.000020    3.62s      13562     8.0932  ...
STEP_RE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+s\s+[\d]+\s+([\d.]+)"
)


def parse_latest(log_path):
    """Return (step, loss, grad_norm) from the last data line in the log."""
    if not os.path.exists(log_path):
        return None
    last = None
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                last = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
    return last


def print_comparison():
    results = {}
    for cell, fname in LOGS.items():
        path = os.path.join(LOG_DIR, fname)
        results[cell] = parse_latest(path)

    a_loss = results["A"][1] if results["A"] else None

    print(f"\n{'Cell':<6} {'Step':>7} {'Loss':>8} {'GradNorm':>10} {'Gap vs A':>10}")
    print("-" * 45)
    for cell in ["A", "B", "C"]:
        r = results[cell]
        if r is None:
            print(f"{cell:<6} {'—':>7} {'—':>8} {'—':>10} {'—':>10}")
            continue
        step, loss, gn = r
        gap = f"+{loss - a_loss:.4f}" if a_loss else "—"
        print(f"{cell:<6} {step:>7} {loss:>8.4f} {gn:>10.2f} {gap:>10}")

    if a_loss:
        b = results["B"]
        c = results["C"]
        if b and b[1] < a_loss:
            print(f"\n  >> Cell B has surpassed Cell A (loss {b[1]:.4f} < {a_loss:.4f})")
        if c and c[1] < a_loss:
            print(f"\n  >> Cell C has surpassed Cell A (loss {c[1]:.4f} < {a_loss:.4f})")


if __name__ == "__main__":
    watch = None
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        watch = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    if watch:
        print(f"Watching every {watch}s. Press Ctrl+C to stop.")
        try:
            while True:
                print_comparison()
                sys.stdout.flush()
                time.sleep(watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_comparison()
