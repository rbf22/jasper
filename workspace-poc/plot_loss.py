#!/usr/bin/env python3
"""Plot loss vs log(step) for Cell A, B, C using plotext."""
import re
import math
import plotext as plt

def parse_log(path):
    data = []
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r'\s*(\d+)\s+([\d.]+)\s', line)
                if m:
                    data.append((int(m.group(1)), float(m.group(2))))
    except FileNotFoundError:
        pass
    return data

def merge_logs(paths):
    all_data = []
    for p in paths:
        all_data.extend(parse_log(p))
    by_step = {}
    for step, loss in all_data:
        by_step[step] = loss
    return sorted(by_step.items())

cell_a = parse_log('logs/cell_a_stability_fix_20260810.log')
cell_b = parse_log('logs/cell_b_stability_fix_20260810.log')
cell_c = merge_logs([
    'logs/cell_c_stability_fix_20260810.log',
    'logs/cell_c_stability_fix_20260810b.log',
])

def to_log_step(steps):
    return [math.log10(max(s, 1)) for s in steps]

cell_a_x = to_log_step([s for s, _ in cell_a])
cell_a_y = [l for _, l in cell_a]
cell_b_x = to_log_step([s for s, _ in cell_b])
cell_b_y = [l for _, l in cell_b]
cell_c_x = to_log_step([s for s, _ in cell_c])
cell_c_y = [l for _, l in cell_c]

cell_a_best = min(cell_a_y) if cell_a_y else 0
cell_a_final = cell_a_y[-1] if cell_a_y else 0

plt.clf()
plt.theme("pro")
plt.title("Loss vs log10(step)")
plt.xlabel("log10(step)")
plt.ylabel("loss")

# Only plot Cell A if it has data within the y-axis range
cell_a_in_range = [p for p in cell_a if 0.8 <= p[1] <= 2.0]
if cell_a_in_range:
    plt.plot(cell_a_x, cell_a_y, label=f"Cell A (control, cur={cell_a_y[-1]:.3f})", marker="dot")
if cell_b:
    plt.plot(cell_b_x, cell_b_y, label=f"Cell B (workspace, cur={cell_b_y[-1]:.3f})", marker="big")
if cell_c:
    plt.plot(cell_c_x, cell_c_y, label=f"Cell C (workspace+recur, cur={cell_c_y[-1]:.3f})", marker="x")

if 0.8 <= cell_a_best <= 2.0:
    plt.hline(cell_a_best)
if 0.8 <= cell_a_final <= 2.0:
    plt.hline(cell_a_final)

plt.ylim(0.8, 2.0)
plt.xticks([2, 3, 4], ["100", "1000", "10000"])
plt.xlim(2, 4)
plt.plotsize(120, 35)

# Build the plot, then save and print
plt.build()
plt.save_fig("/tmp/loss_plot.txt", keep_colors=False)

# Print the saved file
with open("/tmp/loss_plot.txt") as f:
    print(f.read())

if cell_a:
    print(f"  Cell A: steps 0-{cell_a[-1][0]}, best={cell_a_best:.3f}, current={cell_a_final:.3f}")
if cell_b:
    print(f"  Cell B: steps 0-{cell_b[-1][0]}, current={cell_b_y[-1]:.3f}")
if cell_c:
    print(f"  Cell C: steps 0-{cell_c[-1][0]}, current={cell_c_y[-1]:.3f}")
