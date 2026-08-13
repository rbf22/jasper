"""Minimal reproduction of ttnn C++ heap leak — v2.

v1 (linear + mul + add) showed decelerating growth, NOT linear.
The full model backward uses reshape/permute/transpose/sum extensively.
This version adds those ops to try to reproduce the linear leak.
"""
import gc
import ctypes
import tracemalloc
import torch
import ttnn


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024


def gc_trim():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


# --- Setup ---
device = ttnn.open_device(device_id=0)

D = 384
B, T = 8, 128
H = 4
d_h = D // H

# Persistent tensors
x = ttnn.from_torch(
    torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
)
w = ttnn.from_torch(
    torch.randn(D, D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
)
wt = ttnn.transpose(w, 0, 1)

ITERS = 300

# --- Warmup ---
for _ in range(5):
    y = ttnn.linear(x, wt)
    y4 = ttnn.reshape(y, [B, T, H, d_h])
    y4p = ttnn.permute(y4, [0, 2, 1, 3])
    y4p_t = ttnn.transpose(y4p, -2, -1)
    scores = ttnn.matmul(y4p, y4p_t)
    scores = ttnn.mul(scores, scores)
    y_sum = ttnn.sum(scores, dim=-1)
    y2 = ttnn.reshape(y_sum, [B, T, H])
    y = ttnn.add(y2, x[:, :, :H])
    ttnn.synchronize_device(device)
    ttnn.deallocate(y)
    ttnn.deallocate(y_sum)
    ttnn.deallocate(y4p)
    ttnn.deallocate(y4p_t)
    ttnn.deallocate(scores)
    ttnn.deallocate(y2)

ttnn.synchronize_device(device)
gc_trim()
tracemalloc.start(25)

rss0 = rss_mb()
tm0, _ = tracemalloc.get_traced_memory()
print(f"Baseline:  RSS={rss0:.1f} MB  Python={tm0/1024/1024:.1f} MB")
print(f"Running {ITERS} iterations...")
print(f"{'Iter':>6}  {'RSS':>10}  {'Delta':>8}  {'Rate':>12}  {'Python':>10}")
print("-" * 60)

for i in range(ITERS):
    # Mimic retention backward: linear, reshape, permute, matmul, transpose, mul, add, sum
    y = ttnn.linear(x, wt)                           # (B, T, D)
    y4 = ttnn.reshape(y, [B, T, H, d_h])              # reshape (view)
    y4p = ttnn.permute(y4, [0, 2, 1, 3])              # permute (copy) -> (B, H, T, d_h)
    y4p_t = ttnn.transpose(y4p, -2, -1)               # transpose (copy) -> (B, H, d_h, T)
    scores = ttnn.matmul(y4p, y4p_t)                  # matmul -> (B, H, T, T)
    scores = ttnn.mul(scores, scores)                 # elementwise mul
    y_sum = ttnn.sum(scores, dim=-1)                  # reduction -> (B, H, T)
    y2 = ttnn.reshape(y_sum, [B, T, H])               # reshape back
    y = ttnn.add(y2, x[:, :, :H])                     # add (slice of x)
    ttnn.synchronize_device(device)
    ttnn.deallocate(y)
    ttnn.deallocate(y_sum)
    ttnn.deallocate(y4p)
    ttnn.deallocate(y4p_t)
    ttnn.deallocate(scores)
    ttnn.deallocate(y2)

    if (i + 1) % 50 == 0 or i == 0:
        ttnn.synchronize_device(device)
        gc_trim()
        rss = rss_mb()
        tm, _ = tracemalloc.get_traced_memory()
        delta = rss - rss0
        rate = delta / (i + 1)
        print(f"{i+1:6d}  {rss:8.1f} MB  {delta:+6.1f} MB  {rate:8.3f} MB/it  {tm/1024/1024:8.1f} MB")

tracemalloc.stop()
print(f"\nDone. RSS grew by {rss_mb() - rss0:.1f} MB over {ITERS} iterations.")

ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(wt)
ttnn.close_device(device)
