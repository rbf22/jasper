"""Minimal reproduction of ttnn C++ heap memory leak.

BUG: ttnn leaks ~1 KB of host C++ heap memory per ttnn operation call.
The leaked memory contains JIT kernel compilation metadata (kernel source
paths, define strings, kernel binary references) that is stored per
program invocation and never freed, even when the JIT cache reports 100%
hits and the program cache entry count stays constant.

ROOT CAUSE (identified via heaptrack + brk heap string dump):
  The ttnn/tt-metal runtime stores per-invocation kernel metadata on the
  C++ heap (brk segment). This includes:
    - Kernel source file paths (e.g. "reader_interleaved_no_bcast.cpp")
    - Kernel define strings (e.g. "BINARY_SFPU_OP", "BCAST_INPUT")
    - Kernel binary references (e.g. "mul_binary_tile")
  This metadata accumulates at ~1 KB per op call and is never freed.
  The program cache itself (entry count) does not grow — only the
  per-invocation metadata leaks.

EVIDENCE:
  1. heaptrack: 414,100 leaked allocations, 46 MB total leaked
  2. brk heap string dump: leaked region contains kernel source paths
     and define strings, repeating per-op-invocation
  3. mallinfo2: brk heap grows (uordblks) but glibc doesn't report it
     correctly because ttnn uses a custom allocator layer
  4. Per-op leak rate: ~1.0 KB/op (measured by running 1x vs 20x ops)
  5. Same-shape ops leak the same as different-shape ops (not new kernel
     configs — it's per-invocation metadata)
  6. ttnn.empty does NOT leak; only ops that invoke kernels leak
  7. device.disable_and_clear_program_cache() reclaims the leaked memory
  8. ttnn.close_device() + ttnn.open_device() also reclaims it

LEAK RATE:
  - ~1 KB per ttnn operation (mul, linear, matmul, etc.)
  - This repro: ~20 ops/iteration → ~22 KB/iteration → ~6.5 MB / 300 iters
  - Full 14-layer model: ~700 ops/iteration → ~270 KB/iteration → ~2.8 GB/hour

WORKAROUNDS:
  1. Periodic device close/reopen (cheapest, preserves JIT disk cache):
       ttnn.close_device(device)
       device = ttnn.open_device(device_id=0)
     Cost: ~20 seconds for device reinit, no kernel recompilation
  2. Periodic program cache clear (reclaims leak but forces recompile):
       device.disable_and_clear_program_cache()
       device.enable_program_cache()
     Cost: kernel recompilation on next iteration

ENVIRONMENT:
  - Architecture: Blackhole (P300)
  - ttnn: see `pip show ttnn`
  - TT-KMD: 2.10.0
  - Python: 3.12
  - OS: Ubuntu 24.04

REPRODUCTION:
  TT_VISIBLE_DEVICES=0 python repro_ttnn_leak.py

EXPECTED: RSS stays flat (delta ~0 MB)
ACTUAL:   RSS grows linearly (+6.5 MB over 300 iterations)
"""
import gc
import ctypes
import tracemalloc
import torch
import ttnn


def rss_mb():
    """Process RSS in MB from /proc/self/statm."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024


def gc_trim():
    """Force Python GC + libc malloc_trim to release freed heap."""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


# --- Setup ---
device = ttnn.open_device(device_id=0)

D = 384
B, T = 8, 128
H = 4
d_h = D // H

# Persistent weights (created once, reused across all iterations)
w_qkv = ttnn.from_torch(
    torch.randn(D, 4 * D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
)
w_out = ttnn.from_torch(
    torch.randn(D, D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
)
w_out_t = ttnn.transpose(w_out, 0, 1)  # pre-cached transpose for backward

ITERS = 300


def forward(x):
    """Forward pass that caches intermediates for backward (like autograd).

    This mimics a retention/attention layer:
      qkv = linear(x, w_qkv)
      q, k, v, g = split(qkv)
      scores = q @ k^T
      out = scores @ v
      out = sigmoid(g) * out
      return linear(out, w_out)
    """
    c = {}
    c["x"] = x
    y = ttnn.linear(x, w_qkv)
    y4 = ttnn.reshape(y, [B, T, H, 4 * d_h])
    q = ttnn.permute(ttnn.slice(y4, [0, 0, 0, 0], [B, T, H, d_h]), [0, 2, 1, 3])
    k = ttnn.permute(ttnn.slice(y4, [0, 0, 0, d_h], [B, T, H, 2 * d_h]), [0, 2, 1, 3])
    v = ttnn.permute(ttnn.slice(y4, [0, 0, 0, 2 * d_h], [B, T, H, 3 * d_h]), [0, 2, 1, 3])
    g = ttnn.reshape(ttnn.slice(y4, [0, 0, 0, 3 * d_h], [B, T, H, 4 * d_h]), [B, T, D])
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
    out = ttnn.matmul(scores, v)
    out = ttnn.permute(out, [0, 2, 1, 3])
    out = ttnn.reshape(out, [B, T, D])
    g_sig = ttnn.sigmoid(g)
    out_gated = ttnn.mul(out, g_sig)
    result = ttnn.linear(out_gated, w_out)
    # Cache for backward
    c["q"], c["k"], c["v"] = q, k, v
    c["g"], c["scores"], c["out_gated"] = g, scores, out_gated
    # Clean up intermediates not needed for backward
    ttnn.deallocate(y)
    ttnn.deallocate(y4)
    ttnn.deallocate(g_sig)
    ttnn.deallocate(out)
    return result, c


def backward(grad_out, c):
    """Backward pass reading cached intermediates from forward."""
    # grad through out_proj
    grad_out_gated = ttnn.linear(grad_out, w_out_t)
    # grad through gate (sigmoid backward)
    grad_gate = ttnn.mul(grad_out_gated, c["g"])
    grad_gate = ttnn.mul(grad_gate, c["out_gated"])
    ttnn.deallocate(grad_gate)
    # grad through scores@v
    grad_out_4d = ttnn.permute(
        ttnn.reshape(grad_out_gated, [B, T, H, d_h]), [0, 2, 1, 3])
    grad_scores = ttnn.matmul(grad_out_4d, ttnn.transpose(c["v"], -2, -1))
    grad_v = ttnn.matmul(ttnn.transpose(c["scores"], -2, -1), grad_out_4d)
    ttnn.deallocate(grad_out_4d)
    ttnn.deallocate(grad_v)
    # grad through qk = q @ k^T
    grad_q = ttnn.matmul(grad_scores, c["k"])
    grad_k = ttnn.matmul(ttnn.transpose(grad_scores, -2, -1), c["q"])
    ttnn.deallocate(grad_scores)
    ttnn.deallocate(grad_q)
    ttnn.deallocate(grad_k)
    # grad through qkv linear
    grad_x = ttnn.matmul(grad_out_gated, w_qkv)
    ttnn.deallocate(grad_out_gated)
    return grad_x


# --- Warmup ---
for _ in range(3):
    x = ttnn.from_torch(
        torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out, cache = forward(x)
    ttnn.synchronize_device(device)
    grad = ttnn.from_torch(
        torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gx = backward(grad, cache)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)
    ttnn.deallocate(gx)
    ttnn.deallocate(grad)
    for k, v in cache.items():
        if k != "x":
            ttnn.deallocate(v)
    ttnn.deallocate(x)

ttnn.synchronize_device(device)
gc_trim()
tracemalloc.start(25)

# --- Main test loop ---
rss0 = rss_mb()
tm0, _ = tracemalloc.get_traced_memory()
print(f"Baseline:  RSS={rss0:.1f} MB  Python={tm0/1024/1024:.1f} MB")
print(f"Running {ITERS} iterations (forward+backward with cache)...")
print(f"{'Iter':>6}  {'RSS':>10}  {'Delta':>8}  {'Rate':>12}  {'Python':>10}")
print("-" * 60)

for i in range(ITERS):
    x = ttnn.from_torch(
        torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out, cache = forward(x)
    ttnn.synchronize_device(device)
    grad = ttnn.from_torch(
        torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gx = backward(grad, cache)
    ttnn.synchronize_device(device)
    # Explicitly deallocate everything
    ttnn.deallocate(out)
    ttnn.deallocate(gx)
    ttnn.deallocate(grad)
    for k, v in cache.items():
        if k != "x":
            ttnn.deallocate(v)
    ttnn.deallocate(x)

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
print(f"Python tracemalloc grew by 0 bytes (leak is in C++ runtime).")

ttnn.deallocate(w_qkv)
ttnn.deallocate(w_out)
ttnn.deallocate(w_out_t)
ttnn.close_device(device)
