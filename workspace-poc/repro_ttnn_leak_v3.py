"""Minimal repro v3: forward caches tensors, backward reads them.

v1 and v2 (stateless ops) showed only 0.008 MB/iter — decelerating.
The real backward reads cached tensors from the forward. Maybe the leak
is from the inter-op tensor dependency tracking in the ttnn runtime.
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


device = ttnn.open_device(device_id=0)

D = 384
B, T = 8, 128
H = 4
d_h = D // H

# Persistent weights
w = ttnn.from_torch(
    torch.randn(D, 4 * D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
)
wt = ttnn.transpose(w, 0, 1)
w_out = ttnn.from_torch(
    torch.randn(D, D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
)
w_out_t = ttnn.transpose(w_out, 0, 1)

ITERS = 300


def forward_with_cache(x):
    """Mimic retention forward: cache intermediates for backward."""
    c = {}
    c["x"] = x
    y = ttnn.linear(x, w)                        # qkv+gate
    c["qkvg"] = y
    y4 = ttnn.reshape(y, [B, T, H, 4 * d_h])
    # Split into q, k, v, g (use slices)
    q = ttnn.slice(y4, [0, 0, 0, 0], [B, T, H, d_h])
    k = ttnn.slice(y4, [0, 0, 0, d_h], [B, T, H, 2 * d_h])
    v = ttnn.slice(y4, [0, 0, 0, 2 * d_h], [B, T, H, 3 * d_h])
    g = ttnn.slice(y4, [0, 0, 0, 3 * d_h], [B, T, H, 4 * d_h])
    q = ttnn.permute(q, [0, 2, 1, 3])  # (B, H, T, d_h)
    k = ttnn.permute(k, [0, 2, 1, 3])
    v = ttnn.permute(v, [0, 2, 1, 3])
    c["q"], c["k"], c["v"] = q, k, v
    c["g"] = ttnn.reshape(g, [B, T, D])
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))  # (B, H, T, T)
    c["scores"] = scores
    out = ttnn.matmul(scores, v)  # (B, H, T, d_h)
    c["out_4d"] = out
    out = ttnn.permute(out, [0, 2, 1, 3])  # (B, T, H, d_h)
    out = ttnn.reshape(out, [B, T, D])
    c["out_flat"] = out
    g_sig = ttnn.sigmoid(ttnn.reshape(g, [B, T, D]))
    out_gated = ttnn.mul(out, g_sig)
    c["out_gated"] = out_gated
    c["gate"] = ttnn.reshape(g, [B, T, D])
    result = ttnn.linear(out_gated, w_out)
    ttnn.deallocate(g_sig)
    return result, c


def backward_with_cache(grad_out, c):
    """Mimic retention backward: read cached tensors, compute gradients."""
    # grad through out_proj
    grad_out_gated = ttnn.linear(grad_out, w_out_t)
    out_gated_2d = ttnn.reshape(c["out_gated"], [B * T, D])
    grad_out_2d = ttnn.reshape(grad_out, [B * T, D])
    grad_w_out = ttnn.matmul(ttnn.transpose(out_gated_2d, 0, 1), grad_out_2d)
    _safe = ttnn.deallocate
    _safe(grad_w_out)

    # grad through gate (sigmoid backward)
    grad_gate = ttnn.mul(grad_out_gated, c["gate"])
    grad_gate = ttnn.mul(grad_gate, c["out_flat"])
    _safe(grad_gate)

    # grad through scores@v
    grad_out_4d = ttnn.permute(
        ttnn.reshape(grad_out_gated, [B, T, H, d_h]), [0, 2, 1, 3])
    grad_scores = ttnn.matmul(grad_out_4d, ttnn.transpose(c["v"], -2, -1))
    grad_v = ttnn.matmul(ttnn.transpose(c["scores"], -2, -1), grad_out_4d)
    _safe(grad_out_4d)
    _safe(grad_v)

    # grad through qk = q @ k^T
    grad_q = ttnn.matmul(grad_scores, c["k"])
    grad_k = ttnn.matmul(ttnn.transpose(grad_scores, -2, -1), c["q"])
    _safe(grad_scores)
    _safe(grad_q)
    _safe(grad_k)

    # grad through qkv linear
    grad_x = ttnn.matmul(grad_out_gated, w)
    _safe(grad_out_gated)

    return grad_x


# Warmup
for _ in range(3):
    x = ttnn.from_torch(
        torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out, cache = forward_with_cache(x)
    ttnn.synchronize_device(device)
    grad = ttnn.from_torch(
        torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gx = backward_with_cache(grad, cache)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)
    ttnn.deallocate(gx)
    ttnn.deallocate(grad)
    # Clean cache (but not x — it's passed in)
    for k, v in cache.items():
        if k != "x":
            ttnn.deallocate(v)
    ttnn.deallocate(x)

ttnn.synchronize_device(device)
gc_trim()
tracemalloc.start(25)

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
    out, cache = forward_with_cache(x)
    ttnn.synchronize_device(device)
    grad = ttnn.from_torch(
        torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gx = backward_with_cache(grad, cache)
    ttnn.synchronize_device(device)
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

ttnn.deallocate(w)
ttnn.deallocate(wt)
ttnn.deallocate(w_out)
ttnn.deallocate(w_out_t)
ttnn.close_device(device)
