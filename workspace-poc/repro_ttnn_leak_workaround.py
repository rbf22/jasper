"""Test if closing/reopening the device resets the C++ leak."""
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


D = 384
B, T = 8, 128
H = 4
d_h = D // H
ITERS = 100


def run_phase(device, w_qkv, w_out, w_out_t, n_iters, label):
    """Run n_iters of forward+backward, return (rss_start, rss_end)."""

    def forward(x):
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
        c["q"], c["k"], c["v"] = q, k, v
        c["g"], c["scores"], c["out_gated"] = g, scores, out_gated
        ttnn.deallocate(y)
        ttnn.deallocate(y4)
        ttnn.deallocate(g_sig)
        ttnn.deallocate(out)
        return result, c

    def backward(grad_out, c):
        grad_out_gated = ttnn.linear(grad_out, w_out_t)
        grad_gate = ttnn.mul(grad_out_gated, c["g"])
        grad_gate = ttnn.mul(grad_gate, c["out_gated"])
        ttnn.deallocate(grad_gate)
        grad_out_4d = ttnn.permute(
            ttnn.reshape(grad_out_gated, [B, T, H, d_h]), [0, 2, 1, 3])
        grad_scores = ttnn.matmul(grad_out_4d, ttnn.transpose(c["v"], -2, -1))
        grad_v = ttnn.matmul(ttnn.transpose(c["scores"], -2, -1), grad_out_4d)
        ttnn.deallocate(grad_out_4d)
        ttnn.deallocate(grad_v)
        grad_q = ttnn.matmul(grad_scores, c["k"])
        grad_k = ttnn.matmul(ttnn.transpose(grad_scores, -2, -1), c["q"])
        ttnn.deallocate(grad_scores)
        ttnn.deallocate(grad_q)
        ttnn.deallocate(grad_k)
        grad_x = ttnn.matmul(grad_out_gated, w_qkv)
        ttnn.deallocate(grad_out_gated)
        return grad_x

    # Warmup
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
    rss_start = rss_mb()
    print(f"\n{label}: start RSS={rss_start:.1f} MB", flush=True)

    for i in range(n_iters):
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
    rss_end = rss_mb()
    delta = rss_end - rss_start
    print(f"{label}: end RSS={rss_end:.1f} MB  delta={delta:+.1f} MB  rate={delta/n_iters:.3f} MB/it", flush=True)
    return rss_start, rss_end


# --- Phase 1: run on device 0, measure leak ---
device = ttnn.open_device(device_id=0)

w_qkv = ttnn.from_torch(
    torch.randn(D, 4 * D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w_out = ttnn.from_torch(
    torch.randn(D, D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w_out_t = ttnn.transpose(w_out, 0, 1)

rss1_start, rss1_end = run_phase(device, w_qkv, w_out, w_out_t, ITERS, "Phase 1 (before close/reopen)")

# --- Close device, trim, reopen ---
ttnn.deallocate(w_qkv)
ttnn.deallocate(w_out)
ttnn.deallocate(w_out_t)
ttnn.close_device(device)
gc_trim()
rss_after_close = rss_mb()
print(f"\nAfter close_device + gc_trim: RSS={rss_after_close:.1f} MB (was {rss1_end:.1f} MB before close)", flush=True)

# Reopen
device = ttnn.open_device(device_id=0)
w_qkv = ttnn.from_torch(
    torch.randn(D, 4 * D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w_out = ttnn.from_torch(
    torch.randn(D, D, dtype=torch.bfloat16) * 0.02,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w_out_t = ttnn.transpose(w_out, 0, 1)

rss_after_reopen = rss_mb()
print(f"After reopen_device: RSS={rss_after_reopen:.1f} MB", flush=True)

# --- Phase 2: run again, measure leak ---
rss2_start, rss2_end = run_phase(device, w_qkv, w_out, w_out_t, ITERS, "Phase 2 (after close/reopen)")

# --- Summary ---
print(f"\n=== SUMMARY ===")
print(f"Phase 1: {rss1_start:.1f} -> {rss1_end:.1f} MB  (+{rss1_end - rss1_start:.1f} MB)")
print(f"After close: {rss_after_close:.1f} MB  (reclaimed {rss1_end - rss_after_close:.1f} MB)")
print(f"After reopen: {rss_after_reopen:.1f} MB")
print(f"Phase 2: {rss2_start:.1f} -> {rss2_end:.1f} MB  (+{rss2_end - rss2_start:.1f} MB)")

reclaimed = rss1_end - rss_after_close
if reclaimed > 1.0:
    print(f"\nWORKAROUND CONFIRMED: close_device reclaims {reclaimed:.1f} MB of leaked C++ memory.")
    print(f"Periodic close/reopen can reset the leak without restarting the process.")
else:
    print(f"\nWORKAROUND FAILED: close_device only reclaimed {reclaimed:.1f} MB.")

ttnn.deallocate(w_qkv)
ttnn.deallocate(w_out)
ttnn.deallocate(w_out_t)
ttnn.close_device(device)
