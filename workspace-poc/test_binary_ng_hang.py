"""Focused hang repro: 2 retention-like layers + reshape.

This tests the exact pattern that caused the device hang with the partial
buffer binding patch (only c_buffer registered):

  1. Layer 1 runs binary_ng ops (ttnn.mul) → program descriptors cached
  2. Layer 1's intermediate tensors are deallocated
  3. Layer 2 runs the same binary_ng ops → program cache HIT
  4. Fast path patches only registered buffer positions
  5. If a_buffer/b_buffer were NOT registered, reader keeps stale addresses
  6. Reader kernel reads from deallocated device memory → corruption → hang

With the complete fix (all buffer positions registered), all addresses
are patched on cache hit → no hang.

Usage:
  TT_VISIBLE_DEVICES=0 TT_METAL_HOME=/home/rfenwick/Documents/tt-metal-src \
  .tt-venv/bin/python -u test_binary_ng_hang.py
"""
import sys
import time
import torch
import ttnn

TT_METAL_HOME = "/home/rfenwick/Documents/tt-metal-src"


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024


def main():
    device = ttnn.open_device(device_id=0)

    D = 384
    B, T = 4, 128
    H = 4
    d_h = D // H

    # Persistent weights for 2 layers
    w1 = ttnn.from_torch(
        torch.randn(D, 4 * D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    w1_out = ttnn.from_torch(
        torch.randn(D, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    w2 = ttnn.from_torch(
        torch.randn(D, 4 * D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    w2_out = ttnn.from_torch(
        torch.randn(D, D, dtype=torch.bfloat16) * 0.02,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )

    def retention_layer_forward(x, w_qkv, w_out):
        """Single retention-like layer with binary_ng ops + reshape."""
        y = ttnn.linear(x, w_qkv)                       # (B, T, 4D)
        y4 = ttnn.reshape(y, [B, T, H, 4 * d_h])        # reshape (view)
        q = ttnn.slice(y4, [0, 0, 0, 0], [B, T, H, d_h])
        k = ttnn.slice(y4, [0, 0, 0, d_h], [B, T, H, 2 * d_h])
        v = ttnn.slice(y4, [0, 0, 0, 2 * d_h], [B, T, H, 3 * d_h])
        g = ttnn.slice(y4, [0, 0, 0, 3 * d_h], [B, T, H, 4 * d_h])
        q = ttnn.permute(q, [0, 2, 1, 3])  # (B, H, T, d_h)
        k = ttnn.permute(k, [0, 2, 1, 3])
        v = ttnn.permute(v, [0, 2, 1, 3])
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        out = ttnn.matmul(scores, v)                    # (B, H, T, d_h)
        out = ttnn.permute(out, [0, 2, 1, 3])           # (B, T, H, d_h)
        out = ttnn.reshape(out, [B, T, D])              # reshape (view)
        g_flat = ttnn.reshape(g, [B, T, D])
        g_sig = ttnn.sigmoid(g_flat)                    # unary op
        out_gated = ttnn.mul(out, g_sig)                # binary_ng (MUL) — key op
        result = ttnn.linear(out_gated, w_out)

        # Deallocate intermediates — this is what triggers stale buffers
        # on cache hit if buffer bindings are incomplete.
        ttnn.deallocate(y)
        ttnn.deallocate(y4)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(g)
        ttnn.deallocate(scores)
        ttnn.deallocate(out)
        ttnn.deallocate(g_flat)
        ttnn.deallocate(g_sig)
        ttnn.deallocate(out_gated)
        return result

    ITERS = 50
    print(f"2-layer retention + reshape hang repro ({ITERS} iterations)")
    print(f"{'Iter':>6}  {'Time':>8}  {'RSS':>10}  Status")
    print("-" * 50)

    rss0 = rss_mb()
    for i in range(ITERS):
        t0 = time.time()
        x = ttnn.from_torch(
            torch.randn(B, T, D, dtype=torch.bfloat16) * 0.02,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        # Layer 1 — first call: cache miss, programs compiled
        h1 = retention_layer_forward(x, w1, w1_out)
        ttnn.synchronize_device(device)
        ttnn.deallocate(x)

        # Layer 2 — same op shapes: cache HIT, fast path patches buffers
        # This is where the hang occurred with incomplete buffer bindings.
        h2 = retention_layer_forward(h1, w2, w2_out)
        ttnn.synchronize_device(device)
        ttnn.deallocate(h1)
        ttnn.deallocate(h2)

        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            rss = rss_mb()
            print(f"{i+1:6d}  {elapsed:6.3f}s  {rss:8.1f} MB  OK")

    rss_end = rss_mb()
    print(f"\nSUCCESS: Completed {ITERS} iterations without hanging.")
    print(f"RSS: {rss0:.1f} → {rss_end:.1f} MB (delta: {rss_end - rss0:+.1f} MB)")

    ttnn.deallocate(w1)
    ttnn.deallocate(w1_out)
    ttnn.deallocate(w2)
    ttnn.deallocate(w2_out)
    ttnn.close_device(device)
    print("Device closed cleanly.")


if __name__ == "__main__":
    main()
