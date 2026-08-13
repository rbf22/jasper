"""Test _fused_scale_decay in a loop directly."""
import gc, ctypes, sys, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTRetentionLayer

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

device = ttnn.open_device(device_id=0)

B, H, T = 8, 4, 128
BH = B * H

# Pre-allocate input tensors (stable shapes, stable buffers)
scores_raw = ttnn.from_torch(torch.randn(BH * T, T, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
D_decay = ttnn.from_torch(torch.randn(H * T, T, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
scale = 0.125

# Warmup
for _ in range(3):
    out = TTRetentionLayer._fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out, force=True)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"Test (_fused_scale_decay loop, stable tensors): Baseline RSS={rss0//1024}MB")

for i in range(1000):
    out = TTRetentionLayer._fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out, force=True)
    if i in {0, 99, 199, 499, 999}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        delta = rss - rss0
        rate = delta / (i + 1)
        print(f"  iter {i:4d}: RSS={rss//1024}MB  delta={delta//1024:+d}MB  rate={rate/1024:.3f} MB/iter")

# Now test with FRESH output tensor every call (like the model does)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest (_fused_scale_decay loop, fresh output tensor via ttnn.empty): Baseline RSS={rss0//1024}MB")

for i in range(1000):
    # The function internally calls ttnn.empty, so out is always a fresh tensor
    out = TTRetentionLayer._fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out, force=True)
    if i in {0, 99, 199, 499, 999}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        delta = rss - rss0
        rate = delta / (i + 1)
        print(f"  iter {i:4d}: RSS={rss//1024}MB  delta={delta//1024:+d}MB  rate={rate/1024:.3f} MB/iter")

ttnn.deallocate(scores_raw, force=True)
ttnn.deallocate(D_decay, force=True)
ttnn.close_device(device)
