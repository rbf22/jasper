"""Isolate the leak to specific operations."""
import gc, ctypes, sys, torch, ttnn

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

device = ttnn.open_device(device_id=0)

# Test 1: Just ttnn.mul in a loop (binary_ng, the op we fixed)
a = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Warmup
for _ in range(3):
    c = ttnn.mul(a, b)
    ttnn.synchronize_device(device)
    ttnn.deallocate(c, force=True)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"Test 1 (ttnn.mul loop): Baseline RSS={rss0//1024}MB")

for i in range(500):
    c = ttnn.mul(a, b)
    ttnn.synchronize_device(device)
    ttnn.deallocate(c, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.deallocate(a, force=True)
ttnn.deallocate(b, force=True)

# Test 2: ttnn.add (another binary_ng)
a = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 2 (ttnn.add loop): Baseline RSS={rss0//1024}MB")

for i in range(500):
    c = ttnn.add(a, b)
    ttnn.synchronize_device(device)
    ttnn.deallocate(c, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.deallocate(a, force=True)
ttnn.deallocate(b, force=True)

# Test 3: ttnn.softmax
a = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 3 (ttnn.softmax loop): Baseline RSS={rss0//1024}MB")

for i in range(500):
    c = ttnn.softmax(a, dim=-1)
    ttnn.synchronize_device(device)
    ttnn.deallocate(c, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.deallocate(a, force=True)

# Test 4: ttnn.matmul
a = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(8, 384, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 4 (ttnn.matmul loop): Baseline RSS={rss0//1024}MB")

for i in range(500):
    c = ttnn.matmul(a, b)
    ttnn.synchronize_device(device)
    ttnn.deallocate(c, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.deallocate(a, force=True)
ttnn.deallocate(b, force=True)

# Test 5: ttnn.silu (unary)
a = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 5 (ttnn.silu loop): Baseline RSS={rss0//1024}MB")

for i in range(500):
    c = ttnn.silu(a)
    ttnn.synchronize_device(device)
    ttnn.deallocate(c, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.deallocate(a, force=True)

ttnn.close_device(device)
