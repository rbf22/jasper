"""Test if creating and deallocating device tensors leaks."""
import gc, ctypes, sys, torch, ttnn

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

device = ttnn.open_device(device_id=0)

# Test 1: Create + dealloc device tensor in a loop
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"Test 1 (create+dealloc device tensor): Baseline RSS={rss0//1024}MB")
for i in range(500):
    t = torch.randn(8, 128, 384, dtype=torch.bfloat16)
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(tt, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

# Test 2: Create many tensors, dealloc all at once (like model forward)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 2 (create 20 + dealloc all): Baseline RSS={rss0//1024}MB")
for i in range(500):
    tensors = []
    for j in range(20):
        t = torch.randn(8, 128, 384, dtype=torch.bfloat16)
        tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tensors.append(tt)
    ttnn.synchronize_device(device)
    for tt in tensors:
        ttnn.deallocate(tt, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

# Test 3: Chain of ops with create+dealloc (like model forward)
a = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(8, 128, 384, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 3 (chain of ops with dealloc): Baseline RSS={rss0//1024}MB")
for i in range(500):
    # Chain: mul -> add -> silu -> mul -> add
    c1 = ttnn.mul(a, b)
    c2 = ttnn.add(c1, a)
    ttnn.deallocate(c1, force=True)
    c3 = ttnn.silu(c2)
    ttnn.deallocate(c2, force=True)
    c4 = ttnn.mul(c3, b)
    ttnn.deallocate(c3, force=True)
    c5 = ttnn.add(c4, a)
    ttnn.deallocate(c4, force=True)
    ttnn.synchronize_device(device)
    ttnn.deallocate(c5, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.deallocate(a, force=True)
ttnn.deallocate(b, force=True)

# Test 4: ttnn.zeros create+dealloc (like cross_entropy_loss does)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 4 (ttnn.zeros create+dealloc): Baseline RSS={rss0//1024}MB")
for i in range(500):
    z = ttnn.zeros((8, 1, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(z, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.close_device(device)
