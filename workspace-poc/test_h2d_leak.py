"""Test if ttnn.from_torch leaks in a loop."""
import gc, ctypes, sys, torch, ttnn

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

device = ttnn.open_device(device_id=0)

# Test 1: ttnn.from_torch in a loop (host-to-device transfer)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"Test 1 (ttnn.from_torch loop): Baseline RSS={rss0//1024}MB")
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

# Test 2: ttnn.from_torch + ttnn.to_torch (round-trip)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 2 (from_torch + to_torch loop): Baseline RSS={rss0//1024}MB")
for i in range(500):
    t = torch.randn(8, 128, 384, dtype=torch.bfloat16)
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.synchronize_device(device)
    t2 = ttnn.to_torch(tt)
    ttnn.deallocate(tt, force=True)
    del t2
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

# Test 3: Just synchronize_device in a loop (no ops)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 3 (synchronize_device only): Baseline RSS={rss0//1024}MB")
for i in range(500):
    ttnn.synchronize_device(device)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

# Test 4: Embedding lookup in a loop
embedding_weight = ttnn.from_torch(
    torch.eye(128, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest 4 (embedding loop): Baseline RSS={rss0//1024}MB")
for i in range(500):
    ids = torch.randint(0, 128, (8, 128), dtype=torch.int32)
    ids_tt = ttnn.from_torch(ids.unsqueeze(-1), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.embedding(ids_tt, embedding_weight, layout=ttnn.TILE_LAYOUT)
    ttnn.synchronize_device(device)
    ttnn.deallocate(ids_tt, force=True)
    ttnn.deallocate(out, force=True)
    if i in {0, 99, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.deallocate(embedding_weight, force=True)
ttnn.close_device(device)
