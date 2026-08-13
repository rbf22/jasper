"""Measure heap growth from just tensor create/destroy, no forward/backward."""
import gc, ctypes, sys, torch, ttnn

libc = ctypes.CDLL("libc.so.6")

class mallinfo2_t(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]

libc.mallinfo2.restype = mallinfo2_t

def heap_used():
    return libc.mallinfo2().uordblks

device = ttnn.open_device(device_id=0)

B, T, D = 8, 128, 256

# Warmup
for _ in range(5):
    t = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(t, force=True)
    del t

gc.collect()
libc.malloc_trim(0)
used0 = heap_used()
print(f"Baseline (tensor create/destroy only): used={used0//1024}KB")

for i in range(500):
    # Simulate tensor creation/destruction like the test does
    t1 = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t2 = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(t1, force=True)
    ttnn.deallocate(t2, force=True)
    del t1, t2

    if i in {0, 99, 199, 299, 399, 499}:
        gc.collect()
        libc.malloc_trim(0)
        used = heap_used()
        delta = used - used0
        print(f"  iter {i:3d}: used={used//1024}KB  delta={delta//1024:+d}KB  rate={delta/(i+1)/1024:.3f} KB/iter")

ttnn.close_device(device)
