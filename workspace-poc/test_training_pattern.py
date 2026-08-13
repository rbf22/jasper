"""Simulate the training loop pattern: fresh x each iteration, use clear_caches."""
import gc, ctypes, sys, torch, ttnn

libc = ctypes.CDLL("libc.so.6")

class mallinfo2_t(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t), ("ordblks", ctypes.c_size_t), ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t), ("hblkhd", ctypes.c_size_t), ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t), ("uordblks", ctypes.c_size_t), ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]

libc.mallinfo2.restype = mallinfo2_t

def heap_used():
    return libc.mallinfo2().uordblks

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTRetentionLayer, _safe_deallocate
from train_ttnn import build_model_config
import yaml

device = ttnn.open_device(device_id=0)

with open("configs/cell_a_tt.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["micro_batch_size"] = 0
model_config = build_model_config(cfg)

layer = TTRetentionLayer(model_config, device)

B, T, D = 8, 128, model_config.d_model

# Warmup with fresh x each time (like training loop)
for _ in range(5):
    x = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = layer.forward(x)
    ttnn.synchronize_device(device)
    grad_out = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_x, grads = layer.backward(grad_out)
    ttnn.synchronize_device(device)
    _safe_deallocate(out); _safe_deallocate(grad_out); _safe_deallocate(grad_x)
    for g in grads.values(): _safe_deallocate(g)
    # Use clear_caches pattern (sync + dealloc)
    ttnn.synchronize_device(device)
    layer._deallocate_cache()
    layer._deallocate_cache_history()
    _safe_deallocate(x)

gc.collect(); libc.malloc_trim(0)
used0 = heap_used()
print(f"Baseline: used={used0//1024}KB")

# Test: training loop pattern with _deallocate_cache
for i in range(200):
    x = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = layer.forward(x)
    ttnn.synchronize_device(device)
    grad_out = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_x, grads = layer.backward(grad_out)
    ttnn.synchronize_device(device)
    _safe_deallocate(out); _safe_deallocate(grad_out); _safe_deallocate(grad_x)
    for g in grads.values(): _safe_deallocate(g)
    # Clear caches (training loop pattern)
    ttnn.synchronize_device(device)
    layer._deallocate_cache()
    layer._deallocate_cache_history()
    _safe_deallocate(x)
    gc.collect(); libc.malloc_trim(0)
    if i in {0, 9, 49, 99, 199}:
        used = heap_used()
        delta = used - used0
        print(f"  [training pattern] iter {i:3d}: delta={delta//1024:+d}KB  rate={delta/(i+1)/1024:.3f} KB/iter")

ttnn.close_device(device)
