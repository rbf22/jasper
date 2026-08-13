"""Measure heap growth per major step in the training iteration."""
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

x = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Warmup
for _ in range(5):
    out = layer.forward(x)
    ttnn.synchronize_device(device)
    grad_out = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_x, grads = layer.backward(grad_out)
    ttnn.synchronize_device(device)
    _safe_deallocate(out); _safe_deallocate(grad_out); _safe_deallocate(grad_x)
    for g in grads.values(): _safe_deallocate(g)

gc.collect(); libc.malloc_trim(0)

# Track per-step leaks
step_names = ["forward", "sync1", "from_torch", "backward", "sync2", "dealloc"]
step_totals = {n: 0 for n in step_names}
step_counts = {n: 0 for n in step_names}

for i in range(200):
    h0 = heap_used()
    out = layer.forward(x)
    h1 = heap_used()
    ttnn.synchronize_device(device)
    h2 = heap_used()
    grad_out = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    h3 = heap_used()
    grad_x, grads = layer.backward(grad_out)
    h4 = heap_used()
    ttnn.synchronize_device(device)
    h5 = heap_used()
    _safe_deallocate(out); _safe_deallocate(grad_out); _safe_deallocate(grad_x)
    for g in grads.values(): _safe_deallocate(g)
    h6 = heap_used()

    deltas = [h1-h0, h2-h1, h3-h2, h4-h3, h5-h4, h6-h5]
    for name, delta in zip(step_names, deltas):
        if delta > 0:
            step_totals[name] += delta
            step_counts[name] += 1

    if i in {0, 49, 99, 199}:
        gc.collect(); libc.malloc_trim(0)
        total_delta = heap_used() - h0
        print(f"  iter {i:3d}: total_delta={total_delta//1024:+d}KB")
        for name in step_names:
            if step_counts[name] > 0:
                print(f"    {name:10s}: total={step_totals[name]//1024}KB count={step_counts[name]} avg={step_totals[name]/step_counts[name]:.0f} bytes")

print("\n=== Final per-step summary (200 iters) ===")
for name in step_names:
    if step_counts[name] > 0:
        print(f"  {name:10s}: total={step_totals[name]//1024}KB count={step_counts[name]} avg={step_totals[name]/step_counts[name]:.0f} bytes")
total_all = sum(step_totals.values())
print(f"  {'TOTAL':10s}: {total_all//1024}KB ({total_all/200/1024:.1f} KB/iter)")

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
