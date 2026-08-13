"""Measure heap growth around specific Python-level operations to find the 64KB/iter leak outside launch."""
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

# Measure individual phases
phases = {"from_torch": 0, "forward": 0, "sync1": 0, "backward": 0, "sync2": 0, "dealloc": 0, "gc_trim": 0}
counts = {k: 0 for k in phases}

for i in range(50):
    # Phase 1: from_torch (grad_out creation)
    h0 = heap_used()
    grad_out = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    h1 = heap_used()
    phases["from_torch"] += h1 - h0; counts["from_torch"] += 1

    # Phase 2: forward
    h0 = heap_used()
    out = layer.forward(x)
    h1 = heap_used()
    phases["forward"] += h1 - h0; counts["forward"] += 1

    # Phase 3: sync
    h0 = heap_used()
    ttnn.synchronize_device(device)
    h1 = heap_used()
    phases["sync1"] += h1 - h0; counts["sync1"] += 1

    # Phase 4: backward
    h0 = heap_used()
    grad_x, grads = layer.backward(grad_out)
    h1 = heap_used()
    phases["backward"] += h1 - h0; counts["backward"] += 1

    # Phase 5: sync
    h0 = heap_used()
    ttnn.synchronize_device(device)
    h1 = heap_used()
    phases["sync2"] += h1 - h0; counts["sync2"] += 1

    # Phase 6: dealloc
    h0 = heap_used()
    _safe_deallocate(out); _safe_deallocate(grad_out); _safe_deallocate(grad_x)
    for g in grads.values(): _safe_deallocate(g)
    h1 = heap_used()
    phases["dealloc"] += h1 - h0; counts["dealloc"] += 1

    # Phase 7: gc + trim
    h0 = heap_used()
    gc.collect(); libc.malloc_trim(0)
    h1 = heap_used()
    phases["gc_trim"] += h1 - h0; counts["gc_trim"] += 1

print("=== Per-phase heap delta (50 iterations) ===")
total = 0
for phase in ["from_torch", "forward", "sync1", "backward", "sync2", "dealloc", "gc_trim"]:
    avg = phases[phase] / counts[phase]
    per_iter = phases[phase] / 50
    print(f"  {phase:12s}: total={phases[phase]:>10d}  avg={avg:>8.0f}  per_iter={per_iter/1024:>8.2f}KB")
    total += phases[phase]
print(f"  {'TOTAL':12s}: total={total:>10d}  per_iter={total/50/1024:>8.2f}KB")

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
