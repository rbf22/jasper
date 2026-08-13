"""Check if the program cache is growing per iteration."""
import gc, ctypes, sys, os, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024

device = ttnn.open_device(device_id=0)

with open("configs/cell_a_tt.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["micro_batch_size"] = 0
model_config = build_model_config(cfg)
model = TTWRAPModel(model_config, device)

B, T, V = 8, 128, 128

# Warmup
for _ in range(3):
    input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    labels = torch.randint(0, V, (B, T), dtype=torch.int32)
    logits = model.forward(input_ids, k_value=None)
    loss_val, grad_logits = cross_entropy_loss(logits, labels)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)
    _safe_deallocate(grad_logits)
    for g in grads.values():
        _safe_deallocate(g)
    model.clear_caches()

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)

# Check if program cache is enabled
try:
    print(f"Program cache enabled: {ttnn._ttnn.program_cache.is_enabled()}")
except:
    print("Program cache API not accessible from Python")

rss0 = rss_mb()
print(f"Baseline RSS: {rss0:.0f} MB")

# Also track via /proc/self/maps to see if new mappings appear
def maps_size_kb():
    try:
        with open("/proc/self/maps") as f:
            total = 0
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    addr_range = parts[0]
                    start, end = addr_range.split("-")
                    total += (int(end, 16) - int(start, 16)) // 1024
            return total
    except:
        return 0

maps0 = maps_size_kb()
print(f"Baseline maps total: {maps0} KB")

for i in range(200):
    input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    labels = torch.randint(0, V, (B, T), dtype=torch.int32)
    logits = model.forward(input_ids, k_value=None)
    loss_val, grad_logits = cross_entropy_loss(logits, labels)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)
    _safe_deallocate(grad_logits)
    for g in grads.values():
        _safe_deallocate(g)
    model.clear_caches()

    if i in {0, 9, 49, 99, 199}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_mb()
        maps = maps_size_kb()
        print(f"  iter {i:3d}: RSS={rss:.0f}MB (+{rss-rss0:.0f})  "
              f"maps={maps}KB (+{maps-maps0})")

ttnn.close_device(device)
