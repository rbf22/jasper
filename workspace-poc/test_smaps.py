"""Track smaps_rollup to find what's growing."""
import gc, ctypes, sys, os, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024

def smaps_rollup():
    try:
        with open("/proc/self/smaps_rollup") as f:
            data = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    data[parts[0].rstrip(':')] = int(parts[1])
            return data
    except:
        return {}

def top_mappings(n=10):
    """Find the largest memory mappings."""
    try:
        regions = []
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                addr_range = parts[0]
                start, end = addr_range.split("-")
                size = int(end, 16) - int(start, 16)
                desc = parts[5] if len(parts) >= 6 else ""
                regions.append((size, desc, parts[1]))
        regions.sort(reverse=True)
        return regions[:n]
    except:
        return []

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

rss0 = rss_mb()
sr0 = smaps_rollup()
print(f"Baseline: RSS={rss0:.0f}MB")
if sr0:
    print(f"  smaps: Pss={sr0.get('Pss',0)//1024}KB Rss={sr0.get('Rss',0)//1024}KB "
          f"Private_Clean={sr0.get('Private_Clean',0)//1024}KB "
          f"Private_Dirty={sr0.get('Private_Dirty',0)//1024}KB "
          f"Shared_Clean={sr0.get('Shared_Clean',0)//1024}KB "
          f"Shared_Dirty={sr0.get('Shared_Dirty',0)//1024}KB "
          f"Anonymous={sr0.get('Anonymous',0)//1024}KB")

print("  Top 10 mappings:")
for size, desc, perms in top_mappings(10):
    print(f"    {size//1024//1024:4d}MB  {perms}  {desc[:60]}")

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

    if i in {0, 49, 99, 199}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_mb()
        sr = smaps_rollup()
        print(f"\n  iter {i:3d}: RSS={rss:.0f}MB (+{rss-rss0:.0f})")
        if sr:
            print(f"    smaps: Pss={sr.get('Pss',0)//1024}KB (+{(sr.get('Pss',0)-sr0.get('Pss',0))//1024}) "
                  f"Rss={sr.get('Rss',0)//1024}KB (+{(sr.get('Rss',0)-sr0.get('Rss',0))//1024}) "
                  f"Private_Dirty={sr.get('Private_Dirty',0)//1024}KB (+{(sr.get('Private_Dirty',0)-sr0.get('Private_Dirty',0))//1024}) "
                  f"Anonymous={sr.get('Anonymous',0)//1024}KB (+{(sr.get('Anonymous',0)-sr0.get('Anonymous',0))//1024})")

ttnn.close_device(device)
