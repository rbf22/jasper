"""Find the specific anonymous mapping that's growing."""
import gc, ctypes, sys, os, torch, ttnn, re

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

def parse_smaps_anon():
    """Parse /proc/self/smaps and return anonymous mappings with their addresses."""
    mappings = []
    with open("/proc/self/smaps") as f:
        current = {}
        for line in f:
            line = line.strip()
            m = re.match(r'^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)', line)
            if m:
                if current:
                    mappings.append(current)
                path = m.group(7).strip()
                current = {
                    'addr': m.group(1) + '-' + m.group(2),
                    'start': int(m.group(1), 16),
                    'size': int(m.group(2), 16) - int(m.group(1), 16),
                    'perms': m.group(3),
                    'path': path,
                    'Rss': 0, 'Pss': 0, 'Private_Dirty': 0, 'Anonymous': 0,
                }
            elif ':' in line and current:
                parts = line.split(':')
                key = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    val = parts[1].strip().split()[0]
                    if key in ('Rss', 'Pss', 'Private_Dirty', 'Anonymous'):
                        try:
                            current[key] = int(val)
                        except:
                            pass
        if current:
            mappings.append(current)
    # Filter to anonymous only (no path or [heap])
    return [m for m in mappings if not m['path'] or m['path'] == '[anon_hugepage (deleted)]']

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

rss0 = rss_kb()
anon0 = parse_smaps_anon()
# Sort by size descending
anon0_sorted = sorted(anon0, key=lambda m: -m['size'])
print(f"Baseline: RSS={rss0//1024}MB  anon_count={len(anon0)}")
print("  Top 20 anonymous mappings by size:")
for m in anon0_sorted[:20]:
    print(f"    {m['size']//1024//1024:6d}MB  RSS={m['Rss']//1024:4d}MB  "
          f"PrivDirty={m['Private_Dirty']//1024:4d}MB  "
          f"perms={m['perms']}  {m['addr']}  {m['path'][:30]}")

# Also sort by RSS
anon0_by_rss = sorted(anon0, key=lambda m: -m['Rss'])
print("\n  Top 20 anonymous mappings by RSS:")
for m in anon0_by_rss[:20]:
    print(f"    {m['size']//1024//1024:6d}MB  RSS={m['Rss']//1024:4d}MB  "
          f"PrivDirty={m['Private_Dirty']//1024:4d}MB  "
          f"perms={m['perms']}  {m['addr']}  {m['path'][:30]}")

# Build a map by start address for comparison
anon0_map = {m['start']: m for m in anon0}

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

    if i == 199:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        anon = parse_smaps_anon()
        anon_map = {m['start']: m for m in anon}
        print(f"\n  iter 199: RSS={rss//1024}MB (+{(rss-rss0)//1024})  anon_count={len(anon)}")

        # Find mappings that grew
        grew = []
        for start, m in anon_map.items():
            prev = anon0_map.get(start)
            if prev:
                delta = m['Rss'] - prev['Rss']
                if delta > 100:  # >100KB
                    grew.append((delta, m, prev))
            else:
                # New mapping
                grew.append((m['Rss'], m, None))
        grew.sort(key=lambda x: -x[0])
        print("  Mappings that grew (by RSS delta):")
        for delta, m, prev in grew[:20]:
            prev_rss = prev['Rss'] if prev else 0
            print(f"    ΔRSS={delta//1024:+5d}MB  RSS={m['Rss']//1024:4d}MB (was {prev_rss//1024}MB)  "
                  f"size={m['size']//1024//1024:4d}MB  "
                  f"PrivDirty={m['Private_Dirty']//1024:4d}MB  "
                  f"perms={m['perms']}  {m['addr']}  {m['path'][:30]}")

ttnn.close_device(device)
