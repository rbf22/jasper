"""Find which specific memory mapping is growing."""
import gc, ctypes, sys, os, torch, ttnn, re

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

def parse_smaps():
    """Parse /proc/self/smaps and return per-mapping RSS."""
    mappings = []
    with open("/proc/self/smaps") as f:
        current = {}
        for line in f:
            line = line.strip()
            # Header line: start-end perms offset dev inode path
            m = re.match(r'^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)', line)
            if m:
                if current:
                    mappings.append(current)
                current = {
                    'start': m.group(1), 'end': m.group(2),
                    'perms': m.group(3), 'path': m.group(7).strip(),
                    'Rss': 0, 'Pss': 0, 'Private_Dirty': 0, 'Anonymous': 0,
                    'Size': int(m.group(2), 16) - int(m.group(1), 16)
                }
            elif ':' in line:
                parts = line.split(':')
                key = parts[0].strip()
                val = parts[1].strip().split()[0] if len(parts) > 1 and parts[1].strip() else '0'
                if key in ('Rss', 'Pss', 'Private_Dirty', 'Anonymous'):
                    try:
                        current[key] = int(val)
                    except:
                        pass
        if current:
            mappings.append(current)
    return mappings

def summarize_mappings(mappings):
    """Group by path and sum RSS."""
    from collections import defaultdict
    groups = defaultdict(lambda: {'Rss': 0, 'Pss': 0, 'Private_Dirty': 0, 'Anonymous': 0, 'count': 0})
    for m in mappings:
        path = m['path'] if m['path'] else '(anonymous)'
        # Truncate device paths
        if '/dev/tenstorrent' in path:
            path = '/dev/tenstorrent/*'
        groups[path]['Rss'] += m['Rss']
        groups[path]['Pss'] += m['Pss']
        groups[path]['Private_Dirty'] += m['Private_Dirty']
        groups[path]['Anonymous'] += m['Anonymous']
        groups[path]['count'] += 1
    return groups

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
mappings0 = parse_smaps()
groups0 = summarize_mappings(mappings0)
print(f"Baseline: RSS={rss0//1024}MB")
print("  Top mappings by RSS:")
for path, stats in sorted(groups0.items(), key=lambda x: -x[1]['Rss'])[:15]:
    print(f"    RSS={stats['Rss']//1024:4d}MB  Pss={stats['Pss']//1024:4d}MB  "
          f"PrivDirty={stats['Private_Dirty']//1024:4d}MB  Anon={stats['Anonymous']//1024:4d}MB  "
          f"count={stats['count']}  {path[:50]}")

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

    if i in {49, 199}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        mappings = parse_smaps()
        groups = summarize_mappings(mappings)
        print(f"\n  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")
        # Show only mappings that grew
        grew = []
        for path, stats in groups.items():
            prev = groups0.get(path, {'Rss': 0, 'Pss': 0, 'Private_Dirty': 0, 'Anonymous': 0, 'count': 0})
            delta_rss = stats['Rss'] - prev['Rss']
            if delta_rss > 100:  # >100KB
                grew.append((path, stats, prev, delta_rss))
        grew.sort(key=lambda x: -x[3])
        for path, stats, prev, delta_rss in grew[:15]:
            print(f"    ΔRSS={delta_rss//1024:+4d}MB  "
                  f"RSS={stats['Rss']//1024:4d}MB  PrivDirty={stats['Private_Dirty']//1024:4d}MB  "
                  f"Anon={stats['Anonymous']//1024:4d}MB  count={stats['count']}  {path[:50]}")

ttnn.close_device(device)
