"""Find which mappings grow over 1000 iterations."""
import gc, ctypes, sys, os, torch, ttnn, re

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

def parse_smaps():
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
                    'start': int(m.group(1), 16),
                    'size': int(m.group(2), 16) - int(m.group(1), 16),
                    'perms': m.group(3),
                    'path': path,
                    'Rss': 0, 'Private_Dirty': 0, 'Anonymous': 0,
                }
            elif ':' in line and current:
                parts = line.split(':')
                key = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    val = parts[1].strip().split()[0]
                    if key in ('Rss', 'Private_Dirty', 'Anonymous'):
                        try:
                            current[key] = int(val)
                        except:
                            pass
        if current:
            mappings.append(current)
    return mappings

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
mappings0 = {m['start']: m for m in parse_smaps()}
print(f"Baseline: RSS={rss0//1024}MB  total_mappings={len(mappings0)}")

for i in range(1000):
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

    if i == 999:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        mappings = {m['start']: m for m in parse_smaps()}
        print(f"\n  iter 999: RSS={rss//1024}MB (+{(rss-rss0)//1024})  total_mappings={len(mappings)}")

        # Find all mappings that grew
        grew = []
        new_count = 0
        for start, m in mappings.items():
            prev = mappings0.get(start)
            if prev:
                delta = m['Rss'] - prev['Rss']
                if delta > 100:
                    grew.append((delta, m, prev))
            else:
                new_count += 1
                if m['Rss'] > 100:
                    grew.append((m['Rss'], m, None))

        grew.sort(key=lambda x: -x[0])
        print(f"  New mappings: {new_count}")
        print(f"  Mappings that grew (by RSS delta):")
        total_delta = 0
        for delta, m, prev in grew[:30]:
            prev_rss = prev['Rss'] if prev else 0
            total_delta += delta
            print(f"    ΔRSS={delta//1024:+5d}MB  RSS={m['Rss']//1024:4d}MB (was {prev_rss//1024}MB)  "
                  f"size={m['size']//1024//1024:4d}MB  "
                  f"perms={m['perms']}  path={m['path'][:40]}")
        print(f"  Total delta from grew list: {total_delta//1024}MB")

ttnn.close_device(device)
