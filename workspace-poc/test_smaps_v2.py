"""Track which mappings grow during retention layer execution - look at ALL mappings."""
import gc, ctypes, sys, os, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTRetentionLayer, _safe_deallocate
from train_ttnn import build_model_config
import yaml

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

def dump_all_mappings():
    """Return dict of {start_addr: (size_kb, rss_kb, perms, path)} for all mappings."""
    import re
    maps = {}
    pid = os.getpid()
    with open(f"/proc/{pid}/smaps") as f:
        content = f.read()
    blocks = re.split(r"\n(?=[0-9a-f]+-[0-9a-f]+ )", content)
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        header = lines[0]
        parts = header.split(None, 5)
        if len(parts) < 5:
            continue
        addr_range = parts[0]
        perms = parts[1]
        path = parts[5] if len(parts) >= 6 else ""
        start_addr = addr_range.split("-")[0]
        size_kb = 0
        rss_kb = 0
        for line in lines[1:]:
            if line.startswith("Size:"):
                size_kb = int(line.split()[1])
            elif line.startswith("Rss:"):
                rss_kb = int(line.split()[1])
        if size_kb > 100:
            maps[start_addr] = (size_kb, rss_kb, perms, path)
    return maps

device = ttnn.open_device(device_id=0)

with open("configs/cell_a_tt.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["micro_batch_size"] = 0
model_config = build_model_config(cfg)

layer = TTRetentionLayer(model_config, device)
B, T, D = 8, 128, model_config.d_model
x = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Warmup
for _ in range(3):
    out = layer.forward(x)
    ttnn.synchronize_device(device)
    grad_out = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_x, grads = layer.backward(grad_out)
    ttnn.synchronize_device(device)
    _safe_deallocate(out)
    _safe_deallocate(grad_out)
    _safe_deallocate(grad_x)
    for g in grads.values():
        _safe_deallocate(g)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
maps0 = dump_all_mappings()
rss0 = rss_kb()
total_rss0 = sum(v[1] for v in maps0.values())
print(f"Baseline RSS={rss0//1024}MB, {len(maps0)} mappings, sum_rss={total_rss0//1024}MB")

for i in range(200):
    out = layer.forward(x)
    ttnn.synchronize_device(device)
    grad_out = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_x, grads = layer.backward(grad_out)
    ttnn.synchronize_device(device)
    _safe_deallocate(out)
    _safe_deallocate(grad_out)
    _safe_deallocate(grad_x)
    for g in grads.values():
        _safe_deallocate(g)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
maps1 = dump_all_mappings()
rss1 = rss_kb()
total_rss1 = sum(v[1] for v in maps1.values())
print(f"After 200 iters: RSS={rss1//1024}MB, {len(maps1)} mappings, sum_rss={total_rss1//1024}MB, delta={((rss1-rss0)//1024):+d}MB")

# Find grown/new mappings
print("\n--- Grown/changed mappings (delta_rss > 100KB) ---")
for addr in sorted(set(maps0.keys()) | set(maps1.keys())):
    s0, r0, p0, path0 = maps0.get(addr, (0, 0, "", ""))
    s1, r1, p1, path1 = maps1.get(addr, (0, 0, "", ""))
    drss = r1 - r0
    if abs(drss) > 100:
        label = path1 or path0 or "(anon)"
        print(f"  {addr}: RSS {r0}->{r1} KB (delta={drss:+d}KB) [{label}]")

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
