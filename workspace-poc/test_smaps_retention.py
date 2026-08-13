"""Track which anonymous mappings grow during retention layer execution."""
import gc, ctypes, sys, os, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTRetentionLayer, _safe_deallocate
from train_ttnn import build_model_config
import yaml

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

def dump_anon_mappings():
    """Return dict of {address: (size_kb, rss_kb, label)} for anon mappings."""
    maps = {}
    pid = os.getpid()
    with open(f"/proc/{pid}/smaps") as f:
        content = f.read()
    for block in content.split("\n\n"):
        lines = block.strip().split("\n")
        if not lines:
            continue
        header = lines[0]
        parts = header.split()
        if len(parts) < 5:
            continue
        addr_range = parts[0]
        perms = parts[1]
        # Look for anonymous mappings (no file path)
        is_anon = len(parts) == 5 and parts[4] == "0"
        if not is_anon:
            # Also check for mappings with no path at end
            if len(parts) >= 6:
                path = parts[-1]
                if path.startswith("[") or path == "":
                    is_anon = True
            continue  # Skip file-backed
        if perms == "---p":  # Skip PROT_NONE reservations for now
            pass
        size_kb = 0
        rss_kb = 0
        for line in lines[1:]:
            if line.startswith("Size:"):
                size_kb = int(line.split()[1])
            elif line.startswith("Rss:"):
                rss_kb = int(line.split()[1])
        if size_kb > 100:  # Only track significant mappings
            maps[addr_range] = (size_kb, rss_kb, perms)
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
maps0 = dump_anon_mappings()
rss0 = rss_kb()
print(f"Baseline RSS={rss0//1024}MB, {len(maps0)} anon mappings")

# Run 200 iterations
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
maps1 = dump_anon_mappings()
rss1 = rss_kb()
print(f"After 200 iters: RSS={rss1//1024}MB, {len(maps1)} anon mappings, delta={((rss1-rss0)//1024):+d}MB")

# Compare mappings
print("\n--- Grown mappings ---")
for addr in sorted(set(maps0.keys()) | set(maps1.keys())):
    s0, r0, p0 = maps0.get(addr, (0, 0, ""))
    s1, r1, p1 = maps1.get(addr, (0, 0, ""))
    drss = r1 - r0
    if drss > 0:
        print(f"  {addr} {p0 or p1}: size {s0}->{s1} KB, RSS {r0}->{r1} KB, delta_rss={drss} KB")

print("\n--- New mappings ---")
for addr in sorted(set(maps1.keys()) - set(maps0.keys())):
    s1, r1, p1 = maps1[addr]
    if r1 > 0:
        print(f"  {addr} {p1}: size={s1} KB, RSS={r1} KB")

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
