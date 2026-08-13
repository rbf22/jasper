"""Use malloc_info to dump heap state before and after retention layer iterations."""
import gc, ctypes, sys, os, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTRetentionLayer, _safe_deallocate
from train_ttnn import build_model_config
import yaml

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

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
rss0 = rss_kb()

# Dump malloc stats to stderr before
libc = ctypes.CDLL("libc.so.6")
print(f"=== BEFORE: RSS={rss0//1024}MB ===", file=sys.stderr)
libc.malloc_stats()
print(f"=== END BEFORE ===", file=sys.stderr)

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
rss1 = rss_kb()

print(f"=== AFTER 200 iters: RSS={rss1//1024}MB (delta={((rss1-rss0)//1024):+d}MB) ===", file=sys.stderr)
libc.malloc_stats()
print(f"=== END AFTER ===", file=sys.stderr)

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
