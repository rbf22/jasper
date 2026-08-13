"""Test single retention layer forward+backward for leak."""
import gc, ctypes, sys, torch, ttnn

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

# Create a single retention layer
layer = TTRetentionLayer(model_config, device)

B, T, D = 8, 128, model_config.d_model

# Create input
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
print(f"Test (single retention layer fwd+bwd): Baseline RSS={rss0//1024}MB")

for i in range(500):
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
    # Don't manually clear cache — just let the next forward overwrite it
    # and Python GC handle the old tensors

    if i in {0, 99, 199, 299, 399, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        delta = rss - rss0
        rate = delta / (i + 1)
        print(f"  iter {i:3d}: RSS={rss//1024}MB  delta={delta//1024:+d}MB  rate={rate/1024:.3f} MB/iter")

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
