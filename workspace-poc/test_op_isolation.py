"""Test with just ttnn operations (no custom kernels) to isolate the leak."""
import gc, ctypes, sys, os, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import _safe_deallocate
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

B, T, D = 8, 128, model_config.d_model

# Create tensors that mimic what the retention layer uses
x = ttnn.from_torch(torch.randn(B, T, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Simulate the operations that the retention layer does (without custom kernels)
# The retention layer does: matmul, softmax, mul, RoPE, etc.
# Let's just do a simple chain of operations that matches the number of ops

# Warmup
for _ in range(3):
    # Simulate forward: a few matmuls and elementwise ops
    w = ttnn.from_torch(torch.randn(D, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.matmul(x, w)
    out2 = ttnn.softmax(out, dim=-1)
    out3 = ttnn.mul(out2, out)
    ttnn.synchronize_device(device)
    _safe_deallocate(w)
    _safe_deallocate(out)
    _safe_deallocate(out2)
    _safe_deallocate(out3)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"Baseline RSS={rss0//1024}MB", flush=True)

# Run 500 iterations of simple ops
for i in range(500):
    w = ttnn.from_torch(torch.randn(D, D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.matmul(x, w)
    out2 = ttnn.softmax(out, dim=-1)
    out3 = ttnn.mul(out2, out)
    ttnn.synchronize_device(device)
    _safe_deallocate(w)
    _safe_deallocate(out)
    _safe_deallocate(out2)
    _safe_deallocate(out3)
    if i % 100 == 99:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i+1:3d}: RSS={rss//1024}MB  delta={((rss-rss0)//1024):+d}MB  rate={((rss-rss0)/1024.0/(i+1)*1000):.1f} KB/iter", flush=True)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss1 = rss_kb()
print(f"After 500 iters: RSS={rss1//1024}MB (delta={((rss1-rss0)//1024):+d}MB)", flush=True)

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
