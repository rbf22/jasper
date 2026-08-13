"""Test just the _fused_scale_decay custom kernel in isolation."""
import gc, ctypes, sys, os, struct, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import _safe_deallocate, TTRetentionLayer
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

B, H, T, D = 8, 4, 128, model_config.d_model
BH = B * H

# Create input tensors
scores_raw = ttnn.from_torch(torch.randn(BH, T, T, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
D_decay = ttnn.from_torch(torch.randn(1, H, T, T, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
scale = 0.125

# Warmup
for _ in range(3):
    out = TTRetentionLayer._fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device)
    ttnn.synchronize_device(device)
    _safe_deallocate(out)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"Baseline RSS={rss0//1024}MB", flush=True)

# Run 500 iterations of just _fused_scale_decay
for i in range(500):
    out = TTRetentionLayer._fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device)
    ttnn.synchronize_device(device)
    _safe_deallocate(out)
    if i % 100 == 99:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i+1:3d}: RSS={rss//1024}MB  delta={((rss-rss0)//1024):+d}MB  rate={((rss-rss0)/1024.0/(i+1)*1000):.1f} KB/iter", flush=True)

gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss1 = rss_kb()
print(f"After 500 iters: RSS={rss1//1024}MB (delta={((rss1-rss0)//1024):+d}MB)", flush=True)

ttnn.deallocate(scores_raw, force=True)
ttnn.deallocate(D_decay, force=True)
ttnn.close_device(device)
