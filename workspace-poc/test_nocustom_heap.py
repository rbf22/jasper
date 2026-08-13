"""Test retention layer with _fused_scale_decay replaced by ttnn.mul to isolate leak source."""
import gc, ctypes, sys, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTRetentionLayer, _safe_deallocate
from train_ttnn import build_model_config
import yaml

# Monkey-patch _fused_scale_decay to use ttnn.mul instead of custom kernel
def _fused_scale_decay_ttnn(scores_raw, D_decay, scale, B, H, T, device):
    """Replace custom kernel with standard ttnn ops: scores = scores_raw * scale * D_decay."""
    # Reshape to 4D for broadcast: scores_raw (B,H,T,T), D_decay (1,H,T,T)
    scores_4d = ttnn.reshape(scores_raw, [B, H, T, T])
    decay_4d = ttnn.reshape(D_decay, [1, H, T, T])
    scaled = ttnn.mul(scores_4d, scale)
    out_4d = ttnn.mul(scaled, decay_4d)
    out_2d = ttnn.reshape(out_4d, [B * H * T, T])
    _safe_deallocate(scaled)
    _safe_deallocate(out_4d)
    return out_2d

TTRetentionLayer._fused_scale_decay = staticmethod(_fused_scale_decay_ttnn)

libc = ctypes.CDLL("libc.so.6")

class mallinfo2_t(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]

libc.mallinfo2.restype = mallinfo2_t

def heap_info():
    info = libc.mallinfo2()
    return info.arena, info.uordblks, info.fordblks, info.hblkhd

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
libc.malloc_trim(0)
arena0, used0, free0, mmap0 = heap_info()
rss0 = rss_kb()
print(f"Baseline (ttnn.mul replacement): RSS={rss0//1024}MB arena={arena0//1024}KB used={used0//1024}KB free={free0//1024}KB mmap={mmap0//1024}KB")

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

    if i in {0, 49, 99, 199, 299, 399, 499}:
        gc.collect()
        libc.malloc_trim(0)
        arena, used, free, mmap = heap_info()
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB arena={arena//1024}KB used={used//1024}KB free={free//1024}KB mmap={mmap//1024}KB  "
              f"delta_arena={(arena-arena0)//1024:+d}KB delta_used={(used-used0)//1024:+d}KB delta_mmap={(mmap-mmap0)//1024:+d}KB")

ttnn.deallocate(x, force=True)
ttnn.close_device(device)
