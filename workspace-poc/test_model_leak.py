"""Narrow down: is the leak in forward, backward, or loss?"""
import gc, ctypes, sys, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4

device = ttnn.open_device(device_id=0)

with open("configs/cell_a_tt.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["micro_batch_size"] = 0
model_config = build_model_config(cfg)
model = TTWRAPModel(model_config, device)

B, T, V = 8, 128, 128

# Warmup all paths
for _ in range(3):
    input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    labels = torch.randint(0, V, (B, T), dtype=torch.int32)
    logits = model.forward(input_ids, k_value=None)
    ttnn.synchronize_device(device)
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

# Test A: Forward only
rss0 = rss_kb()
print(f"Test A (forward only): Baseline RSS={rss0//1024}MB")
for i in range(200):
    input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    logits = model.forward(input_ids, k_value=None)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)
    model.clear_caches()
    if i in {0, 99, 199}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

# Test B: Forward + loss only (no backward)
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest B (forward + loss): Baseline RSS={rss0//1024}MB")
for i in range(200):
    input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    labels = torch.randint(0, V, (B, T), dtype=torch.int32)
    logits = model.forward(input_ids, k_value=None)
    loss_val, grad_logits = cross_entropy_loss(logits, labels)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)
    _safe_deallocate(grad_logits)
    model.clear_caches()
    if i in {0, 99, 199}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

# Test C: Forward + loss + backward
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
rss0 = rss_kb()
print(f"\nTest C (forward + loss + backward): Baseline RSS={rss0//1024}MB")
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
    if i in {0, 99, 199}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        print(f"  iter {i:3d}: RSS={rss//1024}MB (+{(rss-rss0)//1024})")

ttnn.close_device(device)
