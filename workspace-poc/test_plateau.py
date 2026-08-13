"""Run 5000 iterations to see if RSS growth plateaus (demand paging) or is linear (real leak)."""
import gc, ctypes, sys, os, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024

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

rss0 = rss_mb()
print(f"Baseline: RSS={rss0:.0f}MB")

for i in range(5000):
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

    if i in {0, 99, 499, 999, 1999, 2999, 3999, 4999}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_mb()
        delta = rss - rss0
        rate = delta / (i + 1)
        print(f"  iter {i:5d}: RSS={rss:.0f}MB  delta={delta:+.0f}MB  rate={rate:.3f} MB/iter")

ttnn.close_device(device)
