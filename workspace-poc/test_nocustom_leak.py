"""Test model leak with custom kernels disabled — replace _fused_scale_decay with ttnn.mul."""
import gc, ctypes, sys, torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
import model_ttnn

# Monkey-patch _fused_scale_decay to use regular ttnn ops
def _patched_scale_decay(scores_raw, D_decay, scale, B, H, T, device):
    """Replace fused kernel with: scores = scores_raw * scale * D_decay.
    scores_raw: (BH*T, T), D_decay: (H*T, T) — broadcast over batch B."""
    # Reshape to 4D for broadcast: (B, H, T, T) * (1, H, T, T)
    scores_4d = ttnn.reshape(scores_raw, [B, H, T, T])
    D_4d = ttnn.reshape(D_decay, [1, H, T, T])
    # Scale (scalar mul works in 4D) — creates new tensor, not a view
    scaled = ttnn.mul(scores_4d, scale)
    # Decay (broadcast mul) — creates new tensor
    out_4d = ttnn.mul(scaled, D_4d)
    ttnn.deallocate(scaled, force=True)
    # Reshape back to 2D — may be a view of out_4d
    out = ttnn.reshape(out_4d, [B * H * T, T])
    # Don't deallocate out_4d if out is a view of it
    # The caller will deallocate out; out_4d will be cleaned up by clear_caches
    return out

model_ttnn.TTRetentionLayer._fused_scale_decay = staticmethod(_patched_scale_decay)

# Also patch _fused_gate_backward if it exists
def _patched_gate_backward(grad_out_gated, gate, out_flat, B, T, D, device):
    """Replace fused kernel with: grad_out_flat = grad_out_gated * gate, grad_g = sum(grad_out_gated * out_flat)."""
    grad_out_flat = ttnn.mul(grad_out_gated, gate)
    # grad_g = sum over B,T of (grad_out_gated * out_flat)
    product = ttnn.mul(grad_out_gated, out_flat)
    grad_g = ttnn.sum(product, dim=(0, 1))
    ttnn.deallocate(product, force=True)
    return grad_out_flat, grad_g

if hasattr(model_ttnn.TTRetentionLayer, '_fused_gate_backward'):
    model_ttnn.TTRetentionLayer._fused_gate_backward = staticmethod(_patched_gate_backward)

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
print(f"Test (model with custom kernels DISABLED): Baseline RSS={rss0//1024}MB")

for i in range(500):
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
    if i in {0, 99, 199, 299, 399, 499}:
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rss = rss_kb()
        delta = rss - rss0
        rate = delta / (i + 1)
        print(f"  iter {i:3d}: RSS={rss//1024}MB  delta={delta//1024:+d}MB  rate={rate/1024:.3f} MB/iter")

ttnn.close_device(device)
