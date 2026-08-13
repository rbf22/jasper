"""Quick test: run model forward only (no backward) to isolate hang location."""
import sys, time, torch, ttnn
sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import load_config

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024

device = ttnn.open_device(device_id=0)

cfg_dict = load_config("configs/cell_a_tt.yaml")
config = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ModelConfig, k)})
model = TTWRAPModel(config, device)
print(f"Model: {len(model.layers)} layers, d_model={config.d_model}")

B, T = 4, 128
input_ids = torch.randint(0, config.vocab_size, (B, T), dtype=torch.int32)
labels = torch.randint(0, config.vocab_size, (B, T), dtype=torch.int32)

print("Running forward-only x5...")
for i in range(5):
    t0 = time.time()
    logits = model.forward(input_ids, device)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  fwd {i}: {t1-t0:.3f}s  RSS={rss_mb():.0f}MB")
    ttnn.deallocate(logits)

print("Forward-only OK. Now forward+backward x3...")
from train_ttnn import cross_entropy_loss
for i in range(3):
    t0 = time.time()
    logits = model.forward(input_ids, device)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  fwd {i}: {t1-t0:.3f}s")

    t2 = time.time()
    loss, loss_tensor = cross_entropy_loss(logits, labels)
    ttnn.synchronize_device(device)
    t3 = time.time()
    print(f"  loss {i}: {t3-t2:.3f}s  loss={loss:.4f}")

    t4 = time.time()
    grads = model.backward(loss_tensor)
    ttnn.synchronize_device(device)
    t5 = time.time()
    print(f"  bwd {i}: {t5-t4:.3f}s  grads={len(grads)}  RSS={rss_mb():.0f}MB")
    ttnn.deallocate(logits)
    ttnn.deallocate(loss_tensor)

print("All done!")
ttnn.close_device(device)
