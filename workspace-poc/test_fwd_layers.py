"""Test with varying layer counts to isolate which layer's binary_ng hangs."""
import sys, time, torch, ttnn
sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import load_config

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024

device = ttnn.open_device(device_id=0)

cfg_dict = load_config("configs/cell_a_tt.yaml")

for n_layers in [1, 2, 4, 8, 14]:
    print(f"\n=== Testing {n_layers} layers ===", flush=True)
    config = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ModelConfig, k)})
    config.n_layers = n_layers
    config.vocab_size = 128
    model = TTWRAPModel(config, device)

    B, T = 4, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T), dtype=torch.int32)

    t0 = time.time()
    try:
        logits = model.forward(input_ids, device)
        ttnn.synchronize_device(device)
        t1 = time.time()
        print(f"  {n_layers}L forward: {t1-t0:.3f}s OK", flush=True)
        ttnn.deallocate(logits)
    except Exception as e:
        print(f"  {n_layers}L forward FAILED: {e}", flush=True)
    del model

print("\nAll done!")
ttnn.close_device(device)
