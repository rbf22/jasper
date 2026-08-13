"""Minimal hang repro: 2 forward passes with sync."""
import sys, time, torch, ttnn
sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import load_config

device = ttnn.open_device(device_id=0)

cfg_dict = load_config("configs/cell_a_tt.yaml")
config = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ModelConfig, k)})
config.n_layers = 14
config.vocab_size = 128
model = TTWRAPModel(config, device)
print(f"Model: {len(model.layers)} layers", flush=True)

B, T = 4, 128
input_ids = torch.randint(0, config.vocab_size, (B, T), dtype=torch.int32)

for i in range(3):
    print(f"--- forward {i} start ---", flush=True)
    t0 = time.time()
    logits = model.forward(input_ids, device)
    print(f"--- forward {i} returned, syncing ---", flush=True)
    ttnn.synchronize_device(device)
    t1 = time.time()
    print(f"  fwd {i}: {t1-t0:.3f}s OK", flush=True)
    ttnn.deallocate(logits)

print("All done!")
ttnn.close_device(device)
