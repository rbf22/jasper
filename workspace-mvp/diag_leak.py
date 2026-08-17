"""Focused leak diagnostic: track RSS per step and per phase.

Runs 20 steps and reports:
- RSS before/after each step
- RSS delta per step
- RSS delta per phase (forward, loss, backward, optimizer, clear_caches)
- Python object counts to detect reference leaks
"""
import os, sys, gc, time, torch, ttnn, math
sys.path.insert(0, os.path.dirname(__file__))

# Mesh descriptor setup
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('ttnn')
for name in ['p150_mesh_graph_descriptor.textproto', 'p300_mesh_graph_descriptor.textproto']:
    if spec and spec.submodule_search_locations:
        p = Path(next(iter(spec.submodule_search_locations))) / 'tt_metal' / 'fabric' / 'mesh_graph_descriptors' / name
        if p.is_file():
            os.environ['TT_MESH_GRAPH_DESC_PATH'] = str(p)
            break

from tt_text_latent_memory_model import TTTextLatentMemoryModel, TTTextLatentMemoryConfig, _safe_deallocate
from train_native_tt import TTAdamW, cross_entropy_loss_and_grad, clip_grad_norm, get_lr
from challenge_data import ChallengeDataset
import random

def get_rss():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024.0  # MB
    return 0.0

def get_device_memory(device):
    """Get device memory usage if available."""
    try:
        view = ttnn.get_memory_view(device)
        return {
            'dram': view.dram_per_bank[0].total_allocated_bytes / 1e6 if view.dram_per_bank else 0,
            'l1': sum(b.total_allocated_bytes for b in view.l1_per_bank) / 1e6 if view.l1_per_bank else 0,
        }
    except Exception:
        return {}

device = ttnn.open_device(device_id=0)

config = TTTextLatentMemoryConfig(
    vocab_size=50257, d_model=128, n_encoder_layers=2, n_decoder_layers=2,
    n_heads=4, n_slots=8, max_reasoning_steps=3, expand=2,
    max_prompt_len=64, max_answer_len=16,
)
dtype = ttnn.bfloat16
model = TTTextLatentMemoryModel(config, device, dtype=dtype)
print(f"Model: {model.get_num_params():,} params")

# Find data files
train_path = os.path.join(os.path.dirname(__file__), 'data', 'tiny_challenges_train.txt')
valid_path = os.path.join(os.path.dirname(__file__), 'data', 'tiny_challenges_valid.txt')
dataset = ChallengeDataset(
    train_path=train_path, valid_path=valid_path,
    max_prompt_len=64, max_answer_len=16,
)
optimizer = TTAdamW(model.get_params(), lr=6e-4, weight_decay=0.1)

# Warmup - run 3 steps to stabilize
print("Warmup...")
for i in range(3):
    rng = random.Random(42 + i)
    prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
        dataset.sample_batch(4, "train", rng=rng)
    logits_tt = model.forward_train(prompt_ids, prompt_mask, dec_input, ans_mask)
    loss_val, grad_logits = cross_entropy_loss_and_grad(
        logits_tt, ans_targets, ans_mask, device, dtype=dtype)
    _safe_deallocate(logits_tt)
    grads = model.backward(grad_logits)
    _safe_deallocate(grad_logits)
    grad_norm = clip_grad_norm(grads, 1.0)
    optimizer.step(grads, model)
    for g in grads.values():
        _safe_deallocate(g)
    model.clear_caches()
    gc.collect()

print(f"After warmup RSS: {get_rss():.1f} MB")
print()

# Now measure each phase for 20 steps
print("=== Per-step RSS tracking ===")
rss_before = get_rss()
print(f"Baseline RSS: {rss_before:.1f} MB")

step_rss = []
grad_norms = []
for step in range(50):
    rng = random.Random(100 + step)
    prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
        dataset.sample_batch(4, "train", rng=rng)

    rss_pre = get_rss()

    # Forward
    logits_tt = model.forward_train(prompt_ids, prompt_mask, dec_input, ans_mask)
    rss_post_fwd = get_rss()

    # Loss
    loss_val, grad_logits = cross_entropy_loss_and_grad(
        logits_tt, ans_targets, ans_mask, device, dtype=dtype)
    _safe_deallocate(logits_tt)
    rss_post_loss = get_rss()

    # Backward
    grads = model.backward(grad_logits)
    _safe_deallocate(grad_logits)
    rss_post_bwd = get_rss()

    # Clip
    grad_norm = clip_grad_norm(grads, 1.0)
    grad_norms.append(grad_norm)
    rss_post_clip = get_rss()

    # Optimizer
    optimizer.step(grads, model)
    rss_post_opt = get_rss()

    # Deallocate grads
    for g in grads.values():
        _safe_deallocate(g)
    rss_post_dealloc = get_rss()

    # Clear caches
    model.clear_caches()
    rss_post_clear = get_rss()

    # GC
    gc.collect()
    rss_post_gc = get_rss()

    delta = rss_post_gc - rss_pre
    step_rss.append(delta)

    # Check for inf/nan in gradients
    has_inf = math.isinf(grad_norm) or math.isnan(grad_norm)

    if step < 5 or step >= 45 or has_inf or (step % 10 == 0):
        flag = " *** INF/NAN" if has_inf else ""
        print(f"Step {step:3d}: pre={rss_pre:.1f} fwd+{rss_post_fwd-rss_pre:.1f} "
              f"loss+{rss_post_loss-rss_post_fwd:.1f} bwd+{rss_post_bwd-rss_post_loss:.1f} "
              f"clip+{rss_post_clip-rss_post_bwd:.1f} opt+{rss_post_opt-rss_post_clip:.1f} "
              f"dealloc+{rss_post_dealloc-rss_post_opt:.1f} clear+{rss_post_clear-rss_post_dealloc:.1f} "
              f"gc+{rss_post_gc-rss_post_clear:.1f} => delta={delta:+.1f}MB "
              f"grad={grad_norm:.3f}{flag}")
    elif step == 5:
        print("  ... (steps 5-44 omitted, showing inf/nan and every 10th) ...")

print()
total_delta = sum(step_rss)
avg_delta = total_delta / len(step_rss)
print(f"Total RSS delta over {len(step_rss)} steps: {total_delta:.1f} MB")
print(f"Average per-step: {avg_delta:.2f} MB/step")
print(f"Steps with growth: {sum(1 for d in step_rss if d > 0.1)}/{len(step_rss)}")
print(f"Steps with shrink: {sum(1 for d in step_rss if d < -0.1)}/{len(step_rss)}")

# Gradient stability summary
inf_count = sum(1 for g in grad_norms if math.isinf(g) or math.isnan(g))
large_count = sum(1 for g in grad_norms if g > 1e4 and not math.isinf(g))
finite_norms = [g for g in grad_norms if not math.isinf(g) and not math.isnan(g)]
print(f"\n=== Gradient stability ===")
print(f"Total steps: {len(grad_norms)}")
print(f"Inf/Nan gradients: {inf_count}")
print(f"Large (>1e4) gradients: {large_count}")
if finite_norms:
    print(f"Finite grad norm range: {min(finite_norms):.3f} - {max(finite_norms):.3f}")
    print(f"Finite grad norm mean: {sum(finite_norms)/len(finite_norms):.3f}")

# Check device memory
dev_mem = get_device_memory(device)
if dev_mem:
    print(f"\nDevice memory: DRAM={dev_mem.get('dram', 0):.1f}MB L1={dev_mem.get('l1', 0):.1f}MB")

# Check Python object counts
import ctypes
print(f"\nPython refcount sample: {sys.getrefcount(model)} refs to model")

ttnn.close_device(device)
print("Done.")
