#!/usr/bin/env python
"""Test if MALLOC_ARENA_MAX=1 + malloc_trim fixes RSS growth."""
import gc, ctypes, os, sys, time
import torch, ttnn

libc = ctypes.CDLL("libc.so.6")
libc.mallinfo2.restype = ctypes.c_int  # just to avoid error

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import build_model_config, cross_entropy_loss
from data import Vocab, sample_batch
import yaml

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

def get_heap_kb():
    class mallinfo2_t(ctypes.Structure):
        _fields_ = [("arena", ctypes.c_size_t), ("ordblks", ctypes.c_size_t), ("smblks", ctypes.c_size_t),
                     ("hblks", ctypes.c_size_t), ("hblkhd", ctypes.c_size_t), ("usmblks", ctypes.c_size_t),
                     ("fsmblks", ctypes.c_size_t), ("uordblks", ctypes.c_size_t), ("fordblks", ctypes.c_size_t),
                     ("keepcost", ctypes.c_size_t)]
    libc.mallinfo2.restype = mallinfo2_t
    return libc.mallinfo2().uordblks // 1024

device = ttnn.open_device(device_id=0)

with open("configs/cell_b_tt.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["micro_batch_size"] = 0
model_config = build_model_config(cfg)
model = TTWRAPModel(model_config, device)
vocab = Vocab()
seq_len = cfg.get("seq_len", 128)
import random
rng = random.Random(42)

# Warmup
for _ in range(3):
    input_ids, labels, _ = sample_batch(8, seq_len, vocab, rng=rng)
    logits = model.forward(input_ids)
    loss_val, grad_logits = cross_entropy_loss(logits, labels)
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)
    _safe_deallocate(grad_logits)
    for g in grads.values(): _safe_deallocate(g)
    _safe_deallocate(logits)
    model.clear_caches()
    del logits, grad_logits, grads, input_ids, labels
gc.collect()
libc.malloc_trim(0)

rss0 = get_rss_kb()
heap0 = get_heap_kb()
print(f"Baseline: RSS={rss0//1024}MB, heap={heap0}KB")
print(f"{'Step':>6} {'RSS(MB)':>10} {'dRSS(KB)':>10} {'Heap(KB)':>10} {'dHeap(KB)':>10}")
prev_rss = rss0
prev_heap = heap0

for step in range(200):
    input_ids, labels, _ = sample_batch(8, seq_len, vocab, rng=rng)
    logits = model.forward(input_ids)
    loss_val, grad_logits = cross_entropy_loss(logits, labels)
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)
    _safe_deallocate(grad_logits)
    for g in grads.values(): _safe_deallocate(g)
    _safe_deallocate(logits)
    model.clear_caches()
    del logits, grad_logits, grads, input_ids, labels
    gc.collect()
    libc.malloc_trim(0)  # Return freed pages to OS

    if step % 20 == 0 or step == 199:
        rss = get_rss_kb()
        heap = get_heap_kb()
        print(f"{step:>6} {rss//1024:>10} {rss-prev_rss:>10} {heap:>10} {heap-prev_heap:>10}")
        prev_rss = rss
        prev_heap = heap

final_rss = get_rss_kb()
print(f"\nTotal RSS growth: {(final_rss-rss0)//1024} MB over 200 steps = {(final_rss-rss0)/200:.1f} KB/step")
ttnn.close_device(device)
