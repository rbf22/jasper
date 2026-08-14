#!/usr/bin/env python
"""Identify the exact anonymous mappings that grow during training."""
import gc, os, sys, time, re
import torch, ttnn

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import importlib.util
from pathlib import Path
def _find_mgd():
    spec = importlib.util.find_spec("ttnn")
    for name in ["p150_mesh_graph_descriptor.textproto", "p300_mesh_graph_descriptor.textproto"]:
        if spec is not None and spec.submodule_search_locations:
            path = (Path(next(iter(spec.submodule_search_locations)))
                    / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name)
            if path.is_file(): return str(path)
        for p in sys.path:
            candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
            if candidate.is_file(): return str(candidate)
        candidate = Path("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages") / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
        if candidate.is_file(): return str(candidate)
    return None
_mgd = _find_mgd()
if _mgd: os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import build_model_config, cross_entropy_loss
from data import Vocab, sample_batch
import yaml

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'): return int(line.split()[1])
    return 0

def get_maps():
    """Parse /proc/self/maps, return list of (start, end, perms, path)."""
    maps = []
    with open('/proc/self/maps') as f:
        for line in f:
            parts = line.strip().split(None, 5)
            addr_range = parts[0]
            start, end = addr_range.split('-')
            start = int(start, 16)
            end = int(end, 16)
            perms = parts[1] if len(parts) > 1 else ""
            path = parts[5] if len(parts) > 5 else "[anon]"
            maps.append((start, end, perms, path))
    return maps


def main():
    with open("configs/cell_b_tt.yaml") as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

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
        try: ttnn.deallocate(grad_logits)
        except: pass
        for g in grads.values():
            try: ttnn.deallocate(g)
            except: pass
        try: ttnn.deallocate(logits)
        except: pass
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
    gc.collect()
    ttnn.synchronize_device(device)

    # Baseline maps
    maps0 = get_maps()
    rss0 = get_rss_kb()
    print(f"Baseline: RSS={rss0//1024} MB, mappings={len(maps0)}")

    # Show all large anonymous mappings (> 1 MB)
    print("\nLarge anonymous mappings (> 1 MB) at baseline:")
    for start, end, perms, path in maps0:
        size_mb = (end - start) / (1024*1024)
        if path == "[anon]" and size_mb > 1:
            print(f"  {start:#012x}-{end:#012x}  {perms}  {size_mb:>8.1f} MB")

    # Run 30 steps
    for step in range(30):
        input_ids, labels, _ = sample_batch(8, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        try: ttnn.deallocate(grad_logits)
        except: pass
        for g in grads.values():
            try: ttnn.deallocate(g)
            except: pass
        try: ttnn.deallocate(logits)
        except: pass
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
        gc.collect()

    ttnn.synchronize_device(device)
    time.sleep(0.5)

    maps1 = get_maps()
    rss1 = get_rss_kb()
    print(f"\nAfter 30 steps: RSS={rss1//1024} MB, mappings={len(maps1)}  (delta={(rss1-rss0)//1024} MB)")

    # Compare: find new mappings and changed mappings
    maps0_dict = {(start, end): (perms, path) for start, end, perms, path in maps0}
    maps1_dict = {(start, end): (perms, path) for start, end, perms, path in maps1}

    # New mappings
    new_maps = set(maps1_dict.keys()) - set(maps0_dict.keys())
    if new_maps:
        print(f"\n=== New mappings ({len(new_maps)}) ===")
        for start, end in sorted(new_maps):
            perms, path = maps1_dict[(start, end)]
            size_mb = (end - start) / (1024*1024)
            if size_mb > 0.01:  # > 10 KB
                print(f"  {start:#012x}-{end:#012x}  {perms}  {size_mb:>8.1f} MB  {path[:50]}")

    # Removed mappings
    removed_maps = set(maps0_dict.keys()) - set(maps1_dict.keys())
    if removed_maps:
        print(f"\n=== Removed mappings ({len(removed_maps)}) ===")
        for start, end in sorted(removed_maps):
            perms, path = maps0_dict[(start, end)]
            size_mb = (end - start) / (1024*1024)
            if size_mb > 0.01:
                print(f"  {start:#012x}-{end:#012x}  {perms}  {size_mb:>8.1f} MB  {path[:50]}")

    # Changed mappings (same start, different end = grew)
    starts0 = {start: (end, perms, path) for start, end, perms, path in maps0}
    starts1 = {start: (end, perms, path) for start, end, perms, path in maps1}
    common_starts = set(starts0.keys()) & set(starts1.keys())
    grew = []
    for s in common_starts:
        e0, p0, path0 = starts0[s]
        e1, p1, path1 = starts1[s]
        if e1 != e0:
            delta = e1 - e0
            grew.append((s, e0, e1, delta, path1))
    if grew:
        print(f"\n=== Mappings that grew (same start, expanded end) ===")
        for s, e0, e1, delta, path in sorted(grew, key=lambda x: -x[3]):
            print(f"  {s:#012x}: {e0:#012x} -> {e1:#012x}  delta={delta//1024:>6} KB  {path[:50]}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
