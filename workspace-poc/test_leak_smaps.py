#!/usr/bin/env python
"""Monitor /proc/self/smaps during model training to find which mapping grows."""
import gc
import os
import sys
import time
import re

import torch
import ttnn

# Same device setup as train_ttnn.py
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
            if path.is_file():
                return str(path)
        for p in sys.path:
            candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
            if candidate.is_file():
                return str(candidate)
        candidate = Path("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages") / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
        if candidate.is_file():
            return str(candidate)
    return None

_mgd = _find_mgd()
if _mgd:
    os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import TTWRAPModel, ModelConfig
from train_ttnn import build_model_config, cross_entropy_loss
from data import Vocab, sample_batch
import yaml


def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0


# Regex for smaps mapping header: address-range perm offset dev inode [path]
HEADER_RE = re.compile(r'^[0-9a-f]+-[0-9a-f]+ ')


def get_smaps_summary():
    """Parse /proc/self/smaps and return list of {path, Size, Rss, Anonymous, ...}."""
    mappings = []
    current = None
    with open('/proc/self/smaps') as f:
        for line in f:
            if HEADER_RE.match(line):
                if current:
                    mappings.append(current)
                parts = line.strip().split(None, 5)
                path = parts[5] if len(parts) > 5 else "[anon]"
                current = {"path": path, "header": parts[0]}
            elif current is not None:
                if ':' in line:
                    key, _, val = line.strip().partition(':')
                    try:
                        current[key] = int(val.split()[0])
                    except (ValueError, IndexError):
                        pass
    if current:
        mappings.append(current)

    mappings.sort(key=lambda m: m.get('Rss', 0), reverse=True)
    return mappings


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

    # Baseline smaps
    print("=== Baseline (after warmup) ===")
    rss0 = get_rss_kb()
    print(f"Total RSS: {rss0//1024} MB")
    smaps0 = get_smaps_summary()
    for m in smaps0[:10]:
        print(f"  RSS={m.get('Rss',0)//1024:>5} MB  Anon={m.get('Anonymous',0)//1024:>5} MB  "
              f"Size={m.get('Size',0)//1024:>5} MB  {m.get('path','')[:55]}")

    # Run 40 steps
    for step in range(40):
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

    print("\n=== After 40 steps ===")
    rss1 = get_rss_kb()
    print(f"Total RSS: {rss1//1024} MB  (delta={(rss1-rss0)//1024} MB)")
    smaps1 = get_smaps_summary()
    for m in smaps1[:10]:
        print(f"  RSS={m.get('Rss',0)//1024:>5} MB  Anon={m.get('Anonymous',0)//1024:>5} MB  "
              f"Size={m.get('Size',0)//1024:>5} MB  {m.get('path','')[:55]}")

    # Show which mappings grew
    print("\n=== Mappings that grew (> 500 KB) ===")
    smaps0_dict = {}
    for m in smaps0:
        key = m.get('header', m.get('path', ''))
        smaps0_dict[key] = m
    for m in smaps1:
        key = m.get('header', m.get('path', ''))
        m0 = smaps0_dict.get(key, {})
        delta_rss = m.get('Rss', 0) - m0.get('Rss', 0)
        if delta_rss > 500:  # > 500 KB
            print(f"  RSS delta={delta_rss//1024:>5} MB  "
                  f"Rss={m.get('Rss',0)//1024:>5} MB  "
                  f"Size={m.get('Size',0)//1024:>5} MB  "
                  f"Anon={m.get('Anonymous',0)//1024:>5} MB  "
                  f"{m.get('path','')[:60]}")

    # Show new mappings that appeared
    smaps1_keys = {m.get('header', m.get('path', '')) for m in smaps1}
    smaps0_keys = {m.get('header', m.get('path', '')) for m in smaps0}
    new_keys = smaps1_keys - smaps0_keys
    if new_keys:
        print(f"\n=== New mappings ({len(new_keys)}) ===")
        for m in smaps1:
            key = m.get('header', m.get('path', ''))
            if key in new_keys and m.get('Rss', 0) > 100:
                print(f"  RSS={m.get('Rss',0)//1024:>5} MB  "
                      f"Size={m.get('Size',0)//1024:>5} MB  "
                      f"Anon={m.get('Anonymous',0)//1024:>5} MB  "
                      f"{m.get('path','')[:60]}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
