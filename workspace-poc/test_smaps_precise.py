#!/usr/bin/env python
"""Precise smaps comparison: which mapping's RSS grows during training?"""
import gc
import os
import sys
import time
import re

import torch
import ttnn

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

HEADER_RE = re.compile(r'^([0-9a-f]+)-([0-9a-f]+) ')

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

def parse_smaps():
    """Parse /proc/self/smaps, return dict keyed by address range."""
    mappings = {}
    current_key = None
    current = {}
    with open('/proc/self/smaps') as f:
        for line in f:
            m = HEADER_RE.match(line)
            if m:
                if current_key:
                    mappings[current_key] = current
                current_key = m.group(1) + "-" + m.group(2)
                parts = line.strip().split(None, 5)
                current = {
                    "addr": current_key,
                    "path": parts[5] if len(parts) > 5 else "[anon]",
                    "Size": 0, "Rss": 0, "Pss": 0, "Anonymous": 0,
                    "Private_Dirty": 0, "Swap": 0,
                }
            elif current_key and ':' in line:
                key, _, val = line.strip().partition(':')
                try:
                    current[key] = int(val.split()[0])
                except (ValueError, IndexError):
                    pass
    if current_key:
        mappings[current_key] = current
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
    rss0 = get_rss_kb()
    smaps0 = parse_smaps()
    total_rss0 = sum(m["Rss"] for m in smaps0.values())
    print(f"Baseline: RSS={rss0//1024} MB, smaps total Rss={total_rss0//1024} MB, "
          f"mappings={len(smaps0)}")

    # Show top 10 by Size (virtual size)
    print("\nTop 10 by virtual Size:")
    for m in sorted(smaps0.values(), key=lambda x: x["Size"], reverse=True)[:10]:
        print(f"  Size={m['Size']//1024:>6} MB  Rss={m['Rss']//1024:>5} MB  "
              f"Anon={m['Anonymous']//1024:>5} MB  {m['path'][:50]}")

    # Run 20 steps
    for step in range(20):
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

    rss1 = get_rss_kb()
    smaps1 = parse_smaps()
    total_rss1 = sum(m["Rss"] for m in smaps1.values())
    print(f"\nAfter 20 steps: RSS={rss1//1024} MB, smaps total Rss={total_rss1//1024} MB  "
          f"(delta RSS={(rss1-rss0)//1024} MB, smaps delta={(total_rss1-total_rss0)//1024} MB)")

    # Find which mappings grew
    print("\n=== Mappings with RSS growth > 100 KB ===")
    grew = []
    for key, m1 in smaps1.items():
        m0 = smaps0.get(key, {"Rss": 0, "Size": 0, "path": "[new]", "Anonymous": 0})
        delta_rss = m1["Rss"] - m0["Rss"]
        if delta_rss > 100:
            grew.append((delta_rss, m1, m0))

    grew.sort(key=lambda x: x[0], reverse=True)
    for delta_rss, m1, m0 in grew:
        print(f"  delta_Rss={delta_rss//1024:>5} MB  "
              f"Rss={m0['Rss']//1024:>4}->{m1['Rss']//1024:>4} MB  "
              f"Size={m1['Size']//1024:>6} MB  "
              f"Anon={m1['Anonymous']//1024:>5} MB  "
              f"{m1['path'][:55]}")

    # Also check for new mappings
    new_keys = set(smaps1.keys()) - set(smaps0.keys())
    if new_keys:
        print(f"\n=== New mappings ({len(new_keys)}) ===")
        for key in new_keys:
            m = smaps1[key]
            if m["Rss"] > 10:
                print(f"  Rss={m['Rss']//1024:>5} MB  Size={m['Size']//1024:>6} MB  "
                      f"Anon={m['Anonymous']//1024:>5} MB  {m['path'][:55]}")

    # Summary: is the growth in one big mapping or many small ones?
    total_growth = sum(d for d, _, _ in grew)
    print(f"\nTotal growth in mappings that grew: {total_growth//1024} MB")
    print(f"Number of mappings that grew: {len(grew)}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
