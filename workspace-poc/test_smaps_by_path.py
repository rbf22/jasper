#!/usr/bin/env python
"""Compare smaps by path (not address) to find which mapping type grows."""
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

HEADER_RE = re.compile(r'^([0-9a-f]+)-([0-9a-f]+) ')

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'): return int(line.split()[1])
    return 0

def parse_smaps_by_path():
    """Parse /proc/self/smaps, group by path, return {path: {Rss, Size, Anonymous, count}}."""
    by_path = {}
    current = None
    with open('/proc/self/smaps') as f:
        for line in f:
            m = HEADER_RE.match(line)
            if m:
                parts = line.strip().split(None, 5)
                path = parts[5] if len(parts) > 5 else "[anon]"
                current = {"path": path, "Rss": 0, "Size": 0, "Anonymous": 0}
            elif current and ':' in line:
                key, _, val = line.strip().partition(':')
                try:
                    current[key] = int(val.split()[0])
                except (ValueError, IndexError):
                    pass
            elif m is None and current and line.strip() and not ':' in line.split()[0]:
                # End of entry
                pass
        # Last entry
        if current:
            p = current["path"]
            if p not in by_path:
                by_path[p] = {"Rss": 0, "Size": 0, "Anonymous": 0, "count": 0}
            by_path[p]["Rss"] += current["Rss"]
            by_path[p]["Size"] += current["Size"]
            by_path[p]["Anonymous"] += current["Anonymous"]
            by_path[p]["count"] += 1
    return by_path

def parse_smaps_by_path_v2():
    """Parse /proc/self/smaps properly, group by path."""
    by_path = {}
    current = None
    with open('/proc/self/smaps') as f:
        for line in f:
            m = HEADER_RE.match(line)
            if m:
                # Save previous
                if current:
                    p = current["path"]
                    if p not in by_path:
                        by_path[p] = {"Rss": 0, "Size": 0, "Anonymous": 0, "count": 0}
                    by_path[p]["Rss"] += current["Rss"]
                    by_path[p]["Size"] += current["Size"]
                    by_path[p]["Anonymous"] += current["Anonymous"]
                    by_path[p]["count"] += 1
                parts = line.strip().split(None, 5)
                path = parts[5] if len(parts) > 5 else "[anon]"
                current = {"path": path, "Rss": 0, "Size": 0, "Anonymous": 0}
            elif current and ':' in line:
                key, _, val = line.strip().partition(':')
                try:
                    current[key] = int(val.split()[0])
                except (ValueError, IndexError):
                    pass
    # Save last
    if current:
        p = current["path"]
        if p not in by_path:
            by_path[p] = {"Rss": 0, "Size": 0, "Anonymous": 0, "count": 0}
        by_path[p]["Rss"] += current["Rss"]
        by_path[p]["Size"] += current["Size"]
        by_path[p]["Anonymous"] += current["Anonymous"]
        by_path[p]["count"] += 1
    return by_path


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

    rss0 = get_rss_kb()
    smaps0 = parse_smaps_by_path_v2()
    total_rss0 = sum(m["Rss"] for m in smaps0.values())
    print(f"Baseline: RSS={rss0//1024} MB, smaps total={total_rss0//1024} MB, paths={len(smaps0)}")

    # Show top 10 by RSS
    print("\nTop 10 by RSS:")
    for p, m in sorted(smaps0.items(), key=lambda x: x[1]["Rss"], reverse=True)[:10]:
        print(f"  Rss={m['Rss']//1024:>5} MB  Size={m['Size']//1024:>6} MB  "
              f"Anon={m['Anonymous']//1024:>5} MB  n={m['count']:>3}  {p[:50]}")

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
    smaps1 = parse_smaps_by_path_v2()
    total_rss1 = sum(m["Rss"] for m in smaps1.values())
    print(f"\nAfter 20 steps: RSS={rss1//1024} MB, smaps total={total_rss1//1024} MB  "
          f"(delta={(rss1-rss0)//1024} MB)")

    # Compare by path
    print("\n=== Paths with RSS growth > 50 KB ===")
    all_paths = set(smaps0.keys()) | set(smaps1.keys())
    grew = []
    for p in all_paths:
        m0 = smaps0.get(p, {"Rss": 0, "Size": 0, "Anonymous": 0, "count": 0})
        m1 = smaps1.get(p, {"Rss": 0, "Size": 0, "Anonymous": 0, "count": 0})
        delta = m1["Rss"] - m0["Rss"]
        if abs(delta) > 50:
            grew.append((delta, p, m0, m1))

    grew.sort(key=lambda x: x[0], reverse=True)
    for delta, p, m0, m1 in grew:
        print(f"  delta={delta//1024:>5} MB  Rss={m0['Rss']//1024:>4}->{m1['Rss']//1024:>4} MB  "
              f"Size={m1['Size']//1024:>6} MB  Anon={m1['Anonymous']//1024:>5} MB  "
              f"n={m1['count']:>3}  {p[:55]}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
