#!/usr/bin/env python
"""Longer test: does anonymous mapping growth plateau or continue indefinitely?
Track mapping count, virtual size, and RSS over 100 steps."""
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

def get_anon_stats():
    """Count anonymous mappings and their total Size/Rss from smaps."""
    count = 0
    total_size = 0
    total_rss = 0
    total_anon = 0
    current = None
    with open('/proc/self/smaps') as f:
        for line in f:
            m = HEADER_RE.match(line)
            if m:
                if current:
                    parts = line.strip().split(None, 5)
                    path = parts[5] if len(parts) > 5 else "[anon]"
                    if path == "[anon]":
                        count += 1
                        total_size += current.get("Size", 0)
                        total_rss += current.get("Rss", 0)
                        total_anon += current.get("Anonymous", 0)
                parts = line.strip().split(None, 5)
                path = parts[5] if len(parts) > 5 else "[anon]"
                current = {"path": path, "Size": 0, "Rss": 0, "Anonymous": 0}
            elif current and ':' in line:
                key, _, val = line.strip().partition(':')
                try: current[key] = int(val.split()[0])
                except: pass
    # Last entry
    if current and current.get("path") == "[anon]":
        count += 1
        total_size += current.get("Size", 0)
        total_rss += current.get("Rss", 0)
        total_anon += current.get("Anonymous", 0)
    return count, total_size, total_rss, total_anon


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

    print(f"{'step':>5}  {'RSS_MB':>7}  {'d_RSS':>7}  {'anon_n':>7}  {'anon_vsize_MB':>14}  "
          f"{'anon_rss_MB':>12}  {'anon_anon_MB':>13}  {'cache':>5}")

    rss0 = get_rss_kb()
    n0, vs0, ar0, aa0 = get_anon_stats()
    cache0 = device.num_program_cache_entries()
    print(f"{'init':>5}  {rss0//1024:>7}  {'':>7}  {n0:>7}  {vs0//1024:>14}  "
          f"{ar0//1024:>12}  {aa0//1024:>13}  {cache0:>5}")

    prev_rss = rss0
    for step in range(100):
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

        if step % 10 == 0 or step == 99:
            rss = get_rss_kb()
            n, vs, ar, aa = get_anon_stats()
            cache = device.num_program_cache_entries()
            d_rss = rss - prev_rss
            print(f"{step:>5}  {rss//1024:>7}  {d_rss//1024:>7}  {n:>7}  {vs//1024:>14}  "
                  f"{ar//1024:>12}  {aa//1024:>13}  {cache:>5}")
            prev_rss = rss

    # Final summary
    rss_final = get_rss_kb()
    n_final, vs_final, ar_final, aa_final = get_anon_stats()
    total_growth = rss_final - rss0
    print(f"\n=== Summary (100 steps) ===")
    print(f"  RSS:     {rss0//1024} -> {rss_final//1024} MB  ({total_growth//1024} MB, {total_growth/100:.0f} KB/step)")
    print(f"  Anon n:  {n0} -> {n_final}  (+{n_final - n0})")
    print(f"  Anon VS: {vs0//1024} -> {vs_final//1024} MB  (+{(vs_final-vs0)//1024} MB)")
    print(f"  Anon RSS:{ar0//1024} -> {ar_final//1024} MB  (+{(ar_final-ar0)//1024} MB)")
    print(f"  Anon Anon:{aa0//1024} -> {aa_final//1024} MB  (+{(aa_final-aa0)//1024} MB)")
    print(f"  Cache:   {cache0} -> {device.num_program_cache_entries()}")

    if n_final == n0:
        print(f"\n  >> No new mappings created. Growth is from page faults in existing mappings.")
        print(f"  >> Virtual size grew {(vs_final-vs0)//1024} MB — this is from existing mappings expanding or new ones.")
        print(f"  >> If VS is stable, growth will plateau when all pages are touched.")
    else:
        print(f"\n  >> {n_final - n0} new mappings created. This indicates active allocation, not just page faults.")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
