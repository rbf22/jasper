#!/usr/bin/env python
"""Use strace to track mmap/munmap calls and find the net mmap growth source."""
import gc
import os
import sys
import time
import subprocess
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


def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0


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

    # Now attach strace to ourselves to track mmap/munmap for exactly 10 steps
    pid = os.getpid()
    print(f"Attaching strace to PID {pid} for 10 training steps...")

    # Start strace in background, tracking mmap/munmap/mprotect/brk
    strace_log = "/tmp/strace_mmap.log"
    strace_proc = subprocess.Popen(
        ["strace", "-f", "-p", str(pid),
         "-e", "trace=mmap,munmap,mprotect,brk,mremap",
         "-o", strace_log],
        stderr=subprocess.PIPE)

    # Wait for strace to attach
    time.sleep(2)

    rss_before = get_rss_kb()
    print(f"RSS before 10 steps: {rss_before//1024} MB")

    # Run exactly 10 steps
    for step in range(10):
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
    time.sleep(1)

    rss_after = get_rss_kb()
    print(f"RSS after 10 steps: {rss_after//1024} MB  (delta={(rss_after-rss_before)//1024} MB)")

    # Stop strace
    strace_proc.terminate()
    strace_proc.wait(timeout=5)

    # Parse strace log
    print(f"\nParsing strace log: {strace_log}")

    mmap_count = 0
    munmap_count = 0
    brk_count = 0
    mprotect_count = 0
    mremap_count = 0

    # Track mmap sizes (only anonymous ones)
    anon_mmap_sizes = []  # sizes of MAP_ANONYMOUS mmaps
    anon_mmap_total = 0
    file_mmap_total = 0
    munmap_total = 0

    mmap_re = re.compile(r'mmap\((.*?),\s*(\d+),\s*(\w+),\s*(\w+),')
    munmap_re = re.compile(r'munmap\((\w+),\s*(\d+)\)\s*=\s*(\d+)')

    with open(strace_log) as f:
        for line in f:
            if 'mmap(' in line and 'munmap' not in line:
                mmap_count += 1
                m = mmap_re.search(line)
                if m:
                    size = int(m.group(2))
                    flags = m.group(4)
                    if 'MAP_ANONYMOUS' in flags:
                        anon_mmap_total += size
                        anon_mmap_sizes.append(size)
                    else:
                        file_mmap_total += size
            elif 'munmap(' in line:
                munmap_count += 1
                m = munmap_re.search(line)
                if m:
                    size = int(m.group(2))
                    munmap_total += size
            elif 'brk(' in line:
                brk_count += 1
            elif 'mprotect(' in line:
                mprotect_count += 1
            elif 'mremap(' in line:
                mremap_count += 1

    net_anon = anon_mmap_total - munmap_total
    print(f"\n=== strace summary (10 steps) ===")
    print(f"  mmap calls:   {mmap_count}")
    print(f"  munmap calls: {munmap_count}")
    print(f"  brk calls:    {brk_count}")
    print(f"  mprotect:     {mprotect_count}")
    print(f"  mremap:       {mremap_count}")
    print(f"  anon mmap total:  {anon_mmap_total//1024//1024:.1f} MB")
    print(f"  file mmap total:  {file_mmap_total//1024//1024:.1f} MB")
    print(f"  munmap total:     {munmap_total//1024//1024:.1f} MB")
    print(f"  NET (anon - unmap): {net_anon//1024:.0f} KB  ({net_anon/1024/1024:.1f} MB)")
    print(f"  RSS delta:        {(rss_after-rss_before):.0f} KB  ({(rss_after-rss_before)/1024:.1f} MB)")

    # Show largest anon mmaps
    if anon_mmap_sizes:
        anon_mmap_sizes.sort(reverse=True)
        print(f"\n  Largest 20 anon mmap sizes (KB):")
        for s in anon_mmap_sizes[:20]:
            print(f"    {s//1024:>8} KB")

    # Show size distribution
    if anon_mmap_sizes:
        buckets = {}
        for s in anon_mmap_sizes:
            bucket = s // 1024 if s < 1024*1024 else (s // (1024*1024)) * 1024
            buckets[bucket] = buckets.get(bucket, 0) + 1
        print(f"\n  Anon mmap size distribution (KB bucket -> count):")
        for bucket in sorted(buckets.keys()):
            print(f"    {bucket:>8} KB: {buckets[bucket]:>4} calls")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
