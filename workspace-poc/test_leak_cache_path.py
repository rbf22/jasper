#!/usr/bin/env python
"""Test whether the generic_op slow-path cache hit is the RSS leak source.

This test does two things:
1. Calls generic_op repeatedly and measures RSS growth (slow path)
2. Calls equivalent standard ttnn ops repeatedly and measures RSS growth (no generic_op)

If the generic_op path leaks significantly more than the standard ops,
the slow-path ProgramDescriptor copy is the likely cause.
"""

import os
import sys
import gc
import time
import struct
import resource
import pathlib

# Must set env before importing ttnn
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
_device_id = 0
os.environ.setdefault("TT_VISIBLE_DEVICES", str(_device_id))

# Find mesh graph descriptor (same logic as train_ttnn.py)
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

import ttnn
from ttnn._ttnn.program_descriptor import VectorUInt32

def get_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def get_heap_kb():
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        class mallinfo2_entry(ctypes.Structure):
            _fields_ = [("arena", ctypes.c_int64),
                        ("ordblks", ctypes.c_int64),
                        ("smblks", ctypes.c_int64),
                        ("hblks", ctypes.c_int64),
                        ("hblkhd", ctypes.c_int64),
                        ("usmblks", ctypes.c_int64),
                        ("fsmblks", ctypes.c_int64),
                        ("uordblks", ctypes.c_int64),
                        ("fordblks", ctypes.c_int64),
                        ("keepcost", ctypes.c_int64)]
        class mallinfo2_struct(ctypes.Structure):
            _fields_ = [("total", mallinfo2_entry)]
        info = mallinfo2_struct()
        libc.mallinfo2(ctypes.byref(info))
        return info.total.uordblks // 1024  # KB
    except Exception:
        return 0

def get_smaps_anon_kb(pid=None):
    """Get total anonymous mapping size from /proc/pid/smaps."""
    if pid is None:
        pid = os.getpid()
    try:
        total_pss = 0
        total_rss = 0
        with open(f"/proc/{pid}/smaps", "r") as f:
            in_anon = False
            for line in f:
                if line.startswith("Size:"):
                    in_anon = False
                if "Anonymous" in line:
                    # Check if previous mapping header suggests anon
                    pass
                if line.startswith("Pss:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        total_pss += int(parts[1])
                if line.startswith("Rss:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        total_rss += int(parts[1])
        return total_rss, total_pss
    except Exception:
        return 0, 0

def measure_rss(label, n_iters, func, warmup=5):
    """Run func n_iters times and measure RSS growth."""
    # Warmup
    for _ in range(warmup):
        func()

    gc.collect()
    time.sleep(0.5)

    rss_start = get_rss_kb()
    heap_start = get_heap_kb()
    smaps_rss_start, smaps_pss_start = get_smaps_anon_kb()

    for i in range(n_iters):
        func()
        if (i + 1) % 50 == 0:
            gc.collect()

    gc.collect()
    time.sleep(0.5)

    rss_end = get_rss_kb()
    heap_end = get_heap_kb()
    smaps_rss_end, smaps_pss_end = get_smaps_anon_kb()

    rss_delta = rss_end - rss_start
    heap_delta = heap_end - heap_start
    smaps_rss_delta = smaps_rss_end - smaps_rss_start
    smaps_pss_delta = smaps_pss_end - smaps_pss_start

    per_iter_rss = rss_delta / n_iters if n_iters > 0 else 0
    per_iter_heap = heap_delta / n_iters if n_iters > 0 else 0

    print(f"\n=== {label} ({n_iters} iters) ===")
    print(f"  RSS:       {rss_start} -> {rss_end} KB  (delta={rss_delta} KB, {per_iter_rss:.1f} KB/iter)")
    print(f"  Heap:      {heap_start} -> {heap_end} KB  (delta={heap_delta} KB, {per_iter_heap:.1f} KB/iter)")
    print(f"  Smaps RSS: {smaps_rss_start} -> {smaps_rss_end} KB  (delta={smaps_rss_delta} KB)")
    print(f"  Smaps PSS: {smaps_pss_start} -> {smaps_pss_end} KB  (delta={smaps_pss_delta} KB)")

    return per_iter_rss, per_iter_heap


def main():
    device = ttnn.open_device(device_id=0)

    # Test parameters
    B, H, T = 4, 4, 128  # batch, heads, seq_len
    BH = B * H
    N_ITERS = 200

    # Create input tensors (persistent, not recreated each iter)
    scores_raw = ttnn.empty([BH * T, T], dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    D_decay = ttnn.empty([H * T, T], dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)
    scale = 0.125

    # --- Test 1: generic_op with custom kernel (slow path) ---
    _KERNEL_DIR = str(pathlib.Path(__file__).parent / "kernels")

    def run_generic_op():
        out = ttnn.empty([BH * T, T], dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        num_cores_x, num_cores_y = grid.x, grid.y
        num_cores_total = num_cores_x * num_cores_y

        tiled_cols = (T + 31) // 32
        total_tiles = ((BH * T + 31) // 32) * tiled_cols
        HT_tiles = (H * T + 31) // 32
        num_cores = min(total_tiles, num_cores_total)
        bpc_base = total_tiles // num_cores
        bpc_rem = total_tiles % num_cores

        all_cores = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
        ])

        tile_bytes = 32 * 32 * 2
        cb_scores, cb_D = 0, 1
        cb_out = 16

        def _cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * tile_bytes, core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

        cbs = [_cb(cb_scores, 2), _cb(cb_D, 2), _cb(cb_out, 2)]

        reader_ct = []
        reader_ct.extend(ttnn.TensorAccessorArgs(scores_raw).get_compile_time_args())
        reader_ct.extend(ttnn.TensorAccessorArgs(D_decay).get_compile_time_args())
        writer_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

        scale_bits = struct.unpack('I', struct.pack('f', float(scale)))[0]

        reader_rt, writer_rt, compute_rt = [], [], []
        ts = 0
        for i in range(num_cores_total):
            cx, cy = i // num_cores_y, i % num_cores_y
            coord = ttnn.CoreCoord(cx, cy)
            ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
            reader_rt.append((coord, VectorUInt32([
                scores_raw.buffer_address(), D_decay.buffer_address(),
                ntpc, ts, tiled_cols, HT_tiles])))
            writer_rt.append((coord, VectorUInt32([
                out.buffer_address(), ntpc, ts])))
            compute_rt.append((coord, VectorUInt32([ntpc, scale_bits])))
            ts += ntpc

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
            runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
            runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32([]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))

        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                         semaphores=[], cbs=cbs)
        ttnn.generic_op(io_tensors=[scores_raw, D_decay, out],
                        program_descriptor=program)
        ttnn.deallocate(out, force=True)

    # --- Test 2: standard ttnn ops (no generic_op) ---
    # _fused_scale_decay computes: scores = scores_raw * scale * D_decay
    # scores_raw is (BH*T, T), D_decay is (H*T, T)
    # We need to broadcast D_decay over batch B.
    # Reshape D_decay to (1, H*T, T) and broadcast to (B, H*T, T)
    D_decay_3d = ttnn.reshape(D_decay, [1, H * T, T])
    D_decay_broadcast = ttnn.expand(D_decay_3d, [B, H * T, T])
    D_decay_2d = ttnn.reshape(D_decay_broadcast, [BH * T, T])

    def run_standard_ops():
        # scores = scores_raw * scale * D_decay
        scaled = ttnn.mul(scores_raw, scale)
        out = ttnn.mul(scaled, D_decay_2d)
        ttnn.deallocate(scaled, force=True)
        ttnn.deallocate(out, force=True)

    # --- Test 3: generic_op with custom_program_hash set (still slow path, but skips hash walk) ---
    # Pre-compute hash
    pre_hash = None

    def run_generic_op_with_hash():
        nonlocal pre_hash
        out = ttnn.empty([BH * T, T], dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        num_cores_x, num_cores_y = grid.x, grid.y
        num_cores_total = num_cores_x * num_cores_y

        tiled_cols = (T + 31) // 32
        total_tiles = ((BH * T + 31) // 32) * tiled_cols
        HT_tiles = (H * T + 31) // 32
        num_cores = min(total_tiles, num_cores_total)
        bpc_base = total_tiles // num_cores
        bpc_rem = total_tiles % num_cores

        all_cores = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
        ])

        tile_bytes = 32 * 32 * 2
        cb_scores, cb_D = 0, 1
        cb_out = 16

        def _cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * tile_bytes, core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

        cbs = [_cb(cb_scores, 2), _cb(cb_D, 2), _cb(cb_out, 2)]

        reader_ct = []
        reader_ct.extend(ttnn.TensorAccessorArgs(scores_raw).get_compile_time_args())
        reader_ct.extend(ttnn.TensorAccessorArgs(D_decay).get_compile_time_args())
        writer_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

        scale_bits = struct.unpack('I', struct.pack('f', float(scale)))[0]

        reader_rt, writer_rt, compute_rt = [], [], []
        ts = 0
        for i in range(num_cores_total):
            cx, cy = i // num_cores_y, i % num_cores_y
            coord = ttnn.CoreCoord(cx, cy)
            ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
            reader_rt.append((coord, VectorUInt32([
                scores_raw.buffer_address(), D_decay.buffer_address(),
                ntpc, ts, tiled_cols, HT_tiles])))
            writer_rt.append((coord, VectorUInt32([
                out.buffer_address(), ntpc, ts])))
            compute_rt.append((coord, VectorUInt32([ntpc, scale_bits])))
            ts += ntpc

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
            runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
            runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32([]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))

        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                         semaphores=[], cbs=cbs)

        # Set custom hash to skip hash walk
        if pre_hash is None:
            pre_hash = ttnn.compute_program_descriptor_hash(program)
        program.custom_program_hash = pre_hash

        ttnn.generic_op(io_tensors=[scores_raw, D_decay, out],
                        program_descriptor=program)
        ttnn.deallocate(out, force=True)

    # --- Run tests ---
    print(f"Device program cache entries: {device.num_program_cache_entries()}")
    print(f"Test parameters: B={B}, H={H}, T={T}, iters={N_ITERS}")

    # Test standard ops first (baseline)
    rss_std, heap_std = measure_rss("Standard ttnn ops (no generic_op)", N_ITERS, run_standard_ops)
    print(f"  Cache entries after: {device.num_program_cache_entries()}")

    # Clear program cache between tests
    device.clear_program_cache()
    gc.collect()
    time.sleep(1)

    # Test generic_op (slow path)
    rss_gen, heap_gen = measure_rss("generic_op (slow path, no buffer_bindings)", N_ITERS, run_generic_op)
    print(f"  Cache entries after: {device.num_program_cache_entries()}")

    # Clear program cache between tests
    device.clear_program_cache()
    gc.collect()
    time.sleep(1)

    # Test generic_op with custom hash (still slow path)
    rss_hash, heap_hash = measure_rss("generic_op with custom_program_hash (slow path)", N_ITERS, run_generic_op_with_hash)
    print(f"  Cache entries after: {device.num_program_cache_entries()}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Standard ops:       {rss_std:.1f} KB/iter RSS, {heap_std:.1f} KB/iter heap")
    print(f"generic_op:         {rss_gen:.1f} KB/iter RSS, {heap_gen:.1f} KB/iter heap")
    print(f"generic_op+hash:    {rss_hash:.1f} KB/iter RSS, {heap_hash:.1f} KB/iter heap")
    print(f"\nDelta (generic - standard): {rss_gen - rss_std:.1f} KB/iter RSS")

    ttnn.deallocate(D_decay_2d, force=True)
    ttnn.deallocate(D_decay_broadcast, force=True)
    ttnn.deallocate(D_decay_3d, force=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
