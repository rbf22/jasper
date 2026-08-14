#!/usr/bin/env python
"""Test if ttnn.generic_op leaks memory."""
import gc, os, sys
import torch, ttnn

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import _KERNEL_DIR
from ttnn._ttnn.program_descriptor import VectorUInt32

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

device = ttnn.open_device(device_id=0)

# Create input tensors
x1 = ttnn.from_torch(torch.ones([8*4*128, 192], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
x2 = ttnn.from_torch(torch.ones([8*4*128, 192], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
cos = ttnn.from_torch(torch.ones([8*4*128, 192], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
sin = ttnn.from_torch(torch.ones([8*4*128, 192], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

BH_T = 8 * 4 * 128
d_half = 192
tiled_cols = (d_half + 31) // 32
total_tiles = ((BH_T + 31) // 32) * tiled_cols

grid = device.compute_with_storage_grid_size()
num_cores_x, num_cores_y = grid.x, grid.y
num_cores_total = num_cores_x * num_cores_y
num_cores = min(total_tiles, num_cores_total)
bpc_base = total_tiles // num_cores
bpc_rem = total_tiles % num_cores

all_cores = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                  ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
])

tile_bytes = 32 * 32 * 2

def _cb(idx, n):
    return ttnn.CBDescriptor(
        total_size=n * tile_bytes, core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

cbs = [_cb(0, 2), _cb(1, 2), _cb(2, 2), _cb(3, 2), _cb(16, 2), _cb(17, 2)]

reader_ct = []
for t in [x1, x2, cos, sin]:
    reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

# Warmup
for _ in range(3):
    out0 = ttnn.empty([BH_T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out1 = ttnn.empty([BH_T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    writer_ct = []
    for t in [out0, out1]:
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    
    reader_rt, writer_rt, compute_rt = [], [], []
    ts = 0
    for i in range(num_cores_total):
        cx, cy = i // num_cores_y, i % num_cores_y
        coord = ttnn.CoreCoord(cx, cy)
        ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
        reader_rt.append((coord, VectorUInt32([
            x1.buffer_address(), x2.buffer_address(),
            cos.buffer_address(), sin.buffer_address(),
            ntpc, ts, 128//32, tiled_cols])))
        writer_rt.append((coord, VectorUInt32([
            out0.buffer_address(), out1.buffer_address(),
            ntpc, ts])))
        compute_rt.append((coord, VectorUInt32([ntpc])))
        ts += ntpc

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/rope4d_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
        runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/rope4d_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
        runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/rope4d_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=VectorUInt32([]),
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op(io_tensors=[x1, x2, cos, sin, out0, out1], program_descriptor=program)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out0)
    ttnn.deallocate(out1)

gc.collect()
rss0 = get_rss_kb()
print(f"Baseline RSS: {rss0//1024} MB")
prev = rss0

for step in range(200):
    out0 = ttnn.empty([BH_T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out1 = ttnn.empty([BH_T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    writer_ct = []
    for t in [out0, out1]:
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    
    reader_rt, writer_rt, compute_rt = [], [], []
    ts = 0
    for i in range(num_cores_total):
        cx, cy = i // num_cores_y, i % num_cores_y
        coord = ttnn.CoreCoord(cx, cy)
        ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
        reader_rt.append((coord, VectorUInt32([
            x1.buffer_address(), x2.buffer_address(),
            cos.buffer_address(), sin.buffer_address(),
            ntpc, ts, 128//32, tiled_cols])))
        writer_rt.append((coord, VectorUInt32([
            out0.buffer_address(), out1.buffer_address(),
            ntpc, ts])))
        compute_rt.append((coord, VectorUInt32([ntpc])))
        ts += ntpc

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/rope4d_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
        runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/rope4d_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
        runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_DIR}/rope4d_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=VectorUInt32([]),
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op(io_tensors=[x1, x2, cos, sin, out0, out1], program_descriptor=program)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out0)
    ttnn.deallocate(out1)

    if step % 50 == 49:
        gc.collect()
        rss = get_rss_kb()
        print(f"  step {step:>4}: RSS={rss//1024:>5} MB, delta={rss-prev:>6} KB")
        prev = rss

final = get_rss_kb()
print(f"\nTotal: {(final-rss0)//1024} MB / 200 steps = {(final-rss0)/200:.1f} KB/step")
ttnn.close_device(device)
