# Restart Instructions: TT-Metal Patches for WRAP Memory Leak + Scalar Fix

## Context

We are stabilizing Tenstorrent production training for the WRAP model.
There are three TT-metal native patches that need to be applied to the
tt-metal source tree, built, and installed into the pip package. Two of
them were previously working (no memory leak). The third (scalar fix) is
new and has been written but the buffer_bindings portion of patch 2 was
lost and needs to be reconstructed from the chat histories.

## Current State (as of this session)

### tt-metal source tree
- Location: `/home/rfenwick/Documents/tt-metal-src/`
- Git commit: `7f6364a11dafadf141b6c87358073d9e3d1dd22f` (clean upstream, never committed)
- Build dir: `/home/rfenwick/Documents/tt-metal-src/build_Release/`
- Working tree has 3 patches applied BUT is missing the buffer_bindings
  part of patch 2 (see below). The current working tree uses the slow
  path (descriptor rebuild) for binary_ng cache hits, which works but
  does NOT fix the per-cache-hit allocation leak.

### Jasper repo
- Location: `/home/rfenwick/Documents/jasper/`
- Git HEAD: `23fe558` (last committed state with leak patches + WRAP renames)
- Working tree has uncommitted changes: WRAP renames, optimizer scalar
  fix (tensor-tensor mul workaround in train_ttnn.py), and user's
  deallocation changes in model_ttnn.py
- These uncommitted changes are NOT lost — they are in the working tree.

### Pip package
- Location: `/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/lib64/`
- `.bak` files are the ORIGINAL unpatched pip binaries (from Aug 12 08:46)
- Currently installed `.so` files are from the current build (no buffer_bindings)

### Chat histories with full patch details
- `/home/rfenwick/.local/share/devin/cli/summaries/history_0b98ea0b8cb14e8d.md` (most recent)
- `/home/rfenwick/.local/share/devin/cli/summaries/history_00448cefe58248c4.md` (earlier, has more detail on buffer_bindings)
- `/home/rfenwick/Documents/jasper/workspace-poc/AGENTS.md` (lines 1143-1346 document the patches)

## The Three Patches

### Patch 1: DataCollector flat vector (DONE, working)

Replace `std::unordered_map<uint16_t, uint64_t> runtime_id_to_program_id_`
with a flat `std::vector<uint64_t>` + `std::vector<bool>` for validity.

Files:
- `tt_metal/impl/dispatch/data_collector.hpp` — replace the map with vectors
- `tt_metal/impl/dispatch/data_collector.cpp` — lazy resize + direct index

This is already in the current working tree and is correct. No changes needed.

### Patch 2: binary_ng cache hash + buffer_bindings (PARTIALLY DONE)

**Part 2a: Hash includes tensor shapes (DONE, working)**

`compute_program_hash` in `binary_ng_device_operation.cpp` — the scalar
path now includes `input_tensor_a.logical_shape()` and `shard_volumes`
in the hash. This is already in the current working tree and is correct.

**Part 2b: buffer_bindings for cache-hit fast path (MISSING — needs reconstruction)**

This is the part that was lost. The AGENTS.md (lines 1220-1224) describes it:

> Registered buffer address bindings for reader (arg[0] = a, arg[7]/arg[15] = b)
> and writer (arg[0] or arg[1] = c) kernels. On cache hits with matching
> shapes, `apply_resolved_bindings()` patches only the buffer addresses,
> avoiding a full `create_descriptor()` call and its ~7 KB allocation.

The approach uses `KernelDescriptor::emplace_runtime_args()` with `Buffer*`
instead of `runtime_args.emplace_back()` with `buffer->address()` (uint32_t).
When `Buffer*` is pushed, the framework auto-registers a BufferBinding at
that position. On cache hits, `apply_resolved_bindings()` patches just the
buffer addresses, avoiding the full descriptor rebuild.

The buffer address positions in `binary_ng_program_factory.cpp` are:

**Reader (kernel index 0):**
- arg[0] = a.buffer()->address() (always)
- arg[7] = b.buffer()->address() (row_major, if b.has_value())
- arg[15] = b.buffer()->address() (non-row_major, if b.has_value())

**Writer (kernel index 1):**
- tensor_b case: arg[0] = c.buffer()->address() (both row_major and non-row_major)
- scalar case: arg[0] = c.buffer()->address() (row_major)
- scalar case: arg[1] = c.buffer()->address() (non-row_major, arg[0] is packed_scalar)

To implement: replace the `writer_desc.runtime_args.emplace_back(...)` and
`reader_desc.runtime_args.emplace_back(...)` calls with `emplace_runtime_args()`
using `KernelDescriptor::RTArgList`, pushing `Buffer*` at the buffer address
positions and `uint32_t` everywhere else.

IMPORTANT: The previous session tried this and got a hang. The AGENTS.md
says this was because of cache hash collisions (different shapes, same hash).
With the hash fix (part 2a) now in place, cache hits only occur when shapes
match, so the buffer_bindings should work. But this needs to be verified
carefully — if it hangs, debug WHY before reverting. The hang was previously
caused by reader and writer disagreeing on tile counts due to stale shape
args from a hash collision, NOT by the buffer_bindings mechanism itself.

The chat history `history_00448cefe58248c4.md` likely has the exact
buffer_bindings code that was working. Search for `emplace_runtime_args`
and `RTArgList` in that file.

### Patch 3: get_dynamic_runtime_args for scalar re-application (DONE, working)

This is new (not in the original patches). It fixes the
`ttnn.mul(tensor, scalar)` correctness bug where the scalar value was
frozen from the first cache miss.

Added `get_dynamic_runtime_args` to `BinaryNgDeviceOperation` that
re-applies the packed scalar on cache hits. The implementation is in
`binary_ng_program_factory.cpp` and the declaration is in
`binary_ng_device_operation.hpp`.

This is already in the current working tree and is correct. The scalar
mul test passes (0.9*0.1=0.09, 0.9*0.5=0.45, 0.9*2.0=1.8, 0.9*10.0=9.0).

## Build and Install

```bash
# Build (from tt-metal-src/build_Release/)
CMAKE=/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/cmake/data/bin/cmake
cd /home/rfenwick/Documents/tt-metal-src/build_Release
$CMAKE --build . -j$(nproc)

# Install (copy .so files to pip package)
PKG=/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt
cp build_Release/tt_metal/libtt_metal.so $PKG/lib64/libtt_metal.so
cp build_Release/ttnn/_ttnn.so $PKG/lib64/_ttnn.so
cp build_Release/ttnn/_ttnn.so $PKG/tt-metal/ttnn/ttnn/_ttnn.so
cp build_Release/ttnn/_ttnncpp.so $PKG/lib64/_ttnncpp.so

# Reset device after install
~/.tenstorrent-venv/bin/tt-smi -r
```

Note: The unity build caches aggressively. If a source file change is not
picked up, delete the `.o` file first:
```bash
rm -f build_Release/ttnn/cpp/ttnn/operations/eltwise/binary_ng/CMakeFiles/ttnn_op_eltwise_binary_ng.dir/Unity/unity_0_cxx.cxx.o
```

## Verification

### 1. Scalar mul correctness
```bash
cd /home/rfenwick/Documents/jasper/workspace-poc
P150=$PKG/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=$P150 .tt-venv/bin/python -c "
import os, torch, ttnn
os.environ.setdefault('TT_METAL_LOGGER_LEVEL', 'ERROR')
device = ttnn.open_device(device_id=0)
t = ttnn.from_torch(torch.tensor([[0.9]], dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
for s in [0.1, 0.5, 2.0, 10.0]:
    r = ttnn.mul(t, s)
    print(f'  mul(0.9, {s}) = {ttnn.to_torch(r).float().item():.6f}')
ttnn.close_device(device)
" 2>&1 | grep -viE "nanobind|leaked"
```
Expected: 0.09, 0.45, 1.8, 9.0

### 2. No hang (model leak test)
```bash
TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=$P150 .tt-venv/bin/python test_model_leak.py 2>&1 | grep -viE "nanobind|leaked"
```
Expected: completes without hanging. Forward ~+17MB, backward ~+52MB over 200 iters
(allocator fragmentation, not a sustained leak — should plateau over longer runs).

### 3. Training stability
```bash
TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=$P150 .tt-venv/bin/python test_training_stability.py 2>&1 | grep -viE "nanobind|leaked"
```
Expected: 50 steps, loss ~4.9, grad norm ~1.4, no divergence.

### 4. CPU tests
```bash
cd /home/rfenwick/Documents/jasper/workspace-poc
/home/rfenwick/Documents/jasper/.tt-venv/bin/pytest test_data.py test_text_data.py test_gate_clamp.py test_ws_qknorm_gradcheck.py test_gradients.py test_config.py test_gradient_accum.py -v
```
Expected: 85 passed.

## What To Do

1. **First**: Commit the current jasper repo working tree (WRAP renames +
   optimizer fix + deallocation changes) so nothing is lost.

2. **Then**: Search `history_00448cefe58248c4.md` for the buffer_bindings
   code that was working. Look for `emplace_runtime_args`, `RTArgList`,
   `Buffer*`, `buffer_bindings` in that file.

3. **Apply** the buffer_bindings to `binary_ng_program_factory.cpp`,
   replacing the `runtime_args.emplace_back()` calls for reader and writer
   with `emplace_runtime_args()` using `RTArgList` with `Buffer*` at the
   buffer address positions.

4. **Build, install, and test** — if it hangs, debug the hang. The hash
   fix should prevent the stale-shape issue that caused the previous hang.

5. **Commit the tt-metal source** after each successful patch so we have
   git history. The tt-metal-src repo has no commits — create a branch
   and commit after each patch works.

6. **Run all verification tests** (scalar mul, no hang, training stability,
   CPU tests).

7. **Then launch production training** for all 3 cells.

## Key Paths

- tt-metal source: `/home/rfenwick/Documents/tt-metal-src/`
- tt-metal build: `/home/rfenwick/Documents/tt-metal-src/build_Release/`
- jasper repo: `/home/rfenwick/Documents/jasper/`
- model code: `/home/rfenwick/Documents/jasper/workspace-poc/model_ttnn.py`
- training code: `/home/rfenwick/Documents/jasper/workspace-poc/train_ttnn.py`
- AGENTS.md: `/home/rfenwick/Documents/jasper/workspace-poc/AGENTS.md`
- pip package: `/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/`
- python: `/home/rfenwick/Documents/jasper/.tt-venv/bin/python`
- device reset: `~/.tenstorrent-venv/bin/tt-smi -r`
- chat history (recent): `/home/rfenwick/.local/share/devin/cli/summaries/history_0b98ea0b8cb14e8d.md`
- chat history (earlier, has buffer_bindings detail): `/home/rfenwick/.local/share/devin/cli/summaries/history_00448cefe58248c4.md`

## Git Discipline

- Commit the jasper repo working tree BEFORE doing anything else.
- Create a branch in tt-metal-src and commit after each successful patch.
- Commit after each successful build+test.
- Do NOT leave working changes uncommitted.
