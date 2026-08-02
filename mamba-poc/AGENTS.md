# Project notes

## Environment

Python venv (has `torch` + `ttnn`, Tenstorrent hardware present):

```
/home/rfenwick/Documents/jasper/.tt-venv/bin/python
```

`ttnn` prints a large nanobind "leaked instance" dump at interpreter exit. This
is a teardown artifact, not a failure — filter it with `2>/dev/null` or
`grep -viE "nanobind|leaked"`.

## Verifying the Mamba-3 layer

```bash
/home/rfenwick/Documents/jasper/.tt-venv/bin/python test_mamba3_parity.py --gradcheck
/home/rfenwick/Documents/jasper/.tt-venv/bin/python test_mamba3_parity.py --tile-aligned
```

Three stages:

- **Stage 0** (`--gradcheck`): float64 `torch.autograd.gradcheck` of
  `mamba3_reference.py`, plus an explicit causality assertion. Validates the
  reference that everything else is measured against.
- **Stage 1**: tt-nn forward vs. reference forward. Catches tt-nn
  tile/broadcast/reshape issues.
- **Stage 2**: `TTMamba3Layer.backward` vs. reference autograd, per parameter.
  Catches manual-gradient math bugs.

`--tile-aligned` uses shapes >= the 32x32 tile size. Sub-tile trailing dims are
where tt-nn most often returns silently padded results, so if a sub-tile config
fails but the tile-aligned one passes, suspect a ttnn op rather than the math.

### Do not gradient-check this layer with finite differences

`test_mamba3_grad.py` (the older test) uses central differences with `eps=2.0`
against a bf16 loss sum. It cannot work:

- bf16 carries ~3 decimal digits, so any eps small enough to be a derivative is
  buried in noise, and `eps=2.0` is measuring a secant across a wildly
  nonlinear region (`exp`/`tanh`/`sigmoid`/`softplus`/`cumsum`).
- FD validates the backward against the layer's *own* forward, so a forward bug
  hides itself. The non-causal decay matrix was invisible to it.

Use `test_mamba3_parity.py` instead.

### Expected error levels

Everything should sit at ~0.02 relative error (bf16 vs fp32 noise floor). A
real math bug shows up as O(1), not O(0.05).

`grad_A_log` is the exception: it is precision-limited, not math-limited, and
grows with T (0.017 @T=16 → 0.085 @T=64). It is the only gradient depending
solely on the decay-matrix path, whose backward takes `row_sum - col_sum` of a
TxT matrix — a difference much smaller than either operand, amplifying upstream
bf16 error by ~O(T). Two mitigations are applied: (1) the decay matrix and its
backward reduction run in fp32, and (2) the two matmuls that feed
`grad_L_accum` (`QK`, `grad_QK`) use `compute_kernel_config` with
`fp32_dest_acc_en=True` — fp32 accumulation in the matmul's destination
register while Q/K/V stay bf16 in memory, no extra DRAM traffic. This is safe
on Blackhole (the Wormhole fp32_dest_acc_en rounding erratum is fixed there —
see `ttnn.matmul`'s docstring). Together these took T=64 from 0.43 (no
mitigation) to 0.14 (fp32 reduction only) to 0.085 (both). See
`test_mamba3_parity.py`'s `PARAM_TOL` comment for the full measurement table.

Broadening `fp32_dest_acc_en` to the other matmuls in the R² attention loop
(`out_rq` accumulation, `grad_V_rk`, `grad_Q_rq`, `grad_K_rk`) was tried and
measured to give **no further improvement anywhere** — `fp32_dest_acc_en` only
raises precision of the accumulator, not of the bf16-quantized Q/K/V tiles
being read, so it only helps ops that are themselves cancellation-sensitive
(the decay-matrix reduction is; ordinary attention matmuls aren't). Don't
re-add it elsewhere without a new measurement justifying the extra HiFi4
compute cost.

If you need `grad_A_log` fully precision-matched (e.g. for a stricter test),
the remaining error is bf16 storage quantization of Q/K/V, not accumulation —
the only way to remove it is to store those specific tensors in fp32, which
doubles their DRAM footprint. Not done here because it wasn't needed: `A_log`
is a single scalar per head, not a large tensor to update precisely, and
`BWD_TOL` on every other parameter is already met.

### Testing gotcha: randomise the parameters

`TTMamba3Layer.__init__` used to set `MIMO_V/Z/O` to constants across the R
axis. Since every other tensor they multiply (V, z) is broadcast identically
across R, that made every MIMO rank numerically identical at init, and that
symmetry is a fixed point of gradient descent — it can never break on its own.
This hid rank-indexing bugs from testing AND was a standing capacity bug in
training (the R "parallel SSMs" permanently degenerated to one effective rank).
Fixed: init now adds small per-rank noise (2% relative) so ranks specialize.

Note: this is unrelated to the former `run_*_v1_degenerate` directories (now
deleted), which were about workspace (Cell B/C) read/write attention collapsing
to uniform — a different, older ablation project using `model.py`'s Mamba-2,
not this layer.

The parity test additionally overrides all parameters with independent random
values (`randomised_params`) regardless of the production init, to maximize
the chance of catching index/broadcast bugs. Keep it that way.

### Known limitation

`ngroups > 1` is not implemented — `ttnn.expand` can only broadcast size-1
dims, so the GQA expansion in `forward()` only works for `ngroups == 1`. There
is an assert in `__init__`.

## Production integration

`TTMambaWorkspaceModel` (in `model_ttnn.py`) uses `TTMamba3Layer` unconditionally
for every non-attention layer, for all cells (A/B/C) and all `configs/cell_*_tt.yaml`
runs. `run_A/`, `run_B/`, `run_C/` symlink `mamba3_layer.py` and `model_ttnn.py`
back to the repo root, so fixes here apply immediately to those run directories —
there is only one copy of the layer in play for current runs.

Smoke-tested end-to-end after the fixes above (3 steps each, `--accum_steps 1`
to skip the full gradient-accumulation loop):

```bash
.tt-venv/bin/python train_ttnn.py --config configs/cell_a_tt.yaml --steps 3 --micro_batch 4 --accum_steps 1 --checkpoint_dir /tmp/x --device 0
.tt-venv/bin/python train_ttnn.py --config configs/cell_b_tt.yaml --steps 3 --micro_batch 4 --accum_steps 1 --checkpoint_dir /tmp/x --device 0
.tt-venv/bin/python train_ttnn.py --config configs/cell_c_tt.yaml --steps 3 --micro_batch 4 --accum_steps 1 --checkpoint_dir /tmp/x --device 0
```

All three ran forward + backward + AdamW step + checkpoint save cleanly with
finite loss and grad norm. This is only a smoke test (3 steps, near-zero LR
during warmup) — it proves the layer doesn't crash or NaN, not that training
converges. Watch the first real run for loss trajectory and grad norm blowups
now that `MIMO_V/Z/O` are randomly initialized (slightly different dynamics
than the old constant init).

If a device errors with "Sysmem mapped at unexpected NOC address" or similar
after a killed/canceled process, reset with:

```bash
~/.tenstorrent-venv/bin/tt-smi -r
```

## Current training run (2026-08-02)

All three cells launched in parallel on devices 0, 1, 2 via `nohup` (safe to
log out). Logs in `logs/cell_{a,b,c}.log`, checkpoints in `checkpoints/`.

```bash
# Launch (from repo root):
TT_VISIBLE_DEVICES=0 nohup .tt-venv/bin/python train_ttnn.py \
    --config configs/cell_a_tt.yaml --device 0 --checkpoint_dir checkpoints \
    > logs/cell_a.log 2>&1 &
# Repeat for B (device 1) and C (device 2)
```

### Early stopping

Configs set `plateau_patience: 1000` (10% of max_steps) and
`plateau_min_delta: 1.0e-3` (0.1% relative improvement). The EMA (beta=0.99)
smooths over ~100 steps × 48 micro-batches = ~4800 samples, making it very
stable. The old defaults (patience=500, min_delta=1e-4) were below the EMA
noise floor — noise could reset the patience counter and early stopping
would never trigger. The new values ensure only genuine improvements reset
the counter.

`checkpoint_interval` was increased from 25 to 100 — saving every 25 steps
with 48 accum steps caused frequent I/O pauses.

### Monitoring

`monitor_training.sh` runs every 10 minutes, writing to `logs/monitor.log`.
Reports per cell: process status, recent training steps, latest checkpoint,
and any errors detected.

```bash
# Start monitor:
nohup ./monitor_training.sh > logs/monitor.log 2>&1 &

# Check status:
tail -50 logs/monitor.log

# Check individual cells:
tail -20 logs/cell_a.log
ps aux | grep train_ttnn | grep -v grep
```

### Measured step times (d_model=384, n_heads=4, T=128, micro_batch=8, accum_steps=48)

| Cell | Device | Time/step | Est. total (10000 steps) |
|---|---|---|---|
| A | 0 | ~4.1s | ~11.4 hours |
| B | 1 | ~4.2s | ~11.7 hours |
| C | 2 | ~11.3s | ~31.3 hours |

Cell C is ~2.7x slower due to the recurrent core (K_max=6 iterations per step).

## Deprecated Mamba-2 checkpoints (removed)

`checkpoints/*.pt` (pure-PyTorch Mamba-2 ablation cells, from `model.py`/
`train.py`) and `run_E_v2_mamba2/` (a Mamba-2 fallback checkpoint dump) have
been deleted — Mamba-2 is deprecated in favor of the fixed Mamba-3 layer.

## Cell rename (2026-07-31)

Cells were renamed after the pure-Mamba2 (old A) and Mamba2+attention at
lr=6e-4 (old B) cells were deprecated and removed. Mapping: old E→A,
old C→B, old D→C. Run directories renamed: `run_E`→`run_A`, `run_C`→`run_B`,
`run_D`→`run_C`. PID files renamed accordingly. Old `run_C_v1_degenerate/`
and `run_D_v1_degenerate/` directories deleted. Config files renamed:
`cell_e_tt.yaml`→`cell_a_tt.yaml`, `cell_c_tt.yaml`→`cell_b_tt.yaml`,
`cell_d_tt.yaml`→`cell_c_tt.yaml`. Non-TT config variants (`cell_c.yaml`,
`cell_d.yaml`, `cell_c_colab.yaml`, `cell_d_colab.yaml`, `cell_d_vast.yaml`)
deleted. GPU runners (`vast_runner.py`, `colab_runner.py`, `colab_runner_ac.py`)
marked deprecated — project moved to tt-nn on Blackhole.

## Custom fused kernels (TTRetentionLayer)

`TTRetentionLayer` in `model_ttnn.py` uses custom tt-metal kernels via
`ttnn.generic_op` to fuse element-wise operations and eliminate intermediate
DRAM writes. Kernel source files are in `kernels/`.

### Fused scale + decay (`_fused_scale_decay`)

Computes `scores = qk * scale * D_decay` in one kernel pass (FPU `mul_tiles`
for `qk * D`, then SFPU `mul_unary_tile` for `* scale`). Replaces 2 `ttnn.mul`
ops with 1 kernel. Used in both forward and backward (backward computes
`grad_qk = grad_scores * scale * D` using the same kernel).

- Kernel files: `kernels/scale_decay_{reader,compute,writer}.cpp`
- Test: `test_scale_decay.py` (standalone kernel test)
- Parity: `test_retention_parity.py --tile-aligned` (forward + backward)
- The kernel requires 2D inputs `(BH*T, T)` — 4D tensors from `ttnn.matmul`
  must be reshaped to 2D before calling `_fused_scale_decay`, then reshaped
  back. Passing 4D tensors directly triggers a `TT_FATAL` rank assertion
  when B=1.
- The compute kernel includes a dummy `mul_tiles` to register 0 as an FPU
  warm-up (Blackhole FPU init issue — first computed output is garbage
  without it). Real result goes to register 1.
- Cache stores `qk` (QK^T before scale) instead of the old `scores_raw`
  (QK^T * scale). Backward recomputes `qk * scale` for `grad_D_decay`.

### Fused RoPE (`_fused_rope_kernel` / `_fused_rope_4d`)

Computes the full RoPE rotation in a single kernel pass:
  `rot1 = x1*cos - x2*sin`
  `rot2 = x1*sin + x2*cos`
using 4 FPU `mul_tiles` (to dest regs 1-4) + 2 SFPU binary ops
(`sub_binary_tile`, `add_binary_tile`) on dest registers. Replaces the
previous 2-pass approach (2 kernel launches + 2 `synchronize_device` + 1
`ttnn.sub` + 1 `ttnn.add` = 6 ops) with 1 kernel launch = 1 op.

- Kernel files: `kernels/rope4d_{reader,compute,writer}.cpp`
- Test: `test_fused_rope_single.py` (standalone kernel test, fwd + bwd)
- Parity: `test_retention_parity.py --tile-aligned` (forward + backward)
- The compute kernel includes a dummy `mul_tiles` to register 0 as an FPU
  warm-up (same Blackhole FPU init issue as scale_decay). Real results go
  to registers 1-4, then SFPU ops write rot1→reg0, rot2→reg1.
- `mul_tiles_init` is called per CB pair inside the acquire window for
  safety (state_configure is a no-op on Blackhole but the math init matters).
- The reader broadcasts cos/sin over B and H via `cos_tile_id = (row %
  T_tiles) * tiles_per_row + col`. Requires `T % 32 == 0`.
- Backward RoPE uses the same kernel with pre-computed `_rope_neg_sin_2d`
  (cached in `_init_rope` to avoid per-call `ttnn.neg`).
- The built-in `ttnn.experimental.rotary_embedding` was tested but requires
  `d_head % 64 == 0` — fails for d_head=96 (d_model=384, n_heads=4).
  The custom kernel has no such restriction.
- `_apply_rope` still does 2 `ttnn.slice` + concat around the kernel call
  because d_half=48 is not tile-aligned. These contribute ~0.07ms (slices)
  and ~0.13ms (concat) per call vs ~0.15ms for the kernel itself.
- **Split-RoPE optimization**: `_apply_rope_split` returns (rot1, rot2)
  without concat, and `_apply_rope_backward_split` takes pre-split grads
  without slicing. The forward uses `qk = rot1_q@rot1_k^T + rot2_q@rot2_k^T`
  (2 matmuls + 1 add) instead of `matmul(concat(rot1_q,rot2_q), ...)` (1
  matmul + 1 concat). The backward uses 4 split matmuls instead of 2
  matmuls + 2 slices. This eliminates 2 concats (forward) and 4 slices
  (backward), replacing them with cheap matmuls. Measured: 9.4% total
  speedup (3.2ms → 2.9ms per fwd+bwd), RoPE itself 25% faster (1.6ms →
  1.2ms). The `_apply_rope` / `_apply_rope_backward` methods (with
  concat/slice) are kept for the non-fused fallback path.

### Fused gate backward (`_fused_gate_backward`)

Computes `grad_out_flat = grad_out_gated * gate` and
`grad_g = grad_out_gated * out_flat * gate * (1 - gate)` in a single
kernel pass. Uses 3 FPU `mul_tiles` (gog*gate, gog*out_flat, gate*gate)
+ 1 `copy_tile` + 3 SFPU binary ops (sub for gate-gate^2, mul for
grad_g). Replaces 6 ttnn ops (ones_like, sub, 4 muls) with 1 kernel.

- Kernel files: `kernels/gate_bwd_{reader,compute,writer}.cpp`
- Parity: `test_retention_parity.py --tile-aligned` (forward + backward)
- **No wall-clock improvement**: the `generic_op` dispatch overhead
  (~0.20ms) is the same as 6 individual ttnn ops (~0.03ms each = 0.18ms).
  The kernel computation is negligible vs dispatch. Kept because it
  reduces op count and may help with larger models where dispatch
  overhead is amortized.
- `ttnn.from_torch` tensors don't work with `generic_op` — they need
  a ttnn op (e.g. `ttnn.mul(x, ones_like(x))`) to be run first. This
  is only an issue in standalone tests; in the model, all inputs to
  custom kernels are outputs of ttnn ops.

### Cached scale_tt

`_scale_tt` is created once in `__init__` as a device tensor, replacing
a per-forward `ttnn.from_torch(torch.tensor([scale]))` call. Saves
~0.1ms per forward pass.

### Optimization ceiling

The layer is at 2.9ms/fwd+bwd for d_model=384, n_heads=4, T=128, B=8.
The remaining bottleneck is `generic_op` dispatch overhead (~0.15-0.20ms
per call), not kernel computation. RoPE (4 calls) = 1.2ms (41%),
scale+decay (2 calls) = 0.4ms (14%), gate backward (1 call) = 0.2ms (7%).
Further optimization requires either:
- A single fused kernel for the entire layer (very complex, fragile)
- ttnn tracing/graph mode to batch dispatches (if available)
- A larger model where compute dominates dispatch overhead
