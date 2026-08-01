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
