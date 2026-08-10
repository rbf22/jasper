# Jasper POC — Project notes

## Architecture

**Jasper** is a workspace-augmented retention network with recurrent core.
It combines four components that have never been combined before:

1. **Retention backbone** (RetNet-style): Decayed linear attention with
   per-head gamma decay and RoPE. Used for all non-attention layers.
2. **Workspace** (Perceiver-style): 16 learnable memory slots with
   read/write via cross-attention, QK-Norm, and ReZero gates. Sits at
   specific layer positions (5 and 10).
3. **Recurrent core**: Layers 6-9 iterated K=6 times with slot state
   chaining across iterations.
4. **Attention residuals** (Kimi K3-style): Learned softmax attention
   over iteration outputs, replacing fixed blend.

No existing architecture combines all four. See the README for the full
novelty analysis.

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

`TTMambaWorkspaceModel` (in `model_ttnn.py`) — the Jasper model — uses
`TTMamba3Layer` (the retention layer) unconditionally for every non-attention
layer, for all cells (A/B/C) and all `configs/cell_*_tt.yaml` runs. `run_A/`, `run_B/`, `run_C/` symlink `mamba3_layer.py` and `model_ttnn.py`
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

## CPU-only pytest tests

The repo has pytest test suites that run on CPU only (no `ttnn` device
required). They are listed explicitly in `pytest.ini` to avoid accidentally
collecting `ttnn`-dependent tests.

```bash
# From workspace-poc/:
/home/rfenwick/Documents/jasper/.tt-venv/bin/pytest test_data.py test_text_data.py -v

# From workspace-mvp/:
/home/rfenwick/Documents/jasper/.tt-venv/bin/pytest test_text_data.py -v
```

| Test file | What it covers |
|-----------|---------------|
| `test_data.py` | Synthetic task generators and verifiers in `data.py` (arithmetic, copy, reverse, sort) |
| `test_text_data.py` | GPT-2 BPE tokenizer wrapper, TinyStories dataset loading, packed-stream label generation in `text_data.py` |
| `test_gate_clamp.py` | Gate value clamping logic (verifies values outside [-0.3, 0.3] are clamped, values inside unchanged) |
| `test_ws_qknorm_gradcheck.py` | Float64 autograd gradcheck of workspace cross-attention backward (including causal masks) |

The `ttnn`-dependent tests (`test_mamba3_parity.py`, `test_retention_parity.py`,
`test_scale_decay.py`, `test_fused_rope_single.py`, `test_scatter_loss.py`)
are **not** in `pytest.ini` — they require a Tenstorrent device and are run
manually as described in their respective sections above.

### Code-as-Data (CaD) framework

Two YAML manifests provide structural documentation of the codebase:

- `architecture.yaml` — Bounded contexts, data flow, dependencies,
  architectural decisions, and stability constraints. A structural
  manifest that complements this file.
- `traceability.yaml` — Maps test functions to the production functions
  they cover, documenting coverage gaps.

Key functions in `data.py` and `text_data.py` have semantic docstring
decorators (`@Pure`, `@Idempotent`, `@Deterministic`) indicating their
contractual properties for testability and reasoning.

## Current training run (2026-08-04, AR core + chain scaling fix)

### Directory rename

The project directories were renamed on 2026-08-04:
- `mamba-poc` → **`workspace-poc`** (synthetic arithmetic training)
- `mamba-mvp` → **`workspace-mvp`** (text POC — TinyStories)

The architecture has evolved well beyond Mamba (retention layers + workspace
+ recurrent core), so the "mamba" prefix was dropped. All symlinks in
`workspace-mvp/` point to `workspace-poc/` for shared model code.

### Active runs

Two Cell C variants are training in parallel on the synthetic tasks:

| Run | Device | Config | Checkpoint dir | Log |
|-----|--------|--------|----------------|-----|
| Cell C AR (primary) | 2 | `cell_c_attn_residual.yaml` | `checkpoints/ar_chain_fix/` | `logs/cell_c_ar_chain_fix2_20260804.log` |
| Cell C K3 (deprioritized) | 3 | `cell_c_attn_residual_k3.yaml` | `checkpoints/k3_chain_fix/` | `logs/cell_c_k3_chainfix2_20260804.log` |

The **AR version is the primary Cell C config**. The K3 variant (short conv +
per-channel decay) is deprioritized due to increased overhead (~13s/step vs
~8s/step) from host-side gamma gradient transfers. It's kept running as a
backup in case the AR version encounters issues, but is not expected to be
used for the text POC.

Cell A (backbone-only control) and Cell B (workspace, no recurrence) completed
training earlier — Cell B early-stopped at step 9404. Their results are valid
(no bugs affected them — the AR chain gradient bug only impacts
`recurrent_core: true` configs).

### Launch commands (from workspace-poc/):

```bash
# Cell C AR (primary) — device 2
cd /home/rfenwick/Documents/jasper/workspace-poc
TT_VISIBLE_DEVICES=2 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_ttnn.py \
    --config configs/cell_c_attn_residual.yaml --device 0 \
    --checkpoint_dir checkpoints/ar_chain_fix \
    > logs/cell_c_ar_chain_fix2_20260804.log 2>&1 &

# Cell C K3 (deprioritized) — device 3 (P300, needs mesh graph descriptor)
TT_VISIBLE_DEVICES=3 TT_MESH_GRAPH_DESC_PATH=/home/rfenwick/tt-boltz/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_ttnn.py \
    --config configs/cell_c_attn_residual_k3.yaml --device 0 \
    --checkpoint_dir checkpoints/k3_chain_fix \
    > logs/cell_c_k3_chainfix2_20260804.log 2>&1 &
```

### Critical divergence thresholds

All previous Cell C divergences occurred at specific step counts. The current
run (with the chain scaling fix) must pass these to be considered stable:

| Step | What happened | Bug | Status |
|------|---------------|-----|--------|
| ~750 | K=6, 1/sqrt(K) slot scaling diverged | Slot chain amplification | Fixed (1/K scaling) |
| ~900 | K=3, no slot scaling diverged | Slot chain amplification | Fixed (1/K scaling) |
| ~1000 | Fixed blend core diverged | Blend chain amplification | Fixed (AR core) |
| ~1050 | First AR run diverged | Gradient double-counting | Fixed (removed pre-init) |
| ~1550 | Second AR run diverged | Unscaled AR chain gradient | Fixed (1/K chain scaling) |

**The key threshold is step 1550.** If the current run passes it without grad
norm climbing, the chain scaling fix is confirmed. Watch for grad norm
climbing from ~40 to ~200 over 200-300 steps — that's the early warning
pattern before each previous divergence.

### History of gradient instability fixes

The workspace cross-attention causes gradient amplification through its
two-pass (read + write) structure. Multiple rounds of fixes were attempted:

1. **Run 1** (bound=3.0, gamma trainable): Cell B diverged at step ~2300,
   Cell C at step ~700. Gamma exp_avg_sq exploded.
2. **Run 2** (bound=2.0, gamma trainable): Cell B diverged at step ~1800,
   Cell C at step ~700. Backbone norms capped but gamma gradients still
   exploded.
3. **Run 3** (bound=2.0, gamma frozen, K=3 for C): Cell B pre-divergence at
   step ~1700, Cell C diverged at step ~900. Gamma freeze helped modestly.
4. **Run 4** (bound=2.0, gamma frozen, K=3 for C): Cell B diverged at step
   ~1400, Cell C at step ~900. Same pattern — divergence at gate ~0.20-0.25.

**Root cause**: The workspace's QK^T had condition numbers of 40,000-112,000
(entropy collapse). When gates opened to ~0.20, the cross-attention gradient
paths amplified backbone gradients multiplicatively. Spectral norm bounds and
gamma freezing were symptomatic fixes — they delayed but didn't prevent
divergence.

### Architecture v2 fixes (2026-08-02)

Three architectural changes based on literature review:

1. **QK Normalization** (`_l2_normalize_heads` in `TTWorkspaceModule`):
   L2-normalizes Q and K along d_head before computing attention scores,
   with a learnable scale parameter (`read_qk_scale`, `write_qk_scale`)
   initialized to 1/√d_head. This bounds attention logits regardless of
   weight magnitudes, eliminating the entropy collapse / ill-conditioned
   QK^T that caused divergence. (Henry et al., 2020 — now standard in
   OLMo 2, Gemma 3, Qwen 3)

2. **ReZero gates** (replaces sigmoid gates): Scalar gates initialized to
   0.0 (true identity at init), no sigmoid. The workspace starts as a
   no-op and grows naturally through gradient descent. This gives the
   backbone time to learn a good representation before the workspace
   starts contributing, and avoids the sigmoid's gradient saturation.
   (Bachlechner et al., 2020)

3. **Component-wise gradient clipping** (`clip_grad_norm` in `train_ttnn.py`):
   Workspace params (ws_*) clipped at `ws_grad_clip` (0.5), backbone params
   at `grad_clip` (1.0). Each group's norm is computed and clipped
   independently. This prevents workspace gradient spikes from dominating
   the global clip and starving the backbone of learning signal.
   (Yang et al., 2022 — EMNLP 2022)

### Causal masking fix (2026-08-08)

**Critical bug fix.** The workspace's cross-attention was bidirectional (no
causal mask), which allowed future tokens — including the answer — to leak
into earlier positions through the workspace's read/write passes. This meant
the model could learn to copy the answer from future positions rather than
computing it from the prompt. A reviewer correctly identified this as
"learning where the answer is, not how to do math."

The fix applies Perceiver-IO style causal masking to both cross-attention
passes (`_build_causal_masks` in `TTWorkspaceModule`):

1. **Read pass** (slots attend over sequence): Slot `i` can only attend to
   positions `[0, (i+1)*T//m - 1]`. Each slot gets a growing receptive field
   — early slots see only early tokens, later slots see more.

2. **Write pass** (sequence attends over slots): Position `t` can only
   attend to slots `[0, (t+1)*m//T]`. This ensures position `t` only reads
   from slots whose receptive field includes positions `≤ t`.

Both masks are applied before softmax (setting masked positions to -1e4),
and gradients at masked positions are zeroed in the backward pass (matching
the `TTAttentionLayer` pattern). The backward pass is verified against
PyTorch autograd in `test_ws_qknorm_gradcheck.py` (float64, all gradients
match to <1e-13 relative error).

**All previous workspace-containing results (Cell B, Cell C) are invalid**
— they were trained with the leakage. Cell B's loss 0.083 vs Cell A's 0.931
may have been largely or entirely due to answer leakage, not genuine
reasoning. All workspace cells must be retrained from scratch with this fix.

The same fix is applied to the PyTorch reference model (`model.py`
`WorkspaceModule`) and the text POC (`workspace-mvp/` uses the same
symlinked `model_ttnn.py`).

### New workspace parameters

The workspace module now has 2 additional parameters:
- `ws_read_qk_scale`: learnable scalar for read attention scale (init 1/√d_head)
- `ws_write_qk_scale`: learnable scalar for write attention scale (init 1/√d_head)

These are included in `get_params()`, `set_params()`, checkpoint save/load,
and the optimizer. The `lr_groups` config gives them 1.0x base LR (same as
backbone), not 0.25x like other workspace weights.

### Gate logging

The training log columns `gz_gr` / `gz_gw` now show the raw ReZero gate
values (not sigmoid). At init they are 0.0. They grow through gradient
descent — values of 0.1-0.3 are expected during training. Negative values
are also possible (the workspace can learn to subtract from the residual).

### Previous fixes still in place

- **Backbone spectral normalization**: caps qkv/out_proj at 2.0 (still active)
- **Gamma freezing**: retention gamma is frozen (still active)
- **Cell C K=6**: restored to full recurrence (slot-state gradient scaling
  fixed — see below)

### Slot-state gradient scaling (2026-08-03)

The 1/sqrt(K) blend scaling normalizes the **residual stream (x)** gradient
accumulation across K iterations, but the **slot state chain** bypassed it.
In the core backward, `grad_slots_in` from each iteration's workspace
backward chained directly to the previous iteration without blend scaling,
causing the workspace's gradient amplification to compound exponentially
across K iterations.

Two scaling attempts:

1. **1/sqrt(K) scaling** (first attempt): scale `grad_slots_in` by `1/sqrt(K)`.
   Per-iteration gain: `A/K + 1 - 1/sqrt(K)`. This requires A ≤ sqrt(K) for
   stability. With K=3, diverged at step ~1150. With K=6, diverged at step ~750
   (more iterations compound the amplification even with better per-iteration
   gain).

2. **1/K scaling** (current): scale `grad_slots_in` by `1/K` instead.
   Per-iteration gain: `A/K² + 1 - 1/sqrt(K)`, which is ≤1 for A ≤ sqrt(K)
   — much more robust. The x residual blend stays at 1/sqrt(K) (correct for
   additive gradient accumulation); only the slot chain uses the more
   conservative 1/K (needed because the slot chain is multiplicative).

### Attention Residual Core (2026-08-04, Kimi K3 style)

Even with 1/K slot scaling, Cell C still diverged at step ~1000 (gradient
norm exploded from ~4 to 10000-39500, loss spiked from 0.78 to ~5.1). The
root cause is the **fixed blend chain** itself: `x = (1-a)*x + a*x_new`
applied K times creates a multiplicative gradient path where each
iteration's gradient flows back through all subsequent iterations' blend
operations, amplifying by a factor of `(1-a)` per iteration. With K=6
and `a = 1/sqrt(6) ≈ 0.408`, the gradient chain has gain `(1-0.408)^6 ≈ 0.07`
per path but there are K paths summing, and the workspace's cross-attention
amplification compounds on top.

The fix replaces the fixed blend with **Attention Residuals** (Bachlechner
et al., 2020; Moonshot AI Kimi K3, 2026 — see `kimi-k3-relevance-notes.md`
§2, ranked #1 priority). Instead of blending, we store all K iteration
outputs (plus the pre-core input x_0) and compute softmax attention over
them with a learned query vector:

```
scores_k = sum_d(x_k * query) * scale       → (B, T) per k
alpha = softmax([scores_0, ..., scores_K])   → (B, T, K+1)
x_final = sum_k alpha_k * x_k                → (B, T, d_model)
```

This gives:
- **Bounded output magnitude**: softmax normalizes weights to sum=1, so
  `|x_final| ≤ max_k |x_k|` — no amplification.
- **Consistent gradient magnitude**: each iteration receives gradient
  `alpha_k * grad_x_final` directly from the attention, not through a
  chain of blend operations. The gradient is bounded by the softmax
  weights (which sum to 1).
- **Learned selection**: the model learns which iterations' outputs to
  weight more, rather than using a fixed 1/sqrt(K) schedule.

The `AttentionResidual` class (`model_ttnn.py`) has two learnable
parameters:
- `ar_query`: (1, d_model) vector, init `randn / sqrt(d_model)`
- `ar_scale`: scalar, init `1/sqrt(d_model)` (standard attention scale)

Both are stored with the `ws_` prefix (`ws_ar_query`, `ws_ar_scale`) so
they get the workspace LR group (0.25x by default, configurable via
`lr_groups` in the YAML). The slot chain still uses 1/K gradient scaling
in backward — the attention residual only replaces the x-path blend.

Config: `configs/cell_c_attn_residual.yaml` — identical to
`cell_c_tt.yaml` but with `attention_residual_core: true`.

Activation: set `attention_residual_core: true` in the config. When
`false` (default), the original fixed blend core is used (backward
compatible). The `AttentionResidual` module is only instantiated when
both `recurrent_core` and `attention_residual_core` are true.

Smoke-tested: 3 steps, loss 4.92→4.75, grad norm 6.6-9.2, checkpoint
saved. Param count: 10,531,219 (385 more than blend: 384 for query + 1
for scale).

### Attention Residual backward bug fix (2026-08-04)

The first Cell C run with the attention residual core diverged at step
~1050 (grad norm 18 → 8628 in one step, same pattern as before). Root
cause was a **gradient double-counting bug** in the AR backward path in
`model_ttnn.py`.

In the recurrent core backward (`backward()` around line 3237), the code
pre-initialized `grad_x = grad_x_list[K_active]` (the AR gradient for the
last iteration's output x_K), then the first loop iteration added
`grad_x_list[iter_num + 1] = grad_x_list[K]` again — making the gradient
flowing through the last iteration's core layers 2x too large. This 2x
amplification propagated through all earlier iterations via the chain
rule, compounding with the workspace's cross-attention amplification
across the K=6 slot chain.

**Fix**: removed the pre-initialization; the loop now handles all AR
gradient additions (first iteration sets `grad_x = ar_grad_x`, subsequent
iterations add). Verified with 100-step test: grad norm stable at 6-9
(no climbing), loss decreasing steadily (4.86 → 2.65). The original
run's deceptively low grad norm (2-3 at start) was caused by aggressive
clipping masking the 2x amplification — the clipping kicked in harder
because of the inflated gradient, hiding the buildup until it overwhelmed
the clip threshold at step ~1050.

### AR chain gradient scaling fix (2026-08-04)

After the double-counting fix, Cell C still diverged — this time at step
~1550 (grad norm 40 → 206 → 15563 in 200 steps). The double-counting fix
delayed the divergence by 500 steps but didn't address the underlying
chain amplification.

**Root cause**: In the AR backward, the gradient from each iteration
flows through the shared core layers (layers 6-9) at **full magnitude**
and chains to the previous iteration. The gradient at iteration 0 is
approximately `sum_k A^(K-k) * |ar_grad_x[k]|`, where A is the layer
backward amplification factor (A > 1 when the workspace cross-attention
is active). For K=6 and A≈2, this gives ~64x amplification.

The fixed blend path had `1/sqrt(K)` scaling on the x gradient chain
(`grad_x_new = blend * grad_x`), which dampened this. The AR path had
no such scaling — the AR gradient was bounded by softmax, but the
chained gradient from subsequent iterations was not.

**Fix**: scale the chained gradient by `1/K` before adding the AR
gradient at each iteration (`slot_scale_tt`, already computed as `1/K`
for active iterations). The AR gradient itself is not scaled (it's
bounded by softmax). Per-iteration chain gain becomes `A/K`, stable
for A < K (vs `A/sqrt(K)` requiring A < sqrt(K)). This matches the
slot chain's existing `1/K` scaling.

Verified: 200-step test shows grad norm stable at 5.7-9.1 (no climbing),
loss decreasing 4.94 → 1.28. Full 10K-step run launched on device 2.

### Backbone ReZero gate fix (2026-08-06) — ROOT CAUSE of all divergences

The AR chain scaling fix (1/K) delayed divergence from step ~1050 to ~2650,
but the model still diverged. Three subsequent fix attempts all failed:
1. `backbone_spectral_norm_bound: 2.0 → 1.5` — diverged at step 2800 (worse:
   out_proj grew to 1.5, making A=5.96 vs original 5.0)
2. `chain_scale_safety: 1.5` (1/(K*1.5) = 1/9 scaling) — diverged at step
   2636 (faster: the step-2600 checkpoint was already in the unstable regime)

**Root cause**: The `TTGatedResidualLayer` (wrapping all backbone layers,
including core layers 6-9) used **sigmoid gates** with init=0, giving
`sigmoid(0) = 0.5`. This means each backbone layer contributes 50% of its
output to the residual stream at initialization.

The v2 architecture fix (2026-08-02) changed the *workspace* gates to
ReZero but **missed the backbone layer gates** — they still used sigmoid.

With sigmoid gates at 0.5, each core layer's gradient self-amplification is:
```
layer_gain = 1 + sigmoid(gate) × σ_layer = 1 + 0.5 × 1.07 = 1.535
```
Over 4 core layers: `A_layers = 1.535⁴ = 5.55`
With workspace contribution: `A_coupled ≈ 5.77`
With K=6 and 1/K scaling: `A/K = 0.962` — **only 4% stability margin**

The system starts training *at the stability boundary*. Any weight growth
pushes A > K, causing divergence. This explains:
- Why divergence always happened around step 1000-2650 (weights grow slowly)
- Why lowering the spectral norm bound made it worse (out_proj grew to fill
  the new bound, increasing A)
- Why the chain safety factor didn't help (the checkpoint was already past
  the boundary)
- Why all runs diverged at `gz_gr ≈ 0.68-0.77` (read gate opening increases
  the workspace's cross-amplification, pushing A_coupled over K)

**Fix**: Changed `TTGatedResidualLayer` from sigmoid to ReZero (no sigmoid,
gate init=0.0). At init, each backbone layer is pure identity (contributes
nothing), giving:
```
layer_gain = 1 + 0.0 × σ_layer = 1.0
A_layers = 1.0⁴ = 1.0
A_coupled ≈ 1.04
A/K = 0.173 — 83% margin
```
Gates grow slowly through gradient descent. Even at gate=0.3 (60% of
sigmoid's 0.5), the margin is still 47%.

Also added `freeze_slot_decay: true` to prevent the slot_decay parameter
from growing above 1.0 (which would make the slot chain divergent).

Config changes in `configs/cell_c_attn_residual.yaml`:
- `chain_scale_safety: 1.0` (reverted from 1.5 — not needed with ReZero)
- `freeze_slot_decay: true` (new — prevents slot chain growth)

Code changes in `model_ttnn.py`:
- `TTGatedResidualLayer.forward`: `sigmoid(self.gate)` → `self.gate`
- `TTGatedResidualLayer.backward`: removed sigmoid derivative, grad is
  now `grad_out * inner` (not `grad_out * inner * sigmoid'(gate)`)
- `ModelConfig.freeze_slot_decay`: new field, drops `ws_slot_decay`
  gradients and skips `slot_decay` in `get_params()` when true

Training restarted from scratch (step 0) — the step-2600 checkpoint has
sigmoid-gate weights that don't transfer to ReZero.

### Gate clamping fix (2026-08-09) — Cell B divergence with causal masking

After the causal masking fix (2026-08-08), Cell B (workspace, no recurrent
core) diverged at step ~2384. The root cause was **ReZero gate overgrowth**
leading to RMSNorm cancellation.

With causal masking, Cell B's read gate grows **negative** (the workspace's
past-only contribution is noise for Cell B, which has no recurrent core to
mediate it). At `read_gate ≈ -0.43`, the pre-norm residual
`decay * slots + gate * read_out` cancels, making RMS very small. RMSNorm
backward divides by RMS, amplifying the gradient by `1/RMS` → gradient
explosion.

Cell C does not have this problem because its attention residual core
mediates gate gradients — the AR softmax bounds the gradient flowing back
to the gates, preventing overgrowth. Cell C's gates are positive (~0.30 at
step 1950) and self-dampening (amplification makes RMS larger → 1/RMS
smaller → gradient smaller).

**Fix** (two parts, both in `configs/cell_b_tt.yaml`):

1. **Gate value clamping** (`gate_clamp_bound: 0.3`): After each optimizer
   step, `read_gate` and `write_gate` are clamped to `[-0.3, 0.3]`. This
   prevents the cancellation regime. Implemented in `train_ttnn.py` after
   `optimizer.step()`. The clamp is a no-op for normal learning (gates
   reach ~0.03-0.30 during training) — it only fires in the pathological
   regime.

2. **Reduced gate LR** (`lr_groups.ws_read_gate: 3.0`, `ws_write_gate: 3.0`):
   Decreased from 10x to 3x base LR. Slows gate growth, giving the backbone
   more time to adapt before the workspace's contribution becomes
   significant.

3. **`freeze_slot_decay: true`** (added to Cell B): Matches Cell C. Prevents
   `slot_decay` from drifting and destabilizing the slot update. Cell B
   previously allowed it to learn (drifted to 0.98), but the drift is an
   unnecessary degree of freedom that adds risk without benefit.

**Safety net for Cell C and text POC**: `gate_clamp_bound: 0.3` is also set
in `cell_c_attn_residual.yaml` and `text_cell_c.yaml`. Cell C's gates are
positive and self-dampening, so the clamp won't constrain normal learning —
it only prevents a catastrophic regime shift if the dynamics change
unexpectedly.

Verified with `test_gate_clamp.py` — the clamp logic is tested in isolation
(confirming values outside [-0.3, 0.3] are clamped, values inside are
unchanged).

### Checkpoint backups from previous runs

- `checkpoints/diverged_run_20260802/` — Run 1 checkpoints
- `checkpoints/diverged_run2_20260802/` — Run 2 checkpoints
- `checkpoints/diverged_run3_20260802/` — Run 3 checkpoints
- `checkpoints/diverged_run4_frozen_gamma/` — Run 4 checkpoints
- `checkpoints/diverged_qknorm_k3/` — QK-Norm K=3 run, no slot scaling (diverged at step ~1150)
- `checkpoints/diverged_qknorm_k6_sqrt_scaling/` — QK-Norm K=6, 1/sqrt(K) slot scaling (diverged at step ~750)

### Device 3 (P300 board) mesh graph descriptor

Device 3 is a P300 board with 1 chip. It requires the **p150** mesh graph
descriptor (not p300 — the p300 descriptor expects 2 chips and fails with
`TT_FATAL: Graph specified in MGD could not fit in the discovered physical
topology`). Always set both env vars when using device 3:

```bash
TT_VISIBLE_DEVICES=3 TT_MESH_GRAPH_DESC_PATH=/home/rfenwick/tt-boltz/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

### Kimi K3 architectural updates (2026-08-04, deprioritized)

Two improvements from Kimi K3 (see `kimi-k3-relevance-notes.md` §3, §5)
were added to `TTRetentionLayer` in `model_ttnn.py`:

1. **Short convolution** (`short_conv: true`): depthwise causal conv1d
   (kernel=3) applied before the QKV projection. Captures local token
   dependencies. Implemented as manual shifted multiply (not
   `ttnn.conv1d`, which has heavy compilation overhead for a 3-tap
   kernel). Identity init (w[:,0]=1, w[:,1:]=0) — starts as a no-op.

2. **Per-channel decay** (`per_channel_decay: true`): expands gamma from
   `(n_heads,)` to `(n_heads, d_head)` for element-wise per-channel
   memory decay. Uses an exp decomposition to avoid materializing a
   `(1, H, d_head, T, T)` decay matrix. The gamma gradient is computed
   on host (4 tensor transfers per step) — **this is the source of the
   ~60% overhead that led to deprioritization**.

Config: `configs/cell_c_attn_residual_k3.yaml`. Param count: 10,543,891.

**Status: deprioritized.** The K3 variant is ~60% slower per step (13s vs 8s)
due to host-side gamma gradient transfers. It's kept as a backup but the AR
version without K3 features is the primary config for both synthetic and text
training.

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

| Cell | Config | Device | Time/step | Est. total (10000 steps) |
|---|---|---|---|---|
| A | backbone-only | 0 | ~4.1s | ~11.4 hours |
| B | + workspace | 1 | ~4.2s | ~11.7 hours |
| C AR | + workspace + recurrence (K=6) | 2 | ~8.5s | ~23.6 hours |
| C K3 | + short conv + per-channel decay | 3 | ~13s | ~36 hours (deprioritized) |

Cell C AR is ~2x slower than A/B due to the recurrent core (K=6 iterations
per step). The K3 variant adds ~50% more overhead from host-side gamma
gradient transfers — this is why it's deprioritized.

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

## Text POC (workspace-mvp/)

The `workspace-mvp/` directory contains the text training pipeline — a
drop-in replacement for the synthetic data pipeline that trains the same
model architecture on TinyStories with GPT-2 BPE tokenization.

### Directory structure

```
workspace-mvp/
├── text_data.py          # BPE tokenizer + TinyStories dataset + batch sampling
├── train_text.py         # Training loop (imports from workspace-poc/train_ttnn.py)
├── eval_text.py          # Perplexity evaluation + text generation
├── configs/
│   └── text_cell_c.yaml  # Cell C AR config adapted for text (50K vocab, seq_len=512)
├── data/
│   ├── tinystories_train.txt         # 1.9GB, ~480M tokens
│   └── tinystories_valid.txt         # 19MB, ~4.8M tokens
├── model_ttnn.py -> ../workspace-poc/model_ttnn.py  (symlink)
├── mamba3_layer.py -> ../workspace-poc/mamba3_layer.py
├── kernels/ -> ../workspace-poc/kernels/
└── venv -> ../.tt-venv/
```

### Key differences from synthetic training

| Aspect | Synthetic (workspace-poc) | Text (workspace-mvp) |
|--------|--------------------------|----------------------|
| Data | Generated arithmetic tasks | TinyStories (real text) |
| Vocab | 128 (char-level) | 50,257 (GPT-2 BPE) |
| Seq len | 128 | 512 |
| Labels | Answer tokens only (-100 elsewhere) | Every position (standard LM) |
| Loss | On-device (V×V identity matrix) | Host-side (for V > 2048) |
| Eval | Task accuracy with verifiers | Perplexity + text generation |
| Params | 10.5M | 29.8M (19.3M from embedding) |

### Host-side loss for large vocab

The on-device `cross_entropy_loss` in `train_ttnn.py` uses a V×V identity
matrix for one-hot encoding — fine for V=128 (32KB), impossible for V=50257
(~5GB in bfloat16). `train_text.py` provides two alternatives:

1. **Host-side loss** (default): float32 on CPU, ~200MB logits transfer per
   micro-batch. Faster for the current 30M model size.
2. **On-device scatter loss**: Uses `ttnn.gather` + `ttnn.scatter_add` — no
   V×V identity matrix, no host transfer of full logits. Slower for the
   current model size (on-device softmax over 50K elements has high dispatch
   overhead) but will be preferred for larger models. Verified against
   host-side loss in `test_scatter_loss.py` (0.3% loss match, 1.6% gradient
   match).

Select via CLI: `--loss_method host` (default) or `--loss_method scatter`.

### Dataset choice rationale

TinyStories was chosen over reasoning datasets (OpenThoughts, NuminaMath,
etc.) because:
- It's the only dataset designed for <10M param models
- ~480M tokens fits the training budget exactly
- It validates that the architecture can learn coherent language at all
- Reasoning datasets (DeepSeek-R1 distilled CoT) are too complex for a 10M
  model — it would memorize surface patterns without understanding reasoning

A two-stage approach is planned: TinyStories for foundation training, then
a small reasoning dataset (OpenThoughts-114k metadata config) for
fine-tuning to test the workspace's multi-hop capability.

### Smoke test results (200 steps)

- Loss: 10.92 → 9.24 (log(50257) ≈ 10.82 at init, confirming correct random init)
- Perplexity: ~50K → 10,595
- Grad norm: 1.2-2.9 (stable, no divergence)
- Speed: ~1.4s/step, 1,400 tok/s (micro_batch=4, seq_len=512, no accum)

### Launch (from workspace-mvp/):

```bash
# Smoke test
TT_VISIBLE_DEVICES=1 python train_text.py \
    --config configs/text_cell_c.yaml --steps 3 \
    --micro_batch 4 --accum_steps 1 \
    --checkpoint_dir /tmp/text_test --device 0

# Full training
cd /home/rfenwick/Documents/jasper/workspace-mvp
TT_VISIBLE_DEVICES=1 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_text.py \
    --config configs/text_cell_c.yaml --device 0 \
    --checkpoint_dir checkpoints \
    > logs/text_cell_c.log 2>&1 &

# Evaluation
TT_VISIBLE_DEVICES=1 python eval_text.py \
    --checkpoint checkpoints/cell_text_step500.pt \
    --config configs/text_cell_c.yaml --device 0 \
    --generate --prompt "Once upon a time"
```
