# WRAP POC — Project notes

## Architecture

**WRAP** is a workspace-augmented retention network with recurrent core.
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

## Production integration

`TTWRAPModel` (in `model_ttnn.py`) — the WRAP model — uses
`TTRetentionLayer` (defined inline in `model_ttnn.py`, RetNet-style decayed
linear attention) for every non-attention layer, for all cells (A/B/C) and
all `configs/cell_*_tt.yaml` runs.

Smoke-tested end-to-end after the fixes above (3 steps each, `--accum_steps 1`
to skip the full gradient-accumulation loop):

```bash
.tt-venv/bin/python train_ttnn.py --config configs/cell_a_tt.yaml --steps 3 --micro_batch 4 --accum_steps 1 --checkpoint_dir /tmp/x --device 0
.tt-venv/bin/python train_ttnn.py --config configs/cell_b_tt.yaml --steps 3 --micro_batch 4 --accum_steps 1 --checkpoint_dir /tmp/x --device 0
.tt-venv/bin/python train_ttnn.py --config configs/cell_c_attn_residual.yaml --steps 3 --micro_batch 4 --accum_steps 1 --checkpoint_dir /tmp/x --device 0
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
| `test_gradients.py` | Float64 autograd gradcheck of all layer backward math: RMSNorm, softmax, RoPE, cross-entropy, retention, attention, AttentionResidual, GatedResidual, full model (12 tests, CPU-only) |
| `test_config.py` | `build_model_config` config parsing: defaults, partial/full configs, YAML loading for all 3 cells, d_inner/d_head properties (10 tests, CPU-only) |
| `test_gradient_accum.py` | `accumulate_grads` math: single/multi-step accumulation, accum_factor scaling, multi-param, fp32 precision, empty dict (6 tests, CPU-only) |

The `ttnn`-dependent tests (`test_retention_parity.py`,
`test_scale_decay.py`, `test_fused_rope_single.py`, `test_scatter_loss.py`,
`test_backward_parity.py`, `test_numerical_grad.py`, `test_memory_leak.py`)
are **not** in `pytest.ini` — they require a Tenstorrent device and are run
manually as described in their respective sections above.

### Device-dependent test scripts

| Test file | What it covers | Run command |
|-----------|---------------|-------------|
| `test_retention_parity.py` | Retention layer forward + backward vs PyTorch reference (3-stage: gradcheck, forward parity, backward parity) | `.tt-venv/bin/python test_retention_parity.py --gradcheck --tile-aligned` |
| `test_backward_parity.py` | ttnn manual backward vs PyTorch autograd for attention, AttentionResidual (active + inactive), and retention layers | `.tt-venv/bin/python test_backward_parity.py -v` |
| `test_numerical_grad.py` | Finite-difference gradient check using ttnn forward only (no PyTorch reference needed). Checks retention layer and AttentionResidual | `.tt-venv/bin/python test_numerical_grad.py -v` |
| `test_memory_leak.py` | Per-layer and per-kernel memory leak isolation tests (18 tests covering custom kernels, ttnn ops, per-layer fwd/bwd, cache cleanup) | `.tt-venv/bin/python test_memory_leak.py -l` (list), `--test custom_rope` (single) |
| `test_scale_decay.py` | Fused scale+decay custom kernel correctness | `.tt-venv/bin/python test_scale_decay.py` |
| `test_fused_rope_single.py` | Fused RoPE custom kernel correctness | `.tt-venv/bin/python test_fused_rope_single.py` |
| `test_gate_bwd.py` | Fused gate backward custom kernel correctness | `.tt-venv/bin/python test_gate_bwd.py` |
| `test_ws_qknorm_gradcheck.py` | Workspace QK-Norm backward vs autograd (can also run on CPU) | `.tt-venv/bin/python test_ws_qknorm_gradcheck.py` |
| `test_checkpoint_roundtrip.py` | Checkpoint save/load round-trip for all 3 cells, weight tying preservation, step counter | `.tt-venv/bin/python test_checkpoint_roundtrip.py` |
| `test_optimizer_state.py` | TTAdamW get_state/load_state round-trip, fp32 master preservation, step_count restoration | `.tt-venv/bin/python test_optimizer_state.py` |
| `test_clear_caches.py` | clear_caches correctness: caches cleared, params survive, idempotent, recurrent core caches | `.tt-venv/bin/python test_clear_caches.py` |
| `test_params.py` | get_params/set_params round-trip, weight tying (token_emb == lm_head), expected param keys | `.tt-venv/bin/python test_params.py` |
| `test_clip_grad_norm.py` | Component-wise gradient clipping: global, ws vs backbone, gamma scaling, no-clip, empty | `.tt-venv/bin/python test_clip_grad_norm.py` |
| `test_recurrent_core.py` | Recurrent core K-value handling: K=1/3/6 forward, K-dependent outputs, AR vs no-AR, backward | `.tt-venv/bin/python test_recurrent_core.py` |
| `test_training_stability.py` | 50-step training stability: loss finite, grad norm bounded, params finite, no divergence | `.tt-venv/bin/python test_training_stability.py` |

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

## Current training run (2026-08-13, gate freeze + fresh restart)

### Evaluation results (2026-08-13)

The existing `eval_ttnn.py` evaluator was fixed (batch-size-1 TT-Metal
matmul limitation — padded to batch 4) and run on all three cells'
latest checkpoints before the v2 restart.

| Cell | Step | Task 1 (chain) | Task 2 (2-var) | Task 3 (recall) | Overall | Depth 2 | Depth 8 |
|------|------|----------------|----------------|-----------------|---------|---------|---------|
| A | 9600 | 0.0% | 97.5% | 100.0% | 65.8% | 66.7% | 63.3% |
| B | 8800 | 0.0% | 42.5% | 5.0% | 15.8% | 30.0% | 3.3% |
| C | 4100 | 2.5% | 40.0% | 40.0% | 27.5% | 43.3% | 20.0% |

Key findings:
- **None of the cells learned Task 1** (chained assignment arithmetic).
  Even A at step 9600 gets 0%. The multi-hop reasoning task is beyond
  what a 10M retention model can learn at any configuration tested.
- **B is worse than A** on Tasks 2 and 3 — the workspace actively hurt
  the backbone. 119 out of ~950 step attempts were wasted on
  restores/skips (12% of training time). Grad norm 250-2100 vs A's ~8.
- **C is better than B** (27.5% vs 15.8% overall) and has stable
  gradients (0 restores), but is still worse than A on Tasks 2/3.
- Teacher-forced accuracy matches free-generation accuracy for all cells,
  confirming the failures are reasoning failures, not decoding artifacts.

### Gate freeze fix (2026-08-13)

**Root cause of B underperforming A:** With ReZero gates initialized at
0, B should start as exactly A (workspace is a no-op at gate=0). But
the 3x gate LR caused gates to drift open due to noisy gradients before
the workspace had learned useful representations, destabilizing the
backbone through the workspace's gradient amplification paths.

**Fix:** Added `gate_freeze_steps` config option. During the freeze
period, `model.freeze_workspace_gates()` forces gates back to 0 after
each optimizer step, and the fp32 master is synced. The workspace is a
true no-op (identity) during freeze, so the backbone trains as if the
workspace doesn't exist. After the freeze, gates learn freely at 1x LR
(reduced from 3x/10x).

Changes:
- `model_ttnn.py`: Added `freeze_workspace_gates()` method to `TTWRAPModel`
- `train_ttnn.py`: Added `gate_freeze_steps` config reading and post-step
  freeze logic
- `train_text.py`: Same gate freeze logic added to the text training loop
- `configs/cell_b_tt.yaml`: `gate_freeze_steps: 3000`, gate LR 3x → 1x
- `configs/cell_c_attn_residual.yaml`: `gate_freeze_steps: 3000`,
  gate/AR LR 10x → 1x
- `configs/text_cell_c_tiny_challenges.yaml`: `gate_freeze_steps: 600`,
  gate/AR LR 10x → 1x

### Active runs

Cell A completed training (step 9999, loss 0.83). Cells B and C were
restarted from scratch with the gate freeze fix. Device 3 is now used
for the text MVP (tiny challenges corpus).

| Run | Device | Config | Checkpoint dir | Log |
|-----|--------|--------|----------------|-----|
| Cell A (completed) | 3 | `cell_a_tt.yaml` | `checkpoints/stability_fix_a/` | `logs/cell_a_wrap_20260812.log` |
| Cell B v2 (fresh start) | 1 | `cell_b_tt.yaml` | `checkpoints/stability_fix_b_v2/` | `logs/cell_b_v2_20260813.log` |
| Cell C v2 (fresh start) | 2 | `cell_c_attn_residual.yaml` | `checkpoints/stability_fix_c_v2/` | `logs/cell_c_v2_20260813.log` |
| Text MVP (tiny challenges) | 3 | `text_cell_c_tiny_challenges.yaml` | `checkpoints/tiny_challenges/` | `logs/text_tiny_challenges_20260813.log` |

Old B and C checkpoints were deleted (13 GB + 6.1 GB). The old runs'
results are invalid — they trained without gate freeze, causing the
workspace to destabilize the backbone.

Training health at start (all gates at 0.0, grad_norm ~8.8 matching A):
- Cell B v2: loss 7.25, grad_norm 8.8 — stable, gates frozen
- Cell C v2: loss 7.30, grad_norm 8.8 — stable, gates frozen
- Text MVP: loss 10.9, grad_norm 1.5 — stable, gates frozen

### Launch commands

```bash
# Cell B v2 — device 1 (fresh start, gate freeze)
cd /home/rfenwick/Documents/jasper/workspace-poc
TT_VISIBLE_DEVICES=1 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_ttnn.py \
    --config configs/cell_b_tt.yaml --device 0 \
    --checkpoint_dir checkpoints/stability_fix_b_v2 \
    >> logs/cell_b_v2_20260813.log 2>&1 &

# Cell C v2 — device 2 (fresh start, gate freeze)
TT_VISIBLE_DEVICES=2 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_ttnn.py \
    --config configs/cell_c_attn_residual.yaml --device 0 \
    --checkpoint_dir checkpoints/stability_fix_c_v2 \
    >> logs/cell_c_v2_20260813.log 2>&1 &

# Text MVP — device 3 (tiny challenges corpus)
cd /home/rfenwick/Documents/jasper/workspace-mvp
TT_VISIBLE_DEVICES=3 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_text.py \
    --config configs/text_cell_c_tiny_challenges.yaml --device 0 \
    --checkpoint_dir checkpoints/tiny_challenges \
    > logs/text_tiny_challenges_20260813.log 2>&1 &
```

### Evaluation

The evaluator (`eval_ttnn.py`) was fixed to handle TT-Metal's batch-size-1
matmul limitation (padded to batch 4, using row 0 only). It also now
calls `model.clear_caches()` between examples to prevent device DRAM
leaks during autoregressive generation.

```bash
cd /home/rfenwick/Documents/jasper/workspace-poc
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python eval_ttnn.py \
    --config configs/cell_b_tt.yaml --device 0 \
    --checkpoint checkpoints/stability_fix_b_v2/cell_B_stepXXXX.pt \
    --n-per-task 10 --depths 2 4 6 8 --show-samples 5
```

`eval_loop.py` provides automated checkpoint monitoring and evaluation,
with plateau detection (no Task 1 improvement over 3 consecutive evals).

### Directory rename (2026-08-04)

The project directories were renamed on 2026-08-04:
- `mamba-poc` → **`workspace-poc`** (synthetic arithmetic training)
- `mamba-mvp` → **`workspace-mvp`** (text POC)

The architecture has evolved well beyond Mamba (retention layers + workspace
+ recurrent core), so the "mamba" prefix was dropped. All symlinks in
`workspace-mvp/` point to `workspace-poc/` for shared model code.

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

**The key threshold is step 1550.** Cell C has now passed step 3500 with
grad_norm ~4.7 and no divergence — the chain scaling fix is confirmed.
Watch for grad norm climbing from ~40 to ~200 over 200-300 steps — that's
the early warning pattern before each previous divergence.

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

Config: `configs/cell_c_attn_residual.yaml` — the primary Cell C config,
with `attention_residual_core: true`.

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

### Gradient stability fix (2026-08-10) — root cause of training divergence

After the gate clamping fix, both Cell B and Cell C still diverged:
- Cell B: stable for 600 steps → grad norm jumped from 4.9 to 569 at step
  650 → 7765 at step 700 (skip-on-spike fired, model frozen permanently)
- Cell C: stable for 2100 steps → grad norm jumped from 4.3 to 116 at
  step 2100 → 116K at step 2200

The gate clamping fix addressed the *consequence* (gate values too large)
but not the *cause* (gate gradient variance). A diagnostic analysis
revealed that **backbone ReZero gate gradients**
(`layer_N_gate`) are the dominant source of instability:

| Parameter | Mean grad | Max grad | Variance (max/min) |
|-----------|-----------|----------|---------------------|
| Cell B `layer_0_gate` | 11.21 | 71.50 | 8677x |
| Cell C `layer_0_gate` | 2.61 | 31.00 | 127x |
| Cell C `layer_7_gate` | 0.92 | 9.81 | 1568x |

The ReZero gate gradient is `grad_gate = sum(grad_out * inner_output)` — a
dot product of the backward gradient and the layer's forward output. When
these align on a particular batch, the gradient is enormous; when
orthogonal, it's near-zero. This is a known high-variance estimator.

**Root cause: three converging factors**

1. **Constant LR + sharp loss landscape = bifurcation.** The grad norm
   trajectory is not a gradual climb — it's a sudden jump. The model
   oscillates around a minimum near a cliff edge, and eventually a
   stochastic step pushes it past the edge. The constant LR (2e-4 for all
   10000 steps) keeps oscillation amplitude fixed even as the model
   converges. Cosine LR decay was already implemented in `get_lr()` but
   not configured.

2. **beta2=0.95 is too low for gate gradient variance.** With 8677x
   variance in gate gradients, AdamW's second moment estimate `v_hat`
   only averages over ~20 steps (beta2=0.95 → effective window 1/(1-0.95)
   = 20). A single spike dominates v_hat, making the update
   `m_hat / sqrt(v_hat)` unstable. With beta2=0.999, v_hat averages over
   ~1000 steps, providing stable normalization.

3. **Weight decay on gates + no recovery from skip-on-spike.** Weight
   decay (0.1) pulls gates toward 0, fighting their growth. And
   skip-on-spike freezes the model permanently when it enters a bad
   state — 934 consecutive skipped steps for Cell B, with no recovery.

**Fix (four changes, all config + optimizer)**

1. **Cosine LR decay** (`cosine_decay_steps: 8000`, `cosine_min_ratio: 0.1`):
   After 200-step warmup, LR decays from 2e-4 to 2e-5 over 8000 steps,
   then stays at 2e-5. This shrinks step size as the model converges,
   preventing the bifurcation. Already implemented in `get_lr()` — just
   needed config. Text POC uses `cosine_decay_steps: 4000` (5000 total
   steps).

2. **Per-parameter beta2** (`beta2_groups: {"suffix:_gate": 0.999}`):
   New `beta2_groups` config field in `TTAdamW` (same matching syntax as
   `lr_groups`/`wd_groups`). Backbone ReZero gates use beta2=0.999
   instead of the global 0.95, giving stable second-moment estimation
   despite extreme gradient variance. All other parameters keep beta2=0.95.

3. **Exclude gates from weight decay** (`wd_groups: {"suffix:_gate": 0.0}`):
   ReZero gates are scalar parameters meant to grow from 0 via gradient
   descent. Weight decay pulls them back toward 0, fighting their purpose.
   This is the standard convention for biases/norms/scalars — gates are
   analogous.

4. **Restore-on-spike** (`spike_action: restore`): When grad norm exceeds
   the threshold (5000), reload the last checkpoint (model + optimizer
   state) instead of skipping the step. Skip-on-spike freezes the model
   in the bad state that caused the spike — it can never recover. Restore
   gives it a fresh start from a known-good state (at most 100 steps old).
   The data RNG is not restored (known limitation), so the triggering
   batches won't reoccur in the same order. Default is `"skip"` for
   backward compatibility.

All four changes are in `configs/cell_b_tt.yaml`, `configs/cell_c_attn_residual.yaml`,
and `configs/text_cell_c.yaml`. The optimizer changes are in `train_ttnn.py`
(`TTAdamW.__init__` and `TTAdamW.step`).

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

## Deprecated SSM checkpoints (removed)

`checkpoints/*.pt` (pure-PyTorch SSM ablation cells, from `model.py`/
`train.py`) and `run_E_v2_mamba2/` (a deprecated SSM layer fallback checkpoint dump) have
been deleted — the deprecated SSM layer is no longer used in favor of the fixed retention layer.

## Cell rename (2026-07-31)

Cells were renamed after the pure-SSM (old A) and SSM+attention at
lr=6e-4 (old B) cells were deprecated and removed. Mapping: old E→A,
old C→B, old D→C. Run directories renamed: `run_E`→`run_A`, `run_C`→`run_B`,
`run_D`→`run_C`. PID files renamed accordingly. Old `run_C_v1_degenerate/`
and `run_D_v1_degenerate/` directories deleted. Config files renamed:
`cell_e_tt.yaml`→`cell_a_tt.yaml`, `cell_c_tt.yaml`→`cell_b_tt.yaml`,
`cell_d_tt.yaml`→`cell_c_tt.yaml` (the old `cell_c_tt.yaml` was later
superseded by `cell_c_attn_residual.yaml` and removed). Non-TT config variants (`cell_c.yaml`,
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
model architecture on text with GPT-2 BPE tokenization. The current
active dataset is the **tiny challenges** corpus (mixed reasoning
puzzles), replacing the earlier TinyStories-only approach.

### Directory structure

```
workspace-mvp/
├── text_data.py          # BPE tokenizer + dataset loading + batch sampling
├── train_text.py         # Training loop (imports from workspace-poc/train_ttnn.py)
├── eval_text.py          # Perplexity evaluation + text generation
├── configs/
│   ├── text_cell_c.yaml              # Cell C AR on TinyStories (legacy)
│   └── text_cell_c_tiny_challenges.yaml  # Cell C AR on tiny challenges (active)
├── tools/
│   ├── prepare_tinystories.py
│   ├── prepare_logic_puzzles.py
│   ├── prepare_brainbashers_style.py
│   ├── prepare_babi.py
│   └── prepare_tiny_challenges.py    # 2M mixed corpus generator
├── data/
│   ├── tiny_challenges_train.txt         # ~292 MB, 78.7M tokens
│   ├── tiny_challenges_valid.txt         # ~3 MB, 787K tokens
│   └── *.tokens.pt                       # Pre-tokenized caches
├── model_ttnn.py -> ../workspace-poc/model_ttnn.py  (symlink)
├── kernels/ -> ../workspace-poc/kernels/
└── venv -> ../.tt-venv/
```

### Key differences from synthetic training

| Aspect | Synthetic (workspace-poc) | Text (workspace-mvp) |
|--------|--------------------------|----------------------|
| Data | Generated arithmetic tasks | Tiny challenges (mixed reasoning puzzles) |
| Vocab | 128 (char-level) | 50,257 (GPT-2 BPE) |
| Seq len | 128 | 512 |
| Labels | Answer tokens only (-100 elsewhere) | Every position (standard LM) |
| Loss | On-device (V×V identity matrix) | Host-side (for V > 2048) |
| Eval | Task accuracy with verifiers | Perplexity + text generation |
| Params | 10.5M | 29.8M (19.3M from embedding) |

### Tiny challenges corpus

The tiny challenges corpus is a 2M-example mixed reasoning dataset
inspired by Enigmata-style synthetic reasoning training. The mixture:

| Component | Fraction | Description |
|-----------|----------|-------------|
| Narrative logic puzzles | 40% | Multi-hop location, arithmetic, swap, attribute puzzles |
| BrainBashers-style puzzles | 55% | Attribute-chain, elimination, ordering puzzles |
| bAbI QA | 5% | HuggingFace `Muennighoff/babi` passages |

- Train: 2,000,000 examples (~292 MB, 78.7M tokens)
- Validation: 20,000 examples (~3 MB, 787K tokens)
- Generated by `tools/prepare_tiny_challenges.py` (requires bAbI source
  files from `tools/prepare_babi.py` first)

The shift from TinyStories to tiny challenges reflects the finding that
the synthetic POC tasks (Task 1 chained arithmetic) were not learnable
by a 10M retention model. The tiny challenges corpus tests whether
framing multi-hop reasoning as text continuation — with the workspace
and recurrent core — can learn reasoning that the synthetic POC could
not.

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

### Gate freeze in text training

`train_text.py` has its own training loop (it does not call the
`train_ttnn.py` loop), so gate freeze logic was added separately:
- Reads `gate_freeze_steps` from the config
- Calls `model.freeze_workspace_gates()` after each optimizer step while
  `step < gate_freeze_steps`
- Clamps gates to `[-gate_clamp_bound, gate_clamp_bound]` after unfreeze
- Syncs optimizer fp32 master copies after gate modifications

Config: `configs/text_cell_c_tiny_challenges.yaml` uses
`gate_freeze_steps: 600` and 1x gate/AR LR (reduced from 10x).

### Current text training run

- Config: `configs/text_cell_c_tiny_challenges.yaml`
- Device: 3
- micro_batch=8, accum_steps=48, effective_batch=384
- seq_len=512, effective_tokens/step=196,608
- lr=1e-4, max_steps=2000
- gate_freeze_steps=600
- Initial loss: 10.92 (≈log(50257), correct random init)
- Initial grad_norm: 1.5, gates 0.0 (frozen)
- Speed: ~112s/step (K=6 recurrent core on 30M model)
- Estimated total: ~62 hours for 2000 steps

### Launch (from workspace-mvp/):

```bash
# Smoke test
TT_VISIBLE_DEVICES=1 python train_text.py \
    --config configs/text_cell_c_tiny_challenges.yaml --steps 3 \
    --micro_batch 4 --accum_steps 1 \
    --checkpoint_dir /tmp/text_test --device 0

# Full training (tiny challenges corpus)
cd /home/rfenwick/Documents/jasper/workspace-mvp
TT_VISIBLE_DEVICES=3 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_text.py \
    --config configs/text_cell_c_tiny_challenges.yaml --device 0 \
    --checkpoint_dir checkpoints/tiny_challenges \
    > logs/text_tiny_challenges_20260813.log 2>&1 &

# Evaluation
TT_VISIBLE_DEVICES=1 python eval_text.py \
    --checkpoint checkpoints/tiny_challenges/cell_text_step500.pt \
    --config configs/text_cell_c_tiny_challenges.yaml --device 0 \
    --generate --prompt "Once upon a time"
```

## Memory leak investigation (2026-08-11)

### Summary

Comprehensive investigation of memory growth during training. The original
~40 GB/hour growth was reduced to ~2.8 GB/hour (14x improvement). The
remaining growth is a **ttnn runtime C++ leak**, not a model code bug.

### Fixes applied to model code

1. **Use-after-free in workspace backward**: `grad_write_attn` and
   `grad_read_attn` were being deallocated before `_softmax_backward`
   consumed them. Removed the premature deallocations.

2. **Reassignment leaks**: Intermediate tensors in `_l2_normalize_heads`,
   `_softmax_backward`, attention softmax backward, and nested sum
   reductions were being dropped without deallocation. Added explicit
   `_safe_deallocate` calls.

3. **`freeze_gamma` gradient dropping**: Gamma gradients were being dropped
   from `all_grads` via dict comprehension without explicit deallocation.
   Now explicitly deallocates dropped gradients.

4. **`old_grad_x` in backward loops**: The old `grad_x` was being silently
   dropped when reassigned in the post-core, pre-core, and core backward
   loops. Added explicit `_safe_deallocate(old_grad_x)`.

5. **`one_hot_emb` in cross_entropy_loss**: The embedding output was being
   reassigned via `ttnn.reshape` without deallocating the original. Added
   explicit deallocation of the pre-reshape tensor.

6. **Unsafe alias deallocations reverted**: `ttnn.reshape` returns a VIEW
   (same buffer address), not a copy. `ttnn.deallocate(force=False)` on a
   view frees the shared buffer, causing use-after-free. Reverted
   deallocations of reshape/permute results that share buffers with cached
   or persistent tensors.

### Key ttnn behavior discovered

- `ttnn.reshape` returns a **view** (same buffer address as input)
- `ttnn.transpose`, `ttnn.permute`, `ttnn.slice` return **copies** (new
  buffer)
- `ttnn.deallocate(force=False)` on a view **succeeds** and frees the
  shared buffer — it does NOT detect that another tensor shares the buffer
- This means `_safe_deallocate` is NOT safe for reshape results that
  share buffers with tensors still in use

### Remaining growth: ttnn runtime C++ leak

**Measurement**: 0.27 MB/iter for Cell A (14 layers), linear over 500+
iterations.

**Evidence that it's a runtime leak, not model code**:

| Test | Rate | Notes |
|------|------|-------|
| Single TTRetentionLayer fwd+bwd | 0.04 MB/it | Decelerating, plateaus |
| TTGatedResidualLayer fwd+bwd | 0.04 MB/it | Decelerating, plateaus |
| Cross-entropy loss alone | 0.017 MB/it | Linear, small |
| ttnn.rms_norm in isolation | 0.001 MB/it | Negligible |
| ttnn.linear in isolation | 0.001 MB/it | Negligible |
| ttnn.embedding in isolation | 0.002 MB/it | Negligible |
| ttnn.softmax in isolation | 0.006 MB/it | Decelerating |
| ttnn.from_torch + deallocate | 0.000 MB/it | No leak |
| Inline transpose in matmul | 0.002 MB/it | Negligible |
| Full model (1 layer) | 0.13 MB/it | Decelerating |
| Full model (14 layers) | 0.27 MB/it | **Linear** |
| Full model (14 layers) + jemalloc | 0.27 MB/it | No improvement |
| Full model + sync every layer | 0.28 MB/it | No improvement |
| Forward only (14 layers) | 0.13 MB/it | Decelerating |
| Forward + loss (no backward) | 0.094 MB/it | Linear, small |
| Forward + backward (no loss) | 0.257 MB/it | **Linear** |
| LM head + norm backward only | 0.135 MB/it | Decelerating |
| 14x layer backward only | 0.253 MB/it | **Linear** |
| 1x fwd+bwd (single layer) | 0.018 MB/it | Linear, tiny |
| 14x fwd+bwd (same layer) | 0.257 MB/it | **Linear** |
| 14x fwd+bwd (14 different layers) | 0.250 MB/it | **Linear** |
| 14x linear with inline transpose | 0.024 MB/it | Linear, small |
| 14x linear with pre-cached transpose | 0.016 MB/it | Linear, small |
| Full model + pre-cached weight transposes | 0.27 MB/it | No improvement |

- `tracemalloc` shows **0.0 MB** Python-level growth — all growth is in
  C++ heap
- `/proc/self/smaps` shows growth is **entirely anonymous memory** (host
  RAM), not device memory mappings
- `gc.collect()` + `malloc_trim(0)` every iteration does NOT reduce growth
- `LD_PRELOAD=libjemalloc.so.2` does NOT reduce growth
- Synchronizing after every layer does NOT reduce growth
- Pre-caching all weight transposes does NOT reduce growth
- JIT cache shows 100% hits — not from kernel compilation
- Individual ttnn ops show ~0.001 MB/iter — negligible
- Growth is proportional to number of layers (model complexity)
- Growth is LINEAR — does not plateau over 500 iterations

**Conclusion**: The ttnn runtime leaks ~0.4 KB per ttnn operation in C++
internal state (command queue metadata, allocator bookkeeping, etc.).
With ~700 ops per iteration (14 layers × ~50 ops each), this gives
~0.27 MB/iter. This is a ttnn runtime bug, not a model code bug.

**Binary search isolation**: The leak is in the backward pass, not the
forward. Forward-only growth decelerates (allocator warmup). Backward
growth is linear and proportional to the number of backward calls:
- 1 backward call/iter: 0.018 MB/iter
- 14 backward calls/iter: 0.257 MB/iter (14 × 0.018 = 0.252, matches)
- Using the same layer 14 times vs 14 different layers: identical leak
- Pre-caching weight transposes: no improvement (not from transposes)
- The leak is per-op in the ttnn C++ runtime, not per-layer or per-tensor

### Production training impact

At ~3 steps/sec:
- Cell A (14 layers): ~2.8 GB/hour
- Cell B/C (14 layers + workspace + recurrent): likely ~3-7 GB/hour

With 128 GB RAM, training can run for ~18-40 hours before OOM. For longer
runs, periodic restarts (every 12-24 hours) are needed. Checkpoint
save/load is working, so restarts can be seamless.

### Root cause identified and patched (2026-08-12)

The leak has **two distinct components**, both in the ttnn/tt-metal C++
runtime. Both grow the brk heap and are NOT reclaimed by
`close_device()`, `clear_program_cache()`, `malloc_trim(0)`, or
`gc.collect()`.

**Both components have been patched by rebuilding tt-metal from source.**
The patched `.so` files are installed in the pip package. See the
"Patches applied" section below.

#### Component 1: `TieRuntimeIdToProgramId` map growth (common path)

Every ttnn operation enqueue calls `update_program_dispatch_commands()`
which calls `TieRuntimeIdToProgramId(program)`. This inserts a new entry
into `DataCollector::runtime_id_to_program_id_` (an
`std::unordered_map<uint16_t, uint64_t>`) keyed by the runtime_id
(narrowed to `uint16_t` from a monotonically increasing `int64_t`).

- Source: `tt_metal/impl/dispatch/data_collector.cpp:134-140`
- Called from: `tt_metal/impl/program/dispatch.cpp:2310`
- The map is bounded at 65536 entries (uint16_t key space) but each
  insert allocates a hash map node on the heap, interleaved with
  `create_descriptor()` allocations, preventing coalescing.
- This affects ALL operations that dispatch programs, including those
  using the fast descriptor path.

#### Component 2: `create_descriptor()` slow path (binary_ng)

Operations that use `ProgramDescriptor` without `emplace_runtime_args()`
take the "slow path" in `DescriptorAdapter::apply_descriptor()` on every
cache hit. This calls `create_descriptor()` which allocates a fresh
`ProgramDescriptor` with kernel source strings, defines vectors, CB
descriptors, and runtime argument vectors (~7 KB total), then frees it
when the local variable goes out of scope.

- Source: `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:598-603`
- The freed memory can't be coalesced because the
  `TieRuntimeIdToProgramId` hash map nodes are interleaved in the heap.
- `binary_ng` operations (`ttnn.mul`, `ttnn.add`, `ttnn.sub`, etc.) use
  this slow path because their factory uses direct
  `runtime_args.emplace_back()` instead of `emplace_runtime_args()` with
  buffer bindings.
- Operations with `emplace_runtime_args()` (e.g., `ttnn.typecast`) take
  the fast path and avoid this component.

#### GDB backtrace evidence

`gdb` `catch syscall brk` during a `ttnn.mul` loop (after skipping
initialization) shows all brk calls originating from:

```
#0  __brk
#1  __GI___sbrk (increment=135168)
#5  __GI___libc_malloc (bytes=5120)
#6  operator new(unsigned long)
#7  BinaryNgDeviceOperation::ProgramFactory::create_descriptor(...)
#9  handle_mesh_adapter_cache_hit<...>()
#10 launch<...>()
#11 ttnn::prim::binary_ng(...)
```

The 5120-byte `operator new` is inside `create_descriptor()`, confirming
the slow path as the brk growth trigger.

#### Measured leak rates (before patch)

| Operation | Path | Leak rate | Notes |
|-----------|------|-----------|-------|
| `ttnn.mul` | slow (create_descriptor) | 811 B/op | Both components |
| `ttnn.typecast` | fast (emplace_runtime_args) | 406 B/op | Common path only |
| `ttnn.empty` | no program dispatch | 0 B/op | No leak |

Thread-safe malloc counter: 200 net unfreed allocations per 100 mul ops
(2 per op), 82 KB net retained. `ttnn.empty` has 0 net allocations.

#### Bounded vs unbounded behavior

The isolated `ttnn.mul` leak **plateaus** after ~10,000 ops because the
`runtime_id_to_program_id_` map's uint16_t key space fills up. After
plateau: 0 KB growth for 90,000+ ops. Total bounded leak: ~9.4 MB.

However, the full WRAP model uses 53 unique ttnn operations with
varying shapes, creating many more unique program cache entries. The
combined heap fragmentation from different-sized `create_descriptor()`
allocations does NOT plateau within 2,000 iterations — it shows linear
growth at 0.26 MB/iter. This is why the C++ patch is necessary.

#### What does NOT reclaim the leak (before patch)

- `device.disable_and_clear_program_cache()` — 0 KB reclaimed
- `ttnn.close_device()` + `ttnn.open_device()` — 0 KB reclaimed
- `malloc_trim(0)` — 0 KB reclaimed (actually grew)
- `gc.collect()` — 0 KB reclaimed

The leak is in global/static `DataCollector` state that persists across
device lifecycle.

### Patches applied (2026-08-12)

tt-metal was cloned at commit `7f6364a11dafadf141b6c87358073d9e3d1dd22f`
(matching the pip-installed `pjrt-plugin-tt` v1.3.0), patched, rebuilt
with clang-20, and the resulting `.so` files replaced in the pip package.

Source tree: `/home/rfenwick/Documents/tt-metal-src/`
Build directory: `/home/rfenwick/Documents/tt-metal-src/build_Release/`

#### Patch 1: Flat array for `runtime_id_to_program_id_`

Replaced `std::unordered_map<uint16_t, uint64_t>` with a flat
`std::vector<uint64_t>` of size 65536, lazily initialized on first use.

- File: `tt_metal/impl/dispatch/data_collector.hpp`
  - Replaced `std::unordered_map<uint16_t, uint64_t> runtime_id_to_program_id_`
    with `std::vector<uint64_t> runtime_id_to_program_id_` plus a
    `kInvalidProgramId` sentinel.
  - Added `#include <limits>`.
- File: `tt_metal/impl/dispatch/data_collector.cpp`
  - `TieRuntimeIdToProgramId()`: lazily resizes the vector to 65536
    entries on first call, then does a direct array write. No per-op
    heap allocation after initialization.
  - `GetKernelSourcesForRuntimeId()`: direct array index instead of
    `map.find()`.

This eliminates Component 1 — the common-path heap fragmentation from
hash map node inserts. The one-time cost is a 512 KB allocation
(65536 * 8 bytes) on first operation dispatch.

#### Patch 2: Cache hash fix + `buffer_bindings` for binary_ng (APPLIED)

**Status: Applied and validated. The fast path now works correctly for
binary_ng, eliminating per-cache-hit descriptor allocation.**

The original problem was that the binary_ng `buffer_bindings` fast path
caused device hangs when both reader and writer bindings were enabled.
Investigation revealed the root cause was a **cache hash collision**, not
a dispatch-level issue.

**Root cause (confirmed by validation instrumentation 2026-08-12):**

`BinaryNgDeviceOperation::compute_program_hash()` in
`binary_ng_device_operation.cpp` hashed on dtype, memory_config, and
shard_volumes — but **not tensor shapes**. For interleaved (non-sharded)
tensors, `shard_volumes` is `std::nullopt`, so two binary_ng calls with
different tensor shapes but same dtype and memory config hashed identically.

Within a single forward pass of a 14-layer model, different layers' binary_ng
operations have different tensor shapes but identical kernel structure. The
first call cache-misses and creates the program. Subsequent calls cache-hit
on the wrong program entry. The fast path then patches only buffer addresses,
leaving all shape-dependent runtime args (tile counts, start IDs, dimensions,
strides) stale from the cache-miss call.

Validation instrumentation (temporary, since removed) compared every runtime
arg after fast-path patching against a fresh descriptor rebuild and found
78,304 mismatches across almost every arg in every kernel, starting in the
first forward.

**Why reader-only and writer-only bindings each passed but combined hung:**

With only reader bindings, the reader had stale shape args but the writer
fell through to the slow path (correct args). The CB handshake completed
because the writer's correct parameters dominated. Same in reverse for
writer-only. With both bound, both had stale shape args — the reader and
writer disagreed on tile counts, causing CB synchronization deadlock.

**Fix (two parts):**

1. **`compute_program_hash`** (`binary_ng_device_operation.cpp`): Added
   `input_tensor_a.tensor_spec().padded_shape()` and
   `input_tensor_b.tensor_spec().padded_shape()` to the hash. Now
   different tensor shapes produce different cache keys, so cache hits
   only occur when shapes match — meaning only buffer addresses need
   patching, which is exactly what the fast path does.

2. **`buffer_bindings`** (`binary_ng_program_factory.cpp`): Registered
   buffer address bindings for reader (arg[0] = a, arg[7]/arg[15] = b)
   and writer (arg[0] or arg[1] = c) kernels. On cache hits with matching
   shapes, `apply_resolved_bindings()` patches only the buffer addresses,
   avoiding a full `create_descriptor()` call and its ~7 KB allocation.

**Validation results (2026-08-13, all 4 patches active):**

All 4 patches: (1) DataCollector flat vector, (2) padded_shape in hash,
(3) buffer_bindings for cache-hit fast path, (4) get_dynamic_runtime_args
for scalar re-application on cache hits.

- Tier 1 — Hang + scalar correctness:
  - 2fwd 14-layer test: PASS (no hang, fwd 0: 0.077s, fwd 1+: 0.012s cache hit)
  - Scalar mul: ttnn.mul(ones, 0.5)=0.5, ttnn.mul(ones, 2.0)=2.0 — PASS
- Tier 2 — Leak test (retention_backward, 50 iters with clear_caches):
  0.25 MB/iter (matches original working version)
- Tier 3 — CPU pytest: 85 passed + 16 gradient tests passed
- Tier 4 — Device correctness:
  - retention_parity: ALL PASS (gradcheck, forward, backward)
  - backward_parity: ALL PASS (4/4, including attention_residual_inactive)
  - checkpoint_roundtrip: 5/5 PASS
  - optimizer_state: 4/4 PASS
  - clear_caches: 6/6 PASS
  - params: 3/3 PASS
  - clip_grad_norm: 5/5 PASS
  - recurrent_core: 4/4 PASS
- Tier 5 — Custom kernels:
  - scale_decay: PASS
  - fused_rope (fwd+bwd): PASS
  - gate_bwd: PASS
- Tier 6 — Training stability (50 steps):
  - All 50 losses finite (range 4.8710–5.0323)
  - All 26 params finite after 50 steps
  - grad_norm stable (~1.42), no divergence

Previously classified pre-existing failures (fixed 2026-08-13):

1. **gate_bwd** (was rel_err 1.0 → now 0.003): `generic_op` requires a
   prior descriptor-based ttnn op to initialize dispatch state. Without
   it, the custom kernel produces all-zeros output. In training, the
   forward pass runs first so this never manifested, but the standalone
   test called the kernel directly after `open_device`. Fixed by adding
   a warmup `ttnn.mul` call in `test_gate_bwd.py` before the kernel.

2. **attention_residual_inactive ar_scale** (was rel_err 0.10 vs tol 0.08
   → now 0.0002): The `AttentionResidual.backward` computed `grad_scale`
   entirely in bf16 on device. With few active iterations (K_active=1),
   the compounded bf16 rounding in `scores_pre_scale`, softmax (`alpha`),
   and softmax backward produced ~10% relative error vs the fp32
   reference. Fixed by recomputing the entire `grad_scale` path (scores,
   alpha, softmax backward, accumulation) in fp32 on host from the
   cached bf16 inputs. The device-side `grad_scores` is still used for
   `grad_x` and `grad_query` paths. Transfer cost is negligible
   (grad_scale is a scalar).

The residual 0.25 MB/iter is from other operations' descriptor rebuilds
and allocator fragmentation, not binary_ng. The fast path eliminates
binary_ng's contribution to per-iteration allocation.

**Key fix (2026-08-13):** The hang was caused by using `logical_shape()`
in the hash instead of `padded_shape()`. `logical_shape()` does not
capture tile padding, causing hash collisions between operations with
different tile-padded shapes but the same logical shape. The
`buffer_bindings` fast path then patched buffer addresses on the wrong
cached program, causing CB synchronization deadlock. Using
`padded_shape()` (matching the original working version) eliminates
these collisions.

#### Build and install

```bash
# Install build deps
sudo apt install clang-20 ninja-build libnuma-dev libhwloc-dev \
  libssl-dev pkg-config libcapstone-dev patchelf

# Capstone header is at /usr/include/capstone/capstone.h but Tracy
# expects <capstone.h> — create a symlink:
sudo ln -sf /usr/include/capstone/capstone.h /usr/include/capstone.h

# Clone at the exact commit
git clone https://github.com/tenstorrent/tt-metal.git tt-metal-src
cd tt-metal-src
git fetch --depth 1 origin 7f6364a11dafadf141b6c87358073d9e3d1dd22f
git checkout 7f6364a11dafadf141b6c87358073d9e3d1dd22f
git submodule update --init --recursive --depth 1

# Build with Tracy enabled (must match original pip package configuration)
pip install cmake ninja  # in the venv
./build_metal.sh --cxx-compiler-path clang++-20 --c-compiler-path clang-20 \
  --release --without-distributed
ninja -C build_Release install

# Back up and replace .so files in pip package
PKG_DIR=.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/lib64
cp $PKG_DIR/libtt_metal.so $PKG_DIR/libtt_metal.so.bak
cp $PKG_DIR/_ttnncpp.so $PKG_DIR/_ttnncpp.so.bak
cp $PKG_DIR/_ttnn.so $PKG_DIR/_ttnn.so.bak
cp build_Release/lib/libtt_metal.so $PKG_DIR/
cp build_Release/lib/_ttnncpp.so $PKG_DIR/
cp build_Release/lib/_ttnn.so $PKG_DIR/
cp build_Release/lib/_ttnn.so .tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/ttnn/ttnn/_ttnn.so

# Fix RPATH so the rebuilt library finds both bundled and system libs
patchelf --set-rpath '$ORIGIN:$ORIGIN/../../pjrt_plugin_tt.libs:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu' $PKG_DIR/libtt_metal.so
patchelf --set-rpath '$ORIGIN:$ORIGIN/../../pjrt_plugin_tt.libs:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu' $PKG_DIR/_ttnncpp.so
patchelf --set-rpath '$ORIGIN/../../lib64:$ORIGIN/../../lib64/../../pjrt_plugin_tt.libs:/usr/lib/x86_64-linux-gnu' .tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/ttnn/ttnn/_ttnn.so

# Set TT_METAL_HOME so JIT compilation can find kernel sources
export TT_METAL_HOME=/path/to/tt-metal-src
```

### Verification (2026-08-12)

#### Isolated `ttnn.mul` leak test (100,000 ops)

**Before patch**: 9,428 KB total, still growing at 70k ops, finally
plateaus at ~9.4 MB. Initial rate 866 B/op, decelerating.

**After patch (Component 1 only)**: 6,600 KB total, **plateaus at 10k
ops with 0 KB growth for the remaining 90,000 ops**. Initial rate 676
B/op (one-time array init), then 0 B/op.

The 6.6 MB one-time cost is the flat array initialization (512 KB) plus
JIT compilation and program cache warmup. After that: **zero leak**.

#### Gradient correctness tests

All 12 gradient tests pass with the patched library:
- RMSNorm, Softmax, RoPE, Cross-entropy, Retention, Attention,
  AttentionResidual, GatedResidual, Full model gradchecks: ALL PASS

#### Full model smoke test

Cell A training smoke test (3 steps, forward + backward + optimizer +
checkpoint) runs successfully with the patched library:
- 14 layers, d_model=384, 10M params
- 0.34s/step, 1486 tokens/sec
- Checkpoint saved successfully

#### Full model leak test (2000 iterations, Cell A)

**Before patch**: 0.26 MB/iter linear growth (512 MB over 2000 iters).
**After patch (Component 1 only)**: 0.26 MB/iter linear growth (512 MB
over 2000 iters). **No improvement.**

Component 1 eliminates the common-path leak (verified in isolated
`ttnn.mul` tests: 0 KB growth after 10k ops). But the full model's
linear growth was dominated by Component 2 — the `create_descriptor()`
slow path in `binary_ng` and other operations that use
`ProgramDescriptor` without `emplace_runtime_args()`.

**After adapter fix (2026-08-12)**: The adapter's
`resolve_bindings`/`apply_resolved_bindings` mechanism now handles CB
buffer patching on cache hits for all operations. binary_ng uses the
slow path (descriptor rebuild) which correctly handles stale buffers.
Isolated leak tests show 0.12-0.25 MB/iter (MINOR — allocator
fragmentation, not a leak). The sustained linear growth is eliminated.

#### Component 2 (binary_ng) slow path — validated

The `buffer_bindings` approach for binary_ng was investigated and not
applied (see "Patch 2" section above). The slow path (descriptor rebuild)
is used for binary_ng cache hits. This correctly handles stale buffer
addresses by rebuilding the descriptor with current tensor buffers on
every cache hit. Combined with Component 1 (flat-array patch), the
full model leak is reduced to allocator fragmentation levels.

### Current status (2026-08-14)

- The patched `.so` files (Component 1 flat-array + binary_ng hash fix +
  Tracy disabled) are installed in the pip package. Original `.so` files
  are backed up as `.bak`.
- The tt-metal source tree with the patches is at
  `/home/rfenwick/Documents/tt-metal-src/`.
- **The full model leak is fixed.** Three components were addressed:
  1. Component 1 (DataCollector flat-array) eliminates the common-path
     heap fragmentation from hash map node inserts.
  2. Component 2 (binary_ng padded_shape hash + buffer_bindings) enables
     the fast path for binary_ng cache hits, avoiding descriptor rebuilds.
  3. Component 3 (Tracy profiler disabled) eliminates the dominant leak
     source — see below.
- **Component 3: Tracy profiler leak (2026-08-14, the dominant leak)**
  - The build had `ENABLE_TRACY=ON` and `TRACY_ON_DEMAND=OFF`.
  - Without `TRACY_ON_DEMAND`, every `ZoneScoped` macro unconditionally
    enqueues to an unbounded `moodycamel::ConcurrentQueue` that grows via
    `rpmalloc` → `mmap(MAP_ANONYMOUS)`.
  - No Tracy client was connected to drain the queue, so it grew linearly.
  - This was the source of the ~350 KB/step anonymous mmap growth that
    caused OOM after ~2000 optimizer steps (16.8 MB/step × 2000 = 33.6 GB).
  - The previous "0.25 MB/iter residual" was actually this Tracy leak,
    not allocator fragmentation as previously concluded.
  - Fix: Rebuilt with `ENABLE_TRACY=OFF` (via `--disable-profiler`).
  - Verification: 200-step test shows RSS plateaus at 338 MB (1 MB total
    growth, 0 KB/step after step 20). Anonymous mappings: 0 new mappings.
    All three .so files have zero rpmalloc symbol references.
- **Validation results (2026-08-14, with Tracy disabled):**
  - 200-step leak test: 336 -> 338 MB (1 MB, plateaus by step 20)
  - Anonymous mappings: 161 -> 161 (0 new), virtual size stable
  - Program cache: stable at 212
  - Previous validation (2026-08-12): 12/12 gradient tests, all smoke tests
- Training requires `TT_METAL_HOME=/home/rfenwick/Documents/tt-metal-src`
  to be set so JIT compilation can find kernel sources.
- If the patched library causes issues, restore the originals:
  ```bash
  PKG_DIR=.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/lib64
  cp $PKG_DIR/libtt_metal.so.bak $PKG_DIR/libtt_metal.so
  cp $PKG_DIR/_ttnncpp.so.bak $PKG_DIR/_ttnncpp.so
  cp $PKG_DIR/_ttnn.so.bak $PKG_DIR/_ttnn.so
  cp $PKG_DIR/_ttnn.so.bak .tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/ttnn/ttnn/_ttnn.so
  ```

### Root cause: _cache retention (2026-08-12, final fix)

**The remaining 98.6 KB/iter leak was in the Python model code, not the
C++ runtime.** Each layer's `forward()` caches intermediate tensors in
`self._cache` for use by `backward()`. After `backward()` completes,
`self._cache` still holds references to all those intermediates. On the
next forward pass, `self._cache` is overwritten with new tensors, but the
old tensors' device buffers were never explicitly deallocated — they
became orphaned device allocations waiting for Python GC.

**The training loop already calls `model.clear_caches()` after backward**
(see `train_ttnn.py:1128`), which deallocates cached intermediates and
drops references. This is the correct fix and is already in place.

**However, the leak tests above (0.12-0.25 MB/iter) were running without
`clear_caches()`**, which is why they showed residual growth. When
`clear_caches()` is called after every iteration (as the training loop
does), the leak is completely eliminated.

#### Verification (2026-08-12, with clear_caches)

**Full model (TTWRAPModel, Cell A config, 500 iterations):**
```
Baseline: used=164959KB
  iter   0: delta=+1247KB  rate=1247.922 KB/iter  (program cache fill)
  iter   9: delta=+1270KB  rate=127.017 KB/iter
  iter  49: delta=+1272KB  rate=25.452 KB/iter
  iter  99: delta=+1272KB  rate=12.725 KB/iter
  iter 199: delta=+1272KB  rate=6.363 KB/iter
  iter 299: delta=+1272KB  rate=4.243 KB/iter
  iter 399: delta=+1272KB  rate=3.182 KB/iter
  iter 499: delta=+1272KB  rate=2.546 KB/iter
```

Growth plateaus at 1272 KB after the first few iterations (one-time
program cache fill). Rate at 500 iterations: **2.5 KB/iter** (vs 98.6
KB/iter without clear_caches). **39x reduction**, and the remaining
growth is just the one-time program cache fill that would plateau with
more iterations.

**Single layer (TTRetentionLayer, 200 iterations, with clear_caches):**
```
Baseline: used=152746KB
  iter   0: delta=+75KB  rate=75.219 KB/iter
  iter  49: delta=+3624KB  rate=72.490 KB/iter
  iter  99: delta=+4856KB  rate=48.561 KB/iter
  iter 199: delta=+4856KB  rate=24.284 KB/iter
```

Plateaus at 4856 KB. No continuous leak.

#### What clear_caches() does

`model.clear_caches()` (in `model_ttnn.py:4853`) iterates all layers and:
1. Calls `ttnn.synchronize_device(device)` so async queue releases refs
2. Calls `_safe_deallocate()` on each cached intermediate (excluding
   persistent model parameters like weights, gates, scale tensors)
3. Sets `layer._cache = {}` to drop Python references
4. Calls `_deallocate_cache_history()` for recurrent core iterations
5. Calls `_deallocate_conv_cache()` if present

The training loop calls this after every gradient accumulation step
(`train_ttnn.py:1128`), so production training does NOT have this leak.

#### Why the leak tests showed residual growth

The leak tests in `test_memory_leak.py` and the isolated tests above
were running forward+backward without calling `clear_caches()`. This
is NOT how the training loop works — the training loop always calls
`clear_caches()` after backward. The leak tests were measuring the
orphaned intermediate retention, not a C++ runtime leak.

#### Summary of all leak fixes

1. **C++ Component 1** (flat array for `TieRuntimeIdToProgramId`):
   Eliminates hash map node allocation fragmentation. Applied to
   `data_collector.cpp/hpp`.
2. **C++ Component 2** (binary_ng cache hash + buffer_bindings):
   Eliminates cache collisions that caused hangs. Applied to
   `binary_ng_device_operation.cpp` and `binary_ng_program_factory.cpp`.
3. **Python `clear_caches()`** (already in training loop):
   Eliminates orphaned intermediate tensor retention. This was the
   dominant remaining leak (~98.6 KB/iter) after the C++ fixes.

All three fixes are required for production training stability. The C++
fixes eliminate the bounded-but-large allocator fragmentation, and the
Python `clear_caches()` eliminates the per-iteration intermediate
retention that was the primary cause of the 12-14 GB/hr growth that
killed previous training runs.

### Test commands

```bash
# Reset devices after crashes
~/.tenstorrent-venv/bin/tt-smi -r

# CPU-only tests (no device required, ~2s)
/home/rfenwick/Documents/jasper/.tt-venv/bin/pytest -v

# Device tests (from workspace-poc/, one at a time on device 0)
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_checkpoint_roundtrip.py
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_optimizer_state.py
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_clear_caches.py
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_params.py
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_clip_grad_norm.py
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_recurrent_core.py
TT_VISIBLE_DEVICES=0 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_training_stability.py

# Memory leak tests (from workspace-poc/)
# train_ttnn.py auto-detects P300 and sets TT_MESH_GRAPH_DESC_PATH;
# for standalone test scripts that don't have auto-detection, set it manually:
TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
/home/rfenwick/Documents/jasper/.tt-venv/bin/python -u test_memory_leak.py --test retention_backward --iterations 200

# Available leak tests: custom_rope, custom_scale_decay, custom_gate_backward,
# program_descriptor, reshape, permute, transpose, slice,
# single_retention, single_attention, single_workspace,
# attention_residual, retention_forward, retention_backward,
# workspace_backward, recurrent_backward, cache_cleanup, per_layer
```
