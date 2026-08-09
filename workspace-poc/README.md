# Jasper POC — Code Guide

**Jasper** is a workspace-augmented retention network with recurrent core —
a novel architecture that combines RetNet-style linear attention, Perceiver-style
external memory (workspace), depth-recurrent iteration, and attention residuals
into a unified model. This is the code for a desktop-scale proof-of-concept that
tests whether Jasper reasons better than parameter-matched controls. The
experiment takes ~1 week and ~$20–30, and produces a go/no-go decision for a
$3–5K language-scale ablation.

See the [root README](../README.md) for the project overview and [desktop-jasper-workspace-poc.md](../desktop-jasper-workspace-poc.md) for the full experiment design.

---

## File map

| File | What it does |
|------|-------------|
| `data.py` | Three synthetic task generators with verifiers, character-level vocabulary, batch generation, and unit tests. Each task has a depth knob `k` controlling reasoning steps. |
| `model.py` | PyTorch reference model (`MambaWorkspaceModel`/`WorkspaceModule`) — no longer used for full training (`model_ttnn.py` is the active Jasper implementation), but kept as the CPU-side correctness reference for `test_ws_backward_cpu.py`. |
| `model_ttnn.py` | Jasper model implementation using `ttnn` ops (bfloat16). Same architecture as `model.py` but compiled for Blackhole chips. Includes the TTWorkspaceModule (perceiver-style workspace with QK-Norm and ReZero gates), custom fused kernels for RoPE, scale+decay, and gate backward, and the retention layer. |
| `mamba3_layer.py` | Fixed Mamba-3 layer (replaces Mamba-2's selective scan with decayed linear attention). Used unconditionally for all non-attention layers. |
| `retention_reference.py` | PyTorch reference implementation of the retention layer (decayed linear attention + RoPE). Used for parity testing. |
| `train_ttnn.py` | Tenstorrent-native training loop. Runs on Blackhole chips via `ttnn`, bfloat16-native computation. Used for the current Quietbox 2 experiments. |
| `eval_ttnn.py` | Checkpoint evaluation script for Tenstorrent. Loads a `.pt` checkpoint, runs the model on generated eval examples, reports accuracy per task and per depth. Supports JSON output for automated tracking. |
| `eval_loop.py` | Automated eval monitoring loop. Polls for new checkpoints every 5 minutes, runs `eval_ttnn.py` on each, saves JSON results, and detects plateaus (3 consecutive evals with <1% Task 1 improvement). |
| `monitor_training.sh` | Background monitor script. Reports training status every 10 minutes to `logs/monitor.log`. |
| `kernels/` | Custom tt-metal compute kernels: `rope4d_{reader,compute,writer}.cpp` (fused RoPE), `scale_decay_{reader,compute,writer}.cpp` (fused scale+decay), `gate_bwd_{reader,compute,writer}.cpp` (fused gate backward). |
| `test_retention_parity.py` | Parity test: tt-nn forward + backward vs. PyTorch reference autograd. Run with `--tile-aligned` for tile-aligned shapes. |
| `test_mamba3_parity.py` | Parity test for the Mamba-3 layer: float64 gradcheck + tt-nn forward/backward vs. reference. |
| `test_fused_rope_single.py` | Standalone test for the fused RoPE kernel (forward + backward). |
| `test_scale_decay.py` | Standalone test for the fused scale+decay kernel. |
| `test_gate_bwd.py` | Standalone test for the fused gate backward kernel. |
| `test_ws_backward_cpu.py` | CPU gradient check for the workspace backward pass. Verifies all gradients (input, weights, slots, gates) against PyTorch autograd to within float32 precision. |
| `test_ws_backward.py` | Device-side gradient check (earlier attempt, superseded by the CPU version). |
| `configs/cell_{a,b,c}_tt.yaml` | YAML configs for each model cell, including Tenstorrent-specific settings. Cell A is the no-workspace control; B adds workspace; C adds recurrent core. `cell_c_attn_residual.yaml` is the primary Cell C config (attention residual core). |
| `run_all_cells_parallel.sh` | Launches all three cells in parallel on separate Blackhole chips (one cell per device). |
| `tt_runner.py` | Sequential training runner for a single Blackhole chip. Supports `--cell`, `--status`, `--clean`, `--smoke` modes. |
| `colab_notebook.ipynb` | Google Colab T4 notebook for running cells A and C (the go/no-go pair) on free GPU time. (Deprecated — project moved to TT.) |
| `requirements.txt` | Python dependencies. |
| `checkpoints/` | Saved during training (gitignored). `cell_{X}_step{N}.pt` per cell per checkpoint interval. |
| `logs/` | Training logs (`cell_{a,b,c}.log`) and monitor log (`monitor.log`). |

**Removed (2026-08-09 cleanup):** `train.py`/`probe.py` (old PyTorch training + R2/R3/R4 analysis
pipeline, superseded by `train_ttnn.py`/`eval_ttnn.py` — note the R2/R3/R4 workspace-slot-probing
analysis currently has no `ttnn` equivalent, so that capability would need to be reimplemented
against `model_ttnn.py` if needed again), `colab_runner.py`/`vast_runner.py` (deprecated GPU
runners), `diagnose_explosion.py`/`diagnose_workspace.py`/`monitor_dynamics.py`/
`profile_retention.py`/`profile_ttnn.py` (one-off debugging/profiling scripts for gradient
instability issues that are now fixed and documented in `AGENTS.md`), and `loader.py` (an
unrelated, misplaced Tenstorrent Forge model-loader file with no references in this project).
Also removed: orphaned `run_A/`, `run_B/`, `run_C/` directories (symlinks pointing at the
pre-rename `mamba-poc/` path, which no longer exists), a stray venv accidentally created inside
this directory, and superseded checkpoints (`causal_fix_b/`, `causal_fix_c/`, and loose
pre-retention-layer `cell_{A,B,C}_step*.pt` files).

---

## The three tasks

All tasks are character-level, generated on-the-fly, and automatically verifiable. Training uses depths 2–8; evaluation extends to 2–16.

### Task 1 — Chained assignment arithmetic (multi-hop composition)

Variables are defined in terms of previous variables, all arithmetic mod 97. The model must follow the dependency chain to compute the queried variable's value.

```
a=7;b=a*3+2;c=b-a;d=c*2;?d;
```
Depth = chain length to the queried variable. Distractor variables (not part of the chain) are inserted at random positions.

### Task 2 — Permutation tracking (SSM stress test)

n labeled items undergo a sequence of swap operations. The model must track one item's final position.

```
n=6;2,5;0,3;5,1;?3;
```
Depth = number of swaps. State-tracking is a documented weakness of linear SSMs — this task tests whether the workspace compensates for the architecture's known gap.

### Task 3 — Single-hop recall (control)

Shallow lookup with distractors. Always depth 1.

```
a=5;b=12;c=8;d=3;e=15;?c;
```
Exists purely for the selective-ablation test (R4): the J-space signature requires that killing the workspace leaves this task intact while Tasks 1–2 collapse.

### Task mix during training

45% Task 1 / 45% Task 2 / 10% Task 3. Fresh examples are sampled every batch — with generated data there is no train set to overfit.

---

## The three cells

All cells are ~25–30M parameters, `d_model=384`, vocabulary ~128 (character-level). They form an ablation ladder — each adds one component so marginal contributions are isolated.

> **Rename note (2026-07-31):** Cells were renamed after the pure-Mamba2 (old A) and Mamba2+attention (old B) cells were deprecated and removed. Old E→A, old C→B, old D→C. See AGENTS.md for details.

| Cell | Architecture | Config flags | Key params |
|------|-------------|--------------|------------|
| **A** | Hybrid (Mamba-3 + 2 attention at positions 5, 10), no workspace | `use_attention=true, use_workspace=false, recurrent_core=false` | The no-workspace control — isolates workspace effect |
| **B** | Hybrid + workspace (Mamba-3 + 2 attention + 16-slot perceiver workspace with QK-Norm + ReZero gates) | `use_attention=true, use_workspace=true, recurrent_core=false` | Does an engineered workspace help without recurrence? |
| **C** | Full architecture (hybrid + workspace + layers 6–9 looped K=6 times with attention residuals) | `use_attention=true, use_workspace=true, recurrent_core=true, attention_residual_core=true` | The go/no-go cell |

Cell C's recurrent core (layers 6–9) is applied K times per forward pass. During training, K is sampled uniformly from {1…6} per batch. At inference, K can be swept (the `k_inference` config field or `--k_override` in code). K was temporarily reduced from 6 to 3 to limit gradient amplification, then restored to 6 after fixing the slot-state gradient scaling (see "Training stabilization" below).

The primary Cell C config is `cell_c_attn_residual.yaml`, which uses an **attention residual core** (Kimi K3-style) instead of the fixed blend. This stores all K iteration outputs and computes learned softmax attention over them, bounding output magnitude and providing consistent gradient magnitude across iterations. See "Attention Residual Core" below.

Cell B removes one Mamba layer (13 vs 14) to compensate for the ~2M workspace parameters, keeping cells parameter-matched.

### Comparison logic

| Comparison | What it isolates |
|------------|-----------------|
| **B vs A** | **Effect of workspace (same LR=2e-4 — clean)** |
| C vs B | Effect of recurrent core (same architecture base) |
| C vs A | Full architecture vs no-workspace control (the R1 test) |

### Model architecture details

The model has three phases in its forward pass:

1. **Pre-core** (layers 0 to `core_start`): standard sequential layer processing. Workspace read/write happens at attention positions.
2. **Recurrent core** (layers `core_start` to `core_end`): the core layers are applied K times. Workspace read/write happens inside the loop on every iteration, so each iteration revises the slot state — this is the TRM-style revision dynamic.
3. **Post-core** (layers `core_end` to `n_layers`): final layers decode the workspace state into token logits.

For cells without a recurrent core (A, B), all layers run once in the pre-core phase — there is no loop.

### The workspace module

The `TTWorkspaceModule` is a perceiver-style cross-attention block with 16 learned slot vectors. Each application does two steps:

1. **Read**: slots attend over hidden states (slots query, hidden states are keys/values)
2. **Write**: hidden states attend over slots (hidden states query, slots are keys/values)

Both passes use **QK Normalization** (L2-normalize Q and K before attention scores,
with learnable scale) to prevent entropy collapse, and **ReZero gates** (scalar,
init=0, no sigmoid) so the workspace starts as a true identity and grows naturally.

Slot state persists across recurrent iterations (passed as `slot_state`), so each loop iteration reads and revises the same workspace — the model can iteratively refine its reasoning in the slots rather than in the token stream.

### Training stabilization

The workspace and recurrent core introduce gradient instability that is not
present in the baseline hybrid model. Multiple rounds of fixes were attempted
(see "History of gradient instability" below), culminating in the current
**architecture v2** design which addresses the root cause rather than symptoms.

#### Architecture v2: QK-Norm + ReZero + component-wise clipping

Three architectural changes, all based on published techniques:

1. **QK Normalization** (`_l2_normalize_heads` in `TTWorkspaceModule`):
   L2-normalizes Q and K along the d_head dimension before computing attention
   scores in both read and write passes. A learnable scale parameter
   (`read_qk_scale`, `write_qk_scale`, initialized to 1/√d_head) replaces the
   fixed 1/√d_head scale. This bounds attention logits regardless of weight
   magnitudes, preventing the **entropy collapse** (ill-conditioned QK^T with
   condition numbers of 40,000–112,000) that caused all previous divergence
   events. (Henry et al., 2020 — now standard in OLMo 2, Gemma 3, Qwen 3)

2. **ReZero gates** (replaces sigmoid gates): Scalar gates initialized to 0.0
   (true identity at init), no sigmoid. Both the workspace gates and the
   backbone layer gates use ReZero. The workspace and backbone both start as
   no-ops and grow naturally through gradient descent. This gives the backbone
   time to learn a good representation before the workspace starts contributing,
   and keeps the gradient self-amplification at 1.0 per layer at init (vs
   sigmoid(0)=0.5 which gives 1 + 0.5×σ per layer). (Bachlechner et al., 2020)

3. **Component-wise gradient clipping** (`clip_grad_norm` in `train_ttnn.py`):
   Workspace parameters (ws_*) are clipped at `ws_grad_clip` (0.5), backbone
   parameters at `grad_clip` (1.0). Each group's norm is computed and clipped
   independently. This prevents workspace gradient spikes from dominating the
   global clip and starving the backbone of learning signal.
   (Yang et al., 2022 — EMNLP 2022)

#### Attention Residual Core (2026-08-04, Kimi K3-style)

The fixed blend core (`x = (1-a)*x + a*x_new` applied K times) creates a
multiplicative gradient path that amplifies across iterations. The fix
replaces it with **Attention Residuals**: store all K iteration outputs
(plus the pre-core input x_0) and compute softmax attention over them with
a learned query vector:

```
scores_k = sum_d(x_k * query) * scale       → (B, T) per k
alpha = softmax([scores_0, ..., scores_K])   → (B, T, K+1)
x_final = sum_k alpha_k * x_k                → (B, T, d_model)
```

This bounds output magnitude (softmax normalizes) and gives each iteration
direct gradient from the attention (bounded by softmax weights), rather than
through a chain of blend operations. The `AttentionResidual` class has two
learnable parameters: `ar_query` (1, d_model) and `ar_scale` (scalar).

Config: `attention_residual_core: true` in the YAML.

#### Chain gradient scaling (2026-08-04)

Even with the AR core, the slot state chain and the shared core layer
gradients compound across K iterations. The chained gradient from each
iteration is scaled by `1/(K × chain_scale_safety)` before adding the AR
gradient. With `chain_scale_safety: 1.0` (current), the per-iteration chain
gain is `A/K`, stable for A < K. The AR gradient itself is not scaled (it's
bounded by softmax).

#### Backbone ReZero gate fix (2026-08-06) — root cause of all divergences

The v2 fix changed the *workspace* gates to ReZero but **missed the backbone
layer gates** — they still used sigmoid with init=0, giving `sigmoid(0) = 0.5`.
This meant each backbone layer contributed 50% of its output at init, creating
a gradient self-amplification of `1 + 0.5 × σ_layer` per layer. Over 4 core
layers: `A_layers = 5.55`, `A_coupled ≈ 5.77`, `A/K = 0.96` — only 4% stability
margin. The system started training at the stability boundary and diverged
as soon as weights grew.

**Fix**: Changed `TTGatedResidualLayer` from sigmoid to ReZero (no sigmoid,
gate init=0.0). At init, each backbone layer is pure identity — amplification
= 1.0 per layer, `A_coupled ≈ 1.04`, `A/K = 0.17` — 83% margin. Gates grow
slowly through gradient descent. Even at gate=0.3 (60% of sigmoid's 0.5), the
margin is still 47%.

Also added `freeze_slot_decay: true` to prevent the slot_decay parameter from
growing above 1.0 (which would make the slot chain divergent).

See AGENTS.md for the complete history of gradient instability fixes.

#### Additional stabilizers (from earlier runs, still in place)

4. **Per-parameter LR groups** (`lr_groups` config field): Workspace
   parameters receive 0.25x the base learning rate (5e-5 vs 2e-4 for the
   backbone). Gates get 10x base LR so they open fast enough to contribute
   within the training budget. QK scale parameters get 1.0x (same as backbone).
   Config: `lr_groups` in the YAML.

5. **1/sqrt(K) residual scaling** (in `model_ttnn.py`): The recurrent core's
   blend factor is scaled by 1/sqrt(K), normalizing gradient accumulation
   across K iterations so the total gradient norm is independent of K. Based
   on the variance-preserving principle from DeepNorm.

6. **Slot parameter normalization** (in `model_ttnn.py`): The learned slot
   parameters are normalized to unit RMS after each optimizer step, breaking
   the positive feedback loop: large slots → sharp attention → large gradients
   → larger slots. Applied via `model.normalize_workspace_slots()`.

7. **Spectral normalization of workspace weights** (in `model_ttnn.py`): All
   eight workspace weight matrices have their spectral norm capped at C=5
   after each optimizer step. Uses 10 steps of power iteration on-device.
   Config: `spectral_norm_bound: 5.0`.

8. **Backbone spectral normalization** (in `model_ttnn.py`): Backbone qkv and
   out_proj weights are capped at `backbone_spectral_norm_bound` (2.0) after
   each optimizer step. The workspace feedback loop caused exponential backbone
   weight growth in earlier runs; this cap prevents it while allowing healthy
   growth (Cell A reaches ~1.6).

9. **Gamma freezing** (`freeze_gamma: true`): Retention gamma (decay parameter)
   is frozen — not optimizer-managed. The O(T²) gamma gradient was structurally
   unstable with the workspace. Gamma barely moves during training (drift <0.03
   from init), so freezing it loses nothing while eliminating that gradient
   path. (RetNet keeps γ fixed for the same reason — Sun et al., 2023)

10. **Slot decay** (`slot_decay_init: 1.0`, `freeze_slot_decay: true`): The slot
    update includes a decay scalar: `slots = norm(decay * slots + gate * read_out)`.
    This makes the slot update contractive — old information fades unless actively
    reinforced, providing a restoring force against attention sharpening. The
    decay is frozen at init to prevent it from growing above 1.0 (which would
    make the slot chain divergent).

11. **Chain gradient scaling** (`chain_scale_safety: 1.0`): The chained gradient
    from each recurrent core iteration is scaled by `1/(K × chain_scale_safety)`
    before adding the AR gradient. With ReZero backbone gates, the per-iteration
    amplification A starts at ~1.0, giving A/K = 0.17 at init — 83% margin. No
    extra safety factor is needed.

#### History of gradient instability

The workspace's two-pass cross-attention (read + write) creates a
multiplicative gradient amplification path. Across multiple rounds of fixes,
the instability pattern was consistent: training is stable for a while, then
gradient norms spike and divergence follows. Each fix addressed a layer of
the problem, peeling back to the next underlying cause.

| Run | Fix | Cell C diverges | Root cause |
|-----|-----|-----------------|------------|
| 1 | None (bound=3.0, gamma trainable) | step ~700 | Gamma gradient explosion |
| 2 | bb_bound=2.0, gamma clip | step ~700 | Same — gamma still unstable |
| 3 | bb_bound=2.0, gamma frozen | step ~900 | Entropy collapse in QK^T |
| 4 | Same as 3 (re-run) | step ~900 | Same |
| v2 | QK-Norm + ReZero (workspace) + comp clipping | step ~1000 | Fixed blend chain amplification |
| AR | Attention residual core | step ~1050 | Gradient double-counting in AR backward |
| AR-fix | AR double-counting fix | step ~1550 | Unscaled AR chain gradient |
| 1/K | Chain gradient scaling (1/K) | step ~2650 | Sigmoid backbone gates (A/K=0.96, 4% margin) |
| bound1.5 | backbone_spectral_norm_bound 2.0→1.5 | step ~2800 | Worse — out_proj grew to 1.5, A=5.96 |
| safety1.5 | chain_scale_safety 1.0→1.5 | step ~2636 | Checkpoint already in unstable regime |
| **ReZero** | **Backbone sigmoid→ReZero gates** | **stable (training)** | **A/K=0.17, 83% margin — root cause fixed** |

**Root cause** (identified 2026-08-06): The backbone layer gates used sigmoid
with init=0, giving `sigmoid(0) = 0.5`. This meant each of the 4 core layers
contributed 50% of its output at init, creating a gradient self-amplification
of `1 + 0.5 × σ_layer ≈ 1.535` per layer. Over 4 layers: `A_layers = 5.55`,
`A_coupled ≈ 5.77`, `A/K = 0.96` — only 4% stability margin. The system started
training at the stability boundary and diverged as soon as weights grew.

The v2 fix (2026-08-02) changed the *workspace* gates to ReZero but missed the
backbone layer gates. The backbone ReZero fix (2026-08-06) completes the v2
architecture: all gates (workspace + backbone) now use ReZero, giving A/K = 0.17
(83% margin) at init. See AGENTS.md for the complete fix history.

### Attention regularizers (entropy + diversity) — IMPLEMENTED, MEASURED HARMFUL, DISABLED

> **Do not re-enable these.** Set to 0 in all configs. At weight 0.01 they produced
> our lowest-ever answer-token loss (0.55 vs the control's 0.94) together with our
> *worst* evaluation (12.5% overall, 28% Task 2 — against 33%/85% for the same
> architecture without them and 62%/80% for the no-workspace control). Generated
> answers were near-constant regardless of input ("19", "13", "18", "18", ...).
>
> **Why:** entropy measures *peakedness*, not *input-dependence*. The cheapest way
> to minimize attention entropy is a **fixed** one-hot routing that ignores the
> input — zero entropy, zero information. The regularizer selected for exactly the
> degeneracy it was meant to cure. Probing the checkpoint with 16 distinct inputs:
> the write path had entropy 0.08–0.33 (very sharp) but TV 0.016 and **98.3% of
> positions routed to the same slot for every input**. A fixed positional bucket,
> not a content-addressed memory.
>
> Use the input-dependence metric (below) to measure selectivity. Never optimize entropy.

The original (now superseded) rationale follows, kept for reproducibility of the negative result.

Even with stable training, diagnostic inspection revealed that the workspace was in a **degenerate state**: read attention was nearly uniform (entropy 3.79/4.85 max), write attention was completely uniform (entropy 2.75/2.77 max), and slots did not change across recurrent core iterations. The workspace was acting as a global average pool rather than selective memory — every position wrote equally to every slot, and every slot read equally from every position.

This degeneracy arises because the cross-entropy loss provides only a diffuse gradient signal for workspace selectivity. The model gets the same loss whether the workspace averages or selects, so there's no pressure to become selective.

Two task-agnostic regularizers address this (both can remain active during language training):

5. **Attention entropy penalty** (`ws_entropy_weight: 0.01`): Minimizes the mean entropy of read/write attention distributions, pushing the workspace toward selective (low-entropy) attention. Directly targets the observed uniformity without specifying *what* to attend to.

6. **Slot diversity penalty** (`ws_diversity_weight: 0.01`): Minimizes the mean cosine similarity between pairs of slots, pushing the 16 slots to carry different information rather than collapsing to identical averaged representations.

Both are computed via host-side PyTorch autograd on the cached attention and slot tensors, with gradients injected into the workspace backward pass at the softmax and slot-state boundaries. They contribute ~5% of the total loss signal — a gentle nudge, not a dominant force. Config: `ws_entropy_weight`, `ws_diversity_weight` in the YAML configs.

### Architectural stability: ReZero gates and slot decay

Both the workspace and the backbone layers use **ReZero gates** (scalar,
init=0.0, no sigmoid). This makes the entire model start as a pure identity
at initialization — no layer contributes anything until its gate grows through
gradient descent. This is critical for the recurrent core: with sigmoid gates
(sigmoid(0)=0.5), the 4 core layers create a gradient self-amplification of
~5.5x per iteration, leaving only 4% stability margin under 1/K chain scaling.
With ReZero gates (gate=0), the amplification is 1.0x — 83% margin.

- **ReZero gates** (scalar, init=0.0, no sigmoid): Both workspace gates
  (read_gate, write_gate) and backbone layer gates use ReZero. The gates grow
  linearly through gradient descent — no sigmoid saturation, no fixed schedule.
  The gate-specific LR group (10x workspace LR for workspace gates, 1.0x for
  backbone gates) ensures the gates reach useful values within the training
  budget.

- **Slot decay** (`slot_decay_init: 1.0`, `freeze_slot_decay: true`): The slot
  update is `slots = norm(decay * slots + gate * read_out)` where `decay` is a
  scalar. This makes the slot update contractive — old information fades unless
  actively reinforced. The decay is frozen at init to prevent it from growing
  above 1.0, which would make the slot chain divergent.

The previous design used sigmoid gates with `gate_init: -2` (sigmoid(-2) ≈
0.12) for the workspace and `gate_init: 0` (sigmoid(0) = 0.5) for the backbone.
The workspace gates were changed to ReZero in the v2 fix (2026-08-02), but the
backbone gates were not — this was the root cause of all subsequent Cell C
divergences, identified and fixed on 2026-08-06.

---

## Diagnostics: measuring selectivity correctly

Two metrics replace the discarded entropy objective. Both exist because entropy and
free-generation accuracy each conflated things we needed to tell apart.

### 1. Attention input-dependence (formerly `diagnose_workspace.py`, removed 2026-08-09)

`diagnose_workspace.py` was a one-off debugging script for the gradient-instability
investigation (2026-08-02) and has been removed now that those issues are fixed (see
`AGENTS.md`). Its methodology is documented here in case it's needed again — it probed
the model with N=16 distinct inputs of **identical token length** (so attention
tensors align), then for each `(head, position)` held the index fixed and measured how
far each input's attention distribution sits from the across-input mean, in **total
variation distance** (`0.5 * L1`, range [0,1]).

| Reading | Meaning |
|---|---|
| `TV = 0` | Attention identical for every input → **fixed routing table**, zero information about input. Entropy is meaningless here. |
| `TV < 0.02` | Degenerate. |
| `TV < 0.10` | Weak — mostly fixed routing with small content-driven perturbation. |
| `TV > 0.10` | Input-dependent (content-addressed). Whether it stores the *right* content is a separate question. |

Also reports **top-1 routing fixed %**: the fraction of `(head, position)` pairs whose
argmax target is the same for *every* probe input. 100% = completely fixed.

Crucially this **cannot be gamed by sharpening** — a fixed one-hot routing scores TV=0
and 100% fixed no matter how peaked it is. Reported **per path** (read and write): a
functioning read path otherwise masks a degenerate write path, and those failures mean
different things. A fixed *write* path means information enters the slots independently
of content, so the slots are positional buckets regardless of how good reads look.

### 2. Teacher-forced answer accuracy (`eval_ttnn.py`)

Free generation is autoregressive, so one wrong digit corrupts every later step. A low
generation score therefore conflates "cannot reason" with "cannot recover from its own
mistakes" (exposure bias). The eval now reports both, feeding the gold prefix at every
answer position for the teacher-forced pass:

```
  task                             gen  tf exact  tf/token
  Task1 (chain arithmetic)       5.0%      2.5%     22.4%
  Task2 (2-var arithmetic)      27.5%     47.5%     47.6%
```

- **tf >> gen** → exposure bias; the model knows each token given a correct history.
- **tf also low** → genuine reasoning failure. Task 1 above is 22.4% per-token against a
  **10% chance baseline** over ten digits, so even with a perfect prefix the model cannot
  compute the answer. That is not a decoding artifact.

### Known metric caveat: answer-loss is diluted by EOS

The label mask (`data.py`) spans the answer digits **and** the terminating EOS. For a
two-digit answer that means ~1/3 of supervised positions are trivially predictable, which
deflates the reported "answer loss." This is why loss 0.55 could coexist with 12.5%
accuracy. Do not compare loss across configurations without keeping this in mind.

---

## How to run

### 1. Environment setup

**Mac / NVIDIA (PyTorch path):**
```bash
# From the repo root
python3 -m venv mamba-poc
source mamba-poc/bin/activate
pip install -r mamba-poc/requirements.txt
```

On Mac (MPS): `torch` ships with MPS support — no special install needed. The `mamba-ssm` CUDA kernels won't install; the pure-PyTorch Mamba2 layer in `model.py` works on MPS directly.

On NVIDIA (CUDA): optionally `pip install mamba-ssm causal-conv1d` for faster kernels, though the pure-PyTorch path works everywhere.

**Tenstorrent Quietbox 2 (ttnn path):**
```bash
# The TT venv is at the repo root
source /home/ttuser/Documents/jasper/.tt-venv/bin/activate
# Uses ttnn with bfloat16-native computation on Blackhole chips
# Kernel cache is stored per-cell in run_X/tt_cache/
```

### 2. Unit tests

```bash
python data.py
```
Runs roundtrip tests on all three task generators and the vocabulary. Should print "All tests passed!"

### 3. Parameter count check

```bash
python model.py
```
Prints parameter counts for all cells. Useful to verify they're roughly matched (~25–30M each).

### 4. Train a cell

**Tenstorrent (Quietbox 2):**
```bash
# The TT venv is at the repo root
source /home/ttuser/Documents/jasper/.tt-venv/bin/activate
# Uses ttnn with bfloat16-native computation on Blackhole chips

# Launch all three cells in parallel (one per device, safe to log out):
mkdir -p logs checkpoints
TT_VISIBLE_DEVICES=0 nohup python train_ttnn.py \
    --config configs/cell_a_tt.yaml --device 0 --checkpoint_dir checkpoints \
    > logs/cell_a.log 2>&1 &
TT_VISIBLE_DEVICES=1 nohup python train_ttnn.py \
    --config configs/cell_b_tt.yaml --device 0 --checkpoint_dir checkpoints \
    > logs/cell_b.log 2>&1 &
TT_VISIBLE_DEVICES=2 nohup python train_ttnn.py \
    --config configs/cell_c_tt.yaml --device 0 --checkpoint_dir checkpoints \
    > logs/cell_c.log 2>&1 &

# Start the background monitor (reports every 10 minutes):
nohup ./monitor_training.sh > logs/monitor.log 2>&1 &
```

Key behaviors:
- **Fresh data every batch** — no dataset file, no overfitting risk
- **Checkpoint every 100 steps** to `checkpoints/cell_X_stepN.pt`
- **Early stopping** with EMA-smoothed loss plateau detection (patience=1000, min_delta=1e-3)
- **Linear warmup** then constant LR (200 warmup steps for all cells)
- **Bfloat16-native** on Tenstorrent (no gradient checkpointing needed — ample device DRAM)
- **Custom fused kernels** for RoPE, scale+decay, and gate backward (see AGENTS.md)

### 5. Automated evaluation

The eval loop monitors all cells for new checkpoints and automatically evaluates them:

```bash
# Start the eval loop on device 0 (monitors cells A, B, C)
python -u eval_loop.py --device 0 --cells A B C --poll-interval 300 --n-per-task 5
```

This produces JSON results in `run_X/eval_results/step_N.json` and prints a summary table with plateau detection. A cell is flagged as plateaued when Task 1 accuracy shows no improvement (≥1%) over 3 consecutive evals.

Manual single-checkpoint evaluation:
```bash
python eval_ttnn.py --config configs/cell_b_tt.yaml --device 0 \
    --checkpoint run_B/checkpoints/cell_B_step100.pt --n-per-task 5 \
    --depths 2 4 6 8 --json-output run_B/eval_results/step_100.json
```

### 6. Run analysis (R2, R3, R4) — removed, needs reimplementation

`probe.py` (the R2/R3/R4 analysis tool) was removed on 2026-08-09 along with the old
PyTorch (`train.py`) pipeline it depended on. It has no `ttnn`/`model_ttnn.py` equivalent
yet. The analyses it used to run, documented here for reference if reimplementing against
`model_ttnn.py`:

- **R2** (K sweep): accuracy vs. inference K (1, 2, 4, 6, 8, 12, 16) at each depth. Success:
  accuracy on deep problems increases monotonically with K.
- **R3** (linear probes): trains linear probes (97-class classifier, mod 97) on workspace
  slot states to decode the queried variable's value. Success: workspace probes
  substantially exceed residual-stream probes from Cell A, and decodability rises across
  loop iterations.
- **R4** (selective ablation): computes the mean workspace slot state across training
  examples, then replaces live slots with this mean at inference. Success (J-space
  signature): Tasks 1–2 drop ≥30 points, Task 3 drops ≤5 points.

### 7. Google Colab (free GPU)

Open `colab_notebook.ipynb` in Colab with T4 GPU enabled. It runs cells A and C (the go/no-go pair) sequentially at the full ~2B-token budget. Each takes ~12 hours on a T4. See [infra-setup-guide.md](../infra-setup-guide.md) for details.

---

## Config reference

All fields in the YAML configs can override `ModelConfig` defaults in `model.py`.

### Model fields

| Field | Default | Description |
|-------|---------|-------------|
| `cell` | `C` | Which cell config to use (A/B/C) — sets the defaults, then overrides below apply |
| `d_model` | 384 | Hidden dimension |
| `n_layers` | 14 (A) or 13 (B/C) | Total layer count |
| `vocab_size` | 128 | Character-level vocabulary (padded) |
| `d_state` | 64 | Mamba2 state dimension |
| `d_conv` | 4 | Mamba2 conv1d width |
| `expand` | 4 | Mamba2 expansion factor |
| `n_heads` | 4 | Attention heads (shared by Mamba2 SSD and attention) |
| `use_attention` | false | Whether to include attention layers |
| `attention_positions` | [5, 10] | Which layer indices are attention (rest are Mamba2) |
| `use_workspace` | false | Whether to include the perceiver workspace module |
| `n_workspace_slots` | 16 | Number of workspace slot vectors |
| `recurrent_core` | false | Whether layers `core_start` to `core_end` are looped K times |
| `core_start` | 6 | First layer of the recurrent core |
| `core_end` | 10 | Last layer of the recurrent core (exclusive) |
| `k_train_max` | 6 | Max K during training (sampled uniformly from {1…k_train_max}). Restored to 6 after slot-state gradient scaling fix. |
| `k_inference` | 3 | K at inference (sweep this for R2) |
| `dropout` | 0.0 | Dropout rate |
| `spectral_norm_bound` | 5.0 | Spectral norm cap for workspace weights (power iteration, 10 steps) |
| `backbone_spectral_norm_bound` | 2.0 | Spectral norm cap for backbone qkv/out_proj weights |
| `chain_scale_safety` | 1.0 | Extra safety factor for recurrent core gradient chain (1/(K×safety)). With ReZero backbone gates, 1.0 gives 83% margin. |
| `freeze_gamma` | true | Freeze retention gamma (don't train it) — eliminates O(T²) gradient instability |
| `freeze_slot_decay` | true | Freeze slot_decay at init — prevents slot chain growth above 1.0 |
| `gate_init` | 0.0 | ReZero gate init value (0.0 = true identity). No longer uses sigmoid. |
| `slot_decay_init` | 1.0 | Initial value for slot decay scalar (1.0 = no decay) |
| `attention_residual_core` | false | Use attention residual core (Kimi K3-style) instead of fixed blend for recurrent core |

### Training fields

| Field | Default | Description |
|-------|---------|-------------|
| `lr` | 2e-4 | Peak learning rate (AdamW). All cells use 2e-4. |
| `lr_groups` | (none) | Per-parameter LR groups: prefix-to-multiplier mapping. E.g. `{ws_: 0.25}` gives workspace params 0.25x the base LR. |
| `max_steps` | 10000 | Total training steps |
| `tokens_per_batch` | 250000 | Target tokens per batch (overrides `batch_size` if set) |
| `seq_len` | 128 | Sequence length |
| `warmup_steps` | 200 | LR warmup steps |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient clipping max norm (backbone parameters) |
| `ws_grad_clip` | 0.5 | Gradient clipping max norm for workspace parameters (ws_*). When set, clipping is component-wise: workspace and backbone are clipped independently. |
| `depth_range` | [2, 8] | Training depth range (inclusive) |
| `eval_interval` | 500 | Steps between evaluations |
| `log_interval` | 50 | Steps between log prints |
| `checkpoint_interval` | 100 | Steps between checkpoints |
| `plateau_patience` | 1000 | Early stopping: steps without improvement before stopping (10% of max_steps) |
| `plateau_min_delta` | 1e-3 | Early stopping: minimum relative improvement to reset patience (0.1%) |
| `seed` | 42 | Random seed for data generation |
| `wandb` | false | Enable wandb logging |
| `wandb_project` | `mamba-workspace-poc` | Wandb project name |
| `run_name` | `cellX-seed1` | Wandb run name |
| `ckpt_dir` | `checkpoints` | Checkpoint directory |
| `resume` | true | Auto-resume from latest checkpoint |

---

## Expected training timeline

### Tenstorrent Quietbox 2 (current)

Three Blackhole chips train cells in parallel (devices 0, 1, 2). All cells run from the repo root with `nohup` — safe to log out.

| Cell | Device | Time/step | Est. total (10000 steps) | Architecture |
|------|--------|-----------|--------------------------|--------------|
| A | 0 | ~4.1s | ~11.4 hours | Hybrid (no workspace, no recurrence) |
| B | 1 | ~4.2s | ~11.7 hours | Hybrid + workspace (QK-Norm + ReZero) |
| C | 2 | ~11.3s | ~31.3 hours | Full (workspace + recurrent core, K_max=6) |

Cell C is ~2.7x slower due to the recurrent core looping K_max=6 times per step.

| Phase | Duration | What |
|-------|----------|------|
| Training | ~12 hours (A/B) to ~31 hours (C) | All three cells train to convergence or early stopping |
| Analysis | ~2 hours | R2 K-sweep, R3 probes, R4 ablation on best checkpoint |

### Mac / NVIDIA / Colab (deprecated)

The project has moved to Tenstorrent hardware. The old PyTorch training path (`train.py`) was
removed on 2026-08-09; `model.py` remains only as a CPU-side correctness reference for
`test_ws_backward_cpu.py`, not a runnable training path. See
[infra-setup-guide.md](../infra-setup-guide.md) for legacy platform notes.

---

## Decision rule

| Outcome | Action | Cost |
|---------|--------|------|
| **Go** | Fund the 300M language-scale ablation (~$3–5K) from the main plan | R1 + R2 pass, and ≥1 of R3/R4 shows workspace doing causal work |
| **Pivot** | Keep the hybrid backbone, drop the workspace — browser plan proceeds on standard hybrid distillation | R1 fails but A's efficiency story stands |
| **Kill** | Novel-architecture track is dead — sampled-attempts-plus-verifier on a standard model remains the plan | C ≤ A everywhere |

---

## Current experimental status (as of August 2026)

Training is running on a Tenstorrent Quietbox 2 (4× Blackhole chips, bfloat16-native). The model is **Jasper** — a workspace-augmented retention network with recurrent core.

### Synthetic training (workspace-poc/)

Cell A (backbone-only control) and Cell B (workspace, no recurrence) completed training. Cell B early-stopped at step 9404. Cell C (full architecture: workspace + recurrent core with attention residuals) is currently training with the backbone ReZero gate fix.

**Cell C training history**: Cell C went through multiple rounds of gradient instability fixes (see "History of gradient instability" above). The root cause was identified on 2026-08-06: backbone layer gates used sigmoid (sigmoid(0)=0.5) instead of ReZero, giving only 4% stability margin. The fix (backbone sigmoid→ReZero) gives 83% margin. Training was restarted from scratch with the fix.

**Current run** (`cell_c_attn_residual.yaml`, `checkpoints/ar_rezero/`):

| Step | Loss | Grad norm | gz_gr | gz_gw | Notes |
|------|------|-----------|-------|-------|-------|
| 0 | 7.48 | 8.86 | 0.000 | 0.000 | All gates at 0 (ReZero init) |
| 50 | 7.08 | 8.97 | -0.013 | 0.013 | Gates growing slowly |
| 100 | 1.91 | 2.81 | -0.049 | 0.047 | Loss dropping fast as gates open |
| 200 | 1.46 | 1.23 | -0.056 | 0.051 | Stable, grad norm flat |

All previous runs showed grad norm oscillation by step 200. This run is stable — the ReZero fix appears to have resolved the root cause.

### Text POC (workspace-mvp/)

A text training pipeline has been set up in `workspace-mvp/` to test Jasper on real language data. It uses TinyStories (~480M tokens, GPT-2 BPE tokenization) with the same Cell C AR architecture. The model is 29.8M params (10.5M architecture + 19.3M embedding for the 50K vocab).

Smoke-tested (200 steps): loss 10.92 → 9.24, perplexity ~50K → 10,595, grad norm stable at 1.2-2.9. Ready for full training launch after the synthetic Cell C run validates the architecture.

See AGENTS.md for full details on the text POC setup.

### Architecture changes since the original plan

1. **Mamba-2 → Retention (Mamba-3)**: The selective scan SSM was replaced with decayed linear attention (RetNet-style). This is simpler, faster on tt-nn, and avoids the causal conv1d issues with ttnn. The decay matrix D[t,s] = gamma^(t-s) is computed on-device. See `mamba3_layer.py`.

2. **Custom fused kernels**: Three custom tt-metal kernels eliminate intermediate DRAM writes and reduce op dispatch overhead:
   - **Fused RoPE**: Full rotation (rot1 = x1*cos - x2*sin, rot2 = x1*sin + x2*cos) in a single kernel pass. Uses split-RoPE optimization to avoid expensive sub-tile concat/slice when d_half=48 is not tile-aligned.
   - **Fused scale + decay**: scores = qk * scale * D_decay in one pass (FPU mul + SFPU mul_unary).
   - **Fused gate backward**: grad_out_flat and grad_g computed in a single kernel pass (3 FPU muls + 3 SFPU binary ops).

3. **Early stopping**: EMA-smoothed loss plateau detection with patience=1000 (10% of max_steps) and min_delta=1e-3 (0.1% relative improvement). Tuned for the effective batch size of 384 (micro_batch=8 × accum_steps=48).

4. **Architecture v2 stabilization**: QK-Norm + ReZero gates (workspace + backbone) + component-wise gradient clipping. Previous fixes (backbone spectral norm, gamma freezing, slot decay freezing) remain in place. K restored to 6 after slot-state gradient scaling fix.

5. **Attention residual core** (2026-08-04): Replaced the fixed blend (`x = (1-a)*x + a*x_new`) with learned softmax attention over all K iteration outputs (Kimi K3-style). This bounds output magnitude and provides consistent gradient magnitude across iterations.

6. **Chain gradient scaling** (2026-08-04): The chained gradient from each recurrent core iteration is scaled by 1/(K×safety) before adding the AR gradient. With ReZero backbone gates, safety=1.0 gives 83% stability margin.

7. **Backbone ReZero gate fix** (2026-08-06): The root cause of all Cell C divergences. Backbone layer gates were changed from sigmoid (sigmoid(0)=0.5, 4% margin) to ReZero (gate=0, 83% margin). Also added `freeze_slot_decay: true` to prevent slot chain growth.

### What we're watching for

- **Cell C stability past step 2650**: All previous runs diverged between step 750–2650. The current run (with backbone ReZero fix) must pass step 2650 to be considered stable. Early signs are positive (grad norm flat at step 200).
- **Cell A vs Cell B**: The clean workspace comparison (same LR=2e-4, A has no workspace). If B beats A on Task 1, the workspace is doing real work.
- **Cell C convergence**: The full architecture (workspace + recurrence) is ~2x slower per step. Will it converge within the training budget?
- **Text POC convergence**: Does Jasper learn coherent language on TinyStories? If so, the workspace + recurrent core works on real text, not just synthetic tasks.
- **Cell C K-sweep (R2)**: Does increasing K at inference improve accuracy on hard problems?
- **Selective ablation (R4)**: Does replacing workspace slots with their mean collapse Tasks 1–2 while leaving Task 3 intact?
