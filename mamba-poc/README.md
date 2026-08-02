# Mamba + Workspace POC — Code Guide

This is the code for a desktop-scale proof-of-concept that tests whether a hybrid Mamba model with an engineered workspace and recurrent core reasons better than parameter-matched controls. The experiment takes ~1 week and ~$20–30, and produces a go/no-go decision for a $3–5K language-scale ablation.

See the [root README](../README.md) for the project overview and [desktop-mamba-workspace-poc.md](../desktop-mamba-workspace-poc.md) for the full experiment design.

---

## File map

| File | What it does |
|------|-------------|
| `data.py` | Three synthetic task generators with verifiers, character-level vocabulary, batch generation, and unit tests. Each task has a depth knob `k` controlling reasoning steps. |
| `model.py` | The full model (`MambaWorkspaceModel`) with three cell configurations behind config flags. Contains the pure-PyTorch Mamba2 SSD layer, multi-head attention with RoPE, perceiver-style workspace module, and the recurrent core loop. |
| `model_ttnn.py` | Tenstorrent-native model implementation using `ttnn` ops (bfloat16). Same architecture as `model.py` but compiled for Blackhole chips. Includes custom fused kernels for RoPE, scale+decay, and gate backward. |
| `mamba3_layer.py` | Fixed Mamba-3 layer (replaces Mamba-2's selective scan with decayed linear attention). Used unconditionally for all non-attention layers. |
| `retention_reference.py` | PyTorch reference implementation of the retention layer (decayed linear attention + RoPE). Used for parity testing. |
| `train.py` | Training loop (PyTorch) with fresh data every batch, 15-min checkpointing with auto-resume, wandb logging, cosine LR schedule, and CLI args for cell selection. (Deprecated — use `train_ttnn.py`.) |
| `train_ttnn.py` | Tenstorrent-native training loop. Runs on Blackhole chips via `ttnn`, bfloat16-native computation. Used for the current Quietbox 2 experiments. |
| `eval_ttnn.py` | Checkpoint evaluation script for Tenstorrent. Loads a `.pt` checkpoint, runs the model on generated eval examples, reports accuracy per task and per depth. Supports JSON output for automated tracking. |
| `eval_loop.py` | Automated eval monitoring loop. Polls for new checkpoints every 5 minutes, runs `eval_ttnn.py` on each, saves JSON results, and detects plateaus (3 consecutive evals with <1% Task 1 improvement). |
| `probe.py` | Analysis scripts for R2 (K sweep — test-time compute scaling), R3 (linear probes on workspace slots — decodability), and R4 (selective workspace ablation — J-space signature). |
| `profile_retention.py` | Per-op profiler for the retention layer. Measures wall-clock time of each operation in forward and backward passes. |
| `monitor_training.sh` | Background monitor script. Reports training status every 10 minutes to `logs/monitor.log`. |
| `kernels/` | Custom tt-metal compute kernels: `rope4d_{reader,compute,writer}.cpp` (fused RoPE), `scale_decay_{reader,compute,writer}.cpp` (fused scale+decay), `gate_bwd_{reader,compute,writer}.cpp` (fused gate backward). |
| `test_retention_parity.py` | Parity test: tt-nn forward + backward vs. PyTorch reference autograd. Run with `--tile-aligned` for tile-aligned shapes. |
| `test_mamba3_parity.py` | Parity test for the Mamba-3 layer: float64 gradcheck + tt-nn forward/backward vs. reference. |
| `test_fused_rope_single.py` | Standalone test for the fused RoPE kernel (forward + backward). |
| `test_scale_decay.py` | Standalone test for the fused scale+decay kernel. |
| `test_gate_bwd.py` | Standalone test for the fused gate backward kernel. |
| `test_ws_backward_cpu.py` | CPU gradient check for the workspace backward pass. Verifies all gradients (input, weights, slots, gates) against PyTorch autograd to within float32 precision. |
| `test_ws_backward.py` | Device-side gradient check (earlier attempt, superseded by the CPU version). |
| `configs/cell_{a,b,c}_tt.yaml` | YAML configs for each model cell, including Tenstorrent-specific settings. Cell A is the no-workspace control; B adds workspace; C adds recurrent core. |
| `run_all_cells_parallel.sh` | Launches all three cells in parallel on separate Blackhole chips (one cell per device). |
| `tt_runner.py` | Sequential training runner for a single Blackhole chip. Supports `--cell`, `--status`, `--clean`, `--smoke` modes. |
| `colab_notebook.ipynb` | Google Colab T4 notebook for running cells A and C (the go/no-go pair) on free GPU time. (Deprecated — project moved to TT.) |
| `colab_runner.py` | Sequential training runner for Colab (single T4 GPU). (Deprecated — project moved to TT.) |
| `vast_runner.py` | Parallel training runner for Vast.ai (dual GPU). (Deprecated — project moved to TT.) |
| `requirements.txt` | Python dependencies. |
| `checkpoints/` | Saved during training (gitignored). `cell_{X}_step{N}.pt` per cell per checkpoint interval. |
| `logs/` | Training logs (`cell_{a,b,c}.log`) and monitor log (`monitor.log`). |

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
| **A** | Hybrid (12 Mamba2 + 2 attention at positions 5, 10), no workspace | `use_attention=true, use_workspace=false, recurrent_core=false` | The no-workspace control — isolates workspace effect |
| **B** | Hybrid + workspace (11 Mamba2 + 2 attention + 16-slot perceiver workspace) | `use_attention=true, use_workspace=true, recurrent_core=false` | Does an engineered workspace help without recurrence? |
| **C** | Full architecture (hybrid + workspace + layers 6–9 looped K times) | `use_attention=true, use_workspace=true, recurrent_core=true` | The go/no-go cell |

Cell C's recurrent core (layers 6–9) is applied K times per forward pass. During training, K is sampled uniformly from {1…6} per batch. At inference, K can be swept (the `k_inference` config field or `--k_override` in code).

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

The `WorkspaceModule` is a perceiver-style cross-attention block with 16 learned slot vectors. Each application does two steps:

1. **Read**: slots attend over hidden states (slots query, hidden states are keys/values)
2. **Write**: hidden states attend over slots (hidden states query, slots are keys/values)

Slot state persists across recurrent iterations (passed as `slot_state`), so each loop iteration reads and revises the same workspace — the model can iteratively refine its reasoning in the slots rather than in the token stream.

### Training stabilization

The workspace and recurrent core introduce four sources of gradient instability that are not present in the baseline hybrid model. Each is addressed with a targeted normalization:

1. **Per-parameter LR groups** (`lr_groups` config field): Workspace parameters receive 0.25x the base learning rate (5e-5 vs 2e-4 for the backbone). The workspace creates a sharper loss landscape where equal-LR training causes workspace gradients to dominate and destabilize. The backbone LR is kept at 2e-4 to match Cell A (the control). Config: `lr_groups: {ws_: 0.25}`

2. **1/sqrt(K) residual scaling** (in `model_ttnn.py` / `model.py`): The recurrent core's blend factor is scaled by 1/sqrt(K), turning the full replacement (x = x_new) into a partial update (x = (1/sqrt(K)) * x_new + (1 - 1/sqrt(K)) * x). This normalizes gradient accumulation across K iterations so the total gradient norm is independent of K. Uses the actual K sampled per batch. Based on the variance-preserving principle from DeepNorm.

3. **Slot parameter normalization** (in `model_ttnn.py` / `model.py`): The learned slot parameters are normalized to unit RMS after each optimizer step, breaking the positive feedback loop: large slots → sharp attention → large gradients → larger slots. This is a parameter constraint (like weight clipping in GANs) that allows slot direction to change freely while constraining magnitude. Applied automatically in the training loop via `model.normalize_workspace_slots()`.

4. **Spectral normalization of workspace weights** (in `model_ttnn.py` / `model.py`): All eight workspace weight matrices (read_q, read_k, read_v, read_out, write_q, write_k, write_v, write_out) have their spectral norm capped at C=5 after each optimizer step. If the spectral norm exceeds C, the matrix is scaled down to C; if below C, it's left unchanged (cap, not target). This addresses the root cause of weight growth directly — with spectral norm ≤ C, unit-RMS slots, and LayerNorm on inputs, attention logits are bounded by C²/sqrt(d_h) ≈ 2.55, giving a max attention ratio of ~164:1 — enough for selective attention but preventing the e^100+ ratios that cause gradient explosion. Uses 10 steps of power iteration on-device (same algorithm as Spectral Normalization GANs, Miyato et al. 2018). Config: `spectral_norm_bound: 5.0`. Applied automatically in the training loop via `model.normalize_workspace_slots()`.

**Why C=5?** The standard spectral normalization bound (used in GANs) is C=1. We tried C=1 first — it eliminated the early gradient spikes and allowed both cells to pass their previous collapse points (Cell B at step 300, Cell C at step 250). However, the collapse re-emerged at peak LR (step 400 for B with grad norm 19.7, step 300 for C with grad norm 61.1). The root cause was a **capacity mismatch**: with C=1, the max attention ratio is only e^(2/sqrt(d_h)) ≈ 1.2:1 (nearly uniform) — the workspace cannot provide selective attention. The backbone adapts fast at peak LR and develops expectations that the workspace will selectively attend to specific positions, but the workspace is trapped at spectral norm 1 and cannot deliver. The resulting tug-of-war at the constraint boundary produces the gradient spike. C=5 allows attention ratios up to ~164:1 — sufficient for selective attention while still bounded. The cap (rather than hard normalization to C) lets weights grow naturally up to the bound, preventing disruption when resuming from checkpoints saved with smaller spectral norms.

All four are deterministic (one hyperparameter: the spectral norm bound C=5) and can be disabled independently for ablation.

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

### Architectural stability: zero-init gates and slot decay

The regularizers proved the workspace CAN learn selective attention (write entropy dropped from 2.75 to 0.224), but the model collapsed at step 450 — the sharpening exceeded what external constraints could stably support. Analysis revealed the root cause: the workspace has a **feedback loop with gain** (read from x → update slots → write back to x), and this loop is amplifying rather than contractive. External constraints (spectral norm, slot norm, regularizers) cap individual components but don't make the loop itself stable.

Two architectural changes make the workspace stable by construction — a "tennis ball in a bucket" rather than a knife edge:

7. **Zero-initialized gates** (`gate_init: -2`): Gate parameters are initialized to -2, giving sigmoid(-2) ≈ 0.12. The workspace starts with a small but non-trivial contribution — the feedback loop is mostly inactive. The gates gradually open as the model learns that the workspace is useful. A gate-specific LR group (10x workspace LR) ensures the gates reach sigmoid(0) = 0.5 within ~1,000 steps, fitting the 3,000-step training budget. This is the same principle as zero-initialization in LoRA, tuned for a shorter training horizon.

8. **Slot decay** (`slot_decay_init: 1.0`): The slot update changes from `slots = norm(slots + gate * read_out)` to `slots = norm(decay * slots + gate * read_out)` where `decay` is a learned scalar initialized at 1.0. The decay makes the slot update contractive: old information naturally fades unless the read gate actively reinforces it. The feedback loop now has a restoring force — if attention gets too sharp and overwrites a slot, the decay pulls it back toward the learned slot embedding.

Together, these make the workspace start stable (near-zero gates) and stay stable (decaying slots). The external constraints become less critical because the architecture itself is contractive rather than amplifying. Config: `gate_init` (default -2), `slot_decay_init` (default 1.0) in the YAML configs.

---

## Diagnostics: measuring selectivity correctly

Two metrics replace the discarded entropy objective. Both exist because entropy and
free-generation accuracy each conflated things we needed to tell apart.

### 1. Attention input-dependence (`diagnose_workspace.py`)

Probes the model with N=16 distinct inputs of **identical token length** (so attention
tensors align), then for each `(head, position)` holds the index fixed and measures how
far each input's attention distribution sits from the across-input mean, in **total
variation distance** (`0.5 * L1`, range [0,1]).

```bash
python diagnose_workspace.py --config configs/cell_b_tt.yaml --device 0 \
    --checkpoint run_B/checkpoints/cell_B_step500.pt --n-probe 16 \
    --json-output diag.json
```

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

**PyTorch (Mac / NVIDIA):**
```bash
python train.py --config configs/cell_c_tt.yaml
```

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

### 6. Run analysis (R2, R3, R4)

After training Cell C:

```bash
python probe.py --checkpoint checkpoints/cellC_latest.pt --config configs/cell_c_tt.yaml --all
```

Or run individual analyses:

```bash
# R2: K sweep — accuracy vs inference K (1, 2, 4, 6, 8, 12, 16) at each depth
python probe.py --checkpoint checkpoints/cellC_latest.pt --config configs/cell_c_tt.yaml --r2

# R3: Linear probes — can you decode intermediate variable values from workspace slots?
python probe.py --checkpoint checkpoints/cellC_latest.pt --config configs/cell_c_tt.yaml --r3

# R4: Selective ablation — replace workspace slots with mean, measure task-specific collapse
python probe.py --checkpoint checkpoints/cellC_latest.pt --config configs/cell_c_tt.yaml --r4
```

**R2** sweeps K from 1 to 16 and reports accuracy per task and per depth. Success: accuracy on deep problems increases monotonically with K.

**R3** trains linear probes (97-class classifier, mod 97) on workspace slot states to decode the queried variable's value. Reports probe accuracy at each K. Success: workspace probes substantially exceed residual-stream probes from Cell A, and decodability rises across loop iterations.

**R4** computes the mean workspace slot state across training examples, then replaces live slots with this mean at inference. Reports accuracy drop per task. Success (J-space signature): Tasks 1–2 drop ≥30 points, Task 3 drops ≤5 points.

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
| `k_train_max` | 6 | Max K during training (sampled uniformly from {1…k_train_max}) |
| `k_inference` | 6 | K at inference (sweep this for R2) |
| `dropout` | 0.0 | Dropout rate |

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
| `grad_clip` | 1.0 | Gradient clipping max norm |
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
| B | 1 | ~4.2s | ~11.7 hours | Hybrid + workspace |
| C | 2 | ~11.3s | ~31.3 hours | Full (workspace + recurrent core, K_max=6) |

Cell C is ~2.7x slower due to the recurrent core looping K_max=6 times per step.

| Phase | Duration | What |
|-------|----------|------|
| Training | ~12 hours (A/B) to ~31 hours (C) | All three cells train to convergence or early stopping |
| Analysis | ~2 hours | R2 K-sweep, R3 probes, R4 ablation on best checkpoint |

### Mac / NVIDIA / Colab (deprecated)

The project has moved to Tenstorrent hardware. The PyTorch path (`train.py`, `model.py`) still works on Mac/NVIDIA but is no longer the primary training path. See [infra-setup-guide.md](../infra-setup-guide.md) for legacy platform setup.

---

## Decision rule

| Outcome | Action | Cost |
|---------|--------|------|
| **Go** | Fund the 300M language-scale ablation (~$3–5K) from the main plan | R1 + R2 pass, and ≥1 of R3/R4 shows workspace doing causal work |
| **Pivot** | Keep the hybrid backbone, drop the workspace — browser plan proceeds on standard hybrid distillation | R1 fails but A's efficiency story stands |
| **Kill** | Novel-architecture track is dead — sampled-attempts-plus-verifier on a standard model remains the plan | C ≤ A everywhere |

---

## Current experimental status (as of August 2026)

Training is running on a Tenstorrent Quietbox 2 (4× Blackhole chips, bfloat16-native). All three cells launched in parallel on August 2, 2026, with custom fused kernels (RoPE, scale+decay, gate backward) and the Mamba-3 retention layer.

> **Note:** The results below use the current cell naming (A/B/C). The former pure-Mamba2 (old A) and Mamba2+attention at lr=6e-4 (old B) cells were deprecated and removed after they plateaued — old A at ~35% overall (0% Task 1), old B at 60% overall (0% Task 1). The remaining cells were renamed: old E→A, old C→B, old D→C. The Mamba-2 selective scan was replaced with a decayed linear attention (retention) layer — see `mamba3_layer.py` and `retention_reference.py`.

### Training progress

| Cell | Device | Step | Loss | LR | Status |
|------|--------|------|------|-----|--------|
| A | 0 | ~100 | 1.57 | 1e-4 (warming up) | Training (no-workspace control) |
| B | 1 | ~100 | 1.56 | 1e-4 (warming up) | Training (workspace) |
| C | 2 | ~50 | 2.27 | 5e-5 (warming up) | Training (full architecture) |

### Architecture changes since the original plan

1. **Mamba-2 → Mamba-3 (retention)**: The selective scan SSM was replaced with decayed linear attention (RetNet-style). This is simpler, faster on tt-nn, and avoids the causal conv1d issues with ttnn. The decay matrix D[t,s] = gamma^(t-s) is computed on-device. See `mamba3_layer.py`.

2. **Custom fused kernels**: Three custom tt-metal kernels eliminate intermediate DRAM writes and reduce op dispatch overhead:
   - **Fused RoPE**: Full rotation (rot1 = x1*cos - x2*sin, rot2 = x1*sin + x2*cos) in a single kernel pass. Uses split-RoPE optimization to avoid expensive sub-tile concat/slice when d_half=48 is not tile-aligned.
   - **Fused scale + decay**: scores = qk * scale * D_decay in one pass (FPU mul + SFPU mul_unary).
   - **Fused gate backward**: grad_out_flat and grad_g computed in a single kernel pass (3 FPU muls + 3 SFPU binary ops).

3. **Early stopping**: EMA-smoothed loss plateau detection with patience=1000 (10% of max_steps) and min_delta=1e-3 (0.1% relative improvement). Tuned for the effective batch size of 384 (micro_batch=8 × accum_steps=48).

### What we're watching for

- **Cell A vs Cell B**: The clean workspace comparison (same LR=2e-4, A has no workspace). If B beats A on Task 1, the workspace is doing real work.
- **Cell C convergence**: The full architecture (workspace + recurrence) is ~2.7x slower per step. Will it converge within the training budget?
- **Cell C K-sweep (R2)**: Does increasing K at inference improve accuracy on hard problems?
- **Selective ablation (R4)**: Does replacing workspace slots with their mean collapse Tasks 1–2 while leaving Task 3 intact?
