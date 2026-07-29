# Mamba + Workspace POC — Code Guide

This is the code for a desktop-scale proof-of-concept that tests whether a hybrid Mamba model with an engineered workspace and recurrent core reasons better than parameter-matched controls. The experiment takes ~1 week and ~$20–30, and produces a go/no-go decision for a $3–5K language-scale ablation.

See the [root README](../README.md) for the project overview and [desktop-mamba-workspace-poc.md](../desktop-mamba-workspace-poc.md) for the full experiment design.

---

## File map

| File | What it does |
|------|-------------|
| `data.py` | Three synthetic task generators with verifiers, character-level vocabulary, batch generation, and unit tests. Each task has a depth knob `k` controlling reasoning steps. |
| `model.py` | The full model (`MambaWorkspaceModel`) with four cell configurations behind config flags. Contains the pure-PyTorch Mamba2 SSD layer, multi-head attention with RoPE, perceiver-style workspace module, and the recurrent core loop. |
| `model_ttnn.py` | Tenstorrent-native model implementation using `ttnn` ops (bfloat16). Same architecture as `model.py` but compiled for Blackhole chips. Adds Cell E config support. |
| `train.py` | Training loop (PyTorch) with fresh data every batch, 15-min checkpointing with auto-resume, wandb logging, cosine LR schedule, and CLI args for cell selection. |
| `train_ttnn.py` | Tenstorrent-native training loop. Runs on Blackhole chips via `ttnn`, bfloat16-native computation. Used for the current Quietbox 2 experiments. |
| `eval_ttnn.py` | Checkpoint evaluation script for Tenstorrent. Loads a `.pt` checkpoint, runs the model on generated eval examples, reports accuracy per task and per depth. Supports JSON output for automated tracking. |
| `eval_loop.py` | Automated eval monitoring loop. Polls for new checkpoints every 5 minutes, runs `eval_ttnn.py` on each, saves JSON results, and detects plateaus (3 consecutive evals with <1% Task 1 improvement). |
| `probe.py` | Analysis scripts for R2 (K sweep — test-time compute scaling), R3 (linear probes on workspace slots — decodability), and R4 (selective workspace ablation — J-space signature). |
| `test_ws_backward_cpu.py` | CPU gradient check for the workspace backward pass. Verifies all gradients (input, weights, slots, gates) against PyTorch autograd to within float32 precision. |
| `test_ws_backward.py` | Device-side gradient check (earlier attempt, superseded by the CPU version). |
| `configs/cell_{a,b,c,d,e}_tt.yaml` | YAML configs for each model cell, including Tenstorrent-specific settings. Cell E is the learning-rate control (B architecture, C/D learning rate). |
| `run_all_cells_parallel.sh` | Launches all four cells in parallel on separate Blackhole chips (one cell per device). |
| `run_phase2.sh` | Launches Phase 2 training: Cell A (resumed) + Cell E (control) on freed devices after B and C converge, alongside the still-running D. |
| `tt_runner.py` | Sequential training runner for a single Blackhole chip. Supports `--cell`, `--status`, `--clean`, `--smoke` modes. |
| `colab_notebook.ipynb` | Google Colab T4 notebook for running cells B and D (the go/no-go pair) on free GPU time. |
| `colab_runner.py` | Sequential training runner for Colab (single T4 GPU). Trains Cell B then Cell D, saves outputs to Google Drive. |
| `vast_runner.py` | Parallel training runner for Vast.ai (dual GPU). Trains Cell B and D simultaneously on separate GPUs. |
| `requirements.txt` | Python dependencies. |
| `checkpoints/` | Saved during training (gitignored). One checkpoint per cell: `cell{X}_latest.pt`. |

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

## The five cells

All cells are ~25–30M parameters, `d_model=384`, vocabulary ~128 (character-level). Cells A–D form an ablation ladder — each adds one component so marginal contributions are isolated. Cell E is a learning-rate control.

| Cell | Architecture | Config flags | Key params |
|------|-------------|--------------|------------|
| **A** | Pure Mamba2 (14 layers) | `use_attention=false, use_workspace=false, recurrent_core=false` | Establishes the SSM floor |
| **B** | Hybrid (12 Mamba2 + 2 attention at positions 5, 10) | `use_attention=true, use_workspace=false, recurrent_core=false` | The real baseline the workspace must beat |
| **C** | Hybrid + workspace (11 Mamba2 + 2 attention + 16-slot perceiver workspace) | `use_attention=true, use_workspace=true, recurrent_core=false` | Does an engineered workspace help without recurrence? |
| **D** | Full architecture (hybrid + workspace + layers 6–9 looped K times) | `use_attention=true, use_workspace=true, recurrent_core=true` | The go/no-go cell |
| **E** | Hybrid, same as B but with C/D's lower LR (2e-4) | `use_attention=true, use_workspace=false, recurrent_core=false` | Learning-rate control — isolates workspace effect from LR effect |

Cell D's recurrent core (layers 6–9) is applied K times per forward pass. During training, K is sampled uniformly from {1…6} per batch. At inference, K can be swept (the `k_inference` config field or `--k_override` in code).

Cell C removes one Mamba layer (13 vs 14) to compensate for the ~2M workspace parameters, keeping cells parameter-matched.

### Comparison logic

| Comparison | What it isolates |
|------------|-----------------|
| B vs A | Effect of attention hybridization (same LR) |
| C vs B | Effect of workspace (different LR — confounded) |
| **C vs E** | **Effect of workspace (same LR=2e-4 — clean)** |
| D vs C | Effect of recurrent core (same architecture base) |
| D vs B | Full architecture vs hybrid baseline (the R1 test) |

### Model architecture details

The model has three phases in its forward pass:

1. **Pre-core** (layers 0 to `core_start`): standard sequential layer processing. Workspace read/write happens at attention positions.
2. **Recurrent core** (layers `core_start` to `core_end`): the core layers are applied K times. Workspace read/write happens inside the loop on every iteration, so each iteration revises the slot state — this is the TRM-style revision dynamic.
3. **Post-core** (layers `core_end` to `n_layers`): final layers decode the workspace state into token logits.

For cells without a recurrent core (A, B, C), all layers run once in the pre-core phase — there is no loop.

### The workspace module

The `WorkspaceModule` is a perceiver-style cross-attention block with 16 learned slot vectors. Each application does two steps:

1. **Read**: slots attend over hidden states (slots query, hidden states are keys/values)
2. **Write**: hidden states attend over slots (hidden states query, slots are keys/values)

Slot state persists across recurrent iterations (passed as `slot_state`), so each loop iteration reads and revises the same workspace — the model can iteratively refine its reasoning in the slots rather than in the token stream.

### Training stabilization

The workspace and recurrent core introduce four sources of gradient instability that are not present in the baseline hybrid model. Each is addressed with a targeted normalization:

1. **Per-parameter LR groups** (`lr_groups` config field): Workspace parameters receive 0.25x the base learning rate (5e-5 vs 2e-4 for the backbone). The workspace creates a sharper loss landscape where equal-LR training causes workspace gradients to dominate and destabilize. The backbone LR is kept at 2e-4 to match Cell E (the control). Config: `lr_groups: {ws_: 0.25}`

2. **1/sqrt(K) residual scaling** (in `model_ttnn.py` / `model.py`): The recurrent core's blend factor is scaled by 1/sqrt(K), turning the full replacement (x = x_new) into a partial update (x = (1/sqrt(K)) * x_new + (1 - 1/sqrt(K)) * x). This normalizes gradient accumulation across K iterations so the total gradient norm is independent of K. Uses the actual K sampled per batch. Based on the variance-preserving principle from DeepNorm.

3. **Slot parameter normalization** (in `model_ttnn.py` / `model.py`): The learned slot parameters are normalized to unit RMS after each optimizer step, breaking the positive feedback loop: large slots → sharp attention → large gradients → larger slots. This is a parameter constraint (like weight clipping in GANs) that allows slot direction to change freely while constraining magnitude. Applied automatically in the training loop via `model.normalize_workspace_slots()`.

4. **Spectral normalization of workspace weights** (in `model_ttnn.py` / `model.py`): All eight workspace weight matrices (read_q, read_k, read_v, read_out, write_q, write_k, write_v, write_out) are divided by their spectral norm (largest singular value) after each optimizer step, constraining the Lipschitz constant to 1. This addresses the root cause of weight growth directly — with spectral norm 1, unit-RMS slots, and LayerNorm on inputs, attention logits are bounded by 1/sqrt(d_h) ≈ 0.07 by construction. Uses 10 steps of power iteration on-device (same algorithm as Spectral Normalization GANs, Miyato et al. 2018). Applied automatically in the training loop via `model.normalize_workspace_slots()`.

All four are deterministic (no new hyperparameters or learnable parameters) and can be disabled independently for ablation.

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
python train.py --config configs/cell_d.yaml
```

**Tenstorrent (Quietbox 2):**
```bash
# Each cell runs in an isolated run_X/ directory with its own kernel cache
cd run_D  # or create with: mkdir -p run_D/checkpoints run_D/logs run_D/tt_cache
# Symlinks to shared code:
ln -s ../configs configs && ln -s ../data data && ln -s ../model_ttnn.py model_ttnn.py
ln -s ../train_ttnn.py train_ttnn.py

# Launch on a specific device (0-3 for 4 Blackhole chips)
TT_METAL_CACHE=$(pwd)/tt_cache TT_VISIBLE_DEVICES=3 \
python -u train_ttnn.py --config configs/cell_d_tt.yaml --device 3 \
    --checkpoint_dir checkpoints --steps 30000
```

Key behaviors:
- **Fresh data every batch** — no dataset file, no overfitting risk
- **Checkpoint every 100 steps** to `run_X/checkpoints/cell_X_stepN.pt`
- **Auto-resume** from latest checkpoint if interrupted (set `resume: true` in config)
- **Cosine LR schedule** with configurable warmup (200 steps for C/D/E, 1500 for A/B)
- **Bfloat16-native** on Tenstorrent (no gradient checkpointing needed — ample device DRAM)

### 5. Automated evaluation

The eval loop monitors all cells for new checkpoints and automatically evaluates them:

```bash
# Start the eval loop on device 0 (monitors cells A, C, D, E)
python -u eval_loop.py --device 0 --cells A C D E --poll-interval 300 --n-per-task 5
```

This produces JSON results in `run_X/eval_results/step_N.json` and prints a summary table with plateau detection. A cell is flagged as plateaued when Task 1 accuracy shows no improvement (≥1%) over 3 consecutive evals.

Manual single-checkpoint evaluation:
```bash
python eval_ttnn.py --config configs/cell_c_tt.yaml --device 0 \
    --checkpoint run_C/checkpoints/cell_C_step100.pt --n-per-task 5 \
    --depths 2 4 6 8 --json-output run_C/eval_results/step_100.json
```

### 6. Run analysis (R2, R3, R4)

After training Cell D:

```bash
python probe.py --checkpoint checkpoints/cellD_latest.pt --config configs/cell_d.yaml --all
```

Or run individual analyses:

```bash
# R2: K sweep — accuracy vs inference K (1, 2, 4, 6, 8, 12, 16) at each depth
python probe.py --checkpoint checkpoints/cellD_latest.pt --config configs/cell_d.yaml --r2

# R3: Linear probes — can you decode intermediate variable values from workspace slots?
python probe.py --checkpoint checkpoints/cellD_latest.pt --config configs/cell_d.yaml --r3

# R4: Selective ablation — replace workspace slots with mean, measure task-specific collapse
python probe.py --checkpoint checkpoints/cellD_latest.pt --config configs/cell_d.yaml --r4
```

**R2** sweeps K from 1 to 16 and reports accuracy per task and per depth. Success: accuracy on deep problems increases monotonically with K.

**R3** trains linear probes (97-class classifier, mod 97) on workspace slot states to decode the queried variable's value. Reports probe accuracy at each K. Success: workspace probes substantially exceed residual-stream probes from Cell B, and decodability rises across loop iterations.

**R4** computes the mean workspace slot state across training examples, then replaces live slots with this mean at inference. Reports accuracy drop per task. Success (J-space signature): Tasks 1–2 drop ≥30 points, Task 3 drops ≤5 points.

### 7. Google Colab (free GPU)

Open `colab_notebook.ipynb` in Colab with T4 GPU enabled. It runs cells B and D (the go/no-go pair) sequentially at the full ~2B-token budget. Each takes ~12 hours on a T4. See [infra-setup-guide.md](../infra-setup-guide.md) for details.

---

## Config reference

All fields in the YAML configs can override `ModelConfig` defaults in `model.py`.

### Model fields

| Field | Default | Description |
|-------|---------|-------------|
| `cell` | `D` | Which cell config to use (A/B/C/D/E) — sets the defaults, then overrides below apply |
| `d_model` | 384 | Hidden dimension |
| `n_layers` | 14 (A/B) or 13 (C/D) | Total layer count |
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
| `lr` | 6e-4 (A/B), 2e-4 (C/D/E) | Peak learning rate (AdamW). C/D/E use lower LR due to workspace gradient dynamics. |
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
| `checkpoint_interval` | 900 | Seconds between checkpoints |
| `seed` | 42 | Random seed for data generation |
| `wandb` | false | Enable wandb logging |
| `wandb_project` | `mamba-workspace-poc` | Wandb project name |
| `run_name` | `cellX-seed1` | Wandb run name |
| `ckpt_dir` | `checkpoints` | Checkpoint directory |
| `resume` | true | Auto-resume from latest checkpoint |

---

## Expected training timeline

### Tenstorrent Quietbox 2 (current)

Four Blackhole chips train cells in parallel. Each cell runs in an isolated `run_X/` directory with its own kernel cache. The eval loop runs on device 0, monitoring all cells.

| Phase | Cells | Duration | What |
|-------|-------|----------|------|
| Phase 1 | B (dev 1), C (dev 2), D (dev 3) | ~2–3 days | B plateaus first, then is stopped to free device 1 |
| Phase 2 | E (dev 1), C (dev 2), D (dev 3) | ~2–3 days | E is the LR control; C and D continue to convergence |
| Analysis | — | ~2 hours | R2 K-sweep, R3 probes, R4 ablation on best checkpoint |

Typical step times: ~65s for B/C/E (no recurrence), ~160s for D (recurrent core, K up to 6).

### Mac / NVIDIA / Colab (alternative)

On Mac (MPS, Track M): run only cells B and D, ~1–2 days per cell. On RunPod 4090 (Track N): all four cells, ~4–6 hours each. See [infra-setup-guide.md](../infra-setup-guide.md) for platform setup.

---

## Decision rule

| Outcome | Action | Cost |
|---------|--------|------|
| **Go** | Fund the 300M language-scale ablation (~$3–5K) from the main plan | R1 + R2 pass, and ≥1 of R3/R4 shows workspace doing causal work |
| **Pivot** | Keep the hybrid backbone, drop the workspace — browser plan proceeds on standard hybrid distillation | R1 fails but B's efficiency story stands |
| **Kill** | Novel-architecture track is dead — sampled-attempts-plus-verifier on a standard model remains the plan | D ≤ B everywhere |

---

## Current experimental status (as of July 2026)

Training is running on a Tenstorrent Quietbox 2 (4× Blackhole chips, bfloat16-native).

### Training progress

| Cell | Device | Step | Loss | LR | Status |
|------|--------|------|------|-----|--------|
| A | — | 300 | — | — | Paused (will resume in Phase 2) |
| B | — | 450 | 1.03 | 6e-4 | **Stopped — plateaued at 0% Task 1** |
| C | 2 | ~150 | 1.34 | 1.5e-4 (warming up) | Training |
| D | 3 | ~47 | 3.01 | 4.7e-5 (warming up) | Training |
| E | 1 | ~2 | 4.92 | 2e-6 (warming up) | **Just launched** (LR control) |

### Evaluation results (automated via eval_loop.py)

| Cell | Step | Overall | Task 1 (multi-hop) | Task 2 (state-track) | Task 3 (recall) |
|------|------|---------|---------------------|----------------------|-----------------|
| A | 200 | 30% | 0% | 80% | 10% |
| A | 300 | 35% | 0% | 75% | 30% |
| B | 100 | 22% | 0% | 60% | 5% |
| B | 200 | 30% | 5% | 80% | 5% |
| B | 300 | 57% | 0% | 75% | 95% |
| B | 400 | 60% | 0% | 80% | 100% |
| C | 100 | 23% | 0% | 60% | 10% |

### Key observations

1. **Cell B has plateaued on Task 1** — 4 consecutive evals at 0% (steps 100–400). It masters Task 3 (100%) and Task 2 (80%) but cannot do multi-hop chain reasoning. This confirms the architecture limitation the workspace is designed to address. Cell B has been stopped and replaced by Cell E on device 1.

2. **Cell C is learning fast** — loss at step 150 (1.34) is approaching B's plateau loss (~1.03), despite a 3× lower learning rate. The workspace appears to accelerate learning. First eval at step 100 showed 23% overall, 0% Task 1 (still early in warmup).

3. **Cell D is progressing** — loss dropping from 4.91 to 3.01 in 47 steps. The recurrent core is learning but slow (~160s/step). First checkpoint at step 100 is ~2.3h away.

4. **Cell E is the critical control** — same architecture as B, same LR as C (2e-4). If C beats E on Task 1, the workspace is doing real work. If E matches C, B's failure was a learning rate issue, not an architectural one.

5. **Backward pass verified** — all workspace gradients match PyTorch autograd to <1e-4 relative error. The earlier gradient explosion in Cell C was a training dynamics issue (sharper loss landscape), not a backward pass bug. Fixed by lowering LR from 6e-4 to 2e-4.

### What we're watching for

- **Cell C step 200+**: Will the workspace produce non-zero Task 1 accuracy? This is the first real test of the architecture.
- **Cell E vs Cell C**: The clean workspace comparison (same LR, E has no workspace).
- **Cell D step 100+**: First eval of the full architecture (workspace + recurrence).
- **Cell D K-sweep (R2)**: Does increasing K at inference improve accuracy on hard problems?
