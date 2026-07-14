# Mamba + Workspace POC — Code Guide

This is the code for a desktop-scale proof-of-concept that tests whether a hybrid Mamba model with an engineered workspace and recurrent core reasons better than parameter-matched controls. The experiment takes ~1 week and ~$20–30, and produces a go/no-go decision for a $3–5K language-scale ablation.

See the [root README](../README.md) for the project overview and [desktop-mamba-workspace-poc.md](../desktop-mamba-workspace-poc.md) for the full experiment design.

---

## File map

| File | What it does |
|------|-------------|
| `data.py` | Three synthetic task generators with verifiers, character-level vocabulary, batch generation, and unit tests. Each task has a depth knob `k` controlling reasoning steps. |
| `model.py` | The full model (`MambaWorkspaceModel`) with four cell configurations behind config flags. Contains the pure-PyTorch Mamba2 SSD layer, multi-head attention with RoPE, perceiver-style workspace module, and the recurrent core loop. |
| `train.py` | Training loop with fresh data every batch, 15-min checkpointing with auto-resume, wandb logging, cosine LR schedule, and CLI args for cell selection. |
| `probe.py` | Analysis scripts for R2 (K sweep — test-time compute scaling), R3 (linear probes on workspace slots — decodability), and R4 (selective workspace ablation — J-space signature). |
| `configs/cell_{a,b,c,d}.yaml` | YAML configs for each of the four model cells. Override any `ModelConfig` field from here. |
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

## The four cells

All cells are ~30M parameters, `d_model=384`, vocabulary ~128 (character-level). Each cell adds one component so marginal contributions are isolated.

| Cell | Architecture | Config flags | Key params |
|------|-------------|--------------|------------|
| **A** | Pure Mamba2 (14 layers) | `use_attention=false, use_workspace=false, recurrent_core=false` | Establishes the SSM floor |
| **B** | Hybrid (12 Mamba2 + 2 attention at positions 5, 10) | `use_attention=true, use_workspace=false, recurrent_core=false` | The real baseline the workspace must beat |
| **C** | Hybrid + workspace (11 Mamba2 + 2 attention + 16-slot perceiver workspace) | `use_attention=true, use_workspace=true, recurrent_core=false` | Does an engineered workspace help without recurrence? |
| **D** | Full architecture (hybrid + workspace + layers 6–9 looped K times) | `use_attention=true, use_workspace=true, recurrent_core=true` | The go/no-go cell |

Cell D's recurrent core (layers 6–9) is applied K times per forward pass. During training, K is sampled uniformly from {1…6} per batch. At inference, K can be swept (the `k_inference` config field or `--k_override` in code).

Cell C removes one Mamba layer (13 vs 14) to compensate for the ~2M workspace parameters, keeping cells parameter-matched.

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

---

## How to run

### 1. Environment setup

```bash
# From the repo root
python3 -m venv mamba-poc
source mamba-poc/bin/activate
pip install -r mamba-poc/requirements.txt
```

On Mac (MPS): `torch` ships with MPS support — no special install needed. The `mamba-ssm` CUDA kernels won't install; the pure-PyTorch Mamba2 layer in `model.py` works on MPS directly.

On NVIDIA (CUDA): optionally `pip install mamba-ssm causal-conv1d` for faster kernels, though the pure-PyTorch path works everywhere.

### 2. Unit tests

```bash
python data.py
```
Runs roundtrip tests on all three task generators and the vocabulary. Should print "All tests passed!"

### 3. Parameter count check

```bash
python model.py
```
Prints parameter counts for all four cells. Useful to verify they're roughly matched (~30M each).

### 4. Train a cell

```bash
python train.py --config configs/cell_d.yaml
```

Key behaviors:
- **Fresh data every batch** — no dataset file, no overfitting risk
- **Checkpoint every 15 min** to `checkpoints/cell{X}_latest.pt`
- **Auto-resume** from latest checkpoint if interrupted (set `resume: true` in config)
- **Wandb logging** — set `wandb: true` in config and `wandb login` first
- **Evaluation every 500 steps** on a fixed eval set (depths 2–16, 20 samples per task per depth)
- **Cosine LR schedule** with 200-step warmup

Override the cell without editing the config:
```bash
python train.py --config configs/cell_a.yaml --cell D
```

Disable wandb:
```bash
python train.py --config configs/cell_d.yaml --no-wandb
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
| `cell` | `D` | Which cell config to use (A/B/C/D) — sets the defaults, then overrides below apply |
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
| `lr` | 6e-4 | Peak learning rate (AdamW) |
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

| When | What | Checkpoint |
|------|------|------------|
| Day 1 | Cells A and B | Confirm A lags B on Task 2 — validates harness against literature |
| Days 3–5 | Cells C and D | R1 read on in-distribution accuracy |
| Day 6 | Full eval sweeps | Depth extrapolation curves, K sweep for D, seed-2 launch on winner |
| Day 7 | Probing and ablation | R3, R4, plot headline figures, write one-page verdict |

On Mac (MPS, Track M): run only cells B and D, ~1–2 days per cell. On RunPod 4090 (Track N): all four cells, ~4–6 hours each. See [infra-setup-guide.md](../infra-setup-guide.md) for platform setup.

---

## Decision rule

| Outcome | Action | Cost |
|---------|--------|------|
| **Go** | Fund the 300M language-scale ablation (~$3–5K) from the main plan | R1 + R2 pass, and ≥1 of R3/R4 shows workspace doing causal work |
| **Pivot** | Keep the hybrid backbone, drop the workspace — browser plan proceeds on standard hybrid distillation | R1 fails but B's efficiency story stands |
| **Kill** | Novel-architecture track is dead — sampled-attempts-plus-verifier on a standard model remains the plan | D ≤ B everywhere |
