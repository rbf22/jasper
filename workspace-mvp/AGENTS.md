# WRAP MVP — Text POC

## Overview

Text training pipeline for **WRAP** — a workspace-augmented retention
network with recurrent core. WRAP combines RetNet-style linear attention,
Perceiver-style external memory (workspace), depth-recurrent iteration, and
attention residuals into a unified architecture.

This directory trains the same WRAP model as `workspace-poc/` but on text
data with GPT-2 BPE tokenization. The current active dataset is the
**tiny challenges** corpus — a 2M-example mixed reasoning dataset
(narrative logic puzzles, BrainBashers-style puzzles, and bAbI QA) —
replacing the earlier TinyStories-only approach. The goal is to test
whether the workspace and recurrent core can learn multi-hop reasoning
when the task is framed as text continuation.

## Environment

Same as `workspace-poc/`:

```
/home/rfenwick/Documents/jasper/.tt-venv/bin/python
```

The venv is symlinked as `venv/`. Model code (`model_ttnn.py`,
`kernels/`) is symlinked from `workspace-poc/` — there is only one copy
of the model code.

## Files

| File | What it does |
|------|-------------|
| `text_data.py` | GPT-2 BPE tokenizer wrapper, dataset loading (TinyStories or tiny challenges), pre-tokenized cache, batch sampling, eval batch generation |
| `train_text.py` | Training loop — imports shared infrastructure from `workspace-poc/train_ttnn.py`, adds host-side loss for large vocab, gate freeze logic |
| `eval_text.py` | Perplexity evaluation + autoregressive text generation |
| `configs/text_cell_c.yaml` | Cell C AR config on TinyStories (legacy) |
| `configs/text_cell_c_tiny_challenges.yaml` | Cell C AR config on tiny challenges (active) |
| `test_text_data.py` | Pytest tests for `text_data.py` (tokenizer, dataset loading, packed-stream labels). CPU-only. |
| `data/tiny_challenges_train.txt` | ~292 MB, 78.7M tokens (pre-tokenized cache at `.tokens.pt`) |
| `data/tiny_challenges_valid.txt` | ~3 MB, 787K tokens |
| `data/tinystories_train.txt` | 1.9GB, ~480M tokens (legacy, unused in current run) |
| `data/tinystories_valid.txt` | 19MB, ~4.8M tokens (legacy) |

## Architecture

Same WRAP architecture as `workspace-poc/` Cell C AR:
- 13 layers, d_model=384, n_heads=4
- Retention layers (decayed linear attention) for non-attention positions
- Attention at layers 5 and 10
- Workspace with 16 slots, QK-Norm, ReZero gates
- **Causal masking on workspace cross-attention** (2026-08-08 fix) —
  the workspace's read/write passes now use Perceiver-IO style causal
  masks to prevent future-token leakage. See `workspace-poc/AGENTS.md`
  for details. This is critical for both synthetic tasks (answer leakage)
  and text (standard LM causality).
- Recurrent core (K=6) with attention residuals
- **ReZero gates on all layers** (workspace + backbone) — backbone gates
  were changed from sigmoid to ReZero on 2026-08-06, fixing the root cause
  of all Cell C training divergences (see `workspace-poc/AGENTS.md`)
- `freeze_slot_decay: true` — slot decay frozen at 1.0 to prevent slot
  chain growth
- `chain_scale_safety: 1.0` — 1/K chain gradient scaling (83% margin with
  ReZero gates, no extra safety factor needed)
- `gate_clamp_bound: 0.3` — ReZero gates clamped to [-0.3, 0.3] after each
  optimizer step (safety net against RMSNorm cancellation; see
  `workspace-poc/AGENTS.md` § "Gate clamping fix")
- `cosine_decay_steps: 4000` — cosine LR decay after warmup (prevents
  constant-LR bifurcation; see `workspace-poc/AGENTS.md` § "Gradient
  stability fix")
- `wd_groups: {"suffix:_gate": 0.0}` — ReZero gates excluded from weight
  decay (scalar params meant to grow from 0)
- `beta2_groups: {"suffix:_gate": 0.999}` — backbone gates use higher
  beta2 for stable normalization of high-variance gate gradients
- `spike_action: restore` — reload last checkpoint on grad spike instead
  of skipping (skip-on-spike freezes model in bad state permanently)
- **`gate_freeze_steps: 600`** — workspace gates held at exactly 0 for
  the first 600 steps, making the workspace a true no-op while the
  backbone learns. After step 600, gates learn freely at 1x LR. This
  prevents the workspace from destabilizing the backbone before it has
  learned useful representations. See `workspace-poc/AGENTS.md` § "Gate
  freeze fix" for details.
- Gate/AR LR reduced from 10x to 1x base LR (controlled gate opening
  after unfreeze)
- 10.5M architecture params + 19.3M embedding = **29.8M total**

The model code is symlinked from `workspace-poc/`, so all architecture
fixes apply automatically. The config
(`configs/text_cell_c_tiny_challenges.yaml`) needs to be kept in sync
with new config fields.

## Key differences from synthetic training

1. **Vocab**: 50,257 (GPT-2 BPE) vs 128 (char-level). The embedding
   dominates parameter count (19.3M of 29.8M).
2. **Loss**: Host-side `cross_entropy_loss_host()` for V > 2048. The
   on-device loss uses a V×V identity matrix (5GB at V=50257 —
   impractical). Host-side computes in float32, transfers gradient back.
3. **Labels**: Every position (standard LM) vs answer-only. The
   `ignore_index=-100` masking in `cross_entropy_loss` handles both.
4. **Seq len**: 512 vs 128. Text needs longer context.
5. **LR**: 1e-4 vs 2e-4. Lower for the larger model.
6. **Eval**: Perplexity + generation vs task accuracy with verifiers.

## Dataset choice

The current active dataset is the **tiny challenges** corpus — a 2M-example
mixed reasoning dataset inspired by Enigmata-style synthetic reasoning
training. The mixture:

| Component | Fraction | Description |
|-----------|----------|-------------|
| Narrative logic puzzles | 40% | Multi-hop location, arithmetic, swap, attribute puzzles |
| BrainBashers-style puzzles | 55% | Attribute-chain, elimination, ordering puzzles |
| bAbI QA | 5% | HuggingFace `Muennighoff/babi` passages |

- Train: 2,000,000 examples (~292 MB, 78.7M tokens)
- Validation: 20,000 examples (~3 MB, 787K tokens)

### Rationale for the shift from TinyStories

TinyStories was the original Stage 1 dataset, chosen because it was the
only dataset designed for <10M param models. However, evaluation of the
synthetic POC tasks showed that **none of the cells (A, B, or C) learned
Task 1** (chained assignment arithmetic) — even A at step 9600 got 0%.
The multi-hop reasoning task was beyond what a 10M retention model could
learn from synthetic character-level data.

The tiny challenges corpus tests whether framing multi-hop reasoning as
**text continuation** — with the workspace and recurrent core — can
learn reasoning that the synthetic POC could not. The puzzles are
designed to require multi-hop reasoning but are expressed in natural
language, which may be easier for the model to learn from than the
compressed character-level format.

### Previous two-stage plan (superseded)

The original plan was:
1. Stage 1: TinyStories — validate architecture learns language
2. Stage 2: Fine-tune on OpenThoughts-114k for multi-hop reasoning

The tiny challenges corpus merges these stages: it tests reasoning
directly in the text format, without a separate language pretraining
stage. If the architecture can learn from this corpus, it validates both
language learning and reasoning in a single run.

## Running

```bash
# Smoke test (3 steps)
TT_VISIBLE_DEVICES=1 python train_text.py \
    --config configs/text_cell_c_tiny_challenges.yaml --steps 3 \
    --micro_batch 4 --accum_steps 1 \
    --checkpoint_dir /tmp/text_test --device 0

# Full training (tiny challenges, 2000 steps)
cd /home/rfenwick/Documents/jasper/workspace-mvp
TT_VISIBLE_DEVICES=3 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_text.py \
    --config configs/text_cell_c_tiny_challenges.yaml --device 0 \
    --checkpoint_dir checkpoints/tiny_challenges \
    > logs/text_tiny_challenges_20260813.log 2>&1 &

# Evaluation (perplexity + generation)
TT_VISIBLE_DEVICES=1 python eval_text.py \
    --checkpoint checkpoints/tiny_challenges/cell_text_step500.pt \
    --config configs/text_cell_c_tiny_challenges.yaml --device 0 \
    --generate --prompt "Once upon a time"
```

### Current training configuration

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

### Gate freeze in text training

`train_text.py` has its own training loop (it does not call the
`train_ttnn.py` loop), so gate freeze logic was added separately:
- Reads `gate_freeze_steps` from the config
- Calls `model.freeze_workspace_gates()` after each optimizer step while
  `step < gate_freeze_steps`
- Clamps gates to `[-gate_clamp_bound, gate_clamp_bound]` after unfreeze
- Syncs optimizer fp32 master copies after gate modifications

## Loss computation

The on-device `cross_entropy_loss` in `train_ttnn.py` uses a V×V identity
matrix for one-hot encoding — fine for V=128 (32KB), impossible for V=50257
(~5GB in bfloat16). Two alternatives are implemented in `train_text.py`:

1. **Host-side loss** (`cross_entropy_loss_host`, default): Computes loss and
   gradient in float32 on the host, transfers gradient back to device. ~200MB
   logits transfer per micro-batch. Faster for the current 30M model size
   (CPU softmax over 50K elements is faster than device softmax due to
   dispatch overhead on the large vocab dimension).

2. **On-device scatter loss** (`cross_entropy_loss_scatter`): Uses
   `ttnn.gather` to extract target probs and `ttnn.scatter_add` to build the
   gradient — no V×V identity matrix, no host transfer of the full logits
   tensor. Only a tiny (B×T×32 ≈ 65K element) transfer for the loss value.
   Slower for the current model size (~2.4s/step vs ~1.4s/step) because the
   on-device softmax over 50K elements has high dispatch overhead. Also
   produces occasional `inf` grad norms in bfloat16 (8/200 steps in testing).
   Will be the preferred method for larger models where the 200MB host
   transfer becomes the bottleneck.

Select via CLI: `--loss_method host` (default) or `--loss_method scatter`.

The `compute_loss()` function dispatches based on vocab size and method:
- V ≤ 2048: on-device identity-matrix loss (from `train_ttnn.py`)
- V > 2048 + `host`: host-side float32 loss
- V > 2048 + `scatter`: on-device scatter-based loss

The scatter loss is verified against the host-side loss in
`test_scatter_loss.py` — loss values match within 0.3% relative error,
gradients match within 1.6% mean relative error (bfloat16 noise floor).

## Data preparation tools

A set of standalone Python scripts in `tools/` generate the `.txt` corpora that
`text_data.py` expects. Output always goes to `workspace-mvp/data/` by default.

| Script | What it generates |
|---|---|
| `tools/prepare_tinystories.py` | Original TinyStories train/valid text files |
| `tools/prepare_logic_puzzles.py` | Narrative multi-hop logic puzzles (location, arithmetic, swap, attribute) |
| `tools/prepare_brainbashers_style.py` | Attribute-chain / elimination / position puzzles |
| `tools/prepare_babi.py` | bAbI QA passages from HuggingFace `Muennighoff/babi` |
| `tools/prepare_tiny_challenges.py` | 2M mixed challenge corpus (logic + BrainBashers + bAbI) |

Run from `workspace-mvp/`:

```bash
python tools/prepare_tiny_challenges.py --n-train 2000000 --n-valid 20000
```

All scripts run in the same venv as the training code (`datasets`, `tiktoken`,
and `torch` are not needed for data prep beyond the existing `tiktoken` import).

## Reasoning corpus configs

New `configs/` YAML files point `train_text.py` at the generated challenge corpora
instead of TinyStories:

- `configs/text_cell_c_logic.yaml` — `logic_puzzles_*.txt`
- `configs/text_cell_c_brainbashers.yaml` — `brainbashers_*.txt`
- `configs/text_cell_c_babi.yaml` — `babi_*.txt`
- `configs/text_cell_c_tiny_challenges.yaml` — `tiny_challenges_*.txt`

These are intended to test whether the workspace and recurrent core can solve
small, multi-hop reasoning problems when the task is framed as answer
continuation.
