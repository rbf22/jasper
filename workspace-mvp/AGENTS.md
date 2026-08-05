# Jasper MVP — Text POC

## Overview

Text training pipeline for **Jasper** — a workspace-augmented retention
network with recurrent core. Jasper combines RetNet-style linear attention,
Perceiver-style external memory (workspace), depth-recurrent iteration, and
attention residuals into a unified architecture.

This directory trains the same Jasper model as `workspace-poc/` but on real
text (TinyStories) instead of synthetic arithmetic tasks. The goal is to
validate that the architecture can learn coherent language before scaling
to larger models and reasoning datasets.

## Environment

Same as `workspace-poc/`:

```
/home/rfenwick/Documents/jasper/.tt-venv/bin/python
```

The venv is symlinked as `venv/`. Model code (`model_ttnn.py`,
`mamba3_layer.py`, `kernels/`) is symlinked from `workspace-poc/` — there
is only one copy of the model code.

## Files

| File | What it does |
|------|-------------|
| `text_data.py` | GPT-2 BPE tokenizer wrapper, TinyStories dataset (pre-tokenized cache), batch sampling, eval batch generation |
| `train_text.py` | Training loop — imports shared infrastructure from `workspace-poc/train_ttnn.py`, adds host-side loss for large vocab |
| `eval_text.py` | Perplexity evaluation + autoregressive text generation |
| `configs/text_cell_c.yaml` | Cell C AR config adapted for text (vocab=50257, seq_len=512, lr=1e-4) |
| `data/tinystories_train.txt` | 1.9GB, ~480M tokens (pre-tokenized cache at `.tokens.pt`) |
| `data/tinystories_valid.txt` | 19MB, ~4.8M tokens |

## Architecture

Same Jasper architecture as `workspace-poc/` Cell C AR:
- 13 layers, d_model=384, n_heads=4
- Retention layers (decayed linear attention) for non-attention positions
- Attention at layers 5 and 10
- Workspace with 16 slots, QK-Norm, ReZero gates
- Recurrent core (K=6) with attention residuals
- 10.5M architecture params + 19.3M embedding = **29.8M total**

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

TinyStories was chosen over reasoning datasets (OpenThoughts, NuminaMath,
etc.) because a 10M param model is too small for complex CoT traces — it
would memorize surface patterns without understanding reasoning.
TinyStories is the only dataset designed for <10M param models.

Planned two-stage approach:
1. **Stage 1 (now)**: TinyStories — validate architecture learns language
2. **Stage 2 (future)**: Fine-tune on OpenThoughts-114k (metadata config)
   to test workspace's multi-hop reasoning capability

## Running

```bash
# Smoke test (3 steps)
TT_VISIBLE_DEVICES=1 python train_text.py \
    --config configs/text_cell_c.yaml --steps 3 \
    --micro_batch 4 --accum_steps 1 \
    --checkpoint_dir /tmp/text_test --device 0

# Full training (5000 steps)
cd /home/rfenwick/Documents/jasper/workspace-mvp
TT_VISIBLE_DEVICES=1 nohup /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_text.py \
    --config configs/text_cell_c.yaml --device 0 \
    --checkpoint_dir checkpoints \
    > logs/text_cell_c.log 2>&1 &

# Evaluation (perplexity + generation)
TT_VISIBLE_DEVICES=1 python eval_text.py \
    --checkpoint checkpoints/cell_text_step500.pt \
    --config configs/text_cell_c.yaml --device 0 \
    --generate --prompt "Once upon a time"
```

## Loss computation

The on-device `cross_entropy_loss` in `train_ttnn.py` uses a V×V identity
matrix for one-hot encoding — fine for V=128 (32KB), impossible for V=50257
(~5GB in bfloat16). Two approaches are used:

1. **Host-side loss** (`cross_entropy_loss_host`): Computes loss and gradient
   in float32 on the host, transfers gradient back to device. ~200MB logits
   transfer per micro-batch. Used as the fallback for V > 2048.

2. **On-device scatter-based loss** (`cross_entropy_loss_scatter`): Uses
   `ttnn.embedding` with a 1×V lookup table to gather target logits (avoiding
   the V×V identity matrix), and `ttnn.embedding` with a V×1 one-hot table
   for the gradient scatter. Both tables are cached. This keeps all
   computation on device — no host transfer of the full logits tensor.

The `compute_loss()` function dispatches based on vocab size and config.
