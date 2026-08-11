# Jasper — Workspace-Recurrent Reasoning for In-Browser Inference

## What this project is

Jasper is a research project building toward a ~1B-parameter reasoning model that can run in a web browser. The core architectural bet: **an explicitly engineered workspace (analogous to Anthropic's J-space finding) combined with a recurrent core (TRM / recurrent-depth transformers) lets a small model trade inference-time compute depth for parameter count, while keeping memory flat for browser deployment.**

This repo contains two things:

1. **A desktop-scale proof-of-concept** (`workspace-poc/`) — four parameter-matched model variants trained on synthetic reasoning tasks on Tenstorrent hardware. This experiment produces a go/no-go decision before any real cloud spend.
2. **Three planning documents** (root `.md` files) — the full architecture specification, the POC plan, and the infrastructure setup guide for scaling up if the POC passes.

---

## The thesis in one paragraph

Reasoning-critical machinery in language models appears to be small (Anthropic's J-space: a compact, privileged internal workspace carrying most multi-step reasoning). Iterative refinement can substitute for parameter count (TRM's recursive revision loop, Geiping et al.'s recurrent-depth transformers). Meanwhile, the binding constraint for in-browser deployment is not parameter count but KV-cache memory growth over long reasoning traces. So the target is a model with an explicitly engineered workspace, a recurrent core that trades loop iterations for depth, and traces short enough that the KV cache stays within mobile-browser memory budgets. A retention backbone (RetNet-style decayed linear attention) makes memory flat in sequence length, while the recurrent core trades loop iterations for depth without growing the KV cache.

---

## Repository structure

```
jasper/
├── README.md                          ← you are here
├── workspace-recurrent-1b-plan.md     ← the full 1B architecture spec & cost assessment
├── desktop-mamba-workspace-poc.md     ← the desktop POC experiment plan
├── infra-setup-guide.md               ← infrastructure ladder (Mac → Kaggle → RunPod → Lambda)
├── kimi-k3-relevance-notes.md         ← Kimi K3 architecture analysis (informs the AR core design)
├── workspace-poc/                     ← synthetic task training (the code)
│   ├── README.md                      ← practical guide to the code (start here for running things)
│   ├── AGENTS.md                      ← project notes, architecture history, debugging guide
│   ├── data.py                        ← 3 synthetic task generators + verifiers + unit tests
│   ├── model_ttnn.py                  ← Jasper model (ttnn, bfloat16) — the active implementation
│   ├── train_ttnn.py                  ← Tenstorrent-native training loop
│   ├── eval_ttnn.py                   ← checkpoint evaluation (per-task, per-depth accuracy)
│   ├── probe.py                       ← R2/R3/R4 analysis (K sweep, linear probes, ablation)
│   ├── configs/
│   │   ├── cell_a_tt.yaml             ← backbone baseline (no workspace, no recurrence)
│   │   ├── cell_b_tt.yaml             ← backbone + workspace (no recurrence)
│   │   └── cell_c_attn_residual.yaml  ← backbone + workspace + recurrent core + attention residual (primary Cell C config)
│   ├── kernels/                       ← custom tt-metal compute kernels (RoPE, scale+decay, gate bwd)
│   └── checkpoints/                   ← saved during training (gitignored)
├── workspace-mvp/                     ← text POC (TinyStories, shared model code via symlinks)
└── paper.tex                          ← paper draft
```

---

## The model cells

The POC trains three parameter-matched variants (~10.5M params each) on synthetic tasks with controllable reasoning depth. Cells A–C form an ablation ladder — each adds one component, so the marginal contribution of each is isolated.

| Cell | Architecture | Layers | What it tests |
|------|-------------|--------|---------------|
| **A** | Hybrid (retention + 2 attention at positions 5, 10), no workspace | 13 | The no-workspace control — isolates workspace effect |
| **B** | Hybrid + workspace (retention + 2 attention + 16-slot perceiver workspace with QK-Norm + ReZero gates) | 13 | Does an engineered workspace help without recurrence? |
| **C** | Full architecture (hybrid + workspace + layers 6–9 looped K=6 times with attention residuals) | 13 | The go/no-go cell — does recurrence + workspace beat the hybrid baseline? |

Cell C is the go/no-go cell. If it doesn't beat Cell B, the architecture bet is dead at zero cloud cost. Cell C uses an **attention residual core** (Kimi K3-style) that stores all K iteration outputs and computes learned softmax attention over them, replacing the fixed blend that caused gradient amplification in earlier runs.

---

## The three synthetic tasks

All tasks are character-level, generated on-the-fly (no fixed dataset, no overfitting risk), and automatically verifiable. Each has a **depth knob** `k` that controls how many reasoning steps a correct answer requires. Training uses depths 2–8; evaluation extends to 2–16, so extrapolation beyond the training range is measured from day one.

| Task | Description | Example | Why it exists |
|------|------------|---------|--------------|
| **1 — Chained assignment arithmetic** | Multi-hop composition: variables defined in terms of previous variables, all mod 97 | `a=7;b=a*3+2;c=b-a;d=c*2;?d;` → answer is `d`'s value | Core multi-hop reasoning task — the model must follow a chain of dependencies |
| **2 — Permutation tracking** | n items, k swap operations, query one item's final position | `n=6;2,5;0,3;5,1;?3;` → where is item 3 after all swaps? | SSM stress test — state-tracking is a documented weakness of linear SSMs; tests whether the workspace compensates |
| **3 — Single-hop recall** | Shallow lookup with distractors | `a=5;b=12;c=8;d=3;e=15;?c;` → answer is `8` | Control task for the selective-ablation test — the J-space signature requires that killing the workspace leaves this intact while Tasks 1–2 collapse |

Training mix: 45% Task 1 / 45% Task 2 / 10% Task 3.

---

## The four pre-registered results

These are the measurements that constitute proof, in ascending order of importance. R3 and R4 are what make this a *J-space* proof rather than just another architecture ablation.

| Result | What it measures | Success criterion |
|--------|-----------------|-------------------|
| **R1 — Capability** | Does Cell D beat Cell B on deep problems? | D beats B by ≥10 accuracy points on Tasks 1–2 at depths 10–16 (beyond training range) |
| **R2 — Test-time compute scaling** | Does increasing K (loop iterations) at inference improve accuracy on hard problems? | Accuracy on deep problems increases monotonically with K; harder depths need higher K to saturate |
| **R3 — Workspace decodability** | Can you linearly decode intermediate reasoning steps from the workspace slots? | Probe accuracy from workspace slots substantially exceeds probes on a matched residual-stream position in Cell B; decodability of later intermediates rises across loop iterations |
| **R4 — Selective ablation** | Does the workspace causally carry multi-step reasoning? | Replacing workspace slots with their training-set mean: Tasks 1–2 collapse (≥30pt drop) while Task 3 barely moves (≤5pt drop) — mirroring Anthropic's J-space finding |

---

## Quick start

```bash
# Tenstorrent Quietbox 2 (primary development environment)
source /home/rfenwick/Documents/jasper/.tt-venv/bin/activate

# Run unit tests on the data generators
python workspace-poc/data.py

# Smoke test (3 steps, verifies forward+backward+checkpoint)
cd workspace-poc
python train_ttnn.py --config configs/cell_c_attn_residual.yaml --steps 3 \
    --micro_batch 4 --accum_steps 1 --checkpoint_dir /tmp/test --device 0

# Train a cell for real
python train_ttnn.py --config configs/cell_c_attn_residual.yaml \
    --device 0 --checkpoint_dir checkpoints/ar_rezero

# Run analysis (R2 K-sweep, R3 probes, R4 ablation) on a trained checkpoint
python probe.py --checkpoint checkpoints/ar_rezero/cell_C_final.pt \
    --config configs/cell_c_attn_residual.yaml --all
```

See `workspace-poc/README.md` for the full code guide and `workspace-poc/AGENTS.md` for the architecture history and debugging notes.

---

## Decision rule

The POC produces a one-page verdict:

- **Go** (fund the 300M language-scale ablation, ~$3–5K): R1 and R2 pass, and at least one of R3/R4 shows the workspace doing real causal work.
- **Pivot** (keep the hybrid, drop the workspace): R1 fails but Cell B's efficiency story stands — the browser plan proceeds on hybrid distillation without the novel architecture.
- **Kill** (novel-architecture track only): Cell D ≤ Cell B everywhere — the sampled-attempts-plus-verifier plan on a standard model remains fully intact, and the week cost nothing but electricity.

A caution: synthetic-task wins at 30M have a real history of not transferring to language at scale. That's why the gate between this experiment and real money is the 300M ablation, not a victory lap. This week buys the right to spend $5K intelligently, not the conclusion itself.

---

## Where to read next

| If you want to understand... | Read this |
|------------------------------|-----------|
| The full 1B architecture spec, training pipeline, and cost assessment | [`workspace-recurrent-1b-plan.md`](workspace-recurrent-1b-plan.md) |
| The desktop POC experiment design (tasks, cells, measurements, timeline) | [`desktop-mamba-workspace-poc.md`](desktop-mamba-workspace-poc.md) |
| How to set up infrastructure (Mac → Kaggle → RunPod → Lambda) | [`infra-setup-guide.md`](infra-setup-guide.md) |
| How the code works and how to run it | [`workspace-poc/README.md`](workspace-poc/README.md) |
| Architecture history, debugging guide, and gradient instability fixes | [`workspace-poc/AGENTS.md`](workspace-poc/AGENTS.md) |
| Kimi K3 architecture analysis (informs the attention residual core) | [`kimi-k3-relevance-notes.md`](kimi-k3-relevance-notes.md) |

---

## Key references

- Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (2025)
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (TRM, 2025)
- Anthropic, "A global workspace in language models" (2026)
- Bachlechner et al., "ReZero is All You Need: Fast Convergence at Large Depth" (2020)
- Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models" (RetNet, 2023)
- Moonshot AI, "Kimi K3" (2026) — attention residual core design
- Henry et al., "Query-Key Normalization for Transformers" (2020)
- Yang et al., "Component-wise Gradient Clipping" (EMNLP 2022)
- Wang et al., "M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models" (2025)
- Google DeepMind, "Relaxed Recursive Transformers" (2024)
