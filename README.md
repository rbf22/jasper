# Jasper — Workspace-Recurrent Reasoning for In-Browser Inference

## What this project is

Jasper is a research project building toward a ~1B-parameter reasoning model that can run in a web browser. The core architectural bet: **an explicitly engineered workspace (analogous to Anthropic's J-space finding) combined with a recurrent core (TRM / recurrent-depth transformers) lets a small model trade inference-time compute depth for parameter count, while keeping memory flat for browser deployment.**

This repo contains two things:

1. **A desktop-scale proof-of-concept** (`mamba-poc/`) — four parameter-matched model variants trained on synthetic reasoning tasks in under a week for ~$20–30. This experiment produces a go/no-go decision before any real cloud spend.
2. **Three planning documents** (root `.md` files) — the full architecture specification, the POC plan, and the infrastructure setup guide for scaling up if the POC passes.

---

## The thesis in one paragraph

Reasoning-critical machinery in language models appears to be small (Anthropic's J-space: a compact, privileged internal workspace carrying most multi-step reasoning). Iterative refinement can substitute for parameter count (TRM's recursive revision loop, Geiping et al.'s recurrent-depth transformers). Meanwhile, the binding constraint for in-browser deployment is not parameter count but KV-cache memory growth over long reasoning traces. So the target is a model with an explicitly engineered workspace, a recurrent core that trades loop iterations for depth, and traces short enough that the KV cache stays within mobile-browser memory budgets. A hybrid Mamba-attention backbone makes memory flat in both sequence length and loop count — the SSM's compressed state plays the workspace role natively.

---

## Repository structure

```
jasper/
├── README.md                          ← you are here
├── workspace-recurrent-1b-plan.md     ← the full 1B architecture spec & cost assessment
├── desktop-mamba-workspace-poc.md     ← the desktop POC experiment plan
├── infra-setup-guide.md               ← infrastructure ladder (Mac → Kaggle → RunPod → Lambda)
└── mamba-poc/                         ← the code
    ├── README.md                      ← practical guide to the code (start here for running things)
    ├── data.py                        ← 3 synthetic task generators + verifiers + unit tests
    ├── model.py                       ← 4 model cells behind config flags
    ├── train.py                       ← training loop with checkpointing, wandb, auto-resume
    ├── probe.py                       ← R2/R3/R4 analysis (K sweep, linear probes, ablation)
    ├── configs/
    │   ├── cell_a.yaml                ← pure Mamba2 (SSM floor)
    │   ├── cell_b.yaml                ← hybrid baseline (Mamba2 + attention)
    │   ├── cell_c.yaml                ← hybrid + workspace
    │   └── cell_d.yaml                ← hybrid + workspace + recurrent core (full architecture)
    ├── kaggle_notebook.ipynb          ← Kaggle T4 runner for cells B and D
    ├── requirements.txt
    └── checkpoints/                   ← saved during training (gitignored)
```

---

## The four model cells

The POC trains four parameter-matched variants (~30M params each) on synthetic tasks with controllable reasoning depth. The cells form an ablation ladder — each adds one component, so the marginal contribution of each is isolated.

| Cell | Architecture | Layers | What it tests |
|------|-------------|--------|---------------|
| **A** | Pure Mamba2 | 14 | SSM floor — establishes the baseline, especially on state-tracking (Task 2) where linear SSMs are known to struggle |
| **B** | Hybrid (Mamba2 + attention) | 12 Mamba2 + 2 attention at positions 5, 10 | The real baseline the workspace must beat — hybridization recovers retrieval capability |
| **C** | Hybrid + workspace | 11 Mamba2 + 2 attention + perceiver workspace (16 slots) | Does an engineered workspace help without recurrence? |
| **D** | Hybrid + workspace + recurrent core | 11 Mamba2 + 2 attention + workspace + layers 6–9 looped K times | The full architecture — does recurrence + workspace beat the hybrid baseline? |

Cell D is the go/no-go cell. If it doesn't beat Cell B, the architecture bet is dead at zero cloud cost.

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
# Set up environment (Mac / MPS)
python3 -m venv mamba-poc
source mamba-poc/bin/activate
pip install -r mamba-poc/requirements.txt

# Run unit tests on the data generators
python mamba-poc/data.py

# Check parameter counts for all four cells
python mamba-poc/model.py

# Smoke test (5M params, ~10 min, Task 1 depth-4 only)
python mamba-poc/train.py --config mamba-poc/configs/cell_d.yaml --smoke-test

# Train a cell for real
python mamba-poc/train.py --config mamba-poc/configs/cell_d.yaml

# Run analysis (R2 K-sweep, R3 probes, R4 ablation) on a trained checkpoint
python mamba-poc/probe.py --checkpoint mamba-poc/checkpoints/cellD_latest.pt --config mamba-poc/configs/cell_d.yaml --all
```

See `mamba-poc/README.md` for the full code guide.

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
| How the code works and how to run it | [`mamba-poc/README.md`](mamba-poc/README.md) |

---

## Key references

- Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (2025)
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (TRM, 2025)
- Anthropic, "A global workspace in language models" (2026)
- Wang et al., "M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models" (2025)
- NVIDIA Nemotron-H / Nano 2 / Nemotron 3 reports (2025–26)
- DeepScaleR (2025)
- Google DeepMind, "Relaxed Recursive Transformers" (2024)
