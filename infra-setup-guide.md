# Infrastructure Setup Guide: Mamba + Workspace Proof-of-Concept

## Companion to "Desktop Proof-of-Concept: Mamba + Engineered Workspace for Reasoning"

*July 2026*

---

## The ladder

| Stage | Platform | Cost | Use for |
|---|---|---|---|
| 1 | Your Mac | Free | Env setup, generators, smoke test (Section 6 of the main plan) |
| 2 | Kaggle | Free (30 GPU-hrs/wk) | Day-1 sanity runs, cell B vs. D on a T4 |
| 3 | RunPod (4090) | ~$15–20 total | Full four-cell grid, seed reruns, real training week |
| 4 (later) | Lambda | Committed spend | 300M-scale ablation, multi-GPU Path A pilot — not needed this week |

Skip AWS and GCP free tier for this experiment: GCP's free tier is CPU-only and trial credits carry a zero GPU quota by default; AWS's cheapest GPU (A10G, ~$1/hr) costs 3x a marketplace 4090 for no benefit at this scale.

---

## Stage 1 — Mac setup (this afternoon)

```bash
# Environment
python3 -m venv mamba-poc
source mamba-poc/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # MPS build ships in base torch on macOS
pip install wandb einops numpy pytest

# mamba-ssm's CUDA kernels don't build on macOS — use the pure-PyTorch fallback instead
pip install mamba2-minimal   # or vendor ~150 lines yourself; both are fine for 30M-param models
```

Verify MPS is visible:
```python
import torch
print(torch.backends.mps.is_available())  # should print True on M2/M3/M4
```

Repo skeleton (hand this + the main plan's Sections 3–6 to Claude Code as the spec):
```
mamba-poc/
  data.py        # 3 task generators + verifiers + unit tests
  model.py        # 4 cells behind config flags: use_attention, use_workspace, recurrent_core, k_train_max
  train.py         # training loop, checkpoint every 15 min, wandb logging
  probe.py         # R3/R4 analysis: linear probes + workspace ablation
  configs/
    cell_a.yaml
    cell_b.yaml
    cell_c.yaml
    cell_d.yaml
```

Run the smoke test (5M params, Task 1 depth-4 only, ~10 min) before touching any cloud platform. Confirm loss drops below the trivial baseline and generated answers pass the verifier.

**Checkpoint discipline matters everywhere below, not just on Colab.** Save every 15 minutes to a small file, log to wandb (free tier) so you can watch runs from your phone, and make `train.py` resume from the latest checkpoint automatically. This is what makes Kaggle's 12-hour session limit and any marketplace flakiness a non-issue.

---

## Stage 2 — Kaggle (day 1)

1. Create a Kaggle account if you don't have one; verify your phone number (required to unlock GPU access).
2. New Notebook → Settings → Accelerator → **GPU T4 x2** (you can use just one). Internet must be **on**.
3. Upload your repo as a Kaggle Dataset (or `git clone` directly in a notebook cell if the repo is on GitHub — private repos need a token in Kaggle Secrets).
4. First cell:
```bash
!pip install -q mamba-ssm causal-conv1d wandb einops
```
T4 is Turing — no bf16 support, so set `dtype=fp16` with gradient scaling in your training config. The `mamba-ssm` Triton kernels are occasionally flaky on T4; if you hit kernel errors, fall back to the pure-PyTorch Mamba2 path from Stage 1.

5. Run **cell B** and **cell D** only (the pair that actually decides go/no-go) at the full ~2B-token budget — each takes roughly 12 hours on a T4, so budget one session per cell. Kaggle gives 30 GPU-hours/week, so both fit with margin.
6. Log wandb runs with a consistent naming scheme (`cellB-seed1-kaggle`, etc.) so Stage 3 runs land in the same project and are directly comparable.

If sessions disconnect: your `train.py` resume logic handles it — just restart the notebook and rerun.

---

## Stage 3 — RunPod (rest of the week)

1. Create an account at runpod.io, add a card, put ~$25 in credit (per-second billing, so unused credit just sits there).
2. **Pods → Deploy → Community Cloud** (cheaper than Secure Cloud; fine for a throwaway run). Filter by **RTX 4090**, pick a listing around $0.30–0.40/hr.
3. Choose the **PyTorch 2.x + CUDA 12.x** template — comes with drivers and PyTorch preinstalled, saves the first 20 minutes.
4. Attach a **Network Volume** (a few GB is plenty) if you want checkpoints to survive if you delete the pod — cheap insurance, do this.
5. SSH in (RunPod gives you the command directly) or use their web terminal:
```bash
git clone <your repo>
cd mamba-poc
pip install -r requirements.txt   # torch, mamba-ssm, causal-conv1d, wandb, einops
wandb login   # paste your API key once
```
6. Launch training in a way that survives disconnects — `tmux` or `nohup`, not a bare foreground process:
```bash
tmux new -s train
python train.py --config configs/cell_a.yaml
# ctrl-b, d to detach; tmux attach -t train to check back in
```
7. Run all four cells (A, B, C, D) at full budget here — this is your primary data. At ~$0.35/hr and ~4–6 hrs per cell on a 4090, the full grid plus a second seed on the winning cell should land around $15–20 total, comfortably inside a week.
8. **Stop (don't just leave) the pod** between runs if you're not actively training — billing is per-second but a stopped pod costs nothing while a running idle one does. Pause overnight if nothing is queued.
9. When the grid is done, `scp` or sync the checkpoints and wandb logs down before terminating the pod — the network volume is a safety net, not a substitute for your own copy.

---

## Stage 4 — Lambda (not this week)

Only relevant once Stage 3 says "go" and you're funding the 300M-scale ablation (~$3–5K) or the Path A pilot from the main plan. At that point: lambdalabs.com → Cloud → reserve an on-demand or 1-click cluster instance (single 8xH100 node is the natural unit for that scale), their Lambda Stack image has CUDA/PyTorch/NCCL preconfigured so multi-GPU setup is close to zero-friction. Worth opening an account and checking current on-demand H100 availability now, since reserved capacity can have a waitlist — no need to spend anything yet.

---

## Budget summary

| Item | Cost |
|---|---|
| Mac setup | $0 |
| Kaggle (cells B, D, sanity check) | $0 |
| RunPod (full grid, cells A–D, ~15–20 4090-hours) | ~$15–20 |
| Second seed on winning cell, contingency | ~$5–10 |
| **Total for the week** | **~$20–30** |

This whole plan is designed so a negative result costs you under $30 and an afternoon of setup, and a positive result gives you exact wandb-logged curves to justify the next $3–5K ablation from the main plan.
