# Desktop Proof-of-Concept: Mamba + Engineered Workspace for Reasoning

## One afternoon to set up, one week to train, one clear go/no-go answer

*Companion to the Workspace-Recurrent 1B Plan — July 2026*

---

## 1. What this experiment proves (and what it doesn't)

The claim under test: **a small hybrid Mamba model with an explicit workspace and a recurrent core reasons better than a parameter-matched model without them, and the workspace causally carries the intermediate reasoning state.** The second half is the J-space part — we don't just want better accuracy, we want to reproduce, at desktop scale, the signature Anthropic found: ablating the workspace selectively destroys multi-step reasoning while leaving shallow recall intact, and intermediate quantities are linearly decodable from the workspace before the model outputs them.

We prove this on synthetic verifiable tasks with controllable depth, not on natural language. That's the honest scope: tiny models can't learn language-scale reasoning in a week, but synthetic depth-controlled tasks are exactly how TRM and the looped-transformer literature established recursion's value, and they give clean curves (accuracy vs. problem depth, accuracy vs. inference-time loop count) that either show the effect or don't. A positive result here funds the 300M language-scale ablation in the main plan; a negative result kills the architecture bet for $0 before any cloud spend.

## 2. Hardware tracks

**Track N (NVIDIA desktop, 16–24GB, e.g. RTX 4090):** the primary spec. Use `mamba-ssm` CUDA kernels. At ~30M params you'll sustain roughly 100–200K tokens/sec, so each training cell (~2B tokens) takes 3–6 hours — the full grid plus seeds fits comfortably in the week with room for re-runs.

**Track M (Apple Silicon, M2/M3 Max or better):** fully viable at reduced scale. The CUDA `mamba-ssm` package won't install; use a pure-PyTorch Mamba2 implementation (`mamba.py` / `mamba2-minimal`) on MPS, or MLX if you want to write the scan yourself (a Mamba2 layer in MLX is ~150 lines and you know this stack). Expect ~5–15K tokens/sec at 25M params, so budget ~1–2 days per cell and cut the grid from four cells to the two that matter most (B vs. D below). If the week feels tight, $50–150 of a rented 4090/A6000 runs the whole grid overnight — but the point of this plan is that it doesn't require it.

Everything below is written for Track N with Track M deltas noted.

## 3. The tasks (all generated, all verifiable, infinite data)

Three generators, each ~50 lines of Python, each with a **depth knob** k that controls how many reasoning steps a correct answer requires. Train on depths 2–8, evaluate on 2–16, so extrapolation beyond trained depth is measured from day one.

**Task 1 — Chained assignment arithmetic (multi-hop composition).** Programs like `a=7; b=a*3+2; c=b-a; d=c*2; print(d)` with all arithmetic mod 97 (keeps the vocabulary tiny and prevents magnitude shortcuts). Depth = chain length to the queried variable. This is the core multi-hop task.

**Task 2 — Permutation tracking (the SSM stress test).** n labeled items, a sequence of k swap operations, query one item's final position. State-tracking is a documented weakness of linear SSMs (they provably struggle with composition tasks of this shape), so this cell tells you whether the workspace *compensates for* the architecture's known gap — the most interesting possible result for the hybrid thesis.

**Task 3 — Single-hop recall (control).** `a=5; b=12; ... ; print(b)` with distractors. Deliberately shallow. This exists purely for the selective-ablation test: the J-space signature requires that killing the workspace leaves this task intact while Tasks 1–2 collapse.

Mix during training: 45% / 45% / 10%. Sample fresh examples every batch — with generated data there is no train set to overfit, which removes a whole category of confounds.

## 4. The model grid

Four parameter-matched cells at ~30M parameters, d_model 384, vocabulary ~128 (character-level over the task alphabet). Track M: run cells B and D only.

**Cell A — pure Mamba2.** 14 Mamba2 layers. Establishes the SSM floor, especially on Task 2.

**Cell B — hybrid baseline.** 12 Mamba2 layers with attention layers at positions 5 and 10 (roughly the 1-in-6 hybrid ratio the Nemotron line validated). This is the *real* baseline the workspace must beat.

**Cell C — hybrid + workspace.** Cell B plus a workspace of m=16 learned slot vectors, implemented perceiver-style rather than as sequence tokens (sequence tokens don't work cleanly with a causal SSM — later positions couldn't write back). Twice per forward pass (after layers 5 and 10), interleave two cheap cross-attention steps: slots attend over the hidden states to *read*, then hidden states attend over the slots to *write back*. ~2M extra parameters; remove one Mamba layer to stay matched.

**Cell D — hybrid + workspace + recurrent core.** Layers 6–9 become a recurrent core applied K times, with the workspace read/write inside the loop, so each iteration revises the slots — this is the TRM-style revision dynamic. Train with K sampled uniformly from {1…6} per batch and truncated backprop through the last 2 iterations (keeps memory flat). At eval, sweep K from 1 to 16.

All cells: AdamW, lr 6e-4 with cosine decay, batch ~0.25M tokens, ~2B tokens per cell (Track M: ~0.5B), bf16 (fp16/float32 fallback on MPS), one seed per cell during the week, second seed on the winner if time allows.

## 5. The measurements that constitute proof

Four pre-registered results, in ascending order of importance.

**R1 — Capability:** D beats B by ≥10 accuracy points on Tasks 1–2 at depths above the training range (k = 10–16). If D only ties B in-distribution, the architecture isn't paying rent.

**R2 — Test-time compute scaling:** for cell D, accuracy on deep problems increases monotonically with inference K, and harder depths need higher K to saturate. This is the recurrent-depth signature — compute depth substituting for parameters — and it's the property the browser deployment story depends on (dial K per device).

**R3 — Workspace decodability (the J-lens analogue):** freeze cell D, cache workspace slots at each loop iteration on Task 1, and train linear probes to decode the *intermediate* variable values (b and c in the example above, not just the final answer). Success: probe accuracy from workspace slots substantially exceeds probes trained on a matched residual-stream position in cell B, and decodability of later intermediates rises across loop iterations — you can literally watch the model work through the chain in its workspace.

**R4 — Selective ablation (the J-space signature):** at inference, replace workspace slots with their training-set mean. Success mirrors Anthropic's finding: Tasks 1–2 collapse (≥30-point drop) while Task 3 recall barely moves (≤5-point drop). This is the causal claim — the workspace isn't decoration, it's where the multi-step reasoning lives.

R3 and R4 are what make this a *J-space* proof rather than just another architecture ablation, and they're an afternoon of analysis code on top of the trained checkpoints.

## 6. The afternoon (setup, ~3–4 hours)

Hour 1: environment. Fresh venv; Track N installs `torch`, `mamba-ssm`, `causal-conv1d`, `wandb`; Track M installs `torch` (MPS) and vendors a pure-PyTorch Mamba2 block. Skeleton repo in the nanoGPT style — a single `model.py`, `train.py`, `data.py` — is genuinely the right shape here; resist framework weight.

Hour 2: data generators plus their verifiers, with unit tests that round-trip generate→solve→check. This is mechanical code and exactly the kind of thing to hand to Claude Code with this document as the spec — the generators, the eval harness, and the probing scripts are all well-specified enough above to scaffold in one session.

Hour 3: model code for all four cells behind config flags (`use_attention`, `use_workspace`, `recurrent_core`, `k_train_max`). The workspace module is ~60 lines; the recurrent loop is ~20.

Hour 4: smoke test — a 5M-parameter cell D on Task 1 depth-4 only, 10 minutes of training. If loss drops below the trivial baseline and generated answers verify, launch the real grid before dinner.

## 7. The week

Days 1–2, cells A and B (Track M: cell B only). Day 2 checkpoint: confirm the known result that A lags B on Task 2 — this validates your harness against the literature before you trust it on the novel cells. Days 3–5, cells C and D. Day 5 checkpoint: R1 read on in-distribution accuracy. Day 6: full eval sweeps — depth extrapolation curves for all cells, the K sweep for D, seed-2 launch on the winner. Day 7: probing and ablation (R3, R4), plot the four headline figures (accuracy-vs-depth per cell; accuracy-vs-K per depth; probe accuracy per loop iteration; ablation deltas per task), and write the one-page verdict.

## 8. Decision rule

**Go** (fund the 300M language-scale ablation, ~$3–5K, from the main plan): R1 and R2 pass, and at least one of R3/R4 shows the workspace doing real causal work. **Pivot** (keep the hybrid, drop the workspace): R1 fails but B's efficiency story stands — the browser plan proceeds on Nemotron/M1-style hybrid distillation without the novel architecture. **Kill** (novel-architecture track only): D ≤ B everywhere — the sampled-attempts-plus-verifier plan on a standard model remains fully intact, and the week cost nothing but electricity.

One caution for the writeup: synthetic-task wins at 30M have a real history of not transferring to language at scale — that's precisely why the gate between this experiment and real money is the 300M ablation, not a victory lap. This week buys you the right to spend $5K intelligently, not the conclusion itself.
