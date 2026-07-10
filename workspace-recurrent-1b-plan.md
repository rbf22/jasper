# A Workspace-Recurrent 1B Reasoning Model for In-Browser Inference

## Implementation Plan, Open-Data Training Pipeline, and Cost Assessment

*Draft for review — July 2026*

---

## 1. What we are building and why

The design bet, synthesizing the threads we've discussed: reasoning-critical machinery in language models appears to be small (Anthropic's J-space finding — a compact, privileged internal workspace carrying most multi-step reasoning), and iterative refinement can substitute for parameter count (TRM's recursive revision loop, Geiping et al.'s recurrent-depth transformers). Meanwhile, the binding constraint for in-browser deployment is not parameter count but KV-cache memory growth over long reasoning traces.

So the target is a ~1B-parameter model with three properties: an explicitly engineered workspace rather than an emergent one, a recurrent core that trades loop iterations for depth so hard problems get more compute without more parameters, and traces short enough (via latent reasoning plus Long2Short training) that the KV cache stays within mobile-browser memory budgets.

This is a research-grade bet, not an engineering exercise. Section 8 sequences it after the lower-risk baseline (distill-plus-verify on a standard architecture) so a failed bet still leaves a shippable model.

## 2. Architecture specification

### 2.1 Backbone

A decoder-only model with three zones, following the prelude/core/coda pattern of recurrent-depth transformers. The prelude is 4 standard layers that embed the input into a working representation. The recurrent core is a block of 4 layers that is applied K times per forward pass, where K is sampled during training and chosen adaptively at inference. The coda is 4 layers that decode the final workspace state into token logits.

Suggested dimensions: hidden size 2048, 16 attention heads with grouped-query attention (4 KV heads), SwiGLU MLPs at 8/3 expansion, RoPE positions, RMSNorm. Use a compact tokenizer (32–50K vocabulary, e.g. the Llama 3 tokenizer) rather than a 150K vocabulary — at d=2048, a 150K vocab spends ~300M parameters on embeddings that would be better spent in the trunk. Total unique parameters land around 0.9–1.1B: roughly 100M embeddings, ~600M across the 12 unique transformer layers, plus the workspace machinery below.

Effective compute depth at inference is 8 + 4K layers: K=2 gives a 16-layer model for easy tokens, K=16 gives a 72-layer model for hard derivation steps.

### 2.2 The engineered workspace

Append m = 64 learned register tokens (the "workspace") to the sequence. These are not vocabulary tokens; they are persistent slots that every recurrent-core iteration reads from and writes to via attention. Sequence positions attend to workspace slots and vice versa; the workspace is re-initialized per generated token from its previous-token state (a recurrent channel across time as well as across depth). This gives the model a small, persistent, high-bandwidth scratchpad — a deliberate analogue of the emergent J-space, and directly probeable: you can train J-lens-style readouts on the workspace slots to verify it is carrying intermediate reasoning content, which doubles as an interpretability and debugging tool.

The workspace adds ~30–60M parameters (cross-attention projections and slot embeddings) and negligible inference cost, since 64 slots are tiny relative to sequence length.

### 2.3 Adaptive halting

During training, sample K from a log-normal Poisson (mean ~6, occasionally up to 24) and use truncated backpropagation through the last few iterations, which keeps training memory flat regardless of K. At inference, halt when the residual change in the workspace state between consecutive iterations drops below a threshold — a convergence criterion rather than a learned halting head, which is simpler and proved stable in the recurrent-depth literature. Expose K_max as a runtime knob: the browser client can dial compute up or down per device.

### 2.4 KV-cache strategy

Recurrence multiplies attention passes, not cache size, provided the recurrent core shares one KV cache slot per position (overwrite per iteration, keeping only the final iteration's KV per position, which the recurrent-depth work showed degrades quality only mildly). This is central to the browser story: a 4K-token visible trace with K=8 loops costs the KV memory of a 4K trace, not a 32K one, while delivering comparable effective compute to a much longer explicit chain of thought.

## 3. Two build paths

**Path A — Retrofit (recommended first).** Initialize from an open ~1B checkpoint (Llama-3.2-1B or Qwen3-0.6B/1.7B-adjacent). Assign its bottom layers to the prelude, top layers to the coda, and initialize the recurrent core by averaging or selecting its middle layers, adding per-iteration LoRA adapters so each loop can specialize slightly (the "relaxed recursive transformer" recipe from Google's 2024 work on converting pretrained LLMs into layer-shared recursive models). Then run 50–100B tokens of continued pretraining with stochastic K to teach the model to use recurrence and the workspace. This path costs single-digit thousands of dollars and answers the key scientific question — does engineered recurrence-plus-workspace help at 1B? — before committing to a from-scratch run.

**Path B — From scratch.** A full pretraining run on ~1T tokens with the architecture native from step one. This is what you do if Path A shows a clear win, since retrofitted recurrence likely underperforms native recurrence (the base checkpoint's representations were never shaped for iteration).

## 4. Training pipeline on open data

### Phase 0 — Pretraining or continued pretraining

Open corpora are fully sufficient here. A sensible mixture: FineWeb-Edu and/or DCLM-baseline for the educational web backbone (~55–60%), The Stack v2 / StarCoder2 data for code (~20%), FineMath plus OpenWebMath and Proof-Pile-2 for mathematics (~15%), and peS2o or similar for scientific text (~5%). Weight math and code above their natural rates: the target domain is verifiable reasoning, and small models cannot afford diffuse capability. For Path B, 1T tokens is the credible minimum; note for calibration that competitive small models (SmolLM2/3, Qwen's small tier) train on 10T+ tokens, which is what separates a research artifact from a leaderboard model.

### Phase 1 — Reasoning SFT (distillation)

Two data sources. First, existing open trace datasets: OpenThoughts-3 (~1.2M traces), OpenR1-Math-220k, NuminaMath, and OpenCodeReasoning give several billion tokens of long-form reasoning across math and code. Second, self-generated teacher data: VibeThinker-1.5B/3B weights are open, so generate traces on open problem sets directly (roughly 500K traces at ~15K tokens each is ~7.5B tokens of generation — cheap from a small teacher on vLLM).

This is where the context-extraction idea from our earlier discussion slots in: preprocess teacher traces to pull recall-shaped content (formula statements, known identities, standard algorithms) into the prompt as provided context, KARD-style, and train the student on the remaining derivation-shaped steps. For the recurrent architecture specifically, also train on *truncated* traces — supervising the model to reach the answer from partial visible reasoning — which pressures the derivation to migrate into the recurrent loop and workspace rather than the token stream.

### Phase 2 — RL with verifiable rewards

GRPO or DAPO via an open framework (veRL, OpenRLHF, or TRL). Problem sources with automatic verification: NuminaMath and Big-Math for math with checkable answers, DeepScaleR's curated set, and code tasks with unit tests (TACO, KodCode, PrimeIntellect's synthetic sets). Reward is correctness with a Long2Short shaping term: among correct rollouts, shorter visible traces receive higher reward with the group mean unchanged (VibeThinker-3B's recipe), plus a mild bonus for solving at lower K, which trains the halting behavior. The public cost anchor: DeepScaleR reached o1-preview-level AIME performance on a 1.5B model for roughly $4,500 of compute (~3,800 A100-hours), so this phase is genuinely affordable; recurrence makes each rollout token slower, so budget 2–4x that.

### Phase 3 — Compression and calibration

Distill the RL checkpoint into its own 4-bit quantization-aware form (QAT or QAT-distillation), since the browser target is W4 regardless and post-training quantization stacks a second capability hit on an already-small model. Calibrate the halting threshold on held-out difficulty tiers so median K stays low on easy inputs.

### Phase 4 — Browser deployment

Export via MLC/web-llm (WebGPU) with ONNX Runtime Web WASM as fallback. The recurrent loop needs no new kernels — it re-invokes the same block — so deployment engineering is modest for the transformer variant. Ship K_max as a device-tier setting and wire the sampled-attempts-plus-verification harness (execute generated code/arithmetic in the JS sandbox, majority-vote on exact answers) around the model; per our earlier discussion, that harness is likely worth more than any single architecture choice.

## 5. Cost assessment

The arithmetic behind the headline numbers. Training FLOPs ≈ 6 × N_eff × D, where N_eff is compute-active parameters per token. With mean K=6, N_eff ≈ (8 unique-depth layers + 24 looped-layer applications) × ~50M params/layer ≈ 1.6B. An H100 at a realistic 40% MFU sustains ~4×10^14 FLOPs/s, and committed/spot pricing runs $2.00–3.50 per H100-hour in mid-2026.

| Item | Compute | Cost (range) |
|---|---|---|
| **Path A pilot** | | |
| Continued pretrain, 75B tokens | ~700 H100-hrs | $1.5–3K |
| Teacher trace generation (7.5B tokens from 1.5–3B teacher, vLLM) | ~600–800 GPU-hrs | $1.5–2.5K |
| SFT (~8B tokens, 2 epochs) | ~150 H100-hrs | $0.5–1K |
| RLVR (GRPO, generation-dominated, recurrence overhead) | ~3–8K GPU-hrs | $8–25K |
| QAT + deployment + eval | — | $1–3K |
| Ablations, restarts, contingency (~40%) | — | $5–12K |
| **Path A total** | | **~$20–45K** |
| **Path B (from scratch, 1T tokens)** | | |
| Pretraining: 6 × 1.6e9 × 1e12 ≈ 1e22 FLOPs | ~7,000 H100-hrs | $15–25K |
| Full post-training pipeline (as above) | | $12–30K |
| Ablations, restarts, contingency (~50–75%) | | $15–35K |
| **Path B total** | | **~$45–90K** |
| **Leaderboard-competitive tier** (10T+ tokens, extensive sweeps) | ~70K+ H100-hrs | **$300K–1M+** |

Non-compute costs: figure 2–3 strong ML engineers for 3–6 months for the pilot; the pipeline complexity (recurrent training loop, truncated BPTT, RL infra) is where projects like this actually stall, not the GPU bill. A 64-GPU cluster runs the Path B pretrain in about four to five days, so wall-clock is not the constraint.

The practical read: Path A is a $20–45K experiment that produces a definitive answer on whether recurrence-plus-workspace beats a same-cost standard 1B distillation. Only that answer justifies Path B.

## 6. Non-transformer architectures: the Mamba question

For this deployment target, hybrid state-space models deserve serious consideration — arguably more than the recurrent-transformer design above, on the inference-efficiency axis.

**Why it fits the problem.** An SSM layer carries a fixed-size state instead of a growing KV cache, so memory is flat in sequence length — which attacks exactly the browser bottleneck (long reasoning traces on constrained WebGPU memory). And the evidence that this works for reasoning specifically is now solid rather than speculative. M1, a hybrid Mamba reasoning model distilled from R1-style teachers and RL-tuned, matched same-scale R1-distilled transformers on AIME/MATH while generating more than 3x faster — and used that throughput to win under fixed wall-clock budgets via self-consistency voting, which is precisely the sampled-attempts strategy in our plan. NVIDIA's Nemotron line (Nemotron-H, Nemotron Nano 2, Nemotron 3) has repeatedly shown hybrid Mamba-attention models matching or beating same-size transformers with up to 3x+ generation throughput on long reasoning traces, at scales from 9B to 120B-MoE, trained on 20T+ tokens. Mamba-3 (ICLR 2026) and Gated DeltaNet variants have further closed the quality gap.

**Known weaknesses, honestly stated.** Pure SSMs are measurably worse at exact in-context recall, copying, and state-tracking (even bit-parity), and multi-step reasoning leans on referring back to earlier intermediate results precisely. The field's answer is hybridization: keep a minority of attention layers (roughly one attention layer per 5–8 SSM layers, some full, some sliding-window) and you recover retrieval capability while keeping ~80–90% of the memory and throughput win. At 1B scale I would not attempt a pure SSM reasoner; a hybrid is the defensible design.

**The browser-specific caveat, which is decisive in the near term.** WebGPU runtimes are mature for transformers (web-llm/MLC, ONNX Runtime Web) but Mamba's selective-scan kernel does not exist off-the-shelf in WGSL. Shipping a hybrid SSM in-browser means writing and optimizing custom kernels or compiling via MLC/TVM and accepting suboptimal performance initially — call it 1–2 engineer-months of skilled GPU work, with Safari/WebKit WebGPU immaturity compounding it. That cost is one-time and falling (the ecosystem is moving this direction), but it is real today.

**Recommendation.** Treat these as composable rather than competing. The recurrent-depth idea is architecture-agnostic — the looped core can be made of Mamba-2/GDN layers instead of attention layers, in which case memory is flat in both sequence length and loop count, and the SSM's compressed state plays the workspace role natively (a fixed-size privileged state carrying the reasoning — the same functional shape as J-space). The pragmatic sequence: build the Path A pilot on the transformer variant (mature browser toolchain, faster iteration), and in parallel run a small hybrid-SSM ablation on identical data at ~300–500M scale (~$3–5K) to measure the quality-per-memory tradeoff on your actual task distribution. If the SSM ablation holds quality within ~2 points at 3x throughput, the production 1B model should be the hybrid, and the WGSL kernel investment gets justified by measurement rather than fashion.

## 7. Risks and de-risking milestones

The main technical risks, each with a cheap early test. Recurrence may not transfer to language reasoning at this scale (looped transformers have shown strong results on algorithmic tasks but no decisive open win over token-space CoT for language) — test at 300M scale on GSM8K-tier tasks before any 1B spend. Retrofitted recurrence may underperform badly because base-model representations weren't shaped for iteration — compare retrofit-vs-scratch at 300M. Latent reasoning reduces trace legibility, which weakens debugging and safety monitoring — mitigate with workspace probes (Section 2.2) and keep a short visible trace rather than eliminating it. RL on a recurrent model is slower per rollout and less battle-tested in open frameworks — budget integration time with veRL, and fall back to SFT-plus-rejection-sampling if RL infra fights back. Finally, WebKit/Safari WebGPU remains the weakest deployment link regardless of architecture — validate an end-to-end quantized demo on target devices in month one, not month six.

## 8. Recommended sequence

Month 0–1: baseline first — standard-architecture 1B/1.5B distill-plus-verify running in-browser end to end (this is the shippable fallback and the yardstick). Month 1–2: 300M-scale ablations — recurrence on/off, workspace on/off, hybrid-SSM core on/off, identical data. Month 2–5: Path A pilot on whichever variant won, through SFT and RL. Decision gate: only if the pilot beats the baseline at matched inference memory does Path B (from-scratch, ~$45–90K) get funded.

---

*Key references: Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (2025); Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (TRM, 2025); Anthropic, "A global workspace in language models" (2026); WeiboAI, VibeThinker-1.5B/3B technical reports (2025–26); Wang et al., "M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models" (2025); NVIDIA Nemotron-H / Nano 2 / Nemotron 3 reports (2025–26); DeepScaleR (2025); Kang et al., KARD (2023); Google DeepMind, "Relaxed Recursive Transformers" (2024).*
