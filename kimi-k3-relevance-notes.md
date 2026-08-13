# Kimi K3 Architecture — Relevance to WRAP

Source: SemiAnalysis, "Kimi K3, The Manos, The Mythos, The Legendos" (2026-08-03)
https://newsletter.semianalysis.com/p/kimi-k3-the-manos-the-mythos-the

## Summary

The Kimi K3 article describes four major architectural innovations: Kimi Delta
Attention (KDA), Attention Residuals, LatentMoE, and Quantile load balancing.
Several of these directly validate design choices we already made in WRAP,
and one (Attention Residuals) suggests a potentially better solution to the
gradient amplification problem in Cell C's recurrent core.

---

## 1. QK L2 Normalization — VALIDATES OUR APPROACH

**Article**: KDA applies L2 norm to query and key "to stabilize the eigenvector
of the transition and the output matrices." This is standard in K3, Kimi Linear,
and modern frontier models.

**WRAP**: We implemented exactly this in `TTWorkspaceModule._l2_normalize_heads`
with learnable `read_qk_scale` / `write_qk_scale` (init 1/sqrt(d_head)). This was
the key fix that eliminated the entropy collapse / ill-conditioned QK^T causing
training divergence (see AGENTS.md, "Architecture v2 fixes").

**Verdict**: Direct validation. Moonshot independently arrived at the same
solution for the same problem (linear attention instability from unbounded QK
logits). No action needed.

---

## 2. Attention Residuals — NEW IDEA FOR CELL C

**Article**: Instead of standard residual connections
(`x_{l+1} = x_l + f_l(x_l)`), each layer performs softmax attention over the
**depth dimension** — attending over representations produced by previous
layers/blocks. Each layer learns a query vector `w_l` (not input-dependent),
and computes:

```
alpha_l = softmax(K_l @ w_l)      # K_l = previous block outputs
h_l = alpha_l^T @ V_l             # weighted sum of previous representations
```

This gives every layer **selective access** to earlier representations, rather
than relying solely on the residual stream to preserve information. Block
Attention Residuals reduces communication from O(Ld) to O(Nd) by grouping layers
into blocks and attending only over completed block outputs.

Key benefits reported:
- 1.25x compute efficiency vs standard residuals
- Consistently lower validation loss, gap widening with scale
- Bounded output magnitude (unlike standard residuals where output grows with depth)
- Consistent gradient magnitude across depth

**WRAP relevance**: Cell C's recurrent core currently blends iterations with
`x = blend * x_new + (1 - blend) * x` (1/sqrt(K) for x, 1/K for slots). This is
a fixed linear combination — every iteration contributes equally, and the
gradient amplification through the slot chain required the conservative 1/K
scaling fix. Attention Residuals could replace this blend with **learned
attention over iteration outputs**, giving the model:
- Selective access to any previous iteration's representation (not just the last)
- Bounded output magnitude (softmax normalizes) — directly addresses the
  gradient amplification problem
- Consistent gradient magnitude across K iterations — the exact property we've
  been trying to achieve with manual scaling hacks

**Proposed experiment**: After current Cell C training completes, implement an
"Attention Residual Core" variant where K iteration outputs are stacked and
attended over with a learned per-iteration query, replacing the fixed blend.
This could allow larger K without the 1/K scaling tax on signal strength.

**Caveat**: Attention Residuals add O(K * d) memory (storing iteration outputs)
and O(K^2 * d) compute (attention over K iterations). For K=6 this is trivial;
for large K it would need the block variant.

---

## 3. Short Convolution Before Q/K/V — POTENTIAL IMPROVEMENT

**Article**: KDA applies a short (left-padded causal) convolution to Q, K, V
before the linear attention computation, "effectively capturing local token
dependencies."

**WRAP**: Our `TTRetentionLayer` (the production linear attention layer) does
NOT have short convolution. The deprecated SSM layer has `d_conv=4` but
retention goes straight from `x` → linear projection → RoPE → attention. Adding
a short depthwise conv (kernel 3-4) before the QKV projection could improve
local token modeling, which matters for arithmetic reasoning where individual
digit positions carry critical information.

**Proposed experiment**: Add a depthwise conv1d (kernel=3, causal) to
`TTRetentionLayer` before the QKV projection. Low risk, small parameter count,
well-established technique.

---

## 4. DeltaNet Delta Rule — POTENTIAL IMPROVEMENT

**Article**: DeltaNet improves on linear attention by replacing the additive
update `S_t = S_{t-1} + beta_t * v_t * k_t^T` with the Delta Rule:
`S_t = S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) * k_t^T`. This performs
**targeted removal** of irrelevant associations, regularizing the growth of S
and improving long-range recall.

**WRAP**: Our retention layer uses simple decayed accumulation
(`D[t,s] = gamma^(t-s)`), which is closer to linear attention than DeltaNet.
The Delta Rule could improve the retention layer's ability to overwrite stale
information — relevant for multi-step reasoning where intermediate results
need to be replaced as computation progresses.

**Proposed experiment**: Implement a DeltaNet-style update in the retention
layer. This is a bigger change (different recurrence structure) but could be
tested as a drop-in replacement for one layer.

---

## 5. Per-Channel Decay (Diagonal Alpha) — POTENTIAL IMPROVEMENT

**Article**: KDA expands the forget gate `alpha` from a scalar (Gated DeltaNet)
into a **diagonal matrix** enabling fine-grained per-channel memory decay and
positional awareness.

**WRAP**: Our retention layer uses per-head gamma scalar
(`gamma[h]` for head h). KDA's per-channel decay would give `gamma[h, d]` for
each channel within each head — d_head times more granularity. This could allow
different feature dimensions to have different memory horizons (e.g., position
features decay fast, semantic features decay slow).

**Proposed experiment**: Expand gamma from `(n_heads,)` to `(n_heads, d_head)`
and apply as element-wise decay instead of scalar decay. Modest parameter
increase, straightforward implementation.

---

## 6. Hybrid Linear:Full Attention Ratio — DATA POINT

**Article**: Kimi Linear found **3:1 KDA:MLA** to be the ideal ratio balancing
performance and efficiency. KDA also serves as a position-aware operator,
replacing RoPE in MLA layers.

**WRAP**: We have 13 layers (Cell B/C) with attention at positions [5, 10],
giving **11:2 ≈ 5.5:1** linear:attention ratio. We're significantly more
linear-heavy than Kimi's optimum. Our retention layer uses RoPE (KDA replaces
it), so we're also carrying RoPE's overhead.

**Proposed experiment**: Try 3:1 ratio (e.g., 10 layers with attention at
[2, 5, 8] for 7:3 ≈ 2.3:1, or 12 layers with attention at [3, 6, 9] for 9:3 =
3:1). More attention layers increase compute cost but may improve reasoning
quality. The 3:1 ratio is a well-validated data point from a frontier model.

---

## 7. LatentMoE + Quantile Load Balancing — NOTED FOR FUTURE

**Article**: LatentMoE compresses routed tokens before dispatch, decompresses
after aggregation. Quantile Balancing (QB) is a hyperparameter-free, aux-loss-free
load balancing technique that computes router biases directly from the
distribution of router scores.

**WRAP**: Not applicable to current architecture (no MoE). Noted for future
scaling if we add mixture-of-experts. QB in particular is interesting because
it eliminates the load-balancing loss hyperparameter that's notoriously hard
to tune.

---

## 8. FlashKDA Chunked Computation — NOTED FOR FUTURE

**Article**: FlashKDA parallelizes the KDA recurrence by unrolling in chunks,
using a UT transform and Neumann factorization for the inverse. Two-kernel
design (K1: chunk prep, K2: recurrent compute). Prefill is O(T*D^2), decode is
constant.

**WRAP**: Our ttnn kernels already fuse element-wise ops to reduce DRAM
traffic (see AGENTS.md, "Custom fused kernels"). The chunked parallel
formulation could inspire optimizations if we need to scale to longer sequences
(currently T=128, where the simple matmul approach is fine).

---

## Priority Ranking for WRAP

| Priority | Idea | Effort | Expected Impact |
|----------|------|--------|-----------------|
| **High** | Attention Residuals for Cell C core | Medium | Could replace 1/K scaling hack with principled solution; enable larger K |
| **Medium** | Short conv before QKV in retention | Low | Better local token modeling for arithmetic |
| **Medium** | Per-channel decay (diagonal gamma) | Low | Finer-grained memory control |
| **Low** | DeltaNet Delta Rule in retention | High | Better memory overwriting, but big architecture change |
| **Low** | 3:1 linear:attention ratio | Medium | Data point from frontier model, but our model is much smaller |
| **Future** | LatentMoE + QB | High | Only relevant when adding MoE |
| **Future** | FlashKDA chunked kernels | High | Only relevant for long sequences |

---

## References

- Kimi K3 tech report: https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf
- Kimi Linear paper: https://arxiv.org/abs/2510.26692
- FlashKDA (open source): https://github.com/MoonshotAI/FlashKDA
- Attention Residuals paper: https://arxiv.org/pdf/2603.15031
- Quantile Balancing blog (Jianlin Su): https://kexue.fm/archives/11619
- vLLM K3 support: https://vllm.ai/blog/2026-07-27-k3
