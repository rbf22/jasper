# Refinements from Cactus Needle (Simple Attention Networks)

*Analysis of Needle by Cactus Compute — a 26M parameter attention-only function-calling model — and its implications for Jasper's architecture. Revised after a codebase review: verdicts and full implementation instructions below.*

Source: [cactus-compute/needle](https://github.com/cactus-compute/needle), [HN discussion](https://news.ycombinator.com/item?id=48111896), [Simple Attention Networks doc](https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md)

---

## Verdict summary

| Refinement | Verdict | Reason |
|---|---|---|
| 1. Gated residuals | **ADOPT** | Bigger than stated: the trunk currently has *no* residuals at all |
| 2. ZCRMSNorm | **ADOPT** | Trivial, pairs with #1; only makes sense with residuals in place |
| 3. z-loss | **ADOPT** | Cheap insurance; fp16 AMP + recurrent loop + reactive-only NaN guard |
| 4. Muon optimizer | **DEFER** | Collapse premise weak here: only 2 attention layers, Mamba is full of SiLU nonlinearities. Revisit only if #1–#2 don't stabilize Cell D |
| 5. INT4 QAT during training | **DEFER to 1B plan** | Needle's rationale is overfitting regularization; Jasper generates fresh data every batch, so it doesn't apply. Keep in 1B Phase 3 |
| 6. Token-level loss weighting | **SKIP** | Already done: `data.py` masks all non-answer positions with `-100` (prompt weight = 0, the extreme version) |
| 7. No-FFN core layers | **SKIP for POC / ablation for 1B** | POC has no FFN blocks anywhere. 1B plan does spec SwiGLU MLPs — keep as an ablation there, not a default |

The three ADOPT items are implemented together below. **They change the architecture, so old checkpoints are incompatible — delete `mamba-poc/checkpoints/cell*_latest.pt` or set `resume: false` before retraining.**

---

# Implementation guide

All changes are in `mamba-poc/model.py` and `mamba-poc/train.py`. Do them in order; steps 1 and 2 are one coherent change.

## Step 1: ZCRMSNorm (`model.py`)

Replace the `RMSNorm` class body. Keep the class name `RMSNorm` so all call sites (`Mamba2Layer.norm`, `WorkspaceModule.norm`/`slot_norm`, final `MambaWorkspaceModel.norm`) pick it up unchanged.

Current code (`model.py`, ~line 59):

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight
```

Replace with:

```python
class RMSNorm(nn.Module):
    """ZCRMSNorm: x * (1 + gamma) / RMS(x), gamma init 0 (identity-scale at init)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * (1 + self.weight)
```

Notes:
- `self.weight` is an `nn.Parameter`, not an `nn.Linear`, so `MambaWorkspaceModel._init_weights` will not clobber the zero init. No other change needed.
- Parameter count is unchanged.

## Step 2: Gated residuals + trunk residuals (`model.py`)

**Finding:** the model currently applies layers as `x = self.layers[i](x)` — the layer output *replaces* the stream. There is no `x + layer(x)` anywhere in the trunk. Fix by wrapping every trunk layer in a gated pre-norm residual block: `x = x + sigmoid(gate) * layer(norm(x))` with `gate` init 0 (so each block starts as a half-damped residual, per Needle).

### 2a. Add a `GatedBlock` wrapper class

Add after the `RMSNorm` class in `model.py`:

```python
class GatedBlock(nn.Module):
    """Gated pre-norm residual wrapper: x + sigmoid(gate) * layer(norm(x)).

    gate init 0 -> sigmoid(0) = 0.5, a damped residual at initialization.
    Lets the recurrent core learn to fade or sharpen each layer per iteration.
    """

    def __init__(self, layer: nn.Module, d_model: int):
        super().__init__()
        self.layer = layer
        self.norm = RMSNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.gate) * self.layer(self.norm(x))
```

### 2b. Wrap trunk layers in `MambaWorkspaceModel.__init__`

Current code (~line 339):

```python
self.layers = nn.ModuleList()
for i in range(config.n_layers):
    if config.use_attention and i in config.attention_positions:
        self.layers.append(AttentionLayer(config))
    else:
        self.layers.append(Mamba2Layer(config))
```

Replace with:

```python
self.layers = nn.ModuleList()
for i in range(config.n_layers):
    if config.use_attention and i in config.attention_positions:
        inner = AttentionLayer(config)
    else:
        inner = Mamba2Layer(config)
    self.layers.append(GatedBlock(inner, config.d_model))
```

No change is needed in `MambaWorkspaceModel.forward` — every `x = self.layers[i](x)` call site now goes through the gated residual automatically, including the K-loop in Phase 2.

### 2c. Gate the workspace residuals in `WorkspaceModule`

The workspace already has residuals; add gates to them. In `WorkspaceModule.__init__` (after `self.slot_norm = RMSNorm(...)`, ~line 282), add:

```python
self.read_gate = nn.Parameter(torch.zeros(1))
self.write_gate = nn.Parameter(torch.zeros(1))
```

In `WorkspaceModule.forward`, change the read residual (~line 307) from:

```python
slots = self.slot_norm(slots + self.read_out(read_out))  # residual + normalize to prevent growth over K iterations
```

to:

```python
slots = self.slot_norm(slots + torch.sigmoid(self.read_gate) * self.read_out(read_out))  # gated residual + normalize to prevent growth over K iterations
```

and the write residual (~line 318) from:

```python
x = x + self.write_out(write_out)  # residual update to hidden states
```

to:

```python
x = x + torch.sigmoid(self.write_gate) * self.write_out(write_out)  # gated residual update to hidden states
```

Notes:
- Gate parameters are plain `nn.Parameter`s, so `_init_weights` leaves them at zero. No init changes needed.
- With residuals now carrying the stream, the raw token embedding flows to the final norm untouched at init — this is the intended "damped identity" starting point that pairs with ZCRMSNorm.
- Param cost: +1 scalar per trunk layer (+13–14), +2 in the workspace, plus the `RMSNorm` inside each `GatedBlock` (`d_model` = 384 params each, ~5.4K total). Negligible; cells stay parameter-matched.

## Step 3: z-loss (`train.py`)

In the loss computation inside the training loop (~line 287), change:

```python
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
loss = loss / grad_accum_steps
```

to:

```python
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
# z-loss: penalize log-partition drift to keep logit scale stable (PaLM-style)
z = torch.logsumexp(shift_logits.float(), dim=-1)  # (B, T-1)
z_mask = shift_labels != -100
z_loss = z_loss_coef * (z[z_mask] ** 2).mean()
loss = loss + z_loss
loss = loss / grad_accum_steps
```

And read the coefficient from the config near the other hyperparameters (~line 192, next to `grad_clip`):

```python
z_loss_coef = cfg.get("z_loss_coef", 1e-4)
```

Notes:
- `1e-4` is the standard PaLM/Needle coefficient. Set `z_loss_coef: 0` in a YAML config to disable.
- `.float()` before `logsumexp` matters under fp16 AMP.
- The masked mean matches the cross-entropy's `ignore_index=-100` positions, so the two terms cover the same tokens.

## Step 4: Verify

```bash
# 1. Param counts still ~30M and matched across cells
cd mamba-poc && python model.py

# 2. End-to-end smoke test (Cell D with recurrence, 200 steps, no wandb)
python train.py --config configs/cell_d.yaml --smoke-test
```

Expect: loss decreases over the 200 smoke steps with no NaN skips. Then clear old checkpoints before real runs:

```bash
rm -f checkpoints/cell*_latest.pt
```

To measure the effect: run Cell D before/after on the same config and compare `depth{N}_acc` at high depths and the frequency of `NaN/Inf loss detected` lines.

---

# Original analysis (retained for reference)

## 1. Gated Residuals (highest impact for recurrent core) — ADOPTED

Needle uses `x = x + sigmoid(gate) * Attn(Norm(x))` with `gate` initialized to 0.

**Correction from code review:** the doc originally claimed Jasper uses standard `x = x + layer(x)` residuals. It doesn't — the trunk has *no* residual connections (layer outputs replace the stream). This makes the change more valuable than originally scoped: it is closer to a bug fix. The recurrent core (Cell D) re-applies the same layers K times; gated residuals let the model dampen or sharpen specific layers per iteration, directly addressing the stability of the recurrent depth loop.

## 2. ZCRMSNorm — ADOPTED

`x * (1 + gamma) / RMS(x)` with `gamma` init=0, identity-scale at initialization. Combined with gated residuals, each block starts as a damped identity. From the nGPT / DeepSeek-V3 line of work. Only meaningful once residuals exist (step 2), which is why the two ship together. Retrain from scratch: shapes are checkpoint-compatible but semantics are not.

## 3. Muon Optimizer — DEFERRED

Needle's argument: without FFN nonlinearities between attention layers, deep attention stacks suffer representation collapse, and Muon's orthogonalized updates prevent it. Jasper has only 2 attention layers and the Mamba2 layers are saturated with SiLU and gating, so the collapse premise doesn't cleanly transfer. A dual optimizer also complicates `save_checkpoint`/`load_checkpoint`/scheduler logic for uncertain gain at 30M scale. Revisit only if Cell D remains unstable after steps 1–3.

## 4. z-loss — ADOPTED

Training uses fp16 AMP on CUDA with tied embeddings, and NaN handling is purely reactive (skip-step guard). The recurrent core amplifies gradient paths K times. z-loss is a ~5-line proactive fix for the same failure class.

## 5. INT4 QAT During Training — DEFERRED to 1B plan

Needle's rationale is that quantization noise regularizes small models with high overfitting risk. Jasper's POC generates fresh data every batch — there is no overfitting to regularize. The deployment-robustness argument is real but belongs where it already is: `workspace-recurrent-1b-plan.md` Phase 3. Adding QAT to the POC would also add noise to the A/B/C/D comparison, which is the POC's purpose.

## 6. Token-Level Loss Weighting — SKIPPED

Needle weights loss by token type (structure 1x, values 4x). Jasper already masks all non-answer positions with `-100` in `data.py` — prompt tokens get zero weight, the extreme version of this idea. Answers are a few characters, so intra-answer weighting has nothing to differentiate.

## 7. No-FFN Parameter Budget — SKIPPED for POC, ablation for 1B

Needle's finding: at <50M scale on structured tasks, FFN params are wasted. The POC has no FFN blocks anywhere (`Mamba2Layer`'s expand-4 `in_proj`/gating plays that role internally; `AttentionLayer` has no MLP), so there is nothing to remove. The 1B plan (`workspace-recurrent-1b-plan.md` §2.1) does spec SwiGLU MLPs at 8/3 expansion — removing FFN from the 4 recurrent-core layers is a legitimate ablation there (cheaper K-loops, freed param budget), but Needle's evidence is at <50M scale on retrieval-style tasks, so treat it as an experiment, not a default.
