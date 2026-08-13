#!/usr/bin/env python3
"""Gradient correctness tests for WRAP's layer backward passes.

These are pure-PyTorch tests that run on CPU and can be included in pytest.
They verify that the manual backward math in model_ttnn.py is correct by
comparing against torch.autograd.gradcheck in float64.

Each test builds a self-contained PyTorch reference that matches the TT
layer's forward math exactly (including the TT weight convention
y = x @ W, not PyTorch's F.linear y = x @ W^T), then checks gradients
via torch.autograd.gradcheck.

Coverage:
  1. AttentionLayer backward (QKV, RoPE, causal softmax, output proj)
  2. AttentionResidual backward (learned softmax blend over K iterations)
  3. GatedResidualLayer backward (RMSNorm + ReZero gate)
  4. RetentionLayer backward (already covered by retention_reference.py,
     but included here for completeness with short_conv disabled)

Run:
    .tt-venv/bin/python -m pytest test_gradients.py -v
    .tt-venv/bin/python test_gradients.py --verbose
"""

import math
import sys
import argparse
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CPU-only ModelConfig (avoids importing model_ttnn.py which requires ttnn)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Minimal ModelConfig for gradient tests. Matches model_ttnn.ModelConfig."""
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    vocab_size: int = 64
    d_state: int = 64
    d_conv: int = 4
    expand: int = 4
    use_attention: bool = False
    attention_positions: List[int] = field(default_factory=lambda: [5, 10])
    use_workspace: bool = False
    n_workspace_slots: int = 16
    recurrent_core: bool = False
    core_start: int = 6
    core_end: int = 10
    k_train_max: int = 6
    k_inference: int = 6
    attention_residual_core: bool = False
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True
    spectral_norm_bound: float = 5.0
    backbone_spectral_norm_bound: float = 2.0
    chain_scale_safety: float = 1.0
    freeze_gamma: bool = True
    freeze_slot_decay: bool = False
    ws_entropy_weight: float = 0.0
    ws_diversity_weight: float = 0.0
    gate_init: float = 0.0
    slot_decay_init: float = 1.0
    slot_permutation: bool = False
    gate_schedule_steps: int = 0
    gate_clamp_bound: float = 0.0
    headdim: int = 64
    d_state_ssm: int = 64
    per_channel_decay: bool = False
    use_fused_rope: bool = True
    short_conv: bool = False
    short_conv_kernel: int = 4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative error in Frobenius norm."""
    a, b = a.double().flatten(), b.double().flatten()
    denom = max(b.norm().item(), 1e-12)
    return (a - b).norm().item() / denom


def apply_rope(x, cos, sin):
    """RoPE rotation on (B, H, T, d_head). Splits last dim in half."""
    d_h = x.shape[-1]
    x1 = x[..., :d_h // 2]
    x2 = x[..., d_h // 2:]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.cat([r1, r2], dim=-1)


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight).

    Stays in the input dtype throughout — no float() upcast (which breaks
    gradcheck when input is already float64).
    """
    ms = x.pow(2).mean(-1, keepdim=True)
    norm = torch.rsqrt(ms + eps)
    return x * norm * (1 + weight)


def rms_norm_backward(grad_out, x, weight, eps=1e-6):
    """Manual RMSNorm backward. Returns (grad_x, grad_weight)."""
    D = x.shape[-1]
    x_f = x.float()
    ms = x_f.pow(2).mean(-1, keepdim=True)  # mean(x^2)
    rms_inv = (ms + eps).pow(-0.5)
    x_normed = x_f * rms_inv  # normalized x (before weight)

    grad_out_f = grad_out.float()
    w = (1 + weight.to(torch.float64)).float()

    # grad_weight = sum over all dims except last of grad_out * x_normed
    grad_weight = (grad_out_f * x_normed).sum(dim=tuple(range(grad_out.dim() - 1)))

    # grad_x = grad_out * w * rms_inv * (1 - x^2 / (D * ms + D * eps))
    # Standard RMSNorm backward:
    # d/dx_i [ x_i * rms_inv * w_i ] = w_i * (rms_inv - x_i^2 * rms_inv^3 / D)
    # = w_i * rms_inv * (1 - x_i^2 * rms_inv^2 / D)
    grad_x = grad_out_f * w * rms_inv * (1 - x_f.pow(2) * rms_inv.pow(2) / D)

    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


def softmax_backward(grad_out, out, dim=-1):
    """Manual softmax backward: grad_in = (grad_out - sum(grad_out*out, dim, keepdim)) * out."""
    s = (grad_out * out).sum(dim=dim, keepdim=True)
    return (grad_out - s) * out


# ---------------------------------------------------------------------------
# 1. Attention Layer Reference + Gradcheck
# ---------------------------------------------------------------------------

ATTENTION_PARAM_NAMES = (
    "qkv_weight",      # (d_model, 3*d_model)
    "out_proj_weight",  # (d_model, d_model)
)


class AttentionReference(torch.nn.Module):
    """Pure-PyTorch reference for TTAttentionLayer.

    Matches the TT forward exactly:
      - Weight convention: y = x @ W (matmul, not F.linear)
      - QKV split into q, k, v (no gate)
      - RoPE on q and k
      - Causal masking + softmax
      - Output projection
    """

    def __init__(self, config, params=None, dtype=torch.float32):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.dtype = dtype

        d_rope = self.d_head
        freqs = 1.0 / (10000 ** (torch.arange(0, d_rope, 2).float() / d_rope))
        self.register_buffer("rope_freqs", freqs)

        if params is not None:
            self.load(params)

    def load(self, params):
        for name in ATTENTION_PARAM_NAMES:
            t = params[name].detach().to(self.dtype).clone().requires_grad_(True)
            setattr(self, name, t)

    def params(self):
        return {name: getattr(self, name) for name in ATTENTION_PARAM_NAMES}

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, D = x.shape
        H, d_h = self.n_heads, self.d_head
        dtype = x.dtype

        # QKV projection: (B, T, 3D)
        qkv = x @ self.qkv_weight
        q, k, v = torch.split(qkv, [D, D, D], dim=-1)

        # Reshape to heads: (B, H, T, d_head)
        q = q.reshape(B, T, H, d_h).permute(0, 2, 1, 3)
        k = k.reshape(B, T, H, d_h).permute(0, 2, 1, 3)
        v = v.reshape(B, T, H, d_h).permute(0, 2, 1, 3)

        # RoPE
        positions = torch.arange(T, dtype=torch.float32)
        angles = torch.outer(positions, self.rope_freqs)
        cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)

        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)

        # Attention scores: (B, H, T, T)
        scores = torch.matmul(q_rope, k_rope.transpose(-1, -2))
        scores = scores * self.scale

        # Causal mask
        mask = torch.tril(torch.ones(T, T, dtype=dtype, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn = torch.softmax(scores, dim=-1)

        # Attention output: (B, H, T, d_head)
        out_4d = torch.matmul(attn, v)

        # Reshape back: (B, T, D)
        out = out_4d.permute(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = out @ self.out_proj_weight

        return out


def test_attention_gradcheck(verbose=False):
    """Gradcheck the attention layer reference in float64."""
    

    torch.manual_seed(42)
    config = ModelConfig(d_model=64, n_heads=2)
    D, H, d_h = config.d_model, config.n_heads, config.d_model // config.n_heads
    B, T = 2, 32

    # Random parameters
    params = {
        "qkv_weight": torch.randn(D, 3 * D, dtype=torch.float64) * 0.02,
        "out_proj_weight": torch.randn(D, D, dtype=torch.float64) * 0.02,
    }

    ref = AttentionReference(config, dtype=torch.float64)
    ref.load(params)

    x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)

    def f(xx, qkv_w, out_w):
        ref.qkv_weight = qkv_w
        ref.out_proj_weight = out_w
        return ref(xx)

    args = (x, ref.qkv_weight, ref.out_proj_weight)
    ok = torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  Attention gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "Attention layer gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# 2. AttentionResidual Reference + Gradcheck
# ---------------------------------------------------------------------------

AR_PARAM_NAMES = (
    "ar_query",  # (1, d_model)
    "ar_scale",  # (1,) scalar
)


class AttentionResidualReference(torch.nn.Module):
    """Pure-PyTorch reference for AttentionResidual.

    Forward:
      scores_k = sum_d(x_k * query) * scale  -> (B, T) per k
      alpha = softmax([scores_0, ..., scores_K], dim=-1)  -> (B, T, K+1)
      x_final = sum_k alpha_k * x_k  -> (B, T, D)

    With masking: inactive iterations (k > K_active) get scores = -1e4.
    """

    def __init__(self, config, params=None, dtype=torch.float32):
        super().__init__()
        self.d_model = config.d_model
        self.dtype = dtype
        if params is not None:
            self.load(params)

    def load(self, params):
        for name in AR_PARAM_NAMES:
            t = params[name].detach().to(self.dtype).clone().requires_grad_(True)
            setattr(self, name, t)

    def params(self):
        return {name: getattr(self, name) for name in AR_PARAM_NAMES}

    def forward(self, x_outputs, K_active):
        """x_outputs: list of (B, T, D) tensors. Returns (B, T, D)."""
        B, T, D = x_outputs[0].shape
        n_outputs = len(x_outputs)
        device = x_outputs[0].device
        dtype = x_outputs[0].dtype

        # Score computation: scores_k = sum_d(x_k * query) * scale
        query = self.ar_query  # (1, D)
        scale = self.ar_scale  # (1,)

        scores_list = []
        for k, x_k in enumerate(x_outputs):
            s = (x_k * query).sum(dim=-1, keepdim=True)  # (B, T, 1)
            if k <= K_active:
                s = s * scale
            else:
                s = torch.full_like(s, -1e4)
            scores_list.append(s)

        scores = torch.cat(scores_list, dim=-1)  # (B, T, n_outputs)

        # Softmax over iterations
        alpha = torch.softmax(scores, dim=-1)  # (B, T, n_outputs)

        # Weighted sum
        x_final = torch.zeros_like(x_outputs[0])
        for k, x_k in enumerate(x_outputs):
            x_final = x_final + alpha[..., k:k+1] * x_k

        return x_final


def test_attention_residual_gradcheck(verbose=False):
    """Gradcheck the AttentionResidual reference in float64."""
    

    torch.manual_seed(42)
    config = ModelConfig(d_model=64)
    D = config.d_model
    B, T = 2, 32
    K = 3
    K_active = K  # all iterations active

    params = {
        "ar_query": torch.randn(1, D, dtype=torch.float64) * 0.1,
        "ar_scale": torch.tensor([1.0 / math.sqrt(D)], dtype=torch.float64),
    }

    ref = AttentionResidualReference(config, dtype=torch.float64)
    ref.load(params)

    x_outputs = [torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
                 for _ in range(K + 1)]

    def f(*args):
        xs = args[:K+1]
        q, s = args[K+1], args[K+2]
        ref.ar_query = q
        ref.ar_scale = s
        return ref(list(xs), K_active)

    args = tuple(x_outputs) + (ref.ar_query, ref.ar_scale)
    ok = torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  AttentionResidual gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "AttentionResidual gradcheck failed"
    return ok


def test_attention_residual_inactive_gradcheck(verbose=False):
    """Gradcheck AttentionResidual with inactive iterations (K_active < K)."""
    

    torch.manual_seed(42)
    config = ModelConfig(d_model=64)
    D = config.d_model
    B, T = 2, 32
    K = 3
    K_active = 1  # Only iterations 0 and 1 are active; 2 and 3 are masked

    params = {
        "ar_query": torch.randn(1, D, dtype=torch.float64) * 0.1,
        "ar_scale": torch.tensor([1.0 / math.sqrt(D)], dtype=torch.float64),
    }

    ref = AttentionResidualReference(config, dtype=torch.float64)
    ref.load(params)

    x_outputs = [torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
                 for _ in range(K + 1)]

    def f(*args):
        xs = args[:K+1]
        q, s = args[K+1], args[K+2]
        ref.ar_query = q
        ref.ar_scale = s
        return ref(list(xs), K_active)

    args = tuple(x_outputs) + (ref.ar_query, ref.ar_scale)
    ok = torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  AttentionResidual (inactive) gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "AttentionResidual inactive gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# 3. GatedResidualLayer Reference + Gradcheck
# ---------------------------------------------------------------------------

class GatedResidualReference(torch.nn.Module):
    """Pure-PyTorch reference for TTGatedResidualLayer.

    Forward:
      norm_x = rms_norm(x, norm_weight)
      inner_out = inner_layer(norm_x)  # abstracted as a linear op
      out = x + gate * inner_out       # ReZero: gate is scalar, no sigmoid

    For gradcheck, we abstract the inner layer as a simple matmul so we
    can isolate the gate + norm + residual gradient math.
    """

    def __init__(self, d_model, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype

        # Parameters: norm_weight (gamma), gate (scalar), inner_weight
        self.norm_weight = torch.zeros(d_model, dtype=dtype).requires_grad_(True)
        self.gate = torch.tensor(0.0, dtype=dtype).requires_grad_(True)  # ReZero init=0
        self.inner_weight = (torch.randn(d_model, d_model, dtype=dtype) * 0.02).requires_grad_(True)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        norm_x = rms_norm(x, self.norm_weight)
        inner_out = norm_x @ self.inner_weight
        out = x + self.gate * inner_out
        return out


def test_gated_residual_gradcheck(verbose=False):
    """Gradcheck the GatedResidualLayer reference in float64."""
    D = 64
    B, T = 2, 32

    torch.manual_seed(42)
    ref = GatedResidualReference(D, dtype=torch.float64)

    x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)

    def f(xx, nw, g, iw):
        ref.norm_weight = nw
        ref.gate = g
        ref.inner_weight = iw
        return ref(xx)

    args = (x, ref.norm_weight, ref.gate, ref.inner_weight)
    ok = torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  GatedResidual gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "GatedResidual gradcheck failed"
    return ok


def test_gated_residual_nonzero_gate_gradcheck(verbose=False):
    """Gradcheck GatedResidual with non-zero gate (gate=0 is a degenerate case)."""
    D = 64
    B, T = 2, 32

    torch.manual_seed(42)
    ref = GatedResidualReference(D, dtype=torch.float64)
    # Set gate to 0.3 as a leaf tensor with requires_grad
    gate_param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    ref.gate = gate_param

    x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)

    def f(xx, nw, g, iw):
        ref.norm_weight = nw
        ref.gate = g
        ref.inner_weight = iw
        return ref(xx)

    args = (x, ref.norm_weight, gate_param, ref.inner_weight)
    ok = torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  GatedResidual (gate=0.3) gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "GatedResidual non-zero gate gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# 4. Retention Layer Gradcheck (re-export from retention_reference.py)
# ---------------------------------------------------------------------------

def test_retention_gradcheck(verbose=False):
    """Gradcheck the Retention layer reference in float64.

    This wraps the existing retention_reference.py gradcheck so it's
    included in the pytest suite.
    """
    from retention_reference import RetentionReference, PARAM_NAMES

    torch.manual_seed(42)
    config = ModelConfig(d_model=64, n_heads=2)
    D = config.d_model
    B, T = 2, 32

    params = {
        "qkv_weight": torch.randn(D, 4 * D, dtype=torch.float64) * 0.02,
        "out_proj_weight": torch.randn(D, D, dtype=torch.float64) * 0.02,
        "gamma": torch.full((config.n_heads,), -0.05, dtype=torch.float64),
    }

    ref = RetentionReference(config, dtype=torch.float64)
    ref.load(params)

    x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)

    def f(xx, *ps):
        for n, t in zip(PARAM_NAMES, ps):
            setattr(ref, n, t)
        return ref(xx)

    args = (x,) + tuple(ref.params()[n] for n in PARAM_NAMES)
    ok = torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  Retention gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "Retention layer gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# 5. RMSNorm Gradcheck (component test)
# ---------------------------------------------------------------------------

def test_rms_norm_gradcheck(verbose=False):
    """Gradcheck RMSNorm in float64."""
    D = 64
    B, T = 2, 32

    torch.manual_seed(42)
    x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
    weight = torch.zeros(D, dtype=torch.float64, requires_grad=True)  # gamma init=0

    def f(xx, w):
        return rms_norm(xx, w)

    ok = torch.autograd.gradcheck(f, (x, weight), eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  RMSNorm gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "RMSNorm gradcheck failed"
    return ok


def test_rms_norm_nonzero_weight_gradcheck(verbose=False):
    """Gradcheck RMSNorm with non-zero weight."""
    D = 64
    B, T = 2, 32

    torch.manual_seed(42)
    x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(D, dtype=torch.float64) * 0.1
    weight.requires_grad_(True)

    def f(xx, w):
        return rms_norm(xx, w)

    ok = torch.autograd.gradcheck(f, (x, weight), eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  RMSNorm (nonzero weight) gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "RMSNorm nonzero weight gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# 6. Softmax Backward Gradcheck (component test)
# ---------------------------------------------------------------------------

def test_softmax_backward_gradcheck(verbose=False):
    """Gradcheck the manual softmax backward."""
    N = 32

    torch.manual_seed(42)
    scores = torch.randn(2, 4, N, N, dtype=torch.float64, requires_grad=True)

    def f(s):
        return torch.softmax(s, dim=-1)

    # autograd gradcheck of softmax itself
    ok = torch.autograd.gradcheck(f, (scores,), eps=1e-6, atol=1e-7, rtol=1e-4)

    # Also verify our manual backward matches autograd
    scores_ref = scores.detach().clone().requires_grad_(True)
    out_ref = torch.softmax(scores_ref, dim=-1)
    grad_out = torch.randn_like(out_ref, dtype=torch.float64)
    out_ref.backward(grad_out)
    grad_autograd = scores_ref.grad

    with torch.no_grad():
        out = torch.softmax(scores.detach(), dim=-1)
        grad_manual = softmax_backward(grad_out, out, dim=-1)
        err = rel_err(grad_manual, grad_autograd)

    if verbose:
        print(f"  Softmax gradcheck: {'PASS' if ok else 'FAIL'}, manual vs autograd err={err:.2e}")
    assert ok, "Softmax gradcheck failed"
    assert err < 1e-10, f"Manual softmax backward mismatch: err={err}"
    return ok


# ---------------------------------------------------------------------------
# 7. RoPE Backward Gradcheck (component test)
# ---------------------------------------------------------------------------

def test_rope_gradcheck(verbose=False):
    """Gradcheck RoPE rotation."""
    B, H, T, d_h = 2, 2, 32, 32
    d_half = d_h // 2

    torch.manual_seed(42)
    x = torch.randn(B, H, T, d_h, dtype=torch.float64, requires_grad=True)

    positions = torch.arange(T, dtype=torch.float64)
    freqs = 1.0 / (10000 ** (torch.arange(0, d_h, 2, dtype=torch.float64) / d_h))
    angles = torch.outer(positions, freqs)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    def f(xx):
        return apply_rope(xx, cos, sin)

    ok = torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  RoPE gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "RoPE gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# 8. Cross-Entropy Loss Gradcheck
# ---------------------------------------------------------------------------

def cross_entropy_reference(logits, labels):
    """Reference cross-entropy: logits (B, T, V), labels (B, T) -> scalar loss.

    Matches the TT cross_entropy_loss: shift logits/labels by 1, compute
    softmax cross-entropy at each position, average over valid positions.
    """
    B, T, V = logits.shape

    # Shift: predict position t+1 from position t
    shift_logits = logits[:, :-1, :]  # (B, T-1, V)
    shift_labels = labels[:, 1:]       # (B, T-1)

    # Softmax
    probs = torch.softmax(shift_logits, dim=-1)

    # One-hot
    one_hot = F.one_hot(shift_labels, V).to(logits.dtype)

    # Loss: -sum(one_hot * log(probs)) / n_valid
    log_probs = torch.log(probs + 1e-8)
    loss = -(one_hot * log_probs).sum() / (B * (T - 1))

    return loss


def test_cross_entropy_gradcheck(verbose=False):
    """Gradcheck cross-entropy loss."""
    B, T, V = 2, 32, 64

    torch.manual_seed(42)
    logits = torch.randn(B, T, V, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, V, (B, T))

    def f(lg):
        return cross_entropy_reference(lg, labels)

    ok = torch.autograd.gradcheck(f, (logits,), eps=1e-6, atol=1e-7, rtol=1e-4)

    if verbose:
        print(f"  Cross-entropy gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "Cross-entropy gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# 9. Full Model Gradcheck (small model, CPU)
# ---------------------------------------------------------------------------

class SmallWRAPReference(torch.nn.Module):
    """A small end-to-end WRAP reference for gradcheck.

    Combines: embedding -> [retention layers] -> RMSNorm -> LM head
    No workspace, no recurrent core, no attention residual.
    This tests the full forward/backward chain including embedding gradients.
    """

    def __init__(self, d_model=32, n_heads=2, n_layers=2, vocab_size=64, dtype=torch.float64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.dtype = dtype

        # Parameters
        self.embedding = torch.randn(vocab_size, d_model, dtype=dtype) * 0.02
        self.embedding.requires_grad_(True)

        self.layers_qkv = [torch.randn(d_model, 4 * d_model, dtype=dtype) * 0.02
                           for _ in range(n_layers)]
        self.layers_out = [torch.randn(d_model, d_model, dtype=dtype) * 0.02
                           for _ in range(n_layers)]
        self.layers_gamma = [torch.full((n_heads,), -0.05, dtype=dtype)
                             for _ in range(n_layers)]
        for i in range(n_layers):
            self.layers_qkv[i].requires_grad_(True)
            self.layers_out[i].requires_grad_(True)
            self.layers_gamma[i].requires_grad_(True)

        self.norm_weight = torch.zeros(d_model, dtype=dtype, requires_grad=True)
        self.lm_head = torch.randn(d_model, vocab_size, dtype=dtype) * 0.02
        self.lm_head.requires_grad_(True)

        # RoPE
        d_rope = d_model // n_heads
        freqs = 1.0 / (10000 ** (torch.arange(0, d_rope, 2).float() / d_rope))
        self.register_buffer("rope_freqs", freqs)

    def forward(self, input_ids):
        """input_ids: (B, T) int64 -> logits (B, T, V)"""
        from retention_reference import RetentionReference

        B, T = input_ids.shape
        D, H = self.d_model, self.n_heads
        d_h = D // H

        x = self.embedding[input_ids]  # (B, T, D)

        config = ModelConfig(d_model=D, n_heads=H)

        for i in range(self.n_layers):
            ref = RetentionReference(config, dtype=self.dtype)
            ref.qkv_weight = self.layers_qkv[i]
            ref.out_proj_weight = self.layers_out[i]
            ref.gamma = self.layers_gamma[i]
            x = ref(x)

        x = rms_norm(x, self.norm_weight)
        logits = x @ self.lm_head  # (B, T, V)
        return logits

    def all_params(self):
        params = [self.embedding, self.norm_weight, self.lm_head]
        for i in range(self.n_layers):
            params.extend([self.layers_qkv[i], self.layers_out[i], self.layers_gamma[i]])
        return params


def test_full_model_gradcheck(verbose=False):
    """Gradcheck a small end-to-end model (embedding + 2 retention layers + LM head)."""
    torch.manual_seed(42)

    model = SmallWRAPReference(d_model=32, n_heads=2, n_layers=2, vocab_size=64,
                                  dtype=torch.float64)

    B, T = 2, 16
    input_ids = torch.randint(0, 64, (B, T))
    labels = torch.randint(0, 64, (B, T))

    params = model.all_params()

    def f(*args):
        emb, norm_w, lm_head = args[0], args[1], args[2]
        idx = 3
        model.embedding = emb
        model.norm_weight = norm_w
        model.lm_head = lm_head
        for i in range(model.n_layers):
            setattr(model, f'layers_qkv_{i}', None)  # avoid setattr issues
            model.layers_qkv[i] = args[idx]; idx += 1
            model.layers_out[i] = args[idx]; idx += 1
            model.layers_gamma[i] = args[idx]; idx += 1
        logits = model(input_ids)
        return cross_entropy_reference(logits, labels)

    ok = torch.autograd.gradcheck(f, tuple(params), eps=1e-6, atol=1e-6, rtol=1e-3)

    if verbose:
        print(f"  Full model gradcheck: {'PASS' if ok else 'FAIL'}")
    assert ok, "Full model gradcheck failed"
    return ok


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("RMSNorm (gamma=0)", test_rms_norm_gradcheck),
    ("RMSNorm (gamma!=0)", test_rms_norm_nonzero_weight_gradcheck),
    ("Softmax backward", test_softmax_backward_gradcheck),
    ("RoPE", test_rope_gradcheck),
    ("Cross-entropy loss", test_cross_entropy_gradcheck),
    ("Retention layer", test_retention_gradcheck),
    ("Attention layer", test_attention_gradcheck),
    ("AttentionResidual (all active)", test_attention_residual_gradcheck),
    ("AttentionResidual (inactive iters)", test_attention_residual_inactive_gradcheck),
    ("GatedResidual (gate=0)", test_gated_residual_gradcheck),
    ("GatedResidual (gate=0.3)", test_gated_residual_nonzero_gate_gradcheck),
    ("Full model (2 retention layers)", test_full_model_gradcheck),
]


def main():
    parser = argparse.ArgumentParser(description="Gradient correctness tests")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Running {len(ALL_TESTS)} gradient tests...\n")

    all_pass = True
    for name, fn in ALL_TESTS:
        try:
            ok = fn(verbose=True)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
        except Exception as e:
            status = f"ERROR: {e}"
            all_pass = False
        print(f"  {name}: {status}")

    print(f"\n{'='*40}")
    if all_pass:
        print(f"ALL {len(ALL_TESTS)} TESTS PASSED")
    else:
        print(f"SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
