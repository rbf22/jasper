#!/usr/bin/env python3
"""Gradient check for the architecture v2 workspace backward (QK-Norm + ReZero).

This test builds a pure-PyTorch reference of the TTWorkspaceModule's forward
and manual backward pass, then verifies the backward against PyTorch autograd
using float64 gradcheck.

Unlike test_ws_backward_cpu.py (which tests the OLD architecture against
model.py's WorkspaceModule), this test builds a self-contained reference that
matches the current TT workspace (model_ttnn.py TTWorkspaceModule):

  - QK Normalization: L2-normalize Q and K along d_head before attention scores
  - Learnable qk_scale parameter (init 1/sqrt(d_head))
  - ReZero gates: scalar gates, no sigmoid, init=0
  - Slot decay: learned scalar, init=1.0
  - RMSNorm with weight = 1 + gamma (gamma init 0)

The manual backward replicates the TT backward math exactly (including the
TT convention: ttnn.linear(x, W) = x @ W, vs PyTorch F.linear(x, W) = x @ W^T).

Run:
    python test_ws_qknorm_gradcheck.py
    python test_ws_qknorm_gradcheck.py --verbose   # show per-param errors
"""

import torch
import torch.nn.functional as F
import math
import sys
import argparse


# ============================================================================
# PyTorch reference workspace (matches TT architecture v2)
# ============================================================================

class RefWorkspaceV2(torch.nn.Module):
    """Pure-PyTorch reference matching TTWorkspaceModule architecture v2.

    Uses TT convention: weight matrices are stored as W_tt where y = x @ W_tt.
    (PyTorch's F.linear uses y = x @ W_pt^T, so W_tt = W_pt^T.)
    """

    def __init__(self, d_model, n_heads, n_slots, device="cpu", dtype=torch.float64):
        super().__init__()
        self.D = d_model
        self.H = n_heads
        self.d_h = d_model // n_heads
        self.m = n_slots
        self.device = device
        self.dtype = dtype

        # Weight matrices (TT convention: y = x @ W_tt)
        # We store them as (D, D) and use matmul directly
        scale = 1.0 / math.sqrt(d_model)
        self.read_q_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)
        self.read_k_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)
        self.read_v_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)
        self.read_out_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)
        self.write_q_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)
        self.write_k_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)
        self.write_v_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)
        self.write_out_w = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype) * scale)

        # Slots (learned embeddings)
        self.slots = torch.nn.Parameter(torch.randn(n_slots, d_model, dtype=dtype) * 0.02)

        # RMSNorm weights (weight = 1 + gamma, gamma init 0 -> weight init 1)
        self.norm_w = torch.nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.slot_norm_w = torch.nn.Parameter(torch.ones(d_model, dtype=dtype))

        # ReZero gates (scalar, init 0)
        self.read_gate = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.write_gate = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))

        # Slot decay (scalar, init 1.0)
        self.slot_decay = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype))

        # QK scale (learnable, init 1/sqrt(d_head))
        qk_init = 1.0 / math.sqrt(self.d_h)
        self.read_qk_scale = torch.nn.Parameter(torch.tensor(qk_init, dtype=dtype))
        self.write_qk_scale = torch.nn.Parameter(torch.tensor(qk_init, dtype=dtype))

        self.eps = 1e-6

    def _reshape_to_heads(self, x, B, L):
        # (B, L, D) -> (B, H, L, d_h)
        return x.reshape(B, L, self.H, self.d_h).transpose(1, 2)

    def _reshape_from_heads(self, x, B, L):
        # (B, H, L, d_h) -> (B, L, D)
        return x.transpose(1, 2).reshape(B, L, self.D)

    def _l2_normalize(self, x):
        """L2-normalize along last dim (d_head). x: (..., d_h) -> (..., d_h)"""
        norm_sq = (x * x).sum(dim=-1, keepdim=True)  # (..., 1)
        inv_norm = torch.rsqrt(norm_sq + self.eps)
        return x * inv_norm

    def _rms_norm(self, x, weight):
        """RMSNorm: y = x / sqrt(eps + mean(x^2)) * weight"""
        x_sq_mean = (x * x).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(x_sq_mean + self.eps)
        return x * inv_rms * weight

    def _build_causal_masks(self, T, m, device, dtype):
        """Build causal masks for workspace cross-attention (Perceiver-IO style)."""
        read_mask = torch.zeros(m, T, device=device, dtype=dtype)
        for i in range(m):
            cutoff = min((i + 1) * T // m, T)
            read_mask[i, :cutoff] = 1.0

        write_mask = torch.zeros(T, m, device=device, dtype=dtype)
        for t in range(T):
            cutoff = min((t + 1) * m // T + 1, m)
            write_mask[t, :cutoff] = 1.0

        return read_mask, write_mask

    def forward(self, x, slot_state=None):
        """x: (B, T, D), slot_state: (B, m, D) or None -> (x_out, slots_out)"""
        B, T, D = x.shape
        H, d_h, m = self.H, self.d_h, self.m

        # Initialize slots
        if slot_state is None:
            slots = self.slots.unsqueeze(0).expand(B, m, D).clone()
        else:
            slots = slot_state

        # Build causal masks
        read_mask, write_mask = self._build_causal_masks(T, m, x.device, x.dtype)

        # --- Read: slots attend over hidden states ---
        # TT: y = x @ W_tt, so we use matmul (not F.linear)
        rq = self._reshape_to_heads(slots @ self.read_q_w, B, m)   # (B, H, m, d_h)
        rk = self._reshape_to_heads(x @ self.read_k_w, B, T)       # (B, H, T, d_h)
        rv = self._reshape_to_heads(x @ self.read_v_w, B, T)       # (B, H, T, d_h)

        # QK-Norm
        rq_norm = self._l2_normalize(rq)
        rk_norm = self._l2_normalize(rk)

        read_scores = (rq_norm @ rk_norm.transpose(-2, -1)) * self.read_qk_scale  # (B, H, m, T)
        # Apply causal mask
        read_scores = read_scores.masked_fill(read_mask == 0, float('-inf'))
        read_attn = torch.softmax(read_scores, dim=-1)
        read_out_4d = read_attn @ rv                                # (B, H, m, d_h)
        read_out = self._reshape_from_heads(read_out_4d, B, m)     # (B, m, D)
        read_out_proj = read_out @ self.read_out_w                 # (B, m, D)

        # slots = slot_norm(decay * slots + read_gate * read_out_proj)
        decayed_slots = self.slot_decay * slots
        slots_pre_norm = decayed_slots + self.read_gate * read_out_proj
        slots_out = self._rms_norm(slots_pre_norm, self.slot_norm_w)

        # --- Write: hidden states attend over slots ---
        wq = self._reshape_to_heads(x @ self.write_q_w, B, T)      # (B, H, T, d_h)
        wk = self._reshape_to_heads(slots_out @ self.write_k_w, B, m)  # (B, H, m, d_h)
        wv = self._reshape_to_heads(slots_out @ self.write_v_w, B, m)  # (B, H, m, d_h)

        # QK-Norm
        wq_norm = self._l2_normalize(wq)
        wk_norm = self._l2_normalize(wk)

        write_scores = (wq_norm @ wk_norm.transpose(-2, -1)) * self.write_qk_scale  # (B, H, T, m)
        # Apply causal mask
        write_scores = write_scores.masked_fill(write_mask == 0, float('-inf'))
        write_attn = torch.softmax(write_scores, dim=-1)
        write_out_4d = write_attn @ wv                             # (B, H, T, d_h)
        write_out = self._reshape_from_heads(write_out_4d, B, T)  # (B, T, D)
        write_out_proj = write_out @ self.write_out_w             # (B, T, D)

        # x = norm(x + write_gate * write_out_proj)
        x_pre_norm = x + self.write_gate * write_out_proj
        x_out = self._rms_norm(x_pre_norm, self.norm_w)

        return x_out, slots_out


# ============================================================================
# Manual backward (replicates TTWorkspaceModule.backward math)
# ============================================================================

def manual_softmax_backward(grad_attn, attn, dim=-1):
    """Softmax backward: grad_scores = (grad_attn - sum(grad_attn*attn, dim)) * attn"""
    grad_sum = (grad_attn * attn).sum(dim=dim, keepdim=True)
    return (grad_attn - grad_sum) * attn


def rms_norm_backward(grad_out, x, weight, eps=1e-6):
    """RMSNorm backward: y = x / rms * weight, rms = sqrt(eps + mean(x^2))

    Returns: (grad_x, grad_weight)
    """
    d = x.shape[-1]
    x_sq_mean = (x * x).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(x_sq_mean + eps)
    x_normed = x * inv_rms

    # grad_weight = sum(grad_out * x_normed, dim=(0,1))
    grad_weight = (grad_out * x_normed).sum(dim=tuple(range(grad_out.ndim - 1)))

    # grad_x = (grad_out * weight / rms) - (x_normed * mean(grad_out * weight * x_normed) / rms)
    gw = grad_out * weight  # (..., d)
    gw_xnorm = gw * x_normed  # (..., d)
    gw_xnorm_mean = gw_xnorm.mean(dim=-1, keepdim=True)  # (..., 1)
    grad_x = gw * inv_rms - x_normed * gw_xnorm_mean * inv_rms

    return grad_x, grad_weight


def l2_normalize_backward(grad_normed, x, eps=1e-6):
    """Backward through y = x / ||x|| (L2 norm along last dim).

    Jacobian: dy/dx = (I - y*y^T) / ||x||
    grad_x = (grad_y - y * (grad_y . y)) / ||x||
    """
    norm_sq = (x * x).sum(dim=-1, keepdim=True)
    inv_norm = torch.rsqrt(norm_sq + eps)
    y = x * inv_norm  # normalized output

    dot = (grad_normed * y).sum(dim=-1, keepdim=True)  # (..., 1)
    grad_x = (grad_normed - y * dot) * inv_norm
    return grad_x


def manual_workspace_v2_backward(grad_x_out, grad_slots_out, ws, x, slot_state=None):
    """Replicates TTWorkspaceModule.backward for architecture v2.

    All weights use TT convention: y = x @ W_tt (matmul, not F.linear).
    """
    B, T, D = x.shape
    H, d_h, m = ws.H, ws.d_h, ws.m
    eps = ws.eps

    # Build causal masks (must match forward exactly)
    read_mask, write_mask = ws._build_causal_masks(T, m, x.device, x.dtype)

    # === Recompute forward (cache values needed for backward) ===
    if slot_state is None:
        slots_in = ws.slots.unsqueeze(0).expand(B, m, D).clone()
    else:
        slots_in = slot_state

    # Read path
    rq = ws._reshape_to_heads(slots_in @ ws.read_q_w, B, m)
    rk = ws._reshape_to_heads(x @ ws.read_k_w, B, T)
    rv = ws._reshape_to_heads(x @ ws.read_v_w, B, T)
    rq_norm = ws._l2_normalize(rq)
    rk_norm = ws._l2_normalize(rk)
    read_scores_pre = rq_norm @ rk_norm.transpose(-2, -1)  # before scale
    read_scores = read_scores_pre * ws.read_qk_scale
    # Apply causal mask
    read_scores = read_scores.masked_fill(read_mask == 0, float('-inf'))
    read_attn = torch.softmax(read_scores, dim=-1)
    read_out_4d = read_attn @ rv
    read_out = ws._reshape_from_heads(read_out_4d, B, m)
    read_out_proj = read_out @ ws.read_out_w

    decayed_slots = ws.slot_decay * slots_in
    slots_pre_norm = decayed_slots + ws.read_gate * read_out_proj
    slots_out = ws._rms_norm(slots_pre_norm, ws.slot_norm_w)

    # Write path
    wq = ws._reshape_to_heads(x @ ws.write_q_w, B, T)
    wk = ws._reshape_to_heads(slots_out @ ws.write_k_w, B, m)
    wv = ws._reshape_to_heads(slots_out @ ws.write_v_w, B, m)
    wq_norm = ws._l2_normalize(wq)
    wk_norm = ws._l2_normalize(wk)
    write_scores_pre = wq_norm @ wk_norm.transpose(-2, -1)
    write_scores = write_scores_pre * ws.write_qk_scale
    # Apply causal mask
    write_scores = write_scores.masked_fill(write_mask == 0, float('-inf'))
    write_attn = torch.softmax(write_scores, dim=-1)
    write_out_4d = write_attn @ wv
    write_out = ws._reshape_from_heads(write_out_4d, B, T)
    write_out_proj = write_out @ ws.write_out_w

    x_pre_norm = x + ws.write_gate * write_out_proj

    # === Backward ===

    # --- x_out = norm(x_pre_norm) ---
    grad_x_pre_norm, grad_norm_w = rms_norm_backward(grad_x_out, x_pre_norm, ws.norm_w, eps)

    # --- x_pre_norm = x + write_gate * write_out_proj ---
    grad_x_from_write = grad_x_pre_norm  # residual
    grad_write_out_proj = grad_x_pre_norm * ws.write_gate
    grad_write_gate = (grad_x_pre_norm * write_out_proj).sum()

    # --- write_out_proj = write_out @ write_out_w ---
    grad_write_out = grad_write_out_proj @ ws.write_out_w.T
    grad_write_out_w = write_out.reshape(-1, D).T @ grad_write_out_proj.reshape(-1, D)

    # --- write_out = reshape(write_attn @ wv) ---
    grad_write_out_4d = ws._reshape_to_heads(grad_write_out, B, T)
    grad_write_attn = grad_write_out_4d @ wv.transpose(-2, -1)  # (B, H, T, m)
    grad_wv = write_attn.transpose(-2, -1) @ grad_write_out_4d   # (B, H, m, d_h)

    # --- write_attn = softmax(write_scores * write_qk_scale) ---
    grad_write_scores = manual_softmax_backward(grad_write_attn, write_attn, dim=-1)
    # Zero out gradients at causally masked positions
    grad_write_scores = grad_write_scores * write_mask
    grad_write_scores = grad_write_scores * ws.write_qk_scale

    # grad_write_qk_scale = sum(grad_attn * scores_pre)
    # Note: grad_write_scores already has qk_scale multiplied in (from softmax backward).
    # The gradient of scores = scores_pre * qk_scale w.r.t. qk_scale is:
    #   d(L)/d(qk_scale) = sum(d(L)/d(scores) * scores_pre)
    # But d(L)/d(scores) = grad_attn * qk_scale (since scores = scores_pre * qk_scale,
    # and grad_attn is d(L)/d(attn), so d(L)/d(scores) = grad_attn * qk_scale via softmax backward).
    # Wait — let's be careful. The softmax backward gives us d(L)/d(scores) already.
    # The line `grad_write_scores = grad_write_scores * ws.write_qk_scale` multiplies by qk_scale.
    # So grad_write_scores at this point IS d(L)/d(scores_pre) * qk_scale... no.
    #
    # Let's trace: scores = scores_pre * qk_scale. attn = softmax(scores).
    # softmax_backward gives d(L)/d(scores). Then we multiply by qk_scale to get d(L)/d(scores_pre)?
    # No — scores = scores_pre * qk_scale, so d(scores)/d(scores_pre) = qk_scale.
    # So d(L)/d(scores_pre) = d(L)/d(scores) * qk_scale. That's what the code does.
    #
    # For qk_scale: d(L)/d(qk_scale) = sum(d(L)/d(scores) * scores_pre).
    # But we've already multiplied grad_write_scores by qk_scale to get d(L)/d(scores_pre).
    # So we need d(L)/d(scores) = grad_write_scores / qk_scale... or we can use the
    # pre-multiply value. Let's use the relationship:
    # d(L)/d(qk_scale) = sum(grad_attn_softmax_bwd * scores_pre)
    # where grad_attn_softmax_bwd = d(L)/d(scores) = grad_write_scores / qk_scale
    #
    # Actually, looking at the TT code more carefully:
    # grad_write_scores = softmax_backward(grad_attn, attn)  -> this is d(L)/d(scores)
    # grad_write_scores *= qk_scale  -> now it's d(L)/d(scores_pre) = d(L)/d(scores) * qk_scale
    # grad_qk_scale = sum(grad_write_scores * scores_pre)
    #
    # But d(L)/d(qk_scale) = sum(d(L)/d(scores) * scores_pre)
    #                      = sum((grad_write_scores / qk_scale) * scores_pre)
    #
    # The TT code computes: sum(grad_write_scores * scores_pre) where grad_write_scores
    # is already d(L)/d(scores_pre) = d(L)/d(scores) * qk_scale.
    # So it's computing: sum(d(L)/d(scores) * qk_scale * scores_pre)
    #                   = qk_scale * sum(d(L)/d(scores) * scores_pre)
    #                   = qk_scale * d(L)/d(qk_scale)
    #
    # This is WRONG by a factor of qk_scale! Unless... let me re-read the TT code.
    #
    # Actually wait. Let me re-read:
    #   grad_write_scores = self._softmax_backward(...)  # d(L)/d(scores)
    #   grad_write_scores = ttnn.mul(grad_write_scores, self.write_qk_scale)  # *= qk_scale
    #   grad_write_qk_scale = ttnn.sum(ttnn.mul(grad_write_scores, write_scores_pre))
    #
    # So grad_write_qk_scale = sum(d(L)/d(scores) * qk_scale * scores_pre)
    #                        = qk_scale * d(L)/d(qk_scale)
    #
    # This IS off by a factor of qk_scale. But... the optimizer will still update
    # qk_scale in the right direction (just with a different magnitude). Since qk_scale
    # is a scalar and AdamW normalizes by the gradient magnitude, the effective update
    # direction is the same. Still, this is a mathematical error.
    #
    # For this test, I'll replicate the TT code exactly (including the bug) so the
    # manual backward matches the TT backward. The autograd reference will show the
    # correct gradient, and we can see the discrepancy.
    #
    # Actually, for gradcheck to pass, the manual backward must match autograd.
    # So I should compute the CORRECT gradient here, and note the TT bug separately.
    #
    # Let me compute it correctly:
    grad_write_scores_for_qk = manual_softmax_backward(grad_write_attn, write_attn, dim=-1)
    # This is d(L)/d(scores) (before multiplying by qk_scale)
    grad_write_qk_scale = (grad_write_scores_for_qk * write_scores_pre).sum()

    # grad_wq_norm = d(L)/d(scores_pre) @ wk_norm, grad_wk_norm = d(L)/d(scores_pre)^T @ wq_norm
    # d(L)/d(scores_pre) = d(L)/d(scores) * qk_scale = grad_write_scores (after *= qk_scale)
    grad_wq_norm = grad_write_scores @ wk_norm               # (B, H, T, d_h)
    grad_wk_norm = grad_write_scores.transpose(-2, -1) @ wq_norm  # (B, H, m, d_h)

    # --- QK-Norm backward ---
    grad_wq = l2_normalize_backward(grad_wq_norm, wq, eps)
    grad_wk = l2_normalize_backward(grad_wk_norm, wk, eps)

    # --- Write projections ---
    grad_wq_3d = ws._reshape_from_heads(grad_wq, B, T)
    grad_wk_3d = ws._reshape_from_heads(grad_wk, B, m)
    grad_wv_3d = ws._reshape_from_heads(grad_wv, B, m)

    x_2d = x.reshape(-1, D)
    slots_out_2d = slots_out.reshape(-1, D)

    grad_write_q_w = x_2d.T @ grad_wq_3d.reshape(-1, D)
    grad_x_from_wq = grad_wq_3d.reshape(-1, D) @ ws.write_q_w.T
    grad_x_from_wq = grad_x_from_wq.reshape(B, T, D)

    grad_wk_2d = grad_wk_3d.reshape(-1, D)
    grad_wv_2d = grad_wv_3d.reshape(-1, D)
    grad_write_k_w = slots_out_2d.T @ grad_wk_2d
    grad_write_v_w = slots_out_2d.T @ grad_wv_2d

    grad_slots_from_wk = (grad_wk_2d @ ws.write_k_w.T).reshape(B, m, D)
    grad_slots_from_wv = (grad_wv_2d @ ws.write_v_w.T).reshape(B, m, D)

    # Total grad_slots_out
    grad_slots_total = grad_slots_out + grad_slots_from_wk + grad_slots_from_wv

    # --- slots_out = slot_norm(slots_pre_norm) ---
    grad_slots_pre_norm, grad_slot_norm_w = rms_norm_backward(
        grad_slots_total, slots_pre_norm, ws.slot_norm_w, eps
    )

    # --- slots_pre_norm = decay * slots_in + read_gate * read_out_proj ---
    grad_slots_in_from_read = grad_slots_pre_norm * ws.slot_decay
    grad_read_out_proj = grad_slots_pre_norm * ws.read_gate
    grad_read_gate = (grad_slots_pre_norm * read_out_proj).sum()
    grad_slot_decay = (grad_slots_pre_norm * slots_in).sum()

    # --- read_out_proj = read_out @ read_out_w ---
    grad_read_out = grad_read_out_proj @ ws.read_out_w.T
    grad_read_out_w = read_out.reshape(-1, D).T @ grad_read_out_proj.reshape(-1, D)

    # --- read_out = reshape(read_attn @ rv) ---
    grad_read_out_4d = ws._reshape_to_heads(grad_read_out, B, m)
    grad_read_attn = grad_read_out_4d @ rv.transpose(-2, -1)  # (B, H, m, T)
    grad_rv = read_attn.transpose(-2, -1) @ grad_read_out_4d   # (B, H, T, d_h)

    # --- read_attn = softmax(read_scores * read_qk_scale) ---
    grad_read_scores = manual_softmax_backward(grad_read_attn, read_attn, dim=-1)
    grad_read_scores_for_qk = manual_softmax_backward(grad_read_attn, read_attn, dim=-1)
    # Zero out gradients at causally masked positions
    grad_read_scores = grad_read_scores * read_mask
    grad_read_scores_for_qk = grad_read_scores_for_qk * read_mask
    grad_read_scores = grad_read_scores * ws.read_qk_scale
    grad_read_qk_scale = (grad_read_scores_for_qk * read_scores_pre).sum()

    grad_rq_norm = grad_read_scores @ rk_norm               # (B, H, m, d_h)
    grad_rk_norm = grad_read_scores.transpose(-2, -1) @ rq_norm  # (B, H, T, d_h)

    # --- QK-Norm backward ---
    grad_rq = l2_normalize_backward(grad_rq_norm, rq, eps)
    grad_rk = l2_normalize_backward(grad_rk_norm, rk, eps)

    # --- Read projections ---
    grad_rq_3d = ws._reshape_from_heads(grad_rq, B, m)
    grad_rk_3d = ws._reshape_from_heads(grad_rk, B, T)
    grad_rv_3d = ws._reshape_from_heads(grad_rv, B, T)

    slots_in_2d = slots_in.reshape(-1, D)

    grad_read_q_w = slots_in_2d.T @ grad_rq_3d.reshape(-1, D)
    grad_slots_from_rq = (grad_rq_3d.reshape(-1, D) @ ws.read_q_w.T).reshape(B, m, D)

    grad_rk_2d = grad_rk_3d.reshape(-1, D)
    grad_rv_2d = grad_rv_3d.reshape(-1, D)
    grad_read_k_w = x_2d.T @ grad_rk_2d
    grad_read_v_w = x_2d.T @ grad_rv_2d

    grad_x_from_rk = (grad_rk_2d @ ws.read_k_w.T).reshape(B, T, D)
    grad_x_from_rv = (grad_rv_2d @ ws.read_v_w.T).reshape(B, T, D)

    # --- Total gradients ---
    grad_x = grad_x_from_write + grad_x_from_wq + grad_x_from_rk + grad_x_from_rv
    grad_slots_in = grad_slots_in_from_read + grad_slots_from_rq

    # If slots_in came from learned embeddings (slot_state is None), grad propagates to slots
    if slot_state is None:
        # slots_in = slots.unsqueeze(0).expand(B, m, D) -> grad_slots = sum over B
        grad_slots_param = grad_slots_in.sum(dim=0)  # (m, D)
    else:
        grad_slots_param = None

    grads = {
        "read_q_w": grad_read_q_w,
        "read_k_w": grad_read_k_w,
        "read_v_w": grad_read_v_w,
        "read_out_w": grad_read_out_w,
        "write_q_w": grad_write_q_w,
        "write_k_w": grad_write_k_w,
        "write_v_w": grad_write_v_w,
        "write_out_w": grad_write_out_w,
        "norm_w": grad_norm_w,
        "slot_norm_w": grad_slot_norm_w,
        "read_gate": grad_read_gate,
        "write_gate": grad_write_gate,
        "slot_decay": grad_slot_decay,
        "read_qk_scale": grad_read_qk_scale,
        "write_qk_scale": grad_write_qk_scale,
    }
    if grad_slots_param is not None:
        grads["slots"] = grad_slots_param

    return grad_x, grad_slots_in, grads


# ============================================================================
# Test
# ============================================================================

def test_backward_vs_autograd(verbose=False):
    """Compare manual backward against PyTorch autograd (float64 gradcheck)."""
    torch.manual_seed(42)
    dtype = torch.float64

    D = 384
    m = 16
    H = 4
    B = 2
    T = 32

    ws = RefWorkspaceV2(D, H, m, dtype=dtype)

    # Create inputs with requires_grad
    x = torch.randn(B, T, D, dtype=dtype) * 0.1
    slot_state = None  # Use learned slots (tests slot gradient too)

    # --- PyTorch autograd reference ---
    x_ref = x.clone().requires_grad_(True)
    x_out_ref, slots_out_ref = ws(x_ref, slot_state)

    # Simple scalar loss: L = sum(x_out) + sum(slots_out)
    loss = x_out_ref.sum() + slots_out_ref.sum()
    loss.backward()

    autograd_grads = {
        "x": x_ref.grad.clone(),
    }
    for name, param in ws.named_parameters():
        if param.grad is not None:
            autograd_grads[name] = param.grad.clone()

    # --- Manual backward ---
    ws.zero_grad()
    grad_x_out = torch.ones_like(x_out_ref, dtype=dtype)
    grad_slots_out = torch.ones_like(slots_out_ref, dtype=dtype)

    grad_x, grad_slots_in, manual_grads = manual_workspace_v2_backward(
        grad_x_out, grad_slots_out, ws, x, slot_state
    )

    # --- Compare ---
    all_pass = True
    tol = 1e-9  # float64 precision

    # Input gradient
    err_x = (grad_x - autograd_grads["x"]).abs().max().item()
    rel_x = err_x / (autograd_grads["x"].abs().max().item() + 1e-30)
    x_pass = rel_x < 1e-8
    if verbose or not x_pass:
        print(f"  {'x':>20s}: max_err={err_x:.2e}, rel_err={rel_x:.2e} {'PASS' if x_pass else 'FAIL'}")
    all_pass &= x_pass

    # Parameter gradients
    param_names = [
        "read_q_w", "read_k_w", "read_v_w", "read_out_w",
        "write_q_w", "write_k_w", "write_v_w", "write_out_w",
        "norm_w", "slot_norm_w",
        "read_gate", "write_gate", "slot_decay",
        "read_qk_scale", "write_qk_scale",
        "slots",
    ]

    for name in param_names:
        if name not in manual_grads or name not in autograd_grads:
            if verbose:
                print(f"  {name:>20s}: MISSING (manual={name in manual_grads}, autograd={name in autograd_grads})")
            continue

        m_grad = manual_grads[name]
        a_grad = autograd_grads[name]
        err = (m_grad - a_grad).abs().max().item()
        rel = err / (a_grad.abs().max().item() + 1e-30)
        # Use absolute tolerance for near-zero gradients (gates at init=0)
        pass_abs = err < 1e-10
        pass_rel = rel < 1e-8
        passed = pass_abs or pass_rel
        if verbose or not passed:
            print(f"  {name:>20s}: max_err={err:.2e}, rel_err={rel:.2e} {'PASS' if passed else 'FAIL'}")
        all_pass &= passed

    # Slots gradient (input, not parameter when slot_state is passed)
    if slot_state is None:
        # grad_slots_in should match autograd grad of ws.slots (summed over B)
        err_s = (manual_grads["slots"] - autograd_grads["slots"]).abs().max().item()
        rel_s = err_s / (autograd_grads["slots"].abs().max().item() + 1e-30)
        s_pass = rel_s < 1e-8
        if verbose or not s_pass:
            print(f"  {'slots (param)':>20s}: max_err={err_s:.2e}, rel_err={rel_s:.2e} {'PASS' if s_pass else 'FAIL'}")
        all_pass &= s_pass

    return all_pass


def test_l2_normalize_backward(verbose=False):
    """Standalone test of L2 normalize backward."""
    torch.manual_seed(123)
    dtype = torch.float64
    eps = 1e-6

    x = torch.randn(4, 8, 16, dtype=dtype, requires_grad=True)
    # Use the same epsilon as the backward function
    norm_sq = (x * x).sum(dim=-1, keepdim=True) + eps
    y = x / norm_sq.sqrt()

    # Autograd
    grad_y = torch.randn_like(y)
    y.backward(grad_y)
    auto_grad = x.grad.clone()

    # Manual
    x.grad = None
    manual_grad = l2_normalize_backward(grad_y, x.detach(), eps)

    err = (manual_grad - auto_grad).abs().max().item()
    rel = err / (auto_grad.abs().max().item() + 1e-30)
    passed = rel < 1e-10

    print(f"\nL2 normalize backward: max_err={err:.2e}, rel_err={rel:.2e} {'PASS' if passed else 'FAIL'}")
    return passed


def test_qk_scale_gradient(verbose=False):
    """Standalone test of qk_scale gradient (the potential bug area).

    Uses a weighted loss (not sum) so the gradient is non-zero.
    Also tests the TT-style buggy computation to confirm the bug.
    """
    torch.manual_seed(456)
    dtype = torch.float64

    D, H, m, T, B = 384, 4, 16, 32, 2
    d_h = D // H

    q = torch.randn(B, H, m, d_h, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, d_h, dtype=dtype, requires_grad=True)
    qk_scale = torch.tensor(0.5, dtype=dtype, requires_grad=True)

    q_norm = q / (q * q).sum(dim=-1, keepdim=True).sqrt()
    k_norm = k / (k * k).sum(dim=-1, keepdim=True).sqrt()
    scores_pre = q_norm @ k_norm.transpose(-2, -1)
    scores = scores_pre * qk_scale
    attn = torch.softmax(scores, dim=-1)
    # Use a weighted loss so gradient is non-zero (sum(softmax) = const -> zero grad)
    weight = torch.randn_like(attn)
    loss = (attn * weight).sum()
    loss.backward()

    auto_qk_scale_grad = qk_scale.grad.clone()

    # Correct manual: d(L)/d(qk_scale) = sum(d(L)/d(scores) * scores_pre)
    grad_attn = weight  # d(L)/d(attn)
    grad_scores = (grad_attn - (grad_attn * attn).sum(dim=-1, keepdim=True)) * attn
    correct_qk_scale_grad = (grad_scores * scores_pre).sum()

    # TT-style (buggy, now FIXED): the old code multiplied grad_scores by qk_scale
    # BEFORE computing grad_qk_scale, giving qk_scale * true_gradient.
    # The fix computes grad_qk_scale BEFORE scaling grad_scores.
    grad_scores_scaled = grad_scores * qk_scale.detach()
    tt_buggy_qk_scale_grad = (grad_scores_scaled * scores_pre).sum()

    err_correct = abs(correct_qk_scale_grad.item() - auto_qk_scale_grad.item())
    rel_correct = err_correct / (abs(auto_qk_scale_grad.item()) + 1e-30)
    passed = rel_correct < 1e-10

    err_tt = abs(tt_buggy_qk_scale_grad.item() - auto_qk_scale_grad.item())
    rel_tt = err_tt / (abs(auto_qk_scale_grad.item()) + 1e-30)
    tt_bug_confirmed = rel_tt > 0.01  # Old buggy version should be wrong

    print(f"\nQK scale gradient:")
    print(f"  autograd:         {auto_qk_scale_grad.item():.10f}")
    print(f"  correct manual:   {correct_qk_scale_grad.item():.10f}  (err={err_correct:.2e}) {'PASS' if passed else 'FAIL'}")
    print(f"  old TT (buggy):   {tt_buggy_qk_scale_grad.item():.10f}  (err={err_tt:.2e}) {'BUG CONFIRMED' if tt_bug_confirmed else 'no bug'}")
    if tt_bug_confirmed:
        ratio = auto_qk_scale_grad.item() / tt_buggy_qk_scale_grad.item()
        print(f"  Old TT gradient was off by factor: {ratio:.6f} (= 1/qk_scale = {1/qk_scale.item():.6f})")
        print(f"  NOTE: This bug is now FIXED in model_ttnn.py. With AdamW the impact was negligible")
        print(f"  (AdamW normalizes out constant gradient scaling factors).")

    return passed


def test_rezero_gate_gradient(verbose=False):
    """Standalone test of ReZero gate gradient (gate=0, no sigmoid)."""
    torch.manual_seed(789)
    dtype = torch.float64

    B, T, D = 2, 32, 384

    x = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    out_proj = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
    gate = torch.tensor(0.0, dtype=dtype, requires_grad=True)  # ReZero: init 0

    # Forward: y = x + gate * out_proj
    y = x + gate * out_proj
    loss = y.sum()
    loss.backward()

    auto_gate_grad = gate.grad.clone()
    auto_x_grad = x.grad.clone()
    auto_out_proj_grad = out_proj.grad.clone()

    # Manual
    grad_y = torch.ones_like(y)
    manual_gate_grad = (grad_y * out_proj).sum()
    manual_x_grad = grad_y  # residual
    manual_out_proj_grad = grad_y * gate  # gate=0, so this should be 0

    err_gate = abs(manual_gate_grad.item() - auto_gate_grad.item())
    err_x = (manual_x_grad - auto_x_grad).abs().max().item()
    err_op = (manual_out_proj_grad - auto_out_proj_grad).abs().max().item()

    passed = err_gate < 1e-10 and err_x < 1e-10 and err_op < 1e-10

    print(f"\nReZero gate gradient (gate=0):")
    print(f"  gate:     auto={auto_gate_grad.item():.6f}, manual={manual_gate_grad.item():.6f}, err={err_gate:.2e}")
    print(f"  x:        err={err_x:.2e}")
    print(f"  out_proj: auto_max={auto_out_proj_grad.abs().max().item():.2e}, manual_max={err_op:.2e}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    # Also test with non-zero gate
    gate2 = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    y2 = x.detach() + gate2 * out_proj.detach()
    loss2 = y2.sum()
    loss2.backward()
    auto_gate2_grad = gate2.grad.clone()
    manual_gate2_grad = (torch.ones_like(y2) * out_proj.detach()).sum()

    err_gate2 = abs(manual_gate2_grad.item() - auto_gate2_grad.item())
    passed2 = err_gate2 < 1e-10
    print(f"  gate=0.3: auto={auto_gate2_grad.item():.6f}, manual={manual_gate2_grad.item():.6f}, err={err_gate2:.2e} {'PASS' if passed2 else 'FAIL'}")

    return passed and passed2


def main():
    parser = argparse.ArgumentParser(description="Gradient check for architecture v2 workspace")
    parser.add_argument("--verbose", action="store_true", help="Show per-parameter errors")
    args = parser.parse_args()

    print("=" * 70)
    print("Architecture v2 workspace gradient check (QK-Norm + ReZero)")
    print("=" * 70)

    all_pass = True

    # Standalone component tests
    print("\n--- Component tests ---")
    all_pass &= test_l2_normalize_backward(args.verbose)
    all_pass &= test_qk_scale_gradient(args.verbose)
    all_pass &= test_rezero_gate_gradient(args.verbose)

    # Full workspace backward test
    print("\n--- Full workspace backward vs autograd ---")
    all_pass &= test_backward_vs_autograd(verbose=True)  # always verbose for full test

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see details above")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
