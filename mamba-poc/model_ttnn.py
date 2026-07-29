"""
Tenstorrent native (tt-nn) implementation of the Mamba + Workspace model.

This is a rewrite of model.py using tt-nn operations directly, bypassing
PyTorch/XLA entirely. The forward pass uses tt-nn ops on the Blackhole device.
The backward pass is computed manually using tt-nn _bw ops and matmul.

Key differences from the PyTorch version:
- No torch.compile, no XLA, no graph tracing — direct device execution
- Manual backward pass (no autograd)
- Weights stored as tt-nn tensors on device
- Optimizer: ttnn.moreh_adamw (TT native AdamW)
- Data: PyTorch data pipeline -> ttnn.from_torch for each batch
"""

import os
# Suppress Metal C++ warnings (e.g. ROW MAJOR tile extraction in ttnn.embedding)
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import math
import torch
import ttnn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Config (shared with model.py)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 14
    vocab_size: int = 128
    d_state: int = 64
    d_conv: int = 4
    expand: int = 4
    n_heads: int = 4
    use_attention: bool = False
    attention_positions: List[int] = field(default_factory=lambda: [5, 10])
    use_workspace: bool = False
    n_workspace_slots: int = 16
    recurrent_core: bool = False
    core_start: int = 6
    core_end: int = 10
    k_train_max: int = 6
    k_inference: int = 6
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True

    @property
    def d_inner(self):
        return self.d_model * self.expand

    @property
    def d_head(self):
        return self.d_inner // self.n_heads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_device(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    """Convert a PyTorch tensor to a tt-nn device tensor."""
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def to_host(t: "ttnn.Tensor") -> torch.Tensor:
    """Convert a tt-nn tensor back to a PyTorch tensor on host."""
    return ttnn.to_torch(t)


def tt_zeros(shape, device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    """Create a zero tensor on device."""
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
        dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def tt_ones(shape, device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    """Create a ones tensor on device."""
    return ttnn.from_torch(
        torch.ones(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
        dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def tt_scalar(val: float, shape, device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    """Create a tensor filled with a scalar value."""
    t = torch.full(shape, val, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


# ---------------------------------------------------------------------------
# RMSNorm (tt-nn native)
# ---------------------------------------------------------------------------

class TTRMSNorm:
    """RMSNorm using ttnn.rms_norm.

    Our PyTorch version uses ZCRMSNorm: x * (1 + gamma) / RMS(x), gamma init 0.
    ttnn.rms_norm computes: x / sqrt(eps + mean(x^2)) * weight.
    To match (1 + gamma) scaling, we store weight = 1 + gamma.
    """

    def __init__(self, d: int, device, eps=1e-6):
        self.d = d
        self.eps = eps
        self.device = device
        # weight = 1 + gamma, gamma init 0 -> weight init 1
        self.weight = ttnn.from_torch(
            torch.ones(d, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)

    def backward(self, grad_out: "ttnn.Tensor", x: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """RMS norm backward computed on device.

        Forward: y = x / rms * weight, where rms = sqrt(eps + mean(x^2))
        Backward:
          grad_x = (grad_out * weight / rms) - (x_normed * mean(grad_out * weight * x_normed) / rms)
          grad_weight = sum(grad_out * x_normed, dim=(0,1))

        Args:
            grad_out: (B, T, d) gradient w.r.t. output
            x: (B, T, d) pre-norm input (cached from forward)

        Returns:
            grad_x: (B, T, d) gradient w.r.t. input
            grad_weight: (d,) gradient w.r.t. weight
        """
        device = self.device
        d = self.d
        eps = self.eps

        # Compute rms = sqrt(eps + mean(x^2)) along last dim
        # ttnn.mean drops the last dim, so we reshape back to (B, T, 1)
        x_sq = ttnn.mul(x, x)  # (B, T, d)
        x_sq_mean = ttnn.mean(x_sq, dim=-1)  # (B, T)
        rms_2d = ttnn.sqrt(ttnn.add(x_sq_mean, eps))  # (B, T)
        rms = ttnn.reshape(rms_2d, [x_sq_mean.shape[0], x_sq_mean.shape[1], 1])  # (B, T, 1)

        # Expand rms to (B, T, d) for element-wise ops (ttnn doesn't support (B,T,1) broadcast)
        rms_exp = ttnn.expand(rms, x.shape)  # (B, T, d)
        inv_rms = ttnn.rsqrt(ttnn.add(x_sq_mean, eps))  # (B, T)
        inv_rms_exp = ttnn.expand(ttnn.reshape(inv_rms, [x_sq_mean.shape[0], x_sq_mean.shape[1], 1]), x.shape)

        # x_normed = x / rms = x * inv_rms
        x_normed = ttnn.mul(x, inv_rms_exp)  # (B, T, d)

        # grad_out * weight (broadcast weight across B, T — (1,1,d) works)
        w = ttnn.reshape(self.weight, [1, 1, d])  # (1, 1, d)
        grad_out_w = ttnn.mul(grad_out, w)  # (B, T, d)

        # grad_out_w / rms = grad_out_w * inv_rms
        grad_out_w_rms = ttnn.mul(grad_out_w, inv_rms_exp)  # (B, T, d)

        # mean(grad_out_w * x_normed) along last dim
        grad_out_w_xnorm = ttnn.mul(grad_out_w, x_normed)  # (B, T, d)
        grad_out_w_xnorm_mean = ttnn.mean(grad_out_w_xnorm, dim=-1)  # (B, T)
        grad_out_w_xnorm_mean_3d = ttnn.reshape(
            grad_out_w_xnorm_mean,
            [grad_out_w_xnorm_mean.shape[0], grad_out_w_xnorm_mean.shape[1], 1])  # (B, T, 1)
        grad_out_w_xnorm_mean_exp = ttnn.expand(grad_out_w_xnorm_mean_3d, x.shape)  # (B, T, d)

        # x_normed * (grad_out_w_xnorm_mean / rms) = x_normed * grad_out_w_xnorm_mean * inv_rms
        correction = ttnn.mul(x_normed, ttnn.mul(grad_out_w_xnorm_mean_exp, inv_rms_exp))  # (B, T, d)

        # grad_x = grad_out_w_rms - correction
        grad_x = ttnn.sub(grad_out_w_rms, correction)

        # grad_weight = sum(grad_out * x_normed, dim=(0, 1))
        grad_weight_full = ttnn.mul(grad_out, x_normed)  # (B, T, d)
        grad_weight = ttnn.sum(grad_weight_full, dim=0)  # (T, d)
        grad_weight = ttnn.sum(grad_weight, dim=0)  # (d,)

        return grad_x, grad_weight

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        return {"weight": self.weight}

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        self.weight = params["weight"]


# ---------------------------------------------------------------------------
# Mamba2 Layer (tt-nn native)
# ---------------------------------------------------------------------------

class TTMamba2Layer:
    """Mamba2 layer using tt-nn operations.

    Forward pass mirrors model.py Mamba2Layer.forward() but uses tt-nn ops.
    Backward pass is manual — computes gradients using _bw ops and matmul.
    """

    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_conv = config.d_conv

        # Initialize weights as PyTorch tensors, then move to device
        # NOTE: ttnn.linear(x, W) computes x @ W, so W must be (in, out) — transposed from PyTorch
        # in_proj: (d_model -> 2*d_inner), weight shape (d_model, 2*d_inner)
        w = torch.randn(self.d_model, 2 * self.d_inner, dtype=torch.bfloat16) * 0.02
        self.in_proj_weight = to_device(w, device)

        # conv1d: (d_inner, 1, d_conv) — depthwise conv (computed on host)
        conv_w = torch.randn(self.d_inner, 1, self.d_conv, dtype=torch.bfloat16) * 0.02
        self.conv1d_weight = to_device(conv_w, device)
        # conv1d bias
        conv_b = torch.zeros(self.d_inner, dtype=torch.bfloat16)
        self.conv1d_bias = to_device(conv_b, device)

        # dt_proj: (d_inner -> n_heads) with bias, weight shape (d_inner, n_heads)
        dt_w = torch.randn(self.d_inner, self.n_heads, dtype=torch.bfloat16) * 0.02
        self.dt_proj_weight = to_device(dt_w, device)
        dt_b = torch.zeros(self.n_heads, dtype=torch.bfloat16)
        self.dt_proj_bias = to_device(dt_b, device)

        # B_proj, C_proj: (d_inner -> d_state) no bias, weight shape (d_inner, d_state)
        self.B_proj_weight = to_device(
            torch.randn(self.d_inner, self.d_state, dtype=torch.bfloat16) * 0.02, device)
        self.C_proj_weight = to_device(
            torch.randn(self.d_inner, self.d_state, dtype=torch.bfloat16) * 0.02, device)

        # out_proj: (d_inner -> d_model) no bias, weight shape (d_inner, d_model)
        self.out_proj_weight = to_device(
            torch.randn(self.d_inner, self.d_model, dtype=torch.bfloat16) * 0.02, device)

        # A_log, D: (n_heads,)
        self.A_log = to_device(
            (torch.randn(self.n_heads, dtype=torch.float32) - 0.5), device, dtype=ttnn.float32)
        self.D = to_device(
            torch.ones(self.n_heads, dtype=torch.float32), device, dtype=ttnn.float32)

        # RMSNorm
        self.norm = TTRMSNorm(self.d_inner, device)

        # Cached intermediate values for backward
        self._cache = {}

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        """
        x: (B, T, d_model) on device
        returns: (B, T, d_model) on device
        """
        B, T, D = x.shape  # these may be ttnn Shape objects
        B, T, D = int(B), int(T), int(D)
        H = self.n_heads
        device = self.device

        # in_proj: (B, T, d_model) -> (B, T, 2*d_inner)
        xz = ttnn.linear(x, self.in_proj_weight)
        # Split into x_branch and z: each (B, T, d_inner)
        x_branch, z = ttnn.split(xz, self.d_inner, dim=-1)

        # conv1d: need (B, d_inner, T) for conv, input is (B, T, d_inner)
        x_conv = ttnn.transpose(x_branch, 1, 2)  # (B, d_inner, T)
        # ttnn.conv1d expects [N, H, W, C] format — reshape
        # For now, use a simple approach: implement conv1d as a matmul with a toeplitz matrix
        # This is simpler and avoids conv1d format issues
        x_conv = self._conv1d_simple(x_conv, T)  # (B, d_inner, T)
        x_conv = ttnn.transpose(x_conv, 1, 2)  # (B, T, d_inner)
        x_conv = ttnn.silu(x_conv)

        # Projections for SSD
        dt_pre = ttnn.linear(x_conv, self.dt_proj_weight, bias=self.dt_proj_bias)
        dt = ttnn.softplus(dt_pre)  # (B, T, n_heads)
        B_mat = ttnn.linear(x_conv, self.B_proj_weight)  # (B, T, d_state)
        C_mat = ttnn.linear(x_conv, self.C_proj_weight)  # (B, T, d_state)

        # V = x_conv reshaped: (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        # ttnn doesn't have a direct view, so we work with what we have
        # For the SSD matmul, we need V_h: (B, n_heads, T, d_head)
        V_h = self._reshape_v(x_conv, B, T, H, self.d_head)  # (B, n_heads, T, d_head)

        # Compute decay: A = exp(A_log), decay = exp(-dt * A)
        A = ttnn.exp(self.A_log)  # (n_heads,)
        # decay = exp(-dt * A): need broadcasting (B, T, H) * (H,) -> (B, T, H)
        # ttnn doesn't support arbitrary broadcasting, so we reshape A
        A_expanded = self._expand_param(A, B, T, H)  # (B, T, H) or (1, 1, H)
        decay = ttnn.exp(ttnn.neg(ttnn.mul(dt, A_expanded)))  # (B, T, n_heads)

        # CB = C @ B^T: (B, T, T)
        B_mat_T = ttnn.transpose(B_mat, 1, 2)  # (B, d_state, T)
        CB = ttnn.matmul(C_mat, B_mat_T)  # (B, T, T)

        # Decay matrix L: (B, H, T, T)
        L = self._compute_decay_matrix(decay, B, T, H)  # (B, H, T, T)

        # scores = CB.unsqueeze(1) * L: (B, H, T, T)
        CB_expanded = self._expand_cb(CB, B, H, T)  # (B, H, T, T)
        scores = ttnn.mul(CB_expanded, L)  # (B, H, T, T)

        # Y = scores @ V_h: (B, H, T, d_head)
        # Upcast to fp32 for accumulation stability
        scores_f32 = ttnn.typecast(scores, ttnn.float32)
        V_h_f32 = ttnn.typecast(V_h, ttnn.float32)
        Y = ttnn.matmul(scores_f32, V_h_f32)  # (B, H, T, d_head)
        Y = ttnn.typecast(Y, ttnn.bfloat16)

        # Skip connection: Y = Y + D * V_h
        D_expanded = self._expand_d(self.D, B, H, T, self.d_head)  # (1, H, 1, 1)
        Y = ttnn.add(Y, ttnn.mul(D_expanded, V_h))

        # Reshape back: (B, T, d_inner)
        Y = self._reshape_y(Y, B, T, H, self.d_inner)  # (B, T, d_inner)

        # Cache Y_ssd (pre-gate SSD output) for grad_z computation in backward
        Y_ssd = Y

        # Gate with z
        z_silu = ttnn.silu(z)
        Y = ttnn.mul(Y, z_silu)

        # Cache pre-norm Y for backward (RMS norm backward needs pre-norm input)
        Y_pre_norm = Y

        # Norm and output projection
        Y = self.norm.forward(Y)
        out = ttnn.linear(Y, self.out_proj_weight)  # (B, T, d_model)

        # Cache for backward
        self._cache = {
            "x": x, "x_branch": x_branch, "z": z, "x_conv": x_conv,
            "dt_pre": dt_pre, "dt": dt, "B_mat": B_mat, "C_mat": C_mat, "V_h": V_h,
            "A": A, "A_expanded": A_expanded,
            "decay": decay, "CB": CB, "CB_expanded": CB_expanded,
            "L": L, "scores": scores, "scores_f32": scores_f32, "V_h_f32": V_h_f32,
            "Y_ssd": Y_ssd,
            "Y_pre_norm": Y_pre_norm, "Y": Y, "z_silu": z_silu,
        }

        return out

    def _conv1d_simple(self, x: "ttnn.Tensor", T: int) -> "ttnn.Tensor":
        """Depthwise conv1d computed on device.

        x: (B, d_inner, T) — input (already transposed)
        weight: (d_inner, 1, d_conv) — depthwise conv weights
        Returns: (B, d_inner, T) — output (causal, same length)

        Implements causal conv as: out[t] = sum_{i=0}^{k-1} w[:,i] * x_padded[t+i]
        where x_padded has k-1 zeros at the start.
        Done as k multiply-adds with sliced input, all on device.
        """
        B, d_inner, _ = x.shape
        B, d_inner = int(B), int(d_inner)
        k = self.d_conv
        device = self.device

        # Pad input on device: (B, d_inner, T) -> (B, d_inner, T + k - 1) with k-1 zeros at start
        # ttnn.pad doesn't support non-zero front padding for tiled tensors, so use concat
        zeros_prefix = ttnn.zeros((B, d_inner, k - 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        x_padded = ttnn.concat([zeros_prefix, x], dim=-1)  # (B, d_inner, T + k - 1)

        # Get conv weights as (d_inner, k) on device
        # conv1d_weight is (d_inner, 1, k) — squeeze the middle dim
        conv_w = ttnn.squeeze(self.conv1d_weight, dim=1)  # (d_inner, k)

        # Accumulate: out = sum_i w[:,i] * x_padded[:, :, i:i+T]
        out = None
        for i in range(k):
            # Slice: x_padded[:, :, i : i+T]  -> (B, d_inner, T)
            x_slice = ttnn.slice(x_padded, [0, 0, i], [B, d_inner, i + T])
            # Weight for this kernel position: (d_inner,) -> (1, d_inner, 1) for broadcasting
            w_i = ttnn.slice(conv_w, [0, i], [d_inner, i + 1])  # (d_inner, 1)
            w_i = ttnn.reshape(w_i, [1, d_inner, 1])  # (1, d_inner, 1)
            # Element-wise multiply with broadcast
            term = ttnn.mul(x_slice, w_i)  # (B, d_inner, T)
            if out is None:
                out = term
            else:
                out = ttnn.add(out, term)

        # Add bias: (d_inner,) -> (1, d_inner, 1)
        conv_b = ttnn.reshape(self.conv1d_bias, [1, d_inner, 1])
        out = ttnn.add(out, conv_b)

        return out

    def _reshape_v(self, x_conv: "ttnn.Tensor", B, T, H, d_head) -> "ttnn.Tensor":
        """Reshape (B, T, d_inner) -> (B, H, T, d_head) on device"""
        # (B, T, d_inner) -> (B, T, H, d_head) -> (B, H, T, d_head)
        V_4d = ttnn.reshape(x_conv, [B, T, H, d_head])
        V_h = ttnn.permute(V_4d, [0, 2, 1, 3])
        return V_h

    def _reshape_y(self, Y: "ttnn.Tensor", B, T, H, d_inner) -> "ttnn.Tensor":
        """Reshape (B, H, T, d_head) -> (B, T, d_inner) on device"""
        # (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, d_inner)
        Y_perm = ttnn.permute(Y, [0, 2, 1, 3])
        Y_out = ttnn.reshape(Y_perm, [B, T, d_inner])
        return Y_out

    def _expand_param(self, param: "ttnn.Tensor", B, T, H) -> "ttnn.Tensor":
        """Expand (H,) param to (1, 1, H) for broadcasting with (B, T, H)"""
        return ttnn.reshape(param, [1, 1, H])

    def _expand_cb(self, CB: "ttnn.Tensor", B, H, T) -> "ttnn.Tensor":
        """Expand (B, T, T) -> (B, H, T, T) on device"""
        # Reshape to (B, 1, T, T) and expand to (B, H, T, T)
        CB_4d = ttnn.reshape(CB, [B, 1, T, T])
        return ttnn.expand(CB_4d, [B, H, T, T])

    def _expand_d(self, D: "ttnn.Tensor", B, H, T, d_head) -> "ttnn.Tensor":
        """Expand (H,) -> (1, H, 1, 1) for broadcasting with (B, H, T, d_head)"""
        return ttnn.reshape(D, [1, H, 1, 1])

    def _compute_decay_matrix(self, decay: "ttnn.Tensor", B, T, H) -> "ttnn.Tensor":
        """Compute the decay matrix L[b, h, t, s] = prod_{i=s+1}^{t} decay[i, h] on device.

        Uses the segment_sum approach (same as HuggingFace Mamba2):
        1. Take log of decay
        2. Expand to (B, H, T, T) and mask lower triangular
        3. Cumsum along t dimension
        4. Mask to causal (lower triangular incl diag)
        5. Exp to get L

        decay: (B, T, H) on device
        Returns: (B, H, T, T) on device
        """
        device = self.device

        # decay: (B, T, H) -> (B, H, T) -> (B, H, T, 1) -> (B, H, T, T)
        decay_perm = ttnn.permute(decay, [0, 2, 1])  # (B, H, T)
        log_decay = ttnn.log(ttnn.add(decay, 1e-8))  # (B, T, H) — clamp to avoid log(0)
        log_decay_perm = ttnn.permute(log_decay, [0, 2, 1])  # (B, H, T)

        # Expand to (B, H, T, T): each row t has log_decay[t] repeated across s
        log_decay_4d = ttnn.reshape(log_decay_perm, [B, H, T, 1])
        log_decay_exp = ttnn.expand(log_decay_4d, [B, H, T, T])  # (B, H, T, T)

        # Mask: keep strictly lower triangular (s < t), zero rest
        # Create mask on device: tril with diagonal=-1
        ones_tt = ttnn.from_torch(torch.ones(T, T, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mask_low = ttnn.tril(ones_tt, diagonal=-1)  # (T, T) — 1 for s < t, 0 otherwise
        zeros_4d = ttnn.zeros((B, H, T, T), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        log_decay_masked = ttnn.where(mask_low, log_decay_exp, zeros_4d)

        # Cumsum along dim=-2 (t dimension)
        log_L = ttnn.cumsum(log_decay_masked, dim=-2)  # (B, H, T, T)

        # Mask: keep lower triangular incl diag (s <= t), set rest to large negative
        mask_causal = ttnn.tril(ones_tt, diagonal=0)  # (T, T) — 1 for s <= t
        neg_inf_4d = ttnn.full((B, H, T, T), -1e4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        log_L = ttnn.where(mask_causal, log_L, neg_inf_4d)

        # Exp to get L
        L = ttnn.exp(log_L)  # (B, H, T, T)

        return L

    def backward(self, grad_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", Dict[str, "ttnn.Tensor"]]:
        """Manual backward pass for Mamba2Layer — fully on device.

        grad_out: (B, T, d_model) — gradient w.r.t. output
        Returns: (grad_input, grads_dict)

        All operations including the SSD backward are computed on device using
        tt-nn ops. No host-side computation or transfers in the backward pass.
        """
        c = self._cache
        B, T, D = int(c["x"].shape[0]), int(c["x"].shape[1]), int(c["x"].shape[2])
        H, d_head, d_inner = self.n_heads, self.d_head, self.d_inner
        d_state = self.d_state
        device = self.device

        # --- Backward step 14: out = linear(Y_post_norm, out_proj_w) ---
        grad_Y = ttnn.linear(grad_out, ttnn.transpose(self.out_proj_weight, 0, 1))
        Y_post_2d = ttnn.reshape(c["Y"], [B * T, d_inner])  # post-norm Y
        grad_out_2d = ttnn.reshape(grad_out, [B * T, self.d_model])
        grad_out_proj_weight = ttnn.matmul(ttnn.transpose(Y_post_2d, 0, 1), grad_out_2d)

        # --- Backward step 13: Y_post = rms_norm(Y_pre) — on device ---
        grad_Y_pre, grad_norm_weight = self.norm.backward(grad_Y, c["Y_pre_norm"])

        # --- Backward step 12: Y_pre = Y_ssd * silu(z) — on device ---
        z = c["z"]
        z_silu = c["z_silu"]  # cached from forward
        grad_Y_ssd = ttnn.mul(grad_Y_pre, z_silu)  # (B, T, d_inner)

        # --- Backward steps 5-11: SSD backward on device (using cached intermediates) ---
        x_conv = c["x_conv"]
        dt_pre = c["dt_pre"]
        dt = c["dt"]
        B_mat = c["B_mat"]
        C_mat = c["C_mat"]
        V_h = c["V_h"]
        A = c["A"]
        A_expanded = c["A_expanded"]
        decay = c["decay"]
        CB = c["CB"]
        CB_expanded = c["CB_expanded"]
        L = c["L"]
        scores = c["scores"]

        # Reshape grad_Y_ssd to 4D: (B, T, d_inner) -> (B, H, T, d_head)
        grad_Y_ssd_4d = self._reshape_v(grad_Y_ssd, B, T, H, d_head)  # (B, H, T, d_head)

        # --- Step 12 backward: Y_ssd_4d = scores @ V_h + D * V_h ---
        # grad_scores = grad_Y_ssd_4d @ V_h^T  — (B, H, T, T)
        V_h_T = ttnn.transpose(V_h, -2, -1)  # (B, H, d_head, T)
        grad_scores = ttnn.matmul(grad_Y_ssd_4d, V_h_T)  # (B, H, T, T)

        # grad_V_h_from_matmul = scores^T @ grad_Y_ssd_4d — (B, H, T, d_head)
        scores_T = ttnn.transpose(scores, -2, -1)  # (B, H, T, T)
        grad_V_h_from_matmul = ttnn.matmul(scores_T, grad_Y_ssd_4d)  # (B, H, T, d_head)

        # grad_V_h_from_skip = D * grad_Y_ssd_4d
        D_expanded = self._expand_d(self.D, B, H, T, d_head)  # (1, H, 1, 1)
        grad_V_h_from_skip = ttnn.mul(D_expanded, grad_Y_ssd_4d)  # (B, H, T, d_head)

        # grad_V_h = grad_V_h_from_matmul + grad_V_h_from_skip
        grad_V_h = ttnn.add(grad_V_h_from_matmul, grad_V_h_from_skip)  # (B, H, T, d_head)

        # grad_D = sum over (B, H, T, d_head) of grad_Y_ssd_4d * V_h -> (H,)
        grad_D_full = ttnn.mul(grad_Y_ssd_4d, V_h)  # (B, H, T, d_head)
        grad_D = ttnn.sum(grad_D_full, dim=0)  # (H, T, d_head)
        grad_D = ttnn.sum(grad_D, dim=1)  # (H, d_head)
        grad_D = ttnn.sum(grad_D, dim=-1)  # (H,)

        # --- Step 10 backward: scores = CB_expanded * L ---
        # grad_CB_expanded = sum over H of grad_scores * L -> (B, T, T)
        grad_CB_expanded = ttnn.mul(grad_scores, L)  # (B, H, T, T)
        grad_CB = ttnn.sum(grad_CB_expanded, dim=1)  # (B, T, T)

        # grad_L = grad_scores * CB_expanded -> (B, H, T, T)
        grad_L = ttnn.mul(grad_scores, CB_expanded)  # (B, H, T, T)

        # --- Step 9 backward: L = exp(log_L) with cumsum and masking ---
        # Recompute log_L (needed for cumsum backward — not cached since it's only used here)
        log_decay = ttnn.log(ttnn.add(decay, 1e-8))  # (B, T, H)
        log_decay_perm = ttnn.permute(log_decay, [0, 2, 1])  # (B, H, T)
        log_decay_4d = ttnn.reshape(log_decay_perm, [B, H, T, 1])
        log_decay_exp = ttnn.expand(log_decay_4d, [B, H, T, T])  # (B, H, T, T)

        ones_tt = ttnn.from_torch(torch.ones(T, T, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mask_low = ttnn.tril(ones_tt, diagonal=-1)  # s < t
        zeros_4d = ttnn.zeros((B, H, T, T), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        log_decay_masked = ttnn.where(mask_low, log_decay_exp, zeros_4d)

        log_L = ttnn.cumsum(log_decay_masked, dim=-2)  # (B, H, T, T)

        mask_causal = ttnn.tril(ones_tt, diagonal=0)  # s <= t

        # Backward through L = exp(log_L) where log_L is masked to causal
        # grad_log_L = grad_L * L (chain rule through exp)
        # But only for s <= t (causal mask); for s > t, grad is 0
        grad_log_L = ttnn.mul(grad_L, L)  # (B, H, T, T)
        grad_log_L = ttnn.where(mask_causal, grad_log_L, zeros_4d)  # zero out s > t

        # Backward through cumsum: reverse cumsum along dim=-2
        grad_log_decay_masked = ttnn.cumsum(grad_log_L, dim=-2, reverse_order=True)  # (B, H, T, T)

        # Backward through masking (strictly lower triangular): zero out s >= t
        grad_log_decay_exp = ttnn.where(mask_low, grad_log_decay_masked, zeros_4d)

        # Backward through expand: sum over s dimension (dim=-1)
        grad_log_decay_perm = ttnn.sum(grad_log_decay_exp, dim=-1)  # (B, H, T)

        # Backward through permute: (B, H, T) -> (B, T, H)
        grad_log_decay = ttnn.permute(grad_log_decay_perm, [0, 2, 1])  # (B, T, H)

        # Backward through log: grad_decay = grad_log_decay / decay
        grad_decay = ttnn.div(grad_log_decay, ttnn.add(decay, 1e-8))  # (B, T, H)

        # --- Step 7 backward: decay = exp(-dt * A) ---
        # grad_decay * decay = grad of (-dt * A)
        grad_neg_dt_A = ttnn.mul(grad_decay, decay)  # (B, T, H)

        # grad_dt = grad_neg_dt_A * (-A)
        neg_A = ttnn.neg(A_expanded)  # (1, 1, H)
        grad_dt = ttnn.mul(grad_neg_dt_A, neg_A)  # (B, T, H)

        # grad_A = sum over (B, T) of grad_neg_dt_A * (-dt) -> (H,)
        neg_dt = ttnn.neg(dt)  # (B, T, H)
        grad_A_full = ttnn.mul(grad_neg_dt_A, neg_dt)  # (B, T, H)
        grad_A = ttnn.sum(grad_A_full, dim=0)  # (T, H)
        grad_A = ttnn.sum(grad_A, dim=0)  # (H,)

        # --- Step 6 backward: A = exp(A_log) ---
        # grad_A_log = grad_A * A
        grad_A_log = ttnn.mul(grad_A, A)  # (H,)

        # --- Step 8 backward: CB = C_mat @ B_mat^T ---
        # grad_C_mat = grad_CB @ B_mat — (B, T, d_state)
        grad_C_mat = ttnn.matmul(grad_CB, B_mat)  # (B, T, T) @ (B, T, d_state) -> (B, T, d_state)
        # grad_B_mat = grad_CB^T @ C_mat — (B, T, d_state)
        grad_CB_T = ttnn.transpose(grad_CB, -2, -1)  # (B, T, T)
        grad_B_mat = ttnn.matmul(grad_CB_T, C_mat)  # (B, T, d_state)

        # --- Step 5 backward: V_h from x_conv reshape ---
        # grad_x_conv_from_V = grad_V_h permuted back to (B, T, d_inner)
        grad_x_conv_from_V = self._reshape_y(grad_V_h, B, T, H, d_inner)  # (B, T, d_inner)

        # --- Steps 3-4 backward: B_mat, C_mat from linear(x_conv) ---
        # grad_B_proj_weight = x_conv^T @ grad_B_mat — (d_inner, d_state)
        x_conv_2d = ttnn.reshape(x_conv, [B * T, d_inner])
        grad_B_mat_2d = ttnn.reshape(grad_B_mat, [B * T, d_state])
        grad_B_proj_weight = ttnn.matmul(ttnn.transpose(x_conv_2d, 0, 1), grad_B_mat_2d)

        # grad_x_conv_from_B = grad_B_mat @ B_proj_w^T — (B, T, d_inner)
        grad_x_conv_from_B = ttnn.matmul(grad_B_mat_2d, ttnn.transpose(self.B_proj_weight, 0, 1))
        grad_x_conv_from_B = ttnn.reshape(grad_x_conv_from_B, [B, T, d_inner])

        # grad_C_proj_weight = x_conv^T @ grad_C_mat — (d_inner, d_state)
        grad_C_mat_2d = ttnn.reshape(grad_C_mat, [B * T, d_state])
        grad_C_proj_weight = ttnn.matmul(ttnn.transpose(x_conv_2d, 0, 1), grad_C_mat_2d)

        # grad_x_conv_from_C = grad_C_mat @ C_proj_w^T — (B, T, d_inner)
        grad_x_conv_from_C = ttnn.matmul(grad_C_mat_2d, ttnn.transpose(self.C_proj_weight, 0, 1))
        grad_x_conv_from_C = ttnn.reshape(grad_x_conv_from_C, [B, T, d_inner])

        # --- Step 2 backward: dt = softplus(dt_pre) ---
        grad_dt_pre = ttnn.softplus_bw(grad_dt, dt_pre)[0]  # (B, T, H)

        # grad_dt_proj_weight = x_conv^T @ grad_dt_pre — (d_inner, H)
        grad_dt_2d = ttnn.reshape(grad_dt_pre, [B * T, H])
        grad_dt_proj_weight = ttnn.matmul(ttnn.transpose(x_conv_2d, 0, 1), grad_dt_2d)

        # grad_dt_proj_bias = sum over (B, T) of grad_dt_pre — (H,)
        grad_dt_proj_bias = ttnn.sum(grad_dt_pre, dim=0)  # (T, H)
        grad_dt_proj_bias = ttnn.sum(grad_dt_proj_bias, dim=0)  # (H,)

        # grad_x_conv_from_dt = grad_dt_pre @ dt_proj_w^T — (B, T, d_inner)
        grad_x_conv_from_dt = ttnn.matmul(grad_dt_2d, ttnn.transpose(self.dt_proj_weight, 0, 1))
        grad_x_conv_from_dt = ttnn.reshape(grad_x_conv_from_dt, [B, T, d_inner])

        # --- Total grad_x_conv (from V, B, C, dt paths) ---
        grad_x_conv = ttnn.add(grad_x_conv_from_V, grad_x_conv_from_B)
        grad_x_conv = ttnn.add(grad_x_conv, grad_x_conv_from_C)
        grad_x_conv = ttnn.add(grad_x_conv, grad_x_conv_from_dt)

        # --- Compute grad_z on device: grad_z = grad_Y_pre * Y_ssd * silu'(z) ---
        # Use cached Y_ssd (pre-gate SSD output)
        Y_ssd = c["Y_ssd"]
        grad_Y_pre_Y_ssd = ttnn.mul(grad_Y_pre, Y_ssd)
        grad_z = ttnn.silu_bw(grad_Y_pre_Y_ssd, z)[0]  # (B, T, d_inner)

        # --- Backward step 2: x_conv = silu(conv1d(x_branch)) — on device ---
        # silu backward: need conv output (pre-silu) — recompute on device
        conv_out = self._conv1d_simple(ttnn.transpose(c["x_branch"], 1, 2), T)  # (B, d_inner, T)
        conv_out = ttnn.transpose(conv_out, 1, 2)  # (B, T, d_inner)
        grad_conv_input = ttnn.silu_bw(grad_x_conv, conv_out)[0]
        grad_conv_input_t = ttnn.transpose(grad_conv_input, 1, 2)  # (B, d_inner, T)
        grad_x_branch = self._conv1d_backward(grad_conv_input_t, c["x_branch"], T, B, d_inner)
        grad_conv1d_weight, grad_conv1d_bias = self._conv1d_param_grads(grad_conv_input_t, c["x_branch"], T, B, d_inner)

        # --- Backward step 1: x_branch, z from split(in_proj(x)) — on device ---
        grad_xz = ttnn.concat([grad_x_branch, grad_z], dim=-1)

        x_2d = ttnn.reshape(c["x"], [B * T, D])
        grad_xz_2d = ttnn.reshape(grad_xz, [B * T, 2 * d_inner])
        grad_in_proj_weight = ttnn.matmul(ttnn.transpose(x_2d, 0, 1), grad_xz_2d)
        grad_x = ttnn.matmul(grad_xz_2d, ttnn.transpose(self.in_proj_weight, 0, 1))
        grad_x = ttnn.reshape(grad_x, [B, T, D])

        grads = {
            "in_proj_weight": grad_in_proj_weight,
            "conv1d_weight": grad_conv1d_weight,
            "conv1d_bias": grad_conv1d_bias,
            "dt_proj_weight": grad_dt_proj_weight,
            "dt_proj_bias": grad_dt_proj_bias,
            "B_proj_weight": grad_B_proj_weight,
            "C_proj_weight": grad_C_proj_weight,
            "out_proj_weight": grad_out_proj_weight,
            "A_log": grad_A_log,
            "D": grad_D,
            "norm_weight": grad_norm_weight,
        }

        return grad_x, grads

    def _conv1d_backward(self, grad_conv_input: "ttnn.Tensor", x_branch: "ttnn.Tensor",
                         T: int, B: int, d_inner: int) -> "ttnn.Tensor":
        """Backward through the depthwise conv1d (device-side).

        grad_conv_input: (B, d_inner, T) — grad w.r.t. conv output (before silu)
        x_branch: (B, T, d_inner) — original input to conv (before transpose)

        Returns: grad_x_branch (B, T, d_inner)
        """
        k = self.d_conv
        device = self.device

        # Get conv weights: (d_inner, k)
        conv_w = ttnn.squeeze(self.conv1d_weight, dim=1)  # (d_inner, k)

        # Backward: grad_x_padded[t+i] += w[:,i] * grad[t]
        # We accumulate k terms, each padded to (B, d_inner, T + k - 1)
        # Using concat for front padding (ttnn.pad doesn't support non-zero front padding)
        grad_x_padded = None
        for i in range(k):
            w_i = ttnn.slice(conv_w, [0, i], [d_inner, i + 1])
            w_i = ttnn.reshape(w_i, [1, d_inner, 1])
            term = ttnn.mul(w_i, grad_conv_input)  # (B, d_inner, T) broadcast
            # Pad term to (B, d_inner, T + k - 1) with zeros at front [0, i) and back [i+T, T+k-1)
            if i > 0:
                zeros_front = ttnn.zeros((B, d_inner, i), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                term = ttnn.concat([zeros_front, term], dim=-1)
            if k - 1 - i > 0:
                zeros_back = ttnn.zeros((B, d_inner, k - 1 - i), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                term = ttnn.concat([term, zeros_back], dim=-1)
            # term is now (B, d_inner, T + k - 1)
            if grad_x_padded is None:
                grad_x_padded = term
            else:
                grad_x_padded = ttnn.add(grad_x_padded, term)

        # Remove front padding: take [:, :, k-1:]
        grad_x_t = ttnn.slice(grad_x_padded, [0, 0, k - 1], [B, d_inner, k - 1 + T])

        # Transpose back: (B, d_inner, T) -> (B, T, d_inner)
        grad_x_branch = ttnn.transpose(grad_x_t, 1, 2)

        return grad_x_branch

    def _conv1d_param_grads(self, grad_conv_input: "ttnn.Tensor", x_branch: "ttnn.Tensor",
                            T: int, B: int, d_inner: int) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """Compute gradients for conv1d weight and bias (device-side).

        grad_conv_input: (B, d_inner, T) — grad w.r.t. conv output (pre-bias, pre-silu)
        x_branch: (B, T, d_inner) — original input
        """
        k = self.d_conv
        device = self.device

        # Transpose x_branch: (B, T, d_inner) -> (B, d_inner, T)
        x_t = ttnn.transpose(x_branch, 1, 2)
        # Pad x: (B, d_inner, T + k - 1) with k-1 zeros at front (using concat)
        zeros_prefix = ttnn.zeros((B, d_inner, k - 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        x_padded = ttnn.concat([zeros_prefix, x_t], dim=-1)

        # grad_w[c, i] = sum_{b,t} grad[b, c, t] * x_padded[b, c, t+i]
        grad_w_list = []
        for i in range(k):
            x_slice = ttnn.slice(x_padded, [0, 0, i], [B, d_inner, i + T])
            product = ttnn.mul(grad_conv_input, x_slice)  # (B, d_inner, T)
            # Sum over dim 0 (B) and dim 2 (T) -> (d_inner,)
            grad_w_i = ttnn.sum(product, dim=0)  # (d_inner, T)
            grad_w_i = ttnn.sum(grad_w_i, dim=-1)  # (d_inner,)
            # Reshape to (d_inner, 1) so concat produces (d_inner, k)
            grad_w_i = ttnn.reshape(grad_w_i, [d_inner, 1])
            grad_w_list.append(grad_w_i)

        # Stack: (d_inner, k) -> (d_inner, 1, k)
        grad_w = ttnn.concat(grad_w_list, dim=-1)  # (d_inner, k)
        grad_w = ttnn.unsqueeze(grad_w, dim=1)  # (d_inner, 1, k)

        # grad_b = sum over B, T of grad
        grad_b = ttnn.sum(grad_conv_input, dim=0)  # (d_inner, T)
        grad_b = ttnn.sum(grad_b, dim=-1)  # (d_inner,)

        return grad_w, grad_b

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        return {
            "in_proj_weight": self.in_proj_weight,
            "conv1d_weight": self.conv1d_weight,
            "conv1d_bias": self.conv1d_bias,
            "dt_proj_weight": self.dt_proj_weight,
            "dt_proj_bias": self.dt_proj_bias,
            "B_proj_weight": self.B_proj_weight,
            "C_proj_weight": self.C_proj_weight,
            "out_proj_weight": self.out_proj_weight,
            "A_log": self.A_log,
            "D": self.D,
            "norm_weight": self.norm.weight,
        }

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        for k, v in params.items():
            if k == "norm_weight":
                self.norm.weight = v
            else:
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Full Model (tt-nn native)
# ---------------------------------------------------------------------------

class TTAttentionLayer:
    """Multi-head self-attention with RoPE and causal masking — tt-nn native.

    Matches the PyTorch AttentionLayer in model.py.
    Forward:  qkv = linear(x); q,k,v = chunk(qkv); q,k = rope(q,k);
              attn = softmax(causal_mask(q*k^T * scale)); out = linear(attn@v)
    Backward: manual, all on device.
    """

    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)

        # QKV projection: (d_model, 3*d_model)
        qkv_w = torch.randn(self.d_model, 3 * self.d_model, dtype=torch.bfloat16) * 0.02
        self.qkv_weight = to_device(qkv_w, device)

        # Output projection: (d_model, d_model)
        out_w = torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02
        self.out_proj_weight = to_device(out_w, device)

        # Precompute RoPE cos/sin tables (T will be fixed at 128 for now)
        # These are transferred to device once and reused
        d_rope = self.d_head
        freqs = 1.0 / (10000 ** (torch.arange(0, d_rope, 2).float() / d_rope))
        # Store freqs for lazy init of cos/sin per T
        self._rope_freqs = freqs
        self._rope_cos = None
        self._rope_sin = None
        self._rope_T = 0

        self._cache = {}

    def _init_rope(self, T: int, device):
        """Initialize RoPE cos/sin tables for sequence length T on device."""
        if self._rope_T == T and self._rope_cos is not None:
            return
        positions = torch.arange(T, dtype=torch.float32)
        angles = torch.outer(positions, self._rope_freqs)  # (T, d_rope/2)
        cos = torch.cos(angles).to(torch.bfloat16)  # (T, d_rope/2)
        sin = torch.sin(angles).to(torch.bfloat16)
        # Shape (1, 1, T, d_rope/2) for broadcasting with (B, H, T, d_head)
        cos_4d = cos.unsqueeze(0).unsqueeze(0)
        sin_4d = sin.unsqueeze(0).unsqueeze(0)
        self._rope_cos = ttnn.from_torch(cos_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._rope_sin = ttnn.from_torch(sin_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._rope_T = T

    def _apply_rope(self, x: "ttnn.Tensor", B, H, T) -> "ttnn.Tensor":
        """Apply RoPE to x: (B, H, T, d_head) -> (B, H, T, d_head).

        Splits x into [x1, x2] along last dim, rotates:
          rotated = cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        """
        d_h = self.d_head
        x1 = ttnn.slice(x, [0, 0, 0, 0], [B, H, T, d_h // 2])
        x2 = ttnn.slice(x, [0, 0, 0, d_h // 2], [B, H, T, d_h])
        x1_cos = ttnn.mul(x1, self._rope_cos)
        x2_sin = ttnn.mul(x2, self._rope_sin)
        x1_sin = ttnn.mul(x1, self._rope_sin)
        x2_cos = ttnn.mul(x2, self._rope_cos)
        rotated = ttnn.concat([ttnn.sub(x1_cos, x2_sin), ttnn.add(x1_sin, x2_cos)], dim=-1)
        return rotated

    def _apply_rope_backward(self, grad_rotated: "ttnn.Tensor", B, H, T) -> "ttnn.Tensor":
        """Backward through RoPE.

        RoPE is a rotation: rotated = [x1*cos - x2*sin, x1*sin + x2*cos]
        grad_x1 = grad_r1*cos + grad_r2*sin
        grad_x2 = -grad_r1*sin + grad_r2*cos
        """
        d_h = self.d_head
        grad_r1 = ttnn.slice(grad_rotated, [0, 0, 0, 0], [B, H, T, d_h // 2])
        grad_r2 = ttnn.slice(grad_rotated, [0, 0, 0, d_h // 2], [B, H, T, d_h])
        grad_x1 = ttnn.add(ttnn.mul(grad_r1, self._rope_cos), ttnn.mul(grad_r2, self._rope_sin))
        grad_x2 = ttnn.add(ttnn.mul(ttnn.neg(grad_r1), self._rope_sin), ttnn.mul(grad_r2, self._rope_cos))
        grad_x = ttnn.concat([grad_x1, grad_x2], dim=-1)
        return grad_x

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        """x: (B, T, d_model) -> out: (B, T, d_model)"""
        B, T, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        H, d_h = self.n_heads, self.d_head
        device = self.device

        self._init_rope(T, device)

        # QKV projection: (B, T, d_model) @ (d_model, 3*d_model) -> (B, T, 3*d_model)
        qkv = ttnn.linear(x, self.qkv_weight)  # (B, T, 3*D)

        # Split into q, k, v: each (B, T, D)
        q = ttnn.slice(qkv, [0, 0, 0], [B, T, D])
        k = ttnn.slice(qkv, [0, 0, D], [B, T, 2 * D])
        v = ttnn.slice(qkv, [0, 0, 2 * D], [B, T, 3 * D])

        # Reshape to (B, H, T, d_head)
        q_4d = ttnn.permute(ttnn.reshape(q, [B, T, H, d_h]), [0, 2, 1, 3])
        k_4d = ttnn.permute(ttnn.reshape(k, [B, T, H, d_h]), [0, 2, 1, 3])
        v_4d = ttnn.permute(ttnn.reshape(v, [B, T, H, d_h]), [0, 2, 1, 3])

        # Apply RoPE to q and k
        q_rope = self._apply_rope(q_4d, B, H, T)
        k_rope = self._apply_rope(k_4d, B, H, T)

        # Attention scores: (B, H, T, T) = Q @ K^T * scale
        scores = ttnn.matmul(q_rope, ttnn.transpose(k_rope, -2, -1))  # (B, H, T, T)
        scale_tt = ttnn.from_torch(torch.tensor([self.scale], dtype=torch.bfloat16),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        scores = ttnn.mul(scores, scale_tt)

        # Causal mask: upper triangular (s > t) -> -inf
        ones_tt = ttnn.from_torch(torch.ones(T, T, dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mask = ttnn.triu(ones_tt, diagonal=1)  # 1 for s > t
        neg_inf = ttnn.full((B, H, T, T), -1e4, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
        scores_masked = ttnn.where(mask, neg_inf, scores)

        # Softmax over last dim
        attn = ttnn.softmax(scores_masked, dim=-1)  # (B, H, T, T)

        # Output: (B, H, T, d_head) = attn @ V
        out_4d = ttnn.matmul(attn, v_4d)  # (B, H, T, d_head)

        # Reshape back: (B, T, d_model)
        out = ttnn.reshape(ttnn.permute(out_4d, [0, 2, 1, 3]), [B, T, D])

        # Output projection: (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
        out = ttnn.linear(out, self.out_proj_weight)

        # Cache for backward
        self._cache = {
            "x": x, "qkv": qkv, "q": q, "k": k, "v": v,
            "q_4d": q_4d, "k_4d": k_4d, "v_4d": v_4d,
            "q_rope": q_rope, "k_rope": k_rope,
            "scores": scores, "scores_masked": scores_masked,
            "attn": attn, "out_4d": out_4d,
            "scale_tt": scale_tt,
        }

        return out

    def backward(self, grad_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", Dict[str, "ttnn.Tensor"]]:
        """Backward pass for attention layer — all on device.

        grad_out: (B, T, d_model)
        Returns: (grad_x, grads_dict)
        """
        c = self._cache
        B, T, D = int(c["x"].shape[0]), int(c["x"].shape[1]), int(c["x"].shape[2])
        H, d_h = self.n_heads, self.d_head
        device = self.device

        # --- Backward through out_proj: out = linear(reshape(attn@v), out_proj_w) ---
        # grad_out_4d = grad_out @ out_proj_w^T — (B, T, D) @ (D, D) -> (B, T, D)
        grad_out_proj = ttnn.linear(grad_out, ttnn.transpose(self.out_proj_weight, 0, 1))

        # grad_out_proj_weight = reshape(attn@v)^T @ grad_out — (D, D)
        out_4d = c["out_4d"]  # (B, H, T, d_head)
        out_2d = ttnn.reshape(ttnn.permute(out_4d, [0, 2, 1, 3]), [B * T, D])  # (B*T, D)
        grad_out_2d = ttnn.reshape(grad_out, [B * T, D])
        grad_out_proj_weight = ttnn.matmul(ttnn.transpose(out_2d, 0, 1), grad_out_2d)  # (D, D)

        # Reshape grad back to 4D: (B*T, D) -> (B, T, H, d_head) -> (B, H, T, d_head)
        grad_out_4d = ttnn.reshape(grad_out_proj, [B, T, H, d_h])
        grad_out_4d = ttnn.permute(grad_out_4d, [0, 2, 1, 3])  # (B, H, T, d_head)

        # --- Backward through out_4d = attn @ v_4d ---
        # grad_attn = grad_out_4d @ v_4d^T — (B, H, T, T)
        v_4d = c["v_4d"]
        grad_attn = ttnn.matmul(grad_out_4d, ttnn.transpose(v_4d, -2, -1))

        # grad_v_4d = attn^T @ grad_out_4d — (B, H, T, d_head)
        attn = c["attn"]
        grad_v_4d = ttnn.matmul(ttnn.transpose(attn, -2, -1), grad_out_4d)

        # --- Backward through softmax ---
        # grad_scores = (grad_attn - sum(grad_attn * attn, dim=-1, keepdim=True)) * attn
        grad_attn_sum = ttnn.sum(ttnn.mul(grad_attn, attn), dim=-1)  # (B, H, T)
        grad_attn_sum = ttnn.reshape(grad_attn_sum, [B, H, T, 1])
        grad_attn_sum = ttnn.expand(grad_attn_sum, [B, H, T, T])
        grad_scores = ttnn.mul(ttnn.sub(grad_attn, grad_attn_sum), attn)

        # --- Backward through causal masking (where mask, -inf, scores) ---
        # The mask sets scores to -inf for s > t. After softmax, those are 0.
        # The gradient through where is: grad_scores is 0 for s > t (since attn=0 there).
        # No additional gradient needed — the where is already accounted for in softmax.
        # But we need to zero out grad_scores for s > t to avoid spurious gradients.
        ones_tt = ttnn.from_torch(torch.ones(T, T, dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mask_causal = ttnn.tril(ones_tt, diagonal=0)  # 1 for s <= t
        zeros_4d = ttnn.zeros((B, H, T, T), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        grad_scores = ttnn.where(mask_causal, grad_scores, zeros_4d)

        # --- Backward through scores = QK^T * scale ---
        scale_tt = c["scale_tt"]
        grad_scores = ttnn.mul(grad_scores, scale_tt)  # undo scale

        # grad_q_rope = grad_scores @ k_rope — (B, H, T, d_head)
        k_rope = c["k_rope"]
        grad_q_rope = ttnn.matmul(grad_scores, k_rope)

        # grad_k_rope = grad_scores^T @ q_rope — (B, H, T, d_head)
        q_rope = c["q_rope"]
        grad_k_rope = ttnn.matmul(ttnn.transpose(grad_scores, -2, -1), q_rope)

        # --- Backward through RoPE ---
        grad_q_4d = self._apply_rope_backward(grad_q_rope, B, H, T)
        grad_k_4d = self._apply_rope_backward(grad_k_rope, B, H, T)

        # --- Backward through reshape (B, H, T, d_head) -> (B, T, D) ---
        # q: (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, D)
        grad_q = ttnn.reshape(ttnn.permute(grad_q_4d, [0, 2, 1, 3]), [B, T, D])
        grad_k = ttnn.reshape(ttnn.permute(grad_k_4d, [0, 2, 1, 3]), [B, T, D])
        grad_v = ttnn.reshape(ttnn.permute(grad_v_4d, [0, 2, 1, 3]), [B, T, D])

        # --- Backward through QKV split: concat([q, k, v]) ---
        grad_qkv = ttnn.concat([grad_q, grad_k, grad_v], dim=-1)  # (B, T, 3*D)

        # --- Backward through QKV linear: qkv = linear(x, qkv_weight) ---
        # grad_qkv_weight = x^T @ grad_qkv — (D, 3*D)
        x_2d = ttnn.reshape(c["x"], [B * T, D])
        grad_qkv_2d = ttnn.reshape(grad_qkv, [B * T, 3 * D])
        grad_qkv_weight = ttnn.matmul(ttnn.transpose(x_2d, 0, 1), grad_qkv_2d)

        # grad_x = grad_qkv @ qkv_weight^T — (B, T, D)
        grad_x = ttnn.matmul(grad_qkv_2d, ttnn.transpose(self.qkv_weight, 0, 1))
        grad_x = ttnn.reshape(grad_x, [B, T, D])

        grads = {
            "qkv_weight": grad_qkv_weight,
            "out_proj_weight": grad_out_proj_weight,
        }

        return grad_x, grads

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        return {"qkv_weight": self.qkv_weight, "out_proj_weight": self.out_proj_weight}

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        if "qkv_weight" in params:
            self.qkv_weight = params["qkv_weight"]
        if "out_proj_weight" in params:
            self.out_proj_weight = params["out_proj_weight"]


class TTGatedResidualLayer:
    """Gated pre-norm residual wrapper: x + sigmoid(gate) * layer(norm(x)).

    Matches the PyTorch GatedResidualLayer in model.py.
    gate init 0 -> sigmoid(0) = 0.5, a damped residual at initialization.
    """

    def __init__(self, layer, d_model: int, device):
        self.layer = layer
        self.norm = TTRMSNorm(d_model, device)
        self.device = device
        # gate init 0 -> sigmoid(0) = 0.5
        self.gate = ttnn.from_torch(
            torch.zeros(1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        normed = self.norm.forward(x)
        inner = self.layer.forward(normed)
        # x + sigmoid(gate) * inner
        gate_val = ttnn.sigmoid(self.gate)  # (1,)
        # Broadcast gate to match inner shape
        gated_inner = ttnn.mul(gate_val, inner)  # (B, T, d_model)
        out = ttnn.add(x, gated_inner)
        # Cache for backward
        self._cached_x = x
        self._cached_inner = inner
        return out

    def backward(self, grad_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", Dict[str, "ttnn.Tensor"]]:
        """Backward through gated residual.

        grad_out: (B, T, d_model)
        Returns: (grad_x, grads_dict) where grads_dict includes layer grads,
        norm grads, and gate grad.
        """
        device = self.device

        # Forward was: y = x + sigmoid(gate) * layer(norm(x))
        # grad_x = grad_out (residual) + sigmoid(gate) * grad_inner
        # grad_gate = grad_out * inner * sigmoid'(gate)
        # grad_inner = grad_out * sigmoid(gate)
        # grad_norm_input = layer.backward(grad_inner) -> grad_x_from_inner
        # Then grad_x = grad_out + grad_x_from_inner (through norm)

        # We need the inner output (layer output) — it's in the layer cache as "out"
        # Actually, the layer forward returns out but doesn't cache it separately
        # We can recompute it or get it from the cache
        # The layer's last output is what was returned by layer.forward()
        # Let's recompute it from the cache
        # Actually, we need to cache it in the gated residual forward

        # grad_inner = grad_out * sigmoid(gate)
        gate_val = ttnn.sigmoid(self.gate)
        grad_inner = ttnn.mul(grad_out, gate_val)

        # grad_gate = sum(grad_out * inner * sigmoid'(gate))
        # sigmoid'(gate) = sigmoid(gate) * (1 - sigmoid(gate))
        # We need inner — cache it in forward
        inner = self._cached_inner
        grad_out_inner = ttnn.mul(grad_out, inner)
        ones = ttnn.from_torch(torch.ones(1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        sig_prime = ttnn.mul(gate_val, ttnn.sub(ones, gate_val))
        grad_gate_val = ttnn.mul(grad_out_inner, sig_prime)
        # Gate is a scalar — sum over all dimensions on device
        grad_gate = ttnn.sum(grad_gate_val, dim=0)
        grad_gate = ttnn.sum(grad_gate, dim=0)
        grad_gate = ttnn.sum(grad_gate, dim=0)  # (1,)

        # Layer backward (through norm and layer)
        grad_norm_input, layer_grads = self.layer.backward(grad_inner)

        # Norm backward — on device
        grad_x_from_norm, grad_norm_w = self.norm.backward(grad_norm_input, self._cached_x)

        # Total grad_x = grad_out (residual) + grad_x_from_norm
        grad_x = ttnn.add(grad_out, grad_x_from_norm)

        # Collect all grads
        all_grads = {}
        for k, v in layer_grads.items():
            all_grads[k] = v
        all_grads["wrapper_norm_weight"] = grad_norm_w
        all_grads["gate"] = grad_gate

        return grad_x, all_grads

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        params = {"gate": self.gate, "wrapper_norm_weight": self.norm.weight}
        params.update(self.layer.get_params())
        return params

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        if "gate" in params:
            self.gate = params["gate"]
        if "wrapper_norm_weight" in params:
            self.norm.weight = params["wrapper_norm_weight"]
        layer_params = {k: v for k, v in params.items() if k not in ("gate", "wrapper_norm_weight")}
        if layer_params:
            self.layer.set_params(layer_params)


class TTWorkspaceModule:
    """Perceiver-style workspace with m learned slot vectors — tt-nn native.

    Two cross-attention passes per application:
      1. Read: slots attend over hidden states (Q from slots, KV from x)
      2. Write: hidden states attend over slots (Q from x, KV from slots)

    Both passes use gated residual connections and RMSNorm.
    No causal masking (bidirectional cross-attention). No RoPE.
    """

    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.n_slots = config.n_workspace_slots
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)

        # Learnable slot embeddings: (n_slots, d_model)
        slots = torch.randn(self.n_slots, self.d_model, dtype=torch.bfloat16) * 0.02
        self.slots = to_device(slots, device)

        # Read projections: Q from slots, K/V from x
        self.read_q_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)
        self.read_k_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)
        self.read_v_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)
        self.read_out_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)

        # Write projections: Q from x, K/V from slots
        self.write_q_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)
        self.write_k_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)
        self.write_v_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)
        self.write_out_weight = to_device(torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02, device)

        # Norms
        self.norm = TTRMSNorm(self.d_model, device)       # for x
        self.slot_norm = TTRMSNorm(self.d_model, device)  # for slots

        # Fixed weight for slot parameter normalization (not learned)
        self._slot_param_norm_weight = ttnn.from_torch(
            torch.ones(self.d_model, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # Gates (init 0 -> sigmoid(0) = 0.5)
        self.read_gate = ttnn.from_torch(torch.zeros(1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.write_gate = ttnn.from_torch(torch.zeros(1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        self._cache = {}

    def _reshape_to_heads(self, x, B, L):
        """(B, L, D) -> (B, H, L, d_head)"""
        return ttnn.permute(ttnn.reshape(x, [B, L, self.n_heads, self.d_head]), [0, 2, 1, 3])

    def _reshape_from_heads(self, x, B, L):
        """(B, H, L, d_head) -> (B, L, D)"""
        return ttnn.reshape(ttnn.permute(x, [0, 2, 1, 3]), [B, L, self.d_model])

    def _softmax_backward(self, grad_attn, attn, B, H, L_q, L_k):
        """Manual softmax backward: grad_scores = (grad_attn - sum(grad_attn*attn, -1, keepdim)) * attn"""
        grad_sum = ttnn.sum(ttnn.mul(grad_attn, attn), dim=-1)  # (B, H, L_q)
        grad_sum = ttnn.reshape(grad_sum, [B, H, L_q, 1])
        grad_sum = ttnn.expand(grad_sum, [B, H, L_q, L_k])
        return ttnn.mul(ttnn.sub(grad_attn, grad_sum), attn)

    def forward(self, x: "ttnn.Tensor", slot_state: Optional["ttnn.Tensor"]) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """x: (B, T, D), slot_state: (B, m, D) or None -> (x_out, slot_state_out)"""
        B, T, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        H, d_h, m = self.n_heads, self.d_head, self.n_slots
        device = self.device

        # Initialize slots: (B, m, D)
        if slot_state is None:
            slots = ttnn.reshape(self.slots, [1, m, D])
            slots = ttnn.expand(slots, [B, m, D])
        else:
            slots = slot_state

        scale_tt = ttnn.from_torch(torch.tensor([self.scale], dtype=torch.bfloat16),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # --- Read: slots attend over hidden states ---
        rq = self._reshape_to_heads(ttnn.linear(slots, self.read_q_weight), B, m)   # (B, H, m, d_h)
        rk = self._reshape_to_heads(ttnn.linear(x, self.read_k_weight), B, T)       # (B, H, T, d_h)
        rv = self._reshape_to_heads(ttnn.linear(x, self.read_v_weight), B, T)       # (B, H, T, d_h)

        read_scores = ttnn.matmul(rq, ttnn.transpose(rk, -2, -1))  # (B, H, m, T)
        read_scores = ttnn.mul(read_scores, scale_tt)
        read_attn = ttnn.softmax(read_scores, dim=-1)               # (B, H, m, T)
        read_out_4d = ttnn.matmul(read_attn, rv)                    # (B, H, m, d_h)
        read_out = self._reshape_from_heads(read_out_4d, B, m)      # (B, m, D)
        read_out_proj = ttnn.linear(read_out, self.read_out_weight)  # (B, m, D)

        # slots = slot_norm(slots + sigmoid(read_gate) * read_out_proj)
        read_gate_val = ttnn.sigmoid(self.read_gate)
        slots_pre_norm = ttnn.add(slots, ttnn.mul(read_gate_val, read_out_proj))
        slots_out = self.slot_norm.forward(slots_pre_norm)

        # --- Write: hidden states attend over slots ---
        wq = self._reshape_to_heads(ttnn.linear(x, self.write_q_weight), B, T)       # (B, H, T, d_h)
        wk = self._reshape_to_heads(ttnn.linear(slots_out, self.write_k_weight), B, m)  # (B, H, m, d_h)
        wv = self._reshape_to_heads(ttnn.linear(slots_out, self.write_v_weight), B, m)  # (B, H, m, d_h)

        write_scores = ttnn.matmul(wq, ttnn.transpose(wk, -2, -1))  # (B, H, T, m)
        write_scores = ttnn.mul(write_scores, scale_tt)
        write_attn = ttnn.softmax(write_scores, dim=-1)              # (B, H, T, m)
        write_out_4d = ttnn.matmul(write_attn, wv)                   # (B, H, T, d_h)
        write_out = self._reshape_from_heads(write_out_4d, B, T)     # (B, T, D)
        write_out_proj = ttnn.linear(write_out, self.write_out_weight)  # (B, T, D)

        # x = norm(x + sigmoid(write_gate) * write_out_proj)
        write_gate_val = ttnn.sigmoid(self.write_gate)
        x_pre_norm = ttnn.add(x, ttnn.mul(write_gate_val, write_out_proj))
        x_out = self.norm.forward(x_pre_norm)

        # Cache for backward
        self._cache = {
            "x": x, "slots_in": slots, "slots_pre_norm": slots_pre_norm,
            "rq": rq, "rk": rk, "rv": rv, "read_attn": read_attn, "read_out_4d": read_out_4d,
            "read_out": read_out, "read_out_proj": read_out_proj,
            "read_gate_val": read_gate_val,
            "slots_out": slots_out,
            "wq": wq, "wk": wk, "wv": wv, "write_attn": write_attn, "write_out_4d": write_out_4d,
            "write_out": write_out, "write_out_proj": write_out_proj,
            "write_gate_val": write_gate_val,
            "x_pre_norm": x_pre_norm, "scale_tt": scale_tt,
            "B": B, "T": T, "m": m,
        }

        return x_out, slots_out

    def backward(self, grad_x_out: "ttnn.Tensor", grad_slots_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor", Dict[str, "ttnn.Tensor"]]:
        """Backward through workspace module — all on device.

        grad_x_out: (B, T, D)
        grad_slots_out: (B, m, D) — gradient w.r.t. slots_out
        Returns: (grad_x, grad_slots_in, grads_dict)
        """
        c = self._cache
        B, T, m = c["B"], c["T"], c["m"]
        H, d_h, D = self.n_heads, self.d_head, self.d_model
        device = self.device
        scale_tt = c["scale_tt"]

        # --- Backward through x_out = norm(x_pre_norm) ---
        grad_x_pre_norm, grad_norm_w = self.norm.backward(grad_x_out, c["x_pre_norm"])

        # --- Backward through x_pre_norm = x + sigmoid(write_gate) * write_out_proj ---
        grad_x_from_write = grad_x_pre_norm  # residual
        grad_write_out_proj = ttnn.mul(grad_x_pre_norm, c["write_gate_val"])

        # grad_write_gate = sum(grad_x_pre_norm * write_out_proj * sigmoid'(write_gate))
        write_gate_val = c["write_gate_val"]
        ones_1 = ttnn.from_torch(torch.ones(1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        write_sig_prime = ttnn.mul(write_gate_val, ttnn.sub(ones_1, write_gate_val))
        grad_write_gate = ttnn.mul(ttnn.mul(grad_x_pre_norm, c["write_out_proj"]), write_sig_prime)
        grad_write_gate = ttnn.sum(ttnn.sum(ttnn.sum(grad_write_gate, dim=0), dim=0), dim=0)

        # --- Backward through write_out_proj = linear(write_out) ---
        grad_write_out = ttnn.linear(grad_write_out_proj, ttnn.transpose(self.write_out_weight, 0, 1))
        grad_write_out_2d = ttnn.reshape(c["write_out"], [B * T, D])
        grad_write_out_proj_2d = ttnn.reshape(grad_write_out_proj, [B * T, D])
        grad_write_out_weight = ttnn.matmul(ttnn.transpose(grad_write_out_2d, 0, 1), grad_write_out_proj_2d)

        # --- Backward through write_out = reshape(write_attn @ wv) ---
        grad_write_out_4d = self._reshape_to_heads(grad_write_out, B, T)  # (B, H, T, d_h)
        wv = c["wv"]
        grad_write_attn = ttnn.matmul(grad_write_out_4d, ttnn.transpose(wv, -2, -1))  # (B, H, T, m)
        grad_wv = ttnn.matmul(ttnn.transpose(c["write_attn"], -2, -1), grad_write_out_4d)  # (B, H, m, d_h)

        # --- Backward through write_attn = softmax(write_scores * scale) ---
        grad_write_scores = self._softmax_backward(grad_write_attn, c["write_attn"], B, H, T, m)
        grad_write_scores = ttnn.mul(grad_write_scores, scale_tt)

        # grad_wq = grad_write_scores @ wk, grad_wk = grad_write_scores^T @ wq
        wk = c["wk"]
        wq = c["wq"]
        grad_wq = ttnn.matmul(grad_write_scores, wk)               # (B, H, T, d_h)
        grad_wk = ttnn.matmul(ttnn.transpose(grad_write_scores, -2, -1), wq)  # (B, H, m, d_h)

        # --- Backward through write projections ---
        # wq = linear(x), wk = linear(slots_out), wv = linear(slots_out)
        grad_wq_3d = self._reshape_from_heads(grad_wq, B, T)
        grad_wk_3d = self._reshape_from_heads(grad_wk, B, m)
        grad_wv_3d = self._reshape_from_heads(grad_wv, B, m)

        x_2d = ttnn.reshape(c["x"], [B * T, D])
        slots_out_2d = ttnn.reshape(c["slots_out"], [B * m, D])

        grad_write_q_weight = ttnn.matmul(ttnn.transpose(x_2d, 0, 1), ttnn.reshape(grad_wq_3d, [B * T, D]))
        grad_x_from_wq = ttnn.reshape(ttnn.matmul(ttnn.reshape(grad_wq_3d, [B * T, D]), ttnn.transpose(self.write_q_weight, 0, 1)), [B, T, D])

        grad_wk_2d = ttnn.reshape(grad_wk_3d, [B * m, D])
        grad_wv_2d = ttnn.reshape(grad_wv_3d, [B * m, D])
        grad_write_k_weight = ttnn.matmul(ttnn.transpose(slots_out_2d, 0, 1), grad_wk_2d)
        grad_write_v_weight = ttnn.matmul(ttnn.transpose(slots_out_2d, 0, 1), grad_wv_2d)

        grad_slots_from_wk = ttnn.reshape(ttnn.matmul(grad_wk_2d, ttnn.transpose(self.write_k_weight, 0, 1)), [B, m, D])
        grad_slots_from_wv = ttnn.reshape(ttnn.matmul(grad_wv_2d, ttnn.transpose(self.write_v_weight, 0, 1)), [B, m, D])

        # Total grad_slots_out (from write path + incoming grad_slots_out)
        grad_slots_total = ttnn.add(grad_slots_out, ttnn.add(grad_slots_from_wk, grad_slots_from_wv))

        # --- Backward through slots_out = slot_norm(slots_pre_norm) ---
        grad_slots_pre_norm, grad_slot_norm_w = self.slot_norm.backward(grad_slots_total, c["slots_pre_norm"])

        # --- Backward through slots_pre_norm = slots_in + sigmoid(read_gate) * read_out_proj ---
        grad_slots_in_from_read = grad_slots_pre_norm  # residual
        grad_read_out_proj = ttnn.mul(grad_slots_pre_norm, c["read_gate_val"])

        # grad_read_gate
        read_gate_val = c["read_gate_val"]
        read_sig_prime = ttnn.mul(read_gate_val, ttnn.sub(ones_1, read_gate_val))
        grad_read_gate = ttnn.mul(ttnn.mul(grad_slots_pre_norm, c["read_out_proj"]), read_sig_prime)
        grad_read_gate = ttnn.sum(ttnn.sum(ttnn.sum(grad_read_gate, dim=0), dim=0), dim=0)

        # --- Backward through read_out_proj = linear(read_out) ---
        grad_read_out = ttnn.linear(grad_read_out_proj, ttnn.transpose(self.read_out_weight, 0, 1))
        read_out_2d = ttnn.reshape(c["read_out"], [B * m, D])
        grad_read_out_proj_2d = ttnn.reshape(grad_read_out_proj, [B * m, D])
        grad_read_out_weight = ttnn.matmul(ttnn.transpose(read_out_2d, 0, 1), grad_read_out_proj_2d)

        # --- Backward through read_out = reshape(read_attn @ rv) ---
        grad_read_out_4d = self._reshape_to_heads(grad_read_out, B, m)  # (B, H, m, d_h)
        rv = c["rv"]
        grad_read_attn = ttnn.matmul(grad_read_out_4d, ttnn.transpose(rv, -2, -1))  # (B, H, m, T)
        grad_rv = ttnn.matmul(ttnn.transpose(c["read_attn"], -2, -1), grad_read_out_4d)  # (B, H, T, d_h)

        # --- Backward through read_attn = softmax(read_scores * scale) ---
        grad_read_scores = self._softmax_backward(grad_read_attn, c["read_attn"], B, H, m, T)
        grad_read_scores = ttnn.mul(grad_read_scores, scale_tt)

        # grad_rq = grad_read_scores @ rk, grad_rk = grad_read_scores^T @ rq
        rk = c["rk"]
        rq = c["rq"]
        grad_rq = ttnn.matmul(grad_read_scores, rk)               # (B, H, m, d_h)
        grad_rk = ttnn.matmul(ttnn.transpose(grad_read_scores, -2, -1), rq)  # (B, H, T, d_h)

        # --- Backward through read projections ---
        grad_rq_3d = self._reshape_from_heads(grad_rq, B, m)
        grad_rk_3d = self._reshape_from_heads(grad_rk, B, T)
        grad_rv_3d = self._reshape_from_heads(grad_rv, B, T)

        slots_in_2d = ttnn.reshape(c["slots_in"], [B * m, D])

        grad_read_q_weight = ttnn.matmul(ttnn.transpose(slots_in_2d, 0, 1), ttnn.reshape(grad_rq_3d, [B * m, D]))
        grad_slots_from_rq = ttnn.reshape(ttnn.matmul(ttnn.reshape(grad_rq_3d, [B * m, D]), ttnn.transpose(self.read_q_weight, 0, 1)), [B, m, D])

        grad_rk_2d = ttnn.reshape(grad_rk_3d, [B * T, D])
        grad_rv_2d = ttnn.reshape(grad_rv_3d, [B * T, D])
        grad_read_k_weight = ttnn.matmul(ttnn.transpose(x_2d, 0, 1), grad_rk_2d)
        grad_read_v_weight = ttnn.matmul(ttnn.transpose(x_2d, 0, 1), grad_rv_2d)

        grad_x_from_rk = ttnn.reshape(ttnn.matmul(grad_rk_2d, ttnn.transpose(self.read_k_weight, 0, 1)), [B, T, D])
        grad_x_from_rv = ttnn.reshape(ttnn.matmul(grad_rv_2d, ttnn.transpose(self.read_v_weight, 0, 1)), [B, T, D])

        # --- Total gradients ---
        grad_x = ttnn.add(ttnn.add(grad_x_from_write, grad_x_from_wq), ttnn.add(grad_x_from_rk, grad_x_from_rv))
        grad_slots_in = ttnn.add(grad_slots_in_from_read, grad_slots_from_rq)

        # If slots_in was from learned slot embeddings (slot_state was None), accumulate grad into slots param
        # This is handled by the caller (model.backward) via grad_slots_in

        grads = {
            "read_q_weight": grad_read_q_weight,
            "read_k_weight": grad_read_k_weight,
            "read_v_weight": grad_read_v_weight,
            "read_out_weight": grad_read_out_weight,
            "write_q_weight": grad_write_q_weight,
            "write_k_weight": grad_write_k_weight,
            "write_v_weight": grad_write_v_weight,
            "write_out_weight": grad_write_out_weight,
            "ws_norm_weight": grad_norm_w,
            "ws_slot_norm_weight": grad_slot_norm_w,
            "read_gate": grad_read_gate,
            "write_gate": grad_write_gate,
        }

        return grad_x, grad_slots_in, grads

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        return {
            "slots": self.slots,
            "read_q_weight": self.read_q_weight, "read_k_weight": self.read_k_weight,
            "read_v_weight": self.read_v_weight, "read_out_weight": self.read_out_weight,
            "write_q_weight": self.write_q_weight, "write_k_weight": self.write_k_weight,
            "write_v_weight": self.write_v_weight, "write_out_weight": self.write_out_weight,
            "ws_norm_weight": self.norm.weight, "ws_slot_norm_weight": self.slot_norm.weight,
            "read_gate": self.read_gate, "write_gate": self.write_gate,
        }

    def normalize_slots(self):
        """Normalize slot parameters to unit RMS to prevent unbounded growth.

        This breaks the feedback loop where large slots → sharp attention →
        large gradients → larger slots. Called after each optimizer step.
        Uses fixed weight=1 (not learned) so the slots are constrained to
        unit RMS regardless of training dynamics.
        """
        self.slots = ttnn.rms_norm(self.slots, weight=self._slot_param_norm_weight, epsilon=1e-6)

    def spectral_normalize_weights(self):
        """Apply spectral normalization to all workspace weight matrices.

        Divides each weight matrix by its spectral norm (largest singular value),
        constraining the Lipschitz constant to 1. This prevents unbounded weight
        growth — the root cause of attention sharpening and gradient explosion.

        Unlike logit clamping (which caps the symptom), spectral norm addresses
        the cause: the weight matrices cannot grow beyond spectral norm 1, so
        the attention logits QK^T/√d are bounded by construction:
            |logit| <= ||W_Q||_2 * ||W_K||_2 * ||slots|| * ||x|| / √d_h
                     <= 1 * 1 * 1 * 1 / √d_h  (with slot norm + LayerNorm)
                     = 1/√d_h ≈ 0.07

        The model can still learn the *direction* of each weight matrix (which
        determines what it attends to), just not the *magnitude* (which
        determines attention sharpness).

        Uses power iteration (10 steps) for on-device computation — no host
        transfer needed. This is the same algorithm used by Spectral Normalization
        GANs (Miyato et al., 2018).
        """
        device = self.device
        weight_names = [
            "read_q_weight", "read_k_weight", "read_v_weight", "read_out_weight",
            "write_q_weight", "write_k_weight", "write_v_weight", "write_out_weight",
        ]

        for name in weight_names:
            W = getattr(self, name)  # (D, D) ttnn tensor
            D = self.d_model

            # Power iteration to estimate spectral norm (largest singular value)
            # v converges to the top right singular vector of W
            # σ_max ≈ ||W @ v|| after convergence
            v = ttnn.from_torch(
                torch.randn(D, 1, dtype=torch.bfloat16) / math.sqrt(D),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            for _ in range(10):
                # u = W @ v, then normalize
                u = ttnn.matmul(W, v)  # (D, 1)
                u_norm = ttnn.sqrt(ttnn.sum(ttnn.mul(u, u)))  # scalar
                u_norm = ttnn.maximum(u_norm, ttnn.from_torch(
                    torch.tensor([1e-6], dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                ))
                u = ttnn.mul(u, ttnn.reciprocal(u_norm))

                # v = W^T @ u, then normalize
                v = ttnn.matmul(ttnn.transpose(W, 0, 1), u)  # (D, 1)
                v_norm = ttnn.sqrt(ttnn.sum(ttnn.mul(v, v)))
                v_norm = ttnn.maximum(v_norm, ttnn.from_torch(
                    torch.tensor([1e-6], dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                ))
                v = ttnn.mul(v, ttnn.reciprocal(v_norm))

            # σ_max = ||W @ v|| (v is now the top right singular vector)
            Wv = ttnn.matmul(W, v)  # (D, 1)
            sigma = ttnn.sqrt(ttnn.sum(ttnn.mul(Wv, Wv)))
            # Clamp sigma to avoid division by zero
            sigma = ttnn.maximum(sigma, ttnn.from_torch(
                torch.tensor([1e-6], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            ))

            # W_normalized = W / sigma
            W_normalized = ttnn.mul(W, ttnn.reciprocal(sigma))
            setattr(self, name, W_normalized)

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        key_map = {
            "slots": "slots", "read_q_weight": "read_q_weight", "read_k_weight": "read_k_weight",
            "read_v_weight": "read_v_weight", "read_out_weight": "read_out_weight",
            "write_q_weight": "write_q_weight", "write_k_weight": "write_k_weight",
            "write_v_weight": "write_v_weight", "write_out_weight": "write_out_weight",
            "ws_norm_weight": "norm", "ws_slot_norm_weight": "slot_norm",
            "read_gate": "read_gate", "write_gate": "write_gate",
        }
        for k, v in params.items():
            if k == "ws_norm_weight":
                self.norm.weight = v
            elif k == "ws_slot_norm_weight":
                self.slot_norm.weight = v
            elif hasattr(self, k):
                setattr(self, k, v)


class TTMambaWorkspaceModel:
    """Full model using tt-nn operations.

    Supports Cell A (pure Mamba2), Cell B (Mamba2 + attention),
    Cell C (Mamba2 + attention + workspace), and Cell D (+ recurrent core).
    """

    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device

        # Embedding: (vocab_size, d_model) — for ttnn.embedding this is the weight
        # For LM head (weight-tied), ttnn.linear(x, W) needs W=(d_model, vocab_size)
        # So we store the embedding as (vocab_size, d_model) and transpose for LM head
        emb_w = torch.randn(config.vocab_size, config.d_model, dtype=torch.bfloat16) * 0.02
        self.token_emb_weight = to_device(emb_w, device)

        # Build layers with gated residual wrappers
        self.layers = []
        self.attention_positions = set(config.attention_positions) if config.use_attention else set()
        for i in range(config.n_layers):
            if config.use_attention and i in config.attention_positions:
                attn_layer = TTAttentionLayer(config, device)
                wrapped = TTGatedResidualLayer(attn_layer, config.d_model, device)
                self.layers.append(wrapped)
            else:
                mamba_layer = TTMamba2Layer(config, device)
                wrapped = TTGatedResidualLayer(mamba_layer, config.d_model, device)
                self.layers.append(wrapped)

        # Workspace module (Cell C/D)
        if config.use_workspace:
            self.workspace = TTWorkspaceModule(config, device)
        else:
            self.workspace = None

        # Final norm and LM head (weight-tied with embedding)
        self.norm = TTRMSNorm(config.d_model, device)
        self.lm_head_weight = self.token_emb_weight  # weight tying

        # Cache for identity matrix (used in embedding backward)
        self._identity_tt = None

    def forward(self, input_ids: torch.Tensor, k_value: int = None) -> "ttnn.Tensor":
        """
        input_ids: (B, T) PyTorch tensor of int indices
        k_value: int or None — number of active recurrent core iterations (Cell D).
                 If None and recurrent_core, uses k_train_max (all active for simplicity).
        Returns: (B, T, vocab_size) tt-nn tensor of logits
        """
        device = self.device
        config = self.config
        B, T = input_ids.shape

        # Cache input_ids for backward
        self._cached_input_ids = input_ids

        # Embedding lookup
        indices = ttnn.from_torch(
            input_ids.to(torch.int32),
            dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
        )
        x = ttnn.embedding(indices, self.token_emb_weight, layout=ttnn.TILE_LAYOUT)  # (B, T, d_model)

        # Determine K for recurrent core
        if config.recurrent_core:
            k_max = config.k_train_max
            K = k_value if k_value is not None else k_max
        else:
            k_max = 0
            K = 0

        # Track all forward calls for backward (in order)
        # Each entry: ("pre", layer_idx, ws_was_none) or ("core", iter, layer_idx, ws_was_none) or ("post", layer_idx, ws_was_none)
        # Plus blend info for recurrent core
        self._fwd_trace = []
        slot_state = None

        # --- Phase 1: pre-core layers (0..core_start-1) ---
        pre_core_end = config.core_start if config.recurrent_core else config.n_layers
        for i in range(pre_core_end):
            if self.workspace is not None and i in self.attention_positions:
                was_none = (slot_state is None)
                x, slot_state = self.workspace.forward(x, slot_state)
                self._fwd_trace.append(("ws", "pre", i, was_none))
            x = self.layers[i].forward(x)
            self._fwd_trace.append(("layer", "pre", i))

        # --- Phase 2: recurrent core (core_start..core_end-1) ---
        if config.recurrent_core:
            core_layer_indices = list(range(config.core_start, config.core_end))
            self._core_blend_info = []  # (iter, active_scalar_tensor, x_before, slot_before, x_new, slot_new)

            # 1/sqrt(K) residual scaling: normalizes gradient accumulation across K
            # iterations so that the total gradient norm is independent of K.
            # Without this, gradients from K iterations sum to ~K*sigma, causing
            # instability at large K. With 1/sqrt(K) scaling, they sum to ~sqrt(K)*sigma.
            # The blend factor becomes active/sqrt(K) instead of active, turning the
            # full replacement (x = x_new) into a partial update:
            #   x = (1 - active/sqrt(K)) * x + (active/sqrt(K)) * x_new
            import math
            k_scale = 1.0 / math.sqrt(K) if K > 0 else 0.0

            for iteration in range(k_max):
                active = 1.0 if iteration < K else 0.0
                # Scale the blend factor by 1/sqrt(K) for gradient normalization
                blend_factor = active * k_scale
                active_tt = ttnn.from_torch(
                    torch.tensor([blend_factor], dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )

                # Run one core iteration
                x_new = x
                slot_state_new = slot_state
                for i in core_layer_indices:
                    if self.workspace is not None and i in self.attention_positions:
                        was_none = (slot_state_new is None)
                        x_new, slot_state_new = self.workspace.forward(x_new, slot_state_new)
                        self._fwd_trace.append(("ws", "core", iteration, i, was_none))
                    x_new = self.layers[i].forward(x_new)
                    self._fwd_trace.append(("layer", "core", iteration, i))

                # Final workspace at end of core iteration
                if self.workspace is not None:
                    was_none = (slot_state_new is None)
                    x_new, slot_state_new = self.workspace.forward(x_new, slot_state_new)
                    self._fwd_trace.append(("ws", "core_end", iteration, -1, was_none))

                # Blend: active iterations partially update state (1/sqrt(K) scaling),
                # padding iterations keep old state.
                if slot_state is None:
                    # First iteration is always active (K >= 1)
                    # Use scaled blend: x = blend * x_new + (1 - blend) * x
                    x = ttnn.add(ttnn.mul(active_tt, x_new), ttnn.mul(ttnn.sub(
                        ttnn.from_torch(torch.tensor([1.0], dtype=torch.bfloat16),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                        active_tt), x))
                    slot_state = ttnn.add(ttnn.mul(active_tt, slot_state_new),
                                          ttnn.mul(ttnn.sub(
                                              ttnn.from_torch(torch.tensor([1.0], dtype=torch.bfloat16),
                                                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                                              active_tt), slot_state_new))
                else:
                    # x = blend * x_new + (1 - blend) * x
                    x = ttnn.add(ttnn.mul(active_tt, x_new), ttnn.mul(ttnn.sub(
                        ttnn.from_torch(torch.tensor([1.0], dtype=torch.bfloat16),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                        active_tt), x))
                    slot_state = ttnn.add(ttnn.mul(active_tt, slot_state_new),
                                          ttnn.mul(ttnn.sub(
                                              ttnn.from_torch(torch.tensor([1.0], dtype=torch.bfloat16),
                                                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                                              active_tt), slot_state))

                self._core_blend_info.append({
                    "iteration": iteration, "active": active,
                    "active_tt": active_tt,
                })

        # --- Phase 3: post-core layers (core_end..n_layers-1) ---
        post_core_start = config.core_end if config.recurrent_core else config.n_layers
        for i in range(post_core_start, config.n_layers):
            if self.workspace is not None and i in self.attention_positions:
                was_none = (slot_state is None)
                x, slot_state = self.workspace.forward(x, slot_state)
                self._fwd_trace.append(("ws", "post", i, was_none))
            x = self.layers[i].forward(x)
            self._fwd_trace.append(("layer", "post", i))

        # Cache for backward
        self._cached_x_pre_final_norm = x

        # Final norm
        x = self.norm.forward(x)

        # LM head (weight-tied: weight = embedding weight transposed)
        lm_head_w = ttnn.transpose(self.lm_head_weight, 0, 1)  # (d_model, vocab_size)
        logits = ttnn.linear(x, lm_head_w)  # (B, T, vocab_size)

        return logits

    def backward(self, grad_logits: "ttnn.Tensor") -> Dict[str, "ttnn.Tensor"]:
        """Backward pass through the full model — fully on device.

        grad_logits: (B, T, vocab_size) — gradient w.r.t. logits
        Returns: dict of parameter gradients
        """
        device = self.device
        config = self.config

        # --- Backward through LM head (weight-tied with embedding) ---
        # logits = x_normed @ emb^T, so:
        # grad_x_normed = grad_logits @ emb  (already on device)
        # grad_emb (from LM head) = x_normed^T @ grad_logits  (on device via matmul)
        emb_w = self.lm_head_weight  # (vocab_size, d_model)
        grad_x_normed = ttnn.linear(grad_logits, emb_w)  # (B, T, d_model)

        # LM head weight gradient on device: grad_lm_head = x_pre^T @ grad_logits
        # x_pre is (B, T, d), grad_logits is (B, T, V)
        # Need to reshape to 2D for matmul: (B*T, d)^T @ (B*T, V) = (d, V)
        x_pre = self._cached_x_pre_final_norm  # (B, T, d_model) on device
        B, T = x_pre.shape[0], x_pre.shape[1]
        x_pre_2d = ttnn.reshape(x_pre, [B * T, config.d_model])  # (B*T, d)
        grad_logits_2d = ttnn.reshape(grad_logits, [B * T, config.vocab_size])  # (B*T, V)
        # grad_lm_head = x_pre_2d^T @ grad_logits_2d = (d, B*T) @ (B*T, V) = (d, V)
        grad_lm_head = ttnn.matmul(ttnn.transpose(x_pre_2d, 0, 1), grad_logits_2d)  # (d, V)

        # --- Backward through final RMS norm (on device using TTRMSNorm.backward) ---
        grad_x, grad_norm_w = self.norm.backward(grad_x_normed, x_pre)

        all_grads = {}

        # Helper to accumulate layer grads (shared params across core iterations accumulate)
        def accum_layer_grads(layer_idx, grads_dict):
            for k, v in grads_dict.items():
                key = f"layer_{layer_idx}_{k}"
                if key in all_grads:
                    all_grads[key] = ttnn.add(all_grads[key], v)
                else:
                    all_grads[key] = v

        def accum_ws_grads(grads_dict):
            for k, v in grads_dict.items():
                key = f"ws_{k}"
                if key in all_grads:
                    all_grads[key] = ttnn.add(all_grads[key], v)
                else:
                    all_grads[key] = v

        # --- Backward through the forward trace in reverse ---
        # We need to handle the recurrent core's blend operation.
        # The trace is a list of ("layer", phase, idx) and ("ws", phase, ...) entries.
        # For the core, layers are shared across iterations, so grads accumulate.

        # Process post-core and pre-core layers (simple reverse)
        # For core, we need to handle blend backward then unroll iterations in reverse.

        # Separate the trace into phases
        trace = self._fwd_trace
        recurrent = config.recurrent_core

        # Find phase boundaries in the trace
        pre_trace = []
        core_trace = []
        post_trace = []
        current_phase = "pre"
        for entry in trace:
            if entry[0] == "layer":
                phase = entry[1]
                if phase == "pre":
                    pre_trace.append(entry)
                elif phase == "core":
                    core_trace.append(entry)
                elif phase == "post":
                    post_trace.append(entry)
            elif entry[0] == "ws":
                phase = entry[1]
                if phase == "pre":
                    pre_trace.append(entry)
                elif phase in ("core", "core_end"):
                    core_trace.append(entry)
                elif phase == "post":
                    post_trace.append(entry)

        # --- Backward through post-core layers (reverse) ---
        grad_slot_state = None  # gradient w.r.t. slot_state
        ones_tt = ttnn.from_torch(torch.tensor([1.0], dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        for entry in reversed(post_trace):
            if entry[0] == "layer":
                idx = entry[2]
                grad_x, layer_grads = self.layers[idx].backward(grad_x)
                accum_layer_grads(idx, layer_grads)
            elif entry[0] == "ws":
                idx = entry[2]
                was_none = entry[3]
                if grad_slot_state is None:
                    B = int(grad_x.shape[0])
                    m = self.workspace.n_slots
                    grad_slot_state = ttnn.zeros((B, m, config.d_model), dtype=ttnn.bfloat16,
                                                  layout=ttnn.TILE_LAYOUT, device=device)
                grad_x, grad_slots_in, ws_grads = self.workspace.backward(grad_x, grad_slot_state)
                accum_ws_grads(ws_grads)
                if was_none:
                    grad_slots_param = ttnn.sum(grad_slots_in, dim=0)
                    if "ws_slots" in all_grads:
                        all_grads["ws_slots"] = ttnn.add(all_grads["ws_slots"], grad_slots_param)
                    else:
                        all_grads["ws_slots"] = grad_slots_param
                    grad_slot_state = None
                else:
                    grad_slot_state = grad_slots_in

        # --- Backward through recurrent core (reverse iterations) ---
        if recurrent:
            blend_info = self._core_blend_info
            # Group core_trace by iteration
            # Each iteration has: ws calls for core layers, layer calls, ws at core_end
            # We need to unroll in reverse iteration order

            # Split core_trace into per-iteration segments
            iter_segments = []  # list of (iter_num, entries)
            current_iter = -1
            current_entries = []
            for entry in core_trace:
                if entry[0] == "ws":
                    phase = entry[1]
                    if phase == "core":
                        iter_num = entry[2]
                    elif phase == "core_end":
                        iter_num = entry[2]
                    if iter_num != current_iter:
                        if current_entries:
                            iter_segments.append((current_iter, current_entries))
                        current_iter = iter_num
                        current_entries = [entry]
                    else:
                        current_entries.append(entry)
                elif entry[0] == "layer":
                    iter_num = entry[2]
                    if iter_num != current_iter:
                        if current_entries:
                            iter_segments.append((current_iter, current_entries))
                        current_iter = iter_num
                        current_entries = [entry]
                    else:
                        current_entries.append(entry)
            if current_entries:
                iter_segments.append((current_iter, current_entries))

            # Unroll in reverse iteration order
            for iter_num, entries in reversed(iter_segments):
                blend = blend_info[iter_num]
                active = blend["active"]
                active_tt = blend["active_tt"]

                # All iterations (including the first) use the scaled blend:
                # x = blend * x_new + (1 - blend) * x_old
                # grad_x_new = blend * grad_x  (flows into core layers)
                # grad_x_old = (1 - blend) * grad_x  (flows to previous iteration or pre-core)
                grad_x_new = ttnn.mul(active_tt, grad_x)
                grad_x_old = ttnn.mul(ttnn.sub(ones_tt, active_tt), grad_x)
                grad_slot_new = ttnn.mul(active_tt, grad_slot_state) if grad_slot_state is not None else None
                grad_slot_old = ttnn.mul(ttnn.sub(ones_tt, active_tt), grad_slot_state) if grad_slot_state is not None else None
                # grad_x for the core iteration = grad_x_new
                # grad_x_old accumulates into the previous iteration's output
                grad_x = grad_x_new
                if grad_slot_old is not None:
                    grad_slot_state = grad_slot_new
                    # grad_slot_old will be added to the previous iteration's grad
                    self._pending_grad_slot_old = grad_slot_old
                else:
                    grad_slot_state = None
                    self._pending_grad_slot_old = None

                # Backward through this iteration's entries (reverse)
                for entry in reversed(entries):
                    if entry[0] == "layer":
                        idx = entry[3]
                        grad_x, layer_grads = self.layers[idx].backward(grad_x)
                        accum_layer_grads(idx, layer_grads)
                    elif entry[0] == "ws":
                        ws_iter = entry[2]
                        idx = entry[3]
                        was_none = entry[4]
                        if grad_slot_state is None:
                            B = int(grad_x.shape[0])
                            m = self.workspace.n_slots
                            grad_slot_state = ttnn.zeros((B, m, config.d_model), dtype=ttnn.bfloat16,
                                                          layout=ttnn.TILE_LAYOUT, device=device)
                        grad_x, grad_slots_in, ws_grads = self.workspace.backward(grad_x, grad_slot_state)
                        accum_ws_grads(ws_grads)
                        if was_none:
                            grad_slots_param = ttnn.sum(grad_slots_in, dim=0)
                            if "ws_slots" in all_grads:
                                all_grads["ws_slots"] = ttnn.add(all_grads["ws_slots"], grad_slots_param)
                            else:
                                all_grads["ws_slots"] = grad_slots_param
                            grad_slot_state = None
                        else:
                            grad_slot_state = grad_slots_in

                # After processing this iteration, add grad_x_old from blend.
                # For iter_num > 0, grad_x_old flows to the previous iteration.
                # For iter_num == 0, grad_x_old flows to the pre-core layers
                # (it will be picked up by the pre-core backward below).
                grad_x = ttnn.add(grad_x, grad_x_old)
                if hasattr(self, '_pending_grad_slot_old') and self._pending_grad_slot_old is not None:
                    if grad_slot_state is None:
                        grad_slot_state = self._pending_grad_slot_old
                    else:
                        grad_slot_state = ttnn.add(grad_slot_state, self._pending_grad_slot_old)
                self._pending_grad_slot_old = None

        # --- Backward through pre-core layers (reverse) ---
        for entry in reversed(pre_trace):
            if entry[0] == "layer":
                idx = entry[2]
                grad_x, layer_grads = self.layers[idx].backward(grad_x)
                accum_layer_grads(idx, layer_grads)
            elif entry[0] == "ws":
                idx = entry[2]
                was_none = entry[3]
                if grad_slot_state is None:
                    B = int(grad_x.shape[0])
                    m = self.workspace.n_slots
                    grad_slot_state = ttnn.zeros((B, m, config.d_model), dtype=ttnn.bfloat16,
                                                  layout=ttnn.TILE_LAYOUT, device=device)
                grad_x, grad_slots_in, ws_grads = self.workspace.backward(grad_x, grad_slot_state)
                accum_ws_grads(ws_grads)
                if was_none:
                    grad_slots_param = ttnn.sum(grad_slots_in, dim=0)
                    if "ws_slots" in all_grads:
                        all_grads["ws_slots"] = ttnn.add(all_grads["ws_slots"], grad_slots_param)
                    else:
                        all_grads["ws_slots"] = grad_slots_param
                    grad_slot_state = None
                else:
                    grad_slot_state = grad_slots_in

        # --- Backward through embedding (on device via one-hot matmul) ---
        # grad_emb = one_hot(input_ids)^T @ grad_x + grad_lm_head^T
        # one_hot: (B*T, V), grad_x: (B*T, d) → grad_emb: (V, d)
        input_ids = self._cached_input_ids
        V = config.vocab_size
        B, T = input_ids.shape
        flat_ids = input_ids.reshape(-1).to(torch.int32)  # (B*T,)

        # Create one-hot via embedding lookup with identity matrix (cached)
        if self._identity_tt is None:
            identity = torch.eye(V, dtype=torch.bfloat16)
            self._identity_tt = ttnn.from_torch(identity, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        identity_tt = self._identity_tt
        label_indices = ttnn.from_torch(flat_ids.unsqueeze(-1), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        one_hot = ttnn.embedding(label_indices, identity_tt, layout=ttnn.TILE_LAYOUT)  # (B*T, 1, V)
        one_hot = ttnn.reshape(one_hot, [B * T, V])  # (B*T, V)

        # grad_x: (B, T, d) → (B*T, d)
        grad_x_2d = ttnn.reshape(grad_x, [B * T, config.d_model])

        # grad_emb = one_hot^T @ grad_x_2d = (V, B*T) @ (B*T, d) = (V, d)
        grad_emb = ttnn.matmul(ttnn.transpose(one_hot, 0, 1), grad_x_2d)  # (V, d)

        # Add LM head gradient (weight-tied): grad_emb += grad_lm_head^T
        # grad_lm_head is (d, V), so grad_lm_head^T is (V, d)
        grad_emb = ttnn.add(grad_emb, ttnn.transpose(grad_lm_head, 0, 1))

        all_grads["token_emb_weight"] = grad_emb
        all_grads["norm_weight"] = grad_norm_w

        return all_grads

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        params = {"token_emb_weight": self.token_emb_weight, "norm_weight": self.norm.weight}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for k, v in layer_params.items():
                params[f"layer_{i}_{k}"] = v
        if self.workspace is not None:
            for k, v in self.workspace.get_params().items():
                params[f"ws_{k}"] = v
        return params

    def get_num_params(self) -> int:
        total = 0
        for t in self.get_params().values():
            t_host = ttnn.to_torch(t)
            total += t_host.numel()
        return total

    def normalize_workspace_slots(self):
        """Normalize workspace parameters after each optimizer step.

        Applies two constraints:
        1. Slot parameter normalization: constrains slot vectors to unit RMS
        2. Spectral normalization: constrains all 8 weight matrices to spectral norm 1

        Together these prevent unbounded weight/slot growth, which causes
        attention sharpening and gradient explosion.
        No-op if the model has no workspace.
        """
        if self.workspace is not None:
            self.workspace.normalize_slots()
            self.workspace.spectral_normalize_weights()

    def save_checkpoint(self, path: str, optimizer_state: dict = None, step: int = 0):
        """Save model checkpoint to a PyTorch state dict file.

        Converts all tt-nn tensors to PyTorch and saves as a .pt file.
        """
        checkpoint = {
            "step": step,
            "config": {
                "d_model": self.config.d_model,
                "d_state": self.config.d_state,
                "d_conv": self.config.d_conv,
                "expand": self.config.expand,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "vocab_size": self.config.vocab_size,
            },
            "model_state": {},
        }

        for name, tt_tensor in self.get_params().items():
            checkpoint["model_state"][name] = ttnn.to_torch(tt_tensor).clone()

        if optimizer_state is not None:
            checkpoint["optimizer_state"] = optimizer_state

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path} (step {step})", flush=True)

    def load_checkpoint(self, path: str, device=None) -> dict:
        """Load model checkpoint from a .pt file.

        Returns the optimizer state if present, else None.
        """
        if device is None:
            device = self.device

        checkpoint = torch.load(path, weights_only=False)
        model_state = checkpoint["model_state"]

        # Restore config if needed
        saved_config = checkpoint.get("config", {})
        if saved_config:
            print(f"Loaded checkpoint from {path} (step {checkpoint.get('step', 0)})", flush=True)

        for name, host_tensor in model_state.items():
            if name == "token_emb_weight":
                dtype = ttnn.bfloat16
            elif "A_log" in name or name.endswith("_D"):
                dtype = ttnn.float32
            else:
                dtype = ttnn.bfloat16
            tt_tensor = ttnn.from_torch(
                host_tensor.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
                dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            self._set_param(name, tt_tensor)

        return checkpoint.get("optimizer_state", None)

    def _set_param(self, name: str, tt_tensor: "ttnn.Tensor"):
        """Set a single parameter by name."""
        if name == "token_emb_weight":
            self.token_emb_weight = tt_tensor
            self.lm_head_weight = tt_tensor
        elif name == "norm_weight":
            self.norm.weight = tt_tensor
        elif name.startswith("layer_"):
            parts = name.split("_", 2)
            layer_idx = int(parts[1])
            param_name = parts[2]
            self.layers[layer_idx].set_params({param_name: tt_tensor})
        elif name.startswith("ws_") and self.workspace is not None:
            ws_param_name = name[3:]  # strip "ws_" prefix
            self.workspace.set_params({ws_param_name: tt_tensor})
        else:
            print(f"WARNING: Unknown param name {name}")


# ---------------------------------------------------------------------------
# Cell configurations (shared with model.py)
# ---------------------------------------------------------------------------

def get_cell_config(cell: str) -> ModelConfig:
    if cell == "A":
        return ModelConfig(
            n_layers=14,
            use_attention=False,
            use_workspace=False,
            recurrent_core=False,
        )
    elif cell == "B":
        return ModelConfig(
            n_layers=14,
            use_attention=True,
            attention_positions=[5, 10],
            use_workspace=False,
            recurrent_core=False,
        )
    elif cell == "C":
        return ModelConfig(
            n_layers=13,
            use_attention=True,
            attention_positions=[5, 10],
            use_workspace=True,
            n_workspace_slots=16,
            recurrent_core=False,
        )
    elif cell == "D":
        return ModelConfig(
            n_layers=13,
            use_attention=True,
            attention_positions=[5, 10],
            use_workspace=True,
            n_workspace_slots=16,
            recurrent_core=True,
            core_start=6,
            core_end=10,
            k_train_max=6,
            k_inference=6,
        )
    elif cell == "E":
        # Control: B architecture (Mamba2 + attention) with C/D learning rate
        return ModelConfig(
            n_layers=14,
            use_attention=True,
            attention_positions=[5, 10],
            use_workspace=False,
            recurrent_core=False,
        )
    else:
        raise ValueError(f"Unknown cell: {cell}")
