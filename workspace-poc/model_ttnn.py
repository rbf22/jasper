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

# Custom kernel support (fused RoPE for retention layer)
from ttnn._ttnn.program_descriptor import VectorUInt32

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


# ---------------------------------------------------------------------------
# Memory management helper
# ---------------------------------------------------------------------------

def _safe_deallocate(tensor):
    """Deallocate a ttnn device tensor, ignoring errors.

    Without explicit deallocation, intermediate tensors from every ttnn op
    accumulate in device DRAM until Python GC runs — but ttnn wrapper objects
    are tiny so GC doesn't see the memory pressure.  This was the root cause
    of the OOM that killed the system when running 3 training processes.
    """
    if tensor is None:
        return
    try:
        ttnn.deallocate(tensor)
    except Exception:
        pass


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
    k_train_max: int = 6                # max K during training (restored to 6 after slot gradient fix)
    k_inference: int = 6                # K at inference (sweep this for R2)
    attention_residual_core: bool = False  # use Attention Residuals (Kimi K3) instead of fixed blend for recurrent core
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True
    spectral_norm_bound: float = 5.0
    backbone_spectral_norm_bound: float = 2.0  # cap on qkv/out_proj spectral norms
    chain_scale_safety: float = 1.0  # extra safety factor for recurrent core gradient chain (1/K*safety)
    freeze_gamma: bool = True            # freeze retention gamma (don't train it) — eliminates O(T²) gradient instability
    freeze_slot_decay: bool = False      # freeze slot_decay (prevents slot chain growth towards 1.0)
    ws_entropy_weight: float = 0.0       # weight for attention entropy regularizer (DISABLED — measured harmful)
    ws_diversity_weight: float = 0.0     # weight for slot diversity regularizer (DISABLED — measured harmful)
    gate_init: float = 0.0              # ReZero gate init (0.0 = true identity, no sigmoid). Previously -2.0 for sigmoid gates.
    slot_decay_init: float = 1.0        # slot decay factor init (1.0 = no decay, <1.0 = forgetful)
    slot_permutation: bool = False      # randomly permute slot indices each forward pass (breaks fixed routing)
    gate_schedule_steps: int = 0        # >0: anneal gates from gate_init to 0 over this many steps (no-op with ReZero)
    gate_clamp_bound: float = 0.0       # >0: clamp ReZero gates to [-bound, bound] after each optimizer step. Prevents RMSNorm cancellation when gates grow large negative (causal masking makes workspace contribution noise → model suppresses it → gate grows negative → pre-norm residual cancels → 1/RMS amplifies gradient). 0.0 = disabled (backward compat).
    # --- Mamba-3 MIMO config ---
    headdim: int = 64                   # SSM head dimension (d_inner // headdim = nheads)
    d_state_m3: int = 64                # SSM state size for Mamba-3 (can differ from d_state)
    mimo_rank: int = 4                  # MIMO rank R (parallel SSMs per head)
    rope_fraction: float = 0.5          # fraction of d_state to apply RoPE to
    ngroups: int = 1                    # number of BC heads (1 = shared B/C across all heads)
    # --- Kimi K3 architectural updates ---
    short_conv: bool = False            # depthwise causal conv1d (kernel=3) before QKV in retention
    short_conv_kernel: int = 3          # conv kernel size (3 = standard in KDA/Mamba2)
    per_channel_decay: bool = False     # per-channel gamma: (n_heads, d_head) instead of (n_heads,)

    @property
    def d_inner(self):
        return self.d_model * self.expand

    @property
    def d_head(self):
        return self.d_inner // self.n_heads

    @property
    def nheads_m3(self):
        """Number of SSM heads for Mamba-3 (d_inner // headdim)."""
        return self.d_inner // self.headdim

    @property
    def num_rope_angles(self):
        """Number of RoPE angles for Mamba-3 complex SSM."""
        split = int(self.d_state_m3 * self.rope_fraction)
        if split % 2 != 0:
            split -= 1
        return split // 2


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
            grad_out: (..., d) gradient w.r.t. output
            x: (..., d) pre-norm input (cached from forward)

        Returns:
            grad_x: (..., d) gradient w.r.t. input
            grad_weight: (d,) gradient w.r.t. weight
        """
        device = self.device
        d = self.d
        eps = self.eps

        # Compute rms = sqrt(eps + mean(x^2)) along last dim
        # REVIEWED: RMS norm backward is called for every norm in the model
        # (final norm, workspace norm, slot norm). All intermediates must be
        # explicitly deallocated to avoid compounding leaks.
        x_sq = ttnn.mul(x, x)
        x_sq_mean = ttnn.mean(x_sq, dim=-1)  # (...,) — drops last dim
        _safe_deallocate(x_sq)
        # Reshape to (..., 1) for broadcasting via mul
        mean_shape = list(int(x_sq_mean.shape[i]) for i in range(len(x_sq_mean.shape))) + [1]
        _xsm_eps = ttnn.add(x_sq_mean, eps)
        _safe_deallocate(x_sq_mean)
        inv_rms = ttnn.rsqrt(_xsm_eps)  # (...,)
        _safe_deallocate(_xsm_eps)
        # REVIEWED: reshape may return a view — do NOT deallocate inv_rms.
        inv_rms_b = ttnn.reshape(inv_rms, mean_shape)  # (..., 1)

        # x_normed = x * inv_rms (broadcast over last dim)
        x_normed = ttnn.mul(x, inv_rms_b)

        # grad_out * weight (broadcast weight across all leading dims)
        w_shape = [1] * (len(grad_out.shape) - 1) + [d]
        # REVIEWED: reshape of persistent self.weight may return a view — do NOT deallocate w.
        w = ttnn.reshape(self.weight, w_shape)
        grad_out_w = ttnn.mul(grad_out, w)

        # grad_out_w / rms = grad_out_w * inv_rms
        grad_out_w_rms = ttnn.mul(grad_out_w, inv_rms_b)

        # mean(grad_out_w * x_normed) along last dim
        grad_out_w_xnorm = ttnn.mul(grad_out_w, x_normed)
        _safe_deallocate(grad_out_w)
        grad_out_w_xnorm_mean = ttnn.mean(grad_out_w_xnorm, dim=-1)  # (...,)
        _safe_deallocate(grad_out_w_xnorm)
        # REVIEWED: reshape may return a view — do NOT deallocate grad_out_w_xnorm_mean.
        grad_out_w_xnorm_mean_b = ttnn.reshape(grad_out_w_xnorm_mean, mean_shape)  # (..., 1)

        # x_normed * (grad_out_w_xnorm_mean / rms)
        _inner = ttnn.mul(grad_out_w_xnorm_mean_b, inv_rms_b)
        correction = ttnn.mul(x_normed, _inner)
        _safe_deallocate(_inner)

        # grad_x = grad_out_w_rms - correction
        grad_x = ttnn.sub(grad_out_w_rms, correction)
        _safe_deallocate(grad_out_w_rms)
        _safe_deallocate(correction)

        # grad_weight = sum(grad_out * x_normed, over all leading dims)
        grad_weight_full = ttnn.mul(grad_out, x_normed)
        _safe_deallocate(x_normed)
        # Sum over all dims except last
        # REVIEWED: nested sum reassignment leak — deallocate intermediates.
        for _ in range(len(grad_weight_full.shape) - 1):
            _gw = ttnn.sum(grad_weight_full, dim=0)
            _safe_deallocate(grad_weight_full)
            grad_weight_full = _gw
        grad_weight = grad_weight_full

        # REVIEWED: inv_rms_b may share buffer with inv_rms — deallocate inv_rms
        # only after inv_rms_b is no longer needed (it's not used after here).
        _safe_deallocate(inv_rms)
        _safe_deallocate(inv_rms_b)

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
# Attention layer (used at configured positions in the backbone)
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

        # Cache scale as a device tensor (avoids per-forward torch→device transfer)
        # REVIEWED: from_torch in the hot path was leaking device tensors every
        # forward call — the old scale_tt was cached in self._cache but a NEW
        # one was created each time, orphaning the previous one on device.
        self._scale_tt = ttnn.from_torch(
            torch.tensor([self.scale], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

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
        # Causal mask caches (depend only on T, reused across forward/backward)
        self._causal_mask_upper = None  # triu(ones, 1) — for forward (s > t → -inf)
        self._causal_mask_lower = None  # tril(ones, 0) — for backward (s <= t → keep)

        self._cache = {}
        self._cache_history = []  # old caches from recurrent core iterations

    def _deallocate_cache(self):
        """Deallocate cached intermediates, excluding persistent tensors."""
        if not self._cache:
            return
        _persistent = set()
        if hasattr(self, '_scale_tt') and self._scale_tt is not None:
            _persistent.add(id(self._scale_tt))
        if hasattr(self, '_causal_mask_upper') and self._causal_mask_upper is not None:
            _persistent.add(id(self._causal_mask_upper))
        if hasattr(self, '_causal_mask_lower') and self._causal_mask_lower is not None:
            _persistent.add(id(self._causal_mask_lower))
        for v in self._cache.values():
            if id(v) not in _persistent and hasattr(v, 'shape'):
                _safe_deallocate(v)
        self._cache = {}

    def _deallocate_cache_history(self):
        """Deallocate all historical caches (from recurrent core iterations)."""
        for old_cache in self._cache_history:
            _persistent = set()
            if hasattr(self, '_scale_tt') and self._scale_tt is not None:
                _persistent.add(id(self._scale_tt))
            if hasattr(self, '_causal_mask_upper') and self._causal_mask_upper is not None:
                _persistent.add(id(self._causal_mask_upper))
            if hasattr(self, '_causal_mask_lower') and self._causal_mask_lower is not None:
                _persistent.add(id(self._causal_mask_lower))
            for v in old_cache.values():
                if id(v) not in _persistent and hasattr(v, 'shape'):
                    _safe_deallocate(v)
        self._cache_history = []

    def _init_rope(self, T: int, device):
        """Initialize RoPE cos/sin tables for sequence length T on device."""
        if self._rope_T == T and self._rope_cos is not None:
            return
        # REVIEWED: deallocate old cached tensors before overwriting (reassignment leak)
        _safe_deallocate(self._rope_cos)
        _safe_deallocate(self._rope_sin)
        _safe_deallocate(self._causal_mask_upper)
        _safe_deallocate(self._causal_mask_lower)
        positions = torch.arange(T, dtype=torch.float32)
        angles = torch.outer(positions, self._rope_freqs)  # (T, d_rope/2)
        cos = torch.cos(angles).to(torch.bfloat16)  # (T, d_rope/2)
        sin = torch.sin(angles).to(torch.bfloat16)
        # Shape (1, 1, T, d_rope/2) for broadcasting with (B, H, T, d_head)
        cos_4d = cos.unsqueeze(0).unsqueeze(0)
        sin_4d = sin.unsqueeze(0).unsqueeze(0)
        self._rope_cos = ttnn.from_torch(cos_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._rope_sin = ttnn.from_torch(sin_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # REVIEWED: Cache causal masks per T — they were being recreated via
        # from_torch + triu/tril on every forward AND backward call (2x per step
        # per attention layer), leaking device tensors.
        ones_tt = ttnn.from_torch(torch.ones(T, T, dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._causal_mask_upper = ttnn.triu(ones_tt, diagonal=1)  # 1 for s > t
        self._causal_mask_lower = ttnn.tril(ones_tt, diagonal=0)  # 1 for s <= t
        _safe_deallocate(ones_tt)
        self._rope_T = T

    def _apply_rope(self, x: "ttnn.Tensor", B, H, T) -> "ttnn.Tensor":
        """Apply RoPE to x: (B, H, T, d_head) -> (B, H, T, d_head).

        Splits x into [x1, x2] along last dim, rotates:
          rotated = cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        """
        # REVIEWED: intermediates must be explicitly deallocated.
        d_h = self.d_head
        x1 = ttnn.slice(x, [0, 0, 0, 0], [B, H, T, d_h // 2])
        x2 = ttnn.slice(x, [0, 0, 0, d_h // 2], [B, H, T, d_h])
        x1_cos = ttnn.mul(x1, self._rope_cos)
        x2_sin = ttnn.mul(x2, self._rope_sin)
        rot1 = ttnn.sub(x1_cos, x2_sin)
        _safe_deallocate(x1_cos)
        _safe_deallocate(x2_sin)
        x1_sin = ttnn.mul(x1, self._rope_sin)
        x2_cos = ttnn.mul(x2, self._rope_cos)
        rot2 = ttnn.add(x1_sin, x2_cos)
        _safe_deallocate(x1_sin)
        _safe_deallocate(x2_cos)
        rotated = ttnn.concat([rot1, rot2], dim=-1)
        _safe_deallocate(rot1)
        _safe_deallocate(rot2)
        return rotated

    def _apply_rope_backward(self, grad_rotated: "ttnn.Tensor", B, H, T) -> "ttnn.Tensor":
        """Backward through RoPE.

        RoPE is a rotation: rotated = [x1*cos - x2*sin, x1*sin + x2*cos]
        grad_x1 = grad_r1*cos + grad_r2*sin
        grad_x2 = -grad_r1*sin + grad_r2*cos
        """
        # REVIEWED: intermediates must be explicitly deallocated.
        d_h = self.d_head
        grad_r1 = ttnn.slice(grad_rotated, [0, 0, 0, 0], [B, H, T, d_h // 2])
        grad_r2 = ttnn.slice(grad_rotated, [0, 0, 0, d_h // 2], [B, H, T, d_h])
        _r1_cos = ttnn.mul(grad_r1, self._rope_cos)
        _r2_sin = ttnn.mul(grad_r2, self._rope_sin)
        grad_x1 = ttnn.add(_r1_cos, _r2_sin)
        _safe_deallocate(_r1_cos)
        _safe_deallocate(_r2_sin)
        _neg_r1 = ttnn.neg(grad_r1)
        _nr1_sin = ttnn.mul(_neg_r1, self._rope_sin)
        _safe_deallocate(_neg_r1)
        _r2_cos = ttnn.mul(grad_r2, self._rope_cos)
        grad_x2 = ttnn.add(_nr1_sin, _r2_cos)
        _safe_deallocate(_nr1_sin)
        _safe_deallocate(_r2_cos)
        grad_x = ttnn.concat([grad_x1, grad_x2], dim=-1)
        _safe_deallocate(grad_x1)
        _safe_deallocate(grad_x2)
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
        # REVIEWED: intermediates must be explicitly deallocated.
        scores = ttnn.matmul(q_rope, ttnn.transpose(k_rope, -2, -1))  # (B, H, T, T)
        # REVIEWED: use cached _scale_tt (persistent) instead of from_torch every call.
        scale_tt = self._scale_tt
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor.
        scores_scaled = ttnn.mul(scores, scale_tt)
        _safe_deallocate(scores)
        scores = scores_scaled

        # Causal mask: upper triangular (s > t) -> -inf
        # REVIEWED: use cached _causal_mask_upper (persistent) instead of
        # from_torch + triu every call.
        mask = self._causal_mask_upper
        neg_inf = ttnn.full((B, H, T, T), -1e4, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
        scores_masked = ttnn.where(mask, neg_inf, scores)
        _safe_deallocate(neg_inf)

        # Softmax over last dim
        attn = ttnn.softmax(scores_masked, dim=-1)  # (B, H, T, T)

        # Output: (B, H, T, d_head) = attn @ V
        out_4d = ttnn.matmul(attn, v_4d)  # (B, H, T, d_head)

        # Reshape back: (B, T, d_model)
        out_flat = ttnn.reshape(ttnn.permute(out_4d, [0, 2, 1, 3]), [B, T, D])

        # Output projection: (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
        out = ttnn.linear(out_flat, self.out_proj_weight)
        _safe_deallocate(out_flat)

        # Save old cache to history before overwriting (forward may be called
        # multiple times in the recurrent core; backward needs each cache)
        if self._cache:
            self._cache_history.append(self._cache)
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
        # NOTE: Do NOT deallocate out_2d or grad_out_2d — reshape may return views.

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
        # REVIEWED: reshape/expand may return views — do NOT deallocate old
        # grad_attn_sum (would crash if view shares buffer with new one).
        grad_attn_sum = ttnn.reshape(grad_attn_sum, [B, H, T, 1])
        grad_attn_sum = ttnn.expand(grad_attn_sum, [B, H, T, T])
        grad_scores = ttnn.mul(ttnn.sub(grad_attn, grad_attn_sum), attn)
        _safe_deallocate(grad_attn)
        _safe_deallocate(grad_attn_sum)

        # --- Backward through causal masking (where mask, -inf, scores) ---
        # The mask sets scores to -inf for s > t. After softmax, those are 0.
        # The gradient through where is: grad_scores is 0 for s > t (since attn=0 there).
        # No additional gradient needed — the where is already accounted for in softmax.
        # But we need to zero out grad_scores for s > t to avoid spurious gradients.
        # REVIEWED: use cached _causal_mask_lower (persistent) instead of
        # from_torch + tril every backward call.
        mask_causal = self._causal_mask_lower
        zeros_4d = ttnn.zeros((B, H, T, T), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # REVIEWED: reassignment leak — ttnn.where creates a new tensor,
        # old grad_scores must be explicitly deallocated.
        grad_scores_new = ttnn.where(mask_causal, grad_scores, zeros_4d)
        _safe_deallocate(grad_scores)
        grad_scores = grad_scores_new
        _safe_deallocate(zeros_4d)

        # --- Backward through scores = QK^T * scale ---
        # Chain rule: d(L)/d(QK^T) = d(L)/d(scores) * scale (this multiplies
        # BY scale, it does not "undo" it — scale has no learnable params
        # here so there's no separate grad_scale term to compute, unlike the
        # workspace's learnable qk_scale in TTWorkspaceModule).
        scale_tt = c["scale_tt"]
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor,
        # old grad_scores must be explicitly deallocated.
        grad_scores_scaled = ttnn.mul(grad_scores, scale_tt)
        _safe_deallocate(grad_scores)
        grad_scores = grad_scores_scaled

        # grad_q_rope = grad_scores @ k_rope — (B, H, T, d_head)
        k_rope = c["k_rope"]
        grad_q_rope = ttnn.matmul(grad_scores, k_rope)

        # grad_k_rope = grad_scores^T @ q_rope — (B, H, T, d_head)
        q_rope = c["q_rope"]
        grad_k_rope = ttnn.matmul(ttnn.transpose(grad_scores, -2, -1), q_rope)
        _safe_deallocate(grad_scores)

        # --- Backward through RoPE ---
        grad_q_4d = self._apply_rope_backward(grad_q_rope, B, H, T)
        grad_k_4d = self._apply_rope_backward(grad_k_rope, B, H, T)
        _safe_deallocate(grad_q_rope)
        _safe_deallocate(grad_k_rope)

        # --- Backward through reshape (B, H, T, d_head) -> (B, T, D) ---
        # q: (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, D)
        # NOTE: Do NOT deallocate grad_q_4d/grad_k_4d/grad_v_4d — permute/reshape
        # may return views sharing the same buffer.
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
        # NOTE: Do NOT deallocate x_2d or grad_qkv_2d — reshape may return views.

        # grad_x = grad_qkv @ qkv_weight^T — (B, T, D)
        grad_x = ttnn.matmul(grad_qkv_2d, ttnn.transpose(self.qkv_weight, 0, 1))
        _safe_deallocate(grad_qkv)
        # REVIEWED: reshape may return a view sharing the matmul buffer.
        # Do NOT deallocate old grad_x — if reshape returned a view, deallocating
        # would free the buffer we're about to return.
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


class TTRetentionLayer:
    """Retention layer (RetNet-style) -- tt-nn native.

    Decayed linear attention: D[t,s] = gamma^(t-s) for s <= t.
    No softmax, no selective scan, no custom kernels -- pure matmuls.

    Forward:
      qkvg = linear(x, W_in)         # (B, T, 4D)
      q, k, v, g = split(qkvg)
      q, k = rope(q), rope(k)
      A = (q @ k^T) * scale * D      # decayed linear attention, D is causal
      O = A @ v                       # (B, H, T, d_h)
      O = reshape(O)                  # (B, T, D)
      O = O * sigmoid(g)              # element-wise output gate
      out = linear(O, W_out)          # (B, T, D)

    Backward: manual, all on device. Pure matmuls + element-wise ops.
    """

    def __init__(self, config: ModelConfig, device, use_fused_rope=False):
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)
        # Cache scale as a device tensor (avoids per-forward torch→device transfer)
        self._scale_tt = ttnn.from_torch(
            torch.tensor([self.scale], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # QKV+gate projection: (d_model, 4*d_model)
        qkv_w = torch.randn(self.d_model, 4 * self.d_model, dtype=torch.bfloat16) * 0.02
        self.qkv_weight = to_device(qkv_w, device)

        # Output projection: (d_model, d_model)
        out_w = torch.randn(self.d_model, self.d_model, dtype=torch.bfloat16) * 0.02
        self.out_proj_weight = to_device(out_w, device)

        # Gamma: per-head decay scalar, stored as log(gamma) for stability.
        # Init gamma ~0.95 (strong but not total decay): log(0.95) ~ -0.051
        # When per_channel_decay=True, gamma is (n_heads, d_head) — per-channel.
        if config.per_channel_decay:
            gamma_init = torch.full((self.n_heads, self.d_head), -0.051, dtype=torch.float32)
            gamma_init += torch.randn(self.n_heads, self.d_head, dtype=torch.float32) * 0.02
        else:
            gamma_init = torch.full((self.n_heads,), -0.051, dtype=torch.float32)
            gamma_init += torch.randn(self.n_heads, dtype=torch.float32) * 0.02
        self.gamma = to_device(gamma_init, device, dtype=ttnn.float32)
        self.per_channel_decay = config.per_channel_decay

        # Short convolution: depthwise causal conv1d (kernel=3) before QKV.
        # Identity init (w[:,0]=1, w[:,1]=w[:,2]=0) — starts as no-op,
        # learns local token mixing through gradient descent.
        # Based on Kimi K3's KDA which applies short conv to Q/K/V before
        # linear attention (kimi-k3-relevance-notes.md §3).
        self.short_conv = config.short_conv
        self.conv_kernel = config.short_conv_kernel
        if self.short_conv:
            conv_w = torch.zeros(self.d_model, self.conv_kernel, dtype=torch.bfloat16)
            conv_w[:, 0] = 1.0  # identity init
            self.conv_weight = to_device(conv_w, device)

        # Precompute RoPE cos/sin tables (lazy, per T)
        d_rope = self.d_head
        freqs = 1.0 / (10000 ** (torch.arange(0, d_rope, 2).float() / d_rope))
        self._rope_freqs = freqs
        self._rope_cos = None
        self._rope_sin = None
        self._rope_cos_2d = None
        self._rope_sin_2d = None
        self._rope_neg_sin_2d = None
        self._rope_T = 0

        # Decay matrix cache (diff and causal mask, precomputed per T on device)
        self._diff_tt = None
        self._causal_tt = None
        self._decay_T = 0
        # Per-channel decay position scaling cache (exp(±pos * log_gamma))
        self._pos_exp_neg = None  # exp(-pos * log_gamma) for V scaling
        self._pos_exp_pos = None   # exp(pos * log_gamma) for output scaling
        self._pc_decay_T = 0

        # Fused RoPE custom kernel (untested — device 0 was down when written).
        # Enable with use_fused_rope=True in constructor once verified.
        self.use_fused_rope = use_fused_rope

        self._cache = {}
        self._cache_history = []  # old caches from recurrent core iterations

    def _deallocate_cache(self):
        """Deallocate cached intermediates, excluding persistent tensors."""
        if not self._cache:
            return
        _persistent = set()
        if hasattr(self, '_scale_tt') and self._scale_tt is not None:
            _persistent.add(id(self._scale_tt))
        if hasattr(self, 'gate') and self.gate is not None:
            _persistent.add(id(self.gate))
        for v in self._cache.values():
            if id(v) not in _persistent and hasattr(v, 'shape'):
                _safe_deallocate(v)
        self._cache = {}

    def _deallocate_cache_history(self):
        """Deallocate all historical caches (from recurrent core iterations)."""
        for old_cache in self._cache_history:
            _persistent = set()
            if hasattr(self, '_scale_tt') and self._scale_tt is not None:
                _persistent.add(id(self._scale_tt))
            if hasattr(self, 'gate') and self.gate is not None:
                _persistent.add(id(self.gate))
            for v in old_cache.values():
                if id(v) not in _persistent and hasattr(v, 'shape'):
                    _safe_deallocate(v)
        self._cache_history = []

    def _deallocate_conv_cache(self):
        """Deallocate short conv cache (x_shifts and w_taps).

        REVIEWED: w_taps are reshapes of persistent self.conv_weight (may be
        views — do NOT deallocate). x_shifts[0] is the input x (not owned).
        x_shifts[1:] are concat results (new tensors — must be deallocated).
        """
        if not hasattr(self, '_conv_cache') or not self._conv_cache:
            return
        c = self._conv_cache
        x_shifts = c.get("x_shifts", [])
        # x_shifts[0] is the input x (not owned by this cache).
        # x_shifts[1:] are concat results — deallocate them.
        for i in range(1, len(x_shifts)):
            _safe_deallocate(x_shifts[i])
        # w_taps are reshapes of self.conv_weight — may be views, do NOT deallocate.
        self._conv_cache = {}

    def _init_rope(self, T, device):
        """Initialize RoPE cos/sin tables for sequence length T on device."""
        if self._rope_T == T and self._rope_cos is not None:
            return
        # REVIEWED: deallocate old cached RoPE tables before overwriting (reassignment leak)
        _safe_deallocate(self._rope_cos)
        _safe_deallocate(self._rope_sin)
        _safe_deallocate(self._rope_cos_2d)
        _safe_deallocate(self._rope_sin_2d)
        _safe_deallocate(self._rope_neg_sin_2d)
        positions = torch.arange(T, dtype=torch.float32)
        angles = torch.outer(positions, self._rope_freqs)
        cos = torch.cos(angles).to(torch.bfloat16)   # (T, d_head//2)
        sin = torch.sin(angles).to(torch.bfloat16)
        cos_4d = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head//2)
        sin_4d = sin.unsqueeze(0).unsqueeze(0)
        self._rope_cos = ttnn.from_torch(cos_4d, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=device)
        self._rope_sin = ttnn.from_torch(sin_4d, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=device)
        # 2D versions for the fused kernel (T, d_half) — tile layout matches
        # the reader's broadcast indexing.
        self._rope_cos_2d = ttnn.from_torch(cos, dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device)
        self._rope_sin_2d = ttnn.from_torch(sin, dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device)
        # Pre-compute negated sin for backward RoPE (avoid per-call ttnn.neg)
        neg_sin = (-sin).to(torch.bfloat16)
        self._rope_neg_sin_2d = ttnn.from_torch(neg_sin, dtype=ttnn.bfloat16,
                                                 layout=ttnn.TILE_LAYOUT, device=device)
        self._rope_T = T

    def _apply_short_conv(self, x, B, T, D):
        """Depthwise causal conv1d (kernel=3) applied to x before QKV.

        out[t] = w0*x[t] + w1*x[t-1] + w2*x[t-2]  (zero-padded left)

        Implemented as shifts + element-wise muls (avoids ttnn.conv1d
        compilation overhead for a 3-tap kernel).

        x: (B, T, D) bf16 TILE on device
        Returns: (B, T, D) convolved, plus caches shifts for backward.
        """
        device = self.device
        K = self.conv_kernel
        w = self.conv_weight  # (D, K) TILE

        # Split weights into per-tap vectors: (1, 1, D) for broadcasting
        w_taps = []
        for k in range(K):
            w_k = ttnn.reshape(ttnn.slice(w, [0, k], [D, k + 1]), [1, 1, D])
            w_taps.append(w_k)

        # Build shifted versions of x (zero-padded left for causality)
        # REVIEWED: intermediates must be explicitly deallocated.
        x_shifts = [x]  # x[t] (no shift)
        for k in range(1, K):
            zeros_k = ttnn.zeros([B, k, D], dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
            x_prev = ttnn.concat([zeros_k, ttnn.slice(x, [0, 0, 0], [B, T - k, D])], dim=1)
            _safe_deallocate(zeros_k)
            x_shifts.append(x_prev)

        # Weighted sum: out = sum_k w_k * x_shift_k
        out = None
        for k in range(K):
            term = ttnn.mul(x_shifts[k], w_taps[k])
            if out is None:
                out = term
            else:
                # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
                _out_new = ttnn.add(out, term)
                _safe_deallocate(out)
                _safe_deallocate(term)
                out = _out_new

        # Cache for backward
        self._conv_cache = {"x_shifts": x_shifts, "w_taps": w_taps, "B": B, "T": T, "D": D}
        return out

    def _short_conv_backward(self, grad_out):
        """Backward through the short conv.

        grad_out: (B, T, D) — gradient w.r.t. conv output
        Returns: (grad_x, grad_conv_weight)
          grad_x: (B, T, D) — gradient w.r.t. pre-conv input
          grad_conv_weight: (D, K) — gradient w.r.t. conv weights
        """
        c = self._conv_cache
        x_shifts = c["x_shifts"]
        B, T, D = c["B"], c["T"], c["D"]
        K = self.conv_kernel
        device = self.device

        # grad_x[t] = w0*grad_out[t] + w1*grad_out[t+1] + w2*grad_out[t+2]
        # (reverse shift: grad_out shifted left, zero-padded right)
        # REVIEWED: intermediates must be explicitly deallocated.
        grad_x_shifts = [grad_out]  # grad_out[t] — do NOT deallocate (input)
        for k in range(1, K):
            zeros_k = ttnn.zeros([B, k, D], dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
            # Shift left: drop first k, pad right with zeros
            g_shifted = ttnn.concat([ttnn.slice(grad_out, [0, k, 0], [B, T, D]), zeros_k], dim=1)
            _safe_deallocate(zeros_k)
            grad_x_shifts.append(g_shifted)

        w_taps = c["w_taps"]
        grad_x = None
        for k in range(K):
            term = ttnn.mul(grad_x_shifts[k], w_taps[k])
            if grad_x is None:
                grad_x = term
            else:
                # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
                _gx_new = ttnn.add(grad_x, term)
                _safe_deallocate(grad_x)
                _safe_deallocate(term)
                grad_x = _gx_new
        # Deallocate shifted gradients (no longer needed)
        for k in range(1, K):
            _safe_deallocate(grad_x_shifts[k])

        # grad_w_k = sum_{b,t} grad_out[b,t,d] * x_shift_k[b,t,d] → (D,)
        grad_conv_weight = None
        for k in range(K):
            gw = ttnn.mul(grad_out, x_shifts[k])  # (B, T, D)
            # REVIEWED: nested sum reassignment leak — deallocate intermediates.
            gw_2 = ttnn.sum(gw, dim=0)  # (T, D)
            _safe_deallocate(gw)
            gw_3 = ttnn.sum(gw_2, dim=0)  # (D,)
            _safe_deallocate(gw_2)
            # REVIEWED: reshape may return a view — do NOT deallocate gw_3.
            gw = ttnn.reshape(gw_3, [D, 1])
            if grad_conv_weight is None:
                grad_conv_weight = gw
            else:
                # REVIEWED: reassignment leak — ttnn.concat creates a new tensor.
                _gcw_new = ttnn.concat([grad_conv_weight, gw], dim=-1)
                _safe_deallocate(grad_conv_weight)
                _safe_deallocate(gw)
                grad_conv_weight = _gcw_new

        return grad_x, grad_conv_weight

    def _get_pc_decay_scales(self, T, device):
        """Compute per-channel decay position scalings.

        For per-channel decay, D[h,d,t,s] = gamma[h,d]^(t-s).
        Using the exp decomposition:
          out[t,d] = exp(t*lg) * sum_s scores[t,s] * exp(-s*lg) * v[s,d]

        Returns:
          pos_exp_neg: (1, H, T, d_head) = exp(-pos * log_gamma)  — scales V
          pos_exp_pos: (1, H, T, d_head) = exp(pos * log_gamma)   — scales output
        """
        if self._pc_decay_T == T and self._pos_exp_neg is not None:
            return self._pos_exp_neg, self._pos_exp_pos

        # REVIEWED: deallocate old cached scales before overwriting (reassignment leak)
        _safe_deallocate(self._pos_exp_neg)
        _safe_deallocate(self._pos_exp_pos)

        H, d_h = self.n_heads, self.d_head
        pos = torch.arange(T, dtype=torch.float32)  # (T,)

        # log_gamma: (H, d_head) fp32 on device
        log_gamma_host = ttnn.to_torch(self.gamma).float()  # (H, d_head)

        # exp(-pos * log_gamma): (H, T, d_head)
        # pos: (T, 1, 1), log_gamma: (1, H, d_head) → broadcast (T, H, d_head)
        exp_neg = torch.exp(-pos.unsqueeze(1).unsqueeze(2) * log_gamma_host.unsqueeze(0))
        exp_pos = torch.exp(pos.unsqueeze(1).unsqueeze(2) * log_gamma_host.unsqueeze(0))

        # Reshape to (1, H, T, d_head) for broadcasting over batch
        exp_neg = exp_neg.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)  # (1, H, T, d_head)
        exp_pos = exp_pos.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)

        self._pos_exp_neg = ttnn.from_torch(exp_neg, dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device)
        self._pos_exp_pos = ttnn.from_torch(exp_pos, dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device)
        self._pc_decay_T = T
        return self._pos_exp_neg, self._pos_exp_pos

    def _apply_rope(self, x, B, H, T):
        """Apply RoPE to x: (B, H, T, d_head) -> (B, H, T, d_head).

        Uses fused custom kernel when self.use_fused_rope is True (4 ops vs 9).
        Falls back to ttnn ops otherwise.
        """
        rot1, rot2 = self._apply_rope_split(x, B, H, T)
        return ttnn.concat([rot1, rot2], dim=-1)

    def _apply_rope_split(self, x, B, H, T):
        """Apply RoPE and return (rot1, rot2) separately — no concat.

        Enables the caller to skip the concat and use split matmuls instead,
        eliminating expensive sub-tile concat/slice ops when d_half is not
        tile-aligned.
        """
        # REVIEWED: intermediates must be explicitly deallocated.
        d_h = self.d_head
        x1 = ttnn.slice(x, [0, 0, 0, 0], [B, H, T, d_h // 2])
        x2 = ttnn.slice(x, [0, 0, 0, d_h // 2], [B, H, T, d_h])
        if self.use_fused_rope:
            return self._fused_rope_4d(
                x1, x2, self._rope_cos_2d, self._rope_sin_2d,
                B, H, T, d_h // 2, self.device)
        else:
            x1_cos = ttnn.mul(x1, self._rope_cos)
            x2_sin = ttnn.mul(x2, self._rope_sin)
            rot1 = ttnn.sub(x1_cos, x2_sin)
            _safe_deallocate(x1_cos)
            _safe_deallocate(x2_sin)
            x1_sin = ttnn.mul(x1, self._rope_sin)
            x2_cos = ttnn.mul(x2, self._rope_cos)
            rot2 = ttnn.add(x1_sin, x2_cos)
            _safe_deallocate(x1_sin)
            _safe_deallocate(x2_cos)
            return rot1, rot2

    def _apply_rope_backward(self, grad_rotated, B, H, T):
        """Backward through RoPE — same rotation with sin negated."""
        d_h = self.d_head
        grad_r1 = ttnn.slice(grad_rotated, [0, 0, 0, 0], [B, H, T, d_h // 2])
        grad_r2 = ttnn.slice(grad_rotated, [0, 0, 0, d_h // 2], [B, H, T, d_h])
        return self._apply_rope_backward_split(grad_r1, grad_r2, B, H, T)

    def _apply_rope_backward_split(self, grad_r1, grad_r2, B, H, T):
        """Backward through RoPE from already-split grads — no slice needed.

        grad_r1, grad_r2: (B, H, T, d_half) each — grads w.r.t. rot1, rot2
        Returns: (B, H, T, d_head) = concat([grad_x1, grad_x2])
        """
        # REVIEWED: intermediates must be explicitly deallocated.
        d_h = self.d_head
        if self.use_fused_rope:
            grad_x1, grad_x2 = self._fused_rope_4d(
                grad_r1, grad_r2, self._rope_cos_2d, self._rope_neg_sin_2d,
                B, H, T, d_h // 2, self.device)
            result = ttnn.concat([grad_x1, grad_x2], dim=-1)
            _safe_deallocate(grad_x1)
            _safe_deallocate(grad_x2)
            return result
        else:
            _r1_cos = ttnn.mul(grad_r1, self._rope_cos)
            _r2_sin = ttnn.mul(grad_r2, self._rope_sin)
            grad_x1 = ttnn.add(_r1_cos, _r2_sin)
            _safe_deallocate(_r1_cos)
            _safe_deallocate(_r2_sin)
            _neg_r1 = ttnn.neg(grad_r1)
            _nr1_sin = ttnn.mul(_neg_r1, self._rope_sin)
            _safe_deallocate(_neg_r1)
            _r2_cos = ttnn.mul(grad_r2, self._rope_cos)
            grad_x2 = ttnn.add(_nr1_sin, _r2_cos)
            _safe_deallocate(_nr1_sin)
            _safe_deallocate(_r2_cos)
            result = ttnn.concat([grad_x1, grad_x2], dim=-1)
            _safe_deallocate(grad_x1)
            _safe_deallocate(grad_x2)
            return result

    @staticmethod
    def _fused_rope_4d(x1, x2, cos, sin, B, H, T, d_half, device):
        """Full fused RoPE rotation via single custom kernel pass.

        Computes in one kernel launch:
          rot1 = x1*cos - x2*sin
          rot2 = x1*sin + x2*cos
        The kernel does 4 FPU muls + 2 SFPU binary ops (sub, add) on dest regs.

        Args:
            x1: (B, H, T, d_half) bf16 TILE — first half of input
            x2: (B, H, T, d_half) bf16 TILE — second half
            cos: (T, d_half) bf16 TILE — cos table (2D, broadcast over B, H)
            sin: (T, d_half) bf16 TILE — sin table (2D, broadcast over B, H)
        Returns:
            rot1: (B, H, T, d_half) bf16 TILE = x1*cos - x2*sin
            rot2: (B, H, T, d_half) bf16 TILE = x1*sin + x2*cos
        """
        rot1_2d, rot2_2d = TTRetentionLayer._fused_rope_kernel(
            x1, x2, cos, sin, B, H, T, d_half, device)
        rot1 = ttnn.reshape(rot1_2d, [B, H, T, d_half])
        rot2 = ttnn.reshape(rot2_2d, [B, H, T, d_half])
        return rot1, rot2

    @staticmethod
    def _fused_rope_kernel(x1, x2, cos, sin, B, H, T, d_half, device):
        """Launch custom kernel: full RoPE in one pass.
        out0 = x1*cos - x2*sin (rot1)
        out1 = x1*sin + x2*cos (rot2)
        """
        BH = B * H
        x1_2d = ttnn.reshape(x1, [BH * T, d_half])
        x2_2d = ttnn.reshape(x2, [BH * T, d_half])

        tiled_cols = (d_half + 31) // 32
        total_tiles = ((BH * T + 31) // 32) * tiled_cols

        out0 = ttnn.empty([BH * T, d_half], dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
        out1 = ttnn.empty([BH * T, d_half], dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        num_cores_x, num_cores_y = grid.x, grid.y
        num_cores_total = num_cores_x * num_cores_y
        num_cores = min(total_tiles, num_cores_total)
        bpc_base = total_tiles // num_cores
        bpc_rem = total_tiles % num_cores

        all_cores = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
        ])

        tile_bytes = 32 * 32 * 2  # bf16
        cb_x1, cb_x2, cb_cos, cb_sin = 0, 1, 2, 3
        cb_out0, cb_out1 = 16, 17

        def _cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * tile_bytes, core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

        cbs = [_cb(cb_x1, 2), _cb(cb_x2, 2), _cb(cb_cos, 2), _cb(cb_sin, 2),
               _cb(cb_out0, 2), _cb(cb_out1, 2)]

        reader_ct = []
        for t in [x1_2d, x2_2d, cos, sin]:
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = []
        for t in [out0, out1]:
            writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        assert T % 32 == 0, f"_fused_rope_kernel requires T to be a multiple of 32, got T={T}"
        T_tiles = T // 32

        reader_rt, writer_rt, compute_rt = [], [], []
        ts = 0
        for i in range(num_cores_total):
            cx, cy = i // num_cores_y, i % num_cores_y
            coord = ttnn.CoreCoord(cx, cy)
            ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
            reader_rt.append((coord, VectorUInt32([
                x1_2d.buffer_address(), x2_2d.buffer_address(),
                cos.buffer_address(), sin.buffer_address(),
                ntpc, ts, T_tiles, tiled_cols])))
            writer_rt.append((coord, VectorUInt32([
                out0.buffer_address(), out1.buffer_address(),
                ntpc, ts])))
            compute_rt.append((coord, VectorUInt32([ntpc])))
            ts += ntpc

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rope4d_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
            runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rope4d_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
            runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rope4d_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32([]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))

        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                         semaphores=[], cbs=cbs)
        ttnn.generic_op(io_tensors=[x1_2d, x2_2d, cos, sin, out0, out1],
                        program_descriptor=program)
        return out0, out1

    @staticmethod
    def _fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device):
        """Fused scale + decay: scores = scores_raw * scale * D_decay.

        Replaces 2 ttnn ops (mul + mul) with 1 custom kernel.
        D_decay is (1, H, T, T) broadcast over batch B.

        Args:
            scores_raw: (B, H, T, T) bf16 TILE — QK^T matmul result
            D_decay: (1, H, T, T) bf16 TILE — decay matrix
            scale: float scalar
        Returns:
            scores: (B, H, T, T) bf16 TILE
        """
        import struct
        BH = B * H
        # 2D tile layout: scores_raw is (BH*T, T), D_decay is (H*T, T)
        tiled_cols = (T + 31) // 32
        total_tiles = ((BH * T + 31) // 32) * tiled_cols
        HT_tiles = (H * T + 31) // 32  # D_decay tile rows

        # Encode scale as uint32 for SFPU mul_unary_tile
        scale_bits = struct.unpack('I', struct.pack('f', float(scale)))[0]

        out = ttnn.empty([BH * T, T], dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        num_cores_x, num_cores_y = grid.x, grid.y
        num_cores_total = num_cores_x * num_cores_y
        num_cores = min(total_tiles, num_cores_total)
        bpc_base = total_tiles // num_cores
        bpc_rem = total_tiles % num_cores

        all_cores = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
        ])

        tile_bytes = 32 * 32 * 2  # bf16
        cb_scores, cb_D = 0, 1
        cb_out = 16

        def _cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * tile_bytes, core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

        cbs = [_cb(cb_scores, 2), _cb(cb_D, 2), _cb(cb_out, 2)]

        # Compile-time args
        reader_ct = []
        reader_ct.extend(ttnn.TensorAccessorArgs(scores_raw).get_compile_time_args())
        reader_ct.extend(ttnn.TensorAccessorArgs(D_decay).get_compile_time_args())
        writer_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

        # Runtime args per core
        reader_rt, writer_rt, compute_rt = [], [], []
        ts = 0
        for i in range(num_cores_total):
            cx, cy = i // num_cores_y, i % num_cores_y
            coord = ttnn.CoreCoord(cx, cy)
            ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
            reader_rt.append((coord, VectorUInt32([
                scores_raw.buffer_address(), D_decay.buffer_address(),
                ntpc, ts, tiled_cols, HT_tiles])))
            writer_rt.append((coord, VectorUInt32([
                out.buffer_address(), ntpc, ts])))
            compute_rt.append((coord, VectorUInt32([ntpc, scale_bits])))
            ts += ntpc

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
            runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
            runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/scale_decay_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32([]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))

        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                         semaphores=[], cbs=cbs)
        ttnn.generic_op(io_tensors=[scores_raw, D_decay, out],
                        program_descriptor=program)
        return out

    @staticmethod
    def _fused_gate_backward(grad_out_gated, gate, out_flat, B, T, D, device):
        """Fused gate backward: computes grad_out_flat and grad_g in one kernel.

        grad_out_flat = grad_out_gated * gate
        grad_g = grad_out_gated * out_flat * gate * (1 - gate)

        All inputs/outputs are (B, T, D) bf16 TILE. Replaces 6 ttnn ops
        (ones_like, sub, 4 muls) with 1 kernel launch.
        """
        BT = B * T
        # All tensors are (B, T, D) → 2D as (BT, D) for tiling
        gog_2d = ttnn.reshape(grad_out_gated, [BT, D])
        gate_2d = ttnn.reshape(gate, [BT, D])
        of_2d = ttnn.reshape(out_flat, [BT, D])

        tiled_cols = (D + 31) // 32
        total_tiles = ((BT + 31) // 32) * tiled_cols

        out0 = ttnn.empty([BT, D], dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
        out1 = ttnn.empty([BT, D], dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        num_cores_x, num_cores_y = grid.x, grid.y
        num_cores_total = num_cores_x * num_cores_y
        num_cores = min(total_tiles, num_cores_total)
        bpc_base = total_tiles // num_cores
        bpc_rem = total_tiles % num_cores

        all_cores = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
        ])

        tile_bytes = 32 * 32 * 2  # bf16
        cb_gog, cb_gate, cb_of = 0, 1, 2
        cb_out0, cb_out1 = 16, 17

        def _cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * tile_bytes, core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

        cbs = [_cb(cb_gog, 2), _cb(cb_gate, 2), _cb(cb_of, 2),
               _cb(cb_out0, 2), _cb(cb_out1, 2)]

        reader_ct = []
        for t in [gog_2d, gate_2d, of_2d]:
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = []
        for t in [out0, out1]:
            writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        reader_rt, writer_rt, compute_rt = [], [], []
        ts = 0
        for i in range(num_cores_total):
            cx, cy = i // num_cores_y, i % num_cores_y
            coord = ttnn.CoreCoord(cx, cy)
            ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
            reader_rt.append((coord, VectorUInt32([
                gog_2d.buffer_address(), gate_2d.buffer_address(), of_2d.buffer_address(),
                ntpc, ts])))
            writer_rt.append((coord, VectorUInt32([
                out0.buffer_address(), out1.buffer_address(),
                ntpc, ts])))
            compute_rt.append((coord, VectorUInt32([ntpc])))
            ts += ntpc

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/gate_bwd_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
            runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/gate_bwd_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
            runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/gate_bwd_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32([]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))

        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                         semaphores=[], cbs=cbs)
        ttnn.generic_op(io_tensors=[gog_2d, gate_2d, of_2d, out0, out1],
                        program_descriptor=program)
        # Reshape back to (B, T, D)
        grad_out_flat = ttnn.reshape(out0, [B, T, D])
        grad_g = ttnn.reshape(out1, [B, T, D])
        return grad_out_flat, grad_g

    def _get_decay_matrix(self, T, device):
        """Compute D[t,s] = gamma^(t-s) for s <= t, else 0 — fully on device.

        Returns: (1, H, T, T) bf16 on device.

        The diff matrix and causal mask are precomputed once per T and cached.
        Per-step, only a broadcast multiply (diff * log_gamma) and exp are needed
        — no host transfer. This eliminates the 26ms/iter host round-trip that
        was 36% of total time in profiling.
        """
        # Cache the diff matrix and causal mask on device per T
        if self._decay_T != T or self._diff_tt is None:
            # REVIEWED: deallocate old cached tensors before overwriting (reassignment leak)
            _safe_deallocate(self._diff_tt)
            _safe_deallocate(self._causal_tt)
            H = self.n_heads
            pos = torch.arange(T, dtype=torch.float32)
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (T, T)
            causal = (diff >= 0).to(torch.bfloat16)
            # Store as (1, 1, T, T) for broadcasting with (1, H, T, T)
            diff_tt = diff.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
            causal_tt = causal.unsqueeze(0).unsqueeze(0)
            self._diff_tt = ttnn.from_torch(diff_tt, dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device)
            self._causal_tt = ttnn.from_torch(causal_tt, dtype=ttnn.bfloat16,
                                               layout=ttnn.TILE_LAYOUT, device=device)
            self._decay_T = T

        # log_D = diff * log_gamma  — broadcast (1,1,T,T) * (1,H,1,1) -> (1,H,T,T)
        # gamma is (H,) stored as log(gamma); reshape to (1, H, 1, 1)
        # REVIEWED: intermediates must be explicitly deallocated. This function
        # is called every forward pass (6x per step for Cell C recurrent core).
        log_gamma_4d = ttnn.reshape(self.gamma, [1, self.n_heads, 1, 1])
        # Convert to bf16 for the multiply (gamma is fp32, diff is bf16)
        log_gamma_bf16 = ttnn.typecast(log_gamma_4d, ttnn.bfloat16)
        # REVIEWED: reshape may return a view of self.gamma (persistent) —
        # do NOT deallocate log_gamma_4d (would free self.gamma's buffer).
        log_D = ttnn.mul(self._diff_tt, log_gamma_bf16)  # (1, H, T, T)
        _safe_deallocate(log_gamma_bf16)
        D = ttnn.exp(log_D)  # (1, H, T, T)
        _safe_deallocate(log_D)
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor.
        D_masked = ttnn.mul(D, self._causal_tt)  # zero out future positions
        _safe_deallocate(D)
        return D_masked

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        H, d_h = self.n_heads, self.d_head
        device = self.device

        self._init_rope(T, device)

        # Short conv before QKV (Kimi K3 §3): depthwise causal conv1d
        x_conv = x
        if self.short_conv:
            x_conv = self._apply_short_conv(x, B, T, D)

        # QKV+gate projection: (B, T, 4D)
        qkvg = ttnn.linear(x_conv, self.qkv_weight)

        # Split: q, k, v, g each (B, T, D)
        q = ttnn.slice(qkvg, [0, 0, 0], [B, T, D])
        k = ttnn.slice(qkvg, [0, 0, D], [B, T, 2 * D])
        v = ttnn.slice(qkvg, [0, 0, 2 * D], [B, T, 3 * D])
        g = ttnn.slice(qkvg, [0, 0, 3 * D], [B, T, 4 * D])

        # Reshape q, k, v to (B, H, T, d_head)
        q_4d = ttnn.permute(ttnn.reshape(q, [B, T, H, d_h]), [0, 2, 1, 3])
        k_4d = ttnn.permute(ttnn.reshape(k, [B, T, H, d_h]), [0, 2, 1, 3])
        v_4d = ttnn.permute(ttnn.reshape(v, [B, T, H, d_h]), [0, 2, 1, 3])

        # Apply RoPE to q and k — keep rot1/rot2 separate to avoid expensive
        # sub-tile concat (d_half=48 is not tile-aligned). Split QK matmul instead:
        # qk = rot1_q @ rot1_k^T + rot2_q @ rot2_k^T
        rot1_q, rot2_q = self._apply_rope_split(q_4d, B, H, T)
        rot1_k, rot2_k = self._apply_rope_split(k_4d, B, H, T)

        # Scores: (B, H, T, T) = (Q @ K^T) * scale
        # Split matmul: qk = q1@k1^T + q2@k2^T (avoids concat of rot1/rot2)
        # REVIEWED: intermediates must be explicitly deallocated.
        _qk1 = ttnn.matmul(rot1_q, ttnn.transpose(rot1_k, -2, -1))
        _qk2 = ttnn.matmul(rot2_q, ttnn.transpose(rot2_k, -2, -1))
        qk = ttnn.add(_qk1, _qk2)
        _safe_deallocate(_qk1)
        _safe_deallocate(_qk2)
        scale_tt = self._scale_tt

        if self.per_channel_decay:
            # Per-channel decay (Kimi K3 §5): gamma is (H, d_head), not (H,).
            # Use exp decomposition to avoid materializing (1, H, d_head, T, T):
            #   out[t,d] = exp(t*lg) * sum_s scores[t,s] * exp(-s*lg) * v[s,d]
            # Scores get scale + causal mask only (no decay).
            pos_exp_neg, pos_exp_pos = self._get_pc_decay_scales(T, device)

            # Ensure causal mask is initialized (needed for scores masking)
            if self._decay_T != T or self._causal_tt is None:
                # REVIEWED: deallocate old _causal_tt before overwriting (reassignment leak)
                _safe_deallocate(self._causal_tt)
                pos_t = torch.arange(T, dtype=torch.float32)
                diff = pos_t.unsqueeze(1) - pos_t.unsqueeze(0)
                causal = (diff >= 0).to(torch.bfloat16)
                causal_tt = causal.unsqueeze(0).unsqueeze(0)
                self._causal_tt = ttnn.from_torch(causal_tt, dtype=ttnn.bfloat16,
                                                   layout=ttnn.TILE_LAYOUT, device=device)
                self._decay_T = T

            # Apply scale + causal mask to scores (no decay matrix)
            # REVIEWED: intermediates must be explicitly deallocated.
            qk_2d = ttnn.reshape(qk, [B * H * T, T])
            # Build (H, T, T) causal mask for the fused kernel (broadcast over B)
            causal_H_2d = ttnn.reshape(
                ttnn.from_torch(
                    ttnn.to_torch(self._causal_tt).squeeze().unsqueeze(0).expand(H, T, T).contiguous(),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                [H * T, T])
            scores_2d = self._fused_scale_decay(qk_2d, causal_H_2d, self.scale, B, H, T, device)
            _safe_deallocate(causal_H_2d)
            # REVIEWED: qk_2d and scores_2d may be views of qk/scores — do NOT deallocate.
            scores = ttnn.reshape(scores_2d, [B, H, T, T])

            # Scale V by exp(-pos * log_gamma): v_scaled[b,h,s,d] = v[b,h,s,d] * exp(-s*lg[h,d])
            v_scaled = ttnn.mul(v_4d, pos_exp_neg)  # (B,H,T,d_h) * (1,H,T,d_h)

            # Output: (B, H, T, d_head) = scores @ v_scaled
            raw_out = ttnn.matmul(scores, v_scaled)

            # Scale output by exp(pos * log_gamma): out[b,h,t,d] = raw_out * exp(t*lg[h,d])
            out_4d = ttnn.mul(raw_out, pos_exp_pos)
        else:
            # Scalar decay (original): D[h,t,s] = gamma[h]^(t-s)
            D_decay = self._get_decay_matrix(T, device)

            # Reshape to 2D for the fused kernel (it expects flat (BH*T, T) layout)
            # REVIEWED: intermediates must be explicitly deallocated.
            qk_2d = ttnn.reshape(qk, [B * H * T, T])
            D_decay_2d = ttnn.reshape(D_decay, [H * T, T])
            scores_2d = self._fused_scale_decay(qk_2d, D_decay_2d, self.scale, B, H, T, device)
            # REVIEWED: D_decay_2d may be a view of D_decay (cached) — do NOT deallocate.
            # qk_2d may be a view of qk (cached) — do NOT deallocate.
            # scores_2d may be a view of scores — do NOT deallocate.
            scores = ttnn.reshape(scores_2d, [B, H, T, T])

            # Output: (B, H, T, d_head) = scores @ V
            out_4d = ttnn.matmul(scores, v_4d)

        # Reshape to (B, T, D): permute (B,H,T,d_h) -> (B,T,H,d_h) -> (B,T,D)
        # REVIEWED: ttnn.permute may return a view of out_4d (cached) — do NOT
        # deallocate the intermediate. out_flat is also cached for backward.
        out_flat = ttnn.reshape(ttnn.permute(out_4d, [0, 2, 1, 3]), [B, T, D])

        # Output gate: sigmoid(g) element-wise over (B, T, D)
        gate = ttnn.sigmoid(g)  # (B, T, D)
        out_gated = ttnn.mul(out_flat, gate)

        # Output projection
        out = ttnn.linear(out_gated, self.out_proj_weight)

        # Save old cache to history before overwriting (forward may be called
        # multiple times in the recurrent core; backward needs each cache)
        if self._cache:
            self._cache_history.append(self._cache)
        # Cache for backward
        self._cache = {
            "x": x, "x_conv": x_conv, "qkvg": qkvg, "q": q, "k": k, "v": v, "g": g,
            "q_4d": q_4d, "k_4d": k_4d, "v_4d": v_4d,
            "rot1_q": rot1_q, "rot2_q": rot2_q,
            "rot1_k": rot1_k, "rot2_k": rot2_k,
            "qk": qk, "scores": scores,
            "out_4d": out_4d, "out_flat": out_flat,
            "gate": gate, "out_gated": out_gated,
            "scale_tt": scale_tt,
        }
        if self.per_channel_decay:
            self._cache["v_scaled"] = v_scaled
            self._cache["raw_out"] = raw_out
            self._cache["pos_exp_neg"] = pos_exp_neg
            self._cache["pos_exp_pos"] = pos_exp_pos
        else:
            self._cache["D_decay"] = D_decay

        return out

    def backward(self, grad_out):
        """Backward pass -- all on device.

        grad_out: (B, T, d_model)
        Returns: (grad_x, grads_dict)
        """
        c = self._cache
        B, T, D = int(c["x"].shape[0]), int(c["x"].shape[1]), int(c["x"].shape[2])
        H, d_h = self.n_heads, self.d_head
        device = self.device

        # --- Backward through out_proj: out = linear(out_gated, out_proj_w) ---
        grad_out_gated = ttnn.linear(grad_out, ttnn.transpose(self.out_proj_weight, 0, 1))  # (B, T, D)

        # grad_out_proj_weight = out_gated^T @ grad_out
        out_gated_2d = ttnn.reshape(c["out_gated"], [B * T, D])
        grad_out_2d = ttnn.reshape(grad_out, [B * T, D])
        grad_out_proj_weight = ttnn.matmul(ttnn.transpose(out_gated_2d, 0, 1), grad_out_2d)
        # NOTE: Do NOT deallocate out_gated_2d or grad_out_2d — reshape may
        # return views sharing buffers with c["out_gated"] (cached, reused
        # across recurrent core iterations) or grad_out (input).

        # --- Backward through gate: out_gated = out_flat * sigmoid(g) ---
        # Fused kernel: grad_out_flat = grad_out_gated * gate
        #               grad_g = grad_out_gated * out_flat * gate * (1 - gate)
        gate = c["gate"]  # (B, T, D)
        grad_out_flat, grad_g = self._fused_gate_backward(
            grad_out_gated, gate, c["out_flat"], B, T, D, device)
        _safe_deallocate(grad_out_gated)

        # --- Backward through reshape: out_flat = reshape(permute(out_4d)) ---
        # out_4d (B, H, T, d_h) -> permute -> (B, T, H, d_h) -> reshape -> (B, T, D)
        grad_out_4d = ttnn.permute(
            ttnn.reshape(grad_out_flat, [B, T, H, d_h]),
            [0, 2, 1, 3]
        )  # (B, H, T, d_h)
        _safe_deallocate(grad_out_flat)

        if self.per_channel_decay:
            # --- Per-channel decay backward ---
            # Forward was:
            #   v_scaled = v * pos_exp_neg
            #   raw_out = scores @ v_scaled
            #   out_4d = raw_out * pos_exp_pos
            #
            # Backward:
            #   grad_raw_out = grad_out_4d * pos_exp_pos
            #   grad_pos_pos = sum(grad_out_4d * raw_out * pos)  (for gamma grad)
            #   grad_scores = grad_raw_out @ v_scaled^T
            #   grad_v_scaled = scores^T @ grad_raw_out
            #   grad_v = grad_v_scaled * pos_exp_neg
            #   grad_pos_neg = sum(grad_v_scaled * v * (-pos))  (for gamma grad)
            pos_exp_pos = c["pos_exp_pos"]
            pos_exp_neg = c["pos_exp_neg"]
            raw_out = c["raw_out"]
            v_scaled = c["v_scaled"]

            # grad_raw_out = grad_out_4d * pos_exp_pos
            grad_raw_out = ttnn.mul(grad_out_4d, pos_exp_pos)
            _safe_deallocate(grad_out_4d)

            # grad_scores = grad_raw_out @ v_scaled^T
            grad_scores = ttnn.matmul(grad_raw_out, ttnn.transpose(v_scaled, -2, -1))

            # grad_v_scaled = scores^T @ grad_raw_out
            grad_v_scaled = ttnn.matmul(ttnn.transpose(c["scores"], -2, -1), grad_raw_out)
            _safe_deallocate(grad_raw_out)

            # grad_v = grad_v_scaled * pos_exp_neg
            grad_v_4d = ttnn.mul(grad_v_scaled, pos_exp_neg)

            # --- gamma gradient for per-channel decay ---
            # gamma[h,d] affects out through pos_exp_pos and pos_exp_neg.
            # pos_exp_pos[t,d] = exp(t * log_gamma[h,d])
            # pos_exp_neg[s,d] = exp(-s * log_gamma[h,d])
            #
            # grad_log_gamma from pos_exp_pos:
            #   d(pos_exp_pos)/d(log_gamma) = pos_exp_pos * pos
            #   grad += sum_{b,t} grad_out_4d[b,h,t,d] * raw_out[b,h,t,d] * pos_exp_pos[h,t,d] * t
            #         = sum_{b,t} grad_raw_out[b,h,t,d] * raw_out[b,h,t,d] * t
            #
            # grad_log_gamma from pos_exp_neg:
            #   d(pos_exp_neg)/d(log_gamma) = -pos * pos_exp_neg
            #   grad += sum_{b,s} grad_v_scaled[b,h,s,d] * v[b,h,s,d] * (-s)
            #
            # Combined: grad_log_gamma[h,d] = sum_t grad_raw_out*raw_out*t - sum_s grad_v_scaled*v*s
            # Computed on host since gamma is fp32 and the position weighting needs
            # the position index — much simpler than device-side indexing.

            # Bring needed tensors to host for gamma gradient
            grad_raw_out_h = ttnn.to_torch(grad_raw_out).float()  # (B,H,T,d_h)
            raw_out_h = ttnn.to_torch(raw_out).float()
            grad_v_scaled_h = ttnn.to_torch(grad_v_scaled).float()
            v_4d_h = ttnn.to_torch(c["v_4d"]).float()
            _safe_deallocate(grad_v_scaled)
            pos = torch.arange(T, dtype=torch.float32)  # (T,)

            # grad from pos_exp_pos: sum over B,T of grad_raw_out * raw_out * t
            # (B,H,T,d_h) * (B,H,T,d_h) * (T,) → sum over B,T → (H, d_h)
            grad_from_pos = (grad_raw_out_h * raw_out_h * pos.view(1, 1, T, 1)).sum(dim=(0, 2))

            # grad from pos_exp_neg: sum over B,S of grad_v_scaled * v * (-s)
            grad_from_neg = (grad_v_scaled_h * v_4d_h * (-pos.view(1, 1, T, 1))).sum(dim=(0, 2))

            grad_log_gamma_h = grad_from_pos + grad_from_neg  # (H, d_h)
            grad_gamma = ttnn.from_torch(grad_log_gamma_h, dtype=ttnn.float32,
                                          layout=ttnn.TILE_LAYOUT, device=device)

            # --- Backward through scores = qk * scale * causal_mask ---
            # (no decay in scores for per-channel mode — just scale + causal)
            qk = c["qk"]
            # Fused: grad_scores * scale * causal_mask
            grad_scores_2d = ttnn.reshape(grad_scores, [B * H * T, T])
            causal_H_2d = ttnn.reshape(
                ttnn.from_torch(
                    ttnn.to_torch(self._causal_tt).squeeze().unsqueeze(0).expand(H, T, T).contiguous(),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                [H * T, T])
            grad_scores_scaled_2d = self._fused_scale_decay(
                grad_scores_2d, causal_H_2d, self.scale, B, H, T, device)
            grad_scores_scaled = ttnn.reshape(grad_scores_scaled_2d, [B, H, T, T])
            _safe_deallocate(causal_H_2d)
            # NOTE: Do NOT deallocate grad_scores_2d or grad_scores_scaled_2d —
            # ttnn.reshape() may return a view sharing the same device buffer.
            # grad_scores_2d shares with grad_scores, grad_scores_scaled_2d shares
            # with grad_scores_scaled (used later for RoPE backward).

        else:
            # --- Scalar decay backward (original) ---
            # --- Backward through out_4d = scores @ v_4d ---
            grad_scores = ttnn.matmul(grad_out_4d, ttnn.transpose(c["v_4d"], -2, -1))  # (B, H, T, T)
            grad_v_4d = ttnn.matmul(ttnn.transpose(c["scores"], -2, -1), grad_out_4d)  # (B, H, T, d_h)
            _safe_deallocate(grad_out_4d)

            # --- Backward through scores = qk * scale * D_decay (fused in forward) ---
            D_decay = c["D_decay"]
            qk = c["qk"]

            # Fused: grad_scores * scale * D in one pass = grad w.r.t. QK^T
            grad_scores_2d = ttnn.reshape(grad_scores, [B * H * T, T])
            D_decay_2d = ttnn.reshape(D_decay, [H * T, T])
            grad_scores_scaled_2d = self._fused_scale_decay(
                grad_scores_2d, D_decay_2d, self.scale, B, H, T, device)
            grad_scores_scaled = ttnn.reshape(grad_scores_scaled_2d, [B, H, T, T])
            # NOTE: Do NOT deallocate grad_scores_2d, D_decay_2d, or
            # grad_scores_scaled_2d — ttnn.reshape() may return a view sharing
            # the same device buffer. grad_scores_2d shares with grad_scores
            # (used below), D_decay_2d shares with D_decay (used below),
            # grad_scores_scaled_2d shares with grad_scores_scaled (used later).

            # grad_D_decay = grad_scores * (qk * scale)
            qk_scaled = ttnn.mul(qk, c["scale_tt"])
            grad_D_decay = ttnn.mul(grad_scores, qk_scaled)  # (B, H, T, T)
            _safe_deallocate(qk_scaled)
            _safe_deallocate(grad_scores)

            # --- Backward through D_decay w.r.t. gamma (log_gamma) ---
            # D[h,t,s] = exp(diff[t,s] * log_gamma[h]) for s <= t
            # dD/d(log_gamma[h]) = D[h,t,s] * diff[t,s]
            # grad_log_gamma[h] = sum_{b,t,s} grad_D_decay[b,h,t,s] * D[h,t,s] * diff[t,s]
            # Computed fully on device to avoid the 33ms/iter host transfer that was
            # 47% of total time in profiling. Uses the cached diff_tt from forward.
            # weighted = grad_D_decay * D_decay * diff  -> (B, H, T, T)
            weighted = ttnn.mul(grad_D_decay, D_decay)      # (B, H, T, T)
            _safe_deallocate(grad_D_decay)
            # REVIEWED: reassignment leak — ttnn.mul creates a new tensor,
            # old `weighted` must be explicitly deallocated.
            weighted_new = ttnn.mul(weighted, self._diff_tt)  # broadcast (1,1,T,T)
            _safe_deallocate(weighted)
            weighted = weighted_new
            # Sum over batch (dim 0) and positions (dims 2, 3) -> (H,)
            grad_log_gamma = ttnn.sum(weighted, dim=0)       # (H, T, T)
            _safe_deallocate(weighted)
            # REVIEWED: reassignment leak — ttnn.sum creates a new tensor,
            # old grad_log_gamma must be explicitly deallocated.
            grad_log_gamma_2 = ttnn.sum(grad_log_gamma, dim=1) # (H, T)
            _safe_deallocate(grad_log_gamma)
            grad_log_gamma_3 = ttnn.sum(grad_log_gamma_2, dim=1) # (H,)
            _safe_deallocate(grad_log_gamma_2)
            # Store as fp32 to match gamma's dtype
            grad_gamma = ttnn.typecast(grad_log_gamma_3, ttnn.float32)
            _safe_deallocate(grad_log_gamma_3)

        # grad_q_rope = grad_scores_scaled @ k_rope (split: no concat needed)
        # grad_k_rope = grad_scores_scaled^T @ q_rope (split: no concat needed)
        grad_rot1_q = ttnn.matmul(grad_scores_scaled, c["rot1_k"])  # (B, H, T, d_half)
        grad_rot2_q = ttnn.matmul(grad_scores_scaled, c["rot2_k"])
        grad_rot1_k = ttnn.matmul(ttnn.transpose(grad_scores_scaled, -2, -1), c["rot1_q"])
        grad_rot2_k = ttnn.matmul(ttnn.transpose(grad_scores_scaled, -2, -1), c["rot2_q"])
        _safe_deallocate(grad_scores_scaled)

        # --- Backward through RoPE (split: no slice needed) ---
        grad_q_4d = self._apply_rope_backward_split(grad_rot1_q, grad_rot2_q, B, H, T)
        grad_k_4d = self._apply_rope_backward_split(grad_rot1_k, grad_rot2_k, B, H, T)
        _safe_deallocate(grad_rot1_q)
        _safe_deallocate(grad_rot2_q)
        _safe_deallocate(grad_rot1_k)
        _safe_deallocate(grad_rot2_k)

        # --- Backward through reshape: (B, H, T, d_head) -> (B, T, D) ---
        # NOTE: ttnn.permute/reshape may return views sharing the same buffer.
        # Do NOT deallocate grad_q_4d/grad_k_4d/grad_v_4d here — grad_q/grad_k/grad_v
        # may share their buffers. They'll be cleaned up by clear_caches() or GC.
        grad_q = ttnn.reshape(ttnn.permute(grad_q_4d, [0, 2, 1, 3]), [B, T, D])
        grad_k = ttnn.reshape(ttnn.permute(grad_k_4d, [0, 2, 1, 3]), [B, T, D])
        grad_v = ttnn.reshape(ttnn.permute(grad_v_4d, [0, 2, 1, 3]), [B, T, D])
        # grad_g is already (B, T, D)

        # --- Backward through QKV split: concat([q, k, v, g]) ---
        grad_qkvg = ttnn.concat([grad_q, grad_k, grad_v, grad_g], dim=-1)  # (B, T, 4D)

        # --- Backward through QKV linear: qkvg = linear(x_conv, qkv_weight) ---
        # Use x_conv (post-short-conv) as the input to the linear, not x
        x_conv = c["x_conv"]
        x_conv_2d = ttnn.reshape(x_conv, [B * T, D])
        grad_qkvg_2d = ttnn.reshape(grad_qkvg, [B * T, 4 * D])
        grad_qkv_weight = ttnn.matmul(ttnn.transpose(x_conv_2d, 0, 1), grad_qkvg_2d)
        # NOTE: Do NOT deallocate x_conv_2d or grad_qkvg_2d — they may be views
        # of x_conv (cached) or grad_qkvg (used below).

        grad_x_conv = ttnn.matmul(grad_qkvg_2d, ttnn.transpose(self.qkv_weight, 0, 1))
        _safe_deallocate(grad_qkvg)
        # REVIEWED: reshape may return a view sharing the matmul buffer.
        # Do NOT deallocate the old grad_x_conv — if reshape returned a view,
        # deallocating would free the buffer we're about to use. The matmul
        # buffer will be reclaimed by GC when grad_x_conv goes out of scope.
        grad_x_conv = ttnn.reshape(grad_x_conv, [B, T, D])

        # --- Backward through short conv (if enabled) ---
        if self.short_conv:
            grad_x, grad_conv_weight = self._short_conv_backward(grad_x_conv)
        else:
            grad_x = grad_x_conv
            grad_conv_weight = None

        grads = {
            "qkv_weight": grad_qkv_weight,
            "out_proj_weight": grad_out_proj_weight,
            "gamma": grad_gamma,
        }
        if grad_conv_weight is not None:
            grads["conv_weight"] = grad_conv_weight

        return grad_x, grads

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        params = {"qkv_weight": self.qkv_weight, "out_proj_weight": self.out_proj_weight,
                  "gamma": self.gamma}
        if self.short_conv:
            params["conv_weight"] = self.conv_weight
        return params

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        if "qkv_weight" in params:
            self.qkv_weight = params["qkv_weight"]
        if "out_proj_weight" in params:
            self.out_proj_weight = params["out_proj_weight"]
        if "gamma" in params:
            self.gamma = params["gamma"]
        if "conv_weight" in params and self.short_conv:
            self.conv_weight = params["conv_weight"]


class TTGatedResidualLayer:
    """ReZero pre-norm residual wrapper: x + gate * layer(norm(x)).

    ReZero gate: scalar (no sigmoid), starts at 0.0 (true identity at init).
    The layer contributes nothing at initialization, and the gate grows
    naturally through gradient descent. This gives the backbone time to
    learn a good representation before the layer starts contributing, and
    keeps the gradient self-amplification at 1.0 per layer at init (vs
    sigmoid(0)=0.5 which gives 1 + 0.5*sigma per layer).

    (Bachlechner et al., 2020 — same ReZero approach used for workspace gates.)
    """

    def __init__(self, layer, d_model: int, device):
        self.layer = layer
        self.norm = TTRMSNorm(d_model, device)
        self.device = device
        # ReZero: gate init 0.0 (true identity, no sigmoid)
        self.gate = ttnn.from_torch(
            torch.zeros(1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._cached_x = None
        self._cached_inner = None
        self._cache_history = []  # old caches from recurrent core iterations

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        normed = self.norm.forward(x)
        inner = self.layer.forward(normed)
        # ReZero: x + gate * inner (no sigmoid)
        # REVIEWED: intermediates must be explicitly deallocated.
        gated_inner = ttnn.mul(self.gate, inner)  # (B, T, d_model)
        out = ttnn.add(x, gated_inner)
        _safe_deallocate(gated_inner)
        # Save old cache to history before overwriting (forward may be called
        # multiple times in the recurrent core; backward needs each cache)
        if self._cached_x is not None:
            self._cache_history.append((self._cached_x, self._cached_inner))
        # Cache for backward
        self._cached_x = x
        self._cached_inner = inner
        return out

    def backward(self, grad_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", Dict[str, "ttnn.Tensor"]]:
        """Backward through ReZero residual.

        grad_out: (B, T, d_model)
        Returns: (grad_x, grads_dict) where grads_dict includes layer grads,
        norm grads, and gate grad.
        """
        device = self.device

        # Forward was: y = x + gate * layer(norm(x))
        # grad_x = grad_out (residual) + gate * grad_inner
        # grad_gate = sum(grad_out * inner)  (no sigmoid derivative)
        # grad_inner = grad_out * gate
        # grad_norm_input = layer.backward(grad_inner) -> grad_x_from_inner
        # Then grad_x = grad_out + grad_x_from_inner (through norm)

        # grad_inner = grad_out * gate (ReZero: direct scalar, no sigmoid)
        grad_inner = ttnn.mul(grad_out, self.gate)

        # grad_gate = sum(grad_out * inner) — no sigmoid derivative
        inner = self._cached_inner
        grad_gate_val = ttnn.mul(grad_out, inner)
        # Gate is a scalar — sum over all dimensions on device
        # REVIEWED: nested sum reassignment leak — deallocate intermediates.
        _gg1 = ttnn.sum(grad_gate_val, dim=0)
        _safe_deallocate(grad_gate_val)
        _gg2 = ttnn.sum(_gg1, dim=0)
        _safe_deallocate(_gg1)
        grad_gate = ttnn.sum(_gg2, dim=0)  # (1,)
        _safe_deallocate(_gg2)

        # Layer backward (through norm and layer)
        grad_norm_input, layer_grads = self.layer.backward(grad_inner)
        _safe_deallocate(grad_inner)

        # Norm backward — on device
        grad_x_from_norm, grad_norm_w = self.norm.backward(grad_norm_input, self._cached_x)
        _safe_deallocate(grad_norm_input)

        # Total grad_x = grad_out (residual) + grad_x_from_norm
        # REVIEWED: intermediate grad_x_from_norm must be deallocated after add.
        grad_x = ttnn.add(grad_out, grad_x_from_norm)
        _safe_deallocate(grad_x_from_norm)

        # Collect all grads
        all_grads = {}
        for k, v in layer_grads.items():
            all_grads[k] = v
        all_grads["wrapper_norm_weight"] = grad_norm_w
        all_grads["gate"] = grad_gate

        return grad_x, all_grads

    def clear_caches(self):
        """Clear cached tensors from the last forward pass.

        Explicitly deallocate cached intermediates to free device buffers,
        then drop references. Model parameters survive (still referenced by
        self.gate, self.norm.weight, etc.).
        """
        _safe_deallocate(self._cached_x)
        _safe_deallocate(self._cached_inner)
        self._cached_x = None
        self._cached_inner = None
        # Deallocate cache history from recurrent core iterations
        for old_x, old_inner in self._cache_history:
            _safe_deallocate(old_x)
            _safe_deallocate(old_inner)
        self._cache_history = []
        if hasattr(self.layer, 'clear_caches'):
            self.layer.clear_caches()
        elif hasattr(self.layer, '_cache'):
            # Retention/attention layers: deallocate cached intermediates.
            # Exclude persistent tensors: scale_tt (self._scale_tt),
            # gate (self.gate), and cached causal masks are model parameters
            # reused across steps.
            _persistent = set()
            if hasattr(self.layer, '_scale_tt'):
                _persistent.add(id(self.layer._scale_tt))
            if hasattr(self.layer, 'gate'):
                _persistent.add(id(self.layer.gate))
            if hasattr(self.layer, '_causal_mask_upper') and self.layer._causal_mask_upper is not None:
                _persistent.add(id(self.layer._causal_mask_upper))
            if hasattr(self.layer, '_causal_mask_lower') and self.layer._causal_mask_lower is not None:
                _persistent.add(id(self.layer._causal_mask_lower))
            for k, v in self.layer._cache.items():
                if id(v) not in _persistent and hasattr(v, 'shape'):
                    _safe_deallocate(v)
            self.layer._cache = {}
            # Deallocate cache history from recurrent core iterations
            if hasattr(self.layer, '_deallocate_cache_history'):
                self.layer._deallocate_cache_history()
            # REVIEWED: deallocate short conv cache if present (latent leak when short_conv=True)
            if hasattr(self.layer, '_deallocate_conv_cache'):
                self.layer._deallocate_conv_cache()

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        params = {"gate": self.gate, "wrapper_norm_weight": self.norm.weight}
        params.update(self.layer.get_params())
        return params

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        if "gate" in params:
            _safe_deallocate(self.gate)
            self.gate = params["gate"]
        if "wrapper_norm_weight" in params:
            _safe_deallocate(self.norm.weight)
            self.norm.weight = params["wrapper_norm_weight"]
        layer_params = {k: v for k, v in params.items() if k not in ("gate", "wrapper_norm_weight")}
        if layer_params:
            self.layer.set_params(layer_params)


class TTWorkspaceModule:
    """Perceiver-style workspace with m learned slot vectors — tt-nn native.

    Two cross-attention passes per application:
      1. Read: slots attend over hidden states (Q from slots, KV from x)
      2. Write: hidden states attend over slots (Q from x, KV from slots)

    Both passes use ReZero gates (scalar, init=0) and RMSNorm.
    QK Normalization is applied to both cross-attention passes to prevent
    entropy collapse and bound attention logits regardless of weight magnitudes.
    Causal masking is applied to both passes (Perceiver-IO style) to prevent
    future-token leakage in autoregressive settings. No RoPE.
    """

    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.n_slots = config.n_workspace_slots
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.spectral_norm_bound = config.spectral_norm_bound

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

        # QK-Norm learnable scale parameters (one per attention pass, shared across heads).
        # Initialized to 1/√d_head (the standard attention scale). QK-Norm normalizes
        # Q and K to unit L2 norm per head, then scales by this learnable scalar.
        # This bounds attention logits to [-scale, +scale] regardless of weight magnitudes,
        # preventing the entropy collapse / ill-conditioned QK^T that caused divergence.
        qk_scale_init = self.scale
        self.read_qk_scale = ttnn.from_torch(torch.tensor([qk_scale_init], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.write_qk_scale = ttnn.from_torch(torch.tensor([qk_scale_init], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Norms
        self.norm = TTRMSNorm(self.d_model, device)       # for x
        self.slot_norm = TTRMSNorm(self.d_model, device)  # for slots

        # Fixed weight for slot parameter normalization (not learned)
        self._slot_param_norm_weight = ttnn.from_torch(
            torch.ones(self.d_model, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # ReZero gates: scalar gates initialized to 0.0 (true identity at init).
        # Unlike sigmoid gates (which start at ~0.12 and have saturating gradients),
        # ReZero gates start at exactly 0 and grow linearly through gradient descent.
        # This gives the backbone time to learn a good representation before the
        # workspace starts contributing, and avoids the sigmoid' gradient saturation
        # that made gates slow to adapt.  (Bachlechner et al., 2020)
        self.read_gate = ttnn.from_torch(torch.tensor([0.0], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.write_gate = ttnn.from_torch(torch.tensor([0.0], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Slot decay: learned scalar that makes the slot update contractive.
        # slots_out = norm(decay * slots_in + gate * read_out)
        # Initialized at slot_decay_init (1.0 = no decay). The decay provides a
        # restoring force: old information fades unless actively reinforced.
        self.slot_decay = ttnn.from_torch(torch.tensor([config.slot_decay_init], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        self._cache = {}
        self._cache_history = []  # old caches from recurrent core iterations
        self._forward_caches = []  # list of dicts, one per forward call (for regularizers)

        # REVIEWED: Cache epsilon tensor for L2 normalization (called 4x per
        # forward + 4x per backward = 8x per workspace call, K times per step
        # in Cell C). Avoids repeated from_torch + deallocate overhead.
        self._eps_tt = ttnn.from_torch(
            torch.tensor([1e-6], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _deallocate_cache(self):
        """Deallocate cached intermediates, excluding persistent tensors."""
        if self._cache:
            _persistent_ids = set()
            if hasattr(self, 'slots') and self.slots is not None:
                _persistent_ids.add(id(self.slots))
            if hasattr(self, 'read_gate') and self.read_gate is not None:
                _persistent_ids.add(id(self.read_gate))
            if hasattr(self, 'write_gate') and self.write_gate is not None:
                _persistent_ids.add(id(self.write_gate))
            for v in self._cache.values():
                if id(v) not in _persistent_ids and hasattr(v, 'shape'):
                    _safe_deallocate(v)
            self._cache = {}

    def _deallocate_cache_history(self):
        """Deallocate all historical caches (from recurrent core iterations)."""
        _persistent_ids = set()
        if hasattr(self, 'slots') and self.slots is not None:
            _persistent_ids.add(id(self.slots))
        if hasattr(self, 'read_gate') and self.read_gate is not None:
            _persistent_ids.add(id(self.read_gate))
        if hasattr(self, 'write_gate') and self.write_gate is not None:
            _persistent_ids.add(id(self.write_gate))
        # REVIEWED: also exclude cached causal masks (stored in _mask_cache).
        # The same mask objects are referenced from every cache entry —
        # deallocating them from history would free the persistent mask buffers.
        if hasattr(self, '_mask_cache'):
            for rm, wm in self._mask_cache.values():
                _persistent_ids.add(id(rm))
                _persistent_ids.add(id(wm))
        for old_cache in self._cache_history:
            for v in old_cache.values():
                if id(v) not in _persistent_ids and hasattr(v, 'shape'):
                    _safe_deallocate(v)
        self._cache_history = []

    def _reshape_to_heads(self, x, B, L):
        """(B, L, D) -> (B, H, L, d_head)"""
        return ttnn.permute(ttnn.reshape(x, [B, L, self.n_heads, self.d_head]), [0, 2, 1, 3])

    def _reshape_from_heads(self, x, B, L):
        """(B, H, L, d_head) -> (B, L, D)"""
        return ttnn.reshape(ttnn.permute(x, [0, 2, 1, 3]), [B, L, self.d_model])

    def _softmax_backward(self, grad_attn, attn, B, H, L_q, L_k):
        """Manual softmax backward: grad_scores = (grad_attn - sum(grad_attn*attn, -1, keepdim)) * attn"""
        # REVIEWED: intermediates must be explicitly deallocated.
        _ga_mul = ttnn.mul(grad_attn, attn)
        grad_sum = ttnn.sum(_ga_mul, dim=-1)  # (B, H, L_q)
        _safe_deallocate(_ga_mul)
        # REVIEWED: reshape/expand may return views — do NOT deallocate grad_sum.
        grad_sum = ttnn.reshape(grad_sum, [B, H, L_q, 1])
        grad_sum = ttnn.expand(grad_sum, [B, H, L_q, L_k])
        _ga_sub = ttnn.sub(grad_attn, grad_sum)
        _safe_deallocate(grad_sum)
        result = ttnn.mul(_ga_sub, attn)
        _safe_deallocate(_ga_sub)
        return result

    def _l2_normalize_heads(self, x, B, H, L):
        """L2-normalize along d_head dimension. x: (B, H, L, d_head) -> normalized (B, H, L, d_head)

        QK-Norm: normalizes Q and K to unit L2 norm per (B, H, L) position before attention.
        This bounds attention logits regardless of weight magnitudes, preventing entropy
        collapse.  (Henry et al., 2020 — now standard in OLMo 2, Gemma 3, Qwen 3)
        """
        # REVIEWED: intermediates must be explicitly deallocated. This function
        # is called in both forward and backward (for scores_pre recomputation).
        # Compute L2 norm along last dim: ||x|| = sqrt(sum(x^2, dim=-1))
        sq = ttnn.mul(x, x)                              # (B, H, L, d_h)
        norm_sq = ttnn.sum(sq, dim=-1)                   # (B, H, L)
        _safe_deallocate(sq)
        # REVIEWED: reshape/expand may return views — do NOT deallocate norm_sq.
        norm_sq = ttnn.reshape(norm_sq, [B, H, L, 1])
        norm_sq = ttnn.expand(norm_sq, [B, H, L, self.d_head])
        # Add epsilon for numerical stability (avoid div-by-zero)
        # REVIEWED: use cached _eps_tt (persistent) instead of from_torch every call.
        norm_sq = ttnn.add(norm_sq, self._eps_tt)
        # rsqrt via SFPU
        inv_norm = ttnn.rsqrt(norm_sq)                   # (B, H, L, d_h) broadcast
        _safe_deallocate(norm_sq)
        result = ttnn.mul(x, inv_norm)
        _safe_deallocate(inv_norm)
        return result

    def _build_causal_masks(self, T: int, m: int, B: int, H: int, device):
        """Build causal masks for workspace cross-attention.

        The workspace has two cross-attention passes that must respect causality
        in an autoregressive setting:

        1. Read pass: slots (m) attend over sequence positions (T).
           Slot i can only attend to positions [0, (i+1)*T//m - 1].
           This gives each slot a growing receptive field — early slots see
           only early tokens, later slots see more. This prevents slots from
           reading future tokens (including the answer).

        2. Write pass: sequence positions (T) attend over slots (m).
           Position t can only attend to slots [0, (t+1)*m//T - 1], clamped to
           at least slot 0 so no row is fully masked.
           This ensures position t only reads from slots whose receptive field
           includes positions <= t, preventing future information from flowing
           back into earlier positions through the workspace. (For the first
           chunk, t < ceil(T/m) - 1, slot 0 hasn't finished reading yet, so
           those positions unavoidably see a slot whose receptive field
           extends slightly past t, bounded by one chunk width — the
           alternative is an all-masked row, which would NaN the softmax.)

        Returns:
            read_mask: (m, T) — 1.0 where attention is allowed, 0.0 where masked
            write_mask: (T, m) — 1.0 where attention is allowed, 0.0 where masked
        """
        # REVIEWED: Cache masks persistently per (T, m) — they're identical
        # across forward calls and were being recreated 12x per step (Cell C),
        # leaking device memory. The masks are small but the leak compounds.
        cache_key = (T, m, device.id())
        if hasattr(self, '_mask_cache') and cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # Read mask: slot i -> positions [0, (i+1)*T//m - 1]
        read_mask_host = torch.zeros(m, T, dtype=torch.bfloat16)
        for i in range(m):
            cutoff = min((i + 1) * T // m, T)
            read_mask_host[i, :cutoff] = 1.0

        # Write mask: position t -> slots [0, (t+1)*m//T - 1] (clamped to >= 1 slot)
        write_mask_host = torch.zeros(T, m, dtype=torch.bfloat16)
        for t in range(T):
            cutoff = max(min((t + 1) * m // T, m), 1)
            write_mask_host[t, :cutoff] = 1.0

        read_mask = ttnn.from_torch(read_mask_host, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=device)
        write_mask = ttnn.from_torch(write_mask_host, dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=device)
        if not hasattr(self, '_mask_cache'):
            self._mask_cache = {}
        self._mask_cache[cache_key] = (read_mask, write_mask)
        return read_mask, write_mask

    def _l2_normalize_backward(self, grad_normed, x, B, H, L):
        """Backward through L2 normalization.

        For y = x / ||x||, the Jacobian is:
          dy/dx = (I - y*y^T) / ||x||
        So: grad_x = (grad_y - y * (grad_y . y)) / ||x||

        x: (B, H, L, d_head) — original (pre-norm) tensor
        grad_normed: (B, H, L, d_head) — gradient w.r.t. normalized output
        Returns: grad_x (B, H, L, d_head)
        """
        # REVIEWED: This function is called 4x per workspace backward (which
        # is called K times per step in Cell C). All intermediates must be
        # explicitly deallocated to avoid compounding leaks.
        # Recompute y = x / ||x|| and ||x||
        sq = ttnn.mul(x, x)
        norm_sq = ttnn.sum(sq, dim=-1)                   # (B, H, L)
        _safe_deallocate(sq)
        # REVIEWED: reshape/expand may return views — do NOT deallocate
        # norm_sq, norm_sq_scalar (would crash if view shares buffer).
        norm_sq_scalar = ttnn.reshape(norm_sq, [B, H, L, 1])
        norm_sq_expanded = ttnn.expand(norm_sq_scalar, [B, H, L, self.d_head])
        # REVIEWED: use cached _eps_tt (persistent) instead of from_torch every call.
        norm_sq_safe = ttnn.add(norm_sq_expanded, self._eps_tt)
        inv_norm = ttnn.rsqrt(norm_sq_safe)
        _safe_deallocate(norm_sq_safe)
        y = ttnn.mul(x, inv_norm)                        # normalized output

        # dot = grad_y . y  (scalar per position)
        _gy_mul = ttnn.mul(grad_normed, y)
        dot = ttnn.sum(_gy_mul, dim=-1)  # (B, H, L)
        _safe_deallocate(_gy_mul)
        # REVIEWED: reshape/expand may return views — do NOT deallocate dot.
        dot = ttnn.reshape(dot, [B, H, L, 1])
        dot = ttnn.expand(dot, [B, H, L, self.d_head])

        # grad_x = (grad_y - y * dot) * inv_norm
        _y_dot = ttnn.mul(y, dot)
        _safe_deallocate(y)
        grad_x = ttnn.sub(grad_normed, _y_dot)
        _safe_deallocate(_y_dot)
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor.
        grad_x_final = ttnn.mul(grad_x, inv_norm)
        _safe_deallocate(grad_x)
        _safe_deallocate(inv_norm)
        return grad_x_final

    def forward(self, x: "ttnn.Tensor", slot_state: Optional["ttnn.Tensor"]) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """x: (B, T, D), slot_state: (B, m, D) or None -> (x_out, slot_state_out)"""
        B, T, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        H, d_h, m = self.n_heads, self.d_head, self.n_slots
        device = self.device

        # Initialize slots: (B, m, D)
        # Slot permutation: when initializing from learned slot embeddings (slot_state is None),
        # randomly permute the slot indices. This breaks the chicken-and-egg problem where
        # identical slot keys at init force position-based routing. With permutation, the
        # model cannot rely on a fixed position-to-slot mapping and must learn content-based
        # addressing. The permutation is cached for the backward pass to un-permute gradients.
        slot_perm = None
        if slot_state is None:
            slots_host = ttnn.to_torch(self.slots)  # (m, D)
            if self.config.slot_permutation:
                slot_perm = torch.randperm(m)
                slots_host = slots_host[slot_perm]  # gather along slot dim
            slots = to_device(slots_host, device)
            # REVIEWED: reshape/expand create new tensors — deallocate intermediates.
            _slots_reshaped = ttnn.reshape(slots, [1, m, D])
            _safe_deallocate(slots)
            slots = ttnn.expand(_slots_reshaped, [B, m, D])
            _safe_deallocate(_slots_reshaped)
        else:
            slots = slot_state

        # --- Read: slots attend over hidden states ---
        rq = self._reshape_to_heads(ttnn.linear(slots, self.read_q_weight), B, m)   # (B, H, m, d_h)
        rk = self._reshape_to_heads(ttnn.linear(x, self.read_k_weight), B, T)       # (B, H, T, d_h)
        rv = self._reshape_to_heads(ttnn.linear(x, self.read_v_weight), B, T)       # (B, H, T, d_h)

        # QK-Norm: L2-normalize Q and K along d_head before attention
        rq_norm = self._l2_normalize_heads(rq, B, H, m)   # (B, H, m, d_h)
        rk_norm = self._l2_normalize_heads(rk, B, H, T)   # (B, H, T, d_h)

        # Build causal masks for cross-attention (prevents future-token leakage)
        read_mask, write_mask = self._build_causal_masks(T, m, B, H, device)

        read_scores = ttnn.matmul(rq_norm, ttnn.transpose(rk_norm, -2, -1))  # (B, H, m, T)
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor.
        read_scores = ttnn.mul(read_scores, self.read_qk_scale)  # learnable scale
        # Apply causal mask: set masked positions to -1e4 before softmax
        neg_inf_read = ttnn.full((B, H, m, T), -1e4, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device)
        # REVIEWED: reassignment leak — ttnn.where creates a new tensor.
        read_scores_masked = ttnn.where(read_mask, read_scores, neg_inf_read)
        _safe_deallocate(read_scores)
        _safe_deallocate(neg_inf_read)
        read_scores = read_scores_masked
        read_attn = ttnn.softmax(read_scores, dim=-1)               # (B, H, m, T)
        _safe_deallocate(read_scores)
        read_out_4d = ttnn.matmul(read_attn, rv)                    # (B, H, m, d_h)
        read_out = self._reshape_from_heads(read_out_4d, B, m)      # (B, m, D)
        _safe_deallocate(read_out_4d)
        read_out_proj = ttnn.linear(read_out, self.read_out_weight)  # (B, m, D)
        # REVIEWED: read_out is cached for backward — do NOT deallocate.

        # slots = slot_norm(decay * slots + read_gate * read_out_proj)
        # ReZero gate: scalar (no sigmoid), starts at 0.0 (true identity)
        read_gate_val = self.read_gate  # ReZero: direct scalar, no sigmoid
        decayed_slots = ttnn.mul(self.slot_decay, slots)
        # REVIEWED: intermediate must be deallocated after add.
        _rg_ro = ttnn.mul(read_gate_val, read_out_proj)
        slots_pre_norm = ttnn.add(decayed_slots, _rg_ro)
        _safe_deallocate(_rg_ro)
        slots_out = self.slot_norm.forward(slots_pre_norm)

        # --- Write: hidden states attend over slots ---
        wq = self._reshape_to_heads(ttnn.linear(x, self.write_q_weight), B, T)       # (B, H, T, d_h)
        wk = self._reshape_to_heads(ttnn.linear(slots_out, self.write_k_weight), B, m)  # (B, H, m, d_h)
        wv = self._reshape_to_heads(ttnn.linear(slots_out, self.write_v_weight), B, m)  # (B, H, m, d_h)

        # QK-Norm: L2-normalize Q and K along d_head before attention
        wq_norm = self._l2_normalize_heads(wq, B, H, T)   # (B, H, T, d_h)
        wk_norm = self._l2_normalize_heads(wk, B, H, m)   # (B, H, m, d_h)

        write_scores = ttnn.matmul(wq_norm, ttnn.transpose(wk_norm, -2, -1))  # (B, H, T, m)
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor.
        write_scores = ttnn.mul(write_scores, self.write_qk_scale)  # learnable scale
        # Apply causal mask: set masked positions to -1e4 before softmax
        neg_inf_write = ttnn.full((B, H, T, m), -1e4, dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device)
        # REVIEWED: reassignment leak — ttnn.where creates a new tensor.
        write_scores_masked = ttnn.where(write_mask, write_scores, neg_inf_write)
        _safe_deallocate(write_scores)
        _safe_deallocate(neg_inf_write)
        write_scores = write_scores_masked
        write_attn = ttnn.softmax(write_scores, dim=-1)              # (B, H, T, m)
        _safe_deallocate(write_scores)
        write_out_4d = ttnn.matmul(write_attn, wv)                   # (B, H, T, d_h)
        write_out = self._reshape_from_heads(write_out_4d, B, T)     # (B, T, D)
        _safe_deallocate(write_out_4d)
        write_out_proj = ttnn.linear(write_out, self.write_out_weight)  # (B, T, D)
        # REVIEWED: write_out is cached for backward — do NOT deallocate.

        # x = norm(x + write_gate * write_out_proj)
        # ReZero gate: scalar (no sigmoid), starts at 0.0 (true identity)
        write_gate_val = self.write_gate  # ReZero: direct scalar, no sigmoid
        # REVIEWED: intermediate must be deallocated after add.
        _wg_wo = ttnn.mul(write_gate_val, write_out_proj)
        x_pre_norm = ttnn.add(x, _wg_wo)
        _safe_deallocate(_wg_wo)
        x_out = self.norm.forward(x_pre_norm)

        # Save old cache to history before overwriting (forward may be called
        # multiple times in the recurrent core; backward needs each cache)
        if self._cache:
            self._cache_history.append(self._cache)
        # Cache for backward
        self._cache = {
            "x": x, "slots_in": slots, "decayed_slots": decayed_slots,
            "slots_pre_norm": slots_pre_norm,
            "rq": rq, "rk": rk, "rv": rv, "read_attn": read_attn,
            "read_out": read_out, "read_out_proj": read_out_proj,
            "read_gate_val": read_gate_val,
            "slots_out": slots_out,
            "wq": wq, "wk": wk, "wv": wv, "write_attn": write_attn,
            "write_out": write_out, "write_out_proj": write_out_proj,
            "write_gate_val": write_gate_val,
            "x_pre_norm": x_pre_norm,
            "B": B, "T": T, "m": m,
            "slot_perm": slot_perm,
            "read_mask": read_mask, "write_mask": write_mask,
        }

        # Accumulate for regularizers (entropy + diversity)
        self._forward_caches.append({
            "read_attn": read_attn,       # (B, H, m, T)
            "write_attn": write_attn,     # (B, H, T, m)
            "slots_out": slots_out,       # (B, m, D)
        })

        return x_out, slots_out

    def backward(self, grad_x_out: "ttnn.Tensor", grad_slots_out: "ttnn.Tensor",
                 extra_grad_read_attn=None, extra_grad_write_attn=None, extra_grad_slots_out=None
                 ) -> Tuple["ttnn.Tensor", "ttnn.Tensor", Dict[str, "ttnn.Tensor"]]:
        """Backward through workspace module — all on device.

        grad_x_out: (B, T, D)
        grad_slots_out: (B, m, D) — gradient w.r.t. slots_out
        extra_grad_*: optional regularizer gradients to add (from entropy/diversity losses)
        Returns: (grad_x, grad_slots_in, grads_dict)
        """
        c = self._cache
        B, T, m = c["B"], c["T"], c["m"]
        H, d_h, D = self.n_heads, self.d_head, self.d_model
        device = self.device

        # --- Backward through x_out = norm(x_pre_norm) ---
        grad_x_pre_norm, grad_norm_w = self.norm.backward(grad_x_out, c["x_pre_norm"])

        # --- Backward through x_pre_norm = x + write_gate * write_out_proj ---
        # ReZero gate: no sigmoid, gate is a direct scalar
        grad_x_from_write = grad_x_pre_norm  # residual
        grad_write_out_proj = ttnn.mul(grad_x_pre_norm, c["write_gate_val"])

        # grad_write_gate = sum(grad_x_pre_norm * write_out_proj)  (no sigmoid derivative)
        grad_write_gate = ttnn.mul(grad_x_pre_norm, c["write_out_proj"])
        # REVIEWED: nested sum reassignment leak — each ttnn.sum creates a new
        # tensor; intermediate results must be explicitly deallocated.
        _gw = ttnn.sum(grad_write_gate, dim=0)
        _safe_deallocate(grad_write_gate)
        _gw2 = ttnn.sum(_gw, dim=0)
        _safe_deallocate(_gw)
        grad_write_gate = ttnn.sum(_gw2, dim=0)
        _safe_deallocate(_gw2)

        # --- Backward through write_out_proj = linear(write_out) ---
        grad_write_out = ttnn.linear(grad_write_out_proj, ttnn.transpose(self.write_out_weight, 0, 1))
        grad_write_out_2d = ttnn.reshape(c["write_out"], [B * T, D])
        grad_write_out_proj_2d = ttnn.reshape(grad_write_out_proj, [B * T, D])
        grad_write_out_weight = ttnn.matmul(ttnn.transpose(grad_write_out_2d, 0, 1), grad_write_out_proj_2d)

        # --- Backward through write_out = reshape(write_attn @ wv) ---
        grad_write_out_4d = self._reshape_to_heads(grad_write_out, B, T)  # (B, H, T, d_h)
        wv = c["wv"]
        grad_write_attn = ttnn.matmul(grad_write_out_4d, ttnn.transpose(wv, -2, -1))  # (B, H, T, m)
        if extra_grad_write_attn is not None:
            # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
            _gwa = ttnn.add(grad_write_attn, extra_grad_write_attn)
            _safe_deallocate(grad_write_attn)
            _safe_deallocate(extra_grad_write_attn)
            grad_write_attn = _gwa
        grad_wv = ttnn.matmul(ttnn.transpose(c["write_attn"], -2, -1), grad_write_out_4d)  # (B, H, m, d_h)

        # --- Backward through write_attn = softmax(write_scores * write_qk_scale) ---
        # softmax_backward gives d(L)/d(scores) where scores = scores_pre * qk_scale
        grad_write_scores = self._softmax_backward(grad_write_attn, c["write_attn"], B, H, T, m)

        # Zero out gradients at causally masked positions (same as attention layer).
        # After softmax, masked positions have attn=0, so the softmax backward
        # already produces ~0 there, but we zero explicitly to avoid spurious
        # gradients from bf16 rounding (matches TTAttentionLayer backward pattern).
        write_mask = c["write_mask"]
        zeros_write = ttnn.zeros((B, H, T, m), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device)
        # REVIEWED: reassignment leak — ttnn.where creates a new tensor.
        _gws = ttnn.where(write_mask, grad_write_scores, zeros_write)
        _safe_deallocate(grad_write_scores)
        grad_write_scores = _gws
        _safe_deallocate(zeros_write)

        # grad_qk_scale = sum(d(L)/d(scores) * scores_pre) — compute BEFORE scaling.
        # IMPORTANT: order matters here. grad_write_scores below is still
        # d(L)/d(scores) (pre-qk_scale) at this point; the next line
        # overwrites it in place with d(L)/d(scores_pre) = d(L)/d(scores) *
        # qk_scale, so grad_qk_scale MUST be computed first or it silently
        # uses the wrong (already-scaled) gradient.
        wq = c["wq"]
        wk = c["wk"]
        wq_norm = self._l2_normalize_heads(wq, B, H, T)
        wk_norm = self._l2_normalize_heads(wk, B, H, m)
        write_scores_pre = ttnn.matmul(wq_norm, ttnn.transpose(wk_norm, -2, -1))
        # write_qk_scale is a scalar param: ttnn.sum with no `dim` reduces
        # over ALL dims (B,H,T,m) -> scalar, per ttnn.sum's documented
        # default behavior (unlike torch.sum this is NOT a no-op/identity).
        _gws_mul = ttnn.mul(grad_write_scores, write_scores_pre)
        grad_write_qk_scale = ttnn.sum(_gws_mul)
        _safe_deallocate(_gws_mul)
        _safe_deallocate(write_scores_pre)

        # Now scale: d(L)/d(scores_pre) = d(L)/d(scores) * qk_scale
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor.
        _gws_scaled = ttnn.mul(grad_write_scores, self.write_qk_scale)
        _safe_deallocate(grad_write_scores)
        grad_write_scores = _gws_scaled

        # grad_wq_norm = d(L)/d(scores_pre) @ wk_norm, etc.
        grad_wq_norm = ttnn.matmul(grad_write_scores, wk_norm)               # (B, H, T, d_h)
        grad_wk_norm = ttnn.matmul(ttnn.transpose(grad_write_scores, -2, -1), wq_norm)  # (B, H, m, d_h)
        _safe_deallocate(wq_norm)
        _safe_deallocate(wk_norm)

        # --- Backward through QK-Norm: wq_norm = L2_normalize(wq), wk_norm = L2_normalize(wk) ---
        grad_wq = self._l2_normalize_backward(grad_wq_norm, wq, B, H, T)
        grad_wk = self._l2_normalize_backward(grad_wk_norm, wk, B, H, m)
        # REVIEWED: grad_wq_norm/grad_wk_norm no longer needed after backward.
        _safe_deallocate(grad_wq_norm)
        _safe_deallocate(grad_wk_norm)

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

        # Total grad_slots_out (from write path + incoming grad_slots_out + regularizer)
        # REVIEWED: intermediate tensors must be explicitly deallocated.
        _gs_wk_wv = ttnn.add(grad_slots_from_wk, grad_slots_from_wv)
        _safe_deallocate(grad_slots_from_wk)
        _safe_deallocate(grad_slots_from_wv)
        grad_slots_total = ttnn.add(grad_slots_out, _gs_wk_wv)
        _safe_deallocate(_gs_wk_wv)
        if extra_grad_slots_out is not None:
            # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
            _gst = ttnn.add(grad_slots_total, extra_grad_slots_out)
            _safe_deallocate(grad_slots_total)
            _safe_deallocate(extra_grad_slots_out)
            grad_slots_total = _gst

        # --- Backward through slots_out = slot_norm(slots_pre_norm) ---
        grad_slots_pre_norm, grad_slot_norm_w = self.slot_norm.backward(grad_slots_total, c["slots_pre_norm"])

        # --- Backward through slots_pre_norm = decay * slots_in + read_gate * read_out_proj ---
        # ReZero gate: no sigmoid, gate is a direct scalar
        grad_slots_in_from_read = ttnn.mul(grad_slots_pre_norm, self.slot_decay)  # residual * decay
        grad_read_out_proj = ttnn.mul(grad_slots_pre_norm, c["read_gate_val"])

        # grad_read_gate = sum(grad_slots_pre_norm * read_out_proj)  (no sigmoid derivative)
        grad_read_gate = ttnn.mul(grad_slots_pre_norm, c["read_out_proj"])
        # REVIEWED: nested sum reassignment leak — deallocate intermediates.
        _grg = ttnn.sum(grad_read_gate, dim=0)
        _safe_deallocate(grad_read_gate)
        _grg2 = ttnn.sum(_grg, dim=0)
        _safe_deallocate(_grg)
        grad_read_gate = ttnn.sum(_grg2, dim=0)
        _safe_deallocate(_grg2)

        # grad_slot_decay = sum(grad_slots_pre_norm * slots_in)
        grad_slot_decay = ttnn.mul(grad_slots_pre_norm, c["slots_in"])
        # REVIEWED: nested sum reassignment leak — deallocate intermediates.
        _gsd = ttnn.sum(grad_slot_decay, dim=0)
        _safe_deallocate(grad_slot_decay)
        _gsd2 = ttnn.sum(_gsd, dim=0)
        _safe_deallocate(_gsd)
        grad_slot_decay = ttnn.sum(_gsd2, dim=0)
        _safe_deallocate(_gsd2)

        # --- Backward through read_out_proj = linear(read_out) ---
        grad_read_out = ttnn.linear(grad_read_out_proj, ttnn.transpose(self.read_out_weight, 0, 1))
        read_out_2d = ttnn.reshape(c["read_out"], [B * m, D])
        grad_read_out_proj_2d = ttnn.reshape(grad_read_out_proj, [B * m, D])
        grad_read_out_weight = ttnn.matmul(ttnn.transpose(read_out_2d, 0, 1), grad_read_out_proj_2d)

        # --- Backward through read_out = reshape(read_attn @ rv) ---
        grad_read_out_4d = self._reshape_to_heads(grad_read_out, B, m)  # (B, H, m, d_h)
        rv = c["rv"]
        grad_read_attn = ttnn.matmul(grad_read_out_4d, ttnn.transpose(rv, -2, -1))  # (B, H, m, T)
        if extra_grad_read_attn is not None:
            # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
            _gra = ttnn.add(grad_read_attn, extra_grad_read_attn)
            _safe_deallocate(grad_read_attn)
            _safe_deallocate(extra_grad_read_attn)
            grad_read_attn = _gra
        grad_rv = ttnn.matmul(ttnn.transpose(c["read_attn"], -2, -1), grad_read_out_4d)  # (B, H, T, d_h)

        # --- Backward through read_attn = softmax(read_scores * read_qk_scale) ---
        # softmax_backward gives d(L)/d(scores) where scores = scores_pre * qk_scale
        grad_read_scores = self._softmax_backward(grad_read_attn, c["read_attn"], B, H, m, T)

        # Zero out gradients at causally masked positions (same as attention layer).
        read_mask = c["read_mask"]
        zeros_read = ttnn.zeros((B, H, m, T), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
        # REVIEWED: reassignment leak — ttnn.where creates a new tensor.
        _grs = ttnn.where(read_mask, grad_read_scores, zeros_read)
        _safe_deallocate(grad_read_scores)
        grad_read_scores = _grs
        _safe_deallocate(zeros_read)

        # grad_qk_scale = sum(d(L)/d(scores) * scores_pre) — compute BEFORE
        # scaling (see the identical note in the write-pass backward above:
        # order matters, grad_read_scores gets overwritten in-place below).
        rk = c["rk"]
        rq = c["rq"]
        rq_norm = self._l2_normalize_heads(rq, B, H, m)
        rk_norm = self._l2_normalize_heads(rk, B, H, T)
        read_scores_pre = ttnn.matmul(rq_norm, ttnn.transpose(rk_norm, -2, -1))
        # ttnn.sum with no `dim` reduces over ALL dims -> scalar (read_qk_scale is a scalar param).
        _grs_mul = ttnn.mul(grad_read_scores, read_scores_pre)
        grad_read_qk_scale = ttnn.sum(_grs_mul)
        _safe_deallocate(_grs_mul)
        _safe_deallocate(read_scores_pre)

        # Now scale: d(L)/d(scores_pre) = d(L)/d(scores) * qk_scale
        # REVIEWED: reassignment leak — ttnn.mul creates a new tensor.
        _grs_scaled = ttnn.mul(grad_read_scores, self.read_qk_scale)
        _safe_deallocate(grad_read_scores)
        grad_read_scores = _grs_scaled

        # grad_rq_norm = d(L)/d(scores_pre) @ rk_norm, etc.
        grad_rq_norm = ttnn.matmul(grad_read_scores, rk_norm)               # (B, H, m, d_h)
        grad_rk_norm = ttnn.matmul(ttnn.transpose(grad_read_scores, -2, -1), rq_norm)  # (B, H, T, d_h)
        _safe_deallocate(rq_norm)
        _safe_deallocate(rk_norm)

        # --- Backward through QK-Norm: rq_norm = L2_normalize(rq), rk_norm = L2_normalize(rk) ---
        grad_rq = self._l2_normalize_backward(grad_rq_norm, rq, B, H, m)
        grad_rk = self._l2_normalize_backward(grad_rk_norm, rk, B, H, T)
        # REVIEWED: grad_rq_norm/grad_rk_norm no longer needed after backward.
        _safe_deallocate(grad_rq_norm)
        _safe_deallocate(grad_rk_norm)

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
        # REVIEWED: intermediate tensors must be explicitly deallocated.
        _gx_write_wq = ttnn.add(grad_x_from_write, grad_x_from_wq)
        _safe_deallocate(grad_x_from_write)
        _safe_deallocate(grad_x_from_wq)
        _gx_rk_rv = ttnn.add(grad_x_from_rk, grad_x_from_rv)
        _safe_deallocate(grad_x_from_rk)
        _safe_deallocate(grad_x_from_rv)
        grad_x = ttnn.add(_gx_write_wq, _gx_rk_rv)
        _safe_deallocate(_gx_write_wq)
        _safe_deallocate(_gx_rk_rv)
        grad_slots_in = ttnn.add(grad_slots_in_from_read, grad_slots_from_rq)
        _safe_deallocate(grad_slots_in_from_read)
        _safe_deallocate(grad_slots_from_rq)

        # Un-permute grad_slots_in if slot permutation was applied in forward.
        # The forward did: slots_permuted = slots_original[perm]
        # So: grad_slots_original[perm] = grad_slots_permuted
        # Which means: grad_slots_original = grad_slots_permuted[inv_perm]
        slot_perm = c.get("slot_perm", None)
        if slot_perm is not None:
            inv_perm = torch.argsort(slot_perm)
            grad_slots_in_host = ttnn.to_torch(grad_slots_in)  # (B, m, D)
            grad_slots_in_host = grad_slots_in_host[:, inv_perm, :]  # un-permute slot dim
            # REVIEWED: reassignment leak — old grad_slots_in is a device tensor.
            _safe_deallocate(grad_slots_in)
            grad_slots_in = to_device(grad_slots_in_host, device)

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
            "slot_decay": grad_slot_decay,
            "read_qk_scale": grad_read_qk_scale,
            "write_qk_scale": grad_write_qk_scale,
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
            "slot_decay": self.slot_decay,
            "read_qk_scale": self.read_qk_scale, "write_qk_scale": self.write_qk_scale,
        }

    def normalize_slots(self):
        """Normalize slot parameters to unit RMS to prevent unbounded growth.

        This breaks the feedback loop where large slots → sharp attention →
        large gradients → larger slots. Called after each optimizer step.
        Uses fixed weight=1 (not learned) so the slots are constrained to
        unit RMS regardless of training dynamics.
        """
        old_slots = self.slots
        self.slots = ttnn.rms_norm(self.slots, weight=self._slot_param_norm_weight, epsilon=1e-6)
        _safe_deallocate(old_slots)

    def spectral_normalize_weights(self):
        """Apply spectral normalization to all workspace weight matrices.

        Divides each weight matrix by its spectral norm (largest singular value),
        then scales to the configured bound. This constrains the Lipschitz constant
        to spectral_norm_bound, preventing unbounded weight growth while allowing
        sufficient attention selectivity.

        With bound C, unit-RMS slots, and LayerNorm on inputs, attention logits are
        bounded by:
            |logit| <= C² * ||slots|| * ||x|| / √d_h
                     <= C² / √d_h  (with slot norm + LayerNorm)

        For C=5, d_h=96: max logit ≈ 2.55, max attention ratio ≈ 164:1 — enough
        for selective attention, but preventing the e^100+ ratios that cause
        gradient explosion.

        Uses power iteration (10 steps) for on-device computation — no host
        transfer needed. This is the same algorithm used by Spectral Normalization
        GANs (Miyato et al., 2018).
        """
        device = self.device
        weight_names = [
            "read_q_weight", "read_k_weight", "read_v_weight", "read_out_weight",
            "write_q_weight", "write_k_weight", "write_v_weight", "write_out_weight",
        ]

        eps_tt = ttnn.from_torch(
            torch.tensor([1e-6], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        bound_tt = ttnn.from_torch(
            torch.tensor([self.spectral_norm_bound], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

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
            Wt = ttnn.transpose(W, 0, 1)  # cache transpose outside loop

            for _ in range(10):
                # u = W @ v, then normalize
                u = ttnn.matmul(W, v)  # (D, 1)
                u_sq = ttnn.mul(u, u)
                u_norm = ttnn.sqrt(ttnn.sum(u_sq))
                _safe_deallocate(u_sq)
                u_norm_c = ttnn.maximum(u_norm, eps_tt)
                _safe_deallocate(u_norm)
                u_recip = ttnn.reciprocal(u_norm_c)
                _safe_deallocate(u_norm_c)
                u_new = ttnn.mul(u, u_recip)
                _safe_deallocate(u)
                _safe_deallocate(u_recip)
                u = u_new

                # v = W^T @ u, then normalize
                v_new = ttnn.matmul(Wt, u)  # (D, 1)
                v_sq = ttnn.mul(v_new, v_new)
                v_norm = ttnn.sqrt(ttnn.sum(v_sq))
                _safe_deallocate(v_sq)
                v_norm_c = ttnn.maximum(v_norm, eps_tt)
                _safe_deallocate(v_norm)
                v_recip = ttnn.reciprocal(v_norm_c)
                _safe_deallocate(v_norm_c)
                v_result = ttnn.mul(v_new, v_recip)
                _safe_deallocate(v)
                _safe_deallocate(v_new)
                _safe_deallocate(v_recip)
                v = v_result
            _safe_deallocate(Wt)
            _safe_deallocate(u)

            # σ_max = ||W @ v|| (v is now the top right singular vector)
            Wv = ttnn.matmul(W, v)  # (D, 1)
            Wv_sq = ttnn.mul(Wv, Wv)
            sigma = ttnn.sqrt(ttnn.sum(Wv_sq))
            _safe_deallocate(Wv_sq)
            sigma_c = ttnn.maximum(sigma, eps_tt)
            _safe_deallocate(sigma)

            # Cap spectral norm at bound: if sigma > bound, scale down to bound.
            # If sigma <= bound, leave weights unchanged (cap, not target).
            # This lets the optimizer grow weights naturally up to the bound,
            # and prevents a sudden scaling jump when resuming from a checkpoint
            # saved with a different (or no) bound.
            # scale = min(1, bound/sigma) = bound / max(bound, sigma)
            # REVIEWED: nested ttnn.maximum creates an intermediate — deallocate it.
            _max_bound_sigma = ttnn.maximum(bound_tt, sigma_c)
            scale_factor = ttnn.mul(bound_tt, ttnn.reciprocal(_max_bound_sigma))
            _safe_deallocate(_max_bound_sigma)
            W_normalized = ttnn.mul(W, scale_factor)
            _safe_deallocate(getattr(self, name))
            setattr(self, name, W_normalized)
            # Deallocate intermediates from power iteration
            _safe_deallocate(v)
            _safe_deallocate(Wv)
            _safe_deallocate(sigma_c)
            _safe_deallocate(scale_factor)

        _safe_deallocate(eps_tt)
        _safe_deallocate(bound_tt)

    def clear_caches(self):
        """Clear cached tensors from the last forward pass.

        Explicitly deallocate cached intermediates to free device buffers.
        Model parameters (gates, slots, weights) survive because they're still
        referenced by self.read_gate, self.slots, etc.
        """
        # Persistent tensors stored in cache that must NOT be deallocated:
        # slots_in = self.slots (parameter), read_gate_val = self.read_gate,
        # write_gate_val = self.write_gate
        # REVIEWED: also exclude cached causal masks (stored in _mask_cache).
        _persistent_ids = set()
        if hasattr(self, 'slots') and self.slots is not None:
            _persistent_ids.add(id(self.slots))
        if hasattr(self, 'read_gate') and self.read_gate is not None:
            _persistent_ids.add(id(self.read_gate))
        if hasattr(self, 'write_gate') and self.write_gate is not None:
            _persistent_ids.add(id(self.write_gate))
        # Add cached causal mask tensors to persistent set
        if hasattr(self, '_mask_cache'):
            for rm, wm in self._mask_cache.values():
                _persistent_ids.add(id(rm))
                _persistent_ids.add(id(wm))
        # Deallocate backward cache intermediates
        for k, v in self._cache.items():
            if id(v) not in _persistent_ids and hasattr(v, 'shape'):
                _safe_deallocate(v)
        self._cache = {}
        # Deallocate cache history from recurrent core iterations
        self._deallocate_cache_history()
        # Deallocate forward caches (regularizer intermediates)
        for fc in self._forward_caches:
            for k, v in fc.items():
                if id(v) not in _persistent_ids and hasattr(v, 'shape'):
                    _safe_deallocate(v)
        self._forward_caches = []
        # REVIEWED: clear any unconsumed regularizer gradients (host-side
        # PyTorch tensors). These should normally be empty after backward,
        # but if forward/backward call counts mismatch, items can remain.
        self._reg_grads = None

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        # NOTE: params are matched to attributes generically via `hasattr` —
        # there is no explicit name allowlist here, so this stays in sync
        # with get_params() automatically as fields are added (e.g.
        # read_qk_scale/write_qk_scale). Only "ws_norm_weight" and
        # "ws_slot_norm_weight" need special-casing because they map to a
        # nested RMSNorm submodule's `.weight` rather than a same-named
        # attribute on self.
        for k, v in params.items():
            if k == "ws_norm_weight":
                _safe_deallocate(self.norm.weight)
                self.norm.weight = v
            elif k == "ws_slot_norm_weight":
                _safe_deallocate(self.slot_norm.weight)
                self.slot_norm.weight = v
            elif hasattr(self, k):
                _safe_deallocate(getattr(self, k))
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Attention Residual Core (Kimi K3 style)
# ---------------------------------------------------------------------------

class AttentionResidual:
    """Attention Residuals for the recurrent core — replaces fixed blend.

    Instead of blending iteration outputs with a fixed linear combination
    (x = blend * x_new + (1 - blend) * x), this module computes softmax
    attention over all iteration outputs (including the pre-core input),
    using a learned query vector that is not input-dependent.

    This gives every iteration selective access to all previous iteration
    representations, with bounded output magnitude (softmax normalizes)
    and consistent gradient magnitude across K iterations — directly
    addressing the gradient amplification that caused Cell C divergence.

    Based on Kimi K3's Attention Residuals (Bachlechner et al., 2020;
    Moonshot AI, 2026). See kimi-k3-relevance-notes.md §2.

    Forward:
        Given K+1 outputs [x_0, x_1, ..., x_K] (each (B, T, d_model)):
        scores_k = sum_d(x_k * query) * scale       → (B, T) per k
        alpha = softmax([scores_0, ..., scores_K])   → (B, T, K+1)
        x_final = sum_k alpha_k * x_k                → (B, T, d_model)

    The query is a single (d_model,) vector shared across all positions and
    batches. The scale is a learnable scalar (init 1/sqrt(d_model)).
    """

    def __init__(self, d_model: int, device, k_max: int):
        self.d_model = d_model
        self.device = device
        self.k_max = k_max  # max number of iterations (for mask precomputation)

        # Learned query vector: (d_model,) → stored as (1, d_model) for tiling
        # Init scale 1/sqrt(d_model) so ||query|| ≈ 1 (unit norm)
        query_init = torch.randn(1, d_model, dtype=torch.bfloat16) / math.sqrt(d_model)
        self.query = ttnn.from_torch(query_init, dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=device)

        # Learnable scale: init 1.0 (not 1/sqrt(d_model)).
        # With 1/sqrt(D)=0.051, softmax scores are nearly uniform (max
        # difference ~0.003), so softmax backward gradients are ~0 and the
        # AR parameters never learn — confirmed frozen across 1600 steps.
        # With scale=1.0, the scores have meaningful spread (||x||≈1,
        # ||query||≈1, so scores ~O(1)), giving the softmax real gradient
        # signal to differentiate between iterations.
        scale_init = 1.0
        self.scale = ttnn.from_torch(torch.tensor([scale_init], dtype=torch.bfloat16),
                                      dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Cache for backward
        self._cache = {}

    def _deallocate_cache(self):
        """Deallocate cached intermediates, excluding persistent tensors."""
        if not self._cache:
            return
        _skip_keys = {"query_b", "K_active", "B", "T", "D"}
        for k, v in self._cache.items():
            if k not in _skip_keys and hasattr(v, 'shape'):
                _safe_deallocate(v)
        self._cache = {}

    def _mask_scores(self, scores_list, K_active, B, T):
        """Apply mask to scores: set inactive iteration scores to -1e4.

        scores_list: list of (B, T, 1) tensors, length k_max+1
        K_active: number of active iterations (scores 0..K_active are kept)
        Returns: list of masked (B, T, 1) scores
        """
        device = self.device
        masked = []
        for k, score in enumerate(scores_list):
            if k <= K_active:
                masked.append(score)
            else:
                # Replace with -1e4 so softmax gives ~0 weight.
                # NOTE: this must be an actual -1e4 constant, not
                # `zeros_like(score) * -1e4` (== 0, not -1e4) — that silent
                # bug let masked/inactive iterations receive non-negligible
                # softmax weight instead of being suppressed.
                shape = [int(d) for d in score.shape]
                masked.append(ttnn.full(shape, -1e4, dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=device))
        return masked

    def forward(self, x_outputs, K_active) -> "ttnn.Tensor":
        """Compute attention residual over iteration outputs.

        Args:
            x_outputs: list of (B, T, d_model) tensors, length k_max+1.
                       x_outputs[0] is the pre-core input, [1..K] are iteration outputs.
            K_active: number of active iterations (0..K_active are included in attention)

        Returns: (B, T, d_model) — attention-weighted combination
        """
        device = self.device
        D = self.d_model
        n_outputs = len(x_outputs)  # k_max + 1

        # Reshape query for broadcasting: (1, d_model) → (1, 1, d_model)
        query_b = ttnn.reshape(self.query, [1, 1, D])

        # Compute scores: score_k = sum_d(x_k * query) for each (B, T) position
        # REVIEWED: intermediates must be explicitly deallocated.
        scores_list = []
        scores_pre_scale = []  # before scale (for backward)
        for k, x_k in enumerate(x_outputs):
            # x_k: (B, T, D), query_b: (1, 1, D) → broadcast mul → (B, T, D)
            xq = ttnn.mul(x_k, query_b)  # (B, T, D)
            score_k = ttnn.sum(xq, dim=-1)  # (B, T) — drops last dim
            _safe_deallocate(xq)
            # Reshape to (B, T, 1) for concat
            B_k, T_k = int(score_k.shape[0]), int(score_k.shape[1])
            # REVIEWED: reshape may return a view — do NOT deallocate score_k.
            score_k = ttnn.reshape(score_k, [B_k, T_k, 1])
            scores_pre_scale.append(score_k)
            # Apply scale
            score_k_scaled = ttnn.mul(score_k, self.scale)
            scores_list.append(score_k_scaled)

        # Mask inactive iterations
        scores_masked = self._mask_scores(scores_list, K_active, int(x_outputs[0].shape[0]), int(x_outputs[0].shape[1]))
        # REVIEWED: _mask_scores returns the same tensors for active iterations
        # and new ttnn.full tensors for inactive ones. The original scaled scores
        # for active iterations are now referenced by scores_masked, so we can
        # safely deallocate the scores_list entries for inactive iterations
        # (they've been replaced by ttnn.full tensors). However, since
        # scores_masked[k] IS scores_list[k] for k <= K_active (same object),
        # we must NOT deallocate those. The inactive ones (k > K_active) in
        # scores_list are not referenced by scores_masked, so deallocate them.
        for k in range(K_active + 1, len(scores_list)):
            _safe_deallocate(scores_list[k])

        # Concat scores: (B, T, n_outputs)
        scores = ttnn.concat(scores_masked, dim=-1)
        # REVIEWED: scores_masked items may be views/aliases — do NOT deallocate
        # them individually. They'll be cleaned up in _deallocate_cache.

        # Softmax over the n_outputs dimension
        alpha = ttnn.softmax(scores, dim=-1)  # (B, T, n_outputs)
        _safe_deallocate(scores)

        # Weighted sum: x_final = sum_k alpha_k * x_k
        B, T = int(x_outputs[0].shape[0]), int(x_outputs[0].shape[1])
        x_final = None
        for k, x_k in enumerate(x_outputs):
            # Slice alpha_k: (B, T, 1)
            alpha_k = ttnn.slice(alpha, [0, 0, k], [B, T, k + 1])  # (B, T, 1)
            # Broadcast multiply: (B, T, D) * (B, T, 1) → (B, T, D)
            term = ttnn.mul(x_k, alpha_k)
            _safe_deallocate(alpha_k)
            if x_final is None:
                x_final = term
            else:
                # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
                _xf_new = ttnn.add(x_final, term)
                _safe_deallocate(x_final)
                _safe_deallocate(term)
                x_final = _xf_new

        # Cache for backward
        self._cache = {
            "x_outputs": x_outputs,
            "scores_pre_scale": scores_pre_scale,  # before scale, before mask
            "scores_masked": scores_masked,  # after scale, after mask
            "alpha": alpha,
            "K_active": K_active,
            "B": B, "T": T, "D": D,
            "query_b": query_b,
        }

        return x_final

    def backward(self, grad_x_final):
        """Backward through attention residual.

        Args:
            grad_x_final: (B, T, d_model) — gradient w.r.t. attention output

        Returns:
            grad_x_list: list of (B, T, d_model) — gradients w.r.t. each x_k
            grads: dict with gradients for query and scale
        """
        c = self._cache
        x_outputs = c["x_outputs"]
        scores_pre_scale = c["scores_pre_scale"]
        scores_masked = c["scores_masked"]
        alpha = c["alpha"]
        K_active = c["K_active"]
        B, T, D = c["B"], c["T"], c["D"]
        query_b = c["query_b"]
        device = self.device
        n_outputs = len(x_outputs)

        # --- Through x_final = sum_k alpha_k * x_k ---
        # grad_alpha_k = sum_d(grad_x_final * x_k) → (B, T, 1) per k
        # grad_x_k_from_v = alpha_k * grad_x_final → (B, T, D)
        # REVIEWED: intermediates must be explicitly deallocated.
        grad_alpha_list = []
        grad_x_from_v = []
        for k, x_k in enumerate(x_outputs):
            # grad_alpha_k = (grad_x_final * x_k).sum(dim=-1)
            gx_xk = ttnn.mul(grad_x_final, x_k)  # (B, T, D)
            grad_alpha_k = ttnn.sum(gx_xk, dim=-1)  # (B, T)
            _safe_deallocate(gx_xk)
            # REVIEWED: reshape may return a view — do NOT deallocate grad_alpha_k.
            grad_alpha_k = ttnn.reshape(grad_alpha_k, [B, T, 1])
            grad_alpha_list.append(grad_alpha_k)

            # grad_x_k_from_v = alpha_k * grad_x_final
            alpha_k = ttnn.slice(alpha, [0, 0, k], [B, T, k + 1])  # (B, T, 1)
            grad_xk_v = ttnn.mul(alpha_k, grad_x_final)  # (B, T, D)
            # REVIEWED: slice creates a new tensor — deallocate after use.
            _safe_deallocate(alpha_k)
            grad_x_from_v.append(grad_xk_v)

        # Concat grad_alpha: (B, T, n_outputs)
        grad_alpha = ttnn.concat(grad_alpha_list, dim=-1)
        # REVIEWED: deallocate individual grad_alpha items after concat.
        for gak in grad_alpha_list:
            _safe_deallocate(gak)

        # --- Through alpha = softmax(scores_masked) ---
        # Manual softmax backward: grad_scores = (grad_alpha - sum(grad_alpha*alpha, -1, keepdim)) * alpha
        # REVIEWED: intermediates must be explicitly deallocated.
        _ga_mul = ttnn.mul(grad_alpha, alpha)
        grad_sum = ttnn.sum(_ga_mul, dim=-1)  # (B, T)
        _safe_deallocate(_ga_mul)
        # REVIEWED: reshape/expand may return views — do NOT deallocate grad_sum.
        grad_sum = ttnn.reshape(grad_sum, [B, T, 1])
        grad_sum = ttnn.expand(grad_sum, [B, T, n_outputs])
        _ga_sub = ttnn.sub(grad_alpha, grad_sum)
        _safe_deallocate(grad_sum)
        grad_scores = ttnn.mul(_ga_sub, alpha)  # (B, T, n_outputs)
        _safe_deallocate(_ga_sub)

        # --- Through scores_masked = scores_list * mask_or_keep ---
        # For active k: scores_masked_k = scores_list_k (no change)
        # For inactive k: scores_masked_k = -1e4 (constant, grad = 0)
        # So grad_scores_list_k = grad_scores_k for active, 0 for inactive
        # Also, scores_list_k = scores_pre_scale_k * scale
        # grad_scale = sum(grad_scores_k * scores_pre_scale_k) for active k
        grad_scale = ttnn.from_torch(
            torch.tensor([0.0], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        grad_scores_pre_scale_list = []
        for k in range(n_outputs):
            grad_score_k = ttnn.slice(grad_scores, [0, 0, k], [B, T, k + 1])  # (B, T, 1)
            if k <= K_active:
                # grad_scale += grad_score_k * scores_pre_scale_k
                # REVIEWED: reassignment + intermediate leak.
                _gs_mul = ttnn.mul(grad_score_k, scores_pre_scale[k])
                _gs_sum = ttnn.sum(_gs_mul)
                _safe_deallocate(_gs_mul)
                _gs_new = ttnn.add(grad_scale, _gs_sum)
                _safe_deallocate(grad_scale)
                _safe_deallocate(_gs_sum)
                grad_scale = _gs_new
                # grad_scores_pre_scale_k = grad_score_k * scale
                grad_spk = ttnn.mul(grad_score_k, self.scale)
            else:
                # Inactive: no gradient flows to scores_pre_scale
                grad_spk = ttnn.zeros_like(grad_score_k)
            # REVIEWED: slice creates a new tensor — deallocate after use.
            _safe_deallocate(grad_score_k)
            grad_scores_pre_scale_list.append(grad_spk)

        # --- Through scores_pre_scale_k = sum_d(x_k * query) ---
        # grad_x_k_from_score = grad_scores_pre_scale_k * query  (broadcast)
        # grad_query += sum over k, positions of grad_scores_pre_scale_k * x_k
        grad_query = ttnn.from_torch(
            torch.zeros(1, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        grad_x_list = []
        for k, x_k in enumerate(x_outputs):
            grad_spk = grad_scores_pre_scale_list[k]  # (B, T, 1)
            # grad_x_k_from_score = grad_spk * query (broadcast over D)
            # query_b is (1, 1, D), grad_spk is (B, T, 1) → mul broadcasts → (B, T, D)
            grad_xk_s = ttnn.mul(grad_spk, query_b)  # (B, T, D)

            # grad_query += sum over (B, T) of grad_spk * x_k
            # grad_spk: (B, T, 1), x_k: (B, T, D) → mul → (B, T, D) → sum to (1, D)
            gq_term = ttnn.mul(grad_spk, x_k)  # (B, T, D)
            # REVIEWED: grad_spk no longer needed — deallocate it.
            _safe_deallocate(grad_spk)
            # REVIEWED: nested sum reassignment leak — deallocate intermediates.
            gq_term_2 = ttnn.sum(gq_term, dim=0)
            _safe_deallocate(gq_term)
            gq_term_3 = ttnn.sum(gq_term_2, dim=0)
            _safe_deallocate(gq_term_2)
            # REVIEWED: reshape may return a view — do NOT deallocate gq_term_3.
            gq_term = ttnn.reshape(gq_term_3, [1, D])
            # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
            _gq_new = ttnn.add(grad_query, gq_term)
            _safe_deallocate(grad_query)
            _safe_deallocate(gq_term)
            grad_query = _gq_new

            # Total grad_x_k = grad_x_k_from_v + grad_x_k_from_score
            # REVIEWED: intermediate grad_xk_s must be deallocated after add.
            grad_x_k = ttnn.add(grad_x_from_v[k], grad_xk_s)
            _safe_deallocate(grad_xk_s)
            _safe_deallocate(grad_x_from_v[k])
            grad_x_list.append(grad_x_k)

        grads = {
            "ar_query": grad_query,
            "ar_scale": grad_scale,
        }

        return grad_x_list, grads

    def clear_caches(self):
        """Clear cached tensors from the last forward pass.

        Explicitly deallocate cached intermediates (x_outputs, scores, alphas)
        to free device buffers. Model parameters (query, scale) survive because
        they're still referenced by self.query, self.scale.
        """
        # query_b is a reshape of self.query (parameter) — may share the same
        # device buffer. Deallocating it would free self.query's buffer.
        # K_active, B, T, D are ints (not tensors).
        _skip_keys = {"query_b", "K_active", "B", "T", "D"}
        for k, v in self._cache.items():
            if k not in _skip_keys and hasattr(v, 'shape'):
                _safe_deallocate(v)
        self._cache = {}

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        return {"ar_query": self.query, "ar_scale": self.scale}

    def set_params(self, params: Dict[str, "ttnn.Tensor"]):
        if "ar_query" in params:
            _safe_deallocate(self.query)
            self.query = params["ar_query"]
        if "ar_scale" in params:
            _safe_deallocate(self.scale)
            self.scale = params["ar_scale"]


class TTMambaWorkspaceModel:
    """Full model using tt-nn operations.

    Supports Cell A (Mamba2 + attention, no workspace — control),
    Cell B (Mamba2 + attention + workspace), and Cell C (+ recurrent core).
    """

    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device

        # Embedding: (vocab_size, d_model) — for ttnn.embedding this is the weight
        # For LM head (weight-tied), ttnn.linear(x, W) needs W=(d_model, vocab_size)
        # So we store the embedding as (vocab_size, d_model) and transpose for LM head
        emb_w = torch.randn(config.vocab_size, config.d_model, dtype=torch.bfloat16) * 0.02
        self.token_emb_weight = to_device(emb_w, device)

        # Build layers with gated residual wrappers.
        # Non-attention layers use TTRetentionLayer (RetNet-style decayed linear
        # attention). Retention maps to pure matmuls, avoiding the bandwidth-bound
        # selective scan and custom kernels that caused 0.17% MFU on Blackhole.
        # The J-space workspace and recurrent core attach identically -- they
        # operate on hidden states regardless of the backbone that produced them.
        self.layers = []
        self.attention_positions = set(config.attention_positions) if config.use_attention else set()
        for i in range(config.n_layers):
            if config.use_attention and i in config.attention_positions:
                attn_layer = TTAttentionLayer(config, device)
                wrapped = TTGatedResidualLayer(attn_layer, config.d_model, device)
                self.layers.append(wrapped)
            else:
                ret_layer = TTRetentionLayer(config, device)
                wrapped = TTGatedResidualLayer(ret_layer, config.d_model, device)
                self.layers.append(wrapped)

        # Workspace module (Cell B/C)
        if config.use_workspace:
            self.workspace = TTWorkspaceModule(config, device)
        else:
            self.workspace = None

        # Attention Residual Core (Cell C with Kimi K3 style residuals)
        # Replaces the fixed blend with learned attention over iteration outputs
        if config.recurrent_core and config.attention_residual_core:
            self.attn_residual = AttentionResidual(config.d_model, device, config.k_train_max)
        else:
            self.attn_residual = None

        # Final norm and LM head (weight-tied with embedding)
        self.norm = TTRMSNorm(config.d_model, device)
        self.lm_head_weight = self.token_emb_weight  # weight tying

        # Cache for identity matrix (used in embedding backward)
        self._identity_tt = None

    def forward(self, input_ids: torch.Tensor, k_value: int = None) -> "ttnn.Tensor":
        """
        input_ids: (B, T) PyTorch tensor of int indices
        k_value: int or None — number of active recurrent core iterations (Cell C).
                 If None and recurrent_core, uses k_train_max (all active for simplicity).
        Returns: (B, T, vocab_size) tt-nn tensor of logits
        """
        device = self.device
        config = self.config
        B, T = input_ids.shape

        # Cache input_ids for backward
        self._cached_input_ids = input_ids

        # Embedding lookup
        # REVIEWED: indices is a temporary device tensor — deallocate after embedding.
        indices = ttnn.from_torch(
            input_ids.to(torch.int32),
            dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
        )
        x = ttnn.embedding(indices, self.token_emb_weight, layout=ttnn.TILE_LAYOUT)  # (B, T, d_model)
        _safe_deallocate(indices)

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
        if self.workspace is not None:
            self.workspace._forward_caches = []
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

            if self.attn_residual is not None:
                # === Attention Residual Core (Kimi K3 style) ===
                # Run K active iterations without blend, storing each output.
                # After all iterations, compute softmax attention over all
                # outputs (including pre-core input x_0) with a learned query.
                # Inactive iterations (k >= K) are skipped entirely.
                #
                # The slot_state chains normally through workspace calls (no
                # blend). The 1/K slot gradient scaling is applied in backward
                # to dampen the slot chain's multiplicative gradient path.
                self._core_blend_info = []
                self._core_x_outputs = [x]  # x_0 = pre-core output

                import math
                chain_safety = config.chain_scale_safety
                slot_scale = 1.0 / (K * chain_safety) if K > 0 else 0.0

                for iteration in range(k_max):
                    active = (iteration < K)
                    if not active:
                        # Inactive: skip core layers, store current x (will be masked)
                        self._core_x_outputs.append(x)
                        # Record blend info with slot_scale for backward
                        slot_scale_tt = ttnn.from_torch(
                            torch.tensor([0.0], dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                        )
                        self._core_blend_info.append({
                            "iteration": iteration, "active": 0.0,
                            "slot_scale_tt": slot_scale_tt,
                        })
                        continue

                    # Run one core iteration (no blend — just pass through)
                    for i in core_layer_indices:
                        if self.workspace is not None and i in self.attention_positions:
                            was_none = (slot_state is None)
                            x, slot_state = self.workspace.forward(x, slot_state)
                            self._fwd_trace.append(("ws", "core", iteration, i, was_none))
                        x = self.layers[i].forward(x)
                        self._fwd_trace.append(("layer", "core", iteration, i))

                    # Final workspace at end of core iteration
                    if self.workspace is not None:
                        was_none = (slot_state is None)
                        x, slot_state = self.workspace.forward(x, slot_state)
                        self._fwd_trace.append(("ws", "core_end", iteration, -1, was_none))

                    # Store output (no blend — x is the raw iteration output)
                    self._core_x_outputs.append(x)

                    # Slot gradient scaling factor for backward
                    slot_scale_tt = ttnn.from_torch(
                        torch.tensor([slot_scale], dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    )
                    self._core_blend_info.append({
                        "iteration": iteration, "active": 1.0,
                        "slot_scale_tt": slot_scale_tt,
                    })

                # Compute attention residual over all stored outputs
                # x_outputs has k_max+1 entries: [x_0, x_1, ..., x_{k_max}]
                # Only entries 0..K are active (rest are duplicates, masked in attention)
                x = self.attn_residual.forward(self._core_x_outputs, K_active=K)
                self._fwd_trace.append(("ar", "core", K))

            else:
                # === Fixed blend core (original) ===
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
                # Slot-state gradient scaling: 1/K (more conservative than 1/sqrt(K))
                # The x residual uses 1/sqrt(K) blend, but the slot chain compounds
                # the workspace's gradient amplification multiplicatively across K
                # iterations. Using 1/K for the slot chain makes the per-iteration
                # gain A/K² + 1 - 1/sqrt(K), which is ≤1 when A ≤ K*sqrt(K)/K = sqrt(K)
                # — much more robust than 1/sqrt(K) which requires A ≤ sqrt(K).
                slot_scale = 1.0 / (K * config.chain_scale_safety) if K > 0 else 0.0

                for iteration in range(k_max):
                    active = 1.0 if iteration < K else 0.0
                    # Scale the blend factor by 1/sqrt(K) for gradient normalization
                    blend_factor = active * k_scale
                    active_tt = ttnn.from_torch(
                        torch.tensor([blend_factor], dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    )
                    # Slot-state gradient scaling factor (1/K for active iterations)
                    slot_blend_factor = active * slot_scale
                    slot_scale_tt = ttnn.from_torch(
                        torch.tensor([slot_blend_factor], dtype=torch.bfloat16),
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
                    # REVIEWED: intermediates must be explicitly deallocated.
                    # Compute ones - active_tt once and reuse for both x and slot paths.
                    ones_tt = ttnn.from_torch(
                        torch.tensor([1.0], dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    one_minus_active = ttnn.sub(ones_tt, active_tt)
                    _safe_deallocate(ones_tt)

                    # x = blend * x_new + (1 - blend) * x
                    _a_xn = ttnn.mul(active_tt, x_new)
                    _oma_x = ttnn.mul(one_minus_active, x)
                    x_new_blended = ttnn.add(_a_xn, _oma_x)
                    _safe_deallocate(_a_xn)
                    _safe_deallocate(_oma_x)

                    if slot_state is None:
                        slot_state = ttnn.add(ttnn.mul(active_tt, slot_state_new),
                                              ttnn.mul(one_minus_active, slot_state_new))
                    else:
                        _a_sn = ttnn.mul(active_tt, slot_state_new)
                        _oma_s = ttnn.mul(one_minus_active, slot_state)
                        slot_state_new_blended = ttnn.add(_a_sn, _oma_s)
                        _safe_deallocate(_a_sn)
                        _safe_deallocate(_oma_s)
                        slot_state = slot_state_new_blended

                    _safe_deallocate(one_minus_active)
                    # Deallocate old x and slot_state_new (replaced by blended versions)
                    _safe_deallocate(x_new)
                    _safe_deallocate(slot_state_new)
                    x = x_new_blended

                    self._core_blend_info.append({
                        "iteration": iteration, "active": active,
                        "active_tt": active_tt, "slot_scale_tt": slot_scale_tt,
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
        # REVIEWED: ttnn.transpose may return a view of self.lm_head_weight
        # (persistent parameter) — do NOT deallocate lm_head_w.
        lm_head_w = ttnn.transpose(self.lm_head_weight, 0, 1)  # (d_model, vocab_size)
        logits = ttnn.linear(x, lm_head_w)  # (B, T, vocab_size)

        return logits

    def compute_workspace_regularizers(self):
        """Compute entropy and diversity regularizer losses from the forward pass.

        Must be called after forward() and before backward().

        Returns: (entropy_loss, diversity_loss) as scalar floats (host-side).
        Also stores gradient contributions in self._reg_grads for the backward pass.

        entropy_loss: mean entropy of read/write attention — minimizing this pushes
                      the workspace toward selective (low-entropy) attention.
        diversity_loss: mean cosine similarity between slots — minimizing this pushes
                        slots to carry different information.

        Both are 0.0 if no workspace or no forward caches.
        """
        if self.workspace is None:
            self._reg_grads = None
            return 0.0, 0.0
        caches = self.workspace._forward_caches
        if not caches:
            self._reg_grads = None
            return 0.0, 0.0

        config = self.config
        entropy_weight = config.ws_entropy_weight
        diversity_weight = config.ws_diversity_weight

        if entropy_weight == 0 and diversity_weight == 0:
            self._reg_grads = None
            return 0.0, 0.0

        # Compute on host using torch autograd for the regularizer gradients.
        # The attention and slot values are already on device; we move them to
        # host, compute losses + gradients with PyTorch autograd, then convert
        # the gradients back to ttnn tensors for injection into the backward pass.
        reg_grads = []  # list of (grad_read_attn, grad_write_attn, grad_slots_out) per ws call

        total_entropy = 0.0
        total_diversity = 0.0
        n_entropy_terms = 0
        n_diversity_terms = 0

        for cache in caches:
            read_attn_tt = cache["read_attn"]     # (B, H, m, T)
            write_attn_tt = cache["write_attn"]   # (B, H, T, m)
            slots_out_tt = cache["slots_out"]     # (B, m, D)

            grad_read = None
            grad_write = None
            grad_slots = None

            # --- Entropy regularizer (host-side autograd) ---
            if entropy_weight > 0:
                read_attn = ttnn.to_torch(read_attn_tt).float().requires_grad_(True)
                write_attn = ttnn.to_torch(write_attn_tt).float().requires_grad_(True)

                # Entropy = -sum(p * log(p)) over last dim
                def entropy_fn(p):
                    return -(p * (p + 1e-8).log()).sum(dim=-1).mean()

                read_ent = entropy_fn(read_attn)
                write_ent = entropy_fn(write_attn)
                ent_loss = (read_ent + write_ent) * entropy_weight
                ent_loss.backward()

                total_entropy += ent_loss.item()
                n_entropy_terms += 1

                grad_read = read_attn.grad.clone()
                grad_write = write_attn.grad.clone()

            # --- Diversity regularizer (host-side autograd) ---
            if diversity_weight > 0:
                slots_out = ttnn.to_torch(slots_out_tt).float().requires_grad_(True)
                B, m, D = slots_out.shape

                # Normalize slots to unit length
                slot_norms = slots_out.norm(dim=-1, keepdim=True) + 1e-8  # (B, m, 1)
                slots_norm = slots_out / slot_norms  # (B, m, D)

                # Cosine similarity matrix: (B, m, m)
                sim = torch.matmul(slots_norm, slots_norm.transpose(-1, -2))

                # Off-diagonal mean (exclude self-similarity)
                eye = torch.eye(m, device=sim.device).unsqueeze(0)  # (1, m, m)
                off_diag = sim * (1 - eye)  # zero out diagonal
                div_loss = off_diag.sum() / (B * m * (m - 1)) * diversity_weight
                div_loss.backward()

                total_diversity += div_loss.item()
                n_diversity_terms += 1

                grad_slots = slots_out.grad.clone()

            reg_grads.append((grad_read, grad_write, grad_slots))

        self._reg_grads = reg_grads

        mean_entropy = total_entropy / max(n_entropy_terms, 1)
        mean_diversity = total_diversity / max(n_diversity_terms, 1)
        return mean_entropy, mean_diversity

    def _pop_reg_grads(self):
        """Pop regularizer gradients for the next workspace backward call (reverse order).

        Returns (extra_read, extra_write, extra_slots) as ttnn tensors on device, or None.
        """
        if not hasattr(self, '_reg_grads') or self._reg_grads is None or len(self._reg_grads) == 0:
            return None, None, None
        grad_read, grad_write, grad_slots = self._reg_grads.pop()
        device = self.device
        extra_read = None
        extra_write = None
        extra_slots = None
        if grad_read is not None:
            extra_read = ttnn.from_torch(grad_read.to(torch.bfloat16),
                                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if grad_write is not None:
            extra_write = ttnn.from_torch(grad_write.to(torch.bfloat16),
                                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if grad_slots is not None:
            extra_slots = ttnn.from_torch(grad_slots.to(torch.bfloat16),
                                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return extra_read, extra_write, extra_slots

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
        # NOTE: Do NOT deallocate x_pre_2d or grad_logits_2d — ttnn.reshape()
        # may return a view sharing the same device buffer as the original.
        # Deallocating would free x_pre (needed by norm.backward) or grad_logits.

        # --- Backward through final RMS norm (on device using TTRMSNorm.backward) ---
        grad_x, grad_norm_w = self.norm.backward(grad_x_normed, x_pre)
        _safe_deallocate(grad_x_normed)

        all_grads = {}

        # Helper to accumulate layer grads (shared params across core iterations accumulate)
        def accum_layer_grads(layer_idx, grads_dict):
            for k, v in grads_dict.items():
                key = f"layer_{layer_idx}_{k}"
                if key in all_grads:
                    old = all_grads[key]
                    all_grads[key] = ttnn.add(old, v)
                    _safe_deallocate(old)
                    _safe_deallocate(v)
                else:
                    all_grads[key] = v

        def accum_ws_grads(grads_dict):
            for k, v in grads_dict.items():
                key = f"ws_{k}"
                if key in all_grads:
                    old = all_grads[key]
                    all_grads[key] = ttnn.add(old, v)
                    _safe_deallocate(old)
                    _safe_deallocate(v)
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
        ar_trace = []  # attention residual entries
        current_phase = "pre"
        for entry in trace:
            if entry[0] == "ar":
                ar_trace.append(entry)
            elif entry[0] == "layer":
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
        # REVIEWED: ones_tt is only needed for the fixed blend core backward
        # (non-AR recurrent path). Create lazily to avoid unnecessary device
        # transfer for non-recurrent cells (A, B) or AR core cells (C).
        ones_tt = None

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
                extra_r, extra_w, extra_s = self._pop_reg_grads()
                grad_x, grad_slots_in, ws_grads = self.workspace.backward(
                    grad_x, grad_slot_state, extra_r, extra_w, extra_s)
                accum_ws_grads(ws_grads)
                if was_none:
                    grad_slots_param = ttnn.sum(grad_slots_in, dim=0)
                    _safe_deallocate(grad_slots_in)
                    if "ws_slots" in all_grads:
                        old_slots_grad = all_grads["ws_slots"]
                        all_grads["ws_slots"] = ttnn.add(old_slots_grad, grad_slots_param)
                        _safe_deallocate(old_slots_grad)
                        _safe_deallocate(grad_slots_param)
                    else:
                        all_grads["ws_slots"] = grad_slots_param
                    grad_slot_state = None
                else:
                    # REVIEWED: old grad_slot_state is being orphaned — deallocate it.
                    if grad_slot_state is not None:
                        _safe_deallocate(grad_slot_state)
                    grad_slot_state = grad_slots_in

        # --- Backward through recurrent core (reverse iterations) ---
        if recurrent:
            blend_info = self._core_blend_info
            if self.attn_residual is not None:
                # === Attention Residual Core backward ===
                # 1. Backward through attention residual → get grad for each stored x
                # 2. Backward through iterations in reverse, adding AR gradients
                grad_x_list, ar_grads = self.attn_residual.backward(grad_x)
                accum_ws_grads(ar_grads)  # ar_query and ar_scale are stored with ws_ prefix

                # Group core_trace by iteration (same as blend path)
                iter_segments = []
                current_iter = -1
                current_entries = []
                for entry in core_trace:
                    if entry[0] == "ws":
                        iter_num = entry[2]
                    elif entry[0] == "layer":
                        iter_num = entry[2]
                    else:
                        continue
                    if iter_num != current_iter:
                        if current_entries:
                            iter_segments.append((current_iter, current_entries))
                        current_iter = iter_num
                        current_entries = [entry]
                    else:
                        current_entries.append(entry)
                if current_entries:
                    iter_segments.append((current_iter, current_entries))

                # K_active = number of active iterations (0..K_active are in attention)
                K_active = ar_trace[0][2] if ar_trace else 0

                # Unroll in reverse iteration order.
                # For each iteration, first add the AR gradient for that
                # iteration's output (x_{iter+1}), then backward through the
                # iteration's core layers.  The AR gradient for x_K is added
                # in the first loop iteration — do NOT pre-initialize grad_x
                # with grad_x_list[K_active], that double-counts it.
                #
                # CRITICAL: scale the chained gradient by 1/K before adding
                # the AR gradient.  Without this, the gradient from each
                # iteration flows through the shared core layers at full
                # magnitude and accumulates exponentially: the gradient at
                # iteration 0 is ~sum_k A^(K-k) * |ar_grad_x[k]|, where A is
                # the layer backward amplification (A > 1 when the workspace
                # cross-attention is active).  For K=6 and A≈2, this gives
                # ~64x amplification, causing divergence at step ~1550.
                #
                # Scaling the chain by 1/K gives per-iteration gain A/K,
                # which is stable for A < K (much more robust than 1/sqrt(K)
                # which requires A < sqrt(K)).  The AR gradient itself is
                # not scaled — it's bounded by softmax (weights sum to 1).
                grad_x = None

                for iter_num, entries in reversed(iter_segments):
                    blend = blend_info[iter_num]
                    slot_scale_tt = blend["slot_scale_tt"]

                    # Add attention residual gradient for this iteration's output
                    # x_outputs[iter_num+1] is the output of iteration iter_num
                    # grad_x_list[iter_num+1] is the AR gradient for that output
                    ar_grad_x = grad_x_list[iter_num + 1]
                    if grad_x is None:
                        grad_x = ar_grad_x
                    else:
                        # Scale chained gradient by 1/(K*safety) (slot_scale_tt
                        # for active iterations) before adding AR gradient
                        scaled_chain = ttnn.mul(slot_scale_tt, grad_x)
                        _safe_deallocate(grad_x)
                        # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
                        # ar_grad_x is consumed and must be deallocated.
                        _gx_new = ttnn.add(scaled_chain, ar_grad_x)
                        _safe_deallocate(scaled_chain)
                        _safe_deallocate(ar_grad_x)
                        grad_x = _gx_new

                    # Backward through this iteration's entries (reverse)
                    # No blend — grad_x flows directly through core layers
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
                            extra_r, extra_w, extra_s = self._pop_reg_grads()
                            grad_x, grad_slots_in, ws_grads = self.workspace.backward(
                                grad_x, grad_slot_state, extra_r, extra_w, extra_s)
                            accum_ws_grads(ws_grads)
                            if was_none:
                                grad_slots_param = ttnn.sum(grad_slots_in, dim=0)
                                _safe_deallocate(grad_slots_in)
                                if "ws_slots" in all_grads:
                                    old = all_grads["ws_slots"]
                                    all_grads["ws_slots"] = ttnn.add(old, grad_slots_param)
                                    _safe_deallocate(old)
                                    _safe_deallocate(grad_slots_param)
                                else:
                                    all_grads["ws_slots"] = grad_slots_param
                                grad_slot_state = None
                            else:
                                # Scale grad_slots_in by 1/K before chaining
                                # (same as blend path — dampens slot chain amplification)
                                # REVIEWED: reassignment leak — old grad_slot_state must be deallocated.
                                if grad_slot_state is not None:
                                    _safe_deallocate(grad_slot_state)
                                grad_slot_state = ttnn.mul(slot_scale_tt, grad_slots_in)
                                _safe_deallocate(grad_slots_in)

                # After all iterations, add AR gradient for x_0 (pre-core output)
                # REVIEWED: grad_x_list[0] is consumed — deallocate it.
                grad_x_final = ttnn.add(grad_x, grad_x_list[0])
                _safe_deallocate(grad_x)
                _safe_deallocate(grad_x_list[0])
                grad_x = grad_x_final

            else:
                # === Fixed blend core backward (original) ===
                # Group core_trace by iteration
                # Each iteration has: ws calls for core layers, layer calls, ws at core_end
                # We need to unroll in reverse iteration order

                # REVIEWED: Create ones_tt lazily here — only the fixed blend
                # core backward needs it (AR core doesn't).
                ones_tt = ttnn.from_torch(torch.tensor([1.0], dtype=torch.bfloat16),
                                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

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
                    slot_scale_tt = blend["slot_scale_tt"]

                    # All iterations (including the first) use the scaled blend:
                    # x = blend * x_new + (1 - blend) * x_old
                    # grad_x_new = blend * grad_x  (flows into core layers)
                    # grad_x_old = (1 - blend) * grad_x  (flows to previous iteration or pre-core)
                    grad_x_new = ttnn.mul(active_tt, grad_x)
                    _safe_deallocate(grad_x)
                    # REVIEWED: compute one_minus_active once, use for both x and slot paths.
                    one_minus_active = ttnn.sub(ones_tt, active_tt)
                    grad_x_old = ttnn.mul(one_minus_active, grad_x_new)
                    grad_slot_new = ttnn.mul(active_tt, grad_slot_state) if grad_slot_state is not None else None
                    grad_slot_old = ttnn.mul(one_minus_active, grad_slot_state) if grad_slot_state is not None else None
                    _safe_deallocate(one_minus_active)
                    # grad_x for the core iteration = grad_x_new
                    # grad_x_old accumulates into the previous iteration's output
                    grad_x = grad_x_new
                    if grad_slot_old is not None:
                        # REVIEWED: old grad_slot_state is being orphaned — deallocate it.
                        _safe_deallocate(grad_slot_state)
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
                            extra_r, extra_w, extra_s = self._pop_reg_grads()
                            grad_x, grad_slots_in, ws_grads = self.workspace.backward(
                                grad_x, grad_slot_state, extra_r, extra_w, extra_s)
                            accum_ws_grads(ws_grads)
                            if was_none:
                                grad_slots_param = ttnn.sum(grad_slots_in, dim=0)
                                _safe_deallocate(grad_slots_in)
                                if "ws_slots" in all_grads:
                                    old = all_grads["ws_slots"]
                                    all_grads["ws_slots"] = ttnn.add(old, grad_slots_param)
                                    _safe_deallocate(old)
                                    _safe_deallocate(grad_slots_param)
                                else:
                                    all_grads["ws_slots"] = grad_slots_param
                                grad_slot_state = None
                            else:
                                # Scale grad_slots_in by 1/K (slot_scale_tt) before
                                # chaining to the previous iteration. The workspace
                                # backward received 1/sqrt(K)-scaled inputs, so
                                # grad_slots_in is already 1/sqrt(K) * true_grad.
                                # Scaling by 1/K gives 1/(K*sqrt(K)) * true_grad,
                                # making the per-iteration gain A/K² + 1 - 1/sqrt(K),
                                # which is ≤1 for A ≤ sqrt(K) — much more robust than
                                # the 1/sqrt(K) scaling which required A ≤ sqrt(K)
                                # for the gain to be ≤1.
                                # REVIEWED: reassignment leak — old grad_slot_state must be deallocated.
                                if grad_slot_state is not None:
                                    _safe_deallocate(grad_slot_state)
                                grad_slot_state = ttnn.mul(slot_scale_tt, grad_slots_in)
                                _safe_deallocate(grad_slots_in)

                    # After processing this iteration, add grad_x_old from blend.
                    # For iter_num > 0, grad_x_old flows to the previous iteration.
                    # For iter_num == 0, grad_x_old flows to the pre-core layers
                    # (it will be picked up by the pre-core backward below).
                    grad_x_summed = ttnn.add(grad_x, grad_x_old)
                    _safe_deallocate(grad_x)
                    _safe_deallocate(grad_x_old)
                    grad_x = grad_x_summed
                    if hasattr(self, '_pending_grad_slot_old') and self._pending_grad_slot_old is not None:
                        if grad_slot_state is None:
                            grad_slot_state = self._pending_grad_slot_old
                        else:
                            old_slot_state = grad_slot_state
                            grad_slot_state = ttnn.add(grad_slot_state, self._pending_grad_slot_old)
                            _safe_deallocate(old_slot_state)
                            _safe_deallocate(self._pending_grad_slot_old)
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
                extra_r, extra_w, extra_s = self._pop_reg_grads()
                grad_x, grad_slots_in, ws_grads = self.workspace.backward(
                    grad_x, grad_slot_state, extra_r, extra_w, extra_s)
                accum_ws_grads(ws_grads)
                if was_none:
                    grad_slots_param = ttnn.sum(grad_slots_in, dim=0)
                    _safe_deallocate(grad_slots_in)
                    if "ws_slots" in all_grads:
                        old_slots_grad = all_grads["ws_slots"]
                        all_grads["ws_slots"] = ttnn.add(old_slots_grad, grad_slots_param)
                        _safe_deallocate(old_slots_grad)
                        _safe_deallocate(grad_slots_param)
                    else:
                        all_grads["ws_slots"] = grad_slots_param
                    grad_slot_state = None
                else:
                    # REVIEWED: old grad_slot_state is being orphaned — deallocate it.
                    if grad_slot_state is not None:
                        _safe_deallocate(grad_slot_state)
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
        # NOTE: Do NOT deallocate one_hot, grad_x_2d, or the transpose results —
        # ttnn.reshape/transpose may return views sharing the same device buffer.

        # Add LM head gradient (weight-tied): grad_emb += grad_lm_head^T
        # grad_lm_head is (d, V), so grad_lm_head^T is (V, d)
        # REVIEWED: transpose may return a view — do NOT deallocate grad_lm_head_t
        # (would free grad_lm_head which is returned in all_grads).
        grad_lm_head_t = ttnn.transpose(grad_lm_head, 0, 1)
        # REVIEWED: reassignment leak — ttnn.add creates a new tensor.
        _ge_new = ttnn.add(grad_emb, grad_lm_head_t)
        _safe_deallocate(grad_emb)
        grad_emb = _ge_new

        all_grads["token_emb_weight"] = grad_emb
        all_grads["norm_weight"] = grad_norm_w

        # Deallocate the ones_tt created for blend backward (if any)
        if ones_tt is not None:
            _safe_deallocate(ones_tt)

        # If gamma is frozen, drop gamma gradients so they don't enter
        # clip_grad_norm or the optimizer.  The backward pass still computes
        # them (they're part of the chain rule for grad_x), but they're
        # harmless if unused — the gamma parameter simply never updates.
        if self.config.freeze_gamma:
            all_grads = {k: v for k, v in all_grads.items()
                         if not k.endswith("_gamma")}
        if self.config.freeze_slot_decay:
            all_grads = {k: v for k, v in all_grads.items()
                         if k != "ws_slot_decay"}

        return all_grads

    def clear_caches(self):
        """Clear all cached tensors from the last forward/backward pass.

        Explicitly deallocates cached intermediates to free device buffers,
        then drops references. Model parameters survive because they're still
        referenced by self.token_emb_weight, self.layers[i].gate, etc.
        """
        _safe_deallocate(self._cached_x_pre_final_norm)
        self._cached_x_pre_final_norm = None
        self._cached_input_ids = None  # host tensor, no device dealloc
        # Deallocate core x outputs and blend info (only exist with recurrent core)
        for x_out in getattr(self, '_core_x_outputs', []):
            _safe_deallocate(x_out)
        self._core_x_outputs = []
        for blend in getattr(self, '_core_blend_info', []):
            for k, v in blend.items():
                if hasattr(v, 'shape'):  # skip ints/floats
                    _safe_deallocate(v)
        self._core_blend_info = []
        self._fwd_trace = []
        for layer in self.layers:
            if hasattr(layer, 'clear_caches'):
                layer.clear_caches()
            elif hasattr(layer, '_cache'):
                # Retention/attention layers: deallocate cached intermediates.
                # Exclude persistent tensors: scale_tt (self._scale_tt),
                # gate (self.gate), and cached causal masks are model
                # parameters reused across steps.
                # REVIEWED: also clean cache_history for recurrent core layers.
                _persistent = set()
                if hasattr(layer, '_scale_tt'):
                    _persistent.add(id(layer._scale_tt))
                if hasattr(layer, 'gate'):
                    _persistent.add(id(layer.gate))
                if hasattr(layer, '_causal_mask_upper') and layer._causal_mask_upper is not None:
                    _persistent.add(id(layer._causal_mask_upper))
                if hasattr(layer, '_causal_mask_lower') and layer._causal_mask_lower is not None:
                    _persistent.add(id(layer._causal_mask_lower))
                for k, v in layer._cache.items():
                    if id(v) not in _persistent and hasattr(v, 'shape'):
                        _safe_deallocate(v)
                layer._cache = {}
                # Deallocate cache history from recurrent core iterations
                if hasattr(layer, '_deallocate_cache_history'):
                    layer._deallocate_cache_history()
                # REVIEWED: deallocate short conv cache if present
                if hasattr(layer, '_deallocate_conv_cache'):
                    layer._deallocate_conv_cache()
        if self.workspace is not None:
            self.workspace.clear_caches()
        if self.attn_residual is not None:
            self.attn_residual.clear_caches()

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        params = {"token_emb_weight": self.token_emb_weight, "norm_weight": self.norm.weight}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for k, v in layer_params.items():
                # Skip gamma if frozen — it stays at init and is not optimizer-managed
                if self.config.freeze_gamma and k == "gamma":
                    continue
                params[f"layer_{i}_{k}"] = v
        if self.workspace is not None:
            for k, v in self.workspace.get_params().items():
                # Skip slot_decay if frozen
                if self.config.freeze_slot_decay and k == "slot_decay":
                    continue
                params[f"ws_{k}"] = v
        if self.attn_residual is not None:
            for k, v in self.attn_residual.get_params().items():
                params[f"ws_{k}"] = v  # use ws_ prefix so lr_groups applies
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

    def spectral_normalize_backbone_weights(self):
        """Cap spectral norm of qkv_weight and out_proj_weight in all backbone layers.

        Uses the same power-iteration approach as the workspace spectral norm.
        The bound (config.backbone_spectral_norm_bound, default 2.0) is chosen
        below the divergence threshold (~2.3) observed in the second run, where
        the workspace's multiplicative gradient amplification causes instability
        once backbone norms exceed ~2.3. Cell A (no workspace) is stable at
        norms up to 6.4, but the workspace adds cross-attention gradient paths
        that amplify backbone weight growth. A bound of 2.0 allows healthy
        growth (Cell B reached loss 0.25 with norms at 2.3) while staying
        safely below the instability threshold.

        Only scales down when sigma > bound — never scales up. This lets the
        optimizer grow weights naturally up to the bound.
        """
        device = self.device
        bound = self.config.backbone_spectral_norm_bound
        bound_tt = ttnn.from_torch(
            torch.tensor([bound], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        eps_tt = ttnn.from_torch(
            torch.tensor([1e-6], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        for wrapped in self.layers:
            inner = wrapped.layer
            for wname in ("qkv_weight", "out_proj_weight"):
                if not hasattr(inner, wname):
                    continue
                W = getattr(inner, wname)
                m, n = int(W.shape[0]), int(W.shape[1])

                # Power iteration to estimate spectral norm (10 steps).
                # Each iteration creates ~15 intermediate ttnn tensors; without
                # explicit deallocation they accumulate in host RAM (ttnn wrapper
                # metadata) and device DRAM until Python GC runs — but GC doesn't
                # see the pressure because wrappers are tiny.  This was the
                # dominant memory leak (~3 GB/hour) that OOM-killed training.
                v = ttnn.from_torch(
                    torch.randn(n, 1, dtype=torch.bfloat16) / math.sqrt(n),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                Wt = ttnn.transpose(W, 0, 1)  # cache transpose outside loop
                for _ in range(10):
                    u = ttnn.matmul(W, v)  # (m, 1)
                    u_sq = ttnn.mul(u, u)
                    u_norm = ttnn.sqrt(ttnn.sum(u_sq))
                    _safe_deallocate(u_sq)
                    u_norm_c = ttnn.maximum(u_norm, eps_tt)
                    _safe_deallocate(u_norm)
                    u_recip = ttnn.reciprocal(u_norm_c)
                    _safe_deallocate(u_norm_c)
                    u_new = ttnn.mul(u, u_recip)
                    _safe_deallocate(u)
                    _safe_deallocate(u_recip)
                    u = u_new

                    v_new = ttnn.matmul(Wt, u)  # (n, 1)
                    v_sq = ttnn.mul(v_new, v_new)
                    v_norm = ttnn.sqrt(ttnn.sum(v_sq))
                    _safe_deallocate(v_sq)
                    v_norm_c = ttnn.maximum(v_norm, eps_tt)
                    _safe_deallocate(v_norm)
                    v_recip = ttnn.reciprocal(v_norm_c)
                    _safe_deallocate(v_norm_c)
                    v_result = ttnn.mul(v_new, v_recip)
                    _safe_deallocate(v)  # old v from previous iteration
                    _safe_deallocate(v_new)
                    _safe_deallocate(v_recip)
                    v = v_result
                _safe_deallocate(Wt)
                _safe_deallocate(u)

                Wv = ttnn.matmul(W, v)
                Wv_sq = ttnn.mul(Wv, Wv)
                sigma = ttnn.sqrt(ttnn.sum(Wv_sq))
                _safe_deallocate(Wv_sq)
                sigma_c = ttnn.maximum(sigma, eps_tt)
                _safe_deallocate(sigma)

                # scale = min(1, bound/sigma) = bound / max(bound, sigma)
                scale_factor = ttnn.mul(bound_tt, ttnn.reciprocal(ttnn.maximum(bound_tt, sigma_c)))
                W_normalized = ttnn.mul(W, scale_factor)
                _safe_deallocate(getattr(inner, wname))
                setattr(inner, wname, W_normalized)
                # Deallocate power iteration intermediates
                _safe_deallocate(v)
                _safe_deallocate(Wv)
                _safe_deallocate(sigma_c)
                _safe_deallocate(scale_factor)

        _safe_deallocate(bound_tt)
        _safe_deallocate(eps_tt)

    def clamp_retention_gammas(self, max_log_gamma: float = -0.02):
        """Clamp retention layer log-gamma to <= max_log_gamma after each optimizer step.

        The retention decay matrix is D[t,s] = gamma^(t-s).  When gamma > 1.0,
        D becomes exponentially growing instead of decaying, causing attention
        scores and gradients to explode.  This is a mathematical stability
        constraint, not regularization: the retention layer is definitionally a
        decay mechanism, and gamma > 1 turns it into an unstable amplifier.

        Even gamma = 1.0 (log_gamma = 0) is unstable: D = 1 for all positions
        means no decay, and the gamma gradient (a sum over T*T terms weighted
        by diff[t,s]) becomes O(T^2) — ~128x larger than weight matrix gradients
        (O(T) sums).  This gradient dominates the global clip, starving all
        other parameters of update signal.

        The default clamp at -0.02 (gamma <= 0.98) ensures D decays:
          D[127] = 0.98^127 = 0.077  (vs 1.0 at gamma=1.0)
        This attenuates the gamma gradient by ~10x, preventing domination.

        gamma is stored as log(gamma) in fp32, so clamping at max_log_gamma
        gives gamma <= exp(max_log_gamma).
        """
        for layer in self.layers:
            inner = layer.layer
            if hasattr(inner, 'gamma'):
                # gamma is fp32 (n_heads,) on device; clamp log_gamma <= max_log_gamma
                g_host = ttnn.to_torch(inner.gamma).float()  # (n_heads,)
                clamped = g_host.clamp(max=max_log_gamma)
                if not torch.equal(g_host, clamped):
                    old_gamma = inner.gamma
                    inner.gamma = ttnn.from_torch(
                        clamped, dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT, device=old_gamma.device())
                    _safe_deallocate(old_gamma)

    def clamp_workspace_gates(self):
        """Clamp ReZero gates to [-gate_clamp_bound, gate_clamp_bound] after each optimizer step.

        ReZero gates are unbounded scalars (no sigmoid). When the read gate
        grows large negative — which happens with causal masking because the
        workspace's past-only contribution is noise that the model learns to
        suppress — the pre-norm residual `decay * slots + gate * read_out`
        experiences cancellation. This makes RMS(slots_pre_norm) very small,
        and RMSNorm backward divides by RMS, amplifying the gradient by
        1/RMS → gradient explosion.

        This is a mathematical stability constraint, not regularization: the
        pre-causal-fix run had gates in [-0.35, 0.43] and was stable, so
        ±0.3 is a safe bound that doesn't constrain normal learning.

        No-op when gate_clamp_bound is 0.0 (disabled) or no workspace.
        """
        bound = self.config.gate_clamp_bound
        if bound <= 0 or self.workspace is None:
            return
        ws = self.workspace
        for attr in ("read_gate", "write_gate"):
            g = getattr(ws, attr)
            g_host = ttnn.to_torch(g).float()
            clamped = g_host.clamp(min=-bound, max=bound)
            if not torch.equal(g_host, clamped):
                old_g = getattr(ws, attr)
                setattr(ws, attr, ttnn.from_torch(
                    clamped.to(torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=g.device()))
                _safe_deallocate(old_g)

    def get_workspace_stats(self):
        """Extract workspace gate, QK scale, and slot-decay scalars to host for logging.

        With ReZero gates, the gate values are direct scalars (no sigmoid).
        Returns dict with read_gate, write_gate, slot_decay, read_qk_scale,
        write_qk_scale as Python floats, or None if no workspace (Cell A).
        Also includes ar_scale (attention residual scale) if present.
        """
        if self.workspace is None:
            return None
        ws = self.workspace
        rg = ttnn.to_torch(ws.read_gate).float().item()
        wg = ttnn.to_torch(ws.write_gate).float().item()
        sd = ttnn.to_torch(ws.slot_decay).float().item()
        rqs = ttnn.to_torch(ws.read_qk_scale).float().item()
        wqs = ttnn.to_torch(ws.write_qk_scale).float().item()
        stats = {
            'read_gate': rg,
            'write_gate': wg,
            'slot_decay': sd,
            'read_qk_scale': rqs,
            'write_qk_scale': wqs,
        }
        if self.attn_residual is not None:
            ars = ttnn.to_torch(self.attn_residual.scale).float().item()
            stats['ar_scale'] = ars
        return stats

    def apply_gate_schedule(self, step: int, gate_schedule_steps: int, gate_init_val: float):
        """Force workspace gates open on a schedule.

        With ReZero gates (init=0), this is a no-op — gates already start at 0
        (true identity) and the optimizer controls them freely from step 0.
        This method is kept for backward compatibility with configs that set
        gate_schedule_steps > 0.

        Previously (with sigmoid gates): linearly annealed gate values from
        gate_init (e.g. -2, sigmoid=0.12) to 0 (sigmoid=0.5) over
        gate_schedule_steps. This ensured the workspace contributed enough that
        the model must learn good routing — at 12% mixing the backbone can
        route around the workspace, so the gates never open
        naturally. Forcing them to 50% creates gradient signal that drives
        content-addressed routing.

        After the schedule ends (step >= gate_schedule_steps), the optimizer
        controls the gates freely.

        No-op if no workspace or gate_schedule_steps <= 0.
        """
        if self.workspace is None or gate_schedule_steps <= 0:
            return
        if step >= gate_schedule_steps:
            return
        # With ReZero gates (init=0), the gate schedule is a no-op — gates
        # already start at 0 (true identity). The optimizer controls them
        # freely from step 0. This method is kept for backward compatibility
        # with configs that set gate_schedule_steps > 0.
        # (Previously annealed sigmoid gates from gate_init to 0.)

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
                "headdim": self.config.headdim,
                "d_state_m3": self.config.d_state_m3,
                "mimo_rank": self.config.mimo_rank,
                "rope_fraction": self.config.rope_fraction,
                "ngroups": self.config.ngroups,
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

        # _set_param → set_params now deallocates old params internally,
        # so no explicit cleanup needed here.
        for name, host_tensor in model_state.items():
            if name == "token_emb_weight":
                dtype = ttnn.bfloat16
            elif any(s in name for s in ("A_log", "_D", "dt_bias", "B_bias", "C_bias",
                                          "MIMO_V", "MIMO_Z", "MIMO_O")):
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
            _safe_deallocate(self.token_emb_weight)
            self.token_emb_weight = tt_tensor
            self.lm_head_weight = tt_tensor  # weight tying — same tensor
        elif name == "norm_weight":
            _safe_deallocate(self.norm.weight)
            self.norm.weight = tt_tensor
        elif name.startswith("layer_"):
            parts = name.split("_", 2)
            layer_idx = int(parts[1])
            param_name = parts[2]
            self.layers[layer_idx].set_params({param_name: tt_tensor})
        elif name in ("ws_ar_query", "ws_ar_scale") and self.attn_residual is not None:
            ar_param_name = name[3:]  # strip "ws_" prefix
            self.attn_residual.set_params({ar_param_name: tt_tensor})
        elif name.startswith("ws_") and self.workspace is not None:
            ws_param_name = name[3:]  # strip "ws_" prefix
            self.workspace.set_params({ws_param_name: tt_tensor})
        else:
            print(f"WARNING: Unknown param name {name}")


# ---------------------------------------------------------------------------
# Cell configurations (shared with model.py)
# ---------------------------------------------------------------------------

def get_cell_config(cell: str) -> ModelConfig:
    """Return the ModelConfig for a given cell (A, B, or C).

    Cell naming was renamed on 2026-07-31: old E→A, old C→B, old D→C.
    The former pure-Mamba2 (old A) and Mamba2+attention (old B) cells were
    deprecated and removed — see AGENTS.md.
    """
    if cell == "A":
        # Control: Mamba2 + attention, no workspace
        return ModelConfig(
            n_layers=14,
            use_attention=True,
            attention_positions=[5, 10],
            use_workspace=False,
            recurrent_core=False,
        )
    elif cell == "B":
        # Hybrid + workspace (perceiver)
        return ModelConfig(
            n_layers=13,
            use_attention=True,
            attention_positions=[5, 10],
            use_workspace=True,
            n_workspace_slots=16,
            recurrent_core=False,
        )
    elif cell == "C":
        # Full architecture: hybrid + workspace + recurrent core
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
    else:
        raise ValueError(f"Unknown cell: {cell}")
