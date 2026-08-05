"""Pure-PyTorch reference for the Retention layer (RetNet-style).

This mirrors TTRetentionLayer.forward op-for-op, but in plain PyTorch so that
autograd can produce exact gradients. It exists purely for validation:

  * Stage 0 (--gradcheck): float64 torch.autograd.gradcheck of this reference.
  * Stage 1: tt-nn forward vs. this reference forward.
  * Stage 2: tt-nn backward vs. autograd on this reference.

Retention replaces Mamba-3's selective scan with decayed linear attention:
D[t,s] = gamma^(t-s) for s <= t. The decay gamma is a single learned scalar
per head (stored as log(gamma)), precomputed into a T x T matrix. No softmax,
no cumsum of learned A*dt, no diagonal correction, no MIMO ranks.

Forward:
  qkvg = linear(x, W_in)          # (B, T, 4D) -- q, k, v, gate
  q, k = rope(q), rope(k)         # standard RoPE on q and k
  A = (q @ k^T) * scale * D       # (B, H, T, T) -- decayed linear attention
  O = A @ v                       # (B, H, T, d_h)
  O = reshape(O)                  # (B, T, D)
  O = O * sigmoid(gate)           # element-wise output gate over full d_model
  out = linear(O, W_out)          # (B, T, D)
"""

import math
import torch
import torch.nn.functional as F


def apply_rope(x, cos, sin):
    """Apply RoPE to x: (B, H, T, d_head) -> (B, H, T, d_head).

    Splits x into [x1, x2] along last dim, rotates:
      rotated = cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)

    cos/sin: (1, 1, T, d_head//2)
    """
    d_h = x.shape[-1]
    x1 = x[..., :d_h // 2]
    x2 = x[..., d_h // 2:]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.cat([r1, r2], dim=-1)


PARAM_NAMES = (
    "qkv_weight",       # (d_model, 4*d_model) -- q, k, v, gate
    "out_proj_weight",   # (d_model, d_model)
    "gamma",             # (n_heads,) -- log(gamma) per head
)


class RetentionReference(torch.nn.Module):
    """Reference Retention layer. Parameter shapes match TTRetentionLayer.get_params()."""

    def __init__(self, config, params=None, dtype=torch.float32, eps=1e-6):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.eps = eps

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)

        # RoPE frequencies
        d_rope = self.d_head
        freqs = 1.0 / (10000 ** (torch.arange(0, d_rope, 2).float() / d_rope))
        self.register_buffer("rope_freqs", freqs)

        if params is not None:
            self.load(params)

    def load(self, params):
        """Install parameters as leaf tensors with requires_grad=True."""
        for name in PARAM_NAMES:
            t = params[name].detach().to(self.dtype).clone().requires_grad_(True)
            setattr(self, name, t)
        return self

    def params(self):
        return {name: getattr(self, name) for name in PARAM_NAMES}

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, D = x.shape
        H, d_h = self.n_heads, self.d_head
        device = x.device
        dtype = x.dtype

        # --- QKV+gate projection: (B, T, 4D) ---
        qkvg = x @ self.qkv_weight  # (B, T, 4D)
        q, k, v, g = torch.split(qkvg, [D, D, D, D], dim=-1)

        # Reshape q, k, v to (B, H, T, d_head)
        q = q.reshape(B, T, H, d_h).permute(0, 2, 1, 3)
        k = k.reshape(B, T, H, d_h).permute(0, 2, 1, 3)
        v = v.reshape(B, T, H, d_h).permute(0, 2, 1, 3)
        # g stays (B, T, D) -- element-wise gate over full d_model

        # --- RoPE ---
        positions = torch.arange(T, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.rope_freqs.to(device))  # (T, d_rope/2)
        cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_h//2)
        sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # --- Decay matrix: (H, T, T) ---
        # gamma stored as log(gamma); D[t,s] = exp(diff[t,s] * log_gamma) for s <= t
        pos = torch.arange(T, device=device, dtype=dtype)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (T, T), diff[t, s] = t - s
        causal = (diff >= 0).to(dtype)
        log_gamma = self.gamma.unsqueeze(1).unsqueeze(2)  # (H, 1, 1)
        log_D = diff.unsqueeze(0) * log_gamma  # (H, T, T)
        D_decay = torch.exp(log_D) * causal.unsqueeze(0)  # (H, T, T)

        # --- Decayed linear attention ---
        scores = torch.matmul(q, k.transpose(-1, -2))  # (B, H, T, T)
        scores = scores * self.scale
        scores = scores * D_decay.unsqueeze(0)  # (B, H, T, T)

        out = torch.matmul(scores, v)  # (B, H, T, d_h)

        # Reshape back to (B, T, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)

        # --- Output gate: sigmoid(g) element-wise over (B, T, D) ---
        gate = torch.sigmoid(g)  # (B, T, D)
        out = out * gate

        # --- Output projection ---
        out = out @ self.out_proj_weight  # (B, T, D)

        return out
