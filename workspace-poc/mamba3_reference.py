"""Pure-PyTorch reference for the Mamba-3 MIMO layer.

This mirrors TTMamba3Layer.forward op-for-op, but in plain PyTorch so that
autograd can produce exact gradients. It exists purely for validation:

  * Stage 1 - forward parity: tt-nn forward vs. this reference forward.
    Isolates tt-nn tile/broadcast/reshape issues.
  * Stage 2 - backward parity: TTMamba3Layer.backward vs. autograd on this
    reference. Isolates manual-gradient math bugs.

Run in float64 to sanity-check the reference against finite differences
(see test_mamba3_parity.py --gradcheck); run in float32 to compare against
the bf16 tt-nn implementation.

The reference implements the *intended* semantics. Where the tt-nn layer
currently deviates, the tt-nn layer is the thing to fix.
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F

# Value substituted into masked (non-causal) positions before exp().
# exp(-1e4) underflows to exactly 0 in fp32/bf16, and unlike -inf it keeps
# the backward pass free of nan.
NEG_MASK = -1e4


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Matches ttnn.rms_norm: x / sqrt(eps + mean(x^2)) * weight."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def shift_by_one(t: torch.Tensor) -> torch.Tensor:
    """Shift (B, T, H) by one along T: prepend zeros, drop the last step."""
    return torch.cat([torch.zeros_like(t[:, :1]), t[:, :-1]], dim=1)


def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               split: int) -> torch.Tensor:
    """Rotate the first `split` channels of t as interleaved (even, odd) pairs.

    t:        (B, T, R, H, N)
    cos/sin:  (B, T, H, n_angles)   broadcast over R
    """
    B, T, R, H, N = t.shape
    rot, passthru = t[..., :split], t[..., split:]

    rot = rot.reshape(B, T, R, H, split // 2, 2)
    t0, t1 = rot[..., 0], rot[..., 1]           # (B, T, R, H, n_angles)

    c = cos.unsqueeze(2)                        # (B, T, 1, H, n_angles)
    s = sin.unsqueeze(2)

    r0 = t0 * c - t1 * s
    r1 = t0 * s + t1 * c

    out = torch.stack([r0, r1], dim=-1).reshape(B, T, R, H, split)
    return torch.cat([out, passthru], dim=-1)


def decay_matrix(ADT: torch.Tensor) -> torch.Tensor:
    """L[b,h,t,s] = exp(sum(ADT[b,h,s+1:t+1])) for s <= t, else 0.

    NOTE: the mask must be applied *before* the exp with a large negative
    fill. Masking to zero before the exp yields exp(0)=1 in the upper
    triangle (i.e. a non-causal layer), and masking after the exp overflows
    because segsum > 0 there.
    """
    B, H, T = ADT.shape
    cs = ADT.cumsum(-1)
    segsum = cs.unsqueeze(-1) - cs.unsqueeze(-2)        # [b,h,t,s] = cs[t]-cs[s]
    causal = torch.ones(T, T, dtype=torch.bool, device=ADT.device).tril()
    segsum = torch.where(causal, segsum, torch.full_like(segsum, NEG_MASK))
    return segsum.exp()


# ---------------------------------------------------------------------------
# Reference layer
# ---------------------------------------------------------------------------

PARAM_NAMES = (
    "in_proj_weight", "out_proj_weight", "dt_bias", "A_log", "D",
    "B_bias", "C_bias", "B_norm_weight", "C_norm_weight",
    "MIMO_V", "MIMO_Z", "MIMO_O",
)


class Mamba3Reference(torch.nn.Module):
    """Reference implementation. Parameter shapes match TTMamba3Layer.get_params()."""

    def __init__(self, config, params: Dict[str, torch.Tensor] = None,
                 dtype=torch.float32, eps: float = 1e-6):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.eps = eps

        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.headdim = config.headdim
        self.nheads = config.nheads_m3
        self.d_state = config.d_state_m3
        self.R = config.mimo_rank
        self.ngroups = config.ngroups
        self.num_rope_angles = config.num_rope_angles

        split = int(self.d_state * config.rope_fraction)
        if split % 2 != 0:
            split -= 1
        self.split_tensor_size = split

        if self.ngroups != 1:
            raise NotImplementedError(
                "ngroups > 1 requires a real GQA repeat; the tt-nn layer only "
                "supports ngroups == 1 (ttnn.expand can only broadcast size-1 dims)."
            )

        self.d_in_proj = (2 * self.d_inner
                          + 2 * self.d_state * self.ngroups * self.R
                          + 3 * self.nheads
                          + self.num_rope_angles)

        if params is not None:
            self.load(params)

    def load(self, params: Dict[str, torch.Tensor]):
        """Install parameters (as leaf tensors with requires_grad=True)."""
        for name in PARAM_NAMES:
            t = params[name].detach().to(self.dtype).clone().requires_grad_(True)
            setattr(self, name, t)
        return self

    def params(self) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in PARAM_NAMES}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        cfg = self.config
        B, T, _ = x.shape
        H, R, N, P = self.nheads, self.R, self.d_state, self.headdim
        n_angles = self.num_rope_angles
        G = self.ngroups

        # --- in_proj, split as [z, x, B, C, dt, A, trap, angles] ---
        proj = x @ self.in_proj_weight

        sizes = [self.d_inner, self.d_inner, N * G * R, N * G * R, H, H, H, n_angles]
        z_raw, x_raw, B_raw, C_raw, dt_raw, A_raw, trap_raw, angles_raw = \
            torch.split(proj, sizes, dim=-1)

        # --- z, V kept in (B, T, H, P) ---
        z = z_raw.reshape(B, T, H, P)
        V = x_raw.reshape(B, T, H, P)

        # --- B, C -> (B, T, R, H, N) ---
        B_mat = B_raw.reshape(B, T, R, G, N).expand(B, T, R, H, N)
        C_mat = C_raw.reshape(B, T, R, G, N).expand(B, T, R, H, N)

        # --- QKNorm over d_state, then per-head/rank bias ---
        B_normed = rms_norm(B_mat, self.B_norm_weight, self.eps)
        C_normed = rms_norm(C_mat, self.C_norm_weight, self.eps)

        K = B_normed + self.B_bias.permute(1, 0, 2).reshape(1, 1, R, H, N)
        Q = C_normed + self.C_bias.permute(1, 0, 2).reshape(1, 1, R, H, N)

        # --- dt, A ---
        DT = F.softplus(dt_raw + self.dt_bias.reshape(1, 1, H))          # (B, T, H)
        A = -F.softplus(A_raw + self.A_log.reshape(1, 1, H))             # (B, T, H)
        ADT = (A * DT).permute(0, 2, 1)                                  # (B, H, T)

        # --- RoPE angles (complex SSM) ---
        angles = angles_raw.reshape(B, T, 1, n_angles).expand(B, T, H, n_angles)
        angles_proc = torch.tanh(angles) * math.pi
        angles_cumsum = (angles_proc * DT.reshape(B, T, H, 1)).cumsum(dim=1)
        cos_a, sin_a = angles_cumsum.cos(), angles_cumsum.sin()

        Q_rot = apply_rope(Q, cos_a, sin_a, self.split_tensor_size)
        K_rot = apply_rope(K, cos_a, sin_a, self.split_tensor_size)

        # --- Exponential-trapezoidal weights ---
        trap = torch.sigmoid(trap_raw)
        gamma = DT * trap
        shifted_gamma = shift_by_one(DT) * (1.0 - shift_by_one(trap))
        factor = gamma + shifted_gamma                                   # (B, T, H)

        K_scaled = K_rot * factor.reshape(B, T, 1, H, 1)

        # --- MIMO up-projections (element-wise over rank) ---
        MV = self.MIMO_V.permute(1, 0, 2).reshape(1, 1, R, H, P)
        MZ = self.MIMO_Z.permute(1, 0, 2).reshape(1, 1, R, H, P)
        V_proj = V.reshape(B, T, 1, H, P) * MV                           # (B, T, R, H, P)
        Z_proj = z.reshape(B, T, 1, H, P) * MZ

        # --- Quadratic (T x T) attention with R^2 cross-rank pairs ---
        L = decay_matrix(ADT)                                            # (B, H, T, T)
        Qp = Q_rot.permute(0, 3, 2, 1, 4)                                # (B, H, R, T, N)
        Kp = K_scaled.permute(0, 3, 2, 1, 4)
        Vp = V_proj.permute(0, 3, 2, 1, 4)                               # (B, H, R, T, P)

        QK = torch.einsum("bhqtn,bhksn->bhqkts", Qp, Kp)
        QK = QK * L[:, :, None, None, :, :]
        out_attn = torch.einsum("bhqkts,bhksp->bhqtp", QK, Vp)           # (B, H, R, T, P)

        # --- Trapezoidal diagonal correction (rotation cancels at t == s) ---
        Qb = Q.permute(0, 1, 3, 2, 4)                                    # (B, T, H, R, N)
        Kb = K.permute(0, 1, 3, 2, 4)
        qk_dot = (Qb @ Kb.transpose(-1, -2)) * shifted_gamma[..., None, None]
        qkv = qk_dot @ V_proj.permute(0, 1, 3, 2, 4)                     # (B, T, H, R, P)
        out_ssm = out_attn - qkv.permute(0, 2, 3, 1, 4)

        # --- D skip, z gate, MIMO output contraction ---
        Vpp = V_proj.permute(0, 3, 2, 1, 4)                              # (B, H, R, T, P)
        out_with_d = out_ssm + self.D.reshape(1, H, 1, 1, 1) * Vpp
        out_gated = out_with_d * F.silu(Z_proj.permute(0, 3, 2, 1, 4))

        out = torch.einsum("bhrtp,hrp->bhtp", out_gated, self.MIMO_O)

        out_flat = out.permute(0, 2, 1, 3).reshape(B, T, self.d_inner)
        return out_flat @ self.out_proj_weight
