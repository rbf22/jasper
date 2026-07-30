"""
Model implementation for the Mamba + Workspace POC.

Four parameter-matched cells at ~30M parameters:
  Cell A: pure Mamba2 (14 layers)
  Cell B: hybrid (12 Mamba2 + 2 attention at positions 5, 10)
  Cell C: hybrid + workspace (11 Mamba2 + 2 attention + perceiver workspace)
  Cell D: hybrid + workspace + recurrent core (layers 6-9 looped K times)

Config flags: use_attention, use_workspace, recurrent_core, k_train_max
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 14
    vocab_size: int = 128
    d_state: int = 64          # Mamba2 state dim
    d_conv: int = 4            # Mamba2 conv width
    expand: int = 4            # Mamba2 expand factor
    n_heads: int = 4           # heads for both Mamba2 SSD and attention
    # Cell flags
    use_attention: bool = False      # Cell A=False, B/C/D=True
    attention_positions: List[int] = field(default_factory=lambda: [5, 10])
    use_workspace: bool = False      # Cell A/B=False, C/D=True
    n_workspace_slots: int = 16
    recurrent_core: bool = False     # Cell D=True
    core_start: int = 6              # recurrent core layer range start
    core_end: int = 10               # recurrent core layer range end (exclusive)
    k_train_max: int = 6             # max K during training (sampled from {1..k_train_max})
    k_inference: int = 6             # K at inference (can be swept)
    # Training
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True  # set False for TT/XLA (ample device DRAM)
    # Derived
    @property
    def d_inner(self):
        return self.d_model * self.expand
    @property
    def d_head(self):
        return self.d_inner // self.n_heads


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """ZCRMSNorm: x * (1 + gamma) / RMS(x), gamma init 0 (identity-scale at init)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        # NOTE: Use pow(-0.5) instead of rsqrt() — TT XLA backend has a bug where
        # rsqrt()/sqrt() inside torch.compile(aot_autograd) corrupts the output graph,
        # producing wrong shapes. pow(-0.5) is mathematically equivalent and works correctly.
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).pow(-0.5)
        return (x.float() * norm).type_as(x) * (1 + self.weight.to(x.dtype))


# ---------------------------------------------------------------------------
# Gated Residual Block
# ---------------------------------------------------------------------------

class GatedBlock(nn.Module):
    """Gated pre-norm residual wrapper: x + sigmoid(gate) * layer(norm(x)).

    gate init 0 -> sigmoid(0) = 0.5, a damped residual at initialization.
    Lets the recurrent core learn to fade or sharpen each layer per iteration.
    """

    def __init__(self, layer: nn.Module, d_model: int, use_checkpoint: bool = True):
        super().__init__()
        self.layer = layer
        self.norm = RMSNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.use_checkpoint = use_checkpoint

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpoint:
            inner = checkpoint(
                self._inner_forward, x, use_reentrant=False
            )
        else:
            inner = self._inner_forward(x)
        return x + torch.sigmoid(self.gate) * inner


# ---------------------------------------------------------------------------
# Mamba2 Layer (pure PyTorch, SSD via O(n^2) attention form)
# ---------------------------------------------------------------------------

class Mamba2Layer(nn.Module):
    """Pure-PyTorch Mamba2 layer using the SSD (State Space Duality) form.

    Computes the selective scan as a decayed attention:
      Y[t] = sum_{s<=t} decay[t,s] * (C[t] @ B[s]) * V[s]

    This is O(T^2) but correct and works on MPS (no CUDA kernels needed).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_conv = config.d_conv

        # Projections
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
        )
        self.dt_proj = nn.Linear(self.d_inner, self.n_heads, bias=True)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Learnable parameters
        self.A_log = nn.Parameter(torch.randn(self.n_heads) - 0.5)
        self.D = nn.Parameter(torch.ones(self.n_heads))

        self.norm = RMSNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        w_dtype = self.conv1d.weight.dtype

        # Project to x_branch and z
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal conv1d — cast input to weight dtype (TT XLA may promote Linear to fp32)
        x_conv = x_branch.transpose(1, 2).to(w_dtype)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # causal: keep first T
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # Projections for SSD
        dt = F.softplus(self.dt_proj(x_conv))  # (B, T, n_heads)
        B_mat = self.B_proj(x_conv)  # (B, T, d_state)
        C_mat = self.C_proj(x_conv)  # (B, T, d_state)

        # V = x_conv reshaped for multi-head SSD
        V = x_conv.view(B, T, self.n_heads, self.d_head)  # (B, T, n_heads, d_head)

        # Compute decay: A = exp(A_log), per-position decay = exp(-dt * A)
        A = torch.exp(self.A_log)  # (n_heads,)
        decay = torch.exp(-dt * A.unsqueeze(0).unsqueeze(0))  # (B, T, n_heads)

        # SSD via O(n^2) attention form
        # CB[b, t, s] = C[b, t, :] @ B[b, s, :]  -- shared across heads
        CB = torch.matmul(C_mat, B_mat.transpose(1, 2))  # (B, T, T)

        # Decay matrix L[b, h, t, s] = prod_{i=s+1}^{t} decay[i, h]
        # Uses segment_sum (HuggingFace approach) instead of outer subtraction
        # of cumsum — the outer subtraction triggers a graph-truncation bug in
        # TT's AOT autograd backend.
        log_decay = torch.log(decay.clamp(min=1e-8))  # (B, T, n_heads)
        # segment_sum operates on the last dim — permute (B, T, H) -> (B, H, T)
        log_decay_h = log_decay.permute(0, 2, 1)  # (B, H, T)
        # Expand: (B, H, T) -> (B, H, T, T) where [t, s] = log_decay_h[t]
        log_decay_exp = log_decay_h.unsqueeze(-1).expand(B, self.n_heads, T, T)  # (B, H, T, T)
        # Mask: keep strictly lower triangular (s < t), zero rest
        mask_low = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
        log_decay_exp = log_decay_exp.masked_fill(~mask_low, 0.0)
        # Cumsum along dim=-2 (the t dimension)
        log_L = torch.cumsum(log_decay_exp, dim=-2)  # (B, H, T, T)
        # Mask: keep lower triangular incl diag (s <= t), -inf rest
        mask_causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        log_L = log_L.masked_fill(~mask_causal, float("-inf"))
        L = torch.exp(log_L)  # (B, H, T, T)

        # scores[b, h, t, s] = CB[b, t, s] * L[b, h, t, s]
        scores = CB.unsqueeze(1) * L  # (B, n_heads, T, T)

        # Y[b, h, t, i] = sum_s scores[b, h, t, s] * V[b, s, h, i]
        # Upcast only this matmul to fp32 — accumulation over T=128 can overflow bf16/fp16
        V_h = V.permute(0, 2, 1, 3)  # (B, n_heads, T, d_head)
        Y = torch.matmul(scores.float(), V_h.float()).type_as(x)

        # Skip connection (D residual)
        Y = Y + self.D.view(1, self.n_heads, 1, 1) * V_h

        # Reshape back: (B, T, d_inner)
        Y = Y.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_inner)

        # Gate with z
        Y = Y * F.silu(z)

        # Norm and output projection
        Y = self.norm(Y)
        out = self.out_proj(Y)  # (B, T, D)

        return out


# ---------------------------------------------------------------------------
# Multi-Head Attention Layer
# ---------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    """Standard multi-head self-attention with RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # RoPE frequencies
        d_rope = self.d_head
        freqs = 1.0 / (10000 ** (torch.arange(0, d_rope, 2).float() / d_rope))
        self.register_buffer("rope_freqs", freqs)

    def _apply_rope(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Apply RoPE to x: (B, n_heads, T, d_head)"""
        positions = torch.arange(T, device=x.device).to(x.dtype)
        angles = torch.outer(positions, self.rope_freqs.to(x.dtype))  # (T, d_rope/2)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_rope/2)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)

        # Causal attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Workspace Module (perceiver-style cross-attention)
# ---------------------------------------------------------------------------

class WorkspaceModule(nn.Module):
    """Perceiver-style workspace with m learned slot vectors.

    Twice per application:
      1. Slots read from hidden states (slots attend over sequence)
      2. Hidden states write back from slots (sequence attends over slots)

    This is the 'engineered J-space' — a compact, persistent scratchpad.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_slots = config.n_workspace_slots
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.spectral_norm_bound = getattr(config, 'spectral_norm_bound', 5.0)

        # Learnable slot embeddings
        self.slots = nn.Parameter(torch.randn(self.n_slots, self.d_model) * 0.02)

        # Read: slots attend over hidden states
        self.read_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_out = nn.Linear(self.d_model, self.d_model, bias=False)

        # Write: hidden states attend over slots
        self.write_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.write_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.write_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.write_out = nn.Linear(self.d_model, self.d_model, bias=False)

        self.norm = RMSNorm(self.d_model)
        self.slot_norm = RMSNorm(self.d_model)
        # Zero-init gates: sigmoid(-5) ≈ 0.007, workspace starts as near-identity
        gate_init = getattr(config, 'gate_init', -2.0)
        self.read_gate = nn.Parameter(torch.tensor([gate_init]))
        self.write_gate = nn.Parameter(torch.tensor([gate_init]))
        # Slot decay: makes the slot update contractive (restoring force)
        slot_decay_init = getattr(config, 'slot_decay_init', 1.0)
        self.slot_decay = nn.Parameter(torch.tensor([slot_decay_init]))

    def normalize_slots(self):
        """Normalize slot parameters to unit RMS to prevent unbounded growth.

        This breaks the feedback loop where large slots → sharp attention →
        large gradients → larger slots. Called after each optimizer step.
        """
        with torch.no_grad():
            rms = self.slots.pow(2).mean().sqrt().clamp(min=1e-6)
            self.slots.data.div_(rms)

    def spectral_normalize_weights(self):
        """Apply spectral normalization to all workspace weight matrices.

        Divides each weight matrix by its spectral norm (largest singular value),
        then scales to the configured bound. This constrains the Lipschitz constant
        to spectral_norm_bound, preventing unbounded weight growth while allowing
        sufficient attention selectivity.

        Uses power iteration (10 steps) for efficiency — same algorithm as
        Spectral Normalization GANs (Miyato et al., 2018).
        """
        bound = self.spectral_norm_bound
        weight_modules = [
            self.read_q, self.read_k, self.read_v, self.read_out,
            self.write_q, self.write_k, self.write_v, self.write_out,
        ]
        with torch.no_grad():
            for module in weight_modules:
                W = module.weight  # (D, D)
                D = W.shape[0]
                v = torch.randn(D, 1, device=W.device, dtype=W.dtype) / math.sqrt(D)
                for _ in range(10):
                    u = W @ v
                    u = u / u.norm().clamp(min=1e-6)
                    v = W.t() @ u
                    v = v / v.norm().clamp(min=1e-6)
                sigma = (W @ v).norm().clamp(min=1e-6)
                # Cap: only scale down if sigma > bound, leave unchanged if below
                scale = min(1.0, bound / sigma.item())
                module.weight.data.mul_(scale)

    def forward(self, x: torch.Tensor, slot_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D) — hidden states
        slot_state: (B, m, D) — incoming workspace state (for recurrence across iterations)
        returns: (x_updated, slot_state_updated)
        """
        B, T, D = x.shape

        # Initialize or use incoming slot state
        if slot_state is None:
            slots = self.slots.unsqueeze(0).expand(B, -1, -1)  # (B, m, D)
        else:
            slots = slot_state

        # --- Read: slots attend over hidden states ---
        rq = self.read_q(slots).view(B, self.n_slots, self.n_heads, self.d_head).transpose(1, 2)
        rk = self.read_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        rv = self.read_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        read_attn = torch.matmul(rq, rk.transpose(-2, -1)) * self.scale  # (B, n_heads, m, T)
        read_attn = F.softmax(read_attn, dim=-1)
        read_out = torch.matmul(read_attn, rv)  # (B, n_heads, m, d_head)
        read_out = read_out.transpose(1, 2).contiguous().view(B, self.n_slots, D)
        slots = self.slot_norm(self.slot_decay * slots + torch.sigmoid(self.read_gate) * self.read_out(read_out))  # decay + gated residual + normalize

        # --- Write: hidden states attend over slots ---
        wq = self.write_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        wk = self.write_k(slots).view(B, self.n_slots, self.n_heads, self.d_head).transpose(1, 2)
        wv = self.write_v(slots).view(B, self.n_slots, self.n_heads, self.d_head).transpose(1, 2)

        write_attn = torch.matmul(wq, wk.transpose(-2, -1)) * self.scale  # (B, n_heads, T, m)
        write_attn = F.softmax(write_attn, dim=-1)
        write_out = torch.matmul(write_attn, wv)  # (B, n_heads, T, d_head)
        write_out = write_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + torch.sigmoid(self.write_gate) * self.write_out(write_out)  # gated residual update to hidden states

        x = self.norm(x)
        return x, slots


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class MambaWorkspaceModel(nn.Module):
    """The full model with config flags for all four cells."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            if config.use_attention and i in config.attention_positions:
                inner = AttentionLayer(config)
            else:
                inner = Mamba2Layer(config)
            self.layers.append(GatedBlock(inner, config.d_model, use_checkpoint=config.use_gradient_checkpointing))

        # Workspace module
        if config.use_workspace:
            self.workspace = WorkspaceModule(config)
        else:
            self.workspace = None

        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        k_override: Optional[int] = None,
        return_workspace_states: bool = False,
        k_value: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids: (B, T)
        k_override: override K for inference (otherwise use config.k_inference)
        return_workspace_states: if True, return workspace states per iteration (for probing)
        k_value: pre-sampled K tensor for training (passed from outside to avoid
                 torch.randint inside the compiled graph — TT hardware doesn't
                 support the remainder op that randint uses internally)

        Returns dict with 'logits' and optionally 'workspace_states'.

        K_max padding: when not probing, the recurrent core always loops k_train_max
        times (static loop count for XLA compilation). Iterations beyond the sampled
        K are masked to identity via blend: x = active * x_new + (1-active) * x.
        """
        config = self.config
        B, T = input_ids.shape
        x = self.token_emb(input_ids)  # (B, T, D)

        # Determine K (number of active recurrent iterations)
        k_max = config.k_train_max  # static loop count for XLA
        if self.training:
            if k_value is not None:
                K = k_value  # pre-generated tensor passed from training loop
            else:
                K = random_k_tensor(config.k_train_max, device=input_ids.device)
        elif k_override is not None:
            K = k_override
        else:
            K = config.k_inference

        # For probing (return_workspace_states), use dynamic loop with exact K (not compiled)
        use_padding = config.recurrent_core and not return_workspace_states

        workspace_states = [] if return_workspace_states else None
        slot_state = None

        # Phase 1: pre-core layers (0..core_start-1)
        pre_core_end = config.core_start if config.recurrent_core else config.n_layers
        for i in range(pre_core_end):
            if config.use_workspace and i in config.attention_positions:
                x, slot_state = self.workspace(x, slot_state)
                if return_workspace_states:
                    workspace_states.append(("pre_layer", i, slot_state.clone()))
            x = self.layers[i](x)

        # Phase 2: recurrent core (core_start..core_end-1)
        if config.recurrent_core:
            core_layers = list(range(config.core_start, config.core_end))

            # 1/sqrt(K) residual scaling: normalizes gradient accumulation across K
            # iterations so that the total gradient norm is independent of K.
            # The blend factor becomes active/sqrt(K) instead of active, turning the
            # full replacement (x = x_new) into a partial update:
            #   x = (1 - active/sqrt(K)) * x + (active/sqrt(K)) * x_new
            import math as _math
            K_val = K.item() if isinstance(K, torch.Tensor) else K
            k_scale = 1.0 / _math.sqrt(K_val) if K_val > 0 else 0.0

            if use_padding:
                # XLA-friendly: always loop k_max times, mask iterations >= K
                for iteration in range(k_max):
                    active = (torch.tensor(float(iteration), device=x.device) < K).to(x.dtype)
                    blend = active * k_scale  # scaled blend factor

                    # Run one core iteration
                    x_new = x
                    slot_state_new = slot_state
                    for i in core_layers:
                        if config.use_workspace and i in config.attention_positions:
                            x_new, slot_state_new = self.workspace(x_new, slot_state_new)
                        x_new = self.layers[i](x_new)
                    if config.use_workspace:
                        x_new, slot_state_new = self.workspace(x_new, slot_state_new)

                    # Blend: active iterations partially update state (1/sqrt(K) scaling),
                    # padding iterations keep old state.
                    x = blend * x_new + (1 - blend) * x
                    slot_state = blend * slot_state_new + (1 - blend) * slot_state
            else:
                # Probing path: loop exactly K times (dynamic, not compiled)
                K_int = int(K.item()) if isinstance(K, torch.Tensor) else K
                for iteration in range(K_int):
                    for i in core_layers:
                        if config.use_workspace and i in config.attention_positions:
                            x, slot_state = self.workspace(x, slot_state)
                            if return_workspace_states:
                                workspace_states.append(("iter", iteration, i, slot_state.clone()))
                        x = self.layers[i](x)

                    if config.use_workspace:
                        x, slot_state = self.workspace(x, slot_state)
                        if return_workspace_states:
                            workspace_states.append(("iter_end", iteration, slot_state.clone()))

        # Phase 3: post-core layers (core_end..n_layers-1)
        post_core_start = config.core_end if config.recurrent_core else config.n_layers
        for i in range(post_core_start, config.n_layers):
            if config.use_workspace and i in config.attention_positions:
                x, slot_state = self.workspace(x, slot_state)
                if return_workspace_states:
                    workspace_states.append(("post_layer", i, slot_state.clone()))
            x = self.layers[i](x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result = {"logits": logits}
        if workspace_states is not None:
            result["workspace_states"] = workspace_states
        return result

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def normalize_workspace_slots(self):
        """Normalize workspace parameters after each optimizer step.

        Applies slot normalization (unit RMS) and spectral normalization
        (spectral norm 1 for all weight matrices). No-op if no workspace.
        """
        if self.workspace is not None:
            self.workspace.normalize_slots()
            self.workspace.spectral_normalize_weights()


def random_k(k_max: int, device: torch.device) -> int:
    """Sample K uniformly from {1..k_max} during training. Returns a Python int."""
    return torch.randint(1, k_max + 1, (1,), device=device).item()


def random_k_tensor(k_max: int, device: torch.device) -> torch.Tensor:
    """Sample K uniformly from {1..k_max} as a tensor (no host sync — XLA-friendly).

    Generated on CPU then moved to device — torch.randint uses a remainder op
    internally that is not supported on TT hardware with UINT32.
    """
    k = torch.randint(1, k_max + 1, (1,), device="cpu").to(torch.float32)
    return k.to(device)


# ---------------------------------------------------------------------------
# Cell configurations
# ---------------------------------------------------------------------------

def get_cell_config(cell: str) -> ModelConfig:
    """Return the ModelConfig for a given cell (A, B, C, or D)."""
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
            n_layers=13,  # remove 1 Mamba layer to compensate for workspace params
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
    else:
        raise ValueError(f"Unknown cell: {cell}")


if __name__ == "__main__":
    # Quick param count check
    for cell in ["A", "B", "C", "D"]:
        config = get_cell_config(cell)
        model = MambaWorkspaceModel(config)
        n_params = model.get_num_params()
        print(f"Cell {cell}: {n_params / 1e6:.1f}M params ({n_params:,})")
