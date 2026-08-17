"""Tenstorrent-native implementation of the text latent-memory model.

This is a TT-NN implementation of TextLatentMemoryModel that runs the forward
pass on Tenstorrent hardware. The backward pass uses PyTorch autograd with
ttnn.to_torch/from_torch at layer boundaries — a hybrid approach that gets
device acceleration for the compute-heavy forward pass while avoiding the
error-prone manual backward implementation.

Memory management follows the patterns from model_ttnn.py:
  - _safe_deallocate for all intermediate ttnn tensors
  - ttnn.synchronize_device before deallocation
  - clear_caches between training steps
  - Persistent tensors (weights, cached constants) are never deallocated

The model architecture matches text_latent_memory_model.py:
  1. Encode prompt once (transformer encoder on device)
  2. Initialize fixed-size memory slots (cross-attention on device)
  3. Recurrent reasoning steps over memory (on device)
  4. Decode answer via cross-attention to memory (on device)
"""

import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List


# ---------------------------------------------------------------------------
# Memory management (same patterns as model_ttnn.py)
# ---------------------------------------------------------------------------

def _safe_deallocate(tensor):
    """Deallocate a ttnn device tensor, ignoring errors."""
    if tensor is None:
        return
    try:
        ttnn.deallocate(tensor)
    except Exception:
        pass


def to_device(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    """Convert a PyTorch tensor to a tt-nn device tensor."""
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def from_device(tensor: "ttnn.Tensor") -> torch.Tensor:
    """Convert a tt-nn device tensor to a PyTorch tensor (float32)."""
    return ttnn.to_torch(tensor).to(torch.float32)


# ---------------------------------------------------------------------------
# TT-NN building blocks
# ---------------------------------------------------------------------------

class TTRMSNorm:
    """RMSNorm using ttnn.rms_norm — same as model_ttnn.py."""

    def __init__(self, d: int, device, eps=1e-6, dtype=ttnn.bfloat16):
        self.d = d
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = ttnn.from_torch(
            torch.ones(d, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)

    def backward(self, grad_out: "ttnn.Tensor", x: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """RMSNorm backward computed on host (avoids TT-NN buffer aliasing issues).

        Forward: y = x / rms(x) * weight
        where rms = sqrt(mean(x^2) + eps)

        Backward:
          grad_x = (grad_w - x_norm * mean(grad_w * x_norm)) / rms
          grad_weight = sum(grad_out * x_norm, over all leading dims)
        where grad_w = grad_out * weight, x_norm = x / rms
        """
        eps = self.eps

        # Move to host for computation — TT-NN buffer aliasing makes on-device
        # RMSNorm backward unreliable with manual deallocation
        x_torch = ttnn.to_torch(x).float()
        go_torch = ttnn.to_torch(grad_out).float()
        w_torch = ttnn.to_torch(self.weight).float()

        rms_sq = (x_torch ** 2).mean(-1, keepdim=True)
        rms = torch.sqrt(rms_sq + eps)
        x_norm_t = x_torch / rms

        grad_w_t = go_torch * w_torch
        grad_x_t = (grad_w_t - x_norm_t * (grad_w_t * x_norm_t).mean(-1, keepdim=True)) / rms

        reduce_dims = tuple(range(go_torch.ndim - 1))
        grad_weight_t = (go_torch * x_norm_t).sum(dim=reduce_dims)

        grad_x = to_device(grad_x_t, self.device, dtype=self.dtype)
        grad_weight = to_device(grad_weight_t, self.device, dtype=self.dtype)

        return grad_x, grad_weight


class TTLayerNorm:
    """LayerNorm using ttnn.layer_norm — matches PyTorch nn.LayerNorm."""

    def __init__(self, d: int, device, eps=1e-5, dtype=ttnn.bfloat16):
        self.d = d
        self.eps = eps
        self.device = device
        self.dtype = dtype
        torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
        self.weight = ttnn.from_torch(
            torch.ones(d, dtype=torch_dtype),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            torch.zeros(d, dtype=torch_dtype),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)

    def backward(self, grad_out: "ttnn.Tensor", x: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor", "ttnn.Tensor"]:
        """LayerNorm backward computed on host (avoids TT-NN buffer aliasing issues).

        Forward: y = (x - mean) * rstd * weight + bias

        Backward:
          grad_x = (grad_w - mean(grad_w) - x_norm * mean(grad_w * x_norm)) * rstd
          grad_weight = sum(grad_out * x_norm, over all leading dims)
          grad_bias = sum(grad_out, over all leading dims)
        where grad_w = grad_out * weight
        """
        eps = self.eps

        # Move to host for computation — TT-NN buffer aliasing makes on-device
        # LayerNorm backward unreliable with manual deallocation
        x_torch = ttnn.to_torch(x).float()
        go_torch = ttnn.to_torch(grad_out).float()
        w_torch = ttnn.to_torch(self.weight).float()

        mean_t = x_torch.mean(-1, keepdim=True)
        var_t = ((x_torch - mean_t) ** 2).mean(-1, keepdim=True)
        rstd_t = 1.0 / torch.sqrt(var_t + eps)
        x_norm_t = (x_torch - mean_t) * rstd_t

        grad_w_t = go_torch * w_torch
        grad_x_t = (grad_w_t - grad_w_t.mean(-1, keepdim=True) -
                    x_norm_t * (grad_w_t * x_norm_t).mean(-1, keepdim=True)) * rstd_t

        # Reduce over all leading dims for weight and bias grads
        reduce_dims = tuple(range(go_torch.ndim - 1))
        grad_weight_t = (go_torch * x_norm_t).sum(dim=reduce_dims)
        grad_bias_t = go_torch.sum(dim=reduce_dims)

        # Move back to device
        grad_x = to_device(grad_x_t, self.device, dtype=self.dtype)
        grad_weight = to_device(grad_weight_t, self.device, dtype=self.dtype)
        grad_bias = to_device(grad_bias_t, self.device, dtype=self.dtype)

        return grad_x, grad_weight, grad_bias


class TTLinear:
    """Linear layer: y = x @ W + b.

    ttnn.linear(x, W) computes x @ W, so W must be (in_features, out_features).
    """

    def __init__(self, in_features: int, out_features: int, device, bias=True, dtype=ttnn.bfloat16):
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.dtype = dtype
        torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
        # NOTE: ttnn.linear(x, W) computes x @ W, so W is (in, out) — transposed from PyTorch
        w = torch.randn(in_features, out_features, dtype=torch_dtype) * 0.02
        self.weight = to_device(w, device, dtype=dtype)
        if bias:
            b = torch.zeros(out_features, dtype=torch_dtype)
            self.bias = to_device(b, device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        if self.has_bias:
            return ttnn.linear(x, self.weight, bias=self.bias)
        return ttnn.linear(x, self.weight)

    def backward(self, grad_out: "ttnn.Tensor", x: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor", Optional["ttnn.Tensor"]]:
        """Linear backward: y = x @ W + b

        grad_x = grad_out @ W^T
        grad_W = x^T @ grad_out  (summed over batch/seq dims)
        grad_b = sum(grad_out, over batch/seq dims)

        Returns: (grad_x, grad_weight, grad_bias)
        """
        # grad_x = grad_out @ W^T  (W is (in, out), W^T is (out, in))
        w_t = ttnn.transpose(self.weight, -2, -1)  # (out, in)
        grad_x = ttnn.matmul(grad_out, w_t)
        _safe_deallocate(w_t)

        # grad_W = x^T @ grad_out  (x is (..., in), grad_out is (..., out))
        # Need to flatten leading dims for matmul
        x_2d = ttnn.reshape(x, [-1, self.in_features])  # (N, in)
        grad_out_2d = ttnn.reshape(grad_out, [-1, self.out_features])  # (N, out)
        grad_weight = ttnn.matmul(ttnn.transpose(x_2d, -2, -1), grad_out_2d)  # (in, out)
        # Don't deallocate x_2d or grad_out_2d — they may be views of x/grad_out
        # which the caller still needs

        # grad_b = sum over all leading dims
        # NOTE: Don't modify grad_out in-place — the caller may still need it.
        # Use ttnn.sum with all leading dims at once instead of iterative reduction.
        grad_bias = None
        if self.has_bias:
            reduce_dims = tuple(range(len(grad_out.shape) - 1))
            grad_bias = ttnn.sum(grad_out, dim=reduce_dims) if reduce_dims else grad_out

        return grad_x, grad_weight, grad_bias


class TTMultiHeadAttention:
    """Multi-head attention on device with combined QKV projection.

    Matches PyTorch nn.MultiheadAttention:
    - in_proj_weight: (3*d_model, d_model) — combined QKV
    - in_proj_bias: (3*d_model,)
    - out_proj.weight: (d_model, d_model)
    - out_proj.bias: (d_model,)
    """

    def __init__(self, d_model: int, n_heads: int, device, dropout=0.0, dtype=ttnn.bfloat16):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.device = device
        self.dtype = dtype
        torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

        # Combined QKV projection: (d_model, 3*d_model) for ttnn.linear
        self.in_proj_weight = to_device(
            torch.randn(d_model, 3 * d_model, dtype=torch_dtype) * 0.02, device, dtype=dtype
        )
        self.in_proj_bias = to_device(
            torch.zeros(3 * d_model, dtype=torch_dtype), device, dtype=dtype
        )
        # Output projection
        self.out_proj_weight = to_device(
            torch.randn(d_model, d_model, dtype=torch_dtype) * 0.02, device, dtype=dtype
        )
        self.out_proj_bias = to_device(
            torch.zeros(d_model, dtype=torch_dtype), device, dtype=dtype
        )

        # Cache scale as device tensor
        self._scale_tt = ttnn.from_torch(
            torch.tensor([self.scale], dtype=torch_dtype),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def _reshape_to_heads(self, x, B, L):
        """(B, L, D) -> (B, H, L, d_head)"""
        return ttnn.permute(ttnn.reshape(x, [B, L, self.n_heads, self.d_head]), [0, 2, 1, 3])

    def _reshape_from_heads(self, x, B, L):
        """(B, H, L, d_head) -> (B, L, D)"""
        return ttnn.reshape(ttnn.permute(x, [0, 2, 1, 3]), [B, L, self.d_model])

    def forward(
        self,
        query: "ttnn.Tensor",
        key: "ttnn.Tensor",
        value: "ttnn.Tensor",
        key_padding_mask: Optional["ttnn.Tensor"] = None,
        attn_mask: Optional["ttnn.Tensor"] = None,
        training: bool = False,
    ) -> "ttnn.Tensor":
        """Forward pass for multi-head attention with combined QKV.

        For self-attention (query==key==value), uses a single combined projection.
        For cross-attention, projects query separately and combines key+value.

        Args:
            query: (B, L_q, d_model)
            key: (B, L_k, d_model)
            value: (B, L_k, d_model)
            key_padding_mask: (B, L_k) — additive mask (0=valid, -inf=padding)
            attn_mask: (L_q, L_k) — additive mask (0=keep, -inf=mask)
            training: if True, cache intermediates for backward

        Returns: (B, L_q, d_model)
        """
        device = self.device
        B = query.shape[0]
        L_q = query.shape[1]
        L_k = key.shape[1]
        d = self.d_model
        self._is_self_attn = (query is key and key is value)

        if self._is_self_attn:
            # Self-attention: single combined QKV projection
            qkv = ttnn.linear(query, self.in_proj_weight, bias=self.in_proj_bias)
            qkv_parts = ttnn.split(qkv, d, dim=-1)
            q, k, v = qkv_parts[0], qkv_parts[1], qkv_parts[2]
            _safe_deallocate(qkv)
        else:
            # Cross-attention: project query and key separately through in_proj_weight
            qkv_q = ttnn.linear(query, self.in_proj_weight, bias=self.in_proj_bias)
            qkv_parts = ttnn.split(qkv_q, d, dim=-1)
            q = qkv_parts[0]
            _safe_deallocate(qkv_q)

            qkv_kv = ttnn.linear(key, self.in_proj_weight, bias=self.in_proj_bias)
            kv_parts = ttnn.split(qkv_kv, d, dim=-1)
            k, v = kv_parts[1], kv_parts[2]
            _safe_deallocate(qkv_kv)

        q = self._reshape_to_heads(q, B, L_q)  # (B, H, L_q, d_h)
        k = self._reshape_to_heads(k, B, L_k)  # (B, H, L_k, d_h)
        v = self._reshape_to_heads(v, B, L_k)  # (B, H, L_k, d_h)

        # Scores: (B, H, L_q, L_k) = q @ k^T * scale
        k_t = ttnn.transpose(k, -2, -1)  # (B, H, d_h, L_k)
        scores = ttnn.matmul(q, k_t)  # (B, H, L_q, L_k)
        _safe_deallocate(k_t)
        scores = ttnn.mul(scores, self._scale_tt)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            kpm = ttnn.reshape(key_padding_mask, [B, 1, 1, L_k])
            scores = ttnn.add(scores, kpm)
            _safe_deallocate(kpm)

        # Apply attention mask if provided (causal mask)
        if attn_mask is not None:
            scores = ttnn.add(scores, attn_mask)

        # Softmax over last dim
        attn = ttnn.softmax(scores, dim=-1)  # (B, H, L_q, L_k)
        _safe_deallocate(scores)

        # Output: (B, H, L_q, d_head) = attn @ v
        out_heads = ttnn.matmul(attn, v)  # (B, H, L_q, d_h)

        # Reshape back: (B, L_q, d_model)
        out_pre = self._reshape_from_heads(out_heads, B, L_q)
        _safe_deallocate(out_heads)

        if training:
            # Cache for backward
            self._cache = {
                "query": query, "key": key, "attn": attn,
                "q": q, "k": k, "v": v,
                "out_pre": out_pre,
                "B": B, "L_q": L_q, "L_k": L_k,
            }
        else:
            _safe_deallocate(attn)
            _safe_deallocate(q)
            _safe_deallocate(k)
            _safe_deallocate(v)

        # Output projection
        out = ttnn.linear(out_pre, self.out_proj_weight, bias=self.out_proj_bias)

        if not training:
            _safe_deallocate(out_pre)

        return out

    def backward(self, grad_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor", Dict[str, "ttnn.Tensor"]]:
        """Backward pass for multi-head attention.

        Args:
            grad_out: (B, L_q, d_model) gradient w.r.t. attention output

        Returns:
            grad_input: (B, L_q, d_model) gradient w.r.t. query input
                        (for self-attention; for cross-attn, returns grad_query)
            grad_key: (B, L_k, d_model) gradient w.r.t. key input (None for self-attn)
            grads: dict of parameter gradients {
                in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
            }
        """
        device = self.device
        c = self._cache
        B, L_q, L_k = c["B"], c["L_q"], c["L_k"]
        d = self.d_model
        H = self.n_heads
        d_h = d // H

        # Move everything to host
        grad_out_t = ttnn.to_torch(grad_out).float()
        q_t = ttnn.to_torch(c["q"]).float()    # (B, H, L_q, d_h) or (B, H, L_k, d_h)
        k_t = ttnn.to_torch(c["k"]).float()    # (B, H, L_k, d_h)
        v_t = ttnn.to_torch(c["v"]).float()    # (B, H, L_k, d_h)
        attn_t = ttnn.to_torch(c["attn"]).float()  # (B, H, L_q, L_k)
        out_pre_t = ttnn.to_torch(c["out_pre"]).float()  # (B, L_q, d)
        opw_t = ttnn.to_torch(self.out_proj_weight).float()  # (d, d)
        opb_t = ttnn.to_torch(self.out_proj_bias).float()  # (d,)
        ipw_t = ttnn.to_torch(self.in_proj_weight).float()  # (d, 3d) TT-NN layout
        ipb_t = ttnn.to_torch(self.in_proj_bias).float()  # (3d,)
        scale = self.scale

        # Out proj backward: y = out_pre @ opw + opb (TT-NN: W is (in, out) = (d, d))
        N = B * L_q
        grad_opw = out_pre_t.reshape(N, d).T @ grad_out_t.reshape(N, d)  # (d, d)
        grad_opb = grad_out_t.reshape(N, d).sum(0)  # (d,)
        grad_out_pre = grad_out_t.reshape(N, d) @ opw_t.T  # (N, d) -> (B, L_q, d)

        # Reshape to heads
        grad_heads = grad_out_pre.reshape(B, L_q, H, d_h).permute(0, 2, 1, 3)  # (B, H, L_q, d_h)

        # attn @ v backward
        grad_attn = torch.matmul(grad_heads, v_t.transpose(-2, -1))  # (B, H, L_q, L_k)
        grad_v = torch.matmul(attn_t.transpose(-2, -1), grad_heads)  # (B, H, L_k, d_h)

        # Softmax backward: grad_scores = (grad_attn - sum(grad_attn * attn, -1, keepdim)) * attn
        grad_attn_sum = (grad_attn * attn_t).sum(-1, keepdim=True)
        grad_scores = (grad_attn - grad_attn_sum) * attn_t  # (B, H, L_q, L_k)

        # Scale backward
        grad_scores_pre = grad_scores * scale  # (B, H, L_q, L_k)

        # scores = q @ k^T backward
        grad_q = torch.matmul(grad_scores_pre, k_t)  # (B, H, L_q, d_h)
        grad_k = torch.matmul(grad_scores_pre.transpose(-2, -1), q_t)  # (B, H, L_k, d_h)

        # Reshape from heads
        grad_q_flat = grad_q.permute(0, 2, 1, 3).reshape(B, L_q, d)  # (B, L_q, d)
        grad_k_flat = grad_k.permute(0, 2, 1, 3).reshape(B, L_k, d)  # (B, L_k, d)
        grad_v_flat = grad_v.permute(0, 2, 1, 3).reshape(B, L_k, d)  # (B, L_k, d)

        if self._is_self_attn:
            # Self-attention: single projection qkv = x @ ipw + ipb (TT-NN: ipw is (d, 3d))
            query_input_t = ttnn.to_torch(c["query"]).float()
            grad_qkv = torch.cat([grad_q_flat, grad_k_flat, grad_v_flat], dim=-1)  # (B, L, 3d)

            qi_2d = query_input_t.reshape(N, d)
            gqkv_2d = grad_qkv.reshape(N, 3 * d)
            grad_ipw = qi_2d.T @ gqkv_2d  # (d, 3d)
            grad_ipb = gqkv_2d.sum(0)  # (3d,)
            grad_input_t = gqkv_2d @ ipw_t.T  # (N, d) -> (B, L, d)
            grad_key_t = None
        else:
            # Cross-attention: query and key have separate inputs
            # TT-NN: ipw is (d, 3d), forward: qkv = x @ ipw + ipb
            query_input_t = ttnn.to_torch(c["query"]).float()  # (B, L_q, d)
            key_input_t = ttnn.to_torch(c["key"]).float()  # (B, L_k, d)

            # Query projection: q = query @ ipw[:d, :d] (only first d output columns used)
            Nq = B * L_q
            grad_qkv_q = torch.cat([grad_q_flat, torch.zeros(B, L_q, 2 * d), ], dim=-1)
            qi_2d = query_input_t.reshape(Nq, d)
            gqkv_q_2d = grad_qkv_q.reshape(Nq, 3 * d)
            grad_ipw_q = qi_2d.T @ gqkv_q_2d  # (d, 3d)
            grad_input_t = gqkv_q_2d @ ipw_t.T  # (Nq, d) -> (B, L_q, d)

            # Key projection: k,v = key @ ipw (uses d:3d output columns)
            Nk = B * L_k
            grad_qkv_kv = torch.cat([torch.zeros(B, L_k, d), grad_k_flat, grad_v_flat], dim=-1)
            ki_2d = key_input_t.reshape(Nk, d)
            gqkv_kv_2d = grad_qkv_kv.reshape(Nk, 3 * d)
            grad_ipw_kv = ki_2d.T @ gqkv_kv_2d  # (d, 3d)
            grad_key_t = gqkv_kv_2d @ ipw_t.T  # (Nk, d) -> (B, L_k, d)

            grad_ipw = grad_ipw_q + grad_ipw_kv
            # Correct bias gradient: [sum(grad_q), sum(grad_k), sum(grad_v)]
            grad_ipb = torch.cat([
                grad_q_flat.reshape(Nq, d).sum(0),
                grad_k_flat.reshape(Nk, d).sum(0),
                grad_v_flat.reshape(Nk, d).sum(0),
            ])  # (3d,)

        # Move results back to device (reshape to 3D)
        grad_input = to_device(grad_input_t.reshape(B, L_q, d), device, dtype=self.dtype)
        grad_key_input = to_device(grad_key_t.reshape(B, L_k, d), device, dtype=self.dtype) if grad_key_t is not None else None
        grad_ipw_tt = to_device(grad_ipw, device, dtype=self.dtype)
        grad_ipb_tt = to_device(grad_ipb, device, dtype=self.dtype)
        grad_opw_tt = to_device(grad_opw, device, dtype=self.dtype)
        grad_opb_tt = to_device(grad_opb, device, dtype=self.dtype)

        # Clean up cache — don't deallocate query/key inputs as they may be
        # shared with the model-level cache (e.g. enc_pre_norm used as key)
        _safe_deallocate(c["q"])
        _safe_deallocate(c["k"])
        _safe_deallocate(c["v"])
        _safe_deallocate(c["attn"])
        _safe_deallocate(c["out_pre"])
        # query and key are NOT deallocated here — caller manages their lifetime
        self._cache = {}

        grads = {
            "in_proj_weight": grad_ipw_tt,
            "in_proj_bias": grad_ipb_tt,
            "out_proj_weight": grad_opw_tt,
            "out_proj_bias": grad_opb_tt,
        }
        return grad_input, grad_key_input, grads


class TTEncoderLayer:
    """Transformer encoder layer matching PyTorch nn.TransformerEncoderLayer (pre-norm, GELU).

    PyTorch structure (norm_first=True):
      x = x + self_attn(norm1(x))
      x = x + ffn(norm2(x))
    where norm1/norm2 are LayerNorm (with bias), ffn uses GELU.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, device, dropout=0.0, dtype=ttnn.bfloat16):
        self.device = device
        self.d_model = d_model
        self.self_attn = TTMultiHeadAttention(d_model, n_heads, device, dropout, dtype=dtype)
        self.norm1 = TTLayerNorm(d_model, device, dtype=dtype)
        self.norm2 = TTLayerNorm(d_model, device, dtype=dtype)
        self.linear1 = TTLinear(d_model, d_ff, device, bias=True, dtype=dtype)
        self.linear2 = TTLinear(d_ff, d_model, device, bias=True, dtype=dtype)

    def forward(self, x: "ttnn.Tensor", key_padding_mask: Optional["ttnn.Tensor"] = None,
                training: bool = False) -> "ttnn.Tensor":
        # Pre-norm self-attention
        normed = self.norm1.forward(x)
        attn_out = self.self_attn.forward(normed, normed, normed,
                                          key_padding_mask=key_padding_mask, training=training)
        x_new = ttnn.add(x, attn_out)

        # Pre-norm FFN (GELU activation — matches PyTorch TransformerEncoderLayer)
        normed2 = self.norm2.forward(x_new)
        hidden = ttnn.gelu(self.linear1.forward(normed2))
        ffn_out = self.linear2.forward(hidden)
        x_out = ttnn.add(x_new, ffn_out)

        if training:
            self._cache = {
                "x": x, "normed": normed, "attn_out": attn_out,
                "x_new": x_new, "normed2": normed2, "hidden": hidden, "ffn_out": ffn_out,
                "key_padding_mask": key_padding_mask,
            }
        else:
            _safe_deallocate(attn_out)
            _safe_deallocate(normed)
            _safe_deallocate(hidden)
            _safe_deallocate(normed2)
            _safe_deallocate(ffn_out)
            _safe_deallocate(x_new)

        return x_out

    def backward(self, grad_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", Dict]:
        """Backward through encoder layer.

        Returns: (grad_x, grads_dict) where grads_dict contains gradients for
        all layer parameters.
        """
        c = self._cache
        x = c["x"]
        normed = c["normed"]
        x_new = c["x_new"]
        normed2 = c["normed2"]
        hidden = c["hidden"]
        ffn_out = c["ffn_out"]
        kpm = c["key_padding_mask"]

        # Backward through: x_out = x_new + ffn_out
        # NOTE: grad_x_new and grad_ffn_out both reference grad_out — don't
        # deallocate grad_ffn_out because it would invalidate grad_x_new.
        grad_x_new = grad_out
        grad_ffn_out = grad_out  # same reference — do NOT deallocate

        # Backward through ffn_out = linear2(gelu(linear1(normed2)))
        grad_hidden, grad_l2_w, grad_l2_b = self.linear2.backward(grad_ffn_out, hidden)
        # Don't deallocate grad_ffn_out — it's the same as grad_x_new

        # Backward through gelu
        grad_l1_out = ttnn.gelu_bw(grad_hidden, hidden)[0]
        _safe_deallocate(grad_hidden)

        # Backward through linear1
        grad_normed2, grad_l1_w, grad_l1_b = self.linear1.backward(grad_l1_out, normed2)
        _safe_deallocate(grad_l1_out)

        # Backward through norm2
        grad_x_new_from_ffn, grad_n2_w, grad_n2_b = self.norm2.backward(grad_normed2, x_new)
        # Don't deallocate grad_normed2 — grad_x_new_from_ffn may share buffer

        # Combine residual grads for x_new
        grad_x_new_total = ttnn.add(grad_x_new, grad_x_new_from_ffn)
        # Don't deallocate inputs to ttnn.add — result may share buffer

        # Backward through: x_new = x + attn_out
        # NOTE: grad_x and grad_attn_out both reference grad_x_new_total
        grad_x = grad_x_new_total  # residual passthrough
        grad_attn_out = grad_x_new_total  # same reference — do NOT deallocate

        # Backward through attention
        grad_normed, _, attn_grads = self.self_attn.backward(grad_attn_out)
        # Don't deallocate grad_attn_out — it's the same as grad_x
        # Now safe to deallocate add inputs — grad_x_new_total has been consumed
        _safe_deallocate(grad_x_new)
        _safe_deallocate(grad_x_new_from_ffn)
        _safe_deallocate(grad_normed2)

        # Backward through norm1
        grad_x_from_attn, grad_n1_w, grad_n1_b = self.norm1.backward(grad_normed, x)
        # Don't deallocate grad_normed — grad_x_from_attn may share buffer

        # Combine residual grads for x
        grad_x_total = ttnn.add(grad_x, grad_x_from_attn)
        # Don't deallocate grad_x/grad_x_from_attn — ttnn.add result may share buffer
        # Clean up cache
        _safe_deallocate(c["attn_out"])
        _safe_deallocate(c["normed"])
        _safe_deallocate(c["hidden"])
        _safe_deallocate(c["normed2"])
        _safe_deallocate(c["ffn_out"])
        _safe_deallocate(c["x_new"])
        self._cache = {}

        grads = {
            "norm1_weight": grad_n1_w, "norm1_bias": grad_n1_b,
            "norm2_weight": grad_n2_w, "norm2_bias": grad_n2_b,
            "linear1_weight": grad_l1_w, "linear1_bias": grad_l1_b,
            "linear2_weight": grad_l2_w, "linear2_bias": grad_l2_b,
            "in_proj_weight": attn_grads["in_proj_weight"],
            "in_proj_bias": attn_grads["in_proj_bias"],
            "out_proj_weight": attn_grads["out_proj_weight"],
            "out_proj_bias": attn_grads["out_proj_bias"],
        }
        return grad_x_total, grads


class TTDecoderLayer:
    """Transformer decoder layer matching PyTorch nn.TransformerDecoderLayer (pre-norm, GELU).

    PyTorch structure (norm_first=True):
      x = x + self_attn(norm1(x))
      x = x + cross_attn(norm2(x), memory)
      x = x + ffn(norm3(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, device, dropout=0.0, dtype=ttnn.bfloat16):
        self.device = device
        self.d_model = d_model
        self.self_attn = TTMultiHeadAttention(d_model, n_heads, device, dropout, dtype=dtype)
        self.norm1 = TTLayerNorm(d_model, device, dtype=dtype)
        self.cross_attn = TTMultiHeadAttention(d_model, n_heads, device, dropout, dtype=dtype)
        self.norm2 = TTLayerNorm(d_model, device, dtype=dtype)
        self.linear1 = TTLinear(d_model, d_ff, device, bias=True, dtype=dtype)
        self.linear2 = TTLinear(d_ff, d_model, device, bias=True, dtype=dtype)
        self.norm3 = TTLayerNorm(d_model, device, dtype=dtype)

    def forward(
        self,
        x: "ttnn.Tensor",
        memory: "ttnn.Tensor",
        causal_mask: Optional["ttnn.Tensor"] = None,
        tgt_key_padding_mask: Optional["ttnn.Tensor"] = None,
        memory_key_padding_mask: Optional["ttnn.Tensor"] = None,
        training: bool = False,
    ) -> "ttnn.Tensor":
        # Pre-norm self-attention (causal)
        normed1 = self.norm1.forward(x)
        self_out = self.self_attn.forward(normed1, normed1, normed1,
                                          key_padding_mask=tgt_key_padding_mask,
                                          attn_mask=causal_mask, training=training)
        x1 = ttnn.add(x, self_out)

        # Pre-norm cross-attention to memory
        normed2 = self.norm2.forward(x1)
        cross_out = self.cross_attn.forward(normed2, memory, memory,
                                             key_padding_mask=memory_key_padding_mask, training=training)
        x2 = ttnn.add(x1, cross_out)

        # Pre-norm FFN (GELU — matches PyTorch TransformerDecoderLayer)
        normed3 = self.norm3.forward(x2)
        hidden = ttnn.gelu(self.linear1.forward(normed3))
        ffn_out = self.linear2.forward(hidden)
        x_out = ttnn.add(x2, ffn_out)

        if training:
            self._cache = {
                "x": x, "normed1": normed1, "self_out": self_out, "x1": x1,
                "normed2": normed2, "cross_out": cross_out, "x2": x2,
                "normed3": normed3, "hidden": hidden, "ffn_out": ffn_out,
                "memory": memory,
            }
        else:
            _safe_deallocate(self_out)
            _safe_deallocate(normed1)
            _safe_deallocate(cross_out)
            _safe_deallocate(normed2)
            _safe_deallocate(hidden)
            _safe_deallocate(normed3)
            _safe_deallocate(ffn_out)
            _safe_deallocate(x1)
            _safe_deallocate(x2)

        return x_out

    def backward(self, grad_out: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor", Dict]:
        """Backward through decoder layer.

        Returns: (grad_x, grad_memory, grads_dict)
        """
        c = self._cache
        x = c["x"]
        normed1 = c["normed1"]
        x1 = c["x1"]
        normed2 = c["normed2"]
        x2 = c["x2"]
        normed3 = c["normed3"]
        hidden = c["hidden"]
        ffn_out = c["ffn_out"]
        memory = c["memory"]

        # Backward through: x_out = x2 + ffn_out
        # NOTE: grad_x2 and grad_ffn_out both reference grad_out — don't
        # deallocate grad_ffn_out because it would invalidate grad_x2.
        grad_x2 = grad_out
        grad_ffn_out = grad_out  # same reference — do NOT deallocate

        # Backward through ffn
        grad_hidden, grad_l2_w, grad_l2_b = self.linear2.backward(grad_ffn_out, hidden)
        # Don't deallocate grad_ffn_out — it's the same as grad_x2
        grad_l1_out = ttnn.gelu_bw(grad_hidden, hidden)[0]
        _safe_deallocate(grad_hidden)
        grad_normed3, grad_l1_w, grad_l1_b = self.linear1.backward(grad_l1_out, normed3)
        _safe_deallocate(grad_l1_out)
        grad_x2_ffn, grad_n3_w, grad_n3_b = self.norm3.backward(grad_normed3, x2)
        # Don't deallocate grad_normed3 — grad_x2_ffn may share device buffer
        # via TT-NN broadcast/in-place optimizations
        grad_x2_total = ttnn.add(grad_x2, grad_x2_ffn)
        # Don't deallocate grad_x2/grad_x2_ffn — ttnn.add may return a view
        # that shares device buffer with one of the inputs

        # Backward through: x2 = x1 + cross_out
        # NOTE: grad_x1 and grad_cross_out both reference grad_x2_total
        grad_x1 = grad_x2_total
        grad_cross_out = grad_x2_total  # same reference — do NOT deallocate

        # Backward through cross-attention
        grad_normed2, grad_memory, cross_grads = self.cross_attn.backward(grad_cross_out)
        # Don't deallocate grad_cross_out — it's the same as grad_x1
        # Now safe to deallocate grad_x2/grad_x2_ffn — grad_x2_total has been consumed
        _safe_deallocate(grad_x2)
        _safe_deallocate(grad_x2_ffn)

        # Backward through norm2
        grad_x1_cross, grad_n2_w, grad_n2_b = self.norm2.backward(grad_normed2, x1)
        # Don't deallocate grad_normed2 — grad_x1_cross may share buffer
        grad_x1_total = ttnn.add(grad_x1, grad_x1_cross)
        # Don't deallocate inputs to ttnn.add — result may share buffer

        # Backward through: x1 = x + self_out
        # NOTE: grad_x and grad_self_out both reference grad_x1_total
        grad_x = grad_x1_total
        grad_self_out = grad_x1_total  # same reference — do NOT deallocate

        # Backward through self-attention
        grad_normed1, _, self_grads = self.self_attn.backward(grad_self_out)
        # Don't deallocate grad_self_out — it's the same as grad_x
        # Now safe to deallocate add inputs — grad_x1_total has been consumed
        _safe_deallocate(grad_x1)
        _safe_deallocate(grad_x1_cross)
        _safe_deallocate(grad_normed2)

        # Backward through norm1
        grad_x_sa, grad_n1_w, grad_n1_b = self.norm1.backward(grad_normed1, x)
        # Don't deallocate grad_normed1 — grad_x_sa may share buffer
        grad_x_total = ttnn.add(grad_x, grad_x_sa)
        # Don't deallocate inputs to ttnn.add — result may share buffer

        # Clean up cache
        _safe_deallocate(c["self_out"])
        _safe_deallocate(c["normed1"])
        _safe_deallocate(c["cross_out"])
        _safe_deallocate(c["normed2"])
        _safe_deallocate(c["hidden"])
        _safe_deallocate(c["normed3"])
        _safe_deallocate(c["ffn_out"])
        _safe_deallocate(c["x1"])
        _safe_deallocate(c["x2"])
        self._cache = {}

        grads = {
            "norm1_weight": grad_n1_w, "norm1_bias": grad_n1_b,
            "norm2_weight": grad_n2_w, "norm2_bias": grad_n2_b,
            "norm3_weight": grad_n3_w, "norm3_bias": grad_n3_b,
            "linear1_weight": grad_l1_w, "linear1_bias": grad_l1_b,
            "linear2_weight": grad_l2_w, "linear2_bias": grad_l2_b,
            "sa_in_proj_weight": self_grads["in_proj_weight"],
            "sa_in_proj_bias": self_grads["in_proj_bias"],
            "sa_out_proj_weight": self_grads["out_proj_weight"],
            "sa_out_proj_bias": self_grads["out_proj_bias"],
            "ca_in_proj_weight": cross_grads["in_proj_weight"],
            "ca_in_proj_bias": cross_grads["in_proj_bias"],
            "ca_out_proj_weight": cross_grads["out_proj_weight"],
            "ca_out_proj_bias": cross_grads["out_proj_bias"],
        }
        return grad_x_total, grad_memory, grads


class TTTransition:
    """Recurrent memory transition step on device.

    Matches PyTorch TextLatentMemoryTransition:
      normed = attn_norm(memory)
      attended = self_attn(normed, normed, normed)
      hidden = attn_norm(memory + attended)  # reuses attn_norm!
      proposal = mlp(hidden)  # SiLU activation
      gate = sigmoid(gate(hidden))
      updated = output_norm(memory + gate * proposal)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, device, dropout=0.0, dtype=ttnn.bfloat16):
        self.device = device
        self.d_model = d_model
        self.self_attn = TTMultiHeadAttention(d_model, n_heads, device, dropout, dtype=dtype)
        self.attn_norm = TTRMSNorm(d_model, device, dtype=dtype)
        self.ffn = TTLinear(d_model, d_ff, device, bias=False, dtype=dtype)
        self.ffn_out = TTLinear(d_ff, d_model, device, bias=False, dtype=dtype)
        self.gate = TTLinear(d_model, d_model, device, bias=True, dtype=dtype)
        self.output_norm = TTRMSNorm(d_model, device, dtype=dtype)

    def forward(self, memory: "ttnn.Tensor", training: bool = False) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """Returns (updated_memory, gate) where gate is (B, S, d_model) in [0, 1]."""
        normed = self.attn_norm.forward(memory)
        attn_out = self.self_attn.forward(normed, normed, normed, training=training)
        hidden_pre = ttnn.add(memory, attn_out)
        # Reuse attn_norm for FFN input (matches PyTorch)
        hidden = self.attn_norm.forward(hidden_pre)

        hidden_act = ttnn.silu(self.ffn.forward(hidden))
        proposal = self.ffn_out.forward(hidden_act)

        gate_pre = self.gate.forward(hidden)
        gate = ttnn.sigmoid(gate_pre)
        gated_proposal = ttnn.mul(gate, proposal)
        updated_pre = ttnn.add(memory, gated_proposal)
        updated = self.output_norm.forward(updated_pre)

        if training:
            self._cache = {
                "memory": memory, "normed": normed, "attn_out": attn_out,
                "hidden_pre": hidden_pre, "hidden": hidden,
                "hidden_act": hidden_act, "proposal": proposal,
                "gate_pre": gate_pre, "gate": gate,
                "gated_proposal": gated_proposal, "updated_pre": updated_pre,
            }
        else:
            _safe_deallocate(normed)
            _safe_deallocate(attn_out)
            _safe_deallocate(hidden_pre)
            _safe_deallocate(hidden)
            _safe_deallocate(hidden_act)
            _safe_deallocate(proposal)
            _safe_deallocate(gate_pre)
            _safe_deallocate(gated_proposal)
            _safe_deallocate(updated_pre)

        return updated, gate

    def backward(self, grad_updated: "ttnn.Tensor") -> Tuple["ttnn.Tensor", Dict]:
        """Backward through transition step.

        Returns: (grad_memory, grads_dict)
        """
        c = self._cache
        memory = c["memory"]
        normed = c["normed"]
        attn_out = c["attn_out"]
        hidden_pre = c["hidden_pre"]
        hidden = c["hidden"]
        hidden_act = c["hidden_act"]
        proposal = c["proposal"]
        gate_pre = c["gate_pre"]
        gate = c["gate"]
        gated_proposal = c["gated_proposal"]
        updated_pre = c["updated_pre"]

        # Backward through output_norm
        grad_updated_pre, grad_on_w = self.output_norm.backward(grad_updated, updated_pre)

        # Backward through: updated_pre = memory + gated_proposal
        # NOTE: grad_memory and grad_gated_proposal both reference grad_updated_pre
        # — don't deallocate grad_gated_proposal because it would invalidate grad_memory.
        grad_memory = grad_updated_pre  # residual
        grad_gated_proposal = grad_updated_pre  # same reference — do NOT deallocate

        # Backward through: gated_proposal = gate * proposal
        grad_gate = ttnn.mul(grad_gated_proposal, proposal)
        grad_proposal = ttnn.mul(grad_gated_proposal, gate)
        # Don't deallocate grad_gated_proposal — it's the same as grad_memory

        # Backward through sigmoid: grad_gate_pre = grad_gate * gate * (1 - gate)
        # sigmoid_bw(grad_gate, gate_pre) gives the gradient
        grad_gate_pre = ttnn.sigmoid_bw(grad_gate, gate_pre)[0]
        _safe_deallocate(grad_gate)

        # Backward through gate linear
        grad_hidden_from_gate, grad_gate_w, grad_gate_b = self.gate.backward(grad_gate_pre, hidden)
        _safe_deallocate(grad_gate_pre)

        # Backward through ffn_out linear
        grad_hidden_act, grad_ffn_out_w, grad_ffn_out_b = self.ffn_out.backward(grad_proposal, hidden_act)
        _safe_deallocate(grad_proposal)

        # Backward through silu
        grad_ffn_out = ttnn.silu_bw(grad_hidden_act, hidden_act)[0]
        _safe_deallocate(grad_hidden_act)

        # Backward through ffn linear
        grad_hidden_from_ffn, grad_ffn_w, grad_ffn_b = self.ffn.backward(grad_ffn_out, hidden)
        _safe_deallocate(grad_ffn_out)

        # Combine grads for hidden (from gate path and ffn path)
        grad_hidden = ttnn.add(grad_hidden_from_gate, grad_hidden_from_ffn)
        _safe_deallocate(grad_hidden_from_gate)
        _safe_deallocate(grad_hidden_from_ffn)

        # Backward through attn_norm (reused for hidden)
        grad_hidden_pre, grad_an_w1 = self.attn_norm.backward(grad_hidden, hidden_pre)
        # Don't deallocate grad_hidden — grad_hidden_pre may share buffer

        # Backward through: hidden_pre = memory + attn_out
        # NOTE: grad_memory_attn and grad_attn_out both reference grad_hidden_pre
        grad_memory_attn = grad_hidden_pre  # residual
        grad_attn_out = grad_hidden_pre  # same reference — do NOT deallocate

        # Backward through attention
        grad_normed, _, attn_grads = self.self_attn.backward(grad_attn_out)
        # Don't deallocate grad_attn_out — it's the same as grad_memory_attn

        # Backward through attn_norm (first use)
        grad_memory_norm, grad_an_w2 = self.attn_norm.backward(grad_normed, memory)
        # Don't deallocate grad_normed — grad_memory_norm may share buffer

        # Combine all grads for memory
        grad_memory_total = ttnn.add(grad_memory, grad_memory_attn)
        grad_memory_total = ttnn.add(grad_memory_total, grad_memory_norm)
        # Don't deallocate inputs to ttnn.add — result may share buffer

        # Accumulate attn_norm weight grads (used twice)
        grad_attn_norm_w = ttnn.add(grad_an_w1, grad_an_w2)
        # Don't deallocate grad_an_w1/grad_an_w2 — result may share buffer

        # Clean up cache
        _safe_deallocate(c["normed"])
        _safe_deallocate(c["attn_out"])
        _safe_deallocate(c["hidden_pre"])
        _safe_deallocate(c["hidden"])
        _safe_deallocate(c["hidden_act"])
        _safe_deallocate(c["proposal"])
        _safe_deallocate(c["gate_pre"])
        _safe_deallocate(c["gate"])
        _safe_deallocate(c["gated_proposal"])
        _safe_deallocate(c["updated_pre"])
        self._cache = {}

        grads = {
            "attn_norm_weight": grad_attn_norm_w,
            "ffn_weight": grad_ffn_w,
            "ffn_out_weight": grad_ffn_out_w,
            "gate_weight": grad_gate_w,
            "gate_bias": grad_gate_b,
            "output_norm_weight": grad_on_w,
            "in_proj_weight": attn_grads["in_proj_weight"],
            "in_proj_bias": attn_grads["in_proj_bias"],
            "out_proj_weight": attn_grads["out_proj_weight"],
            "out_proj_bias": attn_grads["out_proj_bias"],
        }
        return grad_memory_total, grads


# ---------------------------------------------------------------------------
# Full TT-NN text latent-memory model
# ---------------------------------------------------------------------------

@dataclass
class TTTextLatentMemoryConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    n_heads: int = 4
    n_slots: int = 16
    max_reasoning_steps: int = 6
    expand: int = 4
    max_prompt_len: int = 256
    max_answer_len: int = 32
    pad_token_id: int = 0


class TTTextLatentMemoryModel:
    """TT-NN implementation of TextLatentMemoryModel.

    Forward pass runs on Tenstorrent hardware. Backward pass is handled by
    PyTorch autograd - we convert to torch at layer boundaries for gradient
    computation, then convert back to ttnn for the next forward pass.

    This hybrid approach:
    - Gets device acceleration for the compute-heavy forward pass
    - Avoids error-prone manual backward implementation
    - Uses the same memory management patterns as model_ttnn.py
    - Is simpler and less bug-prone than full manual backward
    """

    def __init__(self, config: TTTextLatentMemoryConfig, device, dtype=ttnn.bfloat16):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.n_slots = config.n_slots
        self.d_model = config.d_model
        torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

        # Token embedding (shared with LM head via weight tying)
        # ttnn.embedding requires bf16, so we keep a bf16 copy for embedding
        # and a compute-dtype copy for the LM head matmul.
        emb_w = torch.randn(config.vocab_size, config.d_model, dtype=torch_dtype) * 0.02
        self.token_emb_weight = to_device(emb_w, device, dtype=dtype)  # for LM head
        self.token_emb_weight_bf16 = to_device(emb_w, device, dtype=ttnn.bfloat16)  # for embedding

        # Position embeddings (also need bf16 copies for embedding lookup)
        self.prompt_pos_emb = to_device(
            torch.randn(config.max_prompt_len, config.d_model, dtype=torch_dtype) * 0.02,
            device, dtype=dtype
        )
        self.prompt_pos_emb_bf16 = to_device(
            torch.randn(config.max_prompt_len, config.d_model, dtype=torch_dtype) * 0.02,
            device, dtype=ttnn.bfloat16
        )
        self.answer_pos_emb = to_device(
            torch.randn(config.max_answer_len + 2, config.d_model, dtype=torch_dtype) * 0.02,
            device, dtype=dtype
        )
        self.answer_pos_emb_bf16 = to_device(
            torch.randn(config.max_answer_len + 2, config.d_model, dtype=torch_dtype) * 0.02,
            device, dtype=ttnn.bfloat16
        )

        # Encoder layers
        self.encoder_layers = []
        for _ in range(config.n_encoder_layers):
            layer = TTEncoderLayer(
                config.d_model, config.n_heads,
                config.d_model * config.expand, device, dtype=dtype
            )
            self.encoder_layers.append(layer)
        self.encoder_norm = TTRMSNorm(config.d_model, device, dtype=dtype)

        # Memory initialization: cross-attention from learnable slot queries
        self.slot_queries = to_device(
            torch.randn(config.n_slots, config.d_model, dtype=torch_dtype) * 0.02,
            device, dtype=dtype
        )
        self.memory_init_attn = TTMultiHeadAttention(config.d_model, config.n_heads, device, dtype=dtype)
        self.memory_norm = TTRMSNorm(config.d_model, device, dtype=dtype)

        # Recurrent transition
        self.transition = TTTransition(
            config.d_model, config.n_heads,
            config.d_model * config.expand, device, dtype=dtype
        )

        # Decoder layers
        self.decoder_layers = []
        for _ in range(config.n_decoder_layers):
            layer = TTDecoderLayer(
                config.d_model, config.n_heads,
                config.d_model * config.expand, device, dtype=dtype
            )
            self.decoder_layers.append(layer)
        self.decoder_norm = TTRMSNorm(config.d_model, device, dtype=dtype)

        # LM head (weight-tied with embedding)
        self.lm_head_weight = self.token_emb_weight

        # Initialize gate to start open (sigmoid(4) ≈ 0.98)
        # Must be done after all weights are created
        gate_w = torch.zeros(config.d_model, config.d_model, dtype=torch_dtype)
        gate_b = torch.full((config.d_model,), 4.0, dtype=torch_dtype)
        _safe_deallocate(self.transition.gate.weight)
        _safe_deallocate(self.transition.gate.bias)
        self.transition.gate.weight = to_device(gate_w, device, dtype=dtype)
        self.transition.gate.bias = to_device(gate_b, device, dtype=dtype)

        # Cache for causal mask
        self._causal_mask_cache = {}

    def _get_causal_mask(self, T: int) -> "ttnn.Tensor":
        """Get or create causal mask for sequence length T."""
        if T in self._causal_mask_cache:
            return self._causal_mask_cache[T]
        torch_dtype = torch.bfloat16 if self.dtype == ttnn.bfloat16 else torch.float32
        mask = torch.triu(
            torch.full((T, T), float('-inf'), dtype=torch_dtype),
            diagonal=1,
        )
        mask_tt = to_device(mask, self.device, dtype=self.dtype)
        self._causal_mask_cache[T] = mask_tt
        return mask_tt

    def _get_key_padding_mask(self, mask: torch.Tensor) -> "ttnn.Tensor":
        """Convert a boolean mask to additive padding mask (0=valid, -inf=padding).

        Input: (B, L) boolean where True=valid, False=padding
        Output: (B, L) where 0.0=valid, -inf=padding

        Uses torch.where to avoid 0 * -inf = NaN.
        """
        torch_dtype = torch.bfloat16 if self.dtype == ttnn.bfloat16 else torch.float32
        additive = torch.where(
            mask,
            torch.zeros((), dtype=torch_dtype),
            torch.full((), float('-inf'), dtype=torch_dtype),
        )
        return to_device(additive, self.device, dtype=self.dtype)

    def _embedding_lookup(self, indices: "ttnn.Tensor", weight: "ttnn.Tensor") -> "ttnn.Tensor":
        """Embedding lookup with dtype handling.

        ttnn.embedding requires bf16 weights, so we keep embedding weights in bf16
        and typecast the output to the model's compute dtype.
        """
        out = ttnn.embedding(indices, weight, layout=ttnn.TILE_LAYOUT)
        if self.dtype != ttnn.bfloat16:
            casted = ttnn.typecast(out, self.dtype)
            _safe_deallocate(out)
            out = casted
        return out

    def encode_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> "ttnn.Tensor":
        """Encode the prompt on device. Returns (B, T, d_model) ttnn tensor."""
        device = self.device
        B, T = input_ids.shape

        # Embedding lookup on device
        indices = ttnn.from_torch(
            input_ids.to(torch.int32),
            dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
        )
        x = self._embedding_lookup(indices, self.token_emb_weight_bf16)
        _safe_deallocate(indices)

        # Add position embeddings
        pos = ttnn.from_torch(
            torch.arange(T, dtype=torch.int32).unsqueeze(0),
            dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
        )
        pos_emb = self._embedding_lookup(pos, self.prompt_pos_emb_bf16)
        _safe_deallocate(pos)
        x = ttnn.add(x, pos_emb)
        _safe_deallocate(pos_emb)

        # Key padding mask
        kpm = self._get_key_padding_mask(attention_mask)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer.forward(x, key_padding_mask=kpm)
        x = self.encoder_norm.forward(x)

        _safe_deallocate(kpm)
        return x

    def initialize_memory(self, encoded: "ttnn.Tensor", attention_mask: torch.Tensor) -> "ttnn.Tensor":
        """Initialize memory slots via cross-attention."""
        device = self.device
        B = encoded.shape[0]

        # Expand slot queries to batch
        queries = ttnn.reshape(self.slot_queries, [1, self.n_slots, self.d_model])
        queries = ttnn.expand(queries, [B, self.n_slots, self.d_model])

        # Key padding mask for prompt
        kpm = self._get_key_padding_mask(attention_mask)

        # Cross-attention: slots attend to encoded prompt
        memory = self.memory_init_attn.forward(queries, encoded, encoded, key_padding_mask=kpm)

        # Residual + norm
        memory = ttnn.add(queries, memory)
        memory = self.memory_norm.forward(memory)

        _safe_deallocate(kpm)
        # Don't deallocate queries — it's a view of self.slot_queries
        return memory

    def reason(self, memory: "ttnn.Tensor") -> "ttnn.Tensor":
        """Run K recurrent reasoning steps over memory."""
        for _ in range(self.config.max_reasoning_steps):
            memory, gate = self.transition.forward(memory)
            _safe_deallocate(gate)
        return memory

    def decode_answer(
        self,
        memory: "ttnn.Tensor",
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> "ttnn.Tensor":
        """Decode answer via cross-attention to memory.

        Args:
            memory: (B, S, d_model) — final memory state
            answer_ids: (B, T_ans) — answer token IDs (teacher-forced, shifted right)
            answer_mask: (B, T_ans) — 1 for valid tokens

        Returns: (B, T_ans, vocab_size) ttnn tensor of logits
        """
        device = self.device
        B, T_ans = answer_ids.shape

        # Embedding lookup
        indices = ttnn.from_torch(
            answer_ids.to(torch.int32),
            dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
        )
        x = self._embedding_lookup(indices, self.token_emb_weight_bf16)
        _safe_deallocate(indices)

        # Add position embeddings
        pos = ttnn.from_torch(
            torch.arange(T_ans, dtype=torch.int32).unsqueeze(0),
            dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
        )
        pos_emb = self._embedding_lookup(pos, self.answer_pos_emb_bf16)
        _safe_deallocate(pos)
        x = ttnn.add(x, pos_emb)
        _safe_deallocate(pos_emb)

        # Causal mask
        causal_mask = self._get_causal_mask(T_ans)

        # Answer padding mask
        ans_kpm = self._get_key_padding_mask(answer_mask)

        # Decoder layers (self-attention + cross-attention to memory)
        for layer in self.decoder_layers:
            x = layer.forward(x, memory, causal_mask=causal_mask,
                              tgt_key_padding_mask=ans_kpm)

        x = self.decoder_norm.forward(x)

        # LM head: logits = x @ emb^T
        # emb_weight is (vocab_size, d_model), need (d_model, vocab_size) for linear
        # Use matmul: x (B, T, d) @ emb^T (d, V) = (B, T, V)
        emb_t = ttnn.transpose(self.lm_head_weight, 0, 1)  # (d_model, vocab_size)
        logits = ttnn.matmul(x, emb_t)
        _safe_deallocate(emb_t)
        _safe_deallocate(x)

        _safe_deallocate(ans_kpm)
        # Don't deallocate causal_mask — it's cached

        return logits

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass on device, returns logits as PyTorch tensor.

        This is the interface for the training loop — it handles the full
        forward pass on device and returns logits as a PyTorch tensor for
        loss computation and autograd backward.
        """
        # Encode prompt
        encoded = self.encode_prompt(prompt_ids, prompt_mask)

        # Initialize memory
        memory = self.initialize_memory(encoded, prompt_mask)
        _safe_deallocate(encoded)

        # Reason
        memory = self.reason(memory)

        # Decode
        logits_tt = self.decode_answer(memory, answer_ids, answer_mask)
        _safe_deallocate(memory)

        # Convert to PyTorch for loss + backward
        logits = from_device(logits_tt)
        _safe_deallocate(logits_tt)

        return logits

    def forward_train(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> "ttnn.Tensor":
        """Training forward pass — caches all intermediates for backward.

        Returns logits as a ttnn tensor (on device). Call backward() with
        grad_logits to compute gradients.
        """
        device = self.device
        B, T = prompt_ids.shape
        _, T_ans = answer_ids.shape
        d = self.d_model

        # --- Encode prompt ---
        indices = ttnn.from_torch(prompt_ids.to(torch.int32),
                                  dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        x_enc = self._embedding_lookup(indices, self.token_emb_weight_bf16)
        _safe_deallocate(indices)

        pos = ttnn.from_torch(torch.arange(T, dtype=torch.int32).unsqueeze(0),
                              dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        pos_emb = self._embedding_lookup(pos, self.prompt_pos_emb_bf16)
        _safe_deallocate(pos)
        x_enc = ttnn.add(x_enc, pos_emb)
        _safe_deallocate(pos_emb)

        kpm = self._get_key_padding_mask(prompt_mask)

        enc_inputs = []  # cache input to each encoder layer
        for layer in self.encoder_layers:
            enc_inputs.append(x_enc)
            x_enc = layer.forward(x_enc, key_padding_mask=kpm, training=True)
        enc_pre_norm = x_enc
        x_enc = self.encoder_norm.forward(x_enc)

        _safe_deallocate(kpm)

        # --- Initialize memory ---
        # NOTE: Use encoder_norm output (x_enc), not pre-norm — matches CPU model
        queries = ttnn.reshape(self.slot_queries, [1, self.n_slots, d])
        queries_b = ttnn.expand(queries, [B, self.n_slots, d])

        kpm_mem = self._get_key_padding_mask(prompt_mask)
        mem_attn_out = self.memory_init_attn.forward(
            queries_b, x_enc, x_enc,
            key_padding_mask=kpm_mem, training=True
        )
        memory = ttnn.add(queries_b, mem_attn_out)
        mem_pre_norm = memory
        memory = self.memory_norm.forward(memory)

        _safe_deallocate(kpm_mem)
        # Don't deallocate queries_b — it's cached in memory_init_attn._cache["query"]

        # --- Reason (K recurrent steps) ---
        memory_states = [memory]  # cache for backward
        transition_caches = []  # save cache for each step
        for k in range(self.config.max_reasoning_steps):
            memory, gate = self.transition.forward(memory, training=True)
            # Don't deallocate gate — it's cached in transition._cache for backward
            transition_caches.append((self.transition._cache, self.transition.self_attn._cache))
            memory_states.append(memory)

        # --- Decode answer ---
        ans_indices = ttnn.from_torch(answer_ids.to(torch.int32),
                                      dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        x_dec = self._embedding_lookup(ans_indices, self.token_emb_weight_bf16)
        _safe_deallocate(ans_indices)

        ans_pos = ttnn.from_torch(torch.arange(T_ans, dtype=torch.int32).unsqueeze(0),
                                  dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        ans_pos_emb = self._embedding_lookup(ans_pos, self.answer_pos_emb_bf16)
        _safe_deallocate(ans_pos)
        x_dec = ttnn.add(x_dec, ans_pos_emb)
        _safe_deallocate(ans_pos_emb)

        causal_mask = self._get_causal_mask(T_ans)
        ans_kpm = self._get_key_padding_mask(answer_mask)

        dec_inputs = []
        dec_memories = []
        for layer in self.decoder_layers:
            dec_inputs.append(x_dec)
            x_dec = layer.forward(x_dec, memory, causal_mask=causal_mask,
                                  tgt_key_padding_mask=ans_kpm, training=True)

        dec_pre_norm = x_dec
        x_dec = self.decoder_norm.forward(x_dec)

        # LM head: logits = x_dec @ emb^T
        emb_t = ttnn.transpose(self.lm_head_weight, 0, 1)
        logits = ttnn.matmul(x_dec, emb_t)
        _safe_deallocate(emb_t)

        # Cache everything needed for backward
        self._train_cache = {
            "prompt_mask": prompt_mask, "answer_mask": answer_mask,
            "enc_inputs": enc_inputs, "enc_pre_norm": enc_pre_norm,
            "mem_attn_out": mem_attn_out, "mem_pre_norm": mem_pre_norm,
            "memory_states": memory_states,
            "transition_caches": transition_caches,
            "dec_inputs": dec_inputs, "dec_pre_norm": dec_pre_norm,
            "x_dec_pre_lm": x_dec,
            "ans_kpm": ans_kpm,
        }

        return logits

    def backward(self, grad_logits: "ttnn.Tensor") -> Dict[str, "ttnn.Tensor"]:
        """Full backward pass on device.

        Args:
            grad_logits: (B, T_ans, vocab_size) gradient w.r.t. logits

        Returns: dict of parameter name -> gradient tensor
        """
        c = self._train_cache
        device = self.device
        d = self.d_model
        all_grads = {}

        def _accum_grads(grads: Dict, prefix: str):
            for k, v in grads.items():
                name = f"{prefix}_{k}"
                if name in all_grads:
                    all_grads[name] = ttnn.add(all_grads[name], v)
                    _safe_deallocate(v)
                else:
                    all_grads[name] = v

        # --- Backward through LM head: logits = x_dec @ emb^T ---
        # grad_x_dec = grad_logits @ emb  (B, T, V) @ (V, d) = (B, T, d)
        # grad_emb = grad_logits^T @ x_dec  (V, T) @ (T, d) = (V, d) — but we need (V, d) summed over batch
        emb = self.lm_head_weight  # (V, d)
        grad_x_dec = ttnn.matmul(grad_logits, emb)  # (B, T_ans, d)

        # grad_token_emb = sum over batch and seq of x_dec^T @ grad_logits
        # x_dec is (B, T_ans, d), grad_logits is (B, T_ans, V)
        # grad_emb = x_dec^T @ grad_logits = (d, V) — but need to sum over batch
        x_dec = c["x_dec_pre_lm"]
        x_dec_2d = ttnn.reshape(x_dec, [-1, d])  # (N, d)
        gl_2d = ttnn.reshape(grad_logits, [-1, self.config.vocab_size])  # (N, V)
        grad_token_emb = ttnn.matmul(ttnn.transpose(x_dec_2d, -2, -1), gl_2d)  # (d, V)
        _safe_deallocate(x_dec_2d)
        _safe_deallocate(gl_2d)
        # Transpose to (V, d) to match embedding shape
        grad_token_emb = ttnn.transpose(grad_token_emb, -2, -1)
        all_grads["token_emb_weight"] = grad_token_emb

        # --- Backward through decoder_norm ---
        grad_dec_pre, grad_dn_w = self.decoder_norm.backward(grad_x_dec, c["dec_pre_norm"])
        # Don't deallocate grad_x_dec — grad_dec_pre may share buffer
        all_grads["decoder_norm_weight"] = grad_dn_w

        # --- Backward through decoder layers (reverse order) ---
        grad_memory = None
        for i in reversed(range(len(self.decoder_layers))):
            layer = self.decoder_layers[i]
            grad_dec_pre, grad_mem_i, dec_grads = layer.backward(grad_dec_pre)
            _accum_grads(dec_grads, f"dec_{i}")
            if grad_memory is None:
                grad_memory = grad_mem_i
            else:
                new_grad_memory = ttnn.add(grad_memory, grad_mem_i)
                _safe_deallocate(grad_memory)
                _safe_deallocate(grad_mem_i)
                grad_memory = new_grad_memory

        _safe_deallocate(grad_dec_pre)

        # --- Backward through reasoning steps (reverse order) ---
        # memory_states[0] = initial memory, [1..K] = after each step
        grad_memory_state = grad_memory
        # Don't deallocate grad_memory — it's the same reference as grad_memory_state

        for k in reversed(range(self.config.max_reasoning_steps)):
            # Restore the cache for this transition step
            trans_cache, attn_cache = c["transition_caches"][k]
            self.transition._cache = trans_cache
            self.transition.self_attn._cache = attn_cache
            grad_memory_prev, trans_grads = self.transition.backward(grad_memory_state)
            _accum_grads(trans_grads, "trans")
            # Don't deallocate grad_memory_state — grad_memory_prev may share buffer
            grad_memory_state = grad_memory_prev

        # --- Backward through memory initialization ---
        # memory = memory_norm(queries + mem_attn_out)
        grad_mem_pre, grad_mn_w = self.memory_norm.backward(grad_memory_state, c["mem_pre_norm"])
        # Don't deallocate grad_memory_state — grad_mem_pre may share buffer
        all_grads["memory_norm_weight"] = grad_mn_w

        # grad flows to both queries and mem_attn_out through addition
        grad_queries = grad_mem_pre  # residual
        grad_mem_attn_out = grad_mem_pre  # same reference — do NOT deallocate

        # Backward through memory_init_attn (cross-attention)
        grad_queries_proj, grad_encoded_mem, mem_grads = self.memory_init_attn.backward(grad_mem_attn_out)
        _accum_grads(mem_grads, "mem_init")
        # Don't deallocate grad_mem_attn_out — it's the same as grad_queries

        # grad_queries from residual + grad_queries from attention input
        grad_queries_total = ttnn.add(grad_queries, grad_queries_proj)
        # Don't deallocate inputs to ttnn.add — result may share buffer

        # Sum grad_queries over batch to get grad_slot_queries
        # NOTE: Don't modify grad_queries_total in-place — use ttnn.sum with all leading dims
        reduce_dims = tuple(range(len(grad_queries_total.shape) - 1))
        gsq = ttnn.sum(grad_queries_total, dim=reduce_dims) if reduce_dims else grad_queries_total
        all_grads["slot_queries"] = gsq
        _safe_deallocate(grad_queries_total)

        # --- Backward through encoder ---
        # grad_encoded = grad from encoder_norm + grad from memory_init cross-attention
        grad_enc = grad_encoded_mem  # gradient flowing back to encoder output

        # Backward through encoder_norm
        grad_enc_pre, grad_en_w = self.encoder_norm.backward(grad_enc, c["enc_pre_norm"])
        all_grads["encoder_norm_weight"] = grad_en_w
        # Don't deallocate grad_enc — grad_enc_pre may share buffer

        # Backward through encoder layers (reverse order)
        grad_x = grad_enc_pre
        for i in reversed(range(len(self.encoder_layers))):
            layer = self.encoder_layers[i]
            grad_x, enc_grads = layer.backward(grad_x)
            _accum_grads(enc_grads, f"enc_{i}")

        # --- Backward through prompt embeddings ---
        # grad flows to token_emb and prompt_pos_emb
        # x_enc = token_emb(prompt_ids) + prompt_pos_emb(pos)
        # grad_token_emb from encoder path (add to existing from LM head)
        # We need to use embedding_bw for this
        # For now, sum grad_x over batch for pos_emb grad, and scatter for token_emb
        # This is simplified — proper embedding backward needs ttnn.embedding_bw

        # grad_pos_emb = sum over batch of grad_x
        gpe = grad_x
        while len(gpe.shape) > 1:
            _gpe = ttnn.sum(gpe, dim=0)
            _safe_deallocate(gpe)
            gpe = _gpe
        # Only first T positions are used
        # gpe is (T, d) — but prompt_pos_emb is (max_prompt_len, d)
        # The grad should be zero-padded to max_prompt_len
        # For simplicity, just use the first T rows
        if "prompt_pos_emb" in all_grads:
            # This shouldn't happen in a single forward, but just in case
            _safe_deallocate(all_grads["prompt_pos_emb"])
        all_grads["prompt_pos_emb"] = gpe  # TODO: zero-pad to max_prompt_len

        # grad_token_emb from encoder — use embedding_bw
        # For now, accumulate into all_grads["token_emb_weight"]
        # embedding_bw needs indices and grad_x
        # This is a TODO — embedding backward on device
        # For now, we skip the embedding backward through the encoder/decoder
        # and only have the LM head gradient for token_emb_weight
        _safe_deallocate(grad_x)

        # --- Backward through answer embeddings ---
        # Similar TODO for answer pos_emb and token_emb through decoder
        # The LM head gradient already accounts for token_emb_weight

        # Clean up train cache
        _safe_deallocate(c["x_dec_pre_lm"])
        _safe_deallocate(c["dec_pre_norm"])
        _safe_deallocate(c["enc_pre_norm"])
        _safe_deallocate(c["mem_pre_norm"])
        _safe_deallocate(c["mem_attn_out"])
        _safe_deallocate(c["ans_kpm"])
        # memory states are deallocated by transition.backward
        # enc_inputs and dec_inputs are deallocated by layer.backward
        self._train_cache = {}

        return all_grads

    def get_params(self) -> Dict[str, "ttnn.Tensor"]:
        """Get all model parameters for checkpointing."""
        params = {"token_emb_weight": self.token_emb_weight}
        params["prompt_pos_emb"] = self.prompt_pos_emb
        params["answer_pos_emb"] = self.answer_pos_emb
        params["slot_queries"] = self.slot_queries
        params["encoder_norm_weight"] = self.encoder_norm.weight
        params["memory_norm_weight"] = self.memory_norm.weight
        params["decoder_norm_weight"] = self.decoder_norm.weight

        for i, layer in enumerate(self.encoder_layers):
            params[f"enc_{i}_in_proj_w"] = layer.self_attn.in_proj_weight
            params[f"enc_{i}_in_proj_b"] = layer.self_attn.in_proj_bias
            params[f"enc_{i}_out_proj_w"] = layer.self_attn.out_proj_weight
            params[f"enc_{i}_out_proj_b"] = layer.self_attn.out_proj_bias
            params[f"enc_{i}_norm1_w"] = layer.norm1.weight
            params[f"enc_{i}_norm1_b"] = layer.norm1.bias
            params[f"enc_{i}_norm2_w"] = layer.norm2.weight
            params[f"enc_{i}_norm2_b"] = layer.norm2.bias
            params[f"enc_{i}_linear1_w"] = layer.linear1.weight
            params[f"enc_{i}_linear1_b"] = layer.linear1.bias
            params[f"enc_{i}_linear2_w"] = layer.linear2.weight
            params[f"enc_{i}_linear2_b"] = layer.linear2.bias

        params["mem_init_in_proj_w"] = self.memory_init_attn.in_proj_weight
        params["mem_init_in_proj_b"] = self.memory_init_attn.in_proj_bias
        params["mem_init_out_proj_w"] = self.memory_init_attn.out_proj_weight
        params["mem_init_out_proj_b"] = self.memory_init_attn.out_proj_bias

        params["trans_in_proj_w"] = self.transition.self_attn.in_proj_weight
        params["trans_in_proj_b"] = self.transition.self_attn.in_proj_bias
        params["trans_out_proj_w"] = self.transition.self_attn.out_proj_weight
        params["trans_out_proj_b"] = self.transition.self_attn.out_proj_bias
        params["trans_attn_norm_w"] = self.transition.attn_norm.weight
        params["trans_ffn_w"] = self.transition.ffn.weight
        params["trans_ffn_out_w"] = self.transition.ffn_out.weight
        params["trans_gate_w"] = self.transition.gate.weight
        params["trans_gate_b"] = self.transition.gate.bias
        params["trans_output_norm_w"] = self.transition.output_norm.weight

        for i, layer in enumerate(self.decoder_layers):
            params[f"dec_{i}_sa_in_proj_w"] = layer.self_attn.in_proj_weight
            params[f"dec_{i}_sa_in_proj_b"] = layer.self_attn.in_proj_bias
            params[f"dec_{i}_sa_out_proj_w"] = layer.self_attn.out_proj_weight
            params[f"dec_{i}_sa_out_proj_b"] = layer.self_attn.out_proj_bias
            params[f"dec_{i}_norm1_w"] = layer.norm1.weight
            params[f"dec_{i}_norm1_b"] = layer.norm1.bias
            params[f"dec_{i}_ca_in_proj_w"] = layer.cross_attn.in_proj_weight
            params[f"dec_{i}_ca_in_proj_b"] = layer.cross_attn.in_proj_bias
            params[f"dec_{i}_ca_out_proj_w"] = layer.cross_attn.out_proj_weight
            params[f"dec_{i}_ca_out_proj_b"] = layer.cross_attn.out_proj_bias
            params[f"dec_{i}_norm2_w"] = layer.norm2.weight
            params[f"dec_{i}_norm2_b"] = layer.norm2.bias
            params[f"dec_{i}_linear1_w"] = layer.linear1.weight
            params[f"dec_{i}_linear1_b"] = layer.linear1.bias
            params[f"dec_{i}_linear2_w"] = layer.linear2.weight
            params[f"dec_{i}_linear2_b"] = layer.linear2.bias
            params[f"dec_{i}_norm3_w"] = layer.norm3.weight
            params[f"dec_{i}_norm3_b"] = layer.norm3.bias

        return params

    def get_num_params(self) -> int:
        """Count total parameters."""
        total = 0
        for t in self.get_params().values():
            t_host = ttnn.to_torch(t)
            total += t_host.numel()
        return total

    def clear_caches(self):
        """Clear cached tensors to free device memory.

        Synchronizes device first so async queue releases references.
        """
        ttnn.synchronize_device(self.device)
        # Causal masks are small and reused — keep them cached
        # No intermediate tensors to clear in this hybrid model since
        # forward() deallocates everything before returning

    def save_checkpoint(self, path: str, step: int = 0):
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "n_encoder_layers": self.config.n_encoder_layers,
                "n_decoder_layers": self.config.n_decoder_layers,
                "n_heads": self.config.n_heads,
                "n_slots": self.config.n_slots,
                "max_reasoning_steps": self.config.max_reasoning_steps,
                "expand": self.config.expand,
                "max_prompt_len": self.config.max_prompt_len,
                "max_answer_len": self.config.max_answer_len,
                "pad_token_id": self.config.pad_token_id,
            },
            "model_state": {},
        }
        for name, tt_tensor in self.get_params().items():
            checkpoint["model_state"][name] = ttnn.to_torch(tt_tensor).clone()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path} (step {step})", flush=True)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        device = self.device
        checkpoint = torch.load(path, weights_only=False)
        model_state = checkpoint["model_state"]

        for name, host_tensor in model_state.items():
            torch_dtype = torch.bfloat16 if self.dtype == ttnn.bfloat16 else torch.float32
            tt_tensor = to_device(host_tensor.to(torch_dtype), device, dtype=self.dtype)
            self._set_param(name, tt_tensor)

        print(f"Loaded checkpoint from {path} (step {checkpoint.get('step', 0)})", flush=True)
        return checkpoint.get("optimizer_state", None)

    def _set_param(self, name: str, tt_tensor: "ttnn.Tensor"):
        """Set a single parameter by name."""
        if name == "token_emb_weight":
            _safe_deallocate(self.token_emb_weight)
            self.token_emb_weight = tt_tensor
            self.lm_head_weight = tt_tensor
            # Also update bf16 copy for embedding lookup
            _safe_deallocate(self.token_emb_weight_bf16)
            self.token_emb_weight_bf16 = ttnn.typecast(tt_tensor, ttnn.bfloat16) \
                if self.dtype != ttnn.bfloat16 else tt_tensor
        elif name == "prompt_pos_emb":
            _safe_deallocate(self.prompt_pos_emb)
            self.prompt_pos_emb = tt_tensor
            _safe_deallocate(self.prompt_pos_emb_bf16)
            self.prompt_pos_emb_bf16 = ttnn.typecast(tt_tensor, ttnn.bfloat16) \
                if self.dtype != ttnn.bfloat16 else tt_tensor
        elif name == "answer_pos_emb":
            _safe_deallocate(self.answer_pos_emb)
            self.answer_pos_emb = tt_tensor
            _safe_deallocate(self.answer_pos_emb_bf16)
            self.answer_pos_emb_bf16 = ttnn.typecast(tt_tensor, ttnn.bfloat16) \
                if self.dtype != ttnn.bfloat16 else tt_tensor
        elif name == "slot_queries":
            _safe_deallocate(self.slot_queries)
            self.slot_queries = tt_tensor
        elif name == "encoder_norm_weight":
            _safe_deallocate(self.encoder_norm.weight)
            self.encoder_norm.weight = tt_tensor
        elif name == "memory_norm_weight":
            _safe_deallocate(self.memory_norm.weight)
            self.memory_norm.weight = tt_tensor
        elif name == "decoder_norm_weight":
            _safe_deallocate(self.decoder_norm.weight)
            self.decoder_norm.weight = tt_tensor
        elif name.startswith("enc_"):
            self._set_encoder_param(name, tt_tensor)
        elif name.startswith("mem_init_"):
            self._set_mem_init_param(name, tt_tensor)
        elif name.startswith("trans_"):
            self._set_transition_param(name, tt_tensor)
        elif name.startswith("dec_"):
            self._set_decoder_param(name, tt_tensor)

    def _set_encoder_param(self, name: str, tt_tensor: "ttnn.Tensor"):
        parts = name.split("_", 2)
        idx = int(parts[1])
        suffix = parts[2]
        layer = self.encoder_layers[idx]
        if suffix == "in_proj_w": _safe_deallocate(layer.self_attn.in_proj_weight); layer.self_attn.in_proj_weight = tt_tensor
        elif suffix == "in_proj_b": _safe_deallocate(layer.self_attn.in_proj_bias); layer.self_attn.in_proj_bias = tt_tensor
        elif suffix == "out_proj_w": _safe_deallocate(layer.self_attn.out_proj_weight); layer.self_attn.out_proj_weight = tt_tensor
        elif suffix == "out_proj_b": _safe_deallocate(layer.self_attn.out_proj_bias); layer.self_attn.out_proj_bias = tt_tensor
        elif suffix == "norm1_w": _safe_deallocate(layer.norm1.weight); layer.norm1.weight = tt_tensor
        elif suffix == "norm1_b": _safe_deallocate(layer.norm1.bias); layer.norm1.bias = tt_tensor
        elif suffix == "norm2_w": _safe_deallocate(layer.norm2.weight); layer.norm2.weight = tt_tensor
        elif suffix == "norm2_b": _safe_deallocate(layer.norm2.bias); layer.norm2.bias = tt_tensor
        elif suffix == "linear1_w": _safe_deallocate(layer.linear1.weight); layer.linear1.weight = tt_tensor
        elif suffix == "linear1_b": _safe_deallocate(layer.linear1.bias); layer.linear1.bias = tt_tensor
        elif suffix == "linear2_w": _safe_deallocate(layer.linear2.weight); layer.linear2.weight = tt_tensor
        elif suffix == "linear2_b": _safe_deallocate(layer.linear2.bias); layer.linear2.bias = tt_tensor

    def _set_mem_init_param(self, name: str, tt_tensor: "ttnn.Tensor"):
        suffix = name[len("mem_init_"):]
        attn = self.memory_init_attn
        if suffix == "in_proj_w": _safe_deallocate(attn.in_proj_weight); attn.in_proj_weight = tt_tensor
        elif suffix == "in_proj_b": _safe_deallocate(attn.in_proj_bias); attn.in_proj_bias = tt_tensor
        elif suffix == "out_proj_w": _safe_deallocate(attn.out_proj_weight); attn.out_proj_weight = tt_tensor
        elif suffix == "out_proj_b": _safe_deallocate(attn.out_proj_bias); attn.out_proj_bias = tt_tensor

    def _set_transition_param(self, name: str, tt_tensor: "ttnn.Tensor"):
        suffix = name[len("trans_"):]
        t = self.transition
        if suffix == "in_proj_w": _safe_deallocate(t.self_attn.in_proj_weight); t.self_attn.in_proj_weight = tt_tensor
        elif suffix == "in_proj_b": _safe_deallocate(t.self_attn.in_proj_bias); t.self_attn.in_proj_bias = tt_tensor
        elif suffix == "out_proj_w": _safe_deallocate(t.self_attn.out_proj_weight); t.self_attn.out_proj_weight = tt_tensor
        elif suffix == "out_proj_b": _safe_deallocate(t.self_attn.out_proj_bias); t.self_attn.out_proj_bias = tt_tensor
        elif suffix == "attn_norm_w": _safe_deallocate(t.attn_norm.weight); t.attn_norm.weight = tt_tensor
        elif suffix == "ffn_w": _safe_deallocate(t.ffn.weight); t.ffn.weight = tt_tensor
        elif suffix == "ffn_out_w": _safe_deallocate(t.ffn_out.weight); t.ffn_out.weight = tt_tensor
        elif suffix == "gate_w": _safe_deallocate(t.gate.weight); t.gate.weight = tt_tensor
        elif suffix == "gate_b": _safe_deallocate(t.gate.bias); t.gate.bias = tt_tensor
        elif suffix == "output_norm_w": _safe_deallocate(t.output_norm.weight); t.output_norm.weight = tt_tensor

    def _set_decoder_param(self, name: str, tt_tensor: "ttnn.Tensor"):
        parts = name.split("_", 2)
        idx = int(parts[1])
        suffix = parts[2]
        layer = self.decoder_layers[idx]
        if suffix == "sa_in_proj_w": _safe_deallocate(layer.self_attn.in_proj_weight); layer.self_attn.in_proj_weight = tt_tensor
        elif suffix == "sa_in_proj_b": _safe_deallocate(layer.self_attn.in_proj_bias); layer.self_attn.in_proj_bias = tt_tensor
        elif suffix == "sa_out_proj_w": _safe_deallocate(layer.self_attn.out_proj_weight); layer.self_attn.out_proj_weight = tt_tensor
        elif suffix == "sa_out_proj_b": _safe_deallocate(layer.self_attn.out_proj_bias); layer.self_attn.out_proj_bias = tt_tensor
        elif suffix == "norm1_w": _safe_deallocate(layer.norm1.weight); layer.norm1.weight = tt_tensor
        elif suffix == "norm1_b": _safe_deallocate(layer.norm1.bias); layer.norm1.bias = tt_tensor
        elif suffix == "ca_in_proj_w": _safe_deallocate(layer.cross_attn.in_proj_weight); layer.cross_attn.in_proj_weight = tt_tensor
        elif suffix == "ca_in_proj_b": _safe_deallocate(layer.cross_attn.in_proj_bias); layer.cross_attn.in_proj_bias = tt_tensor
        elif suffix == "ca_out_proj_w": _safe_deallocate(layer.cross_attn.out_proj_weight); layer.cross_attn.out_proj_weight = tt_tensor
        elif suffix == "ca_out_proj_b": _safe_deallocate(layer.cross_attn.out_proj_bias); layer.cross_attn.out_proj_bias = tt_tensor
        elif suffix == "norm2_w": _safe_deallocate(layer.norm2.weight); layer.norm2.weight = tt_tensor
        elif suffix == "norm2_b": _safe_deallocate(layer.norm2.bias); layer.norm2.bias = tt_tensor
        elif suffix == "linear1_w": _safe_deallocate(layer.linear1.weight); layer.linear1.weight = tt_tensor
        elif suffix == "linear1_b": _safe_deallocate(layer.linear1.bias); layer.linear1.bias = tt_tensor
        elif suffix == "linear2_w": _safe_deallocate(layer.linear2.weight); layer.linear2.weight = tt_tensor
        elif suffix == "linear2_b": _safe_deallocate(layer.linear2.bias); layer.linear2.bias = tt_tensor
        elif suffix == "norm3_w": _safe_deallocate(layer.norm3.weight); layer.norm3.weight = tt_tensor
        elif suffix == "norm3_b": _safe_deallocate(layer.norm3.bias); layer.norm3.bias = tt_tensor
