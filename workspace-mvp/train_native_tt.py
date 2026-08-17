#!/usr/bin/env python
"""Native TT training for the text latent-memory model.

All forward, backward, loss, and optimizer steps run on Tenstorrent hardware.
No PyTorch autograd is used for the backward pass.

Usage:
    TT_VISIBLE_DEVICES=0 TT_METAL_HOME=/home/rfenwick/Documents/tt-metal-src \
    /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_native_tt.py \
        --steps 50 --batch_size 4 --smoke_test
"""

import argparse
import ctypes
import math
import os
import random
import time
import sys

# Force glibc to return freed memory to the OS (prevents RSS growth from
# malloc arena expansion when large temporary tensors are allocated/freed)
_libc = ctypes.CDLL("libc.so.6")
def malloc_trim():
    _libc.malloc_trim(0)

# Set env before importing ttnn
os.environ.setdefault("TT_METAL_HOME", "/home/rfenwick/Documents/tt-metal-src")

# P300 fabric mesh graph descriptor setup (same as train_tt_text_latent_memory.py)
_P300_SUBSYSTEM_IDS = {"0x0044", "0x0045", "0x0046"}
def _is_p300():
    try:
        from pathlib import Path
        for entry in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
            sub = (entry / "device" / "subsystem_device").read_text().strip().lower()
            if sub in _P300_SUBSYSTEM_IDS:
                return True
    except Exception:
        pass
    return False
def _find_mesh_graph_descriptor():
    try:
        import importlib.util
        from pathlib import Path
        spec = importlib.util.find_spec("ttnn")
        for name in ["p150_mesh_graph_descriptor.textproto", "p300_mesh_graph_descriptor.textproto"]:
            if spec is not None and spec.submodule_search_locations:
                path = Path(next(iter(spec.submodule_search_locations))) / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if path.is_file():
                    return str(path)
            for p in sys.path:
                candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if candidate.is_file():
                    return str(candidate)
    except Exception:
        pass
    return None
if _is_p300():
    _mgd = _find_mesh_graph_descriptor()
    if _mgd:
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tt_text_latent_memory_model import (
    TTTextLatentMemoryConfig,
    TTTextLatentMemoryModel,
    to_device,
    from_device,
    _safe_deallocate,
)
from challenge_data import ChallengeDataset


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def create_device(device_id: int = 0):
    """Create TT device with proper mesh graph config."""
    device = ttnn.open_device(device_id=device_id)
    return device


# ---------------------------------------------------------------------------
# Cross-entropy loss on device
# ---------------------------------------------------------------------------

def cross_entropy_loss_and_grad(
    logits: "ttnn.Tensor",
    targets: torch.Tensor,
    mask: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
):
    """Compute cross-entropy loss and gradient on host, then move grad to device.

    For large vocabularies (V=50257), we compute the softmax on host in float32
    to avoid the V×V identity matrix issue on device.

    Args:
        logits: (B, T, V) ttnn tensor
        targets: (B, T) torch tensor of target token IDs
        mask: (B, T) torch tensor — 1 for valid tokens, 0 for padding

    Returns:
        loss: scalar (float) — mean cross-entropy over valid tokens
        grad_logits: (B, T, V) ttnn tensor — gradient w.r.t. logits
    """
    # Move logits to host and compute loss+grad in fp32
    logits_torch = ttnn.to_torch(logits)
    logits_f = logits_torch.float()
    del logits_torch  # free the bf16 copy immediately

    # Compute log_softmax (numerically stable)
    log_probs = torch.log_softmax(logits_f, dim=-1)  # (B, T, V)
    del logits_f

    # Gather log_probs at target positions for loss
    targets_long = targets.long()
    gathered = log_probs.gather(2, targets_long.unsqueeze(-1)).squeeze(-1)  # (B, T)
    per_token_loss = -gathered
    del gathered

    # Mask out padding
    mask_f = mask.float()
    n_valid = mask_f.sum().clamp_min(1)
    loss_val = (per_token_loss * mask_f).sum().item() / n_valid.item()
    del per_token_loss

    # Gradient: (probs - one_hot) / N
    # Compute in-place to minimize memory: reuse log_probs tensor
    probs = torch.exp(log_probs)  # (B, T, V)
    del log_probs

    # Subtract 1 at target positions (in-place)
    probs.scatter_(2, targets_long.unsqueeze(-1), probs.gather(2, targets_long.unsqueeze(-1)) - 1.0)
    grad = probs  # rename for clarity (probs is now the gradient before masking)
    grad = grad * mask_f.unsqueeze(-1) / n_valid
    del mask_f, n_valid, probs

    # Move gradient back to device
    grad_tt = to_device(grad, device, dtype=dtype)
    del grad

    return loss_val, grad_tt


# ---------------------------------------------------------------------------
# TTAdamW optimizer (adapted from workspace-poc/train_ttnn.py)
# ---------------------------------------------------------------------------

class TTAdamW:
    """AdamW optimizer for TT-NN tensors — fully on device, mixed precision."""

    def __init__(self, params: dict, lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1):
        self.base_lr = lr
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        self.param_names = list(params.keys())
        self.device = None
        self.param_dtype = {}
        self.exp_avg = {}
        self.exp_avg_sq = {}
        self.master = {}

        for name, tt_tensor in params.items():
            if self.device is None:
                self.device = tt_tensor.device()
            self.param_dtype[name] = tt_tensor.dtype
            shape = tuple(tt_tensor.shape)
            self.exp_avg[name] = ttnn.zeros(
                shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            self.exp_avg_sq[name] = ttnn.zeros(
                shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            new_master = ttnn.typecast(tt_tensor, ttnn.float32)
            if new_master is tt_tensor:
                new_master = ttnn.from_torch(
                    ttnn.to_torch(tt_tensor).clone(),
                    dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=tt_tensor.device()
                )
            self.master[name] = new_master

    def set_lr(self, lr: float):
        self.lr = lr

    def step(self, grads: dict, model: TTTextLatentMemoryModel):
        """Apply one optimizer step — fully on device."""
        self.step_count += 1
        device = self.device

        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.step_count
        bc2 = 1.0 - b2 ** self.step_count

        _b1_tt = ttnn.from_torch(torch.tensor([b1], dtype=torch.float32),
                                 dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _1mb1_tt = ttnn.from_torch(torch.tensor([1.0 - b1], dtype=torch.float32),
                                   dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _inv_bc1_tt = ttnn.from_torch(torch.tensor([1.0 / bc1], dtype=torch.float32),
                                      dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _inv_bc2_tt = ttnn.from_torch(torch.tensor([1.0 / bc2], dtype=torch.float32),
                                      dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _eps_tt = ttnn.from_torch(torch.tensor([self.eps], dtype=torch.float32),
                                  dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        for name in self.param_names:
            if name not in grads:
                continue

            grad = grads[name]
            storage_dtype = self.param_dtype[name]
            m_old = self.exp_avg[name]
            v_old = self.exp_avg_sq[name]
            param_old = self.master[name]

            # Upcast grad to fp32
            grad_fp32 = None
            if grad.dtype != ttnn.float32:
                grad_fp32 = ttnn.typecast(grad, ttnn.float32)
                grad_for_update = grad_fp32
            else:
                grad_for_update = grad

            lr = self.lr
            _b2_tt = ttnn.from_torch(torch.tensor([b2], dtype=torch.float32),
                                     dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            _1mb2_tt = ttnn.from_torch(torch.tensor([1.0 - b2], dtype=torch.float32),
                                       dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            _lr_tt = ttnn.from_torch(torch.tensor([lr], dtype=torch.float32),
                                     dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            _wd_scale_tt = ttnn.from_torch(
                torch.tensor([1.0 - lr * self.weight_decay], dtype=torch.float32),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
            )

            # m = b1*m + (1-b1)*g
            m_b1 = ttnn.mul(m_old, _b1_tt)
            g_1mb1 = ttnn.mul(grad_for_update, _1mb1_tt)
            m = ttnn.add(m_b1, g_1mb1)
            _safe_deallocate(m_b1)
            _safe_deallocate(g_1mb1)

            # v = b2*v + (1-b2)*g^2
            g_sq = ttnn.mul(grad_for_update, grad_for_update)
            v_b2 = ttnn.mul(v_old, _b2_tt)
            g_sq_1mb2 = ttnn.mul(g_sq, _1mb2_tt)
            v = ttnn.add(v_b2, g_sq_1mb2)
            _safe_deallocate(g_sq)
            _safe_deallocate(v_b2)
            _safe_deallocate(g_sq_1mb2)

            # bias correction
            m_hat = ttnn.mul(m, _inv_bc1_tt)
            v_hat = ttnn.mul(v, _inv_bc2_tt)

            # param -= lr * m_hat / (sqrt(v_hat) + eps)
            sqrt_v = ttnn.sqrt(v_hat)
            denom = ttnn.add(sqrt_v, _eps_tt)
            update = ttnn.div(m_hat, denom)
            _safe_deallocate(sqrt_v)
            _safe_deallocate(denom)
            _safe_deallocate(m_hat)
            _safe_deallocate(v_hat)

            update_scaled = ttnn.mul(update, _lr_tt)
            _safe_deallocate(update)
            param = ttnn.sub(param_old, update_scaled)
            _safe_deallocate(update_scaled)

            # weight decay
            param_wd = ttnn.mul(param, _wd_scale_tt)
            _safe_deallocate(param)
            param = param_wd

            # Deallocate per-param scalars and old state
            _safe_deallocate(_b2_tt)
            _safe_deallocate(_1mb2_tt)
            _safe_deallocate(_lr_tt)
            _safe_deallocate(_wd_scale_tt)
            _safe_deallocate(m_old)
            _safe_deallocate(v_old)
            _safe_deallocate(param_old)
            _safe_deallocate(grad_fp32)

            # Store updated state
            self.exp_avg[name] = m
            self.exp_avg_sq[name] = v
            self.master[name] = param

            # Downcast to model dtype and write into model
            if storage_dtype != ttnn.float32:
                param_model = ttnn.typecast(param, storage_dtype)
            else:
                param_model = ttnn.from_torch(
                    ttnn.to_torch(param).clone(),
                    dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=param.device()
                )
            self._set_model_param(model, name, param_model)

        # Deallocate step-level scalars
        _safe_deallocate(_b1_tt)
        _safe_deallocate(_1mb1_tt)
        _safe_deallocate(_inv_bc1_tt)
        _safe_deallocate(_inv_bc2_tt)
        _safe_deallocate(_eps_tt)

    def _set_model_param(self, model: TTTextLatentMemoryModel, name: str, param: "ttnn.Tensor"):
        """Update a model parameter from a device tensor."""
        if name == "token_emb_weight":
            _safe_deallocate(model.token_emb_weight)
            model.token_emb_weight = param
            model.lm_head_weight = param
            # Also update bf16 copy for embedding
            _safe_deallocate(model.token_emb_weight_bf16)
            model.token_emb_weight_bf16 = ttnn.typecast(param, ttnn.bfloat16)
        elif name == "prompt_pos_emb":
            _safe_deallocate(model.prompt_pos_emb)
            model.prompt_pos_emb = param
            _safe_deallocate(model.prompt_pos_emb_bf16)
            model.prompt_pos_emb_bf16 = ttnn.typecast(param, ttnn.bfloat16)
        elif name == "answer_pos_emb":
            _safe_deallocate(model.answer_pos_emb)
            model.answer_pos_emb = param
            _safe_deallocate(model.answer_pos_emb_bf16)
            model.answer_pos_emb_bf16 = ttnn.typecast(param, ttnn.bfloat16)
        elif name == "slot_queries":
            _safe_deallocate(model.slot_queries)
            model.slot_queries = param
        elif name == "encoder_norm_weight":
            _safe_deallocate(model.encoder_norm.weight)
            model.encoder_norm.weight = param
        elif name == "memory_norm_weight":
            _safe_deallocate(model.memory_norm.weight)
            model.memory_norm.weight = param
        elif name == "decoder_norm_weight":
            _safe_deallocate(model.decoder_norm.weight)
            model.decoder_norm.weight = param
        elif name.startswith("enc_"):
            self._set_encoder_param(model, name, param)
        elif name.startswith("dec_"):
            self._set_decoder_param(model, name, param)
        elif name.startswith("mem_init_"):
            self._set_mem_init_param(model, name, param)
        elif name.startswith("trans_"):
            self._set_transition_param(model, name, param)
        else:
            print(f"WARNING: unknown param name in optimizer: {name}", flush=True)

    def _set_encoder_param(self, model, name, param):
        # enc_{i}_{param}
        parts = name.split("_", 2)
        i = int(parts[1])
        layer = model.encoder_layers[i]
        suffix = parts[2]
        if suffix == "in_proj_w":
            _safe_deallocate(layer.self_attn.in_proj_weight)
            layer.self_attn.in_proj_weight = param
        elif suffix == "in_proj_b":
            _safe_deallocate(layer.self_attn.in_proj_bias)
            layer.self_attn.in_proj_bias = param
        elif suffix == "out_proj_w":
            _safe_deallocate(layer.self_attn.out_proj_weight)
            layer.self_attn.out_proj_weight = param
        elif suffix == "out_proj_b":
            _safe_deallocate(layer.self_attn.out_proj_bias)
            layer.self_attn.out_proj_bias = param
        elif suffix == "norm1_w":
            _safe_deallocate(layer.norm1.weight)
            layer.norm1.weight = param
        elif suffix == "norm1_b":
            _safe_deallocate(layer.norm1.bias)
            layer.norm1.bias = param
        elif suffix == "norm2_w":
            _safe_deallocate(layer.norm2.weight)
            layer.norm2.weight = param
        elif suffix == "norm2_b":
            _safe_deallocate(layer.norm2.bias)
            layer.norm2.bias = param
        elif suffix == "linear1_w":
            _safe_deallocate(layer.linear1.weight)
            layer.linear1.weight = param
        elif suffix == "linear1_b":
            _safe_deallocate(layer.linear1.bias)
            layer.linear1.bias = param
        elif suffix == "linear2_w":
            _safe_deallocate(layer.linear2.weight)
            layer.linear2.weight = param
        elif suffix == "linear2_b":
            _safe_deallocate(layer.linear2.bias)
            layer.linear2.bias = param
        else:
            print(f"WARNING: unknown encoder param: {name}", flush=True)

    def _set_decoder_param(self, model, name, param):
        # dec_{i}_{sa|ca|norm|linear}_{...}
        parts = name.split("_", 3)
        i = int(parts[1])
        layer = model.decoder_layers[i]
        prefix = parts[2]
        suffix = parts[3] if len(parts) > 3 else ""
        if prefix == "sa":
            self._set_attn_param(layer.self_attn, suffix, param)
        elif prefix == "ca":
            self._set_attn_param(layer.cross_attn, suffix, param)
        elif prefix == "norm1":
            self._set_norm_param(layer.norm1, suffix, param)
        elif prefix == "norm2":
            self._set_norm_param(layer.norm2, suffix, param)
        elif prefix == "norm3":
            self._set_norm_param(layer.norm3, suffix, param)
        elif prefix == "linear1":
            self._set_linear_param(layer.linear1, suffix, param)
        elif prefix == "linear2":
            self._set_linear_param(layer.linear2, suffix, param)
        else:
            print(f"WARNING: unknown decoder param: {name}", flush=True)

    def _set_mem_init_param(self, model, name, param):
        suffix = name[len("mem_init_"):]
        self._set_attn_param(model.memory_init_attn, suffix, param)

    def _set_transition_param(self, model, name, param):
        suffix = name[len("trans_"):]
        t = model.transition
        if suffix == "in_proj_w":
            _safe_deallocate(t.self_attn.in_proj_weight)
            t.self_attn.in_proj_weight = param
        elif suffix == "in_proj_b":
            _safe_deallocate(t.self_attn.in_proj_bias)
            t.self_attn.in_proj_bias = param
        elif suffix == "out_proj_w":
            _safe_deallocate(t.self_attn.out_proj_weight)
            t.self_attn.out_proj_weight = param
        elif suffix == "out_proj_b":
            _safe_deallocate(t.self_attn.out_proj_bias)
            t.self_attn.out_proj_bias = param
        elif suffix == "attn_norm_w":
            _safe_deallocate(t.attn_norm.weight)
            t.attn_norm.weight = param
        elif suffix == "ffn_w":
            _safe_deallocate(t.ffn.weight)
            t.ffn.weight = param
        elif suffix == "ffn_out_w":
            _safe_deallocate(t.ffn_out.weight)
            t.ffn_out.weight = param
        elif suffix == "gate_w":
            _safe_deallocate(t.gate.weight)
            t.gate.weight = param
        elif suffix == "gate_b":
            _safe_deallocate(t.gate.bias)
            t.gate.bias = param
        elif suffix == "output_norm_w":
            _safe_deallocate(t.output_norm.weight)
            t.output_norm.weight = param
        else:
            print(f"WARNING: unknown transition param: {name}", flush=True)

    @staticmethod
    def _set_attn_param(attn, suffix, param):
        if suffix == "in_proj_w":
            _safe_deallocate(attn.in_proj_weight)
            attn.in_proj_weight = param
        elif suffix == "in_proj_b":
            _safe_deallocate(attn.in_proj_bias)
            attn.in_proj_bias = param
        elif suffix == "out_proj_w":
            _safe_deallocate(attn.out_proj_weight)
            attn.out_proj_weight = param
        elif suffix == "out_proj_b":
            _safe_deallocate(attn.out_proj_bias)
            attn.out_proj_bias = param

    @staticmethod
    def _set_norm_param(norm, suffix, param):
        if suffix == "w":
            _safe_deallocate(norm.weight)
            norm.weight = param
        elif suffix == "b":
            _safe_deallocate(norm.bias)
            norm.bias = param

    @staticmethod
    def _set_linear_param(linear, suffix, param):
        if suffix == "w":
            _safe_deallocate(linear.weight)
            linear.weight = param
        elif suffix == "b":
            _safe_deallocate(linear.bias)
            linear.bias = param


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_grad_norm(grads: dict, max_norm: float) -> float:
    """Clip gradient norm in-place on device. Returns pre-clip norm."""
    if not grads:
        return 0.0

    device = next(iter(grads.values())).device()

    # Compute global norm squared
    norm_sq_tt = ttnn.from_torch(
        torch.tensor([0.0], dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    for name, g in grads.items():
        g_fp32 = g
        if g.dtype != ttnn.float32:
            g_fp32 = ttnn.typecast(g, ttnn.float32)
        sq = ttnn.mul(g_fp32, g_fp32)
        if g.dtype != ttnn.float32:
            _safe_deallocate(g_fp32)
        s = ttnn.sum(sq)
        _safe_deallocate(sq)
        old = norm_sq_tt
        norm_sq_tt = ttnn.add(norm_sq_tt, s)
        _safe_deallocate(old)
        _safe_deallocate(s)

    norm_sq = ttnn.to_torch(norm_sq_tt).item()
    _safe_deallocate(norm_sq_tt)
    norm = math.sqrt(norm_sq)

    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        scale_tt = ttnn.from_torch(
            torch.tensor([scale], dtype=torch.float32),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )
        for name, g in grads.items():
            g_fp32 = g
            if g.dtype != ttnn.float32:
                g_fp32 = ttnn.typecast(g, ttnn.float32)
            clipped = ttnn.mul(g_fp32, scale_tt)
            if g.dtype != ttnn.float32:
                clipped_bf = ttnn.typecast(clipped, g.dtype)
                _safe_deallocate(clipped)
                clipped = clipped_bf
                _safe_deallocate(g_fp32)
            grads[name] = clipped
            if g is not clipped:
                _safe_deallocate(g)
        _safe_deallocate(scale_tt)

    return norm


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_lr(step: int, args) -> float:
    """Learning rate schedule: linear warmup + cosine decay."""
    if step < args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Native TT training for text latent-memory model")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_slots", type=int, default=8)
    parser.add_argument("--max_reasoning_steps", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--max_prompt_len", type=int, default=64)
    parser.add_argument("--max_answer_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--monitor_rss", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = ttnn.bfloat16 if args.precision == "bf16" else ttnn.float32

    # Create device
    print(f"Opening device {args.device}...", flush=True)
    device = create_device(args.device)

    # Create config
    config = TTTextLatentMemoryConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        n_slots=args.n_slots,
        max_reasoning_steps=args.max_reasoning_steps,
        expand=args.expand,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # Create model
    print("Creating TT model...", flush=True)
    model = TTTextLatentMemoryModel(config, device, dtype=dtype)
    n_params = model.get_num_params()
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M) precision={args.precision}", flush=True)

    # Create dataset
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, "tiny_challenges_train.txt")
    valid_path = os.path.join(data_dir, "tiny_challenges_valid.txt")
    dataset = ChallengeDataset(
        train_path=train_path,
        valid_path=valid_path,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    # Create optimizer
    params = model.get_params()
    optimizer = TTAdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW lr={args.lr} wd={args.weight_decay}", flush=True)

    # RSS monitoring
    def get_rss():
        if not args.monitor_rss:
            return 0
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
        except Exception:
            return 0

    # Training loop
    rng = random.Random(args.seed)
    start_time = time.time()
    initial_rss = get_rss()

    print(f"\nStarting native TT training for {args.steps} steps...", flush=True)
    print(f"Initial RSS: {initial_rss:.1f} MB" if args.monitor_rss else "", flush=True)

    for step in range(args.steps):
        lr = get_lr(step, args)
        optimizer.set_lr(lr)

        # Sample batch — use fixed sequence lengths to avoid triggering
        # new kernel compilations on every batch (causes device hangs)
        prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
            dataset.sample_batch(
                args.batch_size, "train", rng,
                fixed_prompt_len=args.max_prompt_len,
                fixed_answer_len=args.max_answer_len + 1,  # +1 for BOS
            )

        # Forward pass (training mode — caches intermediates)
        logits_tt = model.forward_train(prompt_ids, prompt_mask, dec_input, ans_mask)

        # Compute token accuracy from logits before loss (avoids extra eval forward)
        if step % 10 == 0 or args.smoke_test:
            with torch.no_grad():
                logits_torch = ttnn.to_torch(logits_tt).float()
                preds = logits_torch.argmax(-1)
                tok_acc = ((preds == ans_targets).float() * ans_mask.float()).sum() / \
                           ans_mask.float().sum().clamp_min(1)

        # Loss + gradient (host-side loss for large vocab, gradient back to device)
        loss_val, grad_logits = cross_entropy_loss_and_grad(
            logits_tt, ans_targets, ans_mask, device, dtype=dtype
        )
        _safe_deallocate(logits_tt)

        # Backward pass (on device)
        grads = model.backward(grad_logits)
        _safe_deallocate(grad_logits)

        # Gradient clipping
        grad_norm = clip_grad_norm(grads, args.grad_clip)

        # Optimizer step
        optimizer.step(grads, model)

        # Deallocate gradients
        for name, g in grads.items():
            _safe_deallocate(g)

        # Synchronize and clear caches to prevent device memory fragmentation
        model.clear_caches()

        # Return freed host memory to OS (prevents RSS growth from glibc arena)
        if step % 10 == 0:
            malloc_trim()

        # Log
        if step % 10 == 0 or args.smoke_test:
            elapsed = time.time() - start_time
            rss = get_rss()
            rss_str = f" rss={rss:.1f}MB" if args.monitor_rss else ""
            print(
                f"{step:6d} loss={loss_val:.4f} "
                f"tok_acc={tok_acc.item():.3f} "
                f"grad={grad_norm:.3f} "
                f"lr={lr:.2e} "
                f"t={elapsed:.0f}s{rss_str}",
                flush=True,
            )

        # Check for NaN/Inf
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"ERROR: loss became {loss_val} at step {step}", flush=True)
            break

        if grad_norm > 1e4:
            print(f"WARNING: gradient norm {grad_norm} is very large at step {step}", flush=True)

    # Final RSS
    final_rss = get_rss()
    if args.monitor_rss:
        print(f"\nFinal RSS: {final_rss:.1f} MB (delta: {final_rss - initial_rss:.1f} MB)", flush=True)

    elapsed = time.time() - start_time
    print(f"\nTraining complete: {args.steps} steps in {elapsed:.0f}s", flush=True)

    # Close device
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
