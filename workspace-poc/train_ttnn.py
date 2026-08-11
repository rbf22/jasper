"""
Tenstorrent native training script using tt-nn.

Trains the Mamba2 model directly on TT hardware using tt-nn ops,
bypassing PyTorch/XLA entirely. Forward pass runs on device, backward
pass uses a hybrid approach (tt-nn ops + host autograd for SSD), and
the optimizer is AdamW implemented on host.

Usage:
    .tt-venv/bin/python train_ttnn.py --config configs/cell_a_tt.yaml
    .tt-venv/bin/python train_ttnn.py --config configs/cell_a_tt.yaml --profile
    .tt-venv/bin/python train_ttnn.py --config configs/cell_a_tt.yaml --steps 50 --micro_batch 8
"""

import os
import sys

# ── TT_VISIBLE_DEVICES must be set BEFORE any ttnn/torch import ──────────────
# When --device is passed, pin this process to a single physical TT chip so
# multiple processes can run in parallel without driver-level contention.
# TT_VISIBLE_DEVICES remaps the physical device to logical 0, so we always
# open_device(device_id=0) inside the process.  This mirrors the pattern used
# by the tt-boltz quad-card demo (demo_quad/worker.py).
_device_id_from_argv = 0
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _device_id_from_argv = int(sys.argv[_i + 1])
        break
    if _a.startswith("--device="):
        _device_id_from_argv = int(_a.split("=", 1)[1])
        break
os.environ.setdefault("TT_VISIBLE_DEVICES", str(_device_id_from_argv))

# P300 chips need a custom fabric mesh graph descriptor; without it,
# ttnn.open_device aborts with TT_FATAL on tt_cluster.cpp.  This mirrors
# the pattern from the tt-boltz quad-card demo (demo_quad/pool.py).
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
        # Try p150 first (works for single-device and after board reset),
        # then p300 (needed for multi-chip fabric topologies)
        candidates = [
            "p150_mesh_graph_descriptor.textproto",
            "p300_mesh_graph_descriptor.textproto",
        ]
        for name in candidates:
            # ttnn package copy
            if spec is not None and spec.submodule_search_locations:
                path = (
                    Path(next(iter(spec.submodule_search_locations)))
                    / "tt_metal" / "fabric" / "mesh_graph_descriptors"
                    / name
                )
                if path.is_file():
                    return str(path)
            # pjrt_plugin_tt copy
            import sys
            for p in sys.path:
                candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if candidate.is_file():
                    return str(candidate)
            # venv site-packages directly
            venv_path = Path("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages")
            candidate = venv_path / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
            if candidate.is_file():
                return str(candidate)
    except Exception:
        pass
    return None
if _is_p300():
    _mgd = _find_mesh_graph_descriptor()
    if _mgd:
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)

# Suppress Metal C++ warnings (e.g. ROW MAJOR tile extraction in ttnn.embedding)
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import argparse
import gc
import math
import time
import torch
import ttnn
import yaml
import random

# Suppress loguru warnings from ttnn
from loguru import logger
logger.remove()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_ttnn import TTMambaWorkspaceModel, ModelConfig

# Cache for identity matrix on device (avoids recreating every step)
_identity_cache = {}  # (V, device_id) -> ttnn.Tensor
from data import Vocab, sample_batch, generate_eval_set


# ---------------------------------------------------------------------------
# Memory management helpers
# ---------------------------------------------------------------------------

def _safe_deallocate(tensor):
    """Deallocate a ttnn tensor if it's a valid device tensor.

    ttnn.deallocate() releases the on-device buffer immediately.  Without
    explicit deallocation, intermediate tensors from every ttnn op accumulate
    in device DRAM until Python's garbage collector runs — but ttnn wrapper
    objects are tiny, so GC doesn't see the memory pressure and may not run
    for hundreds of steps.  This was the root cause of the OOM that killed
    the system when running 3 training processes in parallel.
    """
    if tensor is None:
        return
    try:
        ttnn.deallocate(tensor)
    except Exception:
        pass  # already deallocated, host tensor, or not a device tensor


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model_config(cfg: dict) -> ModelConfig:
    """Build a ModelConfig from a YAML config dict."""
    return ModelConfig(
        d_model=cfg.get("d_model", 384),
        n_layers=cfg.get("n_layers", 14),
        vocab_size=cfg.get("vocab_size", 128),
        n_heads=cfg.get("n_heads", 4),
        use_attention=cfg.get("use_attention", False),
        attention_positions=cfg.get("attention_positions", [5, 10]),
        use_workspace=cfg.get("use_workspace", False),
        n_workspace_slots=cfg.get("n_workspace_slots", 16),
        recurrent_core=cfg.get("recurrent_core", False),
        core_start=cfg.get("core_start", 6),
        core_end=cfg.get("core_end", 10),
        k_train_max=cfg.get("k_train_max", 3),
        k_inference=cfg.get("k_inference", 6),
        attention_residual_core=cfg.get("attention_residual_core", False),
        use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", False),
        spectral_norm_bound=cfg.get("spectral_norm_bound", 5.0),
        backbone_spectral_norm_bound=cfg.get("backbone_spectral_norm_bound", 2.0),
        chain_scale_safety=cfg.get("chain_scale_safety", 1.0),
        freeze_gamma=cfg.get("freeze_gamma", False),
        freeze_slot_decay=cfg.get("freeze_slot_decay", False),
        ws_entropy_weight=cfg.get("ws_entropy_weight", 0.0),
        ws_diversity_weight=cfg.get("ws_diversity_weight", 0.0),
        gate_init=cfg.get("gate_init", 0.0),
        slot_decay_init=cfg.get("slot_decay_init", 1.0),
        slot_permutation=cfg.get("slot_permutation", False),
        gate_schedule_steps=cfg.get("gate_schedule_steps", 0),
        gate_clamp_bound=cfg.get("gate_clamp_bound", 0.0),
        short_conv=cfg.get("short_conv", False),
        short_conv_kernel=cfg.get("short_conv_kernel", 3),
        per_channel_decay=cfg.get("per_channel_decay", False),
    )


# ---------------------------------------------------------------------------
# Optimizer (device-side AdamW)
# ---------------------------------------------------------------------------

class TTAdamW:
    """AdamW optimizer for tt-nn tensors — fully on device, mixed precision.

    Maintains fp32 master copies of all parameters.  Each step:
      1. Read fp32 master (not the bf16 model param)
      2. Compute AdamW update in fp32
      3. Update fp32 master
      4. Downcast master to bf16 and write into the model

    This is the standard mixed-precision training pattern.  Without it,
    bf16 params at magnitude O(1) or larger (gates at -2.0, norm weights
    at 1.0) are permanently frozen: the per-step update (~1e-4) is far
    below the bf16 ULP (~8e-3 at 1.0, ~1.6e-2 at 2.0), so downcasting
    after each step rounds the update to zero and it never accumulates.

    All optimizer state (exp_avg, exp_avg_sq, master) is kept on device
    in fp32.  No host-device transfers during the optimizer step.

    Supports per-parameter LR groups via lr_groups: a dict mapping
    param name prefixes to LR multipliers. For example:
        lr_groups={"ws_": 0.25}  # workspace params get 0.25x the base LR
    Params not matching any prefix use the base LR (multiplier 1.0).
    """

    def __init__(self, params: dict, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                 lr_groups: dict = None, wd_groups: dict = None, beta2_groups: dict = None):
        self.base_lr = lr
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # Per-parameter LR groups: prefix -> multiplier
        # Matching: if the key starts with "suffix:" it matches name.endswith(key[7:]);
        #           otherwise it matches name.startswith(key). There is no "prefix:"
        #           form — bare keys already mean prefix match.
        # NOTE: group keys are checked in dict iteration order (= config file
        # order, insertion order in Python 3.7+) and the FIRST match wins
        # (`break`). If a param name matches multiple keys (e.g. both a
        # specific "ws_read_gate" entry and a catch-all "ws_" prefix), put
        # the more specific key earlier in the config's lr_groups/wd_groups
        # mapping, or it'll silently get the catch-all's multiplier instead.
        self.lr_groups = lr_groups or {}
        self.param_lr_mult = {}  # name -> multiplier (resolved at init)
        # Per-parameter weight decay groups: same matching as lr_groups
        self.wd_groups = wd_groups or {}
        self.param_wd = {}  # name -> weight decay (resolved at init)
        # Per-parameter beta2 groups: same matching as lr_groups.
        # Allows high-variance parameters (e.g. ReZero gates) to use a higher
        # beta2 (e.g. 0.999) for more stable second-moment estimation.
        self.beta2_groups = beta2_groups or {}
        self.param_beta2 = {}  # name -> beta2 (resolved at init)
        for name in params:
            mult = 1.0
            for key, prefix_mult in self.lr_groups.items():
                if key.startswith("suffix:"):
                    suffix = key[7:]
                    if name.endswith(suffix):
                        mult = prefix_mult
                        break
                elif name.startswith(key):
                    mult = prefix_mult
                    break
            self.param_lr_mult[name] = mult
            # Resolve weight decay: default to global, override with group match
            wd = self.weight_decay
            for key, group_wd in self.wd_groups.items():
                if key.startswith("suffix:"):
                    suffix = key[7:]
                    if name.endswith(suffix):
                        wd = group_wd
                        break
                elif name.startswith(key):
                    wd = group_wd
                    break
            self.param_wd[name] = wd
            # Resolve beta2: default to global, override with group match
            b2 = self.beta2
            for key, group_b2 in self.beta2_groups.items():
                if key.startswith("suffix:"):
                    suffix = key[7:]
                    if name.endswith(suffix):
                        b2 = group_b2
                        break
                elif name.startswith(key):
                    b2 = group_b2
                    break
            self.param_beta2[name] = b2

        self.param_names = list(params.keys())
        self.device = None  # set from first param
        self.param_dtype = {}   # name -> ttnn dtype (model storage dtype, may be bf16)
        self.exp_avg = {}       # name -> device tensor (always fp32)
        self.exp_avg_sq = {}    # name -> device tensor (always fp32)
        self.master = {}        # name -> fp32 master copy on device

        for name, tt_tensor in params.items():
            if self.device is None:
                self.device = tt_tensor.device()
            self.param_dtype[name] = tt_tensor.dtype
            shape = tuple(tt_tensor.shape)
            # Optimizer state always in fp32
            self.exp_avg[name] = ttnn.zeros(shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            self.exp_avg_sq[name] = ttnn.zeros(shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            # fp32 master copy — upcast from model's dtype.
            # Always create a new tensor — never alias to the model param.
            new_master = ttnn.typecast(tt_tensor, ttnn.float32)
            if new_master is tt_tensor:
                new_master = ttnn.from_torch(
                    ttnn.to_torch(tt_tensor).clone(),
                    dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=tt_tensor.device())
            self.master[name] = new_master

    def set_lr(self, lr: float):
        self.lr = lr

    def get_param_lr(self, name: str) -> float:
        """Get the effective LR for a parameter, applying group multipliers."""
        return self.lr * self.param_lr_mult.get(name, 1.0)

    def step(self, grads: dict, model: TTMambaWorkspaceModel):
        """Apply one optimizer step — fully on device, fp32 master weights.

        grads: dict of name -> tt-nn gradient tensor
        model: the model (to update its bf16 parameters)
        """
        self.step_count += 1
        device = self.device

        for name in self.param_names:
            if name not in grads:
                continue

            grad = grads[name]
            storage_dtype = self.param_dtype[name]
            m_old = self.exp_avg[name]     # fp32
            v_old = self.exp_avg_sq[name]  # fp32
            param_old = self.master[name]  # fp32 master

            # Upcast grad to fp32 if needed
            grad_fp32 = None
            if grad.dtype != ttnn.float32:
                grad_fp32 = ttnn.typecast(grad, ttnn.float32)
                grad_for_update = grad_fp32
            else:
                grad_for_update = grad

            lr = self.get_param_lr(name)
            b1, b2, eps = self.beta1, self.param_beta2.get(name, self.beta2), self.eps
            wd = self.param_wd.get(name, self.weight_decay)

            # m = b1*m + (1-b1)*g  — deallocate intermediates
            m_b1 = ttnn.mul(m_old, b1)
            g_1mb1 = ttnn.mul(grad_for_update, 1.0 - b1)
            m = ttnn.add(m_b1, g_1mb1)
            _safe_deallocate(m_b1)
            _safe_deallocate(g_1mb1)

            # v = b2*v + (1-b2)*g^2
            g_sq = ttnn.mul(grad_for_update, grad_for_update)
            v_b2 = ttnn.mul(v_old, b2)
            g_sq_1mb2 = ttnn.mul(g_sq, 1.0 - b2)
            v = ttnn.add(v_b2, g_sq_1mb2)
            _safe_deallocate(g_sq)
            _safe_deallocate(v_b2)
            _safe_deallocate(g_sq_1mb2)

            # bias correction
            bc1 = 1.0 - b1 ** self.step_count
            bc2 = 1.0 - b2 ** self.step_count
            m_hat = ttnn.mul(m, 1.0 / bc1)
            v_hat = ttnn.mul(v, 1.0 / bc2)

            # param -= lr * m_hat / (sqrt(v_hat) + eps)
            sqrt_v = ttnn.sqrt(v_hat)
            denom = ttnn.add(sqrt_v, eps)
            update = ttnn.div(m_hat, denom)
            _safe_deallocate(sqrt_v)
            _safe_deallocate(denom)
            _safe_deallocate(m_hat)
            _safe_deallocate(v_hat)

            update_scaled = ttnn.mul(update, lr)
            _safe_deallocate(update)
            param = ttnn.sub(param_old, update_scaled)
            _safe_deallocate(update_scaled)

            # weight decay
            # NOTE: unless a config sets `wd_groups` to override it, `wd` here
            # is the single global `weight_decay` applied to EVERY parameter,
            # including 1D norm weights, ReZero gates (read_gate/write_gate/
            # backbone gate, all init 0 and meant to grow via gradient
            # descent), and slot_decay/gamma. This differs from the common
            # transformer convention of excluding norms/biases/scalars from
            # weight decay. It hasn't been identified as a bug (current
            # experiments were run this way and gates do grow in practice —
            # see gz_gr/gz_gw in training logs), but if gates seem to
            # struggle to grow or plateau near 0, weight decay pulling them
            # back is a place to check first. Use `wd_groups` in the config
            # (e.g. `wd_groups: {"suffix:_gate": 0.0}`) to exclude specific
            # params instead of changing this default.
            param_wd = ttnn.mul(param, 1.0 - lr * wd)
            _safe_deallocate(param)
            param = param_wd

            # Deallocate old optimizer state (replaced by new tensors above)
            _safe_deallocate(m_old)
            _safe_deallocate(v_old)
            _safe_deallocate(param_old)
            _safe_deallocate(grad_fp32)

            # Store updated optimizer state (fp32)
            self.exp_avg[name] = m
            self.exp_avg_sq[name] = v
            # Update fp32 master
            self.master[name] = param

            # Downcast master to model's storage dtype and write into model.
            # When the storage dtype is fp32, we must NOT alias the model param
            # to the master — the next step's _safe_deallocate(param_old) would
            # free the model's parameter. Create a copy instead.
            if storage_dtype != ttnn.float32:
                param_bf16 = ttnn.typecast(param, storage_dtype)
            else:
                param_bf16 = ttnn.from_torch(
                    ttnn.to_torch(param).clone(),
                    dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=param.device())
            self._set_model_param(model, name, param_bf16)

    def _set_model_param(self, model: TTMambaWorkspaceModel, name: str, param: "ttnn.Tensor"):
        """Update a model parameter from a device tensor (no transfer)."""
        if name == "token_emb_weight":
            _safe_deallocate(model.token_emb_weight)
            model.token_emb_weight = param
            model.lm_head_weight = param  # weight tying
        elif name == "norm_weight":
            _safe_deallocate(model.norm.weight)
            model.norm.weight = param
        elif name.startswith("layer_"):
            parts = name.split("_", 2)  # ["layer", "0", "in_proj_weight"]
            layer_idx = int(parts[1])
            param_name = parts[2]
            model.layers[layer_idx].set_params({param_name: param})
        elif name in ("ws_ar_query", "ws_ar_scale") and model.attn_residual is not None:
            ar_param_name = name[3:]  # strip "ws_" prefix
            model.attn_residual.set_params({ar_param_name: param})
        elif name.startswith("ws_"):
            ws_param_name = name[3:]  # strip "ws_" prefix
            model.workspace.set_params({ws_param_name: param})
        else:
            print(f"WARNING: Unknown param name {name}")

    def get_state(self) -> dict:
        """Return optimizer state for checkpointing (transfers to host)."""
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "exp_avg": {k: ttnn.to_torch(v).clone() for k, v in self.exp_avg.items()},
            "exp_avg_sq": {k: ttnn.to_torch(v).clone() for k, v in self.exp_avg_sq.items()},
            "master": {k: ttnn.to_torch(v).clone() for k, v in self.master.items()},
        }

    def sync_master_from_model(self, model: TTMambaWorkspaceModel, names: set = None):
        """Re-sync fp32 master copies from the model's current parameters.

        Call this after any operation that modifies model parameters outside
        the optimizer (e.g. slot normalization, spectral normalization, gate
        schedule).  Without this, the master would diverge from the model and
        the next optimizer step would overwrite the normalization.

        Args:
            names: if given, only sync these param names.  If None, sync all.
                   Use a restricted set to avoid clobbering sub-ULP fp32
                   accumulation on params that weren't modified externally.
        """
        model_params = model.get_params()
        for name in self.param_names:
            if name in model_params and (names is None or name in names):
                p = model_params[name]
                old_master = self.master.get(name)
                # Always create a new fp32 tensor — never alias the model param.
                # If we set self.master[name] = p (same object), the next
                # optimizer step's _safe_deallocate(param_old) would free the
                # model's actual parameter tensor.
                new_master = ttnn.typecast(p, ttnn.float32)
                # If typecast returned the same object (same dtype, some
                # implementations do this), force a copy via from_torch.
                if new_master is p:
                    new_master = ttnn.from_torch(
                        ttnn.to_torch(p).clone(),
                        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=p.device())
                self.master[name] = new_master
                _safe_deallocate(old_master)

    def load_state(self, state: dict, model: TTMambaWorkspaceModel):
        """Load optimizer state from checkpoint (transfers to device, in fp32)."""
        self.step_count = state.get("step_count", 0)
        self.lr = state.get("lr", self.base_lr)
        for name in self.param_names:
            if name in state.get("exp_avg", {}):
                host_m = state["exp_avg"][name]
                _safe_deallocate(self.exp_avg.get(name))
                self.exp_avg[name] = ttnn.from_torch(
                    host_m.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            if name in state.get("exp_avg_sq", {}):
                host_v = state["exp_avg_sq"][name]
                _safe_deallocate(self.exp_avg_sq.get(name))
                self.exp_avg_sq[name] = ttnn.from_torch(
                    host_v.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            if name in state.get("master", {}):
                host_master = state["master"][name]
                _safe_deallocate(self.master.get(name))
                self.master[name] = ttnn.from_torch(
                    host_master.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            elif name in state.get("exp_avg", {}):
                # Backward compat: no master in old checkpoint — reconstruct
                # from model params (upcast to fp32).  Loses sub-ULP accumulation
                # but is the best we can do.
                model_params = model.get_params()
                if name in model_params:
                    _safe_deallocate(self.master.get(name))
                    p = model_params[name]
                    new_master = ttnn.typecast(p, ttnn.float32)
                    if new_master is p:
                        new_master = ttnn.from_torch(
                            ttnn.to_torch(p).clone(),
                            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=p.device())
                    self.master[name] = new_master


# ---------------------------------------------------------------------------
# Loss function (cross-entropy on device)
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits_tt, labels, ignore_index=-100):
    """Compute cross-entropy loss and gradient w.r.t. logits — fully on device.

    logits_tt: (B, T, vocab_size) tt-nn tensor on device
    labels: (B, T) PyTorch tensor
    Returns: (loss_value, grad_logits_tt)
    """
    device = logits_tt.device()
    B, T, V = labels.shape[0], labels.shape[1], logits_tt.shape[-1]

    # Shift: predict token t+1 from token t — use logits[:, :-1], labels[:, :-1]
    shift_logits = logits_tt[:, :-1, :]  # (B, T-1, V)
    shift_labels = labels[:, :-1]  # (B, T-1) — keep original dtype for mask

    # Create valid mask (1.0 for valid, 0.0 for ignore_index) — small tensor, host transfer is cheap
    valid_mask_2d = (shift_labels != ignore_index).float()  # (B, T-1)
    n_valid = int(valid_mask_2d.sum().item())
    n_valid = max(n_valid, 1)  # avoid division by zero

    # Safe labels: clamp negative to 0 for one-hot embedding lookup
    safe_labels = shift_labels.clamp(min=0).to(torch.int32)  # (B, T-1)
    flat_labels = safe_labels.reshape(-1)  # (B*(T-1),)

    # Compute softmax on device
    probs = ttnn.softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # Create one-hot encoding on device using embedding with identity matrix (cached)
    cache_key = (V, device.id())
    if cache_key not in _identity_cache:
        identity = torch.eye(V, dtype=torch.bfloat16)
        _identity_cache[cache_key] = ttnn.from_torch(identity, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    identity_tt = _identity_cache[cache_key]
    label_indices = ttnn.from_torch(flat_labels.unsqueeze(-1), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    one_hot = ttnn.embedding(label_indices, identity_tt, layout=ttnn.TILE_LAYOUT)  # (B*(T-1), 1, V)
    one_hot = ttnn.reshape(one_hot, [B, T - 1, V])
    _safe_deallocate(label_indices)

    # Apply valid mask to one_hot: zero out invalid positions
    mask_tt = ttnn.from_torch(
        valid_mask_2d.unsqueeze(-1).to(torch.bfloat16),  # (B, T-1, 1)
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    one_hot_masked = ttnn.mul(one_hot, mask_tt)  # (B, T-1, V) — zero at invalid positions
    _safe_deallocate(one_hot)

    # grad_logits = (probs - one_hot_masked) / n_valid
    # At invalid positions, one_hot_masked=0, so grad = probs / n_valid
    # But we want grad=0 at invalid positions, so mask the gradient too
    inv_n_valid = 1.0 / n_valid
    inv_n_valid_tt = ttnn.from_torch(
        torch.tensor([inv_n_valid], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    diff = ttnn.sub(probs, one_hot_masked)
    grad_shift = ttnn.mul(diff, inv_n_valid_tt)  # (B, T-1, V)
    _safe_deallocate(diff)
    _safe_deallocate(inv_n_valid_tt)
    grad_shift_masked = ttnn.mul(grad_shift, mask_tt)  # zero out invalid positions
    _safe_deallocate(grad_shift)
    grad_shift = grad_shift_masked

    # Pad to (B, T, V) — last position has zero gradient
    zeros_pad = ttnn.zeros((B, 1, V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_logits_tt = ttnn.concat([grad_shift, zeros_pad], dim=1)  # (B, T, V)
    _safe_deallocate(grad_shift)
    _safe_deallocate(zeros_pad)

    # Compute loss value on device: loss = -sum(one_hot_masked * log(probs)) / n_valid
    probs_clamped = ttnn.clamp(probs, min=1e-8, max=1.0)
    log_probs = ttnn.log(probs_clamped)  # (B, T-1, V)
    _safe_deallocate(probs_clamped)
    neg_log_probs_at_target = ttnn.mul(one_hot_masked, log_probs)  # (B, T-1, V)
    _safe_deallocate(log_probs)
    _safe_deallocate(one_hot_masked)
    loss_sum = ttnn.sum(neg_log_probs_at_target)  # scalar
    _safe_deallocate(neg_log_probs_at_target)
    loss_value = -ttnn.to_torch(loss_sum).item() / n_valid
    _safe_deallocate(loss_sum)

    # Deallocate remaining intermediates (probs and mask_tt are no longer needed)
    _safe_deallocate(probs)
    _safe_deallocate(mask_tt)

    return loss_value, grad_logits_tt


# ---------------------------------------------------------------------------
# Gradient clipping and accumulation
# ---------------------------------------------------------------------------

def clip_grad_norm(grads: dict, max_norm: float, gamma_scale: float = 128.0,
                   ws_max_norm: float = None) -> float:
    """Clip gradient norm in-place on device. Returns the original (pre-clip) norm.

    Component-wise gradient clipping (Yang et al., 2022):
    When ws_max_norm is provided, gradients are split into two groups:
      - Workspace params (names starting with "ws_"): clipped to ws_max_norm
      - Backbone params (everything else): clipped to max_norm
    Each group's norm is computed and clipped independently. This prevents
    workspace gradient spikes from dominating the global clip and starving
    the backbone of learning signal.

    When ws_max_norm is None, falls back to global clipping (all params to max_norm).

    Gamma parameters (retention decay) are included in the backbone group with
    a scale correction: their gradients are divided by gamma_scale (T) before
    contributing to the norm.  This accounts for the structural O(T^2) gradient
    scale mismatch vs O(T) for weight matrices.

    The returned norm is the global norm (all params combined, with gamma/T
    correction), so the skip-on-spike threshold is comparable across runs.
    """
    if not grads:
        return 0.0

    device = next(iter(grads.values())).device()

    gamma_scale_tt = ttnn.from_torch(
        torch.tensor([1.0 / gamma_scale], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Split grads into workspace and backbone groups
    ws_grads = {}
    bb_grads = {}
    for name, tt_grad in grads.items():
        if name.startswith("ws_"):
            ws_grads[name] = tt_grad
        else:
            bb_grads[name] = tt_grad

    # Compute norm for each group
    def compute_group_norm_sq(group_grads):
        """Compute sum of squared norms on device, return scalar tensor."""
        norm_sq_tt = ttnn.from_torch(
            torch.tensor([0.0], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        for name, tt_grad in group_grads.items():
            if name.endswith("_gamma"):
                g_scaled = ttnn.mul(tt_grad, gamma_scale_tt)
                sq = ttnn.mul(g_scaled, g_scaled)
                _safe_deallocate(g_scaled)
            else:
                sq = ttnn.mul(tt_grad, tt_grad)
            norm_sq = ttnn.sum(sq)
            _safe_deallocate(sq)
            old = norm_sq_tt
            norm_sq_tt = ttnn.add(norm_sq_tt, norm_sq)
            _safe_deallocate(old)
            _safe_deallocate(norm_sq)
        return norm_sq_tt

    def clip_group(group_grads, group_max_norm, group_norm_sq_tt):
        """Clip a group's gradients in-place. Returns pre-clip norm."""
        if not group_grads:
            return 0.0
        group_norm_sq = ttnn.to_torch(group_norm_sq_tt).item()
        group_norm = math.sqrt(group_norm_sq)
        if group_norm > group_max_norm and group_norm > 0:
            scale = group_max_norm / (group_norm + 1e-6)
            scale_tt = ttnn.from_torch(
                torch.tensor([scale], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            for name in group_grads:
                old_grad = group_grads[name]
                group_grads[name] = ttnn.mul(old_grad, scale_tt)
                _safe_deallocate(old_grad)
            _safe_deallocate(scale_tt)
        return group_norm

    # Compute norms for both groups
    bb_norm_sq_tt = compute_group_norm_sq(bb_grads)
    ws_norm_sq_tt = compute_group_norm_sq(ws_grads) if ws_grads else None

    # Clip each group independently
    bb_norm = clip_group(bb_grads, max_norm, bb_norm_sq_tt)
    if ws_max_norm is not None and ws_grads:
        ws_norm = clip_group(ws_grads, ws_max_norm, ws_norm_sq_tt)
    else:
        # No separate workspace clip — clip with backbone
        ws_norm = 0.0
        if ws_grads:
            # Add ws grads to backbone group for global clip
            for name, tt_grad in ws_grads.items():
                sq = ttnn.mul(tt_grad, tt_grad)
                norm_sq = ttnn.sum(sq)
                _safe_deallocate(sq)
                old = bb_norm_sq_tt
                bb_norm_sq_tt = ttnn.add(bb_norm_sq_tt, norm_sq)
                _safe_deallocate(old)
                _safe_deallocate(norm_sq)
            # Recompute and clip combined
            total_norm_sq = ttnn.to_torch(bb_norm_sq_tt).item()
            total_norm = math.sqrt(total_norm_sq)
            if total_norm > max_norm and total_norm > 0:
                scale = max_norm / (total_norm + 1e-6)
                scale_tt = ttnn.from_torch(
                    torch.tensor([scale], dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                for name in bb_grads:
                    old_grad = bb_grads[name]
                    bb_grads[name] = ttnn.mul(old_grad, scale_tt)
                    _safe_deallocate(old_grad)
                for name in ws_grads:
                    old_grad = ws_grads[name]
                    ws_grads[name] = ttnn.mul(old_grad, scale_tt)
                    _safe_deallocate(old_grad)
                _safe_deallocate(scale_tt)
            bb_norm = total_norm

    # Merge clipped grads back
    for name in ws_grads:
        grads[name] = ws_grads[name]
    for name in bb_grads:
        grads[name] = bb_grads[name]

    # Return global norm (for skip-on-spike threshold)
    if ws_norm_sq_tt is not None:
        total_norm_sq_tt = ttnn.add(bb_norm_sq_tt, ws_norm_sq_tt)
        total_norm = math.sqrt(ttnn.to_torch(total_norm_sq_tt).item())
        _safe_deallocate(total_norm_sq_tt)
    else:
        total_norm = bb_norm
    _safe_deallocate(bb_norm_sq_tt)
    _safe_deallocate(ws_norm_sq_tt)
    _safe_deallocate(gamma_scale_tt)
    return total_norm


def accumulate_grads(acc_grads: dict, new_grads: dict, accum_factor: int):
    """Accumulate new gradients into the running sum (host-side).

    acc_grads: dict of name -> host fp32 tensor (running sum)
    new_grads: dict of name -> tt-nn gradient tensor
    accum_factor: divide new grads by this before adding
    """
    for name, tt_grad in new_grads.items():
        grad_host = ttnn.to_torch(tt_grad).float() / accum_factor
        if name not in acc_grads:
            acc_grads[name] = grad_host.clone()
        else:
            acc_grads[name] += grad_host


def host_grads_to_tt(acc_grads: dict, device) -> dict:
    """Convert accumulated host gradients back to tt-nn tensors."""
    tt_grads = {}
    for name, grad_host in acc_grads.items():
        dtype = ttnn.float32 if "A_log" in name or name.endswith("_D") else ttnn.bfloat16
        tt_grads[name] = ttnn.from_torch(
            grad_host.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
    return tt_grads


# ---------------------------------------------------------------------------
# LR warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, base_lr: float, warmup_steps: int,
           cosine_decay_steps: int = 0, cosine_min_ratio: float = 0.1) -> float:
    """Linear warmup from 0 to base_lr over warmup_steps, then optional cosine decay.

    If cosine_decay_steps > 0, after warmup the LR follows a cosine schedule
    from base_lr down to base_lr * cosine_min_ratio over cosine_decay_steps,
    then stays at the minimum for the remainder.
    """
    if warmup_steps <= 0:
        effective_step = step
    elif step < warmup_steps:
        return base_lr * step / warmup_steps
    else:
        effective_step = step - warmup_steps

    if cosine_decay_steps <= 0:
        return base_lr

    if effective_step >= cosine_decay_steps:
        return base_lr * cosine_min_ratio

    import math
    progress = effective_step / cosine_decay_steps
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (cosine_min_ratio + (1.0 - cosine_min_ratio) * cosine_factor)


# ---------------------------------------------------------------------------
# Profiling / bottleneck analysis
# ---------------------------------------------------------------------------

class Profiler:
    """Simple timing profiler for identifying bottlenecks."""

    def __init__(self):
        self.times = {}
        self.counts = {}

    def time_section(self, name: str):
        """Context manager for timing a section."""
        class Timer:
            def __init__(self, profiler, section):
                self.profiler = profiler
                self.section = section

            def __enter__(self):
                self.t0 = time.time()
                return self

            def __exit__(self, *args):
                dt = time.time() - self.t0
                self.profiler.times[self.section] = self.profiler.times.get(self.section, 0) + dt
                self.profiler.counts[self.section] = self.profiler.counts.get(self.section, 0) + 1

        return Timer(self, name)

    def report(self) -> str:
        """Return a formatted report of timing breakdown."""
        lines = []
        total = sum(self.times.values())
        for name in sorted(self.times.keys(), key=lambda x: -self.times[x]):
            t = self.times[name]
            n = self.counts[name]
            pct = 100 * t / total if total > 0 else 0
            avg = t / n if n > 0 else 0
            lines.append(f"  {name:>25s}: {t:>8.3f}s ({pct:>5.1f}%)  avg={avg:.4f}s  n={n}")
        lines.append(f"  {'TOTAL':>25s}: {total:>8.3f}s")
        return "\n".join(lines)

    def reset(self):
        self.times.clear()
        self.counts.clear()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str, steps_override=None, micro_batch_override=None,
          accum_steps_override=None, profile=False,
          checkpoint_dir_override=None, resume=None, device_id=0):
    """Train the model on TT hardware using a YAML config."""
    cfg = load_config(config_path)
    cell = cfg.get("cell", "A")

    # Training hyperparams
    max_steps = steps_override if steps_override else cfg.get("max_steps", 10000)
    tokens_per_batch = cfg.get("tokens_per_batch", 250000)
    seq_len = cfg.get("seq_len", 128)
    base_lr = cfg.get("lr", 6e-4)
    warmup_steps = cfg.get("warmup_steps", 200)
    weight_decay = cfg.get("weight_decay", 0.1)
    # Cosine LR schedule (optional): decays LR from base to base*min_ratio
    # over cosine_decay_steps after warmup. Prevents gates from growing
    # unboundedly after convergence.
    cosine_decay_steps = cfg.get("cosine_decay_steps", 0)
    cosine_min_ratio = cfg.get("cosine_min_ratio", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    ws_grad_clip = cfg.get("ws_grad_clip", 0.5)
    grad_norm_spike_threshold = cfg.get("grad_norm_spike_threshold", 5000.0)
    depth_range = tuple(cfg.get("depth_range", [2, 8]))
    seed = cfg.get("seed", 42)
    eval_interval = cfg.get("eval_interval", 500)
    log_interval = cfg.get("log_interval", 50)
    checkpoint_interval = cfg.get("checkpoint_interval", 900)
    ckpt_dir = checkpoint_dir_override or cfg.get("ckpt_dir", "checkpoints")

    # Micro-batch size: from config override or auto-computed
    if micro_batch_override:
        micro_batch = micro_batch_override
    else:
        micro_batch = cfg.get("micro_batch_size", 0)
        if micro_batch == 0:
            micro_batch = 8  # default conservative

    # Compute gradient accumulation steps
    tokens_per_micro = micro_batch * seq_len
    if accum_steps_override:
        accum_steps = accum_steps_override
    else:
        accum_steps = max(1, tokens_per_batch // tokens_per_micro)
    effective_batch = micro_batch * accum_steps
    effective_tokens = effective_batch * seq_len

    print(f"=== TT-nn Training: Cell {cell} ===", flush=True)
    print(f"Config: {config_path}", flush=True)

    # Open device — always open logical device 0.
    # TT_VISIBLE_DEVICES (set at script start) remaps the physical chip
    # specified by --device to logical 0, so each process only sees its
    # own chip and there is no driver-level contention between processes.
    device = ttnn.open_device(device_id=0)
    print(f"Device: {device} (physical device {device_id} via TT_VISIBLE_DEVICES)", flush=True)

    # Build model config
    model_config = build_model_config(cfg)

    # Check for unsupported features
    # (all cells now supported)

    # Create model
    model = TTMambaWorkspaceModel(model_config, device)
    n_params = model.get_num_params()
    print(f"Model: {model_config.n_layers} layers, d_model={model_config.d_model}, "
          f"n_heads={model_config.n_heads}", flush=True)
    print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)
    print(f"bf16 param memory: {n_params * 2 / 1e6:.1f} MB", flush=True)

    # Create optimizer
    # Per-parameter LR groups: allows different LR for workspace vs backbone
    # Config format: lr_groups: {"ws_": 0.25}  → workspace params get 0.25x base LR
    lr_groups = cfg.get("lr_groups", None)
    wd_groups = cfg.get("wd_groups", None)
    beta2_groups = cfg.get("beta2_groups", None)
    optimizer = TTAdamW(model.get_params(), lr=base_lr, weight_decay=weight_decay,
                        lr_groups=lr_groups, wd_groups=wd_groups,
                        beta2_groups=beta2_groups)
    if lr_groups:
        ws_mult = lr_groups.get("ws_", 1.0)
        ws_lr = base_lr * ws_mult
        gate_mult = lr_groups.get("ws_read_gate", None)
        if gate_mult is not None:
            gate_lr = base_lr * gate_mult
            print(f"LR groups: workspace={ws_lr:.2e} ({ws_mult}x), gates={gate_lr:.2e} ({gate_mult}x), backbone={base_lr:.2e}", flush=True)
        else:
            print(f"LR groups: workspace={ws_lr:.2e} ({ws_mult}x), backbone={base_lr:.2e}", flush=True)

    # Resume from checkpoint
    start_step = 0
    if resume and os.path.exists(resume):
        opt_state = model.load_checkpoint(resume, device=device)
        if opt_state:
            optimizer.load_state(opt_state, model)
            start_step = optimizer.step_count
        print(f"Resumed from step {start_step}", flush=True)

    # Data
    vocab = Vocab()
    rng = random.Random(seed)
    # KNOWN LIMITATION: checkpoints do not save/restore `rng`'s state (see
    # save_checkpoint/load_checkpoint in model_ttnn.py), so resuming from a
    # checkpoint re-seeds `rng` from scratch rather than continuing its
    # sequence. This means the exact data batches and recurrent-core K
    # values sampled after a resume will NOT match an uninterrupted run,
    # even though both use the same `seed`. Not a correctness bug (training
    # is still valid), just not bit-for-bit reproducible across a
    # stop/resume boundary. Fixing this would mean threading `rng.getstate()`
    # through save_checkpoint/load_checkpoint's optimizer_state dict.

    # Checkpoint directory
    os.makedirs(ckpt_dir, exist_ok=True)

    # Print training config
    print(f"\nTraining config:", flush=True)
    print(f"  micro_batch={micro_batch}, accum_steps={accum_steps}, "
          f"effective_batch={effective_batch}", flush=True)
    print(f"  seq_len={seq_len}, tokens_per_batch={tokens_per_batch}", flush=True)
    print(f"  effective_tokens/step={effective_tokens}", flush=True)
    print(f"  lr={base_lr}, warmup={warmup_steps}, weight_decay={weight_decay}, "
          f"grad_clip={grad_clip}, ws_grad_clip={ws_grad_clip}, grad_spike={grad_norm_spike_threshold}", flush=True)
    print(f"  max_steps={max_steps}", flush=True)
    if profile:
        print(f"  Profiling: ENABLED", flush=True)

    # Gate schedule config
    gate_schedule_steps = cfg.get("gate_schedule_steps", 0)
    gate_init_val = cfg.get("gate_init", 0.0)
    if gate_schedule_steps > 0 and model_config.use_workspace:
        print(f"  Gate schedule: anneal from {gate_init_val} to 0.0 over {gate_schedule_steps} steps", flush=True)

    print(f"\n{'Step':>6} {'Loss':>10} {'LR':>10} {'Time':>8} {'tokens/s':>10} {'GradNorm':>10} {'Entropy':>10} {'Diversity':>10} {'gz_gr':>8} {'gz_gw':>8} {'gam_slot':>8}", flush=True)

    # Early stopping: stop if smoothed loss hasn't improved for plateau_patience steps.
    # Uses an EMA of the loss to avoid noise from individual steps, and requires
    # a relative improvement of plateau_min_delta to count as "better".
    plateau_patience = cfg.get("plateau_patience", 500)
    plateau_min_delta = cfg.get("plateau_min_delta", 1e-4)  # relative improvement
    plateau_ema_beta = 0.99  # EMA smoothing factor
    best_loss_ema = float("inf")
    loss_ema = None
    steps_since_best = 0
    if plateau_patience > 0:
        print(f"  Early stopping: patience={plateau_patience}, min_delta={plateau_min_delta}", flush=True)

    profiler = Profiler()
    total_time = 0
    total_tokens = 0
    skipped_steps = 0
    restored_steps = 0
    last_ckpt_path = None  # track last saved checkpoint for restore-on-spike

    for step in range(start_step, max_steps):
        t_step_start = time.time()

        # Update LR (warmup + optional cosine decay)
        current_lr = get_lr(step, base_lr, warmup_steps,
                            cosine_decay_steps, cosine_min_ratio)
        optimizer.set_lr(current_lr)

        # Gate schedule: force gates open before forward pass
        # This overrides the gate parameters so the workspace contributes
        # enough to generate gradient signal for learning content-addressed routing.
        # After gate_schedule_steps, the optimizer controls the gates freely.
        model.apply_gate_schedule(step, gate_schedule_steps, gate_init_val)
        # Sync master if gate schedule was applied (it modifies model params
        # directly, and the optimizer reads from the fp32 master).
        if gate_schedule_steps > 0 and step < gate_schedule_steps:
            optimizer.sync_master_from_model(model)

        # Gradient accumulation
        accum_grads = {}  # host-side accumulated gradients
        step_loss = 0.0
        step_entropy = 0.0
        step_diversity = 0.0

        for accum_idx in range(accum_steps):
            # Sample micro-batch
            if profile:
                with profiler.time_section("data_sample"):
                    input_ids, labels, task_ids = sample_batch(
                        micro_batch, seq_len, vocab, depth_range=depth_range, rng=rng
                    )
            else:
                input_ids, labels, task_ids = sample_batch(
                    micro_batch, seq_len, vocab, depth_range=depth_range, rng=rng
                )

            # Forward — sample K for recurrent core (Cell C)
            # Uses the seeded `rng` (not the global `random` module, which is
            # never seeded here) so that --seed actually makes the K schedule
            # reproducible across runs/resumes, consistent with sample_batch()
            # above.
            k_value = None
            if model_config.recurrent_core:
                k_value = rng.randint(1, model_config.k_train_max)

            if profile:
                with profiler.time_section("forward"):
                    logits = model.forward(input_ids, k_value=k_value)
            else:
                logits = model.forward(input_ids, k_value=k_value)

            # Loss + gradient
            if profile:
                with profiler.time_section("loss"):
                    loss_val, grad_logits = cross_entropy_loss(logits, labels)
            else:
                loss_val, grad_logits = cross_entropy_loss(logits, labels)

            step_loss += loss_val

            # Deallocate logits (no longer needed after loss computation)
            _safe_deallocate(logits)

            # Workspace regularizers (entropy + diversity)
            # Computes regularizer losses and stores gradients for backward
            ent_loss, div_loss = model.compute_workspace_regularizers()
            step_entropy += ent_loss
            step_diversity += div_loss

            # Backward
            if profile:
                with profiler.time_section("backward"):
                    grads = model.backward(grad_logits)
            else:
                grads = model.backward(grad_logits)

            # Deallocate grad_logits (consumed by backward)
            _safe_deallocate(grad_logits)

            # Accumulate gradients (host-side) — transfers to host, then we
            # can deallocate the device-side gradient tensors.
            if profile:
                with profiler.time_section("grad_accum"):
                    accumulate_grads(accum_grads, grads, accum_steps)
            else:
                accumulate_grads(accum_grads, grads, accum_steps)

            # Deallocate device-side gradient tensors (already copied to host)
            for g in grads.values():
                _safe_deallocate(g)

            # Clear model's forward/backward caches to free intermediate tensors
            model.clear_caches()

        # Average loss over accumulation steps
        step_loss /= accum_steps
        step_entropy /= accum_steps
        step_diversity /= accum_steps

        # Convert accumulated gradients to tt-nn tensors
        if profile:
            with profiler.time_section("grad_to_tt"):
                tt_grads = host_grads_to_tt(accum_grads, device)
        else:
            tt_grads = host_grads_to_tt(accum_grads, device)

        # Gradient clipping (component-wise: workspace vs backbone)
        grad_norm = 0.0
        if grad_clip > 0:
            if profile:
                with profiler.time_section("grad_clip"):
                    grad_norm = clip_grad_norm(tt_grads, grad_clip, ws_max_norm=ws_grad_clip)
            else:
                grad_norm = clip_grad_norm(tt_grads, grad_clip, ws_max_norm=ws_grad_clip)

        # Skip-on-spike / restore-on-spike: if the pre-clip gradient norm
        # exceeds a threshold, the model has entered an unstable region.
        # Two modes:
        #   - "skip" (default, backward compat): skip the optimizer step.
        #     This prevents the explosion from worsening but leaves the model
        #     in whatever state caused the spike — it cannot recover.
        #   - "restore": reload the last checkpoint (model + optimizer state).
        #     This gives the model a fresh start from a known-good state.
        #     The data RNG is not restored (known limitation), so the batches
        #     that triggered the spike won't reoccur in the same order.
        skip_step = grad_norm_spike_threshold > 0 and grad_norm > grad_norm_spike_threshold
        if skip_step:
            spike_action = cfg.get("spike_action", "skip")
            if spike_action == "restore" and last_ckpt_path is not None:
                print(f"  *** RESTORE step {step}: grad_norm {grad_norm:.1f} > "
                      f"threshold {grad_norm_spike_threshold:.0f} — "
                      f"restoring from {last_ckpt_path} ***", flush=True)
                # Deallocate grads before restore (checkpoint load creates new tensors)
                for g in tt_grads.values():
                    _safe_deallocate(g)
                del tt_grads, accum_grads
                gc.collect()
                opt_state = model.load_checkpoint(last_ckpt_path, device=device)
                if opt_state:
                    optimizer.load_state(opt_state, model)
                    step = optimizer.step_count  # resume from checkpoint step
                    # Re-apply LR for the restored step
                    current_lr = get_lr(step, base_lr, warmup_steps,
                                        cosine_decay_steps, cosine_min_ratio)
                    optimizer.set_lr(current_lr)
                restored_steps += 1
                skipped_steps += 1
                continue  # skip the rest of this iteration, re-run from restored step
            else:
                print(f"  *** SKIP step {step}: grad_norm {grad_norm:.1f} > "
                      f"threshold {grad_norm_spike_threshold:.0f} — skipping optimizer step ***",
                      flush=True)
                skipped_steps += 1
        else:
            # Optimizer step
            if profile:
                with profiler.time_section("optimizer"):
                    optimizer.step(tt_grads, model)
            else:
                optimizer.step(tt_grads, model)

            # Normalize workspace slot parameters after each optimizer step.
            # This prevents unbounded slot growth which causes attention sharpening
            # and gradient explosion. No-op for cells without a workspace.
            model.normalize_workspace_slots()
            # Re-sync fp32 master copies for the params that normalization touched
            # (slots + 8 workspace weight matrices).  Only sync these — syncing all
            # params would clobber the sub-ULP fp32 accumulation on gates, decay,
            # and norm weights, re-freezing them.
            if model.workspace is not None:
                _ws_norm_names = {"ws_slots"} | {
                    f"ws_{p}" for p in (
                        "read_q_weight", "read_k_weight", "read_v_weight", "read_out_weight",
                        "write_q_weight", "write_k_weight", "write_v_weight", "write_out_weight",
                    )
                }
                optimizer.sync_master_from_model(model, names=_ws_norm_names)

            # Spectral-normalize backbone qkv/out_proj weights after each step.
            # The workspace feedback loop causes exponential spectral norm growth
            # in backbone weights (layer_4 qkv reached 3.29 by step 500 in Cell B
            # vs 1.35 in Cell A).  Capping at backbone_spectral_norm_bound (3.0)
            # breaks the feedback loop while allowing healthy growth.
            model.spectral_normalize_backbone_weights()
            # Re-sync fp32 master for all backbone qkv/out_proj weights
            _backbone_norm_names = set()
            for i, wrapped in enumerate(model.layers):
                inner = wrapped.layer
                if hasattr(inner, 'qkv_weight'):
                    _backbone_norm_names.add(f"layer_{i}_qkv_weight")
                if hasattr(inner, 'out_proj_weight'):
                    _backbone_norm_names.add(f"layer_{i}_out_proj_weight")
            if _backbone_norm_names:
                optimizer.sync_master_from_model(model, names=_backbone_norm_names)

            # Clamp retention layer log-gamma to <= -0.02 (gamma <= 0.98).
            # When gamma > 1.0, the decay matrix D[t,s] = gamma^(t-s) becomes
            # exponentially growing, causing gradient explosion.  Even gamma = 1.0
            # is unstable — the O(T^2) gamma gradient dominates the clip.  The
            # -0.02 clamp ensures D decays (D[127]=0.077), reducing the gradient
            # ~10x.  This is a mathematical stability constraint, not regularization.
            # Skipped when freeze_gamma is set — gamma is not optimizer-managed
            # and stays at its init value (~0.95), so clamping is unnecessary.
            if not model_config.freeze_gamma:
                model.clamp_retention_gammas()
                # Re-sync fp32 master for any gammas that were clamped
                _gamma_names = {f"layer_{i}_gamma" for i in range(len(model.layers))
                                if hasattr(model.layers[i].layer, 'gamma')}
                if _gamma_names:
                    optimizer.sync_master_from_model(model, names=_gamma_names)

            # Clamp workspace ReZero gates to [-gate_clamp_bound, gate_clamp_bound].
            # When gates grow large negative (causal masking makes the workspace's
            # past-only contribution noise → model suppresses it), the pre-norm
            # residual cancels, making RMS very small. RMSNorm backward divides
            # by RMS, amplifying the gradient by 1/RMS → gradient explosion.
            # No-op when gate_clamp_bound is 0.0 (default / backward compat).
            model.clamp_workspace_gates()
            if model_config.gate_clamp_bound > 0 and model.workspace is not None:
                optimizer.sync_master_from_model(
                    model, names={"ws_read_gate", "ws_write_gate"})

        # Deallocate tt_grads (no longer needed after optimizer step / skip)
        for g in tt_grads.values():
            _safe_deallocate(g)
        del tt_grads
        # Free host-side accumulated gradients
        del accum_grads
        # Force garbage collection to reclaim any remaining orphaned tensors.
        # Without this, ttnn wrapper objects (tiny Python objects) don't trigger
        # GC often enough, and device DRAM grows until OOM.
        gc.collect()

        t_step_end = time.time()
        step_time = t_step_end - t_step_start
        total_time += step_time
        total_tokens += effective_tokens
        tokens_per_sec = effective_tokens / step_time

        if step < 50 or step % log_interval == 0 or step == max_steps - 1:
            ws_stats = model.get_workspace_stats()
            if ws_stats is not None:
                gate_str = f"{ws_stats['read_gate']:>8.4f} {ws_stats['write_gate']:>8.4f} {ws_stats['slot_decay']:>8.4f}"
            else:
                gate_str = f"{'-':>8} {'-':>8} {'-':>8}"
            print(f"{step:>6} {step_loss:>10.4f} {current_lr:>10.6f} "
                  f"{step_time:>7.2f}s {tokens_per_sec:>10.0f} {grad_norm:>10.4f} "
                  f"{step_entropy:>10.4f} {step_diversity:>10.4f} {gate_str}", flush=True)

        # Early stopping: track EMA of loss, stop if no improvement for plateau_patience steps.
        # Skip during warmup (LR is ramping, loss behavior is not representative).
        if plateau_patience > 0 and step >= warmup_steps:
            if loss_ema is None:
                loss_ema = step_loss
            else:
                loss_ema = plateau_ema_beta * loss_ema + (1 - plateau_ema_beta) * step_loss
            # Check for improvement (relative improvement over best)
            if loss_ema < best_loss_ema * (1 - plateau_min_delta):
                best_loss_ema = loss_ema
                steps_since_best = 0
            else:
                steps_since_best += 1
            if steps_since_best >= plateau_patience:
                print(f"\n*** Early stopping at step {step}: loss EMA plateaued "
                      f"(best={best_loss_ema:.4f}, current={loss_ema:.4f}, "
                      f"no improvement for {plateau_patience} steps) ***", flush=True)
                break

        # Checkpoint
        if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"cell_{cell}_step{step+1}.pt")
            model.save_checkpoint(ckpt_path, optimizer_state=optimizer.get_state(), step=step+1)
            last_ckpt_path = ckpt_path

        # Profile report every 100 steps during profiling
        if profile and step > 0 and (step + 1) % 100 == 0:
            print(f"\n--- Profile report (steps {step-98}-{step}) ---", flush=True)
            print(profiler.report(), flush=True)
            profiler.reset()
            print("", flush=True)

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, f"cell_{cell}_final.pt")
    model.save_checkpoint(final_path, optimizer_state=optimizer.get_state(), step=max_steps)

    # Final profile report
    if profile and profiler.counts:
        print(f"\n--- Final profile report ---", flush=True)
        print(profiler.report(), flush=True)

    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    print(f"\nTotal time: {total_time:.1f}s", flush=True)
    print(f"Avg step: {total_time/(max_steps-start_step):.2f}s", flush=True)
    print(f"Avg throughput: {avg_tokens_per_sec:.0f} tokens/sec", flush=True)
    print(f"Total tokens: {total_tokens:,}", flush=True)
    if skipped_steps > 0:
        print(f"Skipped steps (grad spike): {skipped_steps} "
              f"(restored: {restored_steps})", flush=True)
    ttnn.close_device(device)
    print("Training complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TT-nn native training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--micro_batch", type=int, default=None,
                        help="Override micro-batch size")
    parser.add_argument("--accum_steps", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-section profiling")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=int, default=0,
                        help="Tenstorrent device ID (0-3 for Quietbox 2)")
    args = parser.parse_args()

    train(
        config_path=args.config,
        steps_override=args.steps,
        micro_batch_override=args.micro_batch,
        accum_steps_override=args.accum_steps,
        profile=args.profile,
        checkpoint_dir_override=args.checkpoint_dir,
        resume=args.resume,
        device_id=args.device,
    )
