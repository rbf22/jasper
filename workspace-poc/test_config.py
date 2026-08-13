"""Tests for build_model_config — config parsing and ModelConfig construction.

CPU-only (no device required). Run with:
    .tt-venv/bin/python -m pytest test_config.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import yaml
from model_ttnn import ModelConfig
from train_ttnn import build_model_config

# Directory containing this file (project root).
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT, "configs")


def _load_yaml(name):
    """Load a YAML config file from configs/ and return its dict."""
    with open(os.path.join(CONFIGS_DIR, name), "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# build_model_config defaults
# ---------------------------------------------------------------------------

def test_defaults():
    """Empty dict returns ModelConfig with all build_model_config defaults."""
    cfg = build_model_config({})
    assert isinstance(cfg, ModelConfig)
    # Fields explicitly defaulted by build_model_config (line 146 of train_ttnn.py)
    assert cfg.d_model == 384
    assert cfg.n_layers == 14
    assert cfg.vocab_size == 128
    assert cfg.n_heads == 4
    assert cfg.use_attention is False
    assert cfg.attention_positions == [5, 10]
    assert cfg.use_workspace is False
    assert cfg.n_workspace_slots == 16
    assert cfg.recurrent_core is False
    assert cfg.core_start == 6
    assert cfg.core_end == 10
    assert cfg.k_train_max == 3
    assert cfg.k_inference == 6
    assert cfg.attention_residual_core is False
    assert cfg.use_gradient_checkpointing is False
    assert cfg.spectral_norm_bound == 5.0
    assert cfg.backbone_spectral_norm_bound == 2.0
    assert cfg.chain_scale_safety == 1.0
    assert cfg.freeze_gamma is False
    assert cfg.freeze_slot_decay is False
    assert cfg.ws_entropy_weight == 0.0
    assert cfg.ws_diversity_weight == 0.0
    assert cfg.gate_init == 0.0
    assert cfg.slot_decay_init == 1.0
    assert cfg.slot_permutation is False
    assert cfg.gate_schedule_steps == 0
    assert cfg.gate_clamp_bound == 0.0
    assert cfg.short_conv is False
    assert cfg.short_conv_kernel == 3
    assert cfg.per_channel_decay is False


def test_partial_config():
    """Partial dict only overrides specified fields, rest are defaults."""
    cfg = build_model_config({"d_model": 256, "n_layers": 6, "use_workspace": True})
    assert cfg.d_model == 256
    assert cfg.n_layers == 6
    assert cfg.use_workspace is True
    # Untouched fields keep build_model_config defaults
    assert cfg.vocab_size == 128
    assert cfg.n_heads == 4
    assert cfg.use_attention is False
    assert cfg.attention_positions == [5, 10]
    assert cfg.n_workspace_slots == 16
    assert cfg.recurrent_core is False
    assert cfg.k_train_max == 3
    assert cfg.k_inference == 6
    assert cfg.freeze_gamma is False
    assert cfg.gate_clamp_bound == 0.0


def test_full_config():
    """All build_model_config fields specified, verify each mapped correctly."""
    raw = {
        "d_model": 512,
        "n_layers": 20,
        "vocab_size": 256,
        "n_heads": 8,
        "use_attention": True,
        "attention_positions": [3, 7, 11],
        "use_workspace": True,
        "n_workspace_slots": 32,
        "recurrent_core": True,
        "core_start": 5,
        "core_end": 15,
        "k_train_max": 4,
        "k_inference": 8,
        "attention_residual_core": True,
        "use_gradient_checkpointing": True,
        "spectral_norm_bound": 3.0,
        "backbone_spectral_norm_bound": 1.5,
        "chain_scale_safety": 0.5,
        "freeze_gamma": True,
        "freeze_slot_decay": True,
        "ws_entropy_weight": 0.01,
        "ws_diversity_weight": 0.02,
        "gate_init": -2.0,
        "slot_decay_init": 0.95,
        "slot_permutation": True,
        "gate_schedule_steps": 100,
        "gate_clamp_bound": 0.3,
        "short_conv": True,
        "short_conv_kernel": 5,
        "per_channel_decay": True,
    }
    cfg = build_model_config(raw)
    assert cfg.d_model == 512
    assert cfg.n_layers == 20
    assert cfg.vocab_size == 256
    assert cfg.n_heads == 8
    assert cfg.use_attention is True
    assert cfg.attention_positions == [3, 7, 11]
    assert cfg.use_workspace is True
    assert cfg.n_workspace_slots == 32
    assert cfg.recurrent_core is True
    assert cfg.core_start == 5
    assert cfg.core_end == 15
    assert cfg.k_train_max == 4
    assert cfg.k_inference == 8
    assert cfg.attention_residual_core is True
    assert cfg.use_gradient_checkpointing is True
    assert cfg.spectral_norm_bound == 3.0
    assert cfg.backbone_spectral_norm_bound == 1.5
    assert cfg.chain_scale_safety == 0.5
    assert cfg.freeze_gamma is True
    assert cfg.freeze_slot_decay is True
    assert cfg.ws_entropy_weight == 0.01
    assert cfg.ws_diversity_weight == 0.02
    assert cfg.gate_init == -2.0
    assert cfg.slot_decay_init == 0.95
    assert cfg.slot_permutation is True
    assert cfg.gate_schedule_steps == 100
    assert cfg.gate_clamp_bound == 0.3
    assert cfg.short_conv is True
    assert cfg.short_conv_kernel == 5
    assert cfg.per_channel_decay is True


# ---------------------------------------------------------------------------
# YAML config files
# ---------------------------------------------------------------------------

def test_cell_a_yaml():
    """configs/cell_a_tt.yaml: use_attention=True, use_workspace=False."""
    cfg = build_model_config(_load_yaml("cell_a_tt.yaml"))
    assert cfg.use_attention is True
    assert cfg.use_workspace is False
    # Cell A is the control — no recurrent core either
    assert cfg.recurrent_core is False
    assert cfg.attention_positions == [5, 10]


def test_cell_b_yaml():
    """configs/cell_b_tt.yaml: use_workspace=True."""
    cfg = build_model_config(_load_yaml("cell_b_tt.yaml"))
    assert cfg.use_workspace is True
    assert cfg.use_attention is True
    assert cfg.n_workspace_slots == 16
    # Cell B has no recurrent core
    assert cfg.recurrent_core is False
    # Cell B clamps gates
    assert cfg.gate_clamp_bound == 0.3


def test_cell_c_yaml():
    """configs/cell_c_attn_residual.yaml: recurrent_core + attention_residual_core."""
    cfg = build_model_config(_load_yaml("cell_c_attn_residual.yaml"))
    assert cfg.recurrent_core is True
    assert cfg.attention_residual_core is True
    assert cfg.use_workspace is True
    assert cfg.use_attention is True
    assert cfg.core_start == 6
    assert cfg.core_end == 10
    assert cfg.k_train_max == 6
    assert cfg.k_inference == 6


# ---------------------------------------------------------------------------
# Edge cases & derived properties
# ---------------------------------------------------------------------------

def test_unknown_keys_ignored():
    """Dict with extra unknown keys doesn't crash (build_model_config uses .get)."""
    raw = {
        "d_model": 192,
        "cell": "Z",               # not a ModelConfig field
        "lr": 1e-4,                # training field, not a ModelConfig field
        "bogus_key": [1, 2, 3],    # completely unknown
    }
    cfg = build_model_config(raw)
    assert cfg.d_model == 192
    # Unknown keys are silently ignored — no AttributeError, no crash
    assert not hasattr(cfg, "cell")
    assert not hasattr(cfg, "lr")
    assert not hasattr(cfg, "bogus_key")


def test_d_inner_property():
    """d_model=384, expand=4 -> d_inner=1536."""
    cfg = build_model_config({"d_model": 384})
    assert cfg.expand == 4
    assert cfg.d_inner == 384 * 4
    assert cfg.d_inner == 1536


def test_d_head_property():
    """d_inner=1536, n_heads=4 -> d_head=384."""
    cfg = build_model_config({"d_model": 384, "n_heads": 4})
    assert cfg.d_inner == 1536
    assert cfg.d_head == 1536 // 4
    assert cfg.d_head == 384


def test_gate_clamp_bound_default():
    """Default gate_clamp_bound is 0.0 (disabled)."""
    cfg = build_model_config({})
    assert cfg.gate_clamp_bound == 0.0
    # Explicitly disabled
    cfg2 = build_model_config({"gate_clamp_bound": 0.0})
    assert cfg2.gate_clamp_bound == 0.0
    # Explicitly enabled
    cfg3 = build_model_config({"gate_clamp_bound": 0.3})
    assert cfg3.gate_clamp_bound == 0.3
