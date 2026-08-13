"""Test TTWRAPModel.clear_caches() correctness.

Verifies that:
1. After forward+backward, clear_caches deallocates cached intermediates
2. Model parameters survive clear_caches (still usable for another forward pass)
3. clear_caches can be called multiple times safely
4. After clear_caches, another forward+backward works correctly

Run: .tt-venv/bin/python test_clear_caches.py
"""

import os, sys
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
# P300 mesh graph descriptor setup
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
                path = (Path(next(iter(spec.submodule_search_locations)))
                        / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name)
                if path.is_file():
                    return str(path)
            for p in sys.path:
                candidate = Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name
                if candidate.is_file():
                    return str(candidate)
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

import torch, ttnn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import ModelConfig, TTWRAPModel, _safe_deallocate

# ---------------------------------------------------------------------------
# Small test configs (kept small for fast device tests)
# ---------------------------------------------------------------------------

SMALL_CONFIG = ModelConfig(
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=128,
    use_attention=True, attention_positions=[0],
    use_workspace=True, n_workspace_slots=4,
    recurrent_core=False,
    freeze_gamma=True,
)

SMALL_CONFIG_C = ModelConfig(
    d_model=128, n_heads=4, n_layers=4,
    vocab_size=128,
    use_attention=True, attention_positions=[0],
    use_workspace=True, n_workspace_slots=4,
    recurrent_core=True, core_start=1, core_end=3,
    k_train_max=3, k_inference=3,
    attention_residual_core=True,
    freeze_gamma=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(config, B=4, T=32):
    return torch.randint(0, config.vocab_size, (B, T))


def _assert_finite(logits_tt, label=""):
    logits_torch = ttnn.to_torch(logits_tt)
    assert torch.isfinite(logits_torch).all().item(), \
        f"{label}: logits contain NaN or Inf"
    return logits_torch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_caches_cleared_after_forward(device):
    """Test 1: After forward, caches are populated; after clear_caches, empty."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)

    input_ids = _make_input(config)
    logits = model.forward(input_ids)
    ttnn.synchronize_device(device)

    # Verify caches are populated after forward
    assert model._cached_x_pre_final_norm is not None, \
        "_cached_x_pre_final_norm is None after forward (expected populated)"
    assert len(model._fwd_trace) > 0, \
        "_fwd_trace is empty after forward (expected populated)"
    print(f"  After forward: _cached_x_pre_final_norm is not None, "
          f"_fwd_trace has {len(model._fwd_trace)} entries")

    # Call clear_caches
    model.clear_caches()

    # Verify caches are cleared
    assert model._cached_x_pre_final_norm is None, \
        "_cached_x_pre_final_norm is not None after clear_caches (expected None)"
    assert model._fwd_trace == [], \
        f"_fwd_trace is {model._fwd_trace} after clear_caches (expected [])"
    print("  After clear_caches: _cached_x_pre_final_norm is None, _fwd_trace is []")

    _safe_deallocate(logits)
    del model


def test_params_survive_clear_caches(device):
    """Test 2: After clear_caches, model parameters are still valid (finite)."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)

    input_ids = _make_input(config)
    logits = model.forward(input_ids)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)
    model.clear_caches()

    # Verify all params are finite
    params = model.get_params()
    assert len(params) > 0, "No params returned from get_params()"
    non_finite = []
    for name, p in params.items():
        p_torch = ttnn.to_torch(p)
        if not torch.isfinite(p_torch).all().item():
            non_finite.append(name)
    assert not non_finite, \
        f"Parameters with NaN/Inf after clear_caches: {non_finite}"
    print(f"  All {len(params)} parameters are finite after clear_caches")

    del model


def test_forward_after_clear(device):
    """Test 3: Forward after clear_caches produces finite logits."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)

    input_ids = _make_input(config)

    # First forward + clear
    logits1 = model.forward(input_ids)
    ttnn.synchronize_device(device)
    _assert_finite(logits1, "first forward")
    _safe_deallocate(logits1)
    model.clear_caches()

    # Second forward after clear
    logits2 = model.forward(input_ids)
    ttnn.synchronize_device(device)
    _assert_finite(logits2, "second forward after clear_caches")
    _safe_deallocate(logits2)
    model.clear_caches()

    print("  Forward after clear_caches produces finite logits")
    del model


def test_multiple_rounds(device):
    """Test 4: 3 rounds of forward+backward+clear_caches work without crash."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)

    input_ids = _make_input(config)

    for i in range(3):
        logits = model.forward(input_ids)
        ttnn.synchronize_device(device)
        _assert_finite(logits, f"round {i+1} forward")

        # Backward with random grad
        B, T, V = input_ids.shape[0], input_ids.shape[1], config.vocab_size
        grad_logits = ttnn.from_torch(
            torch.randn(B, T, V, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)

        # Cleanup
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        _safe_deallocate(logits)
        model.clear_caches()
        print(f"  Round {i+1}: forward+backward+clear_caches OK, logits finite")

    del model


def test_cell_c_clear_caches(device):
    """Test 5: Cell C (recurrent core) caches are cleared properly."""
    config = SMALL_CONFIG_C
    model = TTWRAPModel(config, device)

    input_ids = _make_input(config)
    logits = model.forward(input_ids)
    ttnn.synchronize_device(device)

    # Verify recurrent core caches are populated after forward
    assert len(model._core_x_outputs) > 0, \
        f"_core_x_outputs is empty after forward (expected populated, config has recurrent_core=True)"
    assert len(model._core_blend_info) > 0, \
        f"_core_blend_info is empty after forward (expected populated)"
    print(f"  After forward: _core_x_outputs has {len(model._core_x_outputs)} entries, "
          f"_core_blend_info has {len(model._core_blend_info)} entries")

    # Backward
    B, T, V = input_ids.shape[0], input_ids.shape[1], config.vocab_size
    grad_logits = ttnn.from_torch(
        torch.randn(B, T, V, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)

    # Cleanup backward intermediates
    _safe_deallocate(grad_logits)
    for g in grads.values():
        _safe_deallocate(g)
    _safe_deallocate(logits)

    # Clear caches
    model.clear_caches()

    # Verify recurrent core caches are cleared
    assert model._core_x_outputs == [], \
        f"_core_x_outputs is {model._core_x_outputs} after clear_caches (expected [])"
    assert model._core_blend_info == [], \
        f"_core_blend_info is {model._core_blend_info} after clear_caches (expected [])"
    assert model._cached_x_pre_final_norm is None, \
        "_cached_x_pre_final_norm is not None after clear_caches (expected None)"
    assert model._fwd_trace == [], \
        f"_fwd_trace is {model._fwd_trace} after clear_caches (expected [])"
    print("  After clear_caches: _core_x_outputs=[], _core_blend_info=[], "
          "_cached_x_pre_final_norm=None, _fwd_trace=[]")

    del model


def test_clear_caches_idempotent(device):
    """Test 3b: clear_caches can be called multiple times safely."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)

    input_ids = _make_input(config)
    logits = model.forward(input_ids)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)

    # Call clear_caches 3 times — should not crash
    model.clear_caches()
    model.clear_caches()
    model.clear_caches()

    assert model._cached_x_pre_final_norm is None, \
        "_cached_x_pre_final_norm is not None after 3x clear_caches"
    assert model._fwd_trace == [], \
        f"_fwd_trace is {model._fwd_trace} after 3x clear_caches (expected [])"
    print("  clear_caches called 3x safely, caches remain cleared")

    del model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_test(name, test_fn, device):
    try:
        test_fn(device)
        print(f"PASS: {name}")
        return True
    except AssertionError as e:
        print(f"FAIL: {name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {name}: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("clear_caches() Correctness Tests")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)

    results = []
    results.append(run_test("Caches cleared after forward", test_caches_cleared_after_forward, device))
    results.append(run_test("Parameters survive clear_caches", test_params_survive_clear_caches, device))
    results.append(run_test("Forward works after clear_caches", test_forward_after_clear, device))
    results.append(run_test("clear_caches is idempotent (multiple calls safe)", test_clear_caches_idempotent, device))
    results.append(run_test("3 rounds of fwd+bwd+clear_caches", test_multiple_rounds, device))
    results.append(run_test("Cell C recurrent core caches cleared", test_cell_c_clear_caches, device))

    ttnn.close_device(device)

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"{total - passed} TEST(S) FAILED ({passed}/{total} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
