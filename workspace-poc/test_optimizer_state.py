"""Test TTAdamW optimizer state save/load round-trip.

Verifies that optimizer state (exp_avg, exp_avg_sq, master copies, step_count, lr)
is correctly saved via get_state() and restored via load_state().

Run: .tt-venv/bin/python test_optimizer_state.py
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
from train_ttnn import TTAdamW, build_model_config, cross_entropy_loss

# ---------------------------------------------------------------------------
# Small test config (kept small for fast device tests)
# ---------------------------------------------------------------------------

SMALL_CONFIG = ModelConfig(
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=128,
    use_attention=True, attention_positions=[0],
    use_workspace=True, n_workspace_slots=4,
    recurrent_core=False,
    freeze_gamma=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_train_step(model, optimizer, device, config):
    """Run one forward + backward + optimizer step cycle."""
    B, T = 4, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    logits = model.forward(input_ids)
    ttnn.synchronize_device(device)
    labels = torch.randint(0, config.vocab_size, (B, T))
    loss, grad_logits = cross_entropy_loss(logits, labels)
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)
    optimizer.step(grads, model)
    # Normalize workspace slots after step (matches training loop pattern)
    model.normalize_workspace_slots()
    # Cleanup intermediate tensors
    _safe_deallocate(grad_logits)
    for g in grads.values():
        _safe_deallocate(g)
    _safe_deallocate(logits)
    model.clear_caches()
    return loss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_state_structure(device):
    """Test 1: After 3 steps, get_state() contains correct structure."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)
    optimizer = TTAdamW(model.get_params(), lr=6e-4)

    for i in range(3):
        _run_train_step(model, optimizer, device, config)

    state = optimizer.get_state()

    # Verify step_count
    assert state["step_count"] == 3, \
        f"Expected step_count=3, got {state['step_count']}"

    # Verify exp_avg, exp_avg_sq, master dicts have same keys as model params
    param_names = set(model.get_params().keys())
    exp_avg_keys = set(state["exp_avg"].keys())
    exp_avg_sq_keys = set(state["exp_avg_sq"].keys())
    master_keys = set(state["master"].keys())

    assert exp_avg_keys == param_names, \
        f"exp_avg keys mismatch: missing={param_names - exp_avg_keys}, extra={exp_avg_keys - param_names}"
    assert exp_avg_sq_keys == param_names, \
        f"exp_avg_sq keys mismatch: missing={param_names - exp_avg_sq_keys}, extra={exp_avg_sq_keys - param_names}"
    assert master_keys == param_names, \
        f"master keys mismatch: missing={param_names - master_keys}, extra={master_keys - param_names}"

    # Verify exp_avg/exp_avg_sq are not all zeros (steps were run)
    nonzero_m = sum(1 for v in state["exp_avg"].values() if v.abs().max().item() > 0)
    nonzero_v = sum(1 for v in state["exp_avg_sq"].values() if v.abs().max().item() > 0)
    assert nonzero_m > 0, "exp_avg is all zeros — steps didn't populate state"
    assert nonzero_v > 0, "exp_avg_sq is all zeros — steps didn't populate state"

    print(f"  step_count={state['step_count']}, {len(param_names)} params, "
          f"exp_avg/exp_avg_sq/master keys match model params")
    print(f"  {nonzero_m} exp_avg tensors non-zero, {nonzero_v} exp_avg_sq tensors non-zero")

    del model, optimizer
    return state


def test_state_roundtrip(device):
    """Test 2: Save optimizer state, load into new optimizer, verify match."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)
    optimizer = TTAdamW(model.get_params(), lr=6e-4)

    for i in range(3):
        _run_train_step(model, optimizer, device, config)

    state1 = optimizer.get_state()

    # Create a new optimizer with the same model's params
    optimizer2 = TTAdamW(model.get_params(), lr=6e-4)
    optimizer2.load_state(state1, model)
    state2 = optimizer2.get_state()

    # Verify step_count and lr match
    assert state2["step_count"] == state1["step_count"], \
        f"step_count mismatch: {state2['step_count']} vs {state1['step_count']}"
    assert state2["lr"] == state1["lr"], \
        f"lr mismatch: {state2['lr']} vs {state1['lr']}"

    # Compare exp_avg, exp_avg_sq, master tensors
    for name in state1["exp_avg"]:
        if not torch.allclose(state1["exp_avg"][name], state2["exp_avg"][name], atol=1e-6):
            max_diff = (state1["exp_avg"][name].float() - state2["exp_avg"][name].float()).abs().max().item()
            raise AssertionError(f"exp_avg[{name}] mismatch: max_diff={max_diff:.2e}")
    for name in state1["exp_avg_sq"]:
        if not torch.allclose(state1["exp_avg_sq"][name], state2["exp_avg_sq"][name], atol=1e-6):
            max_diff = (state1["exp_avg_sq"][name].float() - state2["exp_avg_sq"][name].float()).abs().max().item()
            raise AssertionError(f"exp_avg_sq[{name}] mismatch: max_diff={max_diff:.2e}")
    for name in state1["master"]:
        if not torch.allclose(state1["master"][name], state2["master"][name], atol=1e-6):
            max_diff = (state1["master"][name].float() - state2["master"][name].float()).abs().max().item()
            raise AssertionError(f"master[{name}] mismatch: max_diff={max_diff:.2e}")

    print(f"  step_count={state2['step_count']}, lr={state2['lr']} — both match original")
    print(f"  exp_avg, exp_avg_sq, master: all {len(state1['exp_avg'])} tensors match")

    del model, optimizer, optimizer2


def test_fp32_master_preservation(device):
    """Test 3: fp32 master tensors are fp32 after load_state."""
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)
    optimizer = TTAdamW(model.get_params(), lr=6e-4)

    for i in range(3):
        _run_train_step(model, optimizer, device, config)

    state = optimizer.get_state()

    # Verify master tensors in saved state are fp32
    for name, master_t in state["master"].items():
        assert master_t.dtype == torch.float32, \
            f"master[{name}] dtype is {master_t.dtype}, expected torch.float32"

    # Load into new optimizer and verify on-device master is fp32
    optimizer2 = TTAdamW(model.get_params(), lr=6e-4)
    optimizer2.load_state(state, model)

    for name, master_tt in optimizer2.master.items():
        assert master_tt.dtype == ttnn.float32, \
            f"master[{name}] on-device dtype is {master_tt.dtype}, expected ttnn.float32"

    print(f"  All {len(state['master'])} master tensors are fp32 (torch.float32 on host, ttnn.float32 on device)")

    del model, optimizer, optimizer2


def test_step_count_restored(device):
    """Test 4: step_count is restored correctly after load_state.

    Run 3 steps, save state, load into new optimizer, verify step_count=3.
    """
    config = SMALL_CONFIG
    model = TTWRAPModel(config, device)
    optimizer = TTAdamW(model.get_params(), lr=6e-4)

    for i in range(3):
        _run_train_step(model, optimizer, device, config)

    assert optimizer.step_count == 3, \
        f"Original optimizer step_count={optimizer.step_count}, expected 3"

    state = optimizer.get_state()

    # Create fresh optimizer (step_count starts at 0)
    optimizer2 = TTAdamW(model.get_params(), lr=6e-4)
    assert optimizer2.step_count == 0, \
        f"New optimizer step_count={optimizer2.step_count}, expected 0 before load"

    optimizer2.load_state(state, model)
    assert optimizer2.step_count == 3, \
        f"Restored optimizer step_count={optimizer2.step_count}, expected 3"

    print(f"  Original step_count=3, new optimizer starts at 0, after load_state step_count={optimizer2.step_count}")

    del model, optimizer, optimizer2


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
    print("Optimizer State Save/Load Tests")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)

    results = []
    results.append(run_test("Optimizer state structure after 3 steps", test_state_structure, device))
    results.append(run_test("Optimizer state round-trip", test_state_roundtrip, device))
    results.append(run_test("fp32 master preservation", test_fp32_master_preservation, device))
    results.append(run_test("step_count restoration", test_step_count_restored, device))

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
