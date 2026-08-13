"""Test checkpoint save/load round-trip for TTWRAPModel.

Verifies that saving a model checkpoint and loading it back restores
all parameters to identical values. Tests with Cell A (no workspace),
Cell B (workspace), and Cell C (recurrent core) configs.

Run: .tt-venv/bin/python test_checkpoint_roundtrip.py
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

SMALL_CONFIG_A = ModelConfig(
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=128,
    use_attention=True, attention_positions=[0],
    use_workspace=False,
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

def _compare_params(params1, params2, label=""):
    """Compare two param dicts (name -> ttnn.Tensor) by converting to torch."""
    assert set(params1.keys()) == set(params2.keys()), \
        f"{label}: param key mismatch: {set(params1.keys()) ^ set(params2.keys())}"
    mismatches = []
    for name in params1:
        p1 = ttnn.to_torch(params1[name])
        p2 = ttnn.to_torch(params2[name])
        if not torch.allclose(p1, p2, atol=1e-5):
            max_diff = (p1.float() - p2.float()).abs().max().item()
            mismatches.append((name, max_diff))
    assert not mismatches, \
        f"{label}: {len(mismatches)} param(s) differ: " + \
        ", ".join(f"{n} (max_diff={d:.2e})" for n, d in mismatches[:5])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cell_b_roundtrip(device):
    """Test 1: Cell B (workspace) checkpoint round-trip preserves all params."""
    ckpt_path = "/tmp/test_ckpt_roundtrip_cellB.pt"
    model1 = TTWRAPModel(SMALL_CONFIG, device)
    params1 = model1.get_params()
    assert len(params1) > 0, "No params returned from get_params()"
    model1.save_checkpoint(ckpt_path, step=0)

    model2 = TTWRAPModel(SMALL_CONFIG, device)
    opt_state = model2.load_checkpoint(ckpt_path, device=device)
    params2 = model2.get_params()

    _compare_params(params1, params2, label="Cell B")
    print(f"  Cell B: {len(params1)} params compared, all match")

    # cleanup
    del model1, model2
    os.remove(ckpt_path)


def test_cell_a_roundtrip(device):
    """Test 2: Cell A (no workspace) checkpoint round-trip preserves all params."""
    ckpt_path = "/tmp/test_ckpt_roundtrip_cellA.pt"
    model1 = TTWRAPModel(SMALL_CONFIG_A, device)
    params1 = model1.get_params()
    assert len(params1) > 0, "No params returned from get_params()"
    model1.save_checkpoint(ckpt_path, step=0)

    model2 = TTWRAPModel(SMALL_CONFIG_A, device)
    model2.load_checkpoint(ckpt_path, device=device)
    params2 = model2.get_params()

    _compare_params(params1, params2, label="Cell A")
    print(f"  Cell A: {len(params1)} params compared, all match")

    del model1, model2
    os.remove(ckpt_path)


def test_cell_c_roundtrip(device):
    """Test 3: Cell C (recurrent core) checkpoint round-trip preserves all params."""
    ckpt_path = "/tmp/test_ckpt_roundtrip_cellC.pt"
    model1 = TTWRAPModel(SMALL_CONFIG_C, device)
    params1 = model1.get_params()
    assert len(params1) > 0, "No params returned from get_params()"
    model1.save_checkpoint(ckpt_path, step=0)

    model2 = TTWRAPModel(SMALL_CONFIG_C, device)
    model2.load_checkpoint(ckpt_path, device=device)
    params2 = model2.get_params()

    _compare_params(params1, params2, label="Cell C")
    print(f"  Cell C: {len(params1)} params compared, all match")

    del model1, model2
    os.remove(ckpt_path)


def test_weight_tying_preserved(device):
    """Test 4: Weight tying is preserved after load_checkpoint.

    model.token_emb_weight and model.lm_head_weight should be the same
    ttnn tensor object after load_checkpoint.
    """
    ckpt_path = "/tmp/test_ckpt_weight_tying.pt"
    model = TTWRAPModel(SMALL_CONFIG, device)
    model.save_checkpoint(ckpt_path, step=0)

    model2 = TTWRAPModel(SMALL_CONFIG, device)
    model2.load_checkpoint(ckpt_path, device=device)

    assert model2.token_emb_weight is model2.lm_head_weight, \
        "Weight tying broken: token_emb_weight and lm_head_weight are different objects"
    print("  token_emb_weight is lm_head_weight: True (same ttnn tensor object)")

    del model, model2
    os.remove(ckpt_path)


def test_step_counter_preserved(device):
    """Test 5: Step counter is preserved in checkpoint.

    save_checkpoint with step=42 should store step=42 in the checkpoint file.
    """
    ckpt_path = "/tmp/test_ckpt_step.pt"
    model = TTWRAPModel(SMALL_CONFIG, device)
    model.save_checkpoint(ckpt_path, step=42)

    # Load raw checkpoint to verify step field
    checkpoint = torch.load(ckpt_path, weights_only=False)
    saved_step = checkpoint.get("step", None)
    assert saved_step == 42, f"Step mismatch: expected 42, got {saved_step}"
    print(f"  Saved step=42, loaded step={saved_step}")

    del model
    os.remove(ckpt_path)


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
    print("Checkpoint Round-Trip Tests")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)

    results = []
    results.append(run_test("Cell B checkpoint roundtrip", test_cell_b_roundtrip, device))
    results.append(run_test("Cell A checkpoint roundtrip", test_cell_a_roundtrip, device))
    results.append(run_test("Cell C checkpoint roundtrip", test_cell_c_roundtrip, device))
    results.append(run_test("Weight tying preserved after load", test_weight_tying_preserved, device))
    results.append(run_test("Step counter preserved", test_step_counter_preserved, device))

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
