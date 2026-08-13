#!/usr/bin/env python3
"""Test TTWRAPModel parameter get/set round-trip and weight tying.

Verifies:
1. get_params -> set_params round-trip preserves all parameter values
2. Weight tying: token_emb_weight and lm_head_weight are the same tensor
3. Weight tying preserved after set_params
4. All param names in get_params are unique and non-empty

Run: .tt-venv/bin/python test_params.py
"""

import os, sys
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
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
from model_ttnn import ModelConfig, TTWRAPModel

# Cell B (workspace, no recurrent core)
SMALL_CONFIG_B = ModelConfig(
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=128,
    use_attention=True, attention_positions=[0],
    use_workspace=True, n_workspace_slots=4,
    recurrent_core=False,
    freeze_gamma=True,
)
# Cell C (recurrent core with attention residual)
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


def test_roundtrip_and_tying():
    """Test 1 & 2 & 3: round-trip preserves values; weight tying holds."""
    device = ttnn.open_device(device_id=0)
    try:
        model = TTWRAPModel(SMALL_CONFIG_B, device)
        params = model.get_params()

        # --- Test 4: all param names unique and non-empty ---
        names = list(params.keys())
        assert all(isinstance(n, str) and len(n) > 0 for n in names), \
            "All param names must be non-empty strings"
        assert len(names) == len(set(names)), \
            f"Duplicate param names: {[n for n in names if names.count(n) > 1]}"
        print(f"[names] {len(names)} unique non-empty param names: {names}")

        # --- Save torch copies of every param ---
        torch_copies = {}
        for name, tt_param in params.items():
            torch_copies[name] = ttnn.to_torch(tt_param).clone()

        # --- Test 2: weight tying (same Python object) ---
        assert model.token_emb_weight is model.lm_head_weight, \
            "Weight tying broken: token_emb_weight is not lm_head_weight"
        print("[tied] token_emb_weight is lm_head_weight (same object): OK")

        # --- Re-set each param via _set_param (round-trip) ---
        for name, tt_param in params.items():
            new_tt = ttnn.from_torch(
                ttnn.to_torch(tt_param).clone(),
                dtype=tt_param.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            model._set_param(name, new_tt)

        # --- Test 3: weight tying preserved after set_params ---
        assert model.token_emb_weight is model.lm_head_weight, \
            "Weight tying broken after _set_param: token_emb_weight is not lm_head_weight"
        print("[tied] weight tying preserved after _set_param: OK")

        # --- Test 1: all params match the saved copies ---
        params_after = model.get_params()
        all_match = True
        for name, tt_param in params_after.items():
            ref = torch_copies[name]
            cur = ttnn.to_torch(tt_param)
            if not torch.allclose(ref, cur, atol=1e-5):
                max_diff = (ref - cur).abs().max().item()
                print(f"  MISMATCH {name}: max_diff={max_diff:.3e}")
                all_match = False
            else:
                print(f"  match {name}: OK")
        assert all_match, "Round-trip did not preserve all parameter values"
        print("[roundtrip] all params preserved after get->set->get: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_expected_keys_cell_b():
    """Test 3: get_params returns token_emb_weight, norm_weight, plus layer_* keys."""
    device = ttnn.open_device(device_id=0)
    try:
        model = TTWRAPModel(SMALL_CONFIG_B, device)
        params = model.get_params()
        names = set(params.keys())
        assert "token_emb_weight" in names, "Missing token_emb_weight"
        assert "norm_weight" in names, "Missing norm_weight"
        layer_keys = [n for n in names if n.startswith("layer_")]
        assert len(layer_keys) == SMALL_CONFIG_B.n_layers or len(layer_keys) > 0, \
            f"Expected layer_* keys, got {layer_keys}"
        print(f"[keys-B] token_emb_weight, norm_weight present; {len(layer_keys)} layer_* keys")
        return True
    finally:
        ttnn.close_device(device)


def test_expected_keys_cell_c():
    """Test 4: Cell C get_params includes ws_* params (workspace + attention residual)."""
    device = ttnn.open_device(device_id=0)
    try:
        model = TTWRAPModel(SMALL_CONFIG_C, device)
        params = model.get_params()
        names = set(params.keys())
        ws_keys = [n for n in names if n.startswith("ws_")]
        assert len(ws_keys) > 0, f"Cell C should have ws_* params, got {names}"
        # Attention residual contributes ws_ar_query / ws_ar_scale
        assert "ws_ar_query" in names, f"Missing ws_ar_query (attention residual), got {ws_keys}"
        assert "ws_ar_scale" in names, f"Missing ws_ar_scale (attention residual), got {ws_keys}"
        print(f"[keys-C] {len(ws_keys)} ws_* keys: {sorted(ws_keys)}")
        return True
    finally:
        ttnn.close_device(device)


def main():
    tests = [
        ("roundtrip_and_tying", test_roundtrip_and_tying),
        ("expected_keys_cell_b", test_expected_keys_cell_b),
        ("expected_keys_cell_c", test_expected_keys_cell_c),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            ok = fn()
            if ok:
                print(f"PASS: {name}")
                passed += 1
            else:
                print(f"FAIL: {name}")
                failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {name}: {e}")
            failed += 1
    print(f"\n=== Summary: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
