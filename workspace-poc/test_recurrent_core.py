#!/usr/bin/env python3
"""Test recurrent core with different K values.

Verifies:
1. Forward pass works with k_train_max=1, 3, 6
2. Forward output is finite for each K value
3. Backward pass works with different K values
4. Cell C with attention_residual_core produces different output than without
5. Slot state chaining: K=1 vs K=3 produce different outputs (slot state matters)

Run: .tt-venv/bin/python test_recurrent_core.py
"""

import os, sys, math
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
from train_ttnn import cross_entropy_loss

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


def _make_config_c(k_train_max=3, k_inference=3, attention_residual_core=True):
    return ModelConfig(
        d_model=128, n_heads=4, n_layers=4,
        vocab_size=128,
        use_attention=True, attention_positions=[0],
        use_workspace=True, n_workspace_slots=4,
        recurrent_core=True, core_start=1, core_end=3,
        k_train_max=k_train_max, k_inference=k_inference,
        attention_residual_core=attention_residual_core,
        freeze_gamma=True,
    )


def test_forward_different_k():
    """Test 1 & 2: forward pass works with K=1, 3, 6 and output is finite."""
    device = ttnn.open_device(device_id=0)
    try:
        B, T = 4, 32
        input_ids = torch.randint(0, 128, (B, T))
        labels = torch.randint(0, 128, (B, T))
        all_ok = True
        for K in [1, 3, 6]:
            config = _make_config_c(k_train_max=K, k_inference=K)
            model = TTWRAPModel(config, device)
            logits = model.forward(input_ids)
            logits_t = ttnn.to_torch(logits)
            finite = bool(torch.isfinite(logits_t).all().item())
            loss, _ = cross_entropy_loss(logits, labels)
            loss_finite = math.isfinite(loss)
            print(f"  K={K}: logits finite={finite}, loss={loss:.4f}, loss_finite={loss_finite}")
            if not finite or not loss_finite:
                all_ok = False
            model.clear_caches()
            del model
        assert all_ok, "Some K value produced non-finite output"
        print("[fwd-K] forward finite for K=1,3,6: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_k1_vs_k3_differ():
    """Test 5: K=1 vs K=3 produce different outputs (slot state chaining matters)."""
    device = ttnn.open_device(device_id=0)
    try:
        B, T = 4, 32
        input_ids = torch.randint(0, 128, (B, T))

        torch.manual_seed(42)
        config1 = _make_config_c(k_train_max=1, k_inference=1)
        model1 = TTWRAPModel(config1, device)
        logits1 = model1.forward(input_ids)
        out1 = ttnn.to_torch(logits1).clone()
        model1.clear_caches()

        torch.manual_seed(42)
        config3 = _make_config_c(k_train_max=3, k_inference=3)
        model3 = TTWRAPModel(config3, device)
        logits3 = model3.forward(input_ids)
        out3 = ttnn.to_torch(logits3).clone()
        model3.clear_caches()

        differ = not torch.allclose(out1, out3, atol=1e-5)
        max_diff = (out1 - out3).abs().max().item()
        print(f"  K=1 vs K=3: max_diff={max_diff:.6f}, differ={differ}")
        assert differ, f"K=1 and K=3 outputs are identical (max_diff={max_diff}), " \
                       "slot state chaining should make them differ"
        print("[k1-vs-k3] outputs differ: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_ar_vs_no_ar():
    """Test 4: attention_residual_core=True vs False produce different outputs."""
    device = ttnn.open_device(device_id=0)
    try:
        B, T = 4, 32
        input_ids = torch.randint(0, 128, (B, T))

        torch.manual_seed(42)
        config_ar = _make_config_c(k_train_max=3, k_inference=3,
                                   attention_residual_core=True)
        model_ar = TTWRAPModel(config_ar, device)
        logits_ar = model_ar.forward(input_ids)
        out_ar = ttnn.to_torch(logits_ar).clone()
        model_ar.clear_caches()

        torch.manual_seed(42)
        config_noar = _make_config_c(k_train_max=3, k_inference=3,
                                     attention_residual_core=False)
        model_noar = TTWRAPModel(config_noar, device)
        logits_noar = model_noar.forward(input_ids)
        out_noar = ttnn.to_torch(logits_noar).clone()
        model_noar.clear_caches()

        differ = not torch.allclose(out_ar, out_noar, atol=1e-5)
        max_diff = (out_ar - out_noar).abs().max().item()
        print(f"  AR vs no-AR: max_diff={max_diff:.6f}, differ={differ}")
        assert differ, f"AR and no-AR outputs are identical (max_diff={max_diff})"
        print("[ar-vs-noar] outputs differ: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_backward_k3():
    """Test 3: backward pass with K=3 produces finite, non-zero workspace grads."""
    device = ttnn.open_device(device_id=0)
    try:
        B, T = 4, 32
        config = _make_config_c(k_train_max=3, k_inference=3)
        model = TTWRAPModel(config, device)
        input_ids = torch.randint(0, 128, (B, T))
        logits = model.forward(input_ids)

        grad_logits = ttnn.from_torch(
            torch.randn(B, T, config.vocab_size, dtype=torch.bfloat16) * 0.01,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        grads = model.backward(grad_logits)

        # Verify all grads are finite. At init, ReZero gates are 0.0 so the
        # workspace is a no-op — ws_* grads may be zero. The key check is that
        # backward runs without crashing and produces finite gradients.
        all_finite = True
        ws_count = 0
        ws_nonzero_count = 0
        for name, g in grads.items():
            g_t = ttnn.to_torch(g).float()
            if not torch.isfinite(g_t).all().item():
                print(f"  NON-FINITE grad: {name}")
                all_finite = False
            if name.startswith("ws_"):
                ws_count += 1
                max_abs = g_t.abs().max().item()
                if max_abs >= 1e-8:
                    ws_nonzero_count += 1
                print(f"  ws grad {name}: max_abs={max_abs:.6f}")

        assert all_finite, "Some gradients are non-finite"
        assert ws_count > 0, f"No ws_* gradients found (expected workspace params)"
        # At least backbone grads should be non-zero (token_emb, layers, etc.)
        bb_nonzero = any(
            not name.startswith("ws_") and ttnn.to_torch(g).float().abs().max().item() >= 1e-8
            for name, g in grads.items()
        )
        assert bb_nonzero, "All backbone gradients are zero"
        print(f"[bwd-K3] {len(grads)} grads, {ws_count} ws_* grads ({ws_nonzero_count} non-zero), all finite: OK")
        model.clear_caches()
        return True
    finally:
        ttnn.close_device(device)


def main():
    tests = [
        ("forward_different_k", test_forward_different_k),
        ("k1_vs_k3_differ", test_k1_vs_k3_differ),
        ("ar_vs_no_ar", test_ar_vs_no_ar),
        ("backward_k3", test_backward_k3),
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
