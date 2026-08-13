#!/usr/bin/env python3
"""Test clip_grad_norm component-wise clipping correctness.

Verifies:
1. Global clipping (ws_max_norm=None): all grads clipped to max_norm
2. Component-wise clipping (ws_max_norm set): ws_ and backbone grads clipped independently
3. Gamma scaling: gamma grads scaled by 1/gamma_scale before norm computation
4. Returned norm is the pre-clip global norm
5. No clipping when norms are below thresholds

Run: .tt-venv/bin/python test_clip_grad_norm.py
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
from train_ttnn import clip_grad_norm

# Tile-aligned size (32x32) required for TILE_LAYOUT device tensors.
N = 32


def _make_grads(device, specs):
    """specs: list of (name, value) -> dict of name -> tt tensor (ones * value)."""
    grads = {}
    for name, val in specs:
        t = torch.ones(N, N, dtype=torch.bfloat16) * val
        grads[name] = ttnn.from_torch(t, dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=device)
    return grads


def _norm_of_grad(grads, name, gamma_scale=None):
    """Compute the L2 norm of a single grad tensor (host-side)."""
    t = ttnn.to_torch(grads[name]).float()
    if gamma_scale is not None and name.endswith("_gamma"):
        t = t / gamma_scale
    return float(t.norm().item())


def test_global_clip():
    """Test 1: global clipping (ws_max_norm=None) clips all grads to max_norm."""
    device = ttnn.open_device(device_id=0)
    try:
        grads = _make_grads(device, [("layer_0_weight", 10.0)])
        # Pre-clip norm = sqrt(32*32 * 100) = sqrt(102400) = 320
        pre_norm = math.sqrt(N * N * 100.0)
        ret = clip_grad_norm(grads, max_norm=1.0, ws_max_norm=None)
        assert ret > 1.0, f"Returned norm {ret} should be > 1.0 (pre-clip)"
        print(f"[global] returned pre-clip norm={ret:.4f} (expected ~{pre_norm:.1f})")
        # After clipping: max value = 10 * (1.0 / 320) ≈ 0.03125
        max_val = ttnn.to_torch(grads["layer_0_weight"]).float().max().item()
        assert max_val <= 1.0 + 1e-2, f"Grad max {max_val} should be <= ~1.0 after clipping"
        # Post-clip norm should be ~1.0
        post_norm = _norm_of_grad(grads, "layer_0_weight")
        assert abs(post_norm - 1.0) < 0.1, f"Post-clip norm {post_norm} should be ~1.0"
        print(f"[global] post-clip grad max={max_val:.6f}, norm={post_norm:.4f}: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_component_wise_clip():
    """Test 2: ws_ and backbone grads clipped independently."""
    device = ttnn.open_device(device_id=0)
    try:
        grads = _make_grads(device, [
            ("layer_0_weight", 10.0),
            ("ws_read_gate", 5.0),
        ])
        ret = clip_grad_norm(grads, max_norm=1.0, ws_max_norm=0.5)
        # Backbone norm pre-clip = sqrt(1024 * 100) = 320; clipped to 1.0
        bb_post = _norm_of_grad(grads, "layer_0_weight")
        # WS norm pre-clip = sqrt(1024 * 25) = 160; clipped to 0.5
        ws_post = _norm_of_grad(grads, "ws_read_gate")
        print(f"[comp] returned norm={ret:.4f}, bb_post={bb_post:.4f}, ws_post={ws_post:.4f}")
        assert abs(bb_post - 1.0) < 0.1, f"Backbone post-clip norm {bb_post} should be ~1.0"
        assert abs(ws_post - 0.5) < 0.1, f"WS post-clip norm {ws_post} should be ~0.5"
        print("[comp] backbone clipped to ~1.0, ws clipped to ~0.5 independently: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_no_clip_small_norms():
    """Test 3: no clipping when norms are below thresholds."""
    device = ttnn.open_device(device_id=0)
    try:
        grads = _make_grads(device, [("layer_0_weight", 0.01)])
        # Pre-clip norm = sqrt(1024 * 0.0001) = sqrt(0.1024) = 0.32
        pre_val = ttnn.to_torch(grads["layer_0_weight"]).float().max().item()
        ret = clip_grad_norm(grads, max_norm=1.0, ws_max_norm=None)
        post_val = ttnn.to_torch(grads["layer_0_weight"]).float().max().item()
        print(f"[noclip] returned norm={ret:.6f}, pre_val={pre_val:.6f}, post_val={post_val:.6f}")
        assert ret < 1.0, f"Norm {ret} should be < 1.0 (below threshold)"
        assert abs(post_val - pre_val) < 1e-6, \
            f"Grad changed from {pre_val} to {post_val} but should be unchanged"
        print("[noclip] grads unchanged when below threshold: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_gamma_scaling():
    """Test 4: gamma grads scaled by 1/gamma_scale before norm computation."""
    device = ttnn.open_device(device_id=0)
    try:
        gamma_scale = 128.0
        # Gamma grad value 128.0; scaled by 1/128 = 1.0 before norm.
        # Scaled norm = sqrt(1024 * 1.0) = 32.0
        grads = _make_grads(device, [("layer_0_gamma", 128.0)])
        ret = clip_grad_norm(grads, max_norm=0.5, gamma_scale=gamma_scale, ws_max_norm=None)
        # Returned norm should reflect gamma scaling: ~32.0
        expected_scaled_norm = math.sqrt(N * N * (128.0 / gamma_scale) ** 2)
        print(f"[gamma] returned norm={ret:.4f} (expected ~{expected_scaled_norm:.1f})")
        assert ret > 0.5, f"Returned norm {ret} should be > 0.5 (gamma-scaled norm exceeds max_norm)"
        # The grad should have been clipped (norm 32 -> 0.5, scale = 0.5/32)
        post_val = ttnn.to_torch(grads["layer_0_gamma"]).float().max().item()
        pre_val = 128.0
        assert post_val < pre_val, f"Gamma grad {post_val} should be clipped below {pre_val}"
        # Post-clip scaled norm should be ~0.5
        post_scaled_norm = _norm_of_grad(grads, "layer_0_gamma", gamma_scale=gamma_scale)
        assert abs(post_scaled_norm - 0.5) < 0.1, \
            f"Post-clip gamma-scaled norm {post_scaled_norm} should be ~0.5"
        print(f"[gamma] post-clip val={post_val:.4f}, scaled norm={post_scaled_norm:.4f}: OK")
        return True
    finally:
        ttnn.close_device(device)


def test_empty_grads():
    """Test 5: empty grads returns 0.0, no crash."""
    ret = clip_grad_norm({}, max_norm=1.0)
    assert ret == 0.0, f"Empty grads should return 0.0, got {ret}"
    print(f"[empty] returned {ret}: OK")
    return True


def main():
    tests = [
        ("global_clip", test_global_clip),
        ("component_wise_clip", test_component_wise_clip),
        ("no_clip_small_norms", test_no_clip_small_norms),
        ("gamma_scaling", test_gamma_scaling),
        ("empty_grads", test_empty_grads),
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
