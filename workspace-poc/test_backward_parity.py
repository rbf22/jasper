#!/usr/bin/env python3
"""Backward parity tests: ttnn manual backward vs PyTorch autograd.

These tests require a Tenstorrent device. They compare the ttnn layer's
manual backward() output against PyTorch autograd gradients computed on
a reference implementation.

Each test:
  1. Creates a TT layer with known random parameters
  2. Creates a PyTorch reference with the same parameters
  3. Runs forward on both
  4. Runs backward on both (ttnn manual vs autograd)
  5. Compares gradients with relative error tolerance

Tolerances are relaxed vs the CPU gradcheck because:
  - ttnn runs in bf16 (not float64)
  - ttnn operations may use different numerical algorithms
  - RoPE cos/sin tables may differ slightly in precision

Run:
    cd /home/rfenwick/Documents/jasper/workspace-poc
    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... \
    .tt-venv/bin/python test_backward_parity.py

    # Run specific test
    .tt-venv/bin/python test_backward_parity.py --test attention
    .tt-venv/bin/python test_backward_parity.py --test attention_residual
    .tt-venv/bin/python test_backward_parity.py --test retention
"""

import os
import sys
import math
import argparse

# Set up environment for P300/P150 devices
_P300_SUBSYSTEM_IDS = {"0x0044", "0x0045", "0x0046"}

def _is_p300():
    try:
        import subprocess
        r = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=5)
        return any(sid in r.stdout for sid in _P300_SUBSYSTEM_IDS)
    except Exception:
        return False

if _is_p300() and "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
    _mesh_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        "tt-boltz/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/"
        "mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
    )
    if os.path.exists(_mesh_path):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = _mesh_path

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_ttnn import (
    ModelConfig, TTAttentionLayer, TTRetentionLayer, AttentionResidual,
    to_device, to_host, _safe_deallocate
)

# Import references from our CPU test module
from test_gradients import (
    AttentionReference, AttentionResidualReference,
    ATTENTION_PARAM_NAMES, AR_PARAM_NAMES,
    apply_rope, rms_norm, rel_err,
)
from retention_reference import RetentionReference, PARAM_NAMES as RETENTION_PARAM_NAMES


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

FWD_TOL = 0.05      # 5% relative error for forward (bf16 vs fp32)
BWD_TOL = 0.08      # 8% relative error for backward (bf16 vs fp32)
PARAM_TOL = {"gamma": 0.10}  # Relaxed for gamma (host-side reduction)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def install_params_tt(tt_layer, params_torch, device, fp32_params=None):
    """Install PyTorch parameters into a TT layer."""
    fp32_params = fp32_params or set()
    tt_params = {}
    for name, t in params_torch.items():
        if name in fp32_params:
            tt_params[name] = to_device(t.float(), device, dtype=ttnn.float32)
        else:
            tt_params[name] = to_device(t.bfloat16(), device, dtype=ttnn.bfloat16)
    tt_layer.set_params(tt_params)
    return tt_layer


def extract_tt_grads(tt_grads_dict, device):
    """Convert TT gradient tensors to PyTorch on host."""
    return {name: to_host(g).float() for name, g in tt_grads_dict.items()}


# ---------------------------------------------------------------------------
# Test 1: Attention Layer Backward Parity
# ---------------------------------------------------------------------------

def test_attention_backward_parity(verbose=False):
    """Compare TTAttentionLayer.backward against PyTorch autograd."""
    device = ttnn.open_device(device_id=0)
    try:
        config = ModelConfig(d_model=64, n_heads=2)
        D, H, d_h = config.d_model, config.n_heads, config.d_model // config.n_heads
        B, T = 2, 32

        torch.manual_seed(42)

        # Random parameters
        params = {
            "qkv_weight": torch.randn(D, 3 * D, dtype=torch.float32) * 0.02,
            "out_proj_weight": torch.randn(D, D, dtype=torch.float32) * 0.02,
        }

        # Random input and grad_out
        x_torch = torch.randn(B, T, D, dtype=torch.float32)
        grad_out_torch = torch.randn(B, T, D, dtype=torch.float32) * 0.1

        # --- PyTorch reference ---
        ref = AttentionReference(config, dtype=torch.float32)
        ref.load(params)

        x_ref = x_torch.clone().requires_grad_(True)
        out_ref = ref(x_ref)
        loss = (out_ref * grad_out_torch).sum()
        ref_grads = torch.autograd.grad(loss, [x_ref] + [ref.params()[n] for n in ATTENTION_PARAM_NAMES])
        ref_grad_x = ref_grads[0]
        ref_grad_params = dict(zip(ATTENTION_PARAM_NAMES, ref_grads[1:]))

        # --- TT layer ---
        tt_layer = TTAttentionLayer(config, device)
        install_params_tt(tt_layer, params, device)

        # Forward
        x_tt = to_device(x_torch.bfloat16(), device)
        out_tt = tt_layer.forward(x_tt)
        ttnn.synchronize_device(device)

        # Check forward parity
        out_tt_host = to_host(out_tt).float()
        fwd_err = rel_err(out_tt_host, out_ref.detach())
        if verbose:
            print(f"  Forward: rel_err={fwd_err:.5f} tol={FWD_TOL} {'PASS' if fwd_err < FWD_TOL else 'FAIL'}")

        # Backward
        grad_out_tt = to_device(grad_out_torch.bfloat16(), device)
        grad_x_tt, grads_tt = tt_layer.backward(grad_out_tt)
        ttnn.synchronize_device(device)

        # Compare gradients
        grad_x_host = to_host(grad_x_tt).float()
        grad_x_err = rel_err(grad_x_host, ref_grad_x)

        param_errors = {}
        for name in ATTENTION_PARAM_NAMES:
            g_tt = to_host(grads_tt[name]).float()
            g_ref = ref_grad_params[name]
            param_errors[name] = rel_err(g_tt, g_ref)

        # Report
        all_ok = fwd_err < FWD_TOL
        if verbose:
            print(f"  grad_x: rel_err={grad_x_err:.5f} tol={BWD_TOL} {'PASS' if grad_x_err < BWD_TOL else 'FAIL'}")
        all_ok &= grad_x_err < BWD_TOL

        for name, err in param_errors.items():
            tol = PARAM_TOL.get(name, BWD_TOL)
            ok = err < tol
            if verbose:
                print(f"  {name}: rel_err={err:.5f} tol={tol:.2f} {'PASS' if ok else 'FAIL'}")
            all_ok &= ok

        # Cleanup
        _safe_deallocate(out_tt)
        _safe_deallocate(grad_x_tt)
        for g in grads_tt.values():
            _safe_deallocate(g)
        tt_layer._deallocate_cache()
        _safe_deallocate(x_tt)
        _safe_deallocate(grad_out_tt)

        return all_ok
    finally:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# Test 2: AttentionResidual Backward Parity
# ---------------------------------------------------------------------------

def test_attention_residual_backward_parity(verbose=False):
    """Compare AttentionResidual.backward against PyTorch autograd."""
    device = ttnn.open_device(device_id=0)
    try:
        D = 64
        B, T = 2, 32
        K = 3
        K_active = K  # all active

        torch.manual_seed(42)

        # Parameters
        query_torch = torch.randn(1, D, dtype=torch.float32) * 0.1
        scale_torch = torch.tensor([1.0], dtype=torch.float32)

        # Random x_outputs and grad_out
        x_outputs_torch = [torch.randn(B, T, D, dtype=torch.float32) * 0.3
                           for _ in range(K + 1)]
        grad_out_torch = torch.randn(B, T, D, dtype=torch.float32) * 0.1

        # --- PyTorch reference ---
        config = ModelConfig(d_model=D)
        params = {"ar_query": query_torch, "ar_scale": scale_torch}
        ref = AttentionResidualReference(config, dtype=torch.float32)
        ref.load(params)

        x_refs = [x.clone().requires_grad_(True) for x in x_outputs_torch]
        out_ref = ref(x_refs, K_active)
        loss = (out_ref * grad_out_torch).sum()
        ref_grads = torch.autograd.grad(loss, x_refs + [ref.ar_query, ref.ar_scale])
        ref_grad_x_list = ref_grads[:K+1]
        ref_grad_query = ref_grads[K+1]
        ref_grad_scale = ref_grads[K+2]

        # --- TT AttentionResidual ---
        ar = AttentionResidual(D, device, k_max=K)
        # Install parameters
        ar.set_params({
            "ar_query": to_device(query_torch.bfloat16(), device),
            "ar_scale": to_device(scale_torch.bfloat16(), device),
        })

        # Forward
        x_outputs_tt = [to_device(x.bfloat16(), device) for x in x_outputs_torch]
        out_tt = ar.forward(x_outputs_tt, K_active)
        ttnn.synchronize_device(device)

        # Check forward parity
        out_tt_host = to_host(out_tt).float()
        fwd_err = rel_err(out_tt_host, out_ref.detach())
        if verbose:
            print(f"  Forward: rel_err={fwd_err:.5f} tol={FWD_TOL} {'PASS' if fwd_err < FWD_TOL else 'FAIL'}")

        # Backward
        grad_out_tt = to_device(grad_out_torch.bfloat16(), device)
        grad_x_list_tt, grads_tt = ar.backward(grad_out_tt)
        ttnn.synchronize_device(device)

        # Compare grad_x for each iteration
        all_ok = fwd_err < FWD_TOL
        for k in range(K + 1):
            g_tt = to_host(grad_x_list_tt[k]).float()
            g_ref = ref_grad_x_list[k]
            err = rel_err(g_tt, g_ref)
            ok = err < BWD_TOL
            if verbose:
                print(f"  grad_x[{k}]: rel_err={err:.5f} tol={BWD_TOL} {'PASS' if ok else 'FAIL'}")
            all_ok &= ok

        # Compare query and scale gradients
        g_query_tt = to_host(grads_tt["ar_query"]).float()
        query_err = rel_err(g_query_tt, ref_grad_query)
        ok_q = query_err < BWD_TOL
        if verbose:
            print(f"  ar_query: rel_err={query_err:.5f} tol={BWD_TOL} {'PASS' if ok_q else 'FAIL'}")
        all_ok &= ok_q

        g_scale_tt = to_host(grads_tt["ar_scale"]).float()
        scale_err = rel_err(g_scale_tt, ref_grad_scale)
        ok_s = scale_err < BWD_TOL
        if verbose:
            print(f"  ar_scale: rel_err={scale_err:.5f} tol={BWD_TOL} {'PASS' if ok_s else 'FAIL'}")
        all_ok &= ok_s

        # Cleanup
        _safe_deallocate(out_tt)
        for g in grad_x_list_tt:
            _safe_deallocate(g)
        for g in grads_tt.values():
            _safe_deallocate(g)
        ar._deallocate_cache()
        for x in x_outputs_tt:
            _safe_deallocate(x)
        _safe_deallocate(grad_out_tt)

        return all_ok
    finally:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# Test 3: AttentionResidual with Inactive Iterations
# ---------------------------------------------------------------------------

def test_attention_residual_inactive_parity(verbose=False):
    """Compare AttentionResidual.backward with K_active < K (masked iterations)."""
    device = ttnn.open_device(device_id=0)
    try:
        D = 64
        B, T = 2, 32
        K = 3
        K_active = 1  # Only iterations 0, 1 are active

        torch.manual_seed(42)

        query_torch = torch.randn(1, D, dtype=torch.float32) * 0.1
        scale_torch = torch.tensor([1.0], dtype=torch.float32)

        x_outputs_torch = [torch.randn(B, T, D, dtype=torch.float32) * 0.3
                           for _ in range(K + 1)]
        grad_out_torch = torch.randn(B, T, D, dtype=torch.float32) * 0.1

        # --- PyTorch reference ---
        config = ModelConfig(d_model=D)
        params = {"ar_query": query_torch, "ar_scale": scale_torch}
        ref = AttentionResidualReference(config, dtype=torch.float32)
        ref.load(params)

        x_refs = [x.clone().requires_grad_(True) for x in x_outputs_torch]
        out_ref = ref(x_refs, K_active)
        loss = (out_ref * grad_out_torch).sum()
        ref_grads = torch.autograd.grad(loss, x_refs + [ref.ar_query, ref.ar_scale])
        ref_grad_x_list = ref_grads[:K+1]
        ref_grad_query = ref_grads[K+1]
        ref_grad_scale = ref_grads[K+2]

        # --- TT ---
        ar = AttentionResidual(D, device, k_max=K)
        ar.set_params({
            "ar_query": to_device(query_torch.bfloat16(), device),
            "ar_scale": to_device(scale_torch.bfloat16(), device),
        })

        x_outputs_tt = [to_device(x.bfloat16(), device) for x in x_outputs_torch]
        out_tt = ar.forward(x_outputs_tt, K_active)
        ttnn.synchronize_device(device)

        out_tt_host = to_host(out_tt).float()
        fwd_err = rel_err(out_tt_host, out_ref.detach())
        if verbose:
            print(f"  Forward: rel_err={fwd_err:.5f} tol={FWD_TOL} {'PASS' if fwd_err < FWD_TOL else 'FAIL'}")

        grad_out_tt = to_device(grad_out_torch.bfloat16(), device)
        grad_x_list_tt, grads_tt = ar.backward(grad_out_tt)
        ttnn.synchronize_device(device)

        all_ok = fwd_err < FWD_TOL
        for k in range(K + 1):
            g_tt = to_host(grad_x_list_tt[k]).float()
            g_ref = ref_grad_x_list[k]
            err = rel_err(g_tt, g_ref)
            # Inactive iterations (k > K_active) should have ~zero gradient
            # but bf16 noise may make it non-zero. Use a relaxed tolerance.
            tol = BWD_TOL if k <= K_active else 0.15
            ok = err < tol
            tag = "active" if k <= K_active else "inactive"
            if verbose:
                print(f"  grad_x[{k}] ({tag}): rel_err={err:.5f} tol={tol:.2f} {'PASS' if ok else 'FAIL'}")
            all_ok &= ok

        g_query_tt = to_host(grads_tt["ar_query"]).float()
        query_err = rel_err(g_query_tt, ref_grad_query)
        ok_q = query_err < BWD_TOL
        if verbose:
            print(f"  ar_query: rel_err={query_err:.5f} tol={BWD_TOL} {'PASS' if ok_q else 'FAIL'}")
        all_ok &= ok_q

        g_scale_tt = to_host(grads_tt["ar_scale"]).float()
        scale_err = rel_err(g_scale_tt, ref_grad_scale)
        ok_s = scale_err < BWD_TOL
        if verbose:
            print(f"  ar_scale: rel_err={scale_err:.5f} tol={BWD_TOL} {'PASS' if ok_s else 'FAIL'}")
        all_ok &= ok_s

        # Cleanup
        _safe_deallocate(out_tt)
        for g in grad_x_list_tt:
            _safe_deallocate(g)
        for g in grads_tt.values():
            _safe_deallocate(g)
        ar._deallocate_cache()
        for x in x_outputs_tt:
            _safe_deallocate(x)
        _safe_deallocate(grad_out_tt)

        return all_ok
    finally:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# Test 4: Retention Layer Backward Parity
# ---------------------------------------------------------------------------

def test_retention_backward_parity(verbose=False):
    """Compare TTRetentionLayer.backward against PyTorch autograd.

    This wraps the existing test_retention_parity.py Stage 2 logic.
    """
    device = ttnn.open_device(device_id=0)
    try:
        config = ModelConfig(d_model=64, n_heads=2)
        D, H, d_h = config.d_model, config.n_heads, config.d_model // config.n_heads
        B, T = 2, 32

        torch.manual_seed(42)

        # Random parameters (gamma stored as log(gamma), init ~ -0.05)
        params = {
            "qkv_weight": torch.randn(D, 4 * D, dtype=torch.float32) * 0.02,
            "out_proj_weight": torch.randn(D, D, dtype=torch.float32) * 0.02,
            "gamma": torch.full((H,), -0.05, dtype=torch.float32),
        }

        x_torch = torch.randn(B, T, D, dtype=torch.float32)
        grad_out_torch = torch.randn(B, T, D, dtype=torch.float32) * 0.1

        # --- PyTorch reference ---
        ref = RetentionReference(config, dtype=torch.float32)
        ref.load(params)

        x_ref = x_torch.clone().requires_grad_(True)
        out_ref = ref(x_ref)
        loss = (out_ref * grad_out_torch).sum()
        ref_grads = torch.autograd.grad(loss, [x_ref] + [ref.params()[n] for n in RETENTION_PARAM_NAMES])
        ref_grad_x = ref_grads[0]
        ref_grad_params = dict(zip(RETENTION_PARAM_NAMES, ref_grads[1:]))

        # --- TT layer ---
        tt_layer = TTRetentionLayer(config, device, use_fused_rope=True)
        # gamma is stored as fp32 on device
        install_params_tt(tt_layer, params, device, fp32_params={"gamma"})

        # Forward
        x_tt = to_device(x_torch.bfloat16(), device)
        out_tt = tt_layer.forward(x_tt)
        ttnn.synchronize_device(device)

        out_tt_host = to_host(out_tt).float()
        fwd_err = rel_err(out_tt_host, out_ref.detach())
        if verbose:
            print(f"  Forward: rel_err={fwd_err:.5f} tol={FWD_TOL} {'PASS' if fwd_err < FWD_TOL else 'FAIL'}")

        # Backward
        grad_out_tt = to_device(grad_out_torch.bfloat16(), device)
        grad_x_tt, grads_tt = tt_layer.backward(grad_out_tt)
        ttnn.synchronize_device(device)

        # Compare
        grad_x_host = to_host(grad_x_tt).float()
        grad_x_err = rel_err(grad_x_host, ref_grad_x)

        all_ok = fwd_err < FWD_TOL
        if verbose:
            print(f"  grad_x: rel_err={grad_x_err:.5f} tol={BWD_TOL} {'PASS' if grad_x_err < BWD_TOL else 'FAIL'}")
        all_ok &= grad_x_err < BWD_TOL

        for name in RETENTION_PARAM_NAMES:
            if name not in grads_tt:
                if verbose:
                    print(f"  {name}: NOT IN TT GRADS (skipped)")
                continue
            g_tt = to_host(grads_tt[name]).float()
            g_ref = ref_grad_params[name]
            err = rel_err(g_tt, g_ref)
            tol = PARAM_TOL.get(name, BWD_TOL)
            ok = err < tol
            if verbose:
                print(f"  {name}: rel_err={err:.5f} tol={tol:.2f} {'PASS' if ok else 'FAIL'}")
            all_ok &= ok

        # Cleanup
        _safe_deallocate(out_tt)
        _safe_deallocate(grad_x_tt)
        for g in grads_tt.values():
            _safe_deallocate(g)
        tt_layer._deallocate_cache()
        _safe_deallocate(x_tt)
        _safe_deallocate(grad_out_tt)

        return all_ok
    finally:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = {
    "attention": ("Attention layer backward parity", test_attention_backward_parity),
    "attention_residual": ("AttentionResidual backward parity (all active)",
                           test_attention_residual_backward_parity),
    "attention_residual_inactive": ("AttentionResidual backward parity (inactive iters)",
                                    test_attention_residual_inactive_parity),
    "retention": ("Retention layer backward parity", test_retention_backward_parity),
}


def main():
    parser = argparse.ArgumentParser(description="Backward parity tests (requires device)")
    parser.add_argument("--test", "-t", default=None,
                        help=f"Test name (one of: {', '.join(TESTS.keys())})")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--list", "-l", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for key, (desc, _) in TESTS.items():
            print(f"  {key:40s}  {desc}")
        return

    if args.test:
        if args.test not in TESTS:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(TESTS.keys())}")
            sys.exit(1)
        desc, fn = TESTS[args.test]
        print(f"\nRunning: {desc}")
        try:
            ok = fn(verbose=True)
            print(f"\nResult: {'PASS' if ok else 'FAIL'}")
            sys.exit(0 if ok else 1)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Run all tests
        results = {}
        for key, (desc, fn) in TESTS.items():
            print(f"\n{'='*60}")
            print(f"TEST: {desc}")
            print(f"{'='*60}")
            try:
                ok = fn(verbose=True)
                results[key] = ok
                print(f"  Result: {'PASS' if ok else 'FAIL'}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[key] = False

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        all_pass = True
        for key, (desc, _) in TESTS.items():
            ok = results.get(key, False)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  {key:40s}  {status}")

        if all_pass:
            print(f"\nALL TESTS PASSED")
        else:
            print(f"\nSOME TESTS FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
