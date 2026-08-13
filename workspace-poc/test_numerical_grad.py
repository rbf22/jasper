#!/usr/bin/env python3
"""Numerical finite-difference gradient check for the full training step.

This test verifies that the ttnn model's backward gradients match
numerical finite-difference gradients computed on the same ttnn forward
pass. This is the gold standard for gradient correctness — it doesn't
rely on a PyTorch reference, only on the forward pass itself.

Method:
  1. Run forward + backward to get analytic gradients
  2. For each parameter, perturb it by +eps and -eps, re-run forward,
     and compute (loss(+eps) - loss(-eps)) / (2*eps) as the numerical gradient
  3. Compare analytic vs numerical gradients

This is slower than the reference-based parity tests (O(P) forward passes
where P is the number of parameters), so we use a small model and only
check a subset of parameters.

Run:
    cd /home/rfenwick/Documents/jasper/workspace-poc
    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... \
    .tt-venv/bin/python test_numerical_grad.py
"""

import os
import sys
import math
import argparse
import random

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
    ModelConfig, TTWRAPModel, TTRetentionLayer,
    to_device, to_host, _safe_deallocate
)
from train_ttnn import cross_entropy_loss, build_model_config


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative error in Frobenius norm."""
    a, b = a.double().flatten(), b.double().flatten()
    denom = max(b.norm().item(), 1e-12)
    return (a - b).norm().item() / denom


# ---------------------------------------------------------------------------
# Numerical gradient check for a single TTRetentionLayer
# ---------------------------------------------------------------------------

def test_retention_numerical_grad(verbose=False, eps=0.1, tol=0.25):
    """Finite-difference gradient check for TTRetentionLayer.

    Uses bf16 forward, so tolerance is relaxed (25% relative error).
    eps=0.1 is used because bf16 has ~0.8% relative precision, making
    smaller eps values produce noisy finite differences.
    We check a subset of parameter elements (5 per parameter)
    to keep runtime reasonable.
    """
    device = ttnn.open_device(device_id=0)
    try:
        config = ModelConfig(d_model=64, n_heads=2, n_layers=1, vocab_size=64,
                             use_workspace=False, recurrent_core=False,
                             freeze_gamma=False)
        D, H = config.d_model, config.n_heads
        B, T = 2, 32
        V = config.vocab_size

        torch.manual_seed(42)
        random.seed(42)

        # Create model
        model = TTWRAPModel(config, device)

        # Fixed input and labels
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)

        def compute_loss():
            """Forward pass → scalar loss."""
            logits = model.forward(input_ids, k_value=None)
            ttnn.synchronize_device(device)
            loss_val, grad_logits = cross_entropy_loss(logits, labels)
            ttnn.synchronize_device(device)
            _safe_deallocate(grad_logits)
            _safe_deallocate(logits)
            model.clear_caches()
            ttnn.synchronize_device(device)
            return loss_val

        def compute_analytic_grads():
            """Forward + backward → analytic gradients."""
            logits = model.forward(input_ids, k_value=None)
            ttnn.synchronize_device(device)
            loss_val, grad_logits = cross_entropy_loss(logits, labels)
            ttnn.synchronize_device(device)
            _safe_deallocate(logits)
            grads = model.backward(grad_logits)
            ttnn.synchronize_device(device)
            _safe_deallocate(grad_logits)
            # Extract gradients to host before cleanup
            host_grads = {}
            for name, g in grads.items():
                host_grads[name] = to_host(g).float().clone()
                _safe_deallocate(g)
            model.clear_caches()
            ttnn.synchronize_device(device)
            return host_grads

        # Get analytic gradients
        analytic_grads = compute_analytic_grads()

        # Parameters to check (first layer only, subset of elements)
        layer = model.layers[0]
        param_names = ["qkv_weight", "out_proj_weight"]
        if hasattr(layer, 'gamma') and not config.freeze_gamma:
            param_names.append("gamma")

        n_elements_to_check = 5  # per parameter
        all_ok = True
        total_checked = 0

        for pname in param_names:
            tt_param = layer.get_params()[pname]
            param_host = to_host(tt_param).float().clone()
            # Model backward uses prefixed keys: layer_0_qkv_weight
            analytic_key = f"layer_0_{pname}"
            analytic = analytic_grads.get(analytic_key)
            if analytic is None:
                if verbose:
                    print(f"  {pname}: no analytic gradient (key={analytic_key}, skipped)")
                continue

            # Select random elements to check
            flat_param = param_host.flatten()
            flat_analytic = analytic.flatten()
            n_elements = flat_param.shape[0]
            indices = random.sample(range(n_elements), min(n_elements_to_check, n_elements))

            errors = []
            for idx in indices:
                orig_val = flat_param[idx].item()

                # +eps
                flat_param[idx] = orig_val + eps
                new_param = flat_param.reshape(param_host.shape).bfloat16()
                layer.set_params({pname: to_device(new_param, device)})
                loss_plus = compute_loss()

                # -eps
                flat_param[idx] = orig_val - eps
                new_param = flat_param.reshape(param_host.shape).bfloat16()
                layer.set_params({pname: to_device(new_param, device)})
                loss_minus = compute_loss()

                # Restore
                flat_param[idx] = orig_val
                new_param = flat_param.reshape(param_host.shape).bfloat16()
                layer.set_params({pname: to_device(new_param, device)})

                # Numerical gradient
                numerical = (loss_plus - loss_minus) / (2 * eps)
                analytic_val = flat_analytic[idx].item()

                if abs(numerical) < 1e-8 and abs(analytic_val) < 1e-8:
                    err = 0.0
                elif abs(numerical) < 1e-8:
                    err = abs(analytic_val)  # analytic is nonzero, numerical is ~0
                else:
                    err = abs(analytic_val - numerical) / max(abs(numerical), 1e-8)

                errors.append(err)
                total_checked += 1

                if verbose:
                    status = "OK" if err < tol else "MISMATCH"
                    print(f"  {pname}[{idx}]: analytic={analytic_val:+.6f} "
                          f"numerical={numerical:+.6f} err={err:.4f} {status}")

            avg_err = sum(errors) / len(errors) if errors else 0
            max_err = max(errors) if errors else 0
            param_ok = max_err < tol
            all_ok &= param_ok
            if verbose:
                print(f"  {pname}: avg_err={avg_err:.4f} max_err={max_err:.4f} "
                      f"tol={tol} {'PASS' if param_ok else 'FAIL'}")

        if verbose:
            print(f"\n  Total elements checked: {total_checked}")
            print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")

        return all_ok
    finally:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# Numerical gradient check for AttentionResidual
# ---------------------------------------------------------------------------

def test_ar_numerical_grad(verbose=False, eps=0.1, tol=0.25):
    """Finite-difference gradient check for AttentionResidual.

    Uses eps=0.1 (larger than default) because bf16 has ~0.8% relative
    precision, so smaller eps values produce noisy finite differences.
    Tolerance is 25% to account for bf16 + finite-difference truncation.
    """
    from model_ttnn import AttentionResidual

    device = ttnn.open_device(device_id=0)
    try:
        D = 64
        B, T = 2, 32
        K = 2
        K_active = K

        torch.manual_seed(42)
        random.seed(42)

        ar = AttentionResidual(D, device, k_max=K)

        # Fixed x_outputs and grad_out
        x_outputs_torch = [torch.randn(B, T, D, dtype=torch.float32) * 0.3
                           for _ in range(K + 1)]
        grad_out_torch = torch.randn(B, T, D, dtype=torch.float32) * 0.1

        def compute_loss():
            """Forward → scalar loss = sum(out * grad_out)."""
            x_tt = [to_device(x.bfloat16(), device) for x in x_outputs_torch]
            out = ar.forward(x_tt, K_active)
            ttnn.synchronize_device(device)
            out_host = to_host(out).float()
            _safe_deallocate(out)
            ar._deallocate_cache()
            for x in x_tt:
                _safe_deallocate(x)
            ttnn.synchronize_device(device)
            return (out_host * grad_out_torch).sum().item()

        def compute_analytic_grads():
            x_tt = [to_device(x.bfloat16(), device) for x in x_outputs_torch]
            out = ar.forward(x_tt, K_active)
            ttnn.synchronize_device(device)
            grad_out_tt = to_device(grad_out_torch.bfloat16(), device)
            grad_x_list, grads = ar.backward(grad_out_tt)
            ttnn.synchronize_device(device)
            # Extract to host
            host_grads = {}
            host_grads["ar_query"] = to_host(grads["ar_query"]).float().clone()
            host_grads["ar_scale"] = to_host(grads["ar_scale"]).float().clone()
            for i, gx in enumerate(grad_x_list):
                host_grads[f"grad_x[{i}]"] = to_host(gx).float().clone()
                _safe_deallocate(gx)
            for g in grads.values():
                _safe_deallocate(g)
            _safe_deallocate(out)
            _safe_deallocate(grad_out_tt)
            ar._deallocate_cache()
            for x in x_tt:
                _safe_deallocate(x)
            ttnn.synchronize_device(device)
            return host_grads

        analytic_grads = compute_analytic_grads()

        # Check ar_query (first 5 elements)
        query_host = to_host(ar.query).float().clone()
        query_flat = query_host.flatten()
        query_analytic = analytic_grads["ar_query"].flatten()

        all_ok = True
        n_check = min(5, query_flat.shape[0])
        indices = random.sample(range(query_flat.shape[0]), n_check)

        for idx in indices:
            orig = query_flat[idx].item()
            # +eps
            query_flat[idx] = orig + eps
            ar.set_params({"ar_query": to_device(
                query_flat.reshape(query_host.shape).bfloat16(), device)})
            loss_plus = compute_loss()
            # -eps
            query_flat[idx] = orig - eps
            ar.set_params({"ar_query": to_device(
                query_flat.reshape(query_host.shape).bfloat16(), device)})
            loss_minus = compute_loss()
            # Restore
            query_flat[idx] = orig
            ar.set_params({"ar_query": to_device(
                query_flat.reshape(query_host.shape).bfloat16(), device)})

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic_val = query_analytic[idx].item()
            if abs(numerical) < 1e-8 and abs(analytic_val) < 1e-8:
                err = 0.0
            elif abs(numerical) < 1e-8:
                err = abs(analytic_val)
            else:
                err = abs(analytic_val - numerical) / max(abs(numerical), 1e-8)

            ok = err < tol
            all_ok &= ok
            if verbose:
                status = "OK" if ok else "MISMATCH"
                print(f"  ar_query[{idx}]: analytic={analytic_val:+.6f} "
                      f"numerical={numerical:+.6f} err={err:.4f} {status}")

        # Check ar_scale
        scale_host = to_host(ar.scale).float().clone()
        scale_orig = scale_host.item()
        scale_analytic = analytic_grads["ar_scale"].item()

        ar.set_params({"ar_scale": to_device(
            torch.tensor([scale_orig + eps], dtype=torch.float32).bfloat16(), device)})
        loss_plus = compute_loss()
        ar.set_params({"ar_scale": to_device(
            torch.tensor([scale_orig - eps], dtype=torch.float32).bfloat16(), device)})
        loss_minus = compute_loss()
        ar.set_params({"ar_scale": to_device(
            torch.tensor([scale_orig], dtype=torch.float32).bfloat16(), device)})

        numerical = (loss_plus - loss_minus) / (2 * eps)
        if abs(numerical) < 1e-8 and abs(scale_analytic) < 1e-8:
            err = 0.0
        elif abs(numerical) < 1e-8:
            err = abs(scale_analytic)
        else:
            err = abs(scale_analytic - numerical) / max(abs(numerical), 1e-8)

        ok = err < tol
        all_ok &= ok
        if verbose:
            status = "OK" if ok else "MISMATCH"
            print(f"  ar_scale: analytic={scale_analytic:+.6f} "
                  f"numerical={numerical:+.6f} err={err:.4f} {status}")

        if verbose:
            print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}")
        return all_ok
    finally:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = {
    "retention": ("Retention layer numerical gradient check",
                  test_retention_numerical_grad),
    "attention_residual": ("AttentionResidual numerical gradient check",
                           test_ar_numerical_grad),
}


def main():
    parser = argparse.ArgumentParser(
        description="Numerical finite-difference gradient checks (requires device)")
    parser.add_argument("--test", "-t", default=None,
                        help=f"Test name (one of: {', '.join(TESTS.keys())})")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="Finite difference epsilon (default: 0.1, suitable for bf16)")
    parser.add_argument("--tol", type=float, default=0.25,
                        help="Relative error tolerance (default: 0.25, relaxed for bf16)")
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
        print(f"  eps={args.eps} tol={args.tol}")
        try:
            ok = fn(verbose=True, eps=args.eps, tol=args.tol)
            print(f"\nResult: {'PASS' if ok else 'FAIL'}")
            sys.exit(0 if ok else 1)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        results = {}
        for key, (desc, fn) in TESTS.items():
            print(f"\n{'='*60}")
            print(f"TEST: {desc}")
            print(f"{'='*60}")
            try:
                ok = fn(verbose=True, eps=args.eps, tol=args.tol)
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
