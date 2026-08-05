"""Two-stage parity test for TTRetentionLayer against a PyTorch fp32 reference.

  Stage 0 (--gradcheck): float64 gradcheck of the reference itself.
  Stage 1: tt-nn forward   vs. reference forward.       -> tt-nn op issues
  Stage 2: tt-nn backward  vs. reference autograd.      -> manual-gradient math

Usage:
    /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_retention_parity.py
    ... --gradcheck        also run the float64 self-check of the reference
    ... --device 0         select a Tenstorrent device
    ... --tile-aligned     use tile-aligned (>=32) shapes only
"""

import os
import sys
import argparse
import zlib
from pathlib import Path

_argv_device = 0
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _argv_device = int(sys.argv[_i + 1])
        break
os.environ.setdefault("TT_VISIBLE_DEVICES", str(_argv_device))

_P300_SUBSYSTEM_IDS = {"0x0044", "0x0045", "0x0046"}


def _is_p300():
    try:
        for entry in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
            if (entry / "device" / "subsystem_device").read_text().strip().lower() in _P300_SUBSYSTEM_IDS:
                return True
    except Exception:
        pass
    return False


def _find_mgd():
    try:
        import importlib.util
        spec = importlib.util.find_spec("ttnn")
        path = (Path(next(iter(spec.submodule_search_locations))) / "tt_metal" / "fabric"
                / "mesh_graph_descriptors" / "p150_mesh_graph_descriptor.textproto")
        if path.is_file():
            return str(path)
        for p in sys.path:
            c = (Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric"
                 / "mesh_graph_descriptors" / "p150_mesh_graph_descriptor.textproto")
            if c.is_file():
                return str(c)
    except Exception:
        pass
    return None


if _is_p300():
    _mgd = _find_mgd()
    if _mgd:
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

import torch  # noqa: E402
import ttnn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import ModelConfig, to_device, TTRetentionLayer  # noqa: E402
from retention_reference import RetentionReference, PARAM_NAMES  # noqa: E402


# bf16 carries ~3 decimal digits. After a T-long matmul chain, ~2-4% relative
# error against fp32 is expected and is NOT a bug. A real math bug shows up as
# O(1) relative error, not O(0.05).
FWD_TOL = 0.05
BWD_TOL = 0.08

# gamma is stored as fp32 and its gradient involves a host-side reduction,
# so it may have slightly higher error than pure bf16 ops.
PARAM_TOL = {"gamma": 0.10}

# Parameters stored as fp32 on device
FP32_PARAMS = {"gamma"}


def rel_err(a, b):
    """Relative error in the Frobenius norm -- robust to per-element cancellation."""
    a, b = a.double().flatten(), b.double().flatten()
    denom = max(b.norm().item(), 1e-12)
    return (a - b).norm().item() / denom


def randomised_params(config, seed=0):
    """Random init for every parameter."""
    g = torch.Generator().manual_seed(seed)
    D = config.d_model

    def rn(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale

    return {
        "qkv_weight": rn(D, 4 * D, scale=0.2),
        "out_proj_weight": rn(D, D, scale=0.2),
        "gamma": rn(config.n_heads, scale=0.1) - 0.05,  # log(gamma) ~ -0.05
    }


def install_params(tt_layer, params, device):
    tt = {}
    for name, t in params.items():
        if name in FP32_PARAMS:
            tt[name] = to_device(t.float(), device, dtype=ttnn.float32)
        else:
            tt[name] = to_device(t.bfloat16(), device, dtype=ttnn.bfloat16)
    tt_layer.set_params(tt)


# ---------------------------------------------------------------------------
# Stage 0: validate the reference against finite differences in float64
# ---------------------------------------------------------------------------

def stage0_gradcheck(config):
    print("\n=== Stage 0: reference self-check (float64 gradcheck) ===")
    params = randomised_params(config, seed=7)
    ref = RetentionReference(config, dtype=torch.float64)
    ref.load({k: v.double() for k, v in params.items()})

    B, T = 2, 6
    x = (torch.randn(B, T, config.d_model, dtype=torch.float64) * 0.5).requires_grad_(True)

    def f(xx, *ps):
        for n, t in zip(PARAM_NAMES, ps):
            setattr(ref, n, t)
        return ref(xx)

    args = (x,) + tuple(ref.params()[n] for n in PARAM_NAMES)
    ok = torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)
    print(f"  gradcheck: {'PASS' if ok else 'FAIL'}")

    # Causality: perturbing step k must not change outputs before k.
    with torch.no_grad():
        x0 = x.detach()
        x1 = x0.clone()
        x1[:, T // 2:] += 10.0
        o0, o1 = ref(x0), ref(x1)
        causal = torch.allclose(o0[:, :T // 2], o1[:, :T // 2], atol=1e-9)
    print(f"  causality: {'PASS' if causal else 'FAIL -- future is leaking into the past'}")
    return ok and causal


# ---------------------------------------------------------------------------
# Stages 1 & 2
# ---------------------------------------------------------------------------

def stage1_forward(tt_layer, ref, x_torch, device):
    print("\n=== Stage 1: forward parity (tt-nn bf16 vs reference fp32) ===")
    out_tt = ttnn.to_torch(tt_layer.forward(to_device(x_torch.bfloat16(), device))).float()
    out_ref = ref(x_torch.float().clone()).detach()

    err = rel_err(out_tt, out_ref)
    print(f"  tt  std={out_tt.std():.6f}  ref std={out_ref.std():.6f}")
    print(f"  rel_err={err:.5f}  tol={FWD_TOL}  -> {'PASS' if err < FWD_TOL else 'FAIL'}")
    if err >= FWD_TOL:
        print("  NOTE: a forward mismatch invalidates Stage 2. Fix this first.")
    return err < FWD_TOL


def stage2_backward(tt_layer, ref, x_torch, device, grad_out_torch):
    print("\n=== Stage 2: backward parity (tt-nn manual vs reference autograd) ===")

    # Reference: loss = sum(out * grad_out) so dloss/dout == grad_out exactly.
    x_ref = x_torch.float().clone().requires_grad_(True)
    out_ref = ref(x_ref)
    loss = (out_ref * grad_out_torch.float()).sum()
    ref_params = [ref.params()[n] for n in PARAM_NAMES]
    ref_grads = torch.autograd.grad(loss, [x_ref] + ref_params)
    ref_grad_x, ref_grad_params = ref_grads[0], dict(zip(PARAM_NAMES, ref_grads[1:]))

    # tt-nn: forward must run immediately before backward (it populates _cache).
    tt_layer.forward(to_device(x_torch.bfloat16(), device))
    grad_x_tt, grads_tt = tt_layer.backward(to_device(grad_out_torch.bfloat16(), device))

    rows = [("grad_x", rel_err(ttnn.to_torch(grad_x_tt).float(), ref_grad_x))]
    for name in PARAM_NAMES:
        if name not in grads_tt:
            rows.append((name, float("nan")))
            continue
        g_tt = grads_tt[name]
        rows.append((name, rel_err(ttnn.to_torch(g_tt).float(), ref_grad_params[name])))

    width = max(len(n) for n, _ in rows)
    all_ok = True
    for name, err in rows:
        tol = PARAM_TOL.get(name, BWD_TOL)
        ok = err == err and err < tol  # NaN-safe
        all_ok &= ok
        note = "  (relaxed: host-side reduction)" if name in PARAM_TOL else ""
        print(f"  {name:<{width}}  rel_err={err:.5f}  tol={tol:.2f}  "
              f"{'PASS' if ok else 'FAIL'}{note}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gradcheck", action="store_true",
                    help="also run the float64 self-check of the reference")
    ap.add_argument("--tile-aligned", action="store_true",
                    help="use tile-aligned (>=32) shapes to rule out sub-tile ttnn issues")
    args = ap.parse_args()

    if args.tile_aligned:
        config = ModelConfig(d_model=64, n_heads=2)
        B, T = 2, 32
    else:
        config = ModelConfig(d_model=64, n_heads=2)
        B, T = 2, 16

    print(f"config: d_model={config.d_model} H={config.n_heads} "
          f"d_head={config.d_model // config.n_heads} B={B} T={T}")

    results = {}
    if args.gradcheck:
        results["stage0"] = stage0_gradcheck(config)

    device = ttnn.open_device(device_id=0)
    try:
        params = randomised_params(config, seed=zlib.crc32(b"retention") % 2**31)

        tt_layer = TTRetentionLayer(config, device)
        install_params(tt_layer, params, device)

        ref = RetentionReference(config, dtype=torch.float32)
        ref.load(params)

        torch.manual_seed(42)
        x_torch = torch.randn(B, T, config.d_model) * 0.5
        torch.manual_seed(123)
        grad_out_torch = torch.randn(B, T, config.d_model) * 0.5

        results["stage1"] = stage1_forward(tt_layer, ref, x_torch, device)
        results["stage2"] = stage2_backward(tt_layer, ref, x_torch, device, grad_out_torch)
    finally:
        ttnn.close_device(device)

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print("ALL PASS" if ok else "FAILURES FOUND")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
