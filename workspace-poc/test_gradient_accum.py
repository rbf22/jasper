"""Tests for gradient accumulation math — accumulate_grads and host_grads_to_tt.

CPU-only (no device required, uses host-side ttnn tensors). Run with:
    .tt-venv/bin/python -m pytest test_gradient_accum.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import ttnn
from train_ttnn import accumulate_grads


def _to_tt(torch_tensor):
    """Convert a torch tensor to a host-side ttnn tensor (ROW_MAJOR, bf16).

    ROW_MAJOR is used because TILE_LAYOUT requires tile-aligned shapes
    (multiples of 32). No device argument — these are CPU-only tests.
    """
    return ttnn.from_torch(
        torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )


# ---------------------------------------------------------------------------
# accumulate_grads
# ---------------------------------------------------------------------------

def test_accumulate_single_step():
    """Single grad, accum_factor=1 -> acc_grads matches input (scaled by 1/1)."""
    torch_grad = torch.randn(4, 4, dtype=torch.float32)
    tt_grad = _to_tt(torch_grad)

    acc_grads = {}
    accumulate_grads(acc_grads, {"weight": tt_grad}, accum_factor=1)

    assert "weight" in acc_grads
    # accumulate_grads converts to fp32 on host and scales by 1/accum_factor
    expected = torch_grad.float() / 1.0
    assert torch.allclose(acc_grads["weight"], expected, atol=1e-2, rtol=1e-2), (
        f"acc_grads mismatch: {acc_grads['weight']} vs {expected}"
    )
    # Result is stored as fp32
    assert acc_grads["weight"].dtype == torch.float32


def test_accumulate_multiple_steps():
    """4 grads, accum_factor=4 -> result = mean of all grads."""
    grads = [torch.randn(4, 4, dtype=torch.float32) for _ in range(4)]
    tt_grads = [_to_tt(g) for g in grads]

    acc_grads = {}
    for g in tt_grads:
        accumulate_grads(acc_grads, {"weight": g}, accum_factor=4)

    expected = torch.stack(grads).mean(dim=0)
    # bf16 round-trip introduces some error; use a loose tolerance
    assert torch.allclose(acc_grads["weight"], expected, atol=2e-2, rtol=2e-2), (
        f"acc_grads mismatch: {acc_grads['weight']} vs {expected}"
    )


def test_accumulate_accum_factor_2():
    """2 grads, accum_factor=2 -> result = (g1 + g2) / 2."""
    g1 = torch.randn(2, 8, dtype=torch.float32)
    g2 = torch.randn(2, 8, dtype=torch.float32)
    tt1 = _to_tt(g1)
    tt2 = _to_tt(g2)

    acc_grads = {}
    accumulate_grads(acc_grads, {"weight": tt1}, accum_factor=2)
    accumulate_grads(acc_grads, {"weight": tt2}, accum_factor=2)

    expected = (g1 + g2) / 2.0
    assert torch.allclose(acc_grads["weight"], expected, atol=2e-2, rtol=2e-2), (
        f"acc_grads mismatch: {acc_grads['weight']} vs {expected}"
    )


def test_accumulate_multiple_params():
    """Dict with 2 params, verify both accumulated independently."""
    g_a = torch.randn(4, 4, dtype=torch.float32)
    g_b = torch.randn(2, 8, dtype=torch.float32)
    tt_a = _to_tt(g_a)
    tt_b = _to_tt(g_b)

    acc_grads = {}
    accumulate_grads(acc_grads, {"param_a": tt_a, "param_b": tt_b}, accum_factor=1)

    assert "param_a" in acc_grads
    assert "param_b" in acc_grads
    assert torch.allclose(acc_grads["param_a"], g_a, atol=1e-2, rtol=1e-2)
    assert torch.allclose(acc_grads["param_b"], g_b, atol=1e-2, rtol=1e-2)
    # Shapes preserved
    assert acc_grads["param_a"].shape == (4, 4)
    assert acc_grads["param_b"].shape == (2, 8)


def test_accumulate_fp32_precision():
    """Verify accumulation is done in fp32 (not bf16).

    accumulate_grads calls .float() on the to_torch result, so the running
    sum is always fp32. We verify this by checking the dtype of the stored
    tensor and that small differences are preserved (bf16 would round them
    away).
    """
    # Two grads that differ by a small amount — bf16 accumulation would
    # lose the second contribution, but fp32 preserves it.
    base = torch.ones(4, 4, dtype=torch.float32)
    small = torch.full((4, 4), 1e-3, dtype=torch.float32)
    tt_base = _to_tt(base)
    tt_small = _to_tt(small)

    acc_grads = {}
    accumulate_grads(acc_grads, {"weight": tt_base}, accum_factor=1)
    accumulate_grads(acc_grads, {"weight": tt_small}, accum_factor=1)

    # Stored tensor must be fp32
    assert acc_grads["weight"].dtype == torch.float32, (
        f"Expected fp32, got {acc_grads['weight'].dtype}"
    )
    # The small contribution (1e-3) is below bf16 ULP at 1.0 (~8e-3), so
    # if accumulation were bf16 the result would be ~1.0. With fp32 it
    # should be ~1.001. We allow tolerance for the bf16 *input* round-trip
    # (the small value may round to ~0.0009766 = 2^-10 in bf16), but the
    # sum must be strictly greater than 1.0.
    result = acc_grads["weight"]
    assert result.mean().item() > 1.0, (
        f"Small contribution lost — result mean {result.mean().item()} "
        f"should be > 1.0 (fp32 accumulation preserves sub-ULP additions)"
    )


def test_accumulate_empty():
    """Empty new_grads dict, verify no crash and acc_grads unchanged."""
    acc_grads = {"weight": torch.ones(4, 4, dtype=torch.float32)}
    accumulate_grads(acc_grads, {}, accum_factor=4)
    # Nothing added
    assert "weight" in acc_grads
    assert torch.allclose(acc_grads["weight"], torch.ones(4, 4))
    # No new keys introduced
    assert set(acc_grads.keys()) == {"weight"}
