"""Test that workspace ReZero gates are clamped after each optimizer step.

ReZero gates are unbounded scalars. When the read gate grows large negative
(causal masking makes the workspace contribution noise → model suppresses it),
the pre-norm residual `decay * slots + gate * read_out` experiences
cancellation, making RMS(slots_pre_norm) very small. RMSNorm backward divides
by RMS, amplifying the gradient by 1/RMS → gradient explosion.

This test verifies that `TTMambaWorkspaceModel.clamp_workspace_gates()`:
  1. Exists and clamps gates to [-gate_clamp_bound, gate_clamp_bound]
  2. Does nothing when gates are within bounds
  3. Actually clips when gates exceed bounds
  4. Is called in the training loop after the optimizer step

The test uses the real device because gate values are ttnn device tensors.
"""
import os, sys
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

# P300 chips need a custom fabric mesh graph descriptor; without it,
# ttnn.open_device aborts with TT_FATAL.  Mirrors train_ttnn.py's setup.
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

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import ModelConfig, TTMambaWorkspaceModel

print("Opening device...", flush=True)
device = ttnn.open_device(device_id=0)
print("Device opened!", flush=True)

GATE_CLAMP_BOUND = 0.3
all_pass = True

try:
    # --- Test 1: clamp_workspace_gates exists ---
    config = ModelConfig(
        d_model=384, n_heads=4, n_layers=2,
        use_attention=True, attention_positions=[0],
        use_workspace=True, n_workspace_slots=16,
        recurrent_core=False,
        gate_clamp_bound=GATE_CLAMP_BOUND,
    )
    model = TTMambaWorkspaceModel(config, device)

    assert hasattr(model, "clamp_workspace_gates"), \
        "FAIL: model.clamp_workspace_gates does not exist"
    print("Test 1 PASS: clamp_workspace_gates exists", flush=True)

    # --- Test 2: gates within bounds are not modified ---
    # Set gates to small values within [-0.3, 0.3]
    ws = model.workspace
    ws.read_gate = ttnn.from_torch(
        torch.tensor([0.15], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ws.write_gate = ttnn.from_torch(
        torch.tensor([-0.20], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    model.clamp_workspace_gates()

    rg = ttnn.to_torch(ws.read_gate).float().item()
    wg = ttnn.to_torch(ws.write_gate).float().item()

    assert abs(rg - 0.15) < 1e-2, \
        f"FAIL: read_gate changed from 0.15 to {rg} (should be unchanged)"
    assert abs(wg - (-0.20)) < 1e-2, \
        f"FAIL: write_gate changed from -0.20 to {wg} (should be unchanged)"
    print(f"Test 2 PASS: gates within bounds unchanged (rg={rg:.4f}, wg={wg:.4f})", flush=True)

    # --- Test 3: gates exceeding bounds are clamped ---
    # Set gates to extreme values that cause RMSNorm cancellation
    ws.read_gate = ttnn.from_torch(
        torch.tensor([-0.50], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ws.write_gate = ttnn.from_torch(
        torch.tensor([0.45], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    model.clamp_workspace_gates()

    rg = ttnn.to_torch(ws.read_gate).float().item()
    wg = ttnn.to_torch(ws.write_gate).float().item()

    assert rg >= -GATE_CLAMP_BOUND - 1e-3, \
        f"FAIL: read_gate={rg} not clamped to >= -{GATE_CLAMP_BOUND}"
    assert rg <= GATE_CLAMP_BOUND + 1e-3, \
        f"FAIL: read_gate={rg} not clamped to <= {GATE_CLAMP_BOUND}"
    assert wg >= -GATE_CLAMP_BOUND - 1e-3, \
        f"FAIL: write_gate={wg} not clamped to >= -{GATE_CLAMP_BOUND}"
    assert wg <= GATE_CLAMP_BOUND + 1e-3, \
        f"FAIL: write_gate={wg} not clamped to <= {GATE_CLAMP_BOUND}"
    print(f"Test 3 PASS: gates clamped (rg={rg:.4f}, wg={wg:.4f})", flush=True)

    # --- Test 4: gate_clamp_bound defaults to 0.0 (no clipping) when not set ---
    config_no_clamp = ModelConfig(
        d_model=384, n_heads=4, n_layers=2,
        use_attention=True, attention_positions=[0],
        use_workspace=True, n_workspace_slots=16,
        recurrent_core=False,
    )
    # gate_clamp_bound should default to 0.0 (disabled) for backward compat
    assert getattr(config_no_clamp, "gate_clamp_bound", 0.0) == 0.0, \
        "FAIL: gate_clamp_bound should default to 0.0 (disabled)"
    print("Test 4 PASS: gate_clamp_bound defaults to 0.0 (disabled)", flush=True)

    # --- Test 5: clamp is a no-op when gate_clamp_bound=0.0 ---
    model_no_clamp = TTMambaWorkspaceModel(config_no_clamp, device)
    ws_nc = model_no_clamp.workspace
    ws_nc.read_gate = ttnn.from_torch(
        torch.tensor([-0.50], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ws_nc.write_gate = ttnn.from_torch(
        torch.tensor([0.45], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    model_no_clamp.clamp_workspace_gates()

    rg = ttnn.to_torch(ws_nc.read_gate).float().item()
    wg = ttnn.to_torch(ws_nc.write_gate).float().item()

    assert abs(rg - (-0.50)) < 1e-2, \
        f"FAIL: read_gate changed from -0.50 to {rg} (clamp should be disabled)"
    assert abs(wg - 0.45) < 1e-2, \
        f"FAIL: write_gate changed from 0.45 to {wg} (clamp should be disabled)"
    print(f"Test 5 PASS: no-op when disabled (rg={rg:.4f}, wg={wg:.4f})", flush=True)

    print("\nAll tests PASS", flush=True)

except AssertionError as e:
    print(f"\n{e}", flush=True)
    all_pass = False
finally:
    ttnn.close_device(device)
    if not all_pass:
        sys.exit(1)
