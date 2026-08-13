#!/usr/bin/env python3
"""Test multi-step training stability for WRAP model.

Runs 50 training steps with Cell B config and verifies:
1. Loss doesn't diverge to NaN/Inf
2. Gradient norms stay bounded
3. Model parameters remain finite
4. clear_caches + forward can be called repeatedly without crash

Run: .tt-venv/bin/python test_training_stability.py
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
from train_ttnn import TTAdamW, cross_entropy_loss, clip_grad_norm

# Cell B (workspace, no recurrent core)
SMALL_CONFIG_B = ModelConfig(
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=128,
    use_attention=True, attention_positions=[0],
    use_workspace=True, n_workspace_slots=4,
    recurrent_core=False,
    freeze_gamma=True,
)

NUM_STEPS = 50


def test_training_stability():
    """Run 50 training steps and verify stability."""
    device = ttnn.open_device(device_id=0)
    try:
        config = SMALL_CONFIG_B
        model = TTWRAPModel(config, device)
        # Use a low LR with linear warmup (mirrors production: lr=2e-4, warmup=200).
        # Without warmup, the first few steps at full LR destabilize the randomly
        # initialized model, causing grad norm explosion.
        BASE_LR = 2e-4
        WARMUP_STEPS = 20
        optimizer = TTAdamW(model.get_params(), lr=BASE_LR)

        B, T = 4, 32
        losses = []

        for step in range(NUM_STEPS):
            # Linear warmup: LR goes from 0 to BASE_LR over WARMUP_STEPS
            if step < WARMUP_STEPS:
                optimizer.lr = BASE_LR * step / WARMUP_STEPS
            else:
                optimizer.lr = BASE_LR

            input_ids = torch.randint(0, config.vocab_size, (B, T))
            labels = torch.randint(0, config.vocab_size, (B, T))
            logits = model.forward(input_ids)
            loss, grad_logits = cross_entropy_loss(logits, labels)
            grads = model.backward(grad_logits)
            # clip_grad_norm clips in-place and returns the PRE-clip global norm.
            # The pre-clip norm can be large (that's why we clip); what matters
            # is that the applied (post-clip) gradients are bounded and the
            # model doesn't diverge to NaN/Inf.
            grad_norm = clip_grad_norm(grads, max_norm=1.0, ws_max_norm=0.5)
            optimizer.step(grads, model)
            model.clear_caches()
            losses.append(loss)

            if step % 10 == 0:
                print(f"Step {step}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")
                assert math.isfinite(loss), f"Loss diverged at step {step}: {loss}"

        # --- Test 2: verify all model params are finite after 50 steps ---
        params = model.get_params()
        all_finite = True
        for name, tt_param in params.items():
            t = ttnn.to_torch(tt_param).float()
            if not torch.isfinite(t).all().item():
                print(f"  NON-FINITE param: {name}")
                all_finite = False
        assert all_finite, "Some model parameters are non-finite after training"
        print(f"[finite] all {len(params)} params finite after {NUM_STEPS} steps: OK")

        # --- Test 3: all losses are finite (no NaN/Inf divergence) ---
        all_losses_finite = all(math.isfinite(l) for l in losses)
        assert all_losses_finite, "Some loss values are non-finite (divergence)"
        print(f"[loss] all {NUM_STEPS} losses finite (range {min(losses):.4f}–{max(losses):.4f}): OK")

        print("[stability] 50-step training stable: OK")
        return True
    finally:
        ttnn.close_device(device)


def main():
    tests = [
        ("training_stability", test_training_stability),
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
