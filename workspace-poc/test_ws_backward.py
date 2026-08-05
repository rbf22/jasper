#!/usr/bin/env python3
"""Numerical gradient check for the TTWorkspaceModule backward pass.

Compares analytic gradients from workspace.backward() against finite-difference
gradients computed on the forward pass.

Run on a free device (not during training):
    python test_ws_backward.py --device 0
"""

import os
import sys
import argparse
import torch
import ttnn
import numpy as np

# Set TT_VISIBLE_DEVICES before importing model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--B", type=int, default=1, help="batch size")
    parser.add_argument("--T", type=int, default=32, help="sequence length")
    parser.add_argument("--D", type=int, default=384, help="d_model")
    parser.add_argument("--m", type=int, default=16, help="n_workspace_slots")
    parser.add_argument("--H", type=int, default=4, help="n_heads")
    parser.add_argument("--eps", type=float, default=1e-2, help="finite diff epsilon")
    args = parser.parse_args()

    os.environ["TT_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", "p150_mesh_graph_descriptor.textproto")

    from model_ttnn import TTWorkspaceModule, ModelConfig

    device = ttnn.open_device(device_id=0)

    config = ModelConfig(
        d_model=args.D,
        n_heads=args.H,
        n_workspace_slots=args.m,
        use_workspace=True,
    )

    ws = TTWorkspaceModule(config, device)

    B, T, D, m = args.B, args.T, args.D, args.m

    # Create deterministic inputs
    torch.manual_seed(42)
    x_torch = torch.randn(B, T, D, dtype=torch.bfloat16) * 0.1
    slot_state_torch = None  # First call: use learned slots

    # --- Forward pass ---
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_out, slots_out = ws.forward(x_tt, slot_state_torch)

    # Create a simple scalar loss: L = sum(x_out) + sum(slots_out)
    # So grad_x_out = ones, grad_slots_out = ones
    grad_x_out_torch = torch.ones(B, T, D, dtype=torch.bfloat16)
    grad_slots_out_torch = torch.ones(B, m, D, dtype=torch.bfloat16)

    grad_x_out = ttnn.from_torch(grad_x_out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_slots_out = ttnn.from_torch(grad_slots_out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # --- Analytic backward ---
    grad_x, grad_slots_in, ws_grads = ws.backward(grad_x_out, grad_slots_out)

    # Convert to torch for comparison
    grad_x_analytic = ttnn.to_torch(grad_x).float()
    grad_slots_in_analytic = ttnn.to_torch(grad_slots_in).float()

    # --- Finite difference check ---
    # We perturb each input element and measure the change in L = sum(x_out) + sum(slots_out)
    def compute_loss(x_modified):
        x_tt_mod = ttnn.from_torch(x_modified, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        x_out_mod, slots_out_mod = ws.forward(x_tt_mod, slot_state_torch)
        loss = ttnn.to_torch(x_out_mod).float().sum().item() + ttnn.to_torch(slots_out_mod).float().sum().item()
        return loss

    # Check a few random elements of grad_x
    n_check = min(20, B * T * D)
    indices = torch.randint(0, B * T * D, (n_check,))
    x_flat = x_torch.clone()

    print(f"Gradient check for x (input to workspace):")
    print(f"  B={B}, T={T}, D={D}, m={m}, H={args.H}")
    print(f"  Checking {n_check} random elements with eps={args.eps}")
    print()

    max_rel_error = 0.0
    n_pass = 0
    for idx in indices:
        b = idx.item() // (T * D)
        t = (idx.item() % (T * D)) // D
        d = idx.item() % D

        x_plus = x_flat.clone()
        x_plus[b, t, d] += args.eps
        x_minus = x_flat.clone()
        x_minus[b, t, d] -= args.eps

        loss_plus = compute_loss(x_plus)
        loss_minus = compute_loss(x_minus)

        grad_fd = (loss_plus - loss_minus) / (2 * args.eps)
        grad_an = grad_x_analytic[b, t, d].item()

        if abs(grad_fd) > 1e-6:
            rel_error = abs(grad_fd - grad_an) / max(abs(grad_fd), abs(grad_an))
        else:
            rel_error = abs(grad_fd - grad_an)

        max_rel_error = max(max_rel_error, rel_error)
        status = "PASS" if rel_error < 0.1 else "FAIL"
        if status == "PASS":
            n_pass += 1
        if status == "FAIL" or rel_error > 0.01:
            print(f"  [{b},{t},{d}] fd={grad_fd:.6f}, analytic={grad_an:.6f}, rel_err={rel_error:.4f} {status}")

    print(f"\n  Passed: {n_pass}/{n_check}, max relative error: {max_rel_error:.4f}")

    # --- Check weight gradients ---
    print(f"\nGradient check for workspace weights:")

    # Check read_q_weight
    w = ws.read_q_weight
    w_torch = ttnn.to_torch(w).float()
    grad_w_analytic = ttnn.to_torch(ws_grads["read_q_weight"]).float()

    n_check_w = min(20, D * D)
    indices_w = torch.randint(0, D * D, (n_check_w,))

    max_rel_error_w = 0.0
    n_pass_w = 0

    for idx in indices_w:
        i = idx.item() // D
        j = idx.item() % D

        w_plus = w_torch.clone()
        w_plus[i, j] += args.eps
        w_minus = w_torch.clone()
        w_minus[i, j] -= args.eps

        w_plus_tt = ttnn.from_torch(w_plus.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w_minus_tt = ttnn.from_torch(w_minus.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        orig_w = ws.read_q_weight
        ws.read_q_weight = w_plus_tt
        loss_plus = compute_loss(x_flat)
        ws.read_q_weight = w_minus_tt
        loss_minus = compute_loss(x_flat)
        ws.read_q_weight = orig_w  # restore

        grad_fd = (loss_plus - loss_minus) / (2 * args.eps)
        grad_an = grad_w_analytic[i, j].item()

        if abs(grad_fd) > 1e-6:
            rel_error = abs(grad_fd - grad_an) / max(abs(grad_fd), abs(grad_an))
        else:
            rel_error = abs(grad_fd - grad_an)

        max_rel_error_w = max(max_rel_error_w, rel_error)
        status = "PASS" if rel_error < 0.15 else "FAIL"
        if status == "PASS":
            n_pass_w += 1
        if status == "FAIL" or rel_error > 0.01:
            print(f"  read_q_weight[{i},{j}] fd={grad_fd:.6f}, analytic={grad_an:.6f}, rel_err={rel_error:.4f} {status}")

    print(f"\n  read_q_weight: Passed {n_pass_w}/{n_check_w}, max rel error: {max_rel_error_w:.4f}")

    # --- Check gate gradients ---
    print(f"\nGate gradients (analytic only, verify non-zero):")
    for gate_name in ["read_gate", "write_gate"]:
        grad = ttnn.to_torch(ws_grads[gate_name]).float().item()
        print(f"  {gate_name}: grad = {grad:.6f}")

    # --- Check norm weight gradients ---
    print(f"\nNorm weight gradients (analytic):")
    for norm_name in ["ws_norm_weight", "ws_slot_norm_weight"]:
        grad = ttnn.to_torch(ws_grads[norm_name]).float()
        print(f"  {norm_name}: shape={grad.shape}, mean={grad.mean().item():.6f}, max_abs={grad.abs().max().item():.6f}")

    print(f"\n=== Summary ===")
    print(f"Input gradient:  {n_pass}/{n_check} passed, max error {max_rel_error:.4f}")
    print(f"Weight gradient: {n_pass_w}/{n_check_w} passed, max error {max_rel_error_w:.4f}")
    if max_rel_error < 0.1 and max_rel_error_w < 0.15:
        print("VERDICT: Backward pass is CORRECT (within bf16 precision)")
    else:
        print("VERDICT: Backward pass has ERRORS — investigate failures above")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
