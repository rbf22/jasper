#!/usr/bin/env python3
"""Test that cross_entropy_loss_scatter matches cross_entropy_loss_host.

Creates random logits and labels, computes loss+gradient with both methods,
and compares. The scatter loss runs in bfloat16 on device, the host loss in
float32 on host — so we expect ~1-2% relative error (bf16 noise floor).
"""

import os
import sys
import torch
import ttnn

# Set up paths
MVP_DIR = os.path.dirname(os.path.abspath(__file__))
POC_DIR = os.path.realpath(os.path.join(MVP_DIR, "..", "workspace-poc"))
if not os.path.isdir(POC_DIR):
    POC_DIR = os.path.realpath(os.path.join(MVP_DIR, "..", "workspace-poc"))
sys.path.insert(0, POC_DIR)
sys.path.insert(0, MVP_DIR)

from train_text import cross_entropy_loss_scatter, cross_entropy_loss_host


def test_scatter_vs_host():
    """Compare scatter-based and host-based loss on the same inputs."""
    device = ttnn.open_device(device_id=0)

    try:
        # Test with a small vocab first (to verify correctness)
        # Then with a larger vocab (closer to production)
        test_configs = [
            {"B": 2, "T": 64, "V": 256, "name": "small"},
            {"B": 4, "T": 128, "V": 1024, "name": "medium"},
            {"B": 2, "T": 64, "V": 4096, "name": "large"},
        ]

        for cfg in test_configs:
            B, T, V = cfg["B"], cfg["T"], cfg["V"]
            # Tile-align V
            V_padded = ((V + 31) // 32) * 32
            print(f"\n--- {cfg['name']}: B={B}, T={T}, V={V} (padded={V_padded}) ---")

            torch.manual_seed(42)
            # Random logits
            logits = torch.randn(B, T, V_padded, dtype=torch.bfloat16) * 2.0
            # Random labels: some valid, some -100 (ignore)
            labels = torch.randint(0, V, (B, T), dtype=torch.long)
            # Set last position to -100 (no next token)
            labels[:, -1] = -100
            # Set a few random positions to -100
            for b in range(B):
                n_ignore = torch.randint(0, T // 4, (1,)).item()
                if n_ignore > 0:
                    idx = torch.randperm(T - 1)[:n_ignore]
                    labels[b, idx] = -100

            # Transfer logits to device
            logits_tt = ttnn.from_torch(
                logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # Host loss (ground truth in float32)
            loss_host, grad_host = cross_entropy_loss_host(logits_tt, labels)

            # Scatter loss (on device)
            loss_scatter, grad_scatter = cross_entropy_loss_scatter(logits_tt, labels)

            # Compare loss values
            loss_diff = abs(loss_host - loss_scatter)
            loss_rel = loss_diff / max(abs(loss_host), 1e-8)
            print(f"  Loss:   host={loss_host:.6f}  scatter={loss_scatter:.6f}  "
                  f"diff={loss_diff:.6f}  rel={loss_rel:.4f}")

            # Compare gradients
            grad_host_torch = ttnn.to_torch(grad_host).float()  # (B, T, V_padded)
            grad_scatter_torch = ttnn.to_torch(grad_scatter).float()  # (B, T, V_padded)

            # Only compare over the real V (not padding)
            grad_host_real = grad_host_torch[:, :, :V]
            grad_scatter_real = grad_scatter_torch[:, :, :V]

            # Check valid positions only (where label != -100)
            valid_mask = (labels != -100).float()
            # Expand mask to (B, T, V) for comparison
            valid_mask_3d = valid_mask.unsqueeze(-1).expand(-1, -1, V)

            # Relative error on valid positions
            diff = (grad_host_real - grad_scatter_real).abs()
            denom = grad_host_real.abs().clamp(min=1e-6)
            rel_err = (diff / denom * valid_mask_3d).sum() / valid_mask_3d.sum().clamp(min=1)

            # Max absolute error
            max_err = (diff * valid_mask_3d).max().item()

            # Check that invalid positions have zero gradient
            invalid_mask_3d = (1 - valid_mask).unsqueeze(-1).expand(-1, -1, V)
            grad_scatter_invalid = (grad_scatter_real.abs() * invalid_mask_3d).max().item()
            grad_host_invalid = (grad_host_real.abs() * invalid_mask_3d).max().item()

            print(f"  Grad:   mean_rel_err={rel_err:.4f}  max_abs_err={max_err:.6f}")
            print(f"  Invalid pos grad:  scatter={grad_scatter_invalid:.8f}  host={grad_host_invalid:.8f}")

            # Assertions
            assert loss_rel < 0.05, f"Loss relative error too high: {loss_rel}"
            assert rel_err < 0.10, f"Gradient relative error too high: {rel_err}"
            assert grad_scatter_invalid < 1e-4, f"Scatter grad at invalid positions too high: {grad_scatter_invalid}"
            print(f"  PASSED")

        print("\n=== All tests passed ===")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_scatter_vs_host()
