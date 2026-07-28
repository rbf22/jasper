#!/usr/bin/env python3
"""CPU gradient check for the workspace backward logic.

Uses the PyTorch model's forward pass (which is verified correct) and compares
the manual backward pass (replicating TTWorkspaceModule.backward math) against
PyTorch autograd.

Key difference: ttnn.linear(x, W) computes x @ W, while PyTorch F.linear(x, W)
computes x @ W^T. The manual backward replicates the TT convention.
"""

import torch
import torch.nn.functional as F
import math
import sys
from model import WorkspaceModule, ModelConfig


def manual_softmax_backward(grad_attn, attn, dim=-1):
    """Manual softmax backward: grad_scores = (grad_attn - sum(grad_attn*attn, dim, keepdim)) * attn"""
    grad_sum = (grad_attn * attn).sum(dim=dim, keepdim=True)
    return (grad_attn - grad_sum) * attn


def rms_norm_backward(grad_out, x, weight_gamma, eps=1e-6):
    """RMSNorm backward for PyTorch model: y = x * (1 + gamma) / RMS(x).

    weight_gamma: the gamma parameter (init 0), NOT (1+gamma).
    The TT version stores weight = 1 + gamma, so grad_weight_tt = grad_gamma.
    """
    d = x.shape[-1]
    x_sq_mean = x.float().pow(2).mean(-1, keepdim=True)  # (B, T, 1)
    rms = torch.sqrt(x_sq_mean + eps)
    inv_rms = 1.0 / rms
    x_normed = x.float() * inv_rms  # (B, T, d)

    # In PyTorch model: y = x_normed * (1 + gamma)
    # grad w.r.t. (1+gamma) = grad_out * x_normed
    # grad w.r.t. gamma = grad_out * x_normed (same, since d(1+gamma)/d(gamma) = 1)
    w = 1.0 + weight_gamma  # actual weight applied
    grad_out_w = grad_out.float() * w  # (B, T, d)
    grad_out_w_rms = grad_out_w * inv_rms  # (B, T, d)

    grad_out_w_xnorm = grad_out_w * x_normed  # (B, T, d)
    grad_out_w_xnorm_mean = grad_out_w_xnorm.mean(-1, keepdim=True)  # (B, T, 1)

    correction = x_normed * grad_out_w_xnorm_mean * inv_rms  # (B, T, d)
    grad_x = grad_out_w_rms - correction

    # grad w.r.t. gamma (same as grad w.r.t. (1+gamma) since derivative is 1)
    grad_weight = (grad_out.float() * x_normed).sum(dim=(0, 1))  # (d,)

    return grad_x.to(x.dtype), grad_weight


def manual_workspace_backward(grad_x_out, grad_slots_out, ws, cache):
    """Replicates TTWorkspaceModule.backward in PyTorch.

    The TT model uses ttnn.linear(x, W) = x @ W.
    The PyTorch model uses F.linear(x, W) = x @ W^T.
    So TT's W is PyTorch's W^T.

    For the backward:
    - TT: y = x @ W_tt → grad_W_tt = x^T @ grad_y, grad_x = grad_y @ W_tt^T
    - PyTorch: y = x @ W_pt^T → grad_W_pt = grad_y^T @ x, grad_x = grad_y @ W_pt

    Since W_tt = W_pt^T:
    - grad_W_tt = x^T @ grad_y = (grad_y^T @ x)^T = grad_W_pt^T  ✓ (transposed)
    - grad_x = grad_y @ W_tt^T = grad_y @ W_pt = same as PyTorch  ✓ (same)

    So the input gradients match, and the weight gradients are transposed.
    We compute the TT-style weight gradients and compare with PyTorch's transposed.
    """
    c = cache
    B, T, m = c["B"], c["T"], c["m"]
    H = ws.n_heads
    d_h = ws.d_head  # D // H
    D = ws.d_model
    scale = c["scale"]

    # --- Backward through x_out = norm(x_pre_norm) ---
    grad_x_pre_norm, grad_norm_w = rms_norm_backward(grad_x_out, c["x_pre_norm"], ws.norm.weight, ws.norm.eps)

    # --- Backward through x_pre_norm = x + sigmoid(write_gate) * write_out_proj ---
    grad_x_from_write = grad_x_pre_norm  # residual
    grad_write_out_proj = grad_x_pre_norm * c["write_gate_val"]

    # grad_write_gate
    write_gate_val = c["write_gate_val"]
    write_sig_prime = write_gate_val * (1 - write_gate_val)
    grad_write_gate = (grad_x_pre_norm * c["write_out_proj"] * write_sig_prime).sum()

    # --- Backward through write_out_proj = F.linear(write_out, W) = write_out @ W^T ---
    # PyTorch forward: y = x @ W^T, so grad_x = grad_y @ W, grad_W = grad_y^T @ x
    # TT forward: y = x @ W_tt, grad_x = grad_y @ W_tt^T, grad_W = x^T @ grad_y
    # Since W_tt = W_pt^T: TT grad_x = grad_y @ W_pt = same as PyTorch grad_x
    # So for input grad: use matmul(grad_y, W) to match PyTorch convention
    grad_write_out = torch.matmul(grad_write_out_proj.reshape(-1, D), ws.write_out.weight).reshape(B, T, D)
    # TT weight grad: x^T @ grad_y (transposed compared to PyTorch's grad_y^T @ x)
    grad_write_out_weight = torch.matmul(
        c["write_out"].reshape(-1, D).T,
        grad_write_out_proj.reshape(-1, D)
    )  # x^T @ grad_y = (grad_y^T @ x)^T = grad_W_pt^T

    # --- Backward through write_out = reshape(write_attn @ wv) ---
    grad_write_out_4d = grad_write_out.reshape(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)
    wv = c["wv"]
    grad_write_attn = torch.matmul(grad_write_out_4d, wv.transpose(-2, -1))  # (B, H, T, m)
    grad_wv = torch.matmul(c["write_attn"].transpose(-2, -1), grad_write_out_4d)  # (B, H, m, d_h)

    # --- Backward through write_attn = softmax(write_scores * scale) ---
    grad_write_scores = manual_softmax_backward(grad_write_attn, c["write_attn"], dim=-1)
    grad_write_scores = grad_write_scores * scale

    # grad_wq = grad_write_scores @ wk, grad_wk = grad_write_scores^T @ wq
    wk = c["wk"]
    wq = c["wq"]
    grad_wq = torch.matmul(grad_write_scores, wk)  # (B, H, T, d_h)
    grad_wk = torch.matmul(grad_write_scores.transpose(-2, -1), wq)  # (B, H, m, d_h)

    # --- Backward through write projections ---
    grad_wq_3d = grad_wq.transpose(1, 2).contiguous().view(B, T, D)
    grad_wk_3d = grad_wk.transpose(1, 2).contiguous().view(B, m, D)
    grad_wv_3d = grad_wv.transpose(1, 2).contiguous().view(B, m, D)

    x_2d = c["x"].reshape(-1, D)  # (B*T, D)
    slots_out_2d = c["slots_out"].reshape(-1, D)  # (B*m, D)

    # TT weight grads: grad_W = x^T @ grad_y (transposed vs PyTorch's grad_y^T @ x)
    grad_write_q_weight = torch.matmul(x_2d.T, grad_wq_3d.reshape(-1, D))
    # Input grads: grad_y @ W (PyTorch convention, same as TT's grad_y @ W_tt^T = grad_y @ W_pt)
    grad_x_from_wq = torch.matmul(grad_wq_3d.reshape(-1, D), ws.write_q.weight).reshape(B, T, D)

    grad_wk_2d = grad_wk_3d.reshape(-1, D)
    grad_wv_2d = grad_wv_3d.reshape(-1, D)
    grad_write_k_weight = torch.matmul(slots_out_2d.T, grad_wk_2d)
    grad_write_v_weight = torch.matmul(slots_out_2d.T, grad_wv_2d)

    grad_slots_from_wk = torch.matmul(grad_wk_2d, ws.write_k.weight).reshape(B, m, D)
    grad_slots_from_wv = torch.matmul(grad_wv_2d, ws.write_v.weight).reshape(B, m, D)

    # Total grad_slots_out
    grad_slots_total = grad_slots_out + grad_slots_from_wk + grad_slots_from_wv

    # --- Backward through slots_out = slot_norm(slots_pre_norm) ---
    grad_slots_pre_norm, grad_slot_norm_w = rms_norm_backward(
        grad_slots_total, c["slots_pre_norm"], ws.slot_norm.weight, ws.slot_norm.eps
    )

    # --- Backward through slots_pre_norm = slots_in + sigmoid(read_gate) * read_out_proj ---
    grad_slots_in_from_read = grad_slots_pre_norm  # residual
    grad_read_out_proj = grad_slots_pre_norm * c["read_gate_val"]

    # grad_read_gate
    read_gate_val = c["read_gate_val"]
    read_sig_prime = read_gate_val * (1 - read_gate_val)
    grad_read_gate = (grad_slots_pre_norm * c["read_out_proj"] * read_sig_prime).sum()

    # --- Backward through read_out_proj = F.linear(read_out, W) ---
    grad_read_out = torch.matmul(grad_read_out_proj.reshape(-1, D), ws.read_out.weight).reshape(B, m, D)
    grad_read_out_weight = torch.matmul(
        c["read_out"].reshape(-1, D).T,
        grad_read_out_proj.reshape(-1, D)
    )

    # --- Backward through read_out = reshape(read_attn @ rv) ---
    grad_read_out_4d = grad_read_out.reshape(B, m, H, d_h).transpose(1, 2)  # (B, H, m, d_h)
    rv = c["rv"]
    grad_read_attn = torch.matmul(grad_read_out_4d, rv.transpose(-2, -1))  # (B, H, m, T)
    grad_rv = torch.matmul(c["read_attn"].transpose(-2, -1), grad_read_out_4d)  # (B, H, T, d_h)

    # --- Backward through read_attn = softmax(read_scores * scale) ---
    grad_read_scores = manual_softmax_backward(grad_read_attn, c["read_attn"], dim=-1)
    grad_read_scores = grad_read_scores * scale

    # grad_rq = grad_read_scores @ rk, grad_rk = grad_read_scores^T @ rq
    rk = c["rk"]
    rq = c["rq"]
    grad_rq = torch.matmul(grad_read_scores, rk)  # (B, H, m, d_h)
    grad_rk = torch.matmul(grad_read_scores.transpose(-2, -1), rq)  # (B, H, T, d_h)

    # --- Backward through read projections ---
    grad_rq_3d = grad_rq.transpose(1, 2).contiguous().view(B, m, D)
    grad_rk_3d = grad_rk.transpose(1, 2).contiguous().view(B, T, D)
    grad_rv_3d = grad_rv.transpose(1, 2).contiguous().view(B, T, D)

    slots_in_2d = c["slots_in"].reshape(-1, D)  # (B*m, D)

    grad_read_q_weight = torch.matmul(slots_in_2d.T, grad_rq_3d.reshape(-1, D))
    grad_slots_from_rq = torch.matmul(grad_rq_3d.reshape(-1, D), ws.read_q.weight).reshape(B, m, D)

    grad_rk_2d = grad_rk_3d.reshape(-1, D)
    grad_rv_2d = grad_rv_3d.reshape(-1, D)
    grad_read_k_weight = torch.matmul(x_2d.T, grad_rk_2d)
    grad_read_v_weight = torch.matmul(x_2d.T, grad_rv_2d)

    grad_x_from_rk = torch.matmul(grad_rk_2d, ws.read_k.weight).reshape(B, T, D)
    grad_x_from_rv = torch.matmul(grad_rv_2d, ws.read_v.weight).reshape(B, T, D)

    # --- Total gradients ---
    grad_x = grad_x_from_write + grad_x_from_wq + grad_x_from_rk + grad_x_from_rv
    grad_slots_in = grad_slots_in_from_read + grad_slots_from_rq

    grads = {
        "read_q_weight": grad_read_q_weight,
        "read_k_weight": grad_read_k_weight,
        "read_v_weight": grad_read_v_weight,
        "read_out_weight": grad_read_out_weight,
        "write_q_weight": grad_write_q_weight,
        "write_k_weight": grad_write_k_weight,
        "write_v_weight": grad_write_v_weight,
        "write_out_weight": grad_write_out_weight,
        "ws_norm_weight": grad_norm_w,
        "ws_slot_norm_weight": grad_slot_norm_w,
        "read_gate": grad_read_gate,
        "write_gate": grad_write_gate,
    }

    return grad_x, grad_slots_in, grads


def main():
    torch.manual_seed(42)

    D = 384
    m = 16
    H = 4
    B = 2
    T = 32

    config = ModelConfig(
        d_model=D,
        n_heads=H,
        n_workspace_slots=m,
        use_workspace=True,
    )

    ws = WorkspaceModule(config)

    # Create inputs
    x = torch.randn(B, T, D, dtype=torch.float32) * 0.1
    slot_state = None  # First call uses learned slots

    # --- PyTorch autograd reference ---
    x_ref = x.clone().requires_grad_(True)
    x_out_ref, slots_out_ref = ws(x_ref, slot_state)

    # Simple scalar loss: L = sum(x_out) + sum(slots_out)
    loss_ref = x_out_ref.sum() + slots_out_ref.sum()
    loss_ref.backward()

    grad_x_autograd = x_ref.grad
    grad_params_autograd = {}
    for name, param in ws.named_parameters():
        if param.grad is not None:
            grad_params_autograd[name] = param.grad.clone()

    # --- Manual forward (using PyTorch model's forward to get cache) ---
    x_man = x.clone().detach()

    # Run forward and cache intermediate values
    with torch.no_grad():
        B_, T_, D_ = x_man.shape
        H_ = ws.n_heads
        d_h_ = ws.d_head
        m_ = ws.n_slots
        scale = 1.0 / math.sqrt(d_h_)

        slots = ws.slots.unsqueeze(0).expand(B_, -1, -1).clone()

        rq = F.linear(slots, ws.read_q.weight).view(B_, m_, H_, d_h_).transpose(1, 2)
        rk = F.linear(x_man, ws.read_k.weight).view(B_, T_, H_, d_h_).transpose(1, 2)
        rv = F.linear(x_man, ws.read_v.weight).view(B_, T_, H_, d_h_).transpose(1, 2)

        read_scores = torch.matmul(rq, rk.transpose(-2, -1)) * scale
        read_attn = F.softmax(read_scores, dim=-1)
        read_out_4d = torch.matmul(read_attn, rv)
        read_out = read_out_4d.transpose(1, 2).contiguous().view(B_, m_, D_)
        read_out_proj = F.linear(read_out, ws.read_out.weight)

        read_gate_val = torch.sigmoid(ws.read_gate)
        slots_pre_norm = slots + read_gate_val * read_out_proj
        slots_out = ws.slot_norm(slots_pre_norm)

        wq = F.linear(x_man, ws.write_q.weight).view(B_, T_, H_, d_h_).transpose(1, 2)
        wk = F.linear(slots_out, ws.write_k.weight).view(B_, m_, H_, d_h_).transpose(1, 2)
        wv = F.linear(slots_out, ws.write_v.weight).view(B_, m_, H_, d_h_).transpose(1, 2)

        write_scores = torch.matmul(wq, wk.transpose(-2, -1)) * scale
        write_attn = F.softmax(write_scores, dim=-1)
        write_out_4d = torch.matmul(write_attn, wv)
        write_out = write_out_4d.transpose(1, 2).contiguous().view(B_, T_, D_)
        write_out_proj = F.linear(write_out, ws.write_out.weight)

        write_gate_val = torch.sigmoid(ws.write_gate)
        x_pre_norm = x_man + write_gate_val * write_out_proj
        x_out = ws.norm(x_pre_norm)

        cache = {
            "x": x_man, "slots_in": slots, "slots_pre_norm": slots_pre_norm,
            "rq": rq, "rk": rk, "rv": rv, "read_attn": read_attn, "read_out_4d": read_out_4d,
            "read_out": read_out, "read_out_proj": read_out_proj,
            "read_gate_val": read_gate_val,
            "slots_out": slots_out,
            "wq": wq, "wk": wk, "wv": wv, "write_attn": write_attn, "write_out_4d": write_out_4d,
            "write_out": write_out, "write_out_proj": write_out_proj,
            "write_gate_val": write_gate_val,
            "x_pre_norm": x_pre_norm, "scale": scale,
            "B": B_, "T": T_, "m": m_,
        }

    # Verify forward matches
    fwd_diff_x = (x_out_ref.detach() - x_out).abs().max().item()
    fwd_diff_s = (slots_out_ref.detach() - slots_out).abs().max().item()
    print(f"Forward match: x_out diff={fwd_diff_x:.8f}, slots_out diff={fwd_diff_s:.8f}")

    # --- Manual backward ---
    grad_x_out = torch.ones_like(x_out)
    grad_slots_out = torch.ones_like(slots_out)

    grad_x_man, grad_slots_in_man, ws_grads_man = manual_workspace_backward(
        grad_x_out, grad_slots_out, ws, cache
    )

    # --- Compare ---
    print(f"\n=== Gradient Comparison (manual vs autograd) ===\n")

    # Input gradient (should match exactly — same convention)
    diff = (grad_x_autograd - grad_x_man).abs()
    rel = diff / (grad_x_autograd.abs() + 1e-8)
    print(f"grad_x:       max_abs_diff={diff.max().item():.6f}, max_rel_err={rel.max().item():.6f}")

    # Weight gradients — TT uses x^T @ grad_y, PyTorch uses grad_y^T @ x
    # These are transposes of each other. Check transposed match.
    param_names = {
        "read_q_weight": "read_q.weight",
        "read_k_weight": "read_k.weight",
        "read_v_weight": "read_v.weight",
        "read_out_weight": "read_out.weight",
        "write_q_weight": "write_q.weight",
        "write_k_weight": "write_k.weight",
        "write_v_weight": "write_v.weight",
        "write_out_weight": "write_out.weight",
        "ws_norm_weight": "norm.weight",
        "ws_slot_norm_weight": "slot_norm.weight",
        "read_gate": "read_gate",
        "write_gate": "write_gate",
    }

    all_pass = True
    for tt_name, pt_name in param_names.items():
        man_grad = ws_grads_man[tt_name]
        auto_grad = grad_params_autograd[pt_name]

        if "gate" in pt_name:
            # Scalar — should match directly
            diff = (auto_grad - man_grad).abs().max().item()
            rel = diff / (auto_grad.abs().item() + 1e-8)
            status = "PASS" if rel < 0.01 else "FAIL"
            print(f"{tt_name:25s}: diff={diff:.6f}, rel_err={rel:.6f} {status}")
            if rel > 0.01:
                all_pass = False
        else:
            # Weight matrices — TT grad is transposed vs PyTorch
            # For 1-D tensors (norm weights), no transpose needed
            if man_grad.dim() <= 1:
                diff_d = (auto_grad - man_grad).abs().max().item()
                rel = diff_d / (auto_grad.abs() + man_grad.abs() + 1e-8).max().item()
                status = "PASS" if rel < 0.01 else "FAIL"
                print(f"{tt_name:25s}: direct match (1-D), diff={diff_d:.6f}, rel_err={rel:.6f} {status}")
            else:
                diff_t = (auto_grad - man_grad.mT).abs().max().item()
                diff_d = (auto_grad - man_grad).abs().max().item()
                if diff_t < diff_d:
                    rel = diff_t / (auto_grad.abs() + man_grad.mT.abs() + 1e-8).max().item()
                    status = "PASS" if rel < 0.01 else "FAIL"
                    print(f"{tt_name:25s}: TRANSPOSED match, diff={diff_t:.6f}, rel_err={rel:.6f} {status}")
                else:
                    rel = diff_d / (auto_grad.abs() + man_grad.abs() + 1e-8).max().item()
                    status = "PASS" if rel < 0.01 else "FAIL"
                    print(f"{tt_name:25s}: direct match, diff={diff_d:.6f}, rel_err={rel:.6f} {status}")
            if rel > 0.01:
                all_pass = False

    # Slots gradient
    grad_slots_param_man = grad_slots_in_man.sum(dim=0)  # (m, D)
    auto_slots_grad = grad_params_autograd["slots"]
    diff = (auto_slots_grad - grad_slots_param_man).abs().max().item()
    rel = diff / (auto_slots_grad.abs() + 1e-8).max().item()
    status = "PASS" if rel < 0.01 else "FAIL"
    print(f"{'ws_slots':25s}: diff={diff:.6f}, rel_err={rel:.6f} {status}")
    if rel > 0.01:
        all_pass = False

    print(f"\n=== Summary ===")
    if all_pass:
        print("VERDICT: Manual backward matches PyTorch autograd!")
        print("The workspace backward math is CORRECT.")
    else:
        print("VERDICT: Mismatches detected — investigate failures above.")


if __name__ == "__main__":
    main()
