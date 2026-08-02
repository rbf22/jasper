"""Profile TTRetentionLayer to find throughput bottlenecks.

Runs N forward+backward passes with per-op timing, then reports:
  - Per-op time (absolute and percentage)
  - Host-device transfer time (decay matrix, gamma grad reduction)
  - Identify whether we're compute-bound or overhead-bound

Usage:
    TT_VISIBLE_DEVICES=0 .tt-venv/bin/python profile_retention.py
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Device selection from argv
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

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ttnn import ModelConfig, to_device, TTRetentionLayer


class OpProfiler:
    """Wrap ttnn ops to measure per-op time."""
    def __init__(self):
        self.times = {}
        self.counts = {}

    def time(self, name):
        class T:
            def __init__(s, profiler, section):
                s.p = profiler
                s.s = section
            def __enter__(s):
                s.t0 = time.perf_counter()
                return s
            def __exit__(s, *a):
                dt = time.perf_counter() - s.t0
                s.p.times[s.s] = s.p.times.get(s.s, 0) + dt
                s.p.counts[s.s] = s.p.counts.get(s.s, 0) + 1
        return T(self, name)

    def report(self, n_iters):
        total = sum(self.times.values())
        print(f"\n{'='*80}")
        print(f"Per-op profile ({n_iters} fwd+bwd iterations)")
        print(f"{'='*80}")
        print(f"  {'Op':<35s} {'Total':>8s} {'Pct':>6s} {'Per-iter':>10s} {'Calls':>6s}")
        print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*10} {'-'*6}")
        for name in sorted(self.times.keys(), key=lambda x: -self.times[x]):
            t = self.times[name]
            n = self.counts[name]
            pct = 100 * t / total if total > 0 else 0
            per_iter = t / n_iters
            per_call = n / n_iters
            print(f"  {name:<35s} {t:>7.3f}s {pct:>5.1f}% {per_iter:>9.4f}s {per_call:>5.1f}x")
        print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*10} {'-'*6}")
        print(f"  {'TOTAL':<35s} {total:>7.3f}s {'':>5s} {total/n_iters:>9.4f}s")
        print(f"\n  Wall-clock per fwd+bwd: {total/n_iters:.4f}s")
        print(f"  Throughput: {n_iters * B * T / total:.0f} samples/s  ({n_iters * B * T * D / total / 1e9:.2f} GFLOP/s raw)")
        return total


def profile_retention(config, B, T, n_warmup, n_iters, device):
    """Profile retention layer forward + backward."""
    p = OpProfiler()

    # Create layer
    layer = TTRetentionLayer(config, device)

    # Create input and grad
    x_torch = torch.randn(B, T, config.d_model, dtype=torch.bfloat16) * 0.5
    grad_torch = torch.randn(B, T, config.d_model, dtype=torch.bfloat16) * 0.5

    D = config.d_model
    H = config.n_heads
    d_h = config.d_model // config.n_heads

    print(f"Config: d_model={D}, n_heads={H}, d_head={d_h}, B={B}, T={T}")
    print(f"Warmup: {n_warmup} iters, Profile: {n_iters} iters")

    # Warmup (untimed)
    for _ in range(n_warmup):
        layer.forward(to_device(x_torch, device))
        layer.backward(to_device(grad_torch, device))
    ttnn.synchronize_device(device)

    # Profiled iterations — manually instrument the forward and backward
    # by re-implementing them with timing wrappers around each op.
    # This duplicates the layer logic but gives us per-op timing.
    for _ in range(n_iters):
        # ===== FORWARD =====
        x = to_device(x_torch, device)
        grad_out = to_device(grad_torch, device)

        with p.time("fwd: rope_init (host)"):
            layer._init_rope(T, device)

        with p.time("fwd: qkv_linear"):
            qkvg = ttnn.linear(x, layer.qkv_weight)

        with p.time("fwd: qkv_slice"):
            q = ttnn.slice(qkvg, [0, 0, 0], [B, T, D])
            k = ttnn.slice(qkvg, [0, 0, D], [B, T, 2 * D])
            v = ttnn.slice(qkvg, [0, 0, 2 * D], [B, T, 3 * D])
            g = ttnn.slice(qkvg, [0, 0, 3 * D], [B, T, 4 * D])

        with p.time("fwd: reshape_qkv"):
            q_4d = ttnn.permute(ttnn.reshape(q, [B, T, H, d_h]), [0, 2, 1, 3])
            k_4d = ttnn.permute(ttnn.reshape(k, [B, T, H, d_h]), [0, 2, 1, 3])
            v_4d = ttnn.permute(ttnn.reshape(v, [B, T, H, d_h]), [0, 2, 1, 3])

        with p.time("fwd: rope_q"):
            rot1_q, rot2_q = layer._apply_rope_split(q_4d, B, H, T)

        with p.time("fwd: rope_k"):
            rot1_k, rot2_k = layer._apply_rope_split(k_4d, B, H, T)

        with p.time("fwd: decay_matrix (device)"):
            D_decay = layer._get_decay_matrix(T, device)

        with p.time("fwd: qk_matmul (split)"):
            qk = ttnn.add(
                ttnn.matmul(rot1_q, ttnn.transpose(rot1_k, -2, -1)),
                ttnn.matmul(rot2_q, ttnn.transpose(rot2_k, -2, -1))
            )

        with p.time("fwd: scale_tt_create"):
            scale_tt = layer._scale_tt

        with p.time("fwd: fused_scale_decay"):
            qk_2d = ttnn.reshape(qk, [B * H * T, T])
            D_decay_2d = ttnn.reshape(D_decay, [H * T, T])
            scores_2d = layer._fused_scale_decay(qk_2d, D_decay_2d, layer.scale, B, H, T, device)
            scores = ttnn.reshape(scores_2d, [B, H, T, T])

        with p.time("fwd: sv_matmul"):
            out_4d = ttnn.matmul(scores, v_4d)

        with p.time("fwd: out_reshape"):
            out_flat = ttnn.reshape(ttnn.permute(out_4d, [0, 2, 1, 3]), [B, T, D])

        with p.time("fwd: sigmoid_gate"):
            gate = ttnn.sigmoid(g)
            out_gated = ttnn.mul(out_flat, gate)

        with p.time("fwd: out_linear"):
            out = ttnn.linear(out_gated, layer.out_proj_weight)

        # ===== BACKWARD =====
        with p.time("bwd: out_grad_linear"):
            grad_out_gated = ttnn.linear(grad_out, ttnn.transpose(layer.out_proj_weight, 0, 1))

        with p.time("bwd: out_proj_weight_grad"):
            out_gated_2d = ttnn.reshape(out_gated, [B * T, D])
            grad_out_2d = ttnn.reshape(grad_out, [B * T, D])
            grad_out_proj_weight = ttnn.matmul(ttnn.transpose(out_gated_2d, 0, 1), grad_out_2d)

        with p.time("bwd: gate_backward (fused)"):
            grad_out_flat, grad_g = layer._fused_gate_backward(
                grad_out_gated, gate, out_flat, B, T, D, device)

        with p.time("bwd: out_reshape_grad"):
            grad_out_4d = ttnn.permute(ttnn.reshape(grad_out_flat, [B, T, H, d_h]), [0, 2, 1, 3])

        with p.time("bwd: scores_grad_matmul"):
            grad_scores = ttnn.matmul(grad_out_4d, ttnn.transpose(v_4d, -2, -1))

        with p.time("bwd: v_grad_matmul"):
            grad_v_4d = ttnn.matmul(ttnn.transpose(scores, -2, -1), grad_out_4d)

        with p.time("bwd: fused_scale_decay_grad"):
            grad_scores_2d = ttnn.reshape(grad_scores, [B * H * T, T])
            D_decay_2d = ttnn.reshape(D_decay, [H * T, T])
            grad_scores_scaled_2d = layer._fused_scale_decay(
                grad_scores_2d, D_decay_2d, layer.scale, B, H, T, device)
            grad_scores_scaled = ttnn.reshape(grad_scores_scaled_2d, [B, H, T, T])

        with p.time("bwd: grad_D_decay (qk_scaled recompute)"):
            qk_scaled = ttnn.mul(qk, scale_tt)
            grad_D_decay = ttnn.mul(grad_scores, qk_scaled)

        with p.time("bwd: gamma_grad (device reduction)"):
            weighted = ttnn.mul(grad_D_decay, D_decay)
            weighted = ttnn.mul(weighted, layer._diff_tt)
            grad_log_gamma = ttnn.sum(weighted, dim=0)
            grad_log_gamma = ttnn.sum(grad_log_gamma, dim=1)
            grad_log_gamma = ttnn.sum(grad_log_gamma, dim=1)
            grad_gamma = ttnn.typecast(grad_log_gamma, ttnn.float32)

        with p.time("bwd: q_grad_matmul (split)"):
            grad_rot1_q = ttnn.matmul(grad_scores_scaled, rot1_k)
            grad_rot2_q = ttnn.matmul(grad_scores_scaled, rot2_k)

        with p.time("bwd: k_grad_matmul (split)"):
            grad_rot1_k = ttnn.matmul(ttnn.transpose(grad_scores_scaled, -2, -1), rot1_q)
            grad_rot2_k = ttnn.matmul(ttnn.transpose(grad_scores_scaled, -2, -1), rot2_q)

        with p.time("bwd: rope_q_backward"):
            grad_q_4d = layer._apply_rope_backward_split(grad_rot1_q, grad_rot2_q, B, H, T)

        with p.time("bwd: rope_k_backward"):
            grad_k_4d = layer._apply_rope_backward_split(grad_rot1_k, grad_rot2_k, B, H, T)

        with p.time("bwd: reshape_grads"):
            grad_q = ttnn.reshape(ttnn.permute(grad_q_4d, [0, 2, 1, 3]), [B, T, D])
            grad_k = ttnn.reshape(ttnn.permute(grad_k_4d, [0, 2, 1, 3]), [B, T, D])
            grad_v = ttnn.reshape(ttnn.permute(grad_v_4d, [0, 2, 1, 3]), [B, T, D])

        with p.time("bwd: qkv_concat"):
            grad_qkvg = ttnn.concat([grad_q, grad_k, grad_v, grad_g], dim=-1)

        with p.time("bwd: qkv_weight_grad"):
            x_2d = ttnn.reshape(x, [B * T, D])
            grad_qkvg_2d = ttnn.reshape(grad_qkvg, [B * T, 4 * D])
            grad_qkv_weight = ttnn.matmul(ttnn.transpose(x_2d, 0, 1), grad_qkvg_2d)

        with p.time("bwd: x_grad_matmul"):
            grad_x = ttnn.matmul(grad_qkvg_2d, ttnn.transpose(layer.qkv_weight, 0, 1))
            grad_x = ttnn.reshape(grad_x, [B, T, D])

        # Force sync so timing is accurate
        ttnn.synchronize_device(device)

    total = p.report(n_iters)

    # Also measure the un-instrumented layer for comparison
    print(f"\n--- Un-instrumented comparison ---")
    for _ in range(n_warmup):
        layer.forward(to_device(x_torch, device))
        layer.backward(to_device(grad_torch, device))
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        layer.forward(to_device(x_torch, device))
        layer.backward(to_device(grad_torch, device))
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    raw_per_iter = (t1 - t0) / n_iters
    print(f"  Un-instrumented: {raw_per_iter:.4f}s/iter")
    print(f"  Instrumented:    {total/n_iters:.4f}s/iter")
    print(f"  Profiling overhead: {((total/n_iters - raw_per_iter) / raw_per_iter * 100):.1f}%")

    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    config = ModelConfig(d_model=args.d_model, n_heads=args.n_heads)
    B, T = args.batch, args.seq_len

    print(f"Opening device {args.device}...")
    device = ttnn.open_device(device_id=0)
    try:
        profile_retention(config, B, T, args.warmup, args.iters, device)
    finally:
        ttnn.close_device(device)
