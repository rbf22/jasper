"""
Memory leak isolation tests for WRAP's custom kernels and layer backward passes.

These tests require a Tenstorrent device and are NOT in pytest.ini.
Run manually:

    cd /home/rfenwick/Documents/jasper/workspace-poc
    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... \
    .tt-venv/bin/python test_memory_leak.py

Each test runs a specific operation N times and reports RSS growth.
A leak is indicated by sustained linear growth that does NOT plateau.

Usage:
    # Run all tests
    python test_memory_leak.py

    # Run specific test by name
    python test_memory_leak.py --test retention_forward
    python test_memory_leak.py --test retention_backward
    python test_memory_leak.py --test custom_rope
    python test_memory_leak.py --test custom_scale_decay
    python test_memory_leak.py --test custom_gate_backward

    # Adjust iterations
    python test_memory_leak.py --iterations 100
"""

import argparse
import gc
import ctypes
import os
import sys
import time

import torch
import ttnn

# Ensure workspace-poc is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rss_mb() -> float:
    """Current process RSS in MB (resident pages only)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024


def gc_trim():
    """GC + malloc_trim to return fragmented memory to the OS."""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def measure_loop(label: str, n_iters: int, body_fn, warmup_fn=None):
    """Run body_fn n_iters times, measuring RSS every few iterations.

    Prints a table of iteration / RSS / delta and a final verdict.
    Returns (total_delta_mb, steady_state_rate_mb_per_iter).
    """
    # Warmup (one-time allocator/kernel-cache initialization)
    if warmup_fn:
        for _ in range(3):
            warmup_fn()
        ttnn.synchronize_device(_get_device())
    gc_trim()
    rss0 = rss_mb()
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  Baseline RSS: {rss0:.0f} MB")
    print(f"  {'Iter':>6}  {'RSS (MB)':>10}  {'Delta':>8}  {'Rate':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*12}")

    checkpoints = set()
    if n_iters <= 20:
        checkpoints = set(range(n_iters))
    else:
        checkpoints = {0, 4, 9}
        checkpoints.update(range(0, n_iters, max(1, n_iters // 10)))
        checkpoints.add(n_iters - 1)

    last_rss = rss0
    rates = []
    for i in range(n_iters):
        body_fn(i)
        ttnn.synchronize_device(_get_device())
        if i in checkpoints:
            gc_trim()
            rss = rss_mb()
            delta = rss - rss0
            rate = delta / (i + 1)
            rates.append(rate)
            print(f"  {i:6d}  {rss:10.0f}  {delta:+8.0f}  {rate:8.2f} MB/it")
            last_rss = rss

    total_delta = last_rss - rss0
    # Steady-state: average rate of last 50% of checkpoints
    if len(rates) >= 4:
        steady_rates = rates[len(rates) // 2:]
        steady_rate = sum(steady_rates) / len(steady_rates)
    else:
        steady_rate = total_delta / n_iters

    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*12}")
    print(f"  TOTAL: {total_delta:+.0f} MB over {n_iters} iters")
    print(f"  Steady-state rate: {steady_rate:.2f} MB/iter")

    if steady_rate < 0.1:
        verdict = "NO LEAK"
    elif steady_rate < 0.5:
        verdict = "MINOR (likely allocator fragmentation)"
    elif steady_rate < 2.0:
        verdict = "MODERATE LEAK"
    else:
        verdict = "SEVERE LEAK"
    print(f"  Verdict: {verdict}")

    return total_delta, steady_rate


# ---------------------------------------------------------------------------
# Global device handle (lazily initialized)
# ---------------------------------------------------------------------------
_DEVICE = None

def _get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = ttnn.open_device(device_id=0)
    return _DEVICE

def _close_device():
    global _DEVICE
    if _DEVICE is not None:
        ttnn.close_device(_DEVICE)
        _DEVICE = None


# ---------------------------------------------------------------------------
# Test 1: Custom kernel — _fused_rope_kernel
# ---------------------------------------------------------------------------
def test_custom_rope(n_iters=50):
    """Test if the custom RoPE kernel leaks."""
    from model_ttnn import TTRetentionLayer

    device = _get_device()
    B, H, T, d_half = 8, 4, 128, 48  # d_model=384, d_head=96, d_half=48

    # Create persistent inputs
    x1 = ttnn.zeros([B, H, T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x2 = ttnn.zeros([B, H, T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos = ttnn.zeros([T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin = ttnn.zeros([T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        out0, out1 = TTRetentionLayer._fused_rope_kernel(x1, x2, cos, sin, B, H, T, d_half, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out0, force=True)
        ttnn.deallocate(out1, force=True)

    def body(i):
        out0, out1 = TTRetentionLayer._fused_rope_kernel(x1, x2, cos, sin, B, H, T, d_half, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out0, force=True)
        ttnn.deallocate(out1, force=True)

    return measure_loop("Custom kernel: _fused_rope_kernel", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 2: Custom kernel — _fused_scale_decay
# ---------------------------------------------------------------------------
def test_custom_scale_decay(n_iters=50):
    """Test if the custom scale+decay kernel leaks."""
    from model_ttnn import TTRetentionLayer

    device = _get_device()
    B, H, T = 8, 4, 128

    scores_raw = ttnn.zeros([B, H, T, T], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    D_decay = ttnn.zeros([1, H, T, T], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    scale = 0.1

    def warmup():
        out = TTRetentionLayer._fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out, force=True)

    def body(i):
        out = TTRetentionLayer._fused_scale_decay(scores_raw, D_decay, scale, B, H, T, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out, force=True)

    return measure_loop("Custom kernel: _fused_scale_decay", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 3: Custom kernel — _fused_gate_backward
# ---------------------------------------------------------------------------
def test_custom_gate_backward(n_iters=50):
    """Test if the custom gate backward kernel leaks."""
    from model_ttnn import TTRetentionLayer

    device = _get_device()
    B, T, D = 8, 128, 384

    grad_out_gated = ttnn.zeros([B, T, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.full([B, T, D], 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_flat = ttnn.zeros([B, T, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        g0, g1 = TTRetentionLayer._fused_gate_backward(grad_out_gated, gate, out_flat, B, T, D, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(g0, force=True)
        ttnn.deallocate(g1, force=True)

    def body(i):
        g0, g1 = TTRetentionLayer._fused_gate_backward(grad_out_gated, gate, out_flat, B, T, D, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(g0, force=True)
        ttnn.deallocate(g1, force=True)

    return measure_loop("Custom kernel: _fused_gate_backward", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 4: ttnn.generic_op / ProgramDescriptor object leak
# ---------------------------------------------------------------------------
def test_program_descriptor_leak(n_iters=50):
    """Test if creating ProgramDescriptor objects leaks Python/C++ objects.

    This tests whether the kernel descriptor / CB descriptor / core range
    objects created per kernel call are properly cleaned up.
    """
    from model_ttnn import TTRetentionLayer

    device = _get_device()
    B, H, T, d_half = 8, 4, 128, 48

    x1 = ttnn.zeros([B, H, T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x2 = ttnn.zeros([B, H, T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos = ttnn.zeros([T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin = ttnn.zeros([T, d_half], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Also count Python object references
    import weakref
    initial_objs = len(gc.get_objects())

    def warmup():
        out0, out1 = TTRetentionLayer._fused_rope_kernel(x1, x2, cos, sin, B, H, T, d_half, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out0, force=True)
        ttnn.deallocate(out1, force=True)
        gc.collect()

    def body(i):
        out0, out1 = TTRetentionLayer._fused_rope_kernel(x1, x2, cos, sin, B, H, T, d_half, device)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out0, force=True)
        ttnn.deallocate(out1, force=True)
        gc.collect()
        if i in {0, 9, 24, 49}:
            n_objs = len(gc.get_objects())
            print(f"    [gc objects: {n_objs} (delta={n_objs - initial_objs})]", flush=True)

    return measure_loop("ProgramDescriptor object leak (with gc.get_objects count)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 5: Retention layer forward only
# ---------------------------------------------------------------------------
def test_retention_forward(n_iters=50):
    """Test if TTRetentionLayer.forward leaks (without backward)."""
    from model_ttnn import TTWRAPModel
    from train_ttnn import build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_a_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)
    model = TTWRAPModel(model_config, device)

    B, T, V = 8, 128, 128

    def warmup():
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        ttnn.synchronize_device(device)
        from model_ttnn import _safe_deallocate
        _safe_deallocate(logits)
        model.clear_caches()

    def body(i):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        ttnn.synchronize_device(device)
        from model_ttnn import _safe_deallocate
        _safe_deallocate(logits)
        model.clear_caches()

    return measure_loop("Retention model: forward only (full model)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 6: Retention layer forward + backward (no optimizer, no accum)
# ---------------------------------------------------------------------------
def test_retention_backward(n_iters=50):
    """Test if TTRetentionLayer forward + backward leaks."""
    from model_ttnn import TTWRAPModel, _safe_deallocate
    from train_ttnn import cross_entropy_loss, build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_a_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)
    model = TTWRAPModel(model_config, device)

    B, T, V = 8, 128, 128

    def warmup():
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()

    def body(i):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()

    return measure_loop("Retention model: forward + backward (Cell A)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 7: Workspace model forward + backward (Cell B)
# ---------------------------------------------------------------------------
def test_workspace_backward(n_iters=50):
    """Test if TTWorkspaceModule forward + backward leaks (Cell B)."""
    from model_ttnn import TTWRAPModel, _safe_deallocate
    from train_ttnn import cross_entropy_loss, build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_b_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)
    model = TTWRAPModel(model_config, device)

    B, T, V = 8, 128, 128

    def warmup():
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()

    def body(i):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()

    return measure_loop("Workspace model: forward + backward (Cell B)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 8: Recurrent core model forward + backward (Cell C)
# ---------------------------------------------------------------------------
def test_recurrent_backward(n_iters=30):
    """Test if the recurrent core + AR backward leaks (Cell C)."""
    from model_ttnn import TTWRAPModel, _safe_deallocate
    from train_ttnn import cross_entropy_loss, build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_c_attn_residual.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)
    model = TTWRAPModel(model_config, device)

    B, T, V = 8, 128, 128

    def warmup():
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=3)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()

    def body(i):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=3)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()

    return measure_loop("Recurrent core model: forward + backward (Cell C, K=3)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 9: Single retention layer in isolation
# ---------------------------------------------------------------------------
def test_single_retention_layer(n_iters=50):
    """Test a single TTRetentionLayer forward + backward in isolation."""
    from model_ttnn import TTRetentionLayer, _safe_deallocate, TTRMSNorm
    from train_ttnn import build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_a_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)

    # Create a single retention layer (not wrapped in GatedResidual)
    layer = TTRetentionLayer(model_config, device, use_fused_rope=True)

    B, T, D = 8, 128, 384

    def make_inputs():
        x = ttnn.from_torch(torch.zeros(B, T, D, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        grad_out = ttnn.from_torch(torch.zeros(B, T, D, dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return x, grad_out

    def warmup():
        x, grad_out = make_inputs()
        out = layer.forward(x)
        ttnn.synchronize_device(device)
        grad_x, grads = layer.backward(grad_out)
        ttnn.synchronize_device(device)
        _safe_deallocate(out)
        _safe_deallocate(grad_x)
        for g in grads.values():
            _safe_deallocate(g)
        layer._deallocate_cache()
        _safe_deallocate(x)
        _safe_deallocate(grad_out)

    def body(i):
        x, grad_out = make_inputs()
        out = layer.forward(x)
        ttnn.synchronize_device(device)
        grad_x, grads = layer.backward(grad_out)
        ttnn.synchronize_device(device)
        _safe_deallocate(out)
        _safe_deallocate(grad_x)
        for g in grads.values():
            _safe_deallocate(g)
        layer._deallocate_cache()
        _safe_deallocate(x)
        _safe_deallocate(grad_out)

    return measure_loop("Single TTRetentionLayer: forward + backward", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 10: Single attention layer in isolation
# ---------------------------------------------------------------------------
def test_single_attention_layer(n_iters=50):
    """Test a single TTAttentionLayer forward + backward in isolation."""
    from model_ttnn import TTAttentionLayer, _safe_deallocate
    from train_ttnn import build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_a_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)

    layer = TTAttentionLayer(model_config, device)

    B, T, D = 8, 128, 384

    def make_inputs():
        x = ttnn.from_torch(torch.zeros(B, T, D, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        grad_out = ttnn.from_torch(torch.zeros(B, T, D, dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return x, grad_out

    def warmup():
        x, grad_out = make_inputs()
        out = layer.forward(x)
        ttnn.synchronize_device(device)
        grad_x, grads = layer.backward(grad_out)
        ttnn.synchronize_device(device)
        _safe_deallocate(out)
        _safe_deallocate(grad_x)
        for g in grads.values():
            _safe_deallocate(g)
        layer._deallocate_cache()
        _safe_deallocate(x)
        _safe_deallocate(grad_out)

    def body(i):
        x, grad_out = make_inputs()
        out = layer.forward(x)
        ttnn.synchronize_device(device)
        grad_x, grads = layer.backward(grad_out)
        ttnn.synchronize_device(device)
        _safe_deallocate(out)
        _safe_deallocate(grad_x)
        for g in grads.values():
            _safe_deallocate(g)
        layer._deallocate_cache()
        _safe_deallocate(x)
        _safe_deallocate(grad_out)

    return measure_loop("Single TTAttentionLayer: forward + backward", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 11: Single workspace module in isolation
# ---------------------------------------------------------------------------
def test_single_workspace(n_iters=50):
    """Test a single TTWorkspaceModule forward + backward in isolation."""
    from model_ttnn import TTWorkspaceModule, _safe_deallocate
    from train_ttnn import build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_b_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)

    ws = TTWorkspaceModule(model_config, device)

    B, T, D = 8, 128, 384
    x = ttnn.zeros([B, T, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    slots = ttnn.zeros([B, 16, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_x = ttnn.zeros([B, T, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_slots = ttnn.zeros([B, 16, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        x_out, slots_out = ws.forward(x, slots)
        ttnn.synchronize_device(device)
        gx, gs, gw = ws.backward(grad_x, grad_slots)
        ttnn.synchronize_device(device)
        _safe_deallocate(x_out)
        _safe_deallocate(slots_out)
        _safe_deallocate(gx)
        _safe_deallocate(gs)
        for g in gw.values():
            _safe_deallocate(g)
        ws._deallocate_cache()
        ws._forward_caches = []

    def body(i):
        x_out, slots_out = ws.forward(x, slots)
        ttnn.synchronize_device(device)
        gx, gs, gw = ws.backward(grad_x, grad_slots)
        ttnn.synchronize_device(device)
        _safe_deallocate(x_out)
        _safe_deallocate(slots_out)
        _safe_deallocate(gx)
        _safe_deallocate(gs)
        for g in gw.values():
            _safe_deallocate(g)
        ws._deallocate_cache()
        ws._forward_caches = []

    return measure_loop("Single TTWorkspaceModule: forward + backward", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 12: AttentionResidual in isolation
# ---------------------------------------------------------------------------
def test_attention_residual(n_iters=50):
    """Test AttentionResidual forward + backward in isolation."""
    from model_ttnn import AttentionResidual, _safe_deallocate
    from train_ttnn import build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_c_attn_residual.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)

    ar = AttentionResidual(model_config.d_model, device, k_max=3)

    B, T, D = 8, 128, 384
    K = 3
    x_outputs = [ttnn.zeros([B, T, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                 for _ in range(K + 1)]
    grad_out = ttnn.zeros([B, T, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        out = ar.forward(x_outputs, K_active=K)
        ttnn.synchronize_device(device)
        grad_x_list, grads = ar.backward(grad_out)
        ttnn.synchronize_device(device)
        _safe_deallocate(out)
        for gx in grad_x_list:
            _safe_deallocate(gx)
        for g in grads.values():
            _safe_deallocate(g)
        ar._deallocate_cache()

    def body(i):
        out = ar.forward(x_outputs, K_active=K)
        ttnn.synchronize_device(device)
        grad_x_list, grads = ar.backward(grad_out)
        ttnn.synchronize_device(device)
        _safe_deallocate(out)
        for gx in grad_x_list:
            _safe_deallocate(gx)
        for g in grads.values():
            _safe_deallocate(g)
        ar._deallocate_cache()

    return measure_loop("AttentionResidual: forward + backward (K=3)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 13: ttnn.reshape leak test (views vs copies)
# ---------------------------------------------------------------------------
def test_reshape_leak(n_iters=100):
    """Test if ttnn.reshape leaks when the result is discarded.

    reshape may return a view (no new allocation) or a copy (new allocation).
    If it returns copies and we don't deallocate them, they leak.
    """
    device = _get_device()
    B, T, D = 8, 128, 384

    # Create a persistent source tensor
    src = ttnn.zeros([B, T, D], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        r = ttnn.reshape(src, [B * T, D])
        ttnn.synchronize_device(device)
        # Don't deallocate — if it's a view, this is fine; if it's a copy, it leaks
        del r

    def body(i):
        r = ttnn.reshape(src, [B * T, D])
        ttnn.synchronize_device(device)
        del r

    return measure_loop("ttnn.reshape (no dealloc, testing view vs copy)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 14: ttnn.permute leak test
# ---------------------------------------------------------------------------
def test_permute_leak(n_iters=100):
    """Test if ttnn.permute leaks when the result is discarded."""
    device = _get_device()
    B, H, T, dh = 8, 4, 128, 96

    src = ttnn.zeros([B, T, H, dh], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        r = ttnn.permute(src, [0, 2, 1, 3])
        ttnn.synchronize_device(device)
        del r

    def body(i):
        r = ttnn.permute(src, [0, 2, 1, 3])
        ttnn.synchronize_device(device)
        del r

    return measure_loop("ttnn.permute (no dealloc, testing view vs copy)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 15: ttnn.transpose leak test
# ---------------------------------------------------------------------------
def test_transpose_leak(n_iters=100):
    """Test if ttnn.transpose leaks when the result is discarded."""
    device = _get_device()
    B, T, D = 8, 128, 384

    src = ttnn.zeros([D, T], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        r = ttnn.transpose(src, 0, 1)
        ttnn.synchronize_device(device)
        del r

    def body(i):
        r = ttnn.transpose(src, 0, 1)
        ttnn.synchronize_device(device)
        del r

    return measure_loop("ttnn.transpose (no dealloc, testing view vs copy)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 16: ttnn.slice leak test
# ---------------------------------------------------------------------------
def test_slice_leak(n_iters=100):
    """Test if ttnn.slice leaks when the result is deallocated."""
    device = _get_device()
    B, T, D3 = 8, 128, 1152  # QKV concatenated

    src = ttnn.zeros([B, T, D3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def warmup():
        s = ttnn.slice(src, [0, 0, 0], [B, T, 384])
        ttnn.synchronize_device(device)
        ttnn.deallocate(s, force=True)

    def body(i):
        s = ttnn.slice(src, [0, 0, 0], [B, T, 384])
        ttnn.synchronize_device(device)
        ttnn.deallocate(s, force=True)

    return measure_loop("ttnn.slice (with dealloc)", n_iters, body, warmup)


# ---------------------------------------------------------------------------
# Test 17: Cache cleanup completeness
# ---------------------------------------------------------------------------
def test_cache_cleanup(n_iters=30):
    """Test that clear_caches() actually frees all cached tensors.

    Compares RSS after forward (caches populated) vs after clear_caches
    (caches should be freed). If clear_caches doesn't free everything,
    RSS will not return to pre-forward levels.
    """
    from model_ttnn import TTWRAPModel, _safe_deallocate
    from train_ttnn import cross_entropy_loss, build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_a_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)
    model = TTWRAPModel(model_config, device)

    B, T, V = 8, 128, 128

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        model.clear_caches()
        ttnn.synchronize_device(device)
    gc_trim()
    rss_base = rss_mb()
    print(f"\n{'='*60}")
    print(f"TEST: Cache cleanup completeness")
    print(f"  Baseline RSS: {rss_base:.0f} MB")

    for i in range(n_iters):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)

        # Measure before forward
        gc_trim()
        rss_pre = rss_mb()

        # Forward — populates caches
        logits = model.forward(input_ids, k_value=None)
        ttnn.synchronize_device(device)
        gc_trim()
        rss_after_fwd = rss_mb()

        # Dealloc output (not in cache)
        _safe_deallocate(logits)
        ttnn.synchronize_device(device)

        # Clear caches
        model.clear_caches()
        ttnn.synchronize_device(device)
        gc_trim()
        rss_after_clear = rss_mb()

        if i in {0, 4, 9, 14, 19, 24, 29}:
            fwd_delta = rss_after_fwd - rss_pre
            clear_delta = rss_after_clear - rss_pre
            print(f"  iter {i:2d}: pre={rss_pre:.0f} after_fwd={rss_after_fwd:.0f} ({fwd_delta:+.0f}) "
                  f"after_clear={rss_after_clear:.0f} ({clear_delta:+.0f})")

    rss_final = rss_mb()
    total = rss_final - rss_base
    print(f"  TOTAL growth: {total:+.0f} MB over {n_iters} iters")
    print(f"  Verdict: {'NO LEAK' if total < 10 else 'LEAK DETECTED'}")

    return total, total / n_iters


# ---------------------------------------------------------------------------
# Test 18: Per-layer backward leak (isolate which layer leaks)
# ---------------------------------------------------------------------------
def test_per_layer_backward(n_iters=30):
    """Run the full model forward, then backward one layer at a time,
    measuring RSS after each layer's backward to find which layer leaks.
    """
    from model_ttnn import TTWRAPModel, _safe_deallocate
    from train_ttnn import cross_entropy_loss, build_model_config
    import yaml

    device = _get_device()
    with open(os.path.join(os.path.dirname(__file__), "configs/cell_a_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["micro_batch_size"] = 0
    model_config = build_model_config(cfg)
    model = TTWRAPModel(model_config, device)

    B, T, V = 8, 128, 128

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)
        logits = model.forward(input_ids, k_value=None)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()
    gc_trim()
    rss0 = rss_mb()

    print(f"\n{'='*60}")
    print(f"TEST: Per-layer backward leak isolation (Cell A)")
    print(f"  Baseline RSS: {rss0:.0f} MB")
    print(f"  {'Step':>6}  {'RSS':>8}  {'Delta':>8}  {'Layer':>30}")

    layer_names = [f"layer_{i}" for i in range(len(model.layers))]
    for step in range(n_iters):
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
        labels = torch.randint(0, V, (B, T), dtype=torch.int32)

        # Forward
        logits = model.forward(input_ids, k_value=None)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        ttnn.synchronize_device(device)
        _safe_deallocate(logits)

        # Backward
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        model.clear_caches()
        ttnn.synchronize_device(device)
        gc_trim()

        rss = rss_mb()
        if step in {0, 4, 9, 14, 19, 24, 29}:
            print(f"  {step:6d}  {rss:8.0f}  {rss-rss0:+8.0f}  {'full backward':>30}")

    rss_final = rss_mb()
    total = rss_final - rss0
    print(f"  TOTAL: {total:+.0f} MB over {n_iters} iters ({total/n_iters:.1f} MB/iter)")
    return total, total / n_iters


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
TESTS = {
    # Custom kernel tests
    "custom_rope": ("Custom RoPE kernel", test_custom_rope),
    "custom_scale_decay": ("Custom scale+decay kernel", test_custom_scale_decay),
    "custom_gate_backward": ("Custom gate backward kernel", test_custom_gate_backward),
    "program_descriptor": ("ProgramDescriptor object leak", test_program_descriptor_leak),

    # Per-operation tests
    "reshape": ("ttnn.reshape view vs copy", test_reshape_leak),
    "permute": ("ttnn.permute view vs copy", test_permute_leak),
    "transpose": ("ttnn.transpose view vs copy", test_transpose_leak),
    "slice": ("ttnn.slice with dealloc", test_slice_leak),

    # Per-layer tests
    "single_retention": ("Single TTRetentionLayer", test_single_retention_layer),
    "single_attention": ("Single TTAttentionLayer", test_single_attention_layer),
    "single_workspace": ("Single TTWorkspaceModule", test_single_workspace),
    "attention_residual": ("AttentionResidual", test_attention_residual),

    # Full model tests
    "retention_forward": ("Retention model forward only", test_retention_forward),
    "retention_backward": ("Retention model fwd+bwd (Cell A)", test_retention_backward),
    "workspace_backward": ("Workspace model fwd+bwd (Cell B)", test_workspace_backward),
    "recurrent_backward": ("Recurrent core model fwd+bwd (Cell C)", test_recurrent_backward),

    # Cache and per-layer isolation
    "cache_cleanup": ("Cache cleanup completeness", test_cache_cleanup),
    "per_layer": ("Per-layer backward leak isolation", test_per_layer_backward),
}


def main():
    parser = argparse.ArgumentParser(description="Memory leak isolation tests")
    parser.add_argument("--test", "-t", default=None,
                        help=f"Test name (one of: {', '.join(TESTS.keys())})")
    parser.add_argument("--iterations", "-n", type=int, default=None,
                        help="Override default iteration count")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available tests and exit")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for key, (desc, _) in TESTS.items():
            print(f"  {key:30s}  {desc}")
        return

    # Open device
    device = _get_device()
    print(f"Device: {device}")

    if args.test:
        if args.test not in TESTS:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(TESTS.keys())}")
            sys.exit(1)
        desc, fn = TESTS[args.test]
        print(f"\nRunning: {desc}")
        n = args.iterations or 50
        try:
            fn(n)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run all tests
        results = {}
        for key, (desc, fn) in TESTS.items():
            print(f"\n{'#'*60}")
            print(f"# {desc}")
            print(f"{'#'*60}")
            n = args.iterations or 50
            try:
                total, rate = fn(n)
                results[key] = (total, rate)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[key] = (None, None)

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Test':<30s}  {'Total':>8}  {'Rate':>10}  {'Verdict'}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*20}")
        for key, (desc, _) in TESTS.items():
            total, rate = results.get(key, (None, None))
            if total is None:
                print(f"  {key:<30s}  {'ERROR':>8}  {'':>10}  FAILED")
            elif rate < 0.1:
                print(f"  {key:<30s}  {total:>+8.0f}  {rate:>8.2f}  {'NO LEAK':>20}")
            elif rate < 0.5:
                print(f"  {key:<30s}  {total:>+8.0f}  {rate:>8.2f}  {'MINOR':>20}")
            elif rate < 2.0:
                print(f"  {key:<30s}  {total:>+8.0f}  {rate:>8.2f}  {'MODERATE':>20}")
            else:
                print(f"  {key:<30s}  {total:>+8.0f}  {rate:>8.2f}  {'SEVERE':>20}")

    _close_device()


if __name__ == "__main__":
    main()
