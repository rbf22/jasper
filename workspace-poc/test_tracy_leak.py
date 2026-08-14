#!/usr/bin/env python3
"""Memory leak tests for the Tracy profiler fix and general RSS stability.

These tests verify that the dominant memory leak (Tracy profiler's
rpmalloc-based ConcurrentQueue) has been eliminated by the
ENABLE_TRACY=OFF rebuild, and that RSS remains stable during
long-running training loops.

The tests come in two tiers:

1. **Binary symbol check** (no device required): Verifies that the
   installed .so files do not contain rpmalloc/Tracy symbols that
   cause the unbounded mmap growth.

2. **RSS stability test** (device required): Runs N forward+backward
   steps and verifies RSS does not grow linearly. The pass criterion
   is that the steady-state rate (last half of iterations) is below
   a threshold, allowing for one-time page-fault settling.

Run:
    cd /home/rfenwick/Documents/jasper/workspace-poc

    # Binary symbol check only (no device needed):
    .tt-venv/bin/python test_tracy_leak.py --test symbols

    # Full RSS stability test (requires device):
    TT_VISIBLE_DEVICES=0 TT_METAL_HOME=/home/rfenwick/Documents/tt-metal-src \
    .tt-venv/bin/python test_tracy_leak.py --test rss

    # Run all tests:
    TT_VISIBLE_DEVICES=0 TT_METAL_HOME=/home/rfenwick/Documents/tt-metal-src \
    .tt-venv/bin/python test_tracy_leak.py

    # Adjust iterations:
    .tt-venv/bin/python test_tracy_leak.py --iterations 300
"""

import argparse
import gc
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Environment setup (must happen before importing ttnn)
# ---------------------------------------------------------------------------

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
        for name in ["p150_mesh_graph_descriptor.textproto",
                     "p300_mesh_graph_descriptor.textproto"]:
            if spec is not None and spec.submodule_search_locations:
                path = (Path(next(iter(spec.submodule_search_locations)))
                        / "tt_metal" / "fabric" / "mesh_graph_descriptors" / name)
                if path.is_file():
                    return str(path)
            for p in sys.path:
                candidate = (Path(p) / "pjrt_plugin_tt" / "tt-metal" /
                             "tt_metal" / "fabric" / "mesh_graph_descriptors" / name)
                if candidate.is_file():
                    return str(candidate)
            venv_path = Path("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages")
            candidate = (venv_path / "pjrt_plugin_tt" / "tt-metal" /
                         "tt_metal" / "fabric" / "mesh_graph_descriptors" / name)
            if candidate.is_file():
                return str(candidate)
    except Exception:
        pass
    return None

if _is_p300():
    _mgd = _find_mesh_graph_descriptor()
    if _mgd:
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", _mgd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rss_kb():
    """Current process RSS in KB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def gc_trim():
    """GC + malloc_trim to return fragmented memory to the OS."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_HEADER_RE = re.compile(r'^([0-9a-f]+)-([0-9a-f]+) ')

def get_anon_mapping_stats():
    """Count anonymous mappings and their total Size/Rss from /proc/self/smaps.

    Returns (count, total_size_kb, total_rss_kb).
    """
    count = 0
    total_size = 0
    total_rss = 0
    current = None
    with open("/proc/self/smaps") as f:
        for line in f:
            m = _HEADER_RE.match(line)
            if m:
                if current and current.get("path") == "[anon]":
                    count += 1
                    total_size += current.get("Size", 0)
                    total_rss += current.get("Rss", 0)
                parts = line.strip().split(None, 5)
                path = parts[5] if len(parts) > 5 else "[anon]"
                current = {"path": path, "Size": 0, "Rss": 0}
            elif current and ":" in line:
                key, _, val = line.strip().partition(":")
                try:
                    current[key] = int(val.split()[0])
                except (ValueError, IndexError):
                    pass
    if current and current.get("path") == "[anon]":
        count += 1
        total_size += current.get("Size", 0)
        total_rss += current.get("Rss", 0)
    return count, total_size, total_rss


def find_installed_so_files():
    """Find the installed TT-Metal .so files in the pip package."""
    from pathlib import Path
    venv_lib = Path("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages")
    pkg_dir = venv_lib / "pjrt_plugin_tt" / "lib64"
    files = {
        "libtt_metal.so": pkg_dir / "libtt_metal.so",
        "_ttnncpp.so": pkg_dir / "_ttnncpp.so",
        "_ttnn.so": pkg_dir / "_ttnn.so",
    }
    # Also check the ttnn directory for _ttnn.so
    ttnn_so = venv_lib / "pjrt_plugin_tt" / "tt-metal" / "ttnn" / "ttnn" / "_ttnn.so"
    if ttnn_so.is_file():
        files["_ttnn.so (ttnn dir)"] = ttnn_so
    return {k: v for k, v in files.items() if v.is_file()}


def check_no_rpmalloc_symbols(so_path):
    """Check that a .so file does not reference rpmalloc/Tracy queue symbols.

    Returns (passed: bool, found_symbols: list[str]).
    """
    try:
        result = subprocess.run(
            ["nm", "-D", str(so_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            # nm might not work on all files; try strings as fallback
            result = subprocess.run(
                ["strings", str(so_path)],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout
            leak_indicators = ["rpmalloc", "rpfree", "InitRpmalloc"]
            found = [s for s in leak_indicators if s in output]
            return (len(found) == 0, found)

        output = result.stdout
        # Symbols that indicate Tracy's rpmalloc-based leak path
        leak_patterns = [
            "rpmalloc",
            "rpfree",
            "InitRpmalloc",
        ]
        found = []
        for pattern in leak_patterns:
            if pattern in output:
                found.append(pattern)
        return (len(found) == 0, found)
    except Exception as e:
        # If we can't check, fail conservatively
        return (False, [f"nm/strings failed: {e}"])


# ---------------------------------------------------------------------------
# Tier 1: Binary symbol tests (no device required)
# ---------------------------------------------------------------------------

def test_no_rpmalloc_in_installed_libs():
    """Verify that installed .so files do not contain rpmalloc symbols.

    The Tracy profiler leak was caused by rpmalloc allocating anonymous
    mmap regions via moodycamel::ConcurrentQueue. Rebuilding with
    ENABLE_TRACY=OFF removes these symbols entirely.
    """
    so_files = find_installed_so_files()
    assert len(so_files) > 0, "No installed .so files found"

    failures = []
    for name, path in so_files.items():
        passed, found = check_no_rpmalloc_symbols(path)
        if not passed:
            failures.append(f"{name} ({path}): found {found}")
        else:
            print(f"  {name}: no rpmalloc symbols (PASS)")

    assert not failures, \
        f"rpmalloc symbols found in installed libraries (Tracy leak not fixed):\n" + \
        "\n".join(f"  {f}" for f in failures)
    print(f"  All {len(so_files)} .so files are free of rpmalloc symbols")


def test_no_tracy_queue_symbols():
    """Verify that installed .so files do not contain Tracy ConcurrentQueue symbols.

    The moodycamel::ConcurrentQueue is the unbounded queue that grows
    via rpmalloc when no Tracy client is connected.

    We check for the rpmalloc function symbols (the actual leak path)
    and the Tracy queue initialization symbol, not generic Tracy
    class names that may appear in unrelated symbol mangling.
    """
    so_files = find_installed_so_files()
    assert len(so_files) > 0, "No installed .so files found"

    # These are the specific symbols that indicate the leak path:
    # rpmalloc/rpfree are used by Tracy's queue allocator
    # InitRpmalloc initializes the per-thread rpmalloc state
    # GetQueue returns the Tracy profiling queue
    # TracyQueuePrepare/TracyQueueCommit are the enqueue macros
    leak_symbols = [
        "rpmalloc",
        "rpfree",
        "InitRpmalloc",
        "GetQueue",
        "TracyQueuePrepare",
        "TracyQueueCommit",
    ]

    failures = []
    for name, path in so_files.items():
        try:
            result = subprocess.run(
                ["nm", "-D", str(path)],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout
            # Check for exact symbol references (not substrings of other symbols)
            # nm output lines look like: "address T _ZN5tracy12InitRpmallocEv"
            # We check if the pattern appears as a complete symbol name component
            found = []
            for sym in leak_symbols:
                # Match as a C++ mangled symbol component (preceded by a digit
                # or letter boundary in the mangled name, not a substring)
                if re.search(rf'[A-Za-z0-9_]{sym}\b|^{sym}\b', output):
                    found.append(sym)
            if found:
                failures.append(f"{name}: found Tracy leak symbols {found}")
            else:
                print(f"  {name}: no Tracy leak symbols (PASS)")
        except Exception as e:
            failures.append(f"{name}: nm failed: {e}")

    assert not failures, \
        f"Tracy leak symbols found (profiling queue not eliminated):\n" + \
        "\n".join(f"  {f}" for f in failures)
    print(f"  All {len(so_files)} .so files are free of Tracy leak symbols")


# ---------------------------------------------------------------------------
# Tier 2: RSS stability tests (device required)
# ---------------------------------------------------------------------------

def _run_training_steps(model, device, vocab, cfg, n_steps, micro_batch=8):
    """Run n_steps of forward+backward, deallocating properly.

    Returns list of (step, rss_kb) samples.
    """
    import random
    import ttnn
    rng = random.Random(42)
    seq_len = cfg.get("seq_len", 128)
    from train_ttnn import cross_entropy_loss
    from model_ttnn import _safe_deallocate

    samples = []
    for step in range(n_steps):
        input_ids, labels, _ = _sample_batch(micro_batch, seq_len, vocab, rng=rng)
        logits = model.forward(input_ids)
        loss_val, grad_logits = cross_entropy_loss(logits, labels)
        grads = model.backward(grad_logits)
        ttnn.synchronize_device(device)
        _safe_deallocate(grad_logits)
        for g in grads.values():
            _safe_deallocate(g)
        _safe_deallocate(logits)
        model.clear_caches()
        del logits, grad_logits, grads, input_ids, labels
        gc.collect()

        if step % 10 == 0 or step == n_steps - 1:
            samples.append((step, get_rss_kb()))
    return samples


def _sample_batch(batch_size, seq_len, vocab, rng=None):
    """Import and call data.sample_batch."""
    from data import sample_batch
    return sample_batch(batch_size, seq_len, vocab, rng=rng)


def test_rss_stable_over_training(n_steps=100, max_rate_kb_per_step=50.0):
    """Verify RSS does not grow linearly during training.

    The Tracy leak caused ~350 KB/step linear growth. After the fix,
    RSS should plateau within ~20 steps (one-time page-fault settling).

    Pass criterion: the steady-state rate (second half of iterations)
    must be below max_rate_kb_per_step.
    """
    import yaml
    import ttnn

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_ttnn import TTWRAPModel
    from train_ttnn import build_model_config
    from data import Vocab

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "configs", "cell_b_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    model = TTWRAPModel(model_config, device)
    vocab = Vocab()

    # Warmup (one-time allocator/kernel-cache initialization)
    samples_warmup = _run_training_steps(model, device, vocab, cfg, 3)
    gc_trim()
    ttnn.synchronize_device(device)

    rss0 = get_rss_kb()
    print(f"  Baseline RSS after warmup: {rss0 // 1024} MB")

    # Run the actual test
    samples = _run_training_steps(model, device, vocab, cfg, n_steps)

    # Print the RSS trajectory
    print(f"  {'Step':>5}  {'RSS (MB)':>10}  {'Delta (KB)':>12}")
    prev = rss0
    for step, rss in samples:
        delta = rss - prev
        print(f"  {step:>5}  {rss // 1024:>10}  {delta:>12}")
        prev = rss

    # Compute steady-state rate from the second half of iterations
    midpoint = n_steps // 2
    first_half = [(s, r) for s, r in samples if s < midpoint]
    second_half = [(s, r) for s, r in samples if s >= midpoint]

    if len(second_half) >= 2:
        first_rss = second_half[0][1]
        last_rss = second_half[-1][1]
        first_step = second_half[0][0]
        last_step = second_half[-1][0]
        step_span = last_step - first_step
        if step_span > 0:
            steady_rate = (last_rss - first_rss) / step_span
        else:
            steady_rate = 0.0
    else:
        steady_rate = 0.0

    total_growth = samples[-1][1] - rss0
    print(f"\n  Total growth: {total_growth // 1024} MB over {n_steps} steps")
    print(f"  Steady-state rate (2nd half): {steady_rate:.1f} KB/step")
    print(f"  Threshold: {max_rate_kb_per_step:.0f} KB/step")

    ttnn.close_device(device)

    assert steady_rate < max_rate_kb_per_step, \
        f"RSS growing at {steady_rate:.1f} KB/step in steady state " + \
        f"(threshold: {max_rate_kb_per_step:.0f} KB/step). " + \
        f"Tracy leak may not be fixed."
    print(f"  PASS: steady-state rate {steady_rate:.1f} KB/step < {max_rate_kb_per_step:.0f} KB/step")


def test_no_new_anon_mappings(n_steps=50):
    """Verify no new anonymous mappings are created during training.

    The Tracy leak created new mmap(MAP_ANONYMOUS) regions via rpmalloc.
    After the fix, the anonymous mapping count should remain stable.
    """
    import yaml
    import ttnn

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_ttnn import TTWRAPModel
    from train_ttnn import build_model_config
    from data import Vocab

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "configs", "cell_b_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    model = TTWRAPModel(model_config, device)
    vocab = Vocab()

    # Warmup
    _run_training_steps(model, device, vocab, cfg, 3)
    gc_trim()
    ttnn.synchronize_device(device)

    n0, vs0, ar0 = get_anon_mapping_stats()
    print(f"  Baseline: {n0} anon mappings, "
          f"VS={vs0 // 1024} MB, RSS={ar0 // 1024} MB")

    # Run training steps
    _run_training_steps(model, device, vocab, cfg, n_steps)
    ttnn.synchronize_device(device)

    n1, vs1, ar1 = get_anon_mapping_stats()
    print(f"  After {n_steps} steps: {n1} anon mappings, "
          f"VS={vs1 // 1024} MB, RSS={ar1 // 1024} MB")

    new_mappings = n1 - n0
    vs_growth = vs1 - vs0
    rss_growth = ar1 - ar0

    ttnn.close_device(device)

    print(f"  New mappings: {new_mappings}")
    print(f"  VS growth: {vs_growth // 1024} MB")
    print(f"  RSS growth: {rss_growth // 1024} MB")

    # Allow at most 1 new mapping (for minor allocator activity),
    # but not the linear growth seen with the Tracy leak.
    assert new_mappings <= 1, \
        f"{new_mappings} new anonymous mappings created (expected 0-1). " + \
        f"Tracy leak may not be fixed."
    print(f"  PASS: only {new_mappings} new anonymous mappings (<= 1 allowed)")


def test_rss_plateaus(n_steps=200):
    """Verify RSS plateaus (stops growing) after initial settling.

    The Tracy leak showed linear growth that never plateaued. After the
    fix, RSS should settle within ~20 steps and remain flat.

    Pass criterion: the rate in the last quarter of iterations must be
    less than 10 KB/step (effectively zero).
    """
    import yaml
    import ttnn

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_ttnn import TTWRAPModel
    from train_ttnn import build_model_config
    from data import Vocab

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "configs", "cell_b_tt.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_config = build_model_config(cfg)
    device = ttnn.open_device(device_id=0)

    model = TTWRAPModel(model_config, device)
    vocab = Vocab()

    # Warmup
    _run_training_steps(model, device, vocab, cfg, 3)
    gc_trim()
    ttnn.synchronize_device(device)

    rss0 = get_rss_kb()
    print(f"  Baseline RSS: {rss0 // 1024} MB")

    samples = _run_training_steps(model, device, vocab, cfg, n_steps)

    # Check the last quarter
    quarter_point = int(n_steps * 0.75)
    last_quarter = [(s, r) for s, r in samples if s >= quarter_point]

    if len(last_quarter) >= 2:
        first_rss = last_quarter[0][1]
        last_rss = last_quarter[-1][1]
        first_step = last_quarter[0][0]
        last_step = last_quarter[-1][0]
        step_span = last_step - first_step
        if step_span > 0:
            final_rate = (last_rss - first_rss) / step_span
        else:
            final_rate = 0.0
    else:
        final_rate = 0.0

    total_growth = samples[-1][1] - rss0
    print(f"\n  Total growth: {total_growth // 1024} MB over {n_steps} steps")
    print(f"  Final quarter rate: {final_rate:.1f} KB/step")

    ttnn.close_device(device)

    assert final_rate < 10.0, \
        f"RSS still growing at {final_rate:.1f} KB/step in final quarter " + \
        f"(expected < 10 KB/step, i.e. plateaued). " + \
        f"Tracy leak may not be fixed."
    print(f"  PASS: RSS plateaued at {final_rate:.1f} KB/step in final quarter")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(name, fn, *args, **kwargs):
    """Run a single test, printing PASS/FAIL."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")
    try:
        fn(*args, **kwargs)
        print(f"PASS: {name}")
        return True
    except AssertionError as e:
        print(f"FAIL: {name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {name}: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Tracy leak / RSS stability tests")
    parser.add_argument("--test", default="all",
                        choices=["all", "symbols", "rss", "mappings", "plateau"],
                        help="Which test(s) to run")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations for RSS tests")
    args = parser.parse_args()

    print("=" * 60)
    print("Tracy Profiler Leak Tests")
    print("=" * 60)

    results = []

    # Tier 1: Symbol tests (no device needed)
    if args.test in ("all", "symbols"):
        results.append(run_test(
            "No rpmalloc symbols in installed .so files",
            test_no_rpmalloc_in_installed_libs))
        results.append(run_test(
            "No Tracy queue symbols in installed .so files",
            test_no_tracy_queue_symbols))

    # Tier 2: RSS stability tests (device needed)
    if args.test in ("all", "rss"):
        results.append(run_test(
            f"RSS stable over {args.iterations} training steps",
            test_rss_stable_over_training, n_steps=args.iterations))

    if args.test in ("all", "mappings"):
        results.append(run_test(
            "No new anonymous mappings during training",
            test_no_new_anon_mappings, n_steps=50))

    if args.test in ("all", "plateau"):
        results.append(run_test(
            "RSS plateaus after initial settling",
            test_rss_plateaus, n_steps=max(200, args.iterations)))

    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"{total - passed} TEST(S) FAILED ({passed}/{total} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
