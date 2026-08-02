#!/usr/bin/env python3
"""Diagnose the Cell B gradient explosion by comparing checkpoints at steps 400, 500, 700.

Analyzes:
1. Parameter value changes (step 400 -> 700): which changed the most?
2. Optimizer exp_avg_sq (second moment) at step 700: which had largest gradients?
3. Spectral norms of workspace weight matrices at step 400 vs 700.
4. Slot parameter RMS at step 400 vs 700.
5. NaN/Inf check.
"""

import torch
import os
import sys

CKPT_DIR = "/home/rfenwick/Documents/jasper/mamba-poc/checkpoints"
STEPS = [400, 500, 700]
C_BOUND = 5.0  # spectral norm bound from config

# Workspace weight matrices (8 total)
WS_WEIGHT_KEYS = [
    "ws_read_q_weight", "ws_read_k_weight", "ws_read_v_weight", "ws_read_out_weight",
    "ws_write_q_weight", "ws_write_k_weight", "ws_write_v_weight", "ws_write_out_weight",
]

# All workspace-related keys
WS_KEYS = WS_WEIGHT_KEYS + [
    "ws_read_gate", "ws_write_gate", "ws_slot_decay",
    "ws_ws_norm_weight", "ws_ws_slot_norm_weight", "ws_slots",
]

# Friendly names for the report
FRIENDLY = {
    "ws_read_q_weight": "ws_WQ (read query)",
    "ws_read_k_weight": "ws_WK (read key)",
    "ws_read_v_weight": "ws_WV (read value)",
    "ws_read_out_weight": "ws_WO (read output)",
    "ws_write_q_weight": "ws_WQ2 (write query)",
    "ws_write_k_weight": "ws_WK2 (write key)",
    "ws_write_v_weight": "ws_WV2 (write value)",
    "ws_write_out_weight": "ws_WO2 (write output)",
    "ws_read_gate": "ws_read_gate",
    "ws_write_gate": "ws_write_gate",
    "ws_slot_decay": "ws_slot_decay",
    "ws_ws_norm_weight": "ws_ws_norm_weight",
    "ws_ws_slot_norm_weight": "ws_ws_slot_norm_weight",
    "ws_slots": "slot_params (learned slots)",
    "token_emb_weight": "token_emb_weight",
    "norm_weight": "norm_weight (final)",
}


def load_ckpt(step):
    path = os.path.join(CKPT_DIR, f"cell_B_step{step}.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


def friendly(key):
    return FRIENDLY.get(key, key)


def spectral_norm(W):
    """Compute the largest singular value (spectral norm) of matrix W using SVD."""
    if W.dim() == 1:
        return float(W.abs().max())
    # Use fp32 for accuracy
    W32 = W.float()
    try:
        s = torch.linalg.svdvals(W32)
        return float(s[0])
    except Exception:
        # Fallback: power iteration
        D = W32.shape[0]
        v = torch.randn(D, 1) / (D ** 0.5)
        for _ in range(100):
            u = W32 @ v
            u = u / (u.norm() + 1e-12)
            v = W32.T @ u
            v = v / (v.norm() + 1e-12)
        return float((W32 @ v).norm())


def rms(t):
    """RMS of a tensor."""
    return float((t.float() ** 2).mean().sqrt())


def main():
    print("=" * 100)
    print("CELL B GRADIENT EXPLOSION DIAGNOSIS")
    print("Steps: 400 (grad norm ~7.7) -> 500 (~945) -> 700 (~1.37M)")
    print("=" * 100)
    print()

    ckpts = {s: load_ckpt(s) for s in STEPS}

    # ----------------------------------------------------------------
    # 5. NaN / Inf check (do this first — it may explain everything)
    # ----------------------------------------------------------------
    print("=" * 100)
    print("5. NaN / INF CHECK")
    print("=" * 100)
    nan_inf_found = False
    for step in STEPS:
        ms = ckpts[step]["model_state"]
        opt = ckpts[step]["optimizer_state"]
        for k, v in ms.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"  step {step} model_state[{k}]: HAS NaN/Inf!")
                nan_inf_found = True
        for buf_name in ["master", "exp_avg", "exp_avg_sq"]:
            buf = opt.get(buf_name, {})
            for k, v in buf.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"  step {step} optimizer {buf_name}[{k}]: HAS NaN/Inf!")
                    nan_inf_found = True
    if not nan_inf_found:
        print("  No NaN or Inf found in any checkpoint (model_state, master, exp_avg, exp_avg_sq).")
    print()

    # ----------------------------------------------------------------
    # 1. Parameter value changes: step 400 -> 700
    # ----------------------------------------------------------------
    print("=" * 100)
    print("1. PARAMETER VALUE CHANGES (step 400 -> step 700)")
    print("=" * 100)
    print()

    ms400 = ckpts[400]["model_state"]
    ms700 = ckpts[700]["model_state"]

    changes = []
    for k in ms400.keys():
        if k not in ms700:
            continue
        v400 = ms400[k].float()
        v700 = ms700[k].float()
        diff = v700 - v400
        abs_diff_max = float(diff.abs().max())
        abs_diff_mean = float(diff.abs().mean())
        rel_change = abs_diff_max / (float(v400.abs().max()) + 1e-12)
        l2_change = float(diff.norm())
        l2_orig = float(v400.norm())
        rel_l2 = l2_change / (l2_orig + 1e-12)
        changes.append({
            "key": k,
            "friendly": friendly(k),
            "shape": tuple(v400.shape),
            "abs_diff_max": abs_diff_max,
            "abs_diff_mean": abs_diff_mean,
            "rel_change_max": rel_change,
            "l2_change": l2_change,
            "rel_l2": rel_l2,
            "val400_max": float(v400.abs().max()),
            "val700_max": float(v700.abs().max()),
        })

    # Sort by absolute max change
    changes_sorted = sorted(changes, key=lambda x: x["abs_diff_max"], reverse=True)

    print("Top 20 parameters by ABSOLUTE MAX CHANGE (|Δ|_max):")
    print(f"{'Rank':>4}  {'Parameter':<45} {'Shape':<18} {'|Δ|_max':>12} {'|Δ|_mean':>12} {'rel_max':>10} {'rel_L2':>10} {'v400_max':>10} {'v700_max':>10}")
    print("-" * 145)
    for i, c in enumerate(changes_sorted[:20]):
        print(f"{i+1:>4}  {c['friendly']:<45} {str(c['shape']):<18} {c['abs_diff_max']:>12.6f} {c['abs_diff_mean']:>12.6f} {c['rel_change_max']:>10.4f} {c['rel_l2']:>10.4f} {c['val400_max']:>10.4f} {c['val700_max']:>10.4f}")
    print()

    print("Top 20 parameters by RELATIVE L2 CHANGE (||Δ|| / ||orig||):")
    print(f"{'Rank':>4}  {'Parameter':<45} {'Shape':<18} {'rel_L2':>10} {'|Δ|_max':>12} {'|Δ|_mean':>12} {'v400_max':>10} {'v700_max':>10}")
    print("-" * 125)
    changes_rel = sorted(changes, key=lambda x: x["rel_l2"], reverse=True)
    for i, c in enumerate(changes_rel[:20]):
        print(f"{i+1:>4}  {c['friendly']:<45} {str(c['shape']):<18} {c['rel_l2']:>10.4f} {c['abs_diff_max']:>12.6f} {c['abs_diff_mean']:>12.6f} {c['val400_max']:>10.4f} {c['val700_max']:>10.4f}")
    print()

    # ----------------------------------------------------------------
    # 2. Optimizer exp_avg_sq (second moment) at step 700
    # ----------------------------------------------------------------
    print("=" * 100)
    print("2. OPTIMIZER exp_avg_sq (SECOND MOMENT) AT STEP 700")
    print("   (AdamW v_t ≈ EMA of g². Large v_t => large recent gradients.)")
    print("=" * 100)
    print()

    for step in STEPS:
        opt = ckpts[step]["optimizer_state"]
        eas = opt.get("exp_avg_sq", {})
        eas_stats = []
        for k, v in eas.items():
            vf = v.float()
            eas_stats.append({
                "key": k,
                "friendly": friendly(k),
                "shape": tuple(vf.shape),
                "max": float(vf.max()),
                "mean": float(vf.mean()),
                "rms": rms(vf),
            })
        eas_sorted = sorted(eas_stats, key=lambda x: x["max"], reverse=True)

        print(f"--- Step {step}: Top 20 by exp_avg_sq MAX ---")
        print(f"{'Rank':>4}  {'Parameter':<45} {'Shape':<18} {'max':>14} {'mean':>14} {'rms':>14}")
        print("-" * 115)
        for i, c in enumerate(eas_sorted[:20]):
            print(f"{i+1:>4}  {c['friendly']:<45} {str(c['shape']):<18} {c['max']:>14.6f} {c['mean']:>14.6f} {c['rms']:>14.6f}")
        print()

    # Also show exp_avg (first moment) at step 700 for context
    print("--- Step 700: exp_avg (FIRST MOMENT) Top 20 by MAX ---")
    ea = ckpts[700]["optimizer_state"].get("exp_avg", {})
    ea_stats = []
    for k, v in ea.items():
        vf = v.float()
        ea_stats.append({"key": k, "friendly": friendly(k), "shape": tuple(vf.shape),
                         "max": float(vf.abs().max()), "mean": float(vf.abs().mean())})
    ea_sorted = sorted(ea_stats, key=lambda x: x["max"], reverse=True)
    print(f"{'Rank':>4}  {'Parameter':<45} {'Shape':<18} {'|max|':>14} {'|mean|':>14}")
    print("-" * 100)
    for i, c in enumerate(ea_sorted[:20]):
        print(f"{i+1:>4}  {c['friendly']:<45} {str(c['shape']):<18} {c['max']:>14.6f} {c['mean']:>14.6f}")
    print()

    # ----------------------------------------------------------------
    # 3. Spectral norms of workspace weight matrices
    # ----------------------------------------------------------------
    print("=" * 100)
    print(f"3. SPECTRAL NORMS OF WORKSPACE WEIGHT MATRICES (C cap = {C_BOUND})")
    print("=" * 100)
    print()

    print(f"{'Weight':<45} {'step400 σ_max':>14} {'step500 σ_max':>14} {'step700 σ_max':>14} {'400→700 Δ':>12} {'Exceeds C?':>12}")
    print("-" * 110)
    for k in WS_WEIGHT_KEYS:
        sigmas = {}
        for step in STEPS:
            W = ckpts[step]["model_state"][k].float()
            sigmas[step] = spectral_norm(W)
        delta = sigmas[700] - sigmas[400]
        exceeds = "YES" if sigmas[700] > C_BOUND else "no"
        flag = " ***" if sigmas[700] > C_BOUND else ""
        print(f"{friendly(k):<45} {sigmas[400]:>14.4f} {sigmas[500]:>14.4f} {sigmas[700]:>14.4f} {delta:>+12.4f} {exceeds:>12}{flag}")
    print()

    # Also check the non-workspace weight matrices (qkv, out_proj) for comparison
    print("For comparison — retention/attention layer weight matrices (layer 0, 6, 12):")
    print(f"{'Weight':<45} {'step400 σ_max':>14} {'step700 σ_max':>14} {'400→700 Δ':>12}")
    print("-" * 90)
    for layer in [0, 6, 12]:
        for suffix in ["qkv_weight", "out_proj_weight"]:
            k = f"layer_{layer}_{suffix}"
            if k not in ms400:
                continue
            s400 = spectral_norm(ms400[k].float())
            s700 = spectral_norm(ms700[k].float())
            print(f"{k:<45} {s400:>14.4f} {s700:>14.4f} {s700-s400:>+12.4f}")
    print()

    # ----------------------------------------------------------------
    # 4. Slot parameter RMS
    # ----------------------------------------------------------------
    print("=" * 100)
    print("4. SLOT PARAMETER RMS (should be ~1.0 if normalize_slots() is working)")
    print("=" * 100)
    print()

    for step in STEPS:
        slots = ckpts[step]["model_state"]["ws_slots"].float()
        slot_rms = rms(slots)
        per_slot_rms = [rms(slots[i]) for i in range(slots.shape[0])]
        slot_norms = [float(slots[i].norm()) for i in range(slots.shape[0])]
        max_slot = float(slots.abs().max())
        print(f"  Step {step}:")
        print(f"    Overall RMS:        {slot_rms:.6f}")
        print(f"    Max |value|:        {max_slot:.6f}")
        print(f"    Per-slot RMS:       {['%.4f' % x for x in per_slot_rms]}")
        print(f"    Per-slot L2 norm:   {['%.4f' % x for x in slot_norms]}")
        print()

    # ----------------------------------------------------------------
    # Summary: workspace gates and key scalars
    # ----------------------------------------------------------------
    print("=" * 100)
    print("SUMMARY: KEY SCALAR PARAMETERS")
    print("=" * 100)
    print()
    scalar_keys = ["ws_read_gate", "ws_write_gate", "ws_slot_decay"]
    print(f"{'Parameter':<25} {'step400':>14} {'step500':>14} {'step700':>14}")
    print("-" * 70)
    for k in scalar_keys:
        vals = {}
        for step in STEPS:
            v = ckpts[step]["model_state"][k].float()
            vals[step] = float(v.item())
        print(f"{friendly(k):<25} {vals[400]:>14.6f} {vals[500]:>14.6f} {vals[700]:>14.6f}")
    print()

    # Gate sigmoid values
    print("Gate sigmoid values:")
    print(f"{'Parameter':<25} {'step400 σ':>14} {'step500 σ':>14} {'step700 σ':>14}")
    print("-" * 70)
    for k in ["ws_read_gate", "ws_write_gate"]:
        vals = {}
        for step in STEPS:
            v = ckpts[step]["model_state"][k].float()
            vals[step] = float(torch.sigmoid(v).item())
        print(f"{friendly(k):<25} {vals[400]:>14.6f} {vals[500]:>14.6f} {vals[700]:>14.6f}")
    print()

    # Layer gates
    print("Layer gates (sigmoid):")
    print(f"{'Parameter':<25} {'step400 σ':>14} {'step500 σ':>14} {'step700 σ':>14}")
    print("-" * 70)
    for k in sorted(ms400.keys()):
        if "_gate" in k and "ws_" not in k:
            vals = {}
            for step in STEPS:
                v = ckpts[step]["model_state"][k].float()
                vals[step] = float(torch.sigmoid(v).item())
            print(f"{k:<25} {vals[400]:>14.6f} {vals[500]:>14.6f} {vals[700]:>14.6f}")
    print()

    # ----------------------------------------------------------------
    # Cross-check: exp_avg_sq growth 400 -> 700
    # ----------------------------------------------------------------
    print("=" * 100)
    print("BONUS: exp_avg_sq GROWTH (step 400 -> 700), top 15 by growth ratio")
    print("=" * 100)
    print()
    eas400 = ckpts[400]["optimizer_state"].get("exp_avg_sq", {})
    eas700 = ckpts[700]["optimizer_state"].get("exp_avg_sq", {})
    growth = []
    for k in eas400:
        if k not in eas700:
            continue
        m400 = float(eas400[k].float().max())
        m700 = float(eas700[k].float().max())
        ratio = m700 / (m400 + 1e-12)
        growth.append({"key": k, "friendly": friendly(k), "m400": m400, "m700": m700, "ratio": ratio})
    growth_sorted = sorted(growth, key=lambda x: x["ratio"], reverse=True)
    print(f"{'Rank':>4}  {'Parameter':<45} {'step400 max':>14} {'step700 max':>14} {'ratio':>12}")
    print("-" * 95)
    for i, c in enumerate(growth_sorted[:15]):
        print(f"{i+1:>4}  {c['friendly']:<45} {c['m400']:>14.6f} {c['m700']:>14.6f} {c['ratio']:>12.1f}")
    print()

    print("=" * 100)
    print("DIAGNOSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
