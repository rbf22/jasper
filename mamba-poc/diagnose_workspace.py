#!/usr/bin/env python3
"""Diagnostic: inspect workspace slots and attention during Task 1 forward pass.

Generates a Task 1 example with known intermediate values, runs the model
forward, and dumps:
  - Slot states at each workspace call (what's stored in the workspace)
  - Read attention patterns (which positions the slots attend to)
  - Write attention patterns (which slots each position writes to)
  - The model's prediction

This tells us whether the workspace is:
  a) Storing useful intermediate values (slots contain relevant info)
  b) Selectively attending to the right positions (read attention is sharp)
  c) Using the slots to update hidden states (write attention is selective)
  d) Changing behavior across iterations (recurrent core does something different each step)

Usage:
    python diagnose_workspace.py --config configs/cell_c_tt.yaml --device 0 \
        --checkpoint run_C/checkpoints/cell_C_step100.pt
"""

import os
import sys
import argparse
import yaml
import torch
import ttnn
import random
import numpy as np


def _tv(p, q, dim=-1):
    """Total variation distance between two distributions: 0.5 * L1. Range [0, 1]."""
    return 0.5 * (p - q).abs().sum(dim=dim)


def capture_ws_calls(model, ttnn_mod, input_ids):
    """Run a forward pass and capture workspace read/write attention + slots.

    Returns (ws_calls, logits_tt) where ws_calls has one dict per workspace call.
    """
    ws_calls = []
    original_forward = model.workspace.forward

    def instrumented_forward(x, slot_state):
        x_out, slot_state_out = original_forward(x, slot_state)
        cache = model.workspace._cache
        ws_calls.append({
            "slots_in": ttnn_mod.to_torch(cache["slots_in"]).clone(),
            "slots_out": ttnn_mod.to_torch(cache["slots_out"]).clone(),
            "read_attn": ttnn_mod.to_torch(cache["read_attn"]).clone(),
            "write_attn": ttnn_mod.to_torch(cache["write_attn"]).clone(),
            "slot_state_was_none": slot_state is None,
            "B": int(cache["B"]), "T": int(cache["T"]), "m": int(cache["m"]),
        })
        return x_out, slot_state_out

    model.workspace.forward = instrumented_forward
    try:
        with torch.no_grad():
            logits_tt = model.forward(input_ids)
    finally:
        model.workspace.forward = original_forward
    return ws_calls, logits_tt


def analyze_input_dependence(model, vocab, gen_task1, depth, rng, n_probe, n_slots):
    """Measure whether workspace attention actually depends on the INPUT.

    Entropy measures peakedness, not input-dependence. A fixed one-hot routing
    that ignores the input has near-zero entropy and zero information content —
    which is exactly the degenerate solution an entropy penalty selects for.

    This probes the model with n_probe different inputs of identical token length
    (so attention tensors align), then for each (head, position) holds the index
    fixed and measures how far each input's attention distribution sits from the
    across-input mean, in total variation distance.

        TV = 0    -> routing is identical for every input (constant / positional
                     only). The workspace is a fixed bucket, not a content-
                     addressed memory.
        TV > 0    -> routing changes with content. Necessary (not sufficient)
                     for the workspace to be doing real work.

    Also reports top-1 routing agreement: the fraction of (head, position) pairs
    for which every probe input routes to the SAME top slot.
    """
    # Collect n_probe examples that all tokenize to the same length, so the
    # attention tensors are directly comparable position-by-position.
    buckets = {}
    for _ in range(n_probe * 60):
        prompt, _, _ = gen_task1(depth, rng)
        ids = vocab.encode(prompt)
        buckets.setdefault(len(ids), []).append(ids)
        if len(buckets.get(len(ids), [])) >= n_probe:
            break
    length, batch = max(buckets.items(), key=lambda kv: len(kv[1]))
    batch = batch[:n_probe]
    if len(batch) < 2:
        print("Could not collect enough same-length probes; skipping.")
        return None

    print(f"Probing with {len(batch)} distinct inputs of {length} tokens each")
    print()

    # Run each probe separately and stack per workspace call.
    per_input_calls = []
    for ids in batch:
        input_ids = torch.tensor([ids], dtype=torch.long)
        calls, _ = capture_ws_calls(model, ttnn, input_ids)
        per_input_calls.append(calls)

    n_calls = min(len(c) for c in per_input_calls)
    results = []

    for call_idx in range(n_calls):
        # read: (N, H, m, T) | write: (N, H, T, m)
        read = torch.stack([c[call_idx]["read_attn"][0].float() for c in per_input_calls])
        write = torch.stack([c[call_idx]["write_attn"][0].float() for c in per_input_calls])

        # --- Input-dependence: distance from the across-input mean pattern ---
        read_mean = read.mean(dim=0, keepdim=True)     # (1, H, m, T)
        write_mean = write.mean(dim=0, keepdim=True)   # (1, H, T, m)
        read_tv = _tv(read, read_mean).mean().item()   # avg over N,H,m
        write_tv = _tv(write, write_mean).mean().item()  # avg over N,H,T

        # --- Top-1 routing agreement: does every input pick the same slot? ---
        write_top = write.argmax(dim=-1)               # (N, H, T)
        write_agree = (write_top == write_top[0:1]).all(dim=0).float().mean().item()
        read_top = read.argmax(dim=-1)                 # (N, H, m)
        read_agree = (read_top == read_top[0:1]).all(dim=0).float().mean().item()

        # --- Entropy, for direct contrast with the input-dependence numbers ---
        read_ent = -(read * (read + 1e-10).log()).sum(-1).mean().item()
        write_ent = -(write * (write + 1e-10).log()).sum(-1).mean().item()

        results.append({
            "call": call_idx,
            "read_tv": read_tv, "write_tv": write_tv,
            "read_agree": read_agree, "write_agree": write_agree,
            "read_entropy": read_ent, "write_entropy": write_ent,
        })

    print("=" * 80)
    print("INPUT-DEPENDENCE (the metric that matters)")
    print("=" * 80)
    print(f"{'call':>5} {'read TV':>9} {'write TV':>9} {'read fix%':>10} {'write fix%':>11} "
          f"{'read H':>8} {'write H':>8}")
    for r in results:
        print(f"{r['call']:>5} {r['read_tv']:>9.4f} {r['write_tv']:>9.4f} "
              f"{r['read_agree']*100:>9.1f}% {r['write_agree']*100:>10.1f}% "
              f"{r['read_entropy']:>8.3f} {r['write_entropy']:>8.3f}")
    print()
    print("  TV   = mean total variation distance from the across-input mean pattern")
    print("         (0 = attention identical for every input = constant routing)")
    print("  fix% = fraction of (head, position) pairs whose top-1 target is the")
    print("         SAME for every probe input (100% = completely fixed routing)")
    print("  H    = entropy, shown only to contrast with TV: low H + low TV is a")
    print("         degenerate sharp-but-constant routing, NOT selective memory")
    print()

    mean_read_tv = float(np.mean([r["read_tv"] for r in results]))
    mean_write_tv = float(np.mean([r["write_tv"] for r in results]))
    mean_write_agree = float(np.mean([r["write_agree"] for r in results]))

    # Verdict is reported PER PATH. A healthy read path can otherwise mask a
    # fully degenerate write path (or vice versa), and the two failures have
    # different meanings: a fixed write path means information enters the slots
    # independently of content, so the slots are positional buckets rather than
    # a content-addressed memory — regardless of how selective reads look.
    def verdict(name, tv, fixed_frac=None):
        print(f"Mean {name} input-dependence (TV): {tv:.4f}", end="")
        if fixed_frac is not None:
            print(f"   (top-1 fixed for {fixed_frac*100:.1f}% of positions)", end="")
        print()
        if tv < 0.02:
            print(f"  -> {name.upper()} IS DEGENERATE: effectively identical for every input.")
            print(f"     This path is a fixed routing table and carries no information")
            print(f"     about the input. Low entropy here is meaningless.")
            return "degenerate"
        if tv < 0.10:
            print(f"  -> {name.upper()} IS WEAK: barely responds to the input; mostly a")
            print(f"     fixed routing with small content-driven perturbations.")
            return "weak"
        print(f"  -> {name} is input-dependent (content-addressed). Whether it stores")
        print(f"     the RIGHT content is a separate question.")
        return "ok"

    read_v = verdict("read", mean_read_tv)
    write_v = verdict("write", mean_write_tv, mean_write_agree)
    print()
    if "degenerate" in (read_v, write_v):
        print("OVERALL: the workspace is NOT functioning as a content-addressed memory,")
        print("because at least one path ignores the input entirely.")
    elif read_v == "ok" and write_v == "ok":
        print("OVERALL: both paths are input-dependent.")
    else:
        print("OVERALL: partially functional; at least one path is only weakly")
        print("input-dependent.")
    print()
    return {
        "read_tv": mean_read_tv,
        "write_tv": mean_write_tv,
        "write_top1_fixed_frac": mean_write_agree,
        "read_verdict": read_v,
        "write_verdict": write_v,
        "per_call": results,
    }


# Set TT_VISIBLE_DEVICES before importing model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", type=int, required=True, help="Physical device ID")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--depth", type=int, default=4, help="Task 1 chain depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task generation")
    parser.add_argument("--n-probe", type=int, default=16,
                        help="Number of distinct inputs for the input-dependence probe")
    parser.add_argument("--json-output", default=None, help="Save summary metrics as JSON")
    args = parser.parse_args()

    os.environ["TT_VISIBLE_DEVICES"] = str(args.device)

    # Find mesh graph descriptor for P300 chips
    import sys as _sys
    from pathlib import Path as _Path
    for p in _sys.path:
        candidate = _Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / "p150_mesh_graph_descriptor.textproto"
        if candidate.is_file():
            os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", str(candidate))
            break
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_ttnn import TTMambaWorkspaceModel, ModelConfig
    from data import Vocab, gen_task1, VAR_NAMES, MOD
    from train_ttnn import build_model_config

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cell = cfg.get("cell", "?")
    seq_len = cfg.get("seq_len", 128)

    # Open device
    device = ttnn.open_device(device_id=0)
    print(f"Device: {device}")

    # Build model
    model_config = build_model_config(cfg)
    model = TTMambaWorkspaceModel(model_config, device)
    print(f"Cell: {cell}, Params: {model.get_num_params() / 1e6:.2f}M")

    # Load checkpoint
    model.load_checkpoint(args.checkpoint, device=device)
    step = torch.load(args.checkpoint, weights_only=False).get("step", 0)
    print(f"Checkpoint step: {step}")
    print()

    # Generate a Task 1 example with known intermediates
    vocab = Vocab()
    rng = random.Random(args.seed)
    prompt, answer_str, answer_int = gen_task1(args.depth, rng)

    # Parse the prompt to get intermediate values
    # Format: "a=5;b=a+3;c=b*2;d=c-1;?d;" (with possible distractors)
    parts = prompt.rstrip(";").split(";")
    env = {}
    chain_vars = []
    for p in parts:
        if p.startswith("?"):
            continue
        var, expr = p.split("=")
        # Evaluate
        from data import eval_expr
        val = eval_expr(expr, env)
        env[var] = val
        chain_vars.append((var, val, expr))

    print("=" * 80)
    print("TASK 1 EXAMPLE")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Answer: {answer_str} (={answer_int})")
    print(f"Chain depth: {args.depth}")
    print()
    print("Intermediate values:")
    for var, val, expr in chain_vars:
        marker = " <-- QUERY" if var == parts[-1].replace("?", "") else ""
        print(f"  {var} = {expr} = {val}{marker}")
    print()

    # Encode the prompt
    input_ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long)  # (1, T)
    T = input_ids.shape[1]
    print(f"Input length: {T} tokens")
    print(f"Tokens: {' '.join(vocab.decode([t]) for t in input_ids[0].tolist())}")
    print()

    # Run forward pass with hooks to capture workspace state
    # We need to capture the workspace's internal state at each call.
    # The workspace caches read_attn, write_attn, slots_out in self._cache.
    # We'll run the forward, then inspect the cache after each workspace call.

    # Run forward pass, capturing workspace internals
    ws_calls, logits_tt = capture_ws_calls(model, ttnn, input_ids)

    # Get prediction
    logits = ttnn.to_torch(logits_tt)  # (1, T, V)
    # The model predicts the next token at each position
    # The answer should be predicted at the position after "?d;"
    # Find the query position (where "?" appears)
    query_pos = None
    for i, t in enumerate(input_ids[0].tolist()):
        if vocab.itos.get(t) == "?":
            query_pos = i
    if query_pos is not None:
        pred_token = logits[0, query_pos].argmax().item()
        pred_char = vocab.itos.get(pred_token, "?")
        print(f"Model prediction at '?' position: '{pred_char}' (token {pred_token})")
        print(f"Correct answer: '{answer_str}'")
        print(f"Correct: {pred_char == answer_str}")
    else:
        print("Could not find '?' position in input")
    print()

    # Analyze workspace calls
    print("=" * 80)
    print("WORKSPACE ANALYSIS")
    print("=" * 80)
    print(f"Number of workspace calls: {len(ws_calls)}")
    print()

    for i, ws in enumerate(ws_calls):
        print(f"--- Workspace call {i} (slot_state was {'None' if ws['slot_state_was_none'] else 'provided'}) ---")
        slots_in = ws["slots_in"][0]   # (m, D)
        slots_out = ws["slots_out"][0]  # (m, D)
        read_attn = ws["read_attn"][0]  # (H, m, T)
        write_attn = ws["write_attn"][0]  # (H, T, m)

        # Slot analysis: how much did slots change?
        slot_diff = (slots_out - slots_in).norm(dim=-1)  # (m,)
        slot_in_norm = slots_in.norm(dim=-1)  # (m,)
        slot_out_norm = slots_out.norm(dim=-1)  # (m,)

        print(f"  Slot norms (in):  {slot_in_norm.tolist()}")
        print(f"  Slot norms (out): {slot_out_norm.tolist()}")
        print(f"  Slot change:      {slot_diff.tolist()}")
        print(f"  Mean slot change: {slot_diff.mean().item():.4f}")
        print()

        # Read attention analysis: which positions does each slot attend to?
        # read_attn: (H, m, T) — for each head and slot, distribution over T positions
        print(f"  Read attention (slots -> positions):")
        for h in range(read_attn.shape[0]):
            for s in range(read_attn.shape[1]):
                attn = read_attn[h, s]  # (T,)
                # Find top-3 attended positions
                top_vals, top_pos = attn.topk(3)
                # Entropy (high = uniform, low = selective)
                entropy = -(attn * (attn + 1e-10).log()).sum().item()
                max_attn = attn.max().item()
                # Map positions to characters
                top_chars = [vocab.itos.get(input_ids[0, p].item(), "?") for p in top_pos.tolist()]
                print(f"    Head {h}, Slot {s}: entropy={entropy:.3f}, max={max_attn:.3f}, "
                      f"top3={list(zip(top_chars, [f'{v:.3f}' for v in top_vals.tolist()]))}")
        print()

        # Write attention analysis: which slots does each position write to?
        # write_attn: (H, T, m) — for each head and position, distribution over m slots
        print(f"  Write attention (positions -> slots):")
        # Show attention for key positions (where variables are defined)
        for h in range(min(1, write_attn.shape[0])):  # just head 0 for brevity
            for t_pos in range(write_attn.shape[1]):
                attn = write_attn[h, t_pos]  # (m,)
                entropy = -(attn * (attn + 1e-10).log()).sum().item()
                max_attn = attn.max().item()
                char = vocab.itos.get(input_ids[0, t_pos].item(), "?")
                # Only show positions with meaningful characters (not padding)
                if char not in ["<pad>", "<bos>"]:
                    top_vals, top_slots = attn.topk(2)
                    print(f"    Head {h}, Pos {t_pos} ('{char}'): entropy={entropy:.3f}, max={max_attn:.3f}, "
                          f"top2_slots={list(zip(top_slots.tolist(), [f'{v:.3f}' for v in top_vals.tolist()]))}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Check if slots are changing
    all_changes = [ws["slots_out"][0] - ws["slots_in"][0] for ws in ws_calls]
    total_change = sum(c.norm().item() for c in all_changes)
    print(f"Total slot change across all workspace calls: {total_change:.4f}")
    if total_change < 0.1:
        print("  -> Slots are NOT changing. The workspace is not storing anything.")
    elif total_change < 1.0:
        print("  -> Slots are changing slightly. The workspace may be storing weak signals.")
    else:
        print("  -> Slots are changing significantly. The workspace is actively processing.")

    # Check if read attention is selective
    all_read_entropies = []
    for ws in ws_calls:
        read_attn = ws["read_attn"][0]  # (H, m, T)
        for h in range(read_attn.shape[0]):
            for s in range(read_attn.shape[1]):
                attn = read_attn[h, s]
                entropy = -(attn * (attn + 1e-10).log()).sum().item()
                all_read_entropies.append(entropy)
    mean_read_entropy = np.mean(all_read_entropies)
    max_entropy = np.log(seq_len)  # uniform distribution entropy
    print(f"Mean read attention entropy: {mean_read_entropy:.3f} (max={max_entropy:.3f} for uniform)")
    if mean_read_entropy > 0.9 * max_entropy:
        print("  -> Read attention is NEARLY UNIFORM. The workspace is averaging, not selecting.")
    elif mean_read_entropy > 0.5 * max_entropy:
        print("  -> Read attention is moderately selective.")
    else:
        print("  -> Read attention is SHARP. The workspace is selecting specific positions.")

    # Check if write attention is selective
    all_write_entropies = []
    for ws in ws_calls:
        write_attn = ws["write_attn"][0]  # (H, T, m)
        for h in range(write_attn.shape[0]):
            for t_pos in range(write_attn.shape[1]):
                attn = write_attn[h, t_pos]
                entropy = -(attn * (attn + 1e-10).log()).sum().item()
                all_write_entropies.append(entropy)
    mean_write_entropy = np.mean(all_write_entropies)
    max_write_entropy = np.log(model_config.n_workspace_slots)
    print(f"Mean write attention entropy: {mean_write_entropy:.3f} (max={max_write_entropy:.3f} for uniform)")
    if mean_write_entropy > 0.9 * max_write_entropy:
        print("  -> Write attention is NEARLY UNIFORM. Positions are not selectively writing to slots.")
    elif mean_write_entropy > 0.5 * max_write_entropy:
        print("  -> Write attention is moderately selective.")
    else:
        print("  -> Write attention is SHARP. Positions are selectively writing to specific slots.")

    # Check if slots differ across calls (recurrent core doing different things)
    if len(ws_calls) > 1:
        slot_differences = []
        for i in range(1, len(ws_calls)):
            diff = (ws_calls[i]["slots_in"][0] - ws_calls[i-1]["slots_out"][0]).norm().item()
            slot_differences.append(diff)
        print(f"Slot changes between consecutive workspace calls: {slot_differences}")
        if max(slot_differences) < 0.1:
            print("  -> Slots are NOT changing across iterations. The recurrent core is not using iterations differently.")
        else:
            print("  -> Slots ARE changing across iterations. The recurrent core is doing different things each step.")
    print()

    # --- Input-dependence probe ---------------------------------------------
    # This is the decisive measurement. Entropy above tells us whether attention
    # is peaked; this tells us whether it carries any information about the input.
    dep = analyze_input_dependence(
        model, vocab, gen_task1, args.depth, random.Random(args.seed + 1),
        args.n_probe, model_config.n_workspace_slots,
    )

    if args.json_output:
        import json
        summary = {
            "cell": cell,
            "step": step,
            "checkpoint": args.checkpoint,
            "depth": args.depth,
            "gate_read": float(torch.sigmoid(ttnn.to_torch(model.workspace.read_gate).float()).flatten()[0]),
            "gate_write": float(torch.sigmoid(ttnn.to_torch(model.workspace.write_gate).float()).flatten()[0]),
            "total_slot_change": total_change,
            "mean_read_entropy": float(mean_read_entropy),
            "mean_write_entropy": float(mean_write_entropy),
            "input_dependence": dep,
        }
        with open(args.json_output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {args.json_output}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
