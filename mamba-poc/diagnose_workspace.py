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

# Set TT_VISIBLE_DEVICES before importing model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", type=int, required=True, help="Physical device ID")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--depth", type=int, default=4, help="Task 1 chain depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task generation")
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

    # Monkey-patch the workspace forward to capture state
    ws_calls = []
    original_forward = model.workspace.forward

    def instrumented_forward(x, slot_state):
        # Call original
        x_out, slot_state_out = original_forward(x, slot_state)
        # Capture state from cache
        cache = model.workspace._cache
        B = int(cache["B"])
        T_local = int(cache["T"])
        m = int(cache["m"])

        # Extract to torch for analysis
        slots_in = ttnn.to_torch(cache["slots_in"])      # (B, m, D)
        slots_out = ttnn.to_torch(cache["slots_out"])     # (B, m, D)
        read_attn = ttnn.to_torch(cache["read_attn"])     # (B, H, m, T)
        write_attn = ttnn.to_torch(cache["write_attn"])   # (B, H, T, m)

        ws_calls.append({
            "slots_in": slots_in.clone(),
            "slots_out": slots_out.clone(),
            "read_attn": read_attn.clone(),
            "write_attn": write_attn.clone(),
            "slot_state_was_none": slot_state is None,
            "B": B, "T": T_local, "m": m,
        })

        return x_out, slot_state_out

    model.workspace.forward = instrumented_forward

    # Run forward pass
    with torch.no_grad():
        logits_tt = model.forward(input_ids)

    # Restore original
    model.workspace.forward = original_forward

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

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
