#!/usr/bin/env python3
"""Evaluate a TT checkpoint on the task eval set.

Loads a checkpoint, runs the model on generated eval examples, and reports
accuracy per task and per depth. Also shows sample predictions.

Usage:
    python eval_ttnn.py --config configs/cell_b_tt.yaml --device 1 \
        --checkpoint run_B/checkpoints/cell_B_step300.pt

    # Extended depths (extrapolation beyond training range)
    python eval_ttnn.py --config configs/cell_c_attn_residual.yaml --device 0 \
        --checkpoint checkpoints/cell_C_step5000.pt \
        --depths 2 4 6 8 10 12 14 16

    # K-sweep for Cell C (test-time compute scaling, R2)
    python eval_ttnn.py --config configs/cell_c_attn_residual.yaml --device 0 \
        --checkpoint checkpoints/cell_C_step5000.pt \
        --depths 2 4 6 8 10 12 14 16 \
        --k-sweep 1 2 4 6 8 12 16 \
        --json-output /tmp/eval_C_ksweep.json
"""

import os
import sys
import argparse
import yaml
import time
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
    parser.add_argument("--n-per-task", type=int, default=20, help="Examples per task per depth")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--show-samples", type=int, default=10, help="Number of sample predictions to show")
    parser.add_argument("--max-new", type=int, default=10, help="Max tokens to generate")
    parser.add_argument("--json-output", type=str, default=None, help="Save results as JSON to this path")
    parser.add_argument("--k-sweep", type=int, nargs="+", default=None,
                        help="Sweep inference K (recurrent core iterations). "
                             "Only meaningful for Cell C. Runs the full eval at each K value "
                             "and produces a K-vs-accuracy table.")
    args = parser.parse_args()

    # Set TT_VISIBLE_DEVICES before importing ttnn-heavy modules
    os.environ["TT_VISIBLE_DEVICES"] = str(args.device)

    # Find mesh graph descriptor for P300 chips (try p150 first, then p300)
    # p150 works for single-device and after board reset;
    # p300 is needed for multi-chip fabric topologies
    import sys as _sys
    from pathlib import Path as _Path
    _mgd_names = ["p150_mesh_graph_descriptor.textproto", "p300_mesh_graph_descriptor.textproto"]
    _venv_path = _Path("/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages")
    for _name in _mgd_names:
        for p in _sys.path:
            candidate = _Path(p) / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / _name
            if candidate.is_file():
                os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", str(candidate))
                break
        else:
            candidate = _venv_path / "pjrt_plugin_tt" / "tt-metal" / "tt_metal" / "fabric" / "mesh_graph_descriptors" / _name
            if candidate.is_file():
                os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", str(candidate))
                break
        if "TT_MESH_GRAPH_DESC_PATH" in os.environ:
            break
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_ttnn import TTWRAPModel, ModelConfig
    from data import Vocab, generate_eval_set, TASK_VERIFIERS
    from train_ttnn import build_model_config

    TASK_NAMES = {1: "Task1 (chain arithmetic)", 2: "Task2 (2-var arithmetic)", 3: "Task3 (free-form)"}

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
    # Disable slot permutation during eval for deterministic results.
    # The permutation is a training-time augmentation; at eval we want
    # consistent slot ordering to measure the model's learned routing.
    model_config.slot_permutation = False
    model = TTWRAPModel(model_config, device)
    print(f"Cell: {cell}, Params: {model.get_num_params() / 1e6:.2f}M")

    # Load checkpoint
    model.load_checkpoint(args.checkpoint, device=device)
    step = torch.load(args.checkpoint, weights_only=False).get("step", 0)
    print(f"Checkpoint step: {step}")

    # Generate eval set (fixed seed for reproducibility — same examples across K values)
    vocab = Vocab()
    rng = random.Random(123)
    eval_set = generate_eval_set(
        n_per_task_per_depth=args.n_per_task,
        depths=args.depths,
        vocab=vocab,
        seq_len=seq_len,
        rng=rng,
    )
    print(f"Eval set: {len(eval_set)} examples ({args.n_per_task} per task per depth)")
    print()

    # Determine K values to sweep
    if args.k_sweep:
        k_values = args.k_sweep
        print(f"K-sweep: {k_values}")
    else:
        k_values = [None]  # single run, use model default

    # --- Eval function ---
    def run_eval(k_value, show_samples=0):
        """Run evaluation at a given K value. Returns results dict."""
        correct = {1: 0, 2: 0, 3: 0}
        total = {1: 0, 2: 0, 3: 0}
        correct_by_depth = {}
        total_by_depth = {}
        samples_shown = 0

        tf_correct = {1: 0, 2: 0, 3: 0}
        tf_tok_correct = {1: 0, 2: 0, 3: 0}
        tf_tok_total = {1: 0, 2: 0, 3: 0}

        # Per-task per-depth accuracy (for K-sweep heatmap)
        correct_by_task_depth = {}
        total_by_task_depth = {}

        with torch.no_grad():
            for i, ex in enumerate(eval_set):
                input_ids_1 = ex["input_ids"].unsqueeze(0)  # (1, T) — full padded sequence
                task_id = ex["task_id"]
                depth = ex["depth"]
                prompt = ex["prompt"]
                answer_str = ex["answer_str"]

                # Find prompt length (where answer starts)
                prompt_ids = vocab.encode(prompt)
                prompt_len = len(prompt_ids) - 1  # exclude EOS so model predicts first answer char

                # TT-Metal matmul requires batch >= 2 (tile alignment).
                # Replicate the single example to batch size 4 and use row 0.
                BATCH_PAD = 4
                input_ids = input_ids_1.expand(BATCH_PAD, -1).contiguous()  # (4, T)

                # Generate answer tokens autoregressively.
                # IMPORTANT: use the full padded 128-token sequence (same as training),
                # not a truncated prompt. The workspace's cross-attention between slots
                # and the sequence is length-sensitive — truncating changes the attention
                # pattern and causes the model to fail. We overwrite tokens at positions
                # prompt_len..prompt_len+max_new with generated tokens, keeping the rest
                # as padding.
                generated = input_ids.clone()  # (4, T) full padded sequence
                generated_tokens = []

                for gen_step in range(args.max_new):
                    logits_tt = model.forward(generated, k_value=k_value)
                    logits = ttnn.to_torch(logits_tt)  # (4, T, V)
                    # Logit at position (prompt_len + gen_step - 1) predicts token at (prompt_len + gen_step)
                    # Use only row 0 (the actual example)
                    next_logits = logits[:1, prompt_len + gen_step - 1, :]
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated[:1, prompt_len + gen_step] = next_token
                    generated_tokens.append(next_token.item())
                    if next_token.item() == vocab.EOS:
                        break

                response = vocab.decode(generated_tokens)
                is_correct = TASK_VERIFIERS[task_id](prompt, response)

                total[task_id] += 1
                correct[task_id] += int(is_correct)

                # --- Teacher-forced pass: gold prefix at every answer position ---
                # Use the full padded input with answer tokens placed at the correct
                # positions (same as training). The model sees the same 128-token
                # padded sequence format it was trained on.
                answer_tokens = [vocab.stoi[c] for c in answer_str if c in vocab.stoi]
                if answer_tokens:
                    tf_input = input_ids.clone()  # (4, T) full padded
                    for j, tok in enumerate(answer_tokens):
                        if prompt_len + j < tf_input.shape[1]:
                            tf_input[:1, prompt_len + j] = tok
                    tf_logits = ttnn.to_torch(model.forward(tf_input, k_value=k_value))  # (4, T, V)
                    n_tok_ok = 0
                    for j, gold in enumerate(answer_tokens):
                        # logit at index (prompt_len + j - 1) predicts token at prompt_len + j
                        # Use row 0 (the actual example)
                        pred = tf_logits[0, prompt_len + j - 1].argmax().item()
                        n_tok_ok += int(pred == gold)
                    tf_tok_correct[task_id] += n_tok_ok
                    tf_tok_total[task_id] += len(answer_tokens)
                    tf_correct[task_id] += int(n_tok_ok == len(answer_tokens))

                if depth not in correct_by_depth:
                    correct_by_depth[depth] = 0
                    total_by_depth[depth] = 0
                total_by_depth[depth] += 1
                correct_by_depth[depth] += int(is_correct)

                # Per-task per-depth
                td_key = (task_id, depth)
                if td_key not in correct_by_task_depth:
                    correct_by_task_depth[td_key] = 0
                    total_by_task_depth[td_key] = 0
                total_by_task_depth[td_key] += 1
                correct_by_task_depth[td_key] += int(is_correct)

                # Show samples
                if samples_shown < show_samples:
                    status = "✓" if is_correct else "✗"
                    task_name = TASK_NAMES.get(task_id, f"Task{task_id}")
                    k_label = f" K={k_value}" if k_value is not None else ""
                    print(f"  {status} [{task_name} d={depth}{k_label}] "
                          f"prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
                    print(f"    predicted: '{response}'  expected: '{answer_str}'")
                    samples_shown += 1

                # Clear model caches between examples to avoid device memory leaks.
                # The autoregressive generation runs multiple forward passes per
                # example, accumulating cached tensors. Without clearing, device
                # DRAM fills up after ~5 examples.
                try:
                    model.clear_caches()
                except Exception:
                    pass

        # Compute summary stats
        gen_all = sum(correct.values()) / max(sum(total.values()), 1)
        tf_all = sum(tf_correct.values()) / max(sum(total.values()), 1)
        tf_tok_all = sum(tf_tok_correct.values()) / max(sum(tf_tok_total.values()), 1)

        return {
            "correct": correct,
            "total": total,
            "correct_by_depth": correct_by_depth,
            "total_by_depth": total_by_depth,
            "tf_correct": tf_correct,
            "tf_tok_correct": tf_tok_correct,
            "tf_tok_total": tf_tok_total,
            "tf_all": tf_all,
            "tf_tok_all": tf_tok_all,
            "gen_all": gen_all,
            "correct_by_task_depth": correct_by_task_depth,
            "total_by_task_depth": total_by_task_depth,
        }

    def print_results(res, k_label=""):
        """Print results table for a single eval run."""
        correct = res["correct"]
        total = res["total"]
        correct_by_depth = res["correct_by_depth"]
        total_by_depth = res["total_by_depth"]
        tf_correct = res["tf_correct"]
        tf_tok_correct = res["tf_tok_correct"]
        tf_tok_total = res["tf_tok_total"]

        print(f"\nPer-task accuracy (gen = free generation, tf = teacher-forced):")
        print(f"  {'task':<28} {'gen':>7} {'tf exact':>9} {'tf/token':>9}")
        for tid in [1, 2, 3]:
            acc = correct[tid] / max(total[tid], 1)
            tf_acc = tf_correct[tid] / max(total[tid], 1)
            tf_tok = tf_tok_correct[tid] / max(tf_tok_total[tid], 1)
            task_name = TASK_NAMES.get(tid, f"Task{tid}")
            print(f"  {task_name:<28} {acc:>6.1%} {tf_acc:>9.1%} {tf_tok:>9.1%}")

        gen_all = res["gen_all"]
        tf_all = res["tf_all"]
        tf_tok_all = res["tf_tok_all"]
        print(f"\n  Interpretation: tf exact ({tf_all:.1%}) vs gen ({gen_all:.1%}) — ", end="")
        if tf_all - gen_all > 0.15:
            print("large gap => exposure bias.")
            print("  The model largely knows each token given a correct prefix but cannot")
            print("  recover from its own errors. Reasoning is partly there; decoding is not.")
        elif tf_tok_all < 0.3:
            print("both low => genuine failure.")
            print("  Even with the gold prefix the model cannot produce the answer tokens.")
            print("  This is a reasoning failure, not a decoding artifact.")
        else:
            print("similar => decoding is not the bottleneck.")

        print(f"\nPer-depth accuracy:")
        for depth in sorted(total_by_depth.keys()):
            acc = correct_by_depth[depth] / max(total_by_depth[depth], 1)
            print(f"  Depth {depth:>2}: {correct_by_depth[depth]}/{total_by_depth[depth]} = {acc:.1%}")

        overall = sum(correct.values()) / max(sum(total.values()), 1)
        print(f"\nOverall: {sum(correct.values())}/{sum(total.values())} = {overall:.1%}")

    # --- Run eval(s) ---
    all_results = {}

    for k_val in k_values:
        k_label = f"K={k_val}" if k_val is not None else "default"
        print(f"\n{'='*60}")
        print(f"Evaluating: Cell {cell} (step {step}), {k_label}")
        print(f"{'='*60}")

        show = args.show_samples if k_val == k_values[0] else 0
        res = run_eval(k_val, show_samples=show)
        print_results(res, k_label)

        all_results[k_label] = res

    # --- K-sweep summary table ---
    if args.k_sweep and len(k_values) > 1:
        print(f"\n{'='*60}")
        print(f"K-sweep summary: Cell {cell} (step {step})")
        print(f"{'='*60}")

        # Overall accuracy vs K
        print(f"\nOverall accuracy vs K:")
        print(f"  {'K':>4}  {'Overall':>8}  {'Task1':>8}  {'Task2':>8}  {'Task3':>8}")
        for k_val in k_values:
            k_label = f"K={k_val}"
            res = all_results[k_label]
            overall = res["gen_all"]
            t1 = res["correct"][1] / max(res["total"][1], 1)
            t2 = res["correct"][2] / max(res["total"][2], 1)
            t3 = res["correct"][3] / max(res["total"][3], 1)
            print(f"  {k_val:>4}  {overall:>7.1%}  {t1:>7.1%}  {t2:>7.1%}  {t3:>7.1%}")

        # Per-depth accuracy vs K (the R2 curve)
        print(f"\nPer-depth accuracy vs K (R2: test-time compute scaling):")
        depth_header = "  K    " + "  ".join(f"d={d:>2}" for d in sorted(args.depths))
        print(depth_header)
        for k_val in k_values:
            k_label = f"K={k_val}"
            res = all_results[k_label]
            cbd = res["correct_by_depth"]
            tbd = res["total_by_depth"]
            row = f"  {k_val:>4}  "
            for d in sorted(args.depths):
                if d in tbd and tbd[d] > 0:
                    acc = cbd[d] / tbd[d]
                    row += f" {acc:>5.1%}"
                else:
                    row += f"   —  "
            print(row)

        # Per-task per-depth accuracy vs K (most detailed)
        print(f"\nTask 1 (chain arithmetic) accuracy vs K and depth:")
        print(depth_header)
        for k_val in k_values:
            k_label = f"K={k_val}"
            res = all_results[k_label]
            cbtd = res["correct_by_task_depth"]
            tbtd = res["total_by_task_depth"]
            row = f"  {k_val:>4}  "
            for d in sorted(args.depths):
                key = (1, d)
                if key in tbtd and tbtd[key] > 0:
                    acc = cbtd[key] / tbtd[key]
                    row += f" {acc:>5.1%}"
                else:
                    row += f"   —  "
            print(row)

    # --- Save JSON ---
    if args.json_output:
        import json
        if args.k_sweep and len(k_values) > 1:
            # K-sweep JSON: structured by K value
            results = {
                "cell": cell,
                "step": step,
                "checkpoint": args.checkpoint,
                "n_examples": len(eval_set),
                "depths": args.depths,
                "k_sweep": k_values,
                "k_results": {},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            for k_val in k_values:
                k_label = f"K={k_val}"
                res = all_results[k_label]
                results["k_results"][k_label] = {
                    "overall_accuracy": res["gen_all"],
                    "task_accuracy": {
                        f"task{tid}": res["correct"][tid] / max(res["total"][tid], 1)
                        for tid in [1, 2, 3]
                    },
                    "depth_accuracy": {
                        str(d): res["correct_by_depth"][d] / max(res["total_by_depth"][d], 1)
                        for d in sorted(res["total_by_depth"].keys())
                    },
                    "task_depth_accuracy": {
                        f"task{tid}_d{d}": res["correct_by_task_depth"][(tid, d)] / max(res["total_by_task_depth"][(tid, d)], 1)
                        for tid in [1, 2, 3] for d in sorted(args.depths)
                        if (tid, d) in res["total_by_task_depth"]
                    },
                }
        else:
            # Single K run
            k_label = f"K={k_values[0]}" if k_values[0] is not None else "default"
            res = all_results[k_label]
            results = {
                "cell": cell,
                "step": step,
                "checkpoint": args.checkpoint,
                "n_examples": len(eval_set),
                "task_accuracy": {
                    f"task{tid}": res["correct"][tid] / max(res["total"][tid], 1)
                    for tid in [1, 2, 3]
                },
                "teacher_forced_exact": {
                    f"task{tid}": res["tf_correct"][tid] / max(res["total"][tid], 1)
                    for tid in [1, 2, 3]
                },
                "teacher_forced_token": {
                    f"task{tid}": res["tf_tok_correct"][tid] / max(res["tf_tok_total"][tid], 1)
                    for tid in [1, 2, 3]
                },
                "teacher_forced_exact_overall": res["tf_all"],
                "teacher_forced_token_overall": res["tf_tok_all"],
                "depth_accuracy": {
                    str(d): res["correct_by_depth"][d] / max(res["total_by_depth"][d], 1)
                    for d in sorted(res["total_by_depth"].keys())
                },
                "overall_accuracy": res["gen_all"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json_output}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
