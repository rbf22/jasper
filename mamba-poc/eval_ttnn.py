#!/usr/bin/env python3
"""Evaluate a TT checkpoint on the task eval set.

Loads a checkpoint, runs the model on generated eval examples, and reports
accuracy per task and per depth. Also shows sample predictions.

Usage:
    python eval_ttnn.py --config configs/cell_b_tt.yaml --device 1 \
        --checkpoint run_B/checkpoints/cell_B_step300.pt
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
    args = parser.parse_args()

    # Set TT_VISIBLE_DEVICES before importing ttnn-heavy modules
    os.environ["TT_VISIBLE_DEVICES"] = str(args.device)

    # Find mesh graph descriptor for P300 chips (try p300 first, then p150)
    import sys as _sys
    from pathlib import Path as _Path
    _mgd_names = ["p300_mesh_graph_descriptor.textproto", "p150_mesh_graph_descriptor.textproto"]
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
    from model_ttnn import TTMambaWorkspaceModel, ModelConfig
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
    model = TTMambaWorkspaceModel(model_config, device)
    print(f"Cell: {cell}, Params: {model.get_num_params() / 1e6:.2f}M")

    # Load checkpoint
    model.load_checkpoint(args.checkpoint, device=device)
    step = torch.load(args.checkpoint, weights_only=False).get("step", 0)
    print(f"Checkpoint step: {step}")

    # Generate eval set
    vocab = Vocab()
    rng = random.Random(123)  # fixed seed for reproducibility
    eval_set = generate_eval_set(
        n_per_task_per_depth=args.n_per_task,
        depths=args.depths,
        vocab=vocab,
        seq_len=seq_len,
        rng=rng,
    )
    print(f"Eval set: {len(eval_set)} examples ({args.n_per_task} per task per depth)")
    print()

    # Run evaluation
    correct = {1: 0, 2: 0, 3: 0}
    total = {1: 0, 2: 0, 3: 0}
    correct_by_depth = {}
    total_by_depth = {}
    samples_shown = 0

    # Teacher-forced accuracy, measured alongside free generation.
    # Generation is autoregressive: one wrong digit corrupts every later step, so
    # a low generation score conflates "cannot reason" with "cannot recover from
    # its own mistakes" (exposure bias). Teacher-forcing feeds the gold prefix at
    # every answer position, isolating whether the model knows each digit given a
    # correct history. Large tf >> gen gap = exposure bias; tf also low = the
    # model genuinely does not compute the answer.
    tf_correct = {1: 0, 2: 0, 3: 0}      # all answer tokens correct
    tf_tok_correct = {1: 0, 2: 0, 3: 0}  # per-token
    tf_tok_total = {1: 0, 2: 0, 3: 0}

    with torch.no_grad():
        for i, ex in enumerate(eval_set):
            input_ids = ex["input_ids"].unsqueeze(0)  # (1, T)
            task_id = ex["task_id"]
            depth = ex["depth"]
            prompt = ex["prompt"]
            answer_str = ex["answer_str"]

            # Find prompt length (where answer starts)
            prompt_ids = vocab.encode(prompt)
            prompt_len = len(prompt_ids) - 1  # exclude EOS so model predicts first answer char

            # Generate answer tokens autoregressively
            generated = input_ids[:, :prompt_len].clone()
            generated_tokens = []

            for _ in range(args.max_new):
                logits_tt = model.forward(generated)
                logits = ttnn.to_torch(logits_tt)  # (1, T, V)
                next_logits = logits[:, -1, :]  # last position
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                generated_tokens.append(next_token.item())
                if next_token.item() == vocab.EOS:
                    break

            response = vocab.decode(generated_tokens)
            is_correct = TASK_VERIFIERS[task_id](prompt, response)

            total[task_id] += 1
            correct[task_id] += int(is_correct)

            # --- Teacher-forced pass: gold prefix at every answer position ---
            # encode() wraps in BOS/EOS, so take the bare answer chars. This makes
            # prompt_len the index of the first answer token (matches data.py).
            answer_tokens = [vocab.stoi[c] for c in answer_str if c in vocab.stoi]
            if answer_tokens:
                tf_input = torch.cat([
                    input_ids[:, :prompt_len],
                    torch.tensor([answer_tokens], dtype=input_ids.dtype),
                ], dim=1)
                tf_logits = ttnn.to_torch(model.forward(tf_input))  # (1, L, V)
                n_tok_ok = 0
                for j, gold in enumerate(answer_tokens):
                    # logit at index (prompt_len + j - 1) predicts token at prompt_len + j
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

            # Show samples
            if samples_shown < args.show_samples:
                status = "✓" if is_correct else "✗"
                task_name = TASK_NAMES.get(task_id, f"Task{task_id}")
                print(f"  {status} [{task_name} d={depth}] "
                      f"prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
                print(f"    predicted: '{response}'  expected: '{answer_str}'")
                samples_shown += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"Results: Cell {cell} (step {step})")
    print(f"{'='*60}")
    print(f"\nPer-task accuracy (gen = free generation, tf = teacher-forced):")
    print(f"  {'task':<28} {'gen':>7} {'tf exact':>9} {'tf/token':>9}")
    for tid in [1, 2, 3]:
        acc = correct[tid] / max(total[tid], 1)
        tf_acc = tf_correct[tid] / max(total[tid], 1)
        tf_tok = tf_tok_correct[tid] / max(tf_tok_total[tid], 1)
        task_name = TASK_NAMES.get(tid, f"Task{tid}")
        print(f"  {task_name:<28} {acc:>6.1%} {tf_acc:>9.1%} {tf_tok:>9.1%}")

    gen_all = sum(correct.values()) / max(sum(total.values()), 1)
    tf_all = sum(tf_correct.values()) / max(sum(total.values()), 1)
    tf_tok_all = sum(tf_tok_correct.values()) / max(sum(tf_tok_total.values()), 1)
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
        print(f"  Depth {depth}: {correct_by_depth[depth]}/{total_by_depth[depth]} = {acc:.1%}")

    overall = sum(correct.values()) / max(sum(total.values()), 1)
    print(f"\nOverall: {sum(correct.values())}/{sum(total.values())} = {overall:.1%}")

    # Save JSON results if requested
    if args.json_output:
        import json
        results = {
            "cell": cell,
            "step": step,
            "checkpoint": args.checkpoint,
            "n_examples": len(eval_set),
            "task_accuracy": {
                f"task{tid}": correct[tid] / max(total[tid], 1)
                for tid in [1, 2, 3]
            },
            "teacher_forced_exact": {
                f"task{tid}": tf_correct[tid] / max(total[tid], 1)
                for tid in [1, 2, 3]
            },
            "teacher_forced_token": {
                f"task{tid}": tf_tok_correct[tid] / max(tf_tok_total[tid], 1)
                for tid in [1, 2, 3]
            },
            "teacher_forced_exact_overall": tf_all,
            "teacher_forced_token_overall": tf_tok_all,
            "depth_accuracy": {
                str(d): correct_by_depth[d] / max(total_by_depth[d], 1)
                for d in sorted(total_by_depth.keys())
            },
            "overall_accuracy": overall,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json_output}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
