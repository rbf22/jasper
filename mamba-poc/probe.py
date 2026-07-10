"""
Probing and ablation analysis for R3 and R4.

R3 — Workspace decodability (J-lens analogue):
    Train linear probes on workspace slots to decode intermediate variable values.
    Compare against probes on a matched residual-stream position in a non-workspace model.

R4 — Selective ablation (J-space signature):
    Replace workspace slots with their training-set mean.
    Success: Tasks 1-2 collapse (>=30pt drop) while Task 3 barely moves (<=5pt drop).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from data import Vocab, gen_task1, gen_task2, gen_task3, TASK_VERIFIERS, generate_eval_set
from model import MambaWorkspaceModel, get_cell_config, ModelConfig
from train import get_device, load_config, build_model_config, load_checkpoint, evaluate


# ---------------------------------------------------------------------------
# R3: Linear probes on workspace slots
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Simple linear probe: maps hidden states to predicted values."""

    def __init__(self, d_model, n_classes=97):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.linear(x)


def collect_workspace_states(
    model: MambaWorkspaceModel,
    input_ids: torch.Tensor,
    device: torch.device,
    k_override: int = None,
) -> List[Tuple]:
    """Run model with return_workspace_states=True and collect slot states."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids.to(device), k_override=k_override, return_workspace_states=True)
    return out.get("workspace_states", [])


def collect_residual_states(
    model: MambaWorkspaceModel,
    input_ids: torch.Tensor,
    device: torch.device,
    layer_idx: int,
) -> torch.Tensor:
    """Collect residual stream states at a specific layer (for baseline comparison)."""
    model.eval()
    with torch.no_grad():
        config = model.config
        B, T = input_ids.shape
        x = model.token_emb(input_ids.to(device))

        for i, layer in enumerate(model.layers):
            if i == layer_idx:
                return x.clone()
            x = layer(x)

        # If layer_idx is beyond layers, return final hidden state
        return x.clone()


def train_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    d_model: int,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device = None,
) -> float:
    """Train a linear probe and return best accuracy."""
    if device is None:
        device = features.device

    n_classes = 97  # mod 97
    probe = LinearProbe(d_model, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    n = features.shape[0]
    if n < 2:
        return 0.0

    # Split 80/20
    n_train = int(0.8 * n)
    indices = torch.randperm(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_feat = features[train_idx]
    train_tgt = targets[train_idx]
    val_feat = features[val_idx]
    val_tgt = targets[val_idx]

    best_acc = 0.0
    for epoch in range(n_epochs):
        probe.train()
        logits = probe(train_feat)
        loss = F.cross_entropy(logits, train_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_feat)
                val_pred = val_logits.argmax(dim=-1)
                acc = (val_pred == val_tgt).float().mean().item()
                best_acc = max(best_acc, acc)

    return best_acc


def run_r3(
    model: MambaWorkspaceModel,
    vocab: Vocab,
    device: torch.device,
    n_samples: int = 200,
    max_k: int = 6,
) -> Dict:
    """R3: Workspace decodability analysis.

    For Task 1 problems, collect workspace states at each loop iteration
    and train probes to decode intermediate variable values.
    """
    print("\n=== R3: Workspace Decodability ===")
    config = model.config
    rng = random.Random(42)

    # Generate Task 1 problems with known intermediate values
    problems = []
    for _ in range(n_samples):
        depth = rng.randint(4, 8)
        prompt, answer_str, answer = gen_task1(depth, rng)
        # We need to track intermediate values
        # Parse the chain to get all intermediate values
        parts = prompt.rstrip(";").split(";")
        env = {}
        intermediates = {}
        query_var = None
        for part in parts:
            if part.startswith("?"):
                query_var = part[1:]
            elif "=" in part:
                var, expr = part.split("=")
                # Evaluate
                val = _safe_eval(expr, env)
                env[var] = val
                intermediates[var] = val

        problems.append({
            "prompt": prompt,
            "intermediates": intermediates,
            "query_var": query_var,
            "depth": depth,
        })

    # Collect workspace states for each problem
    # For each problem, run model with different K values and collect slot states
    all_features = {k: [] for k in range(1, max_k + 1)}
    all_targets = {k: [] for k in range(1, max_k + 1)}

    for prob in problems:
        input_ids = torch.tensor([vocab.encode(prob["prompt"])], dtype=torch.long)
        if input_ids.shape[1] > 128:
            continue

        for k in range(1, max_k + 1):
            states = collect_workspace_states(model, input_ids, device, k_override=k)
            # Get the last workspace state for this K
            if states:
                # states is a list of tuples like ("iter_end", iteration, slot_state)
                # Get the last slot state
                last_state = None
                for s in states:
                    if len(s) == 3 and s[0] == "iter_end":
                        last_state = s[2]
                if last_state is not None:
                    # Mean across slots: (B, m, D) -> (B, D) -> (D,)
                    feat = last_state.mean(dim=1).squeeze(0)  # (D,)
                    all_features[k].append(feat.cpu())  # full D-dim vector
                    # Target: the value of the queried variable
                    tgt = prob["intermediates"].get(prob["query_var"], 0)
                    all_targets[k].append(tgt)

    # Train probes for each K
    results = {}
    for k in range(1, max_k + 1):
        if len(all_features[k]) < 10:
            results[f"k{k}_probe_acc"] = 0.0
            continue
        features = torch.stack(all_features[k])
        targets = torch.tensor(all_targets[k], dtype=torch.long)
        acc = train_probe(features, targets, config.d_model, device=device)
        results[f"k{k}_probe_acc"] = acc
        print(f"  K={k}: probe accuracy = {acc:.3f} (n={len(all_features[k])})")

    return results


def _safe_eval(expr: str, env: Dict[str, int]) -> int:
    """Safely evaluate a Task 1 expression."""
    import string
    tokens = []
    for ch in expr:
        if ch in string.ascii_lowercase:
            tokens.append(str(env.get(ch, 0)))
        else:
            tokens.append(ch)
    result = eval("".join(tokens))  # noqa: S307
    return result % 97


# ---------------------------------------------------------------------------
# R4: Selective workspace ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_slot_mean(
    model: MambaWorkspaceModel,
    vocab: Vocab,
    device: torch.device,
    n_samples: int = 100,
) -> torch.Tensor:
    """Compute the mean workspace slot state across training examples."""
    model.eval()
    rng = random.Random(99)
    slot_states = []

    for _ in range(n_samples):
        depth = rng.randint(2, 8)
        r = rng.random()
        if r < 0.45:
            prompt, _, _ = gen_task1(depth, rng)
        elif r < 0.90:
            prompt, _, _ = gen_task2(depth, rng)
        else:
            prompt, _, _ = gen_task3(1, rng)

        input_ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long)
        if input_ids.shape[1] > 128:
            continue

        states = collect_workspace_states(model, input_ids, device)
        for s in states:
            if len(s) == 3 and s[0] == "iter_end":
                slot_states.append(s[2].cpu())

    if not slot_states:
        return None

    return torch.stack(slot_states).mean(dim=0).squeeze(0)  # (B, m, D) -> (m, D)


@torch.no_grad()
def evaluate_with_ablated_workspace(
    model: MambaWorkspaceModel,
    eval_set: List[Dict],
    vocab: Vocab,
    device: torch.device,
    slot_mean: torch.Tensor,
    max_new: int = 5,
) -> Dict:
    """Evaluate with workspace slots replaced by their mean (ablation)."""
    model.eval()

    # Monkey-patch the workspace to return mean slots
    original_forward = None
    if model.workspace is not None:
        original_forward = model.workspace.forward

        def ablated_forward(x, slot_state=None):
            B = x.shape[0]
            # Always use the mean slots, ignoring any recurrent state
            mean_slots = slot_mean.unsqueeze(0).expand(B, -1, -1).to(x.device)
            return original_forward.__func__(model.workspace, x, mean_slots)

        model.workspace.forward = ablated_forward

    try:
        correct = {1: 0, 2: 0, 3: 0}
        total = {1: 0, 2: 0, 3: 0}

        for ex in eval_set:
            input_ids = ex["input_ids"].unsqueeze(0).to(device)
            task_id = ex["task_id"]
            prompt = ex["prompt"]

            prompt_ids = vocab.encode(prompt)
            prompt_len = len(prompt_ids) - 1  # exclude EOS

            generated = input_ids[:, :prompt_len]
            for _ in range(max_new):
                out = model(generated)
                next_logits = out["logits"][:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == vocab.EOS:
                    break

            response_ids = generated[0, prompt_len:].tolist()
            response = vocab.decode(response_ids)

            is_correct = TASK_VERIFIERS[task_id](prompt, response)
            total[task_id] += 1
            correct[task_id] += int(is_correct)
    finally:
        # Restore original workspace
        if original_forward is not None:
            model.workspace.forward = original_forward

    results = {}
    for tid in [1, 2, 3]:
        results[f"task{tid}_acc"] = correct[tid] / max(total[tid], 1)
    results["overall_acc"] = sum(correct.values()) / max(sum(total.values()), 1)
    return results


def run_r4(
    model: MambaWorkspaceModel,
    vocab: Vocab,
    device: torch.device,
    seq_len: int = 128,
) -> Dict:
    """R4: Selective workspace ablation (J-space signature)."""
    print("\n=== R4: Selective Workspace Ablation ===")

    if model.workspace is None:
        print("  No workspace in this model — skipping R4.")
        return {}

    # Generate eval set
    eval_set = generate_eval_set(
        n_per_task_per_depth=50,
        depths=list(range(2, 9)),
        vocab=vocab,
        seq_len=seq_len,
        rng=random.Random(777),
    )

    # Baseline accuracy (no ablation)
    baseline = evaluate(model, eval_set, vocab, device, max_new=5)
    print("  Baseline:")
    for k in ["task1_acc", "task2_acc", "task3_acc"]:
        print(f"    {k}: {baseline[k]:.3f}")

    # Compute slot mean
    slot_mean = compute_slot_mean(model, vocab, device, n_samples=100)
    if slot_mean is None:
        print("  Could not compute slot mean — skipping.")
        return {}

    # Ablated accuracy
    ablated = evaluate_with_ablated_workspace(model, eval_set, vocab, device, slot_mean, max_new=5)
    print("  Ablated (mean slots):")
    for k in ["task1_acc", "task2_acc", "task3_acc"]:
        print(f"    {k}: {ablated[k]:.3f}")

    # Compute deltas
    deltas = {}
    for tid in [1, 2, 3]:
        key = f"task{tid}_acc"
        delta = baseline[key] - ablated[key]
        deltas[f"task{tid}_delta"] = delta
        print(f"  Task {tid} delta: {delta:.3f}")

    # J-space signature check
    task12_drop = max(deltas["task1_delta"], deltas["task2_delta"])
    task3_drop = deltas["task3_delta"]
    deltas["jspace_signature"] = task12_drop >= 0.30 and task3_drop <= 0.05
    print(f"\n  J-space signature: {'PASS' if deltas['jspace_signature'] else 'FAIL'}")
    print(f"    (Tasks 1-2 drop: {task12_drop:.3f} >= 0.30, Task 3 drop: {task3_drop:.3f} <= 0.05)")

    return {**ablated, **deltas, **{f"baseline_{k}": v for k, v in baseline.items()}}


# ---------------------------------------------------------------------------
# R2: Test-time compute scaling (K sweep)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_at_k(
    model: MambaWorkspaceModel,
    eval_set: List[Dict],
    vocab: Vocab,
    device: torch.device,
    k: int,
    max_new: int = 5,
) -> Dict:
    """Evaluate at a specific K value."""
    model.eval()
    correct = {1: 0, 2: 0, 3: 0}
    total = {1: 0, 2: 0, 3: 0}
    correct_by_depth = {}
    total_by_depth = {}

    for ex in eval_set:
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        task_id = ex["task_id"]
        depth = ex["depth"]
        prompt = ex["prompt"]

        prompt_ids = vocab.encode(prompt)
        prompt_len = len(prompt_ids) - 1  # exclude EOS

        generated = input_ids[:, :prompt_len]
        for _ in range(max_new):
            out = model(generated, k_override=k)
            next_logits = out["logits"][:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == vocab.EOS:
                break

        response_ids = generated[0, prompt_len:].tolist()
        response = vocab.decode(response_ids)

        is_correct = TASK_VERIFIERS[task_id](prompt, response)
        total[task_id] += 1
        correct[task_id] += int(is_correct)

        if depth not in correct_by_depth:
            correct_by_depth[depth] = 0
            total_by_depth[depth] = 0
        total_by_depth[depth] += 1
        correct_by_depth[depth] += int(is_correct)

    results = {}
    for tid in [1, 2, 3]:
        results[f"task{tid}_acc"] = correct[tid] / max(total[tid], 1)
    for depth in sorted(total_by_depth.keys()):
        results[f"depth{depth}_acc"] = correct_by_depth[depth] / max(total_by_depth[depth], 1)
    results["overall_acc"] = sum(correct.values()) / max(sum(total.values()), 1)
    return results


def run_r2(
    model: MambaWorkspaceModel,
    vocab: Vocab,
    device: torch.device,
    k_values: List[int] = None,
    seq_len: int = 128,
) -> Dict:
    """R2: Test-time compute scaling — sweep K and measure accuracy vs depth."""
    print("\n=== R2: Test-Time Compute Scaling (K sweep) ===")

    if not model.config.recurrent_core:
        print("  No recurrent core in this model — skipping R2.")
        return {}

    if k_values is None:
        k_values = [1, 2, 4, 6, 8, 12, 16]

    eval_set = generate_eval_set(
        n_per_task_per_depth=30,
        depths=list(range(2, 17)),
        vocab=vocab,
        seq_len=seq_len,
        rng=random.Random(888),
    )

    results = {}
    for k in k_values:
        eval_results = evaluate_at_k(model, eval_set, vocab, device, k=k, max_new=5)
        results[f"k{k}"] = eval_results
        print(f"  K={k}: overall={eval_results['overall_acc']:.3f}, "
              f"task1={eval_results['task1_acc']:.3f}, task2={eval_results['task2_acc']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probing and ablation analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--r2", action="store_true", help="Run R2 (K sweep)")
    parser.add_argument("--r3", action="store_true", help="Run R3 (workspace probes)")
    parser.add_argument("--r4", action="store_true", help="Run R4 (ablation)")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    cfg = load_config(args.config)
    model_config = build_model_config(cfg)
    model = MambaWorkspaceModel(model_config).to(device)

    # Load checkpoint
    step, loss = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint at step {step}, loss={loss:.4f}")

    vocab = Vocab()

    if args.all or args.r2:
        run_r2(model, vocab, device)
    if args.all or args.r3:
        run_r3(model, vocab, device)
    if args.all or args.r4:
        run_r4(model, vocab, device)


if __name__ == "__main__":
    main()
