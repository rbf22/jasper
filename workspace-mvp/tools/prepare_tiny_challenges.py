#!/usr/bin/env python3
"""Build a large, mixed "tiny challenges" dataset to replace TinyStories.

Instead of language stories, every training example is a small reasoning problem:
location tracking, attribute chains, elimination, ordering, and bAbI-style QA.
The model's job is to continue the prompt with the answer sentence.

Sources:
  - prepare_logic_puzzles.py (narrative multi-hop stories)
  - prepare_brainbashers_style.py (attribute chains / elimination)
  - babi_train.txt (passage-question-answer)
"""

import random
import json
import argparse
import importlib.util
from pathlib import Path

import tiktoken


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_lines(path, max_n=None):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_n is not None and i >= max_n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Prepare tiny-challenges corpus for Jasper")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--n-train", type=int, default=2_000_000)
    parser.add_argument("--n-valid", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading puzzle generators...", flush=True)
    logic_mod = load_module(Path(__file__).parent / "prepare_logic_puzzles.py")
    bb_mod = load_module(Path(__file__).parent / "prepare_brainbashers_style.py")

    # Load bAbI examples as a small, fixed in-distribution source
    babi_train = load_lines(args.out_dir / "babi_train.txt")
    babi_valid = load_lines(args.out_dir / "babi_valid.txt")
    print(f"  bAbI train: {len(babi_train):,}, valid: {len(babi_valid):,}", flush=True)

    train_path = args.out_dir / "tiny_challenges_train.txt"
    valid_path = args.out_dir / "tiny_challenges_valid.txt"

    enc = tiktoken.get_encoding("gpt2")

    def make_example(rng):
        choice = rng.choices(
            ["logic", "brainbashers", "babi"],
            weights=[40, 55, 5],  # mostly chain/elimination + narrative logic
            k=1,
        )[0]
        if choice == "logic":
            return logic_mod.generate_puzzle(rng)
        if choice == "brainbashers":
            return bb_mod.generate_puzzle(rng)
        return rng.choice(babi_train)

    def make_example_valid(rng):
        choice = rng.choices(
            ["logic", "brainbashers", "babi"],
            weights=[40, 55, 5],
            k=1,
        )[0]
        if choice == "logic":
            return logic_mod.generate_puzzle(rng)
        if choice == "brainbashers":
            return bb_mod.generate_puzzle(rng)
        return rng.choice(babi_valid)

    print("Generating training set...", flush=True)
    rng = random.Random(args.seed)
    n_tokens_est = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for i in range(args.n_train):
            ex = make_example(rng)
            f.write(ex)
            f.write("\n")
            if i < 2000:
                n_tokens_est += len(enc.encode(ex, disallowed_special=())) + 1

    print("Generating validation set...", flush=True)
    rng_valid = random.Random(args.seed + 9999)
    with open(valid_path, "w", encoding="utf-8") as f:
        for _ in range(args.n_valid):
            ex = make_example_valid(rng_valid)
            f.write(ex)
            f.write("\n")

    avg = n_tokens_est / min(args.n_train, 2000)
    estimated_train = int(avg * args.n_train)

    stats = {
        "n_train": args.n_train,
        "n_valid": args.n_valid,
        "seed": args.seed,
        "mix": {"logic": 0.40, "brainbashers": 0.55, "babi": 0.05},
        "avg_tokens_per_example": avg,
        "estimated_train_tokens": estimated_train,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    with open(args.out_dir / "tiny_challenges_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n--- Sample examples ---")
    rng = random.Random(args.seed + 12345)
    for i in range(5):
        print(f"\n{i+1}. {make_example(rng)}")

    print("\n--- Summary ---")
    print(json.dumps(stats, indent=2))
    print(
        "\nNext: use configs/text_cell_c_tiny_challenges.yaml with train_text.py. "
        "The .tokens.pt cache will be built on first TextDataset load.",
        flush=True,
    )


if __name__ == "__main__":
    main()
