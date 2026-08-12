#!/usr/bin/env python3
"""Download the bAbI QA dataset and reformat it for the Jasper text pipeline.

Each output line is one self-contained example: the passage sentences, the
question, and the answer, separated by spaces. Jasper's TextDataset will
pack these with EOS and train next-token prediction so the model learns to
output the answer after the question.

Source: HuggingFace `Muennighoff/babi` (English, all 20 tasks).
"""

import json
import argparse
from pathlib import Path

from datasets import load_dataset
import tiktoken


def normalize(text):
    return " ".join(text.split())


def write_split(split, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in split:
            passage = normalize(ex["passage"])
            question = normalize(ex["question"])
            answer = normalize(ex["answer"])
            line = f"{passage} {question} {answer}"
            f.write(line)
            f.write("\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Prepare bAbI for Jasper")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--dataset", type=str, default="Muennighoff/babi")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...", flush=True)
    ds = load_dataset(args.dataset)
    print("Available splits:", list(ds.keys()), flush=True)

    train_path = args.out_dir / "babi_train.txt"
    valid_path = args.out_dir / "babi_valid.txt"
    test_path = args.out_dir / "babi_test.txt"

    n_train = write_split(ds["train"], train_path)
    n_valid = write_split(ds["validation"], valid_path)
    n_test = write_split(ds["test"], test_path) if "test" in ds else 0

    print(f"  train: {n_train:,} examples", flush=True)
    print(f"  valid: {n_valid:,} examples", flush=True)
    if n_test:
        print(f"  test:  {n_test:,} examples", flush=True)

    # Token stats
    print("\nEstimating tokens...", flush=True)
    enc = tiktoken.get_encoding("gpt2")
    sample_size = min(2000, n_train)
    sample = ds["train"].select(range(sample_size))
    toks = sum(
        len(enc.encode(f"{normalize(ex['passage'])} {normalize(ex['question'])} {normalize(ex['answer'])}", disallowed_special=())) + 1
        for ex in sample
    )
    avg = toks / sample_size
    print(f"  avg tokens per example: {avg:.1f}")
    print(f"  estimated train tokens: {int(avg * n_train):,}")

    stats = {
        "dataset": args.dataset,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "avg_tokens_per_example": avg,
        "estimated_train_tokens": int(avg * n_train),
        "files": [str(train_path), str(valid_path)] + ([str(test_path)] if n_test else []),
    }
    with open(args.out_dir / "babi_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n--- Sample examples ---")
    for i in range(5):
        ex = ds["train"][i]
        print(f"\n{i+1}. {normalize(ex['passage'])} {normalize(ex['question'])} {normalize(ex['answer'])}")

    print("\n--- Summary ---")
    print(json.dumps(stats, indent=2))
    print(
        "\nNext: use configs/text_cell_c_babi.yaml with train_text.py. "
        "The .tokens.pt cache will be built on first TextDataset load.",
        flush=True,
    )


if __name__ == "__main__":
    main()
