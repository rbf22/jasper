#!/usr/bin/env python3
"""Download the TinyStories dataset and format it for the WRAP text-MVP.

Output:
    jasper/workspace-mvp/data/tinystories_train.txt
    jasper/workspace-mvp/data/tinystories_valid.txt

Each line is one story. The WRAP TextDataset will tokenize these files
with GPT-2 BPE and cache them as .tokens.pt on first use.
"""

import os
import json
import argparse
from pathlib import Path

import tiktoken
from datasets import load_dataset


def find_text_field(example):
    """Return the field that holds the story text."""
    for key in ("story", "text", "content", "stories", "sentence"):
        if key in example and isinstance(example[key], str):
            return key
    # Fallback: first string field
    for key, value in example.items():
        if isinstance(value, str):
            return key
    raise ValueError(f"No string field found in example: {list(example.keys())}")


def normalize_story(text):
    """Collapse internal newlines so each story fits on a single line."""
    return " ".join(text.split())


def write_split(split_data, out_path, text_field):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in split_data:
            story = normalize_story(example[text_field])
            if not story:
                continue
            f.write(story)
            f.write("\n")
            total_chars += len(story)
            n += 1
    return n, total_chars


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories for WRAP")
    parser.add_argument(
        "--dataset",
        type=str,
        default="roneneldan/TinyStories",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Where to write the .txt files",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="If set, limit the number of training stories",
    )
    parser.add_argument(
        "--max-valid",
        type=int,
        default=None,
        help="If set, limit the number of validation stories",
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset}...", flush=True)
    ds = load_dataset(args.dataset)
    print("Available splits:", list(ds.keys()), flush=True)

    train = ds["train"]
    valid = ds.get("validation")
    if valid is None:
        # Use test as validation if present; otherwise take a small slice
        valid = ds.get("test")
    if valid is None:
        train_valid = train.train_test_split(test_size=0.001, seed=42)
        train = train_valid["train"]
        valid = train_valid["test"]

    if args.max_train:
        train = train.select(range(min(args.max_train, len(train))))
    if args.max_valid:
        valid = valid.select(range(min(args.max_valid, len(valid))))

    text_field = find_text_field(train[0])
    print(f"Text field: {text_field!r}", flush=True)

    print("\n--- Sample stories ---", flush=True)
    for i in range(min(3, len(train))):
        sample = train[i][text_field].replace("\n", " ")
        print(f"\nExample {i+1}:")
        print(sample[:300] + ("..." if len(sample) > 300 else ""), flush=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"\nWriting validation stories...", flush=True)
    valid_path = args.out_dir / "tinystories_valid.txt"
    n_valid, chars_valid = write_split(valid, valid_path, text_field)
    print(f"  {n_valid:,} stories, {chars_valid:,} chars", flush=True)

    print(f"\nWriting training stories...", flush=True)
    train_path = args.out_dir / "tinystories_train.txt"
    n_train, chars_train = write_split(train, train_path, text_field)
    print(f"  {n_train:,} stories, {chars_train:,} chars", flush=True)

    print("\nEstimating token counts on small samples...", flush=True)
    sample_size = 2000
    sample_valid = valid.select(range(min(sample_size, len(valid))))
    sample_train = train.select(range(min(sample_size, len(train))))
    valid_tokens = sum(
        len(tokenizer.encode(ex[text_field])) + 1  # +1 for EOS
        for ex in sample_valid
    )
    train_tokens = sum(
        len(tokenizer.encode(ex[text_field])) + 1
        for ex in sample_train
    )
    estimated_valid_tokens = int(valid_tokens / len(sample_valid) * n_valid)
    estimated_train_tokens = int(train_tokens / len(sample_train) * n_train)

    stats = {
        "dataset": args.dataset,
        "text_field": text_field,
        "train_stories": n_train,
        "train_chars": chars_train,
        "valid_stories": n_valid,
        "valid_chars": chars_valid,
        "estimated_train_tokens": estimated_train_tokens,
        "estimated_valid_tokens": estimated_valid_tokens,
        "tokenizer": "gpt2",
        "files": [str(train_path), str(valid_path)],
    }

    stats_path = args.out_dir / "tinystories_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n--- Summary ---", flush=True)
    print(json.dumps(stats, indent=2), flush=True)

    print(f"\nStats written to {stats_path}", flush=True)
    print(
        "\nNext: on the training host, run train_text.py with "
        "configs/text_cell_c.yaml. The .tokens.pt cache will be built "
        "automatically on first TextDataset load.",
        flush=True,
    )


if __name__ == "__main__":
    main()
