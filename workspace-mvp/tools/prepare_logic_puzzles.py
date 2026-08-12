#!/usr/bin/env python3
"""Generate narrative multi-hop logic puzzles for the Jasper text-MVP.

Each line is one self-contained story that ends with a question and its
answer. The model is trained with standard next-token prediction, so it must
learn to keep the story state in the workspace and generate the correct final
sentence.

Puzzle types:
  - Person/location tracking
  - Object/location tracking with moves
  - Chained arithmetic in story form
  - Permutation/swap tracking in a row
  - Single-hop attribute recall (control)
"""

import random
import json
import argparse
from pathlib import Path


NAMES = [
    "Lily", "Ben", "Emma", "Max", "Ava", "Noah", "Mia", "Leo", "Zoe", "Jack",
    "Olivia", "Lucas", "Sophia", "Ethan", "Chloe", "Mason", "Aiden", "Grace",
    "Logan", "Zara", "Owen", "Nora", "Eli", "Rosa", "Sam", "Ivy", "Jake",
    "Pia", "Kai", "Mila", "Finn", "Tara", "Duke", "Yara", "Cole", "Wren",
]

PLACES = [
    "kitchen", "garden", "bedroom", "park", "school", "library",
    "playground", "zoo", "beach", "forest", "store", "room",
]

OBJECTS = [
    "red ball", "blue car", "green doll", "yellow boat", "purple block",
    "orange cat", "pink hat", "black shoe", "white star", "brown box",
    "gray cup", "silver key", "gold coin", "tiny bird", "soft bear",
]

ANIMALS = [
    "cat", "dog", "fish", "bird", "frog", "rabbit", "duck", "mouse",
    "monkey", "turtle", "panda", "lion", "bear", "horse", "sheep",
]

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "brown"]

ACTIONS = ["ran", "jumped", "swam", "flew", "walked", "climbed"]


# ---------------------------------------------------------------------------
# Puzzle generators
# ---------------------------------------------------------------------------

def person_location(rng):
    """Track people moving between places."""
    n = rng.randint(2, 4)
    people = rng.sample(NAMES, n)
    places = rng.sample(PLACES, n)
    state = {p: pl for p, pl in zip(people, places)}
    parts = [f"{p} is in the {pl}." for p, pl in zip(people, places)]

    # Apply random moves
    n_moves = rng.randint(1, min(4, n))
    movers = []
    for _ in range(n_moves):
        p = rng.choice(people)
        new_place = rng.choice(PLACES)
        # Avoid no-op and avoid two consecutive identical moves
        while new_place == state[p]:
            new_place = rng.choice(PLACES)
        old = state[p]
        state[p] = new_place
        if p not in movers:
            movers.append(p)
        parts.append(f"{p} went from the {old} to the {new_place}.")

    asked = rng.choice(people)
    answer_place = state[asked]
    parts.append(f"Where is {asked}? {asked} is in the {answer_place}.")
    return " ".join(parts)


def object_location(rng):
    """Track objects being moved between containers/places."""
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    # A second distractor object and place
    obj2 = rng.choice(OBJECTS)
    while obj2 == obj:
        obj2 = rng.choice(OBJECTS)
    place2 = rng.choice(PLACES)
    while place2 == place:
        place2 = rng.choice(PLACES)

    state = {obj: place, obj2: place2}
    parts = [
        f"The {obj} is in the {place}.",
        f"The {obj2} is in the {place2}.",
    ]

    n_moves = rng.randint(1, 3)
    for _ in range(n_moves):
        target = rng.choice([obj, obj2])
        new_place = rng.choice(PLACES)
        while new_place == state[target]:
            new_place = rng.choice(PLACES)
        state[target] = new_place
        parts.append(f"Sue moved the {target} to the {new_place}.")

    asked = obj if rng.random() < 0.5 else obj2
    answer_place = state[asked]
    parts.append(f"Where is the {asked}? The {asked} is in the {answer_place}.")
    return " ".join(parts)


def arithmetic_chain(rng):
    """Narrative chain of arithmetic operations."""
    name = rng.choice(NAMES)
    # Keep all numbers >= 2 so the plural 'cookies/toys' is always grammatical.
    start = rng.randint(4, 9)
    a = rng.randint(2, 6)
    b = rng.randint(2, min(start + a - 2, 5))
    c = rng.randint(2, min(start + a - b, 5))
    total = start + a - b - c

    item = rng.choice(["apples", "cookies", "marbles", "stickers", "toys"])
    parts = [
        f"{name} had {start} {item}.",
        f"{name} found {a} more {item}.",
        f"{name} gave {b} {item} to a friend.",
        f"{name} ate {c} {item}.",
        f"How many {item} does {name} have? {name} has {total} {item}.",
    ]
    return " ".join(parts)


def swap_tracking(rng):
    """Track a sequence of swaps in a row of toys/animals."""
    n = rng.randint(3, 6)
    items = rng.sample(ANIMALS, n)
    positions = list(range(n))
    item_to_pos = {item: i for i, item in enumerate(items)}

    parts = [f"There are {n} animals in a row: {', '.join(items)}."]

    n_swaps = rng.randint(1, min(4, n))
    for _ in range(n_swaps):
        i, j = rng.sample(range(n), 2)
        # Update mapping
        item_i = items[i]
        item_j = items[j]
        positions[i], positions[j] = positions[j], positions[i]
        items[i], items[j] = items[j], items[i]
        parts.append(f"The {item_i} and the {item_j} swapped places.")

    # Ask for a position or an item's place
    if rng.random() < 0.5:
        # Which animal is at position k? (1-indexed)
        k = rng.randint(1, n)
        answer = items[k - 1]
        parts.append(f"Which animal is in position {k}? The {answer} is in position {k}.")
    else:
        # Where is item x?
        answer = rng.choice(items)
        k = items.index(answer) + 1
        parts.append(f"Where is the {answer}? The {answer} is in position {k}.")

    return " ".join(parts)


def attribute_recall(rng):
    """Single-hop distractor recall, mirroring the control task in data.py."""
    animal = rng.choice(ANIMALS)
    color = rng.choice(COLORS)
    action = rng.choice(ACTIONS)
    # Distractor
    animal2 = rng.choice(ANIMALS)
    while animal2 == animal:
        animal2 = rng.choice(ANIMALS)
    color2 = rng.choice(COLORS)
    while color2 == color:
        color2 = rng.choice(COLORS)

    if rng.random() < 0.5:
        parts = [
            f"The {animal} is {color} and {action} fast.",
            f"The {animal2} is {color2} and {action} slow.",
            f"What color is the {animal}? The {animal} is {color}.",
        ]
    else:
        parts = [
            f"The {animal} is {color} and {action} fast.",
            f"The {animal2} is {color2} and {action} slow.",
            f"Which animal {action} fast? The {animal} {action} fast.",
        ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def generate_puzzle(rng):
    generator = rng.choices(
        [person_location, object_location, arithmetic_chain, swap_tracking, attribute_recall],
        weights=[25, 25, 20, 20, 10],
        k=1,
    )[0]
    return generator(rng)


def write_puzzles(n, out_path, seed):
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tokens_est = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            puzzle = generate_puzzle(rng)
            f.write(puzzle)
            f.write("\n")
            if i < 1000:
                # rough token estimate for sanity
                n_tokens_est += len(puzzle.split()) + 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Generate logic puzzle text data for Jasper")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--n-train", type=int, default=500_000)
    parser.add_argument("--n-valid", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = args.out_dir / "logic_puzzles_train.txt"
    valid_path = args.out_dir / "logic_puzzles_valid.txt"

    print("Generating training puzzles...", flush=True)
    write_puzzles(args.n_train, train_path, seed=args.seed)
    print(f"  Wrote {args.n_train:,} puzzles to {train_path}", flush=True)

    print("Generating validation puzzles...", flush=True)
    write_puzzles(args.n_valid, valid_path, seed=args.seed + 9999)
    print(f"  Wrote {args.n_valid:,} puzzles to {valid_path}", flush=True)

    # Show a few samples
    print("\n--- Sample puzzles ---")
    rng = random.Random(args.seed + 12345)
    for i in range(5):
        print(f"\nPuzzle {i+1}: {generate_puzzle(rng)}")

    stats = {
        "n_train": args.n_train,
        "n_valid": args.n_valid,
        "seed": args.seed,
        "types": ["person_location", "object_location", "arithmetic_chain", "swap_tracking", "attribute_recall"],
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    with open(args.out_dir / "logic_puzzle_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\n--- Summary ---")
    print(json.dumps(stats, indent=2))
    print(
        "\nNext: use the logic-puzzle YAML config with train_text.py. "
        "The .tokens.pt cache will be built on first TextDataset load.",
        flush=True,
    )


if __name__ == "__main__":
    main()
