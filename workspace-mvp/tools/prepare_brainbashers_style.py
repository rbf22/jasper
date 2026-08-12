#!/usr/bin/env python3
"""Generate BrainBashers-style logic-grid puzzles for the Jasper text-MVP.

The goal is to force the model to use its workspace / recurrent thinking space
by interleaving several true facts and then asking a question whose answer
requires combining 2-4 of them in a chain.

Styles:
  1. Attribute chain: person -> color -> pet -> food/place
  2. Place & object: person -> place -> object/attribute
  3. Position chain: position -> person -> color/pet
  4. Elimination: negative + positive clues over a small attribute grid

Each puzzle is one line: clues ... question answer.
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

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "brown"]
PETS = ["dog", "cat", "bird", "fish", "rabbit", "mouse", "frog", "turtle", "hamster", "snake"]
FOODS = ["apples", "cookies", "carrots", "grapes", "bananas", "pears", "muffins", "crackers"]
PLACES = ["kitchen", "garden", "bedroom", "park", "school", "library", "office", "store"]
OBJECTS = ["ball", "car", "doll", "book", "box", "cup", "star", "key", "coin", "hat"]
POSITIONS = ["first", "second", "third", "fourth", "fifth"]


def assign(n, pool, rng):
    """Return a list of n unique values from pool, assigned by index."""
    return rng.sample(pool, n)


def chain(rng, templates, n=2):
    """Build a chain of clues from templates, each consuming the next."""
    return [rng.choice(templates) for _ in range(n)]


# ---------------------------------------------------------------------------
# Style 1: person -> color -> (optional) pet / food / place
# ---------------------------------------------------------------------------

def attribute_chain(rng):
    n = rng.randint(3, 4)
    people = assign(n, NAMES, rng)
    colors = assign(n, COLORS, rng)
    person_color = {p: c for p, c in zip(people, colors)}

    # Optional second attribute chain
    use_pet = rng.random() < 0.7
    if use_pet:
        pets = assign(n, PETS, rng)
        person_pet = {p: pt for p, pt in zip(people, pets)}
        color_pet = {c: person_pet[p] for p, c in person_color.items()}
        more = rng.random() < 0.5
        if more:
            foods = assign(n, FOODS, rng)
            person_food = {p: f for p, f in zip(people, foods)}
            pet_food = {pt: person_food[p] for p, pt in person_pet.items()}
            # person -> color -> pet -> food (4-hop)
            target = rng.choice(people)
            t_color = person_color[target]
            t_pet = person_pet[target]
            t_food = person_food[target]
            clues = [
                f"{target} is {t_color}.",
                f"The {t_color} person has a {t_pet}.",
                f"The {t_pet} owner eats {t_food}.",
            ]
            question = f"What does {target} eat?"
            answer = f"{target} eats {t_food}."
        else:
            # person -> color -> pet (3-hop)
            target = rng.choice(people)
            t_color = person_color[target]
            t_pet = person_pet[target]
            clues = [
                f"{target} is {t_color}.",
                f"The {t_color} person has a {t_pet}.",
            ]
            question = f"What pet does {target} have?"
            answer = f"{target} has a {t_pet}."
    else:
        # person <-> place
        places = assign(n, PLACES, rng)
        person_place = {p: pl for p, pl in zip(people, places)}
        color_place = {c: person_place[p] for p, c in person_color.items()}
        more = rng.random() < 0.5
        if more:
            objects = assign(n, OBJECTS, rng)
            person_object = {p: o for p, o in zip(people, objects)}
            place_object = {pl: person_object[p] for p, pl in person_place.items()}
            # person -> color -> place -> object (4-hop)
            target = rng.choice(people)
            t_color = person_color[target]
            t_place = person_place[target]
            t_object = person_object[target]
            clues = [
                f"{target} is {t_color}.",
                f"The {t_color} person lives in the {t_place}.",
                f"The {t_place} has a {t_object}.",
            ]
            question = f"What does {target} have?"
            answer = f"{target} has a {t_object}."
        else:
            # person -> color -> place (3-hop)
            target = rng.choice(people)
            t_color = person_color[target]
            t_place = person_place[target]
            clues = [
                f"{target} is {t_color}.",
                f"The {t_color} person lives in the {t_place}.",
            ]
            question = f"Where does {target} live?"
            answer = f"{target} lives in the {t_place}."

    # Add a few true distractors
    distractors = []
    for _ in range(rng.randint(0, 2)):
        p = rng.choice(people)
        c = person_color[p]
        distractors.append(f"{p} is {c}.")
    rng.shuffle(clues + distractors)
    parts = (clues + distractors)
    rng.shuffle(parts)
    return " ".join(parts) + " " + question + " " + answer


# ---------------------------------------------------------------------------
# Style 2: place -> object & person
# ---------------------------------------------------------------------------

def place_object(rng):
    n = rng.randint(3, 4)
    people = assign(n, NAMES, rng)
    places = assign(n, PLACES, rng)
    objects = assign(n, OBJECTS, rng)
    person_place = {p: pl for p, pl in zip(people, places)}
    place_object = {pl: o for pl, o in zip(places, objects)}
    person_object = {p: place_object[person_place[p]] for p in people}

    # Choose a 3-hop: person -> place -> object
    target = rng.choice(people)
    t_place = person_place[target]
    t_object = person_object[target]
    clues = [
        f"{target} is in the {t_place}.",
        f"The {t_place} contains a {t_object}.",
    ]
    question = f"What does {target} have?"
    answer = f"{target} has a {t_object}."

    # Add a few true distractors
    distractors = []
    for _ in range(rng.randint(0, 2)):
        p = rng.choice(people)
        distractors.append(f"{p} is in the {person_place[p]}.")
    parts = clues + distractors
    rng.shuffle(parts)
    return " ".join(parts) + " " + question + " " + answer


# ---------------------------------------------------------------------------
# Style 3: position chain
# ---------------------------------------------------------------------------

def position_chain(rng):
    n = rng.randint(3, 4)
    people = assign(n, NAMES, rng)
    positions = POSITIONS[:n]
    colors = assign(n, COLORS, rng)
    pets = assign(n, PETS, rng)
    person_pos = {p: pos for p, pos in zip(people, positions)}
    pos_color = {pos: c for pos, c in zip(positions, colors)}
    person_color = {p: pos_color[person_pos[p]] for p in people}
    pos_pet = {pos: pt for pos, pt in zip(positions, pets)}
    person_pet = {p: pos_pet[person_pos[p]] for p in people}

    target = rng.choice(people)
    t_pos = person_pos[target]
    t_color = person_color[target]
    t_pet = person_pet[target]
    clues = [
        f"{target} is in {t_pos} place.",
        f"The {t_pos} place is {t_color}.",
        f"The {t_color} position has a {t_pet}.",
    ]
    question = f"What pet does {target} have?"
    answer = f"{target} has a {t_pet}."

    # Add distractors
    for _ in range(rng.randint(0, 2)):
        p = rng.choice(people)
        distractor = f"{p} is in {person_pos[p]} place."
        clues.append(distractor)
    rng.shuffle(clues)
    return " ".join(clues) + " " + question + " " + answer


# ---------------------------------------------------------------------------
# Style 4: simple elimination with negatives
# ---------------------------------------------------------------------------

def elimination(rng):
    n = 3
    people = assign(n, NAMES, rng)
    colors = assign(n, COLORS, rng)
    person_color = {p: c for p, c in zip(people, colors)}

    target = rng.choice(people)
    t_color = person_color[target]
    # other two people and their colors
    others = [p for p in people if p != target]
    o1, o2 = others

    clues = [
        f"{o1} is {person_color[o1]}.",
        f"{target} is not {person_color[o1]}.",
        f"{o2} is not {t_color}.",
    ]
    # Optionally add a 4th distractor
    if rng.random() < 0.5:
        clues.append(f"{target} is not {person_color[o2]}.")

    rng.shuffle(clues)
    question = f"What color is {target}?"
    answer = f"{target} is {t_color}."
    return " ".join(clues) + " " + question + " " + answer


# ---------------------------------------------------------------------------
# Dispatcher and writer
# ---------------------------------------------------------------------------

GENERATORS = [attribute_chain, place_object, position_chain, elimination]


def generate_puzzle(rng):
    return rng.choice(GENERATORS)(rng)


def write_puzzles(n, out_path, seed):
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(generate_puzzle(rng))
            f.write("\n")
    return n


def main():
    parser = argparse.ArgumentParser(description="Prepare BrainBashers-style logic puzzles for Jasper")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-valid", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = args.out_dir / "brainbashers_train.txt"
    valid_path = args.out_dir / "brainbashers_valid.txt"

    print("Generating training puzzles...", flush=True)
    write_puzzles(args.n_train, train_path, seed=args.seed)
    print(f"  Wrote {args.n_train:,} puzzles to {train_path}", flush=True)

    print("Generating validation puzzles...", flush=True)
    write_puzzles(args.n_valid, valid_path, seed=args.seed + 9999)
    print(f"  Wrote {args.n_valid:,} puzzles to {valid_path}", flush=True)

    print("\n--- Sample puzzles ---")
    rng = random.Random(args.seed + 12345)
    for i in range(5):
        print(f"\nPuzzle {i+1}: {generate_puzzle(rng)}")

    # Token stats
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    toks = 0
    n = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            toks += len(enc.encode(line.strip(), disallowed_special=())) + 1
            n += 1
    avg = toks / n
    print(f"\nAvg tokens per puzzle (sample): {avg:.1f}")
    print(f"Estimated train tokens: {int(avg * args.n_train):,}")

    stats = {
        "n_train": args.n_train,
        "n_valid": args.n_valid,
        "seed": args.seed,
        "styles": ["attribute_chain", "place_object", "position_chain", "elimination"],
        "avg_tokens_per_puzzle": avg,
        "estimated_train_tokens": int(avg * args.n_train),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    with open(args.out_dir / "brainbashers_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n--- Summary ---")
    print(json.dumps(stats, indent=2))
    print(
        "\nNext: use configs/text_cell_c_brainbashers.yaml with train_text.py. "
        "The .tokens.pt cache will be built on first TextDataset load.",
        flush=True,
    )


if __name__ == "__main__":
    main()
