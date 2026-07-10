"""
Data generators, verifiers, and vocabulary for the Mamba + Workspace POC.

Three synthetic tasks with controllable depth:
  Task 1: Chained assignment arithmetic (multi-hop composition) — mod 97
  Task 2: Permutation tracking (SSM stress test)
  Task 3: Single-hop recall (control)

All tasks are character-level, generated on-the-fly, and verifiable.
"""

import random
import string
import torch
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """Character-level vocabulary with ~50 tokens (padded to 128)."""

    PAD = 0
    BOS = 1
    EOS = 2

    CHARS = list(string.ascii_lowercase) + list(string.digits) + list("+-*=;,:?()\n")
    VOCAB_SIZE = 128

    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        for i, ch in enumerate(self.CHARS, start=3):
            self.stoi[ch] = i
            self.itos[i] = ch
        self.stoi["<pad>"] = self.PAD
        self.stoi["<bos>"] = self.BOS
        self.stoi["<eos>"] = self.EOS
        self.itos[self.PAD] = "<pad>"
        self.itos[self.BOS] = "<bos>"
        self.itos[self.EOS] = "<eos>"

    def encode(self, text: str) -> List[int]:
        ids = [self.BOS]
        for ch in text:
            ids.append(self.stoi[ch])
        ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int]) -> str:
        chars = []
        for i in ids:
            if i == self.EOS:
                break
            if i in (self.PAD, self.BOS):
                continue
            chars.append(self.itos.get(i, "?"))
        return "".join(chars)

    def __len__(self):
        return self.VOCAB_SIZE


# ---------------------------------------------------------------------------
# Task 1: Chained assignment arithmetic (mod 97)
# ---------------------------------------------------------------------------

MOD = 97
VAR_NAMES = list(string.ascii_lowercase[:16])  # a through p


def gen_task1(depth: int, rng: random.Random) -> Tuple[str, str, int]:
    """Generate a chained assignment problem.

    Returns (prompt, answer_str, answer_int).
    Depth = chain length from queried variable back to a constant.
    """
    chain_len = depth
    n_distractors = rng.randint(1, 3)

    # Build the chain: each var depends on the previous one
    values = {}
    lines = []
    ops = ["+", "-", "*"]

    # First variable gets a random constant
    values["a"] = rng.randint(0, MOD - 1)
    lines.append(f"a={values['a']}")

    for i in range(1, chain_len):
        var = VAR_NAMES[i]
        prev = VAR_NAMES[i - 1]
        op = rng.choice(ops)
        operand = rng.randint(1, 96)
        if op == "+":
            values[var] = (values[prev] + operand) % MOD
        elif op == "-":
            values[var] = (values[prev] - operand) % MOD
        else:
            values[var] = (values[prev] * operand) % MOD
        lines.append(f"{var}={prev}{op}{operand}")

    # Insert distractor variables at random positions (not part of the chain)
    all_lines = list(lines)
    distractor_vars = VAR_NAMES[chain_len : chain_len + n_distractors]
    for dv in distractor_vars:
        dv_val = rng.randint(0, MOD - 1)
        dline = f"{dv}={dv_val}"
        pos = rng.randint(0, len(all_lines))
        all_lines.insert(pos, dline)

    # Query the last variable in the chain
    query_var = VAR_NAMES[chain_len - 1]
    answer = values[query_var]

    prompt = ";".join(all_lines) + f";?{query_var};"
    answer_str = str(answer)
    return prompt, answer_str, answer


def verify_task1(prompt: str, response: str) -> bool:
    """Verify a Task 1 response by evaluating the chain."""
    try:
        answer = int(response.strip())
        # Parse the prompt to find the queried variable and recompute
        parts = prompt.rstrip(";").split(";")
        query_part = [p for p in parts if p.startswith("?")][0]
        query_var = query_part[1]

        # Evaluate all assignments
        env = {}
        for part in parts:
            if part.startswith("?"):
                continue
            var, expr = part.split("=")
            val = eval_expr(expr, env)
            env[var] = val

        return env[query_var] == answer
    except Exception:
        return False


def eval_expr(expr: str, env: Dict[str, int]) -> int:
    """Evaluate a simple arithmetic expression mod 97."""
    # Replace variable references with their values
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i] in string.ascii_lowercase:
            tokens.append(str(env[expr[i]]))
        else:
            tokens.append(expr[i])
        i += 1
    expr_str = "".join(tokens)
    # Safe eval: only digits and +-*
    result = eval(expr_str)  # noqa: S307 — controlled input
    return result % MOD


# ---------------------------------------------------------------------------
# Task 2: Permutation tracking
# ---------------------------------------------------------------------------


def gen_task2(depth: int, rng: random.Random) -> Tuple[str, str, int]:
    """Generate a permutation tracking problem.

    n items, k=depth swap operations, query one item's final position.
    Returns (prompt, answer_str, answer_int).
    """
    n_items = rng.randint(6, 12)
    n_swaps = depth

    # Track positions: pos[item] = current position
    pos = list(range(n_items))
    # Also track: item_at[pos] = item
    item_at = list(range(n_items))

    swap_lines = []
    for _ in range(n_swaps):
        i, j = rng.sample(range(n_items), 2)
        # Swap items at positions i and j
        item_at[i], item_at[j] = item_at[j], item_at[i]
        swap_lines.append(f"{i},{j}")

    # Query a random item
    query_item = rng.randint(0, n_items - 1)
    # Find its current position
    answer = item_at.index(query_item)

    prompt = f"n={n_items};" + ";".join(swap_lines) + f";?{query_item};"
    answer_str = str(answer)
    return prompt, answer_str, answer


def verify_task2(prompt: str, response: str) -> bool:
    """Verify a Task 2 response by replaying the swaps."""
    try:
        answer = int(response.strip())
        parts = prompt.rstrip(";").split(";")
        n = int(parts[0].split("=")[1])

        item_at = list(range(n))
        query_item = None

        for part in parts[1:]:
            if part.startswith("?"):
                query_item = int(part[1:])
            elif "," in part:
                i, j = map(int, part.split(","))
                item_at[i], item_at[j] = item_at[j], item_at[i]

        if query_item is None:
            return False
        return item_at.index(query_item) == answer
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Task 3: Single-hop recall (control)
# ---------------------------------------------------------------------------


def gen_task3(depth: int, rng: random.Random) -> Tuple[str, str, int]:
    """Generate a single-hop recall problem with distractors.

    depth is ignored (always 1). Returns (prompt, answer_str, answer_int).
    """
    n_vars = rng.randint(4, 8)
    query_idx = rng.randint(0, n_vars - 1)

    lines = []
    values = {}
    for i in range(n_vars):
        var = VAR_NAMES[i]
        val = rng.randint(0, MOD - 1)
        values[var] = val
        lines.append(f"{var}={val}")

    query_var = VAR_NAMES[query_idx]
    answer = values[query_var]

    prompt = ";".join(lines) + f";?{query_var};"
    answer_str = str(answer)
    return prompt, answer_str, answer


def verify_task3(prompt: str, response: str) -> bool:
    """Verify a Task 3 response."""
    try:
        answer = int(response.strip())
        parts = prompt.rstrip(";").split(";")
        query_part = [p for p in parts if p.startswith("?")][0]
        query_var = query_part[1]

        env = {}
        for part in parts:
            if part.startswith("?"):
                continue
            var, val = part.split("=")
            env[var] = int(val)

        return env[query_var] == answer
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    1: gen_task1,
    2: gen_task2,
    3: gen_task3,
}

TASK_VERIFIERS = {
    1: verify_task1,
    2: verify_task2,
    3: verify_task3,
}

TASK_MIX = {1: 0.45, 2: 0.45, 3: 0.10}


def sample_batch(
    batch_size: int,
    seq_len: int,
    vocab: Vocab,
    depth_range: Tuple[int, int] = (2, 8),
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of mixed-task examples.

    Returns:
        input_ids: (B, T) token ids (padded)
        labels: (B, T) labels with -100 for non-answer positions
        task_ids: (B,) task type (1, 2, or 3)
    """
    if rng is None:
        rng = random.Random()

    input_ids = torch.full((batch_size, seq_len), vocab.PAD, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    task_ids = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # Sample task according to mix
        r = rng.random()
        cum = 0.0
        task_id = 1
        for tid, prob in TASK_MIX.items():
            cum += prob
            if r < cum:
                task_id = tid
                break

        task_ids[b] = task_id
        depth = rng.randint(*depth_range)
        prompt, answer_str, _ = TASK_GENERATORS[task_id](depth, rng)

        # Full text: prompt + answer
        full_text = prompt + answer_str
        ids = vocab.encode(full_text)

        # Truncate or pad to seq_len
        if len(ids) > seq_len:
            ids = ids[:seq_len]

        # Find where the answer starts (after the last ';')
        prompt_ids = vocab.encode(prompt)
        answer_start = len(prompt_ids) - 1  # -1 for EOS we'll remove

        # Place in tensor
        input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        # Labels: only the answer tokens (shifted by 1 for next-token prediction)
        # At position answer_start-1 (last prompt char), model should predict first answer char
        for t in range(answer_start - 1, len(ids) - 1):
            labels[b, t] = ids[t + 1]

    return input_ids, labels, task_ids


def generate_eval_set(
    n_per_task_per_depth: int,
    depths: List[int],
    vocab: Vocab,
    seq_len: int,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """Generate a fixed evaluation set.

    Returns list of dicts with: prompt, answer, answer_str, task_id, depth, input_ids, labels.
    """
    if rng is None:
        rng = random.Random()

    examples = []
    for task_id in [1, 2, 3]:
        for depth in depths:
            for _ in range(n_per_task_per_depth):
                prompt, answer_str, answer = TASK_GENERATORS[task_id](depth, rng)
                full_text = prompt + answer_str
                ids = vocab.encode(full_text)
                if len(ids) > seq_len:
                    ids = ids[:seq_len]

                prompt_ids = vocab.encode(prompt)
                answer_start = len(prompt_ids) - 1

                input_ids = torch.full((seq_len,), vocab.PAD, dtype=torch.long)
                input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.long)

                labels = torch.full((seq_len,), -100, dtype=torch.long)
                for t in range(answer_start - 1, len(ids) - 1):
                    labels[t] = ids[t + 1]

                examples.append(
                    {
                        "prompt": prompt,
                        "answer": answer,
                        "answer_str": answer_str,
                        "task_id": task_id,
                        "depth": depth,
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                )

    return examples


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_task1_roundtrip():
    """Generate -> solve -> verify roundtrip for Task 1."""
    rng = random.Random(42)
    for depth in range(2, 9):
        for _ in range(100):
            prompt, answer_str, answer = gen_task1(depth, rng)
            assert verify_task1(prompt, answer_str), f"Task1 failed: {prompt} -> {answer_str}"
    print("Task 1 roundtrip: PASS")


def test_task2_roundtrip():
    """Generate -> solve -> verify roundtrip for Task 2."""
    rng = random.Random(42)
    for depth in range(2, 9):
        for _ in range(100):
            prompt, answer_str, answer = gen_task2(depth, rng)
            assert verify_task2(prompt, answer_str), f"Task2 failed: {prompt} -> {answer_str}"
    print("Task 2 roundtrip: PASS")


def test_task3_roundtrip():
    """Generate -> solve -> verify roundtrip for Task 3."""
    rng = random.Random(42)
    for _ in range(100):
        prompt, answer_str, answer = gen_task3(1, random.Random())
        assert verify_task3(prompt, answer_str), f"Task3 failed: {prompt} -> {answer_str}"
    print("Task 3 roundtrip: PASS")


def test_batch_generation():
    """Test that batch generation produces valid shapes."""
    vocab = Vocab()
    rng = random.Random(42)
    input_ids, labels, task_ids = sample_batch(8, 128, vocab, rng=rng)
    assert input_ids.shape == (8, 128)
    assert labels.shape == (8, 128)
    assert task_ids.shape == (8,)
    assert (task_ids >= 1).all() and (task_ids <= 3).all()
    # At least some labels should be non-negative
    assert (labels >= 0).any()
    print("Batch generation: PASS")


def test_depth_control():
    """Test that depth parameter actually controls problem difficulty."""
    rng = random.Random(42)
    for depth in [2, 4, 8]:
        prompt, _, _ = gen_task1(depth, rng)
        # Count the number of chained variables (non-distractor)
        # The chain should have exactly `depth` variables
        parts = prompt.rstrip(";").split(";")
        # Filter out the query part
        assignments = [p for p in parts if not p.startswith("?") and "=" in p]
        # At least `depth` assignments (chain vars + distractors)
        assert len(assignments) >= depth, f"Depth {depth}: too few assignments"
    print("Depth control: PASS")


def test_vocab():
    """Test vocabulary encode/decode roundtrip."""
    vocab = Vocab()
    text = "a=7;b=a*3+2;?b;23"
    ids = vocab.encode(text)
    decoded = vocab.decode(ids)
    assert decoded == text, f"Vocab roundtrip failed: {decoded} != {text}"
    print("Vocab roundtrip: PASS")


if __name__ == "__main__":
    test_vocab()
    test_task1_roundtrip()
    test_task2_roundtrip()
    test_task3_roundtrip()
    test_batch_generation()
    test_depth_control()
    print("\nAll tests passed!")
