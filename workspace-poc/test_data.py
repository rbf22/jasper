"""CPU-only pytest tests for data.py — synthetic task generators and verifiers.

These tests run without Tenstorrent hardware. They validate:
  - Vocab encode/decode roundtrip
  - Task 1 (chained assignment arithmetic mod 97) generation + verification
  - Task 2 (permutation tracking) generation + verification
  - Task 3 (single-hop recall) generation + verification
  - Batch generation shapes and label masking
  - Depth parameter controls problem difficulty
  - Determinism with seeded RNG
  - Edge cases (depth=1, wrong answers, malformed input)

Usage:
    .tt-venv/bin/python -m pytest test_data.py -v
    .tt-venv/bin/python -m pytest test_data.py -k task1 -v
"""
import random
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (
    Vocab, gen_task1, verify_task1, gen_task2, verify_task2,
    gen_task3, verify_task3, eval_expr, sample_batch, generate_eval_set,
    TASK_GENERATORS, TASK_VERIFIERS, TASK_MIX, MOD, VAR_NAMES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vocab():
    return Vocab()


@pytest.fixture
def rng():
    return random.Random(42)


# ---------------------------------------------------------------------------
# Vocab tests
# ---------------------------------------------------------------------------

class TestVocab:
    """Tests for the character-level vocabulary."""

    def test_encode_decode_roundtrip(self, vocab):
        """@Pure: encode then decode returns original text."""
        text = "a=7;b=a*3+2;?b;23"
        ids = vocab.encode(text)
        decoded = vocab.decode(ids)
        assert decoded == text

    def test_encode_adds_bos_eos(self, vocab):
        """@Pure: encode wraps text with BOS and EOS tokens."""
        ids = vocab.encode("a")
        assert ids[0] == Vocab.BOS
        assert ids[-1] == Vocab.EOS
        assert ids[1] == vocab.stoi["a"]

    def test_decode_stops_at_eos(self, vocab):
        """@Pure: decode stops at EOS, ignores trailing tokens."""
        ids = vocab.encode("a") + [vocab.stoi["z"], vocab.stoi["z"]]
        decoded = vocab.decode(ids)
        assert decoded == "a"

    def test_decode_skips_pad_and_bos(self, vocab):
        """@Pure: decode skips PAD and BOS in the middle of the sequence."""
        ids = [Vocab.PAD, Vocab.BOS, vocab.stoi["a"], Vocab.PAD, Vocab.EOS]
        decoded = vocab.decode(ids)
        assert decoded == "a"

    def test_vocab_size_is_128(self, vocab):
        """@Pure: VOCAB_SIZE is padded to 128."""
        assert vocab.VOCAB_SIZE == 128
        assert len(vocab) == 128

    def test_all_chars_encodable(self, vocab):
        """@Pure: every character in CHARS can be encoded."""
        for ch in Vocab.CHARS:
            ids = vocab.encode(ch)
            assert ids[1] == vocab.stoi[ch], f"Cannot encode char: {ch!r}"

    def test_encode_unknown_char_raises(self, vocab):
        """@Pure: encoding a char not in the vocabulary raises KeyError."""
        with pytest.raises(KeyError):
            vocab.encode("@")

    def test_decode_unknown_id_returns_question(self, vocab):
        """@Pure: decoding an unknown ID returns '?'."""
        decoded = vocab.decode([99])
        assert "?" in decoded or decoded == ""


# ---------------------------------------------------------------------------
# Task 1: Chained assignment arithmetic (mod 97)
# ---------------------------------------------------------------------------

class TestTask1:
    """Tests for chained assignment arithmetic (mod 97)."""

    def test_roundtrip_all_depths(self, rng):
        """@Idempotent: gen_task1 -> verify_task1 for depths 2-8, 100 iters each."""
        for depth in range(2, 9):
            for _ in range(100):
                prompt, answer_str, answer = gen_task1(depth, rng)
                assert verify_task1(prompt, answer_str), \
                    f"Task1 failed at depth {depth}: {prompt} -> {answer_str}"

    def test_answer_in_range(self, rng):
        """@Pure: answer is always in [0, 96] (mod 97)."""
        for _ in range(200):
            _, _, answer = gen_task1(5, rng)
            assert 0 <= answer < MOD

    def test_depth_controls_chain_length(self, rng):
        """@Pure: depth parameter controls the number of chained variables."""
        for depth in [2, 4, 8]:
            prompt, _, _ = gen_task1(depth, rng)
            parts = prompt.rstrip(";").split(";")
            assignments = [p for p in parts if not p.startswith("?") and "=" in p]
            # chain vars + 1-3 distractors
            assert len(assignments) >= depth
            assert len(assignments) <= depth + 3

    def test_deterministic_with_seed(self):
        """@Pure: same seed produces same problem."""
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        p1, a1, _ = gen_task1(5, rng1)
        p2, a2, _ = gen_task1(5, rng2)
        assert p1 == p2
        assert a1 == a2

    def test_wrong_answer_rejected(self, rng):
        """@Pure: verify_task1 rejects incorrect answers."""
        prompt, _, answer = gen_task1(4, rng)
        wrong = (answer + 1) % MOD
        assert not verify_task1(prompt, str(wrong))

    def test_malformed_prompt_rejected(self):
        """@Pure: verify_task1 returns False for malformed prompts."""
        assert not verify_task1("garbage", "42")
        assert not verify_task1("", "0")
        assert not verify_task1("a=5;?a;", "not_a_number")

    def test_eval_expr_simple(self):
        """@Pure: eval_expr handles simple expressions."""
        env = {"a": 5}
        assert eval_expr("a", env) == 5
        assert eval_expr("a+3", env) == 8
        assert eval_expr("a*2", env) == 10
        assert eval_expr("a-1", env) == 4

    def test_eval_expr_mod(self):
        """@Pure: eval_expr applies mod 97."""
        env = {"a": 96}
        assert eval_expr("a+2", env) == 1  # (96+2) % 97 = 1

    def test_chain_uses_first_var_a(self, rng):
        """@Pure: Task 1 chains always start with variable 'a'."""
        prompt, _, _ = gen_task1(3, rng)
        parts = prompt.rstrip(";").split(";")
        assignments = [p for p in parts if not p.startswith("?") and "=" in p]
        # First chain variable is always 'a' with a constant
        a_line = [p for p in assignments if p.startswith("a=")]
        assert len(a_line) == 1

    def test_query_is_last_chain_var(self, rng):
        """@Pure: Task 1 queries the last variable in the chain."""
        for depth in [2, 5, 8]:
            prompt, _, _ = gen_task1(depth, rng)
            parts = prompt.rstrip(";").split(";")
            query = [p for p in parts if p.startswith("?")][0]
            query_var = query[1]
            # Last chain var is VAR_NAMES[depth-1]
            assert query_var == VAR_NAMES[depth - 1]


# ---------------------------------------------------------------------------
# Task 2: Permutation tracking
# ---------------------------------------------------------------------------

class TestTask2:
    """Tests for permutation tracking."""

    def test_roundtrip_all_depths(self, rng):
        """@Idempotent: gen_task2 -> verify_task2 for depths 2-8, 100 iters each."""
        for depth in range(2, 9):
            for _ in range(100):
                prompt, answer_str, answer = gen_task2(depth, rng)
                assert verify_task2(prompt, answer_str), \
                    f"Task2 failed at depth {depth}: {prompt} -> {answer_str}"

    def test_answer_in_range(self, rng):
        """@Pure: answer (position) is in [0, n_items-1]."""
        for _ in range(200):
            prompt, _, answer = gen_task2(5, rng)
            parts = prompt.rstrip(";").split(";")
            n = int(parts[0].split("=")[1])
            assert 0 <= answer < n

    def test_deterministic_with_seed(self):
        """@Pure: same seed produces same problem."""
        rng1 = random.Random(456)
        rng2 = random.Random(456)
        p1, a1, _ = gen_task2(4, rng1)
        p2, a2, _ = gen_task2(4, rng2)
        assert p1 == p2
        assert a1 == a2

    def test_wrong_answer_rejected(self, rng):
        """@Pure: verify_task2 rejects incorrect answers."""
        prompt, _, answer = gen_task2(4, rng)
        parts = prompt.rstrip(";").split(";")
        n = int(parts[0].split("=")[1])
        wrong = (answer + 1) % n
        if wrong == answer:
            wrong = (answer + 2) % n
        assert not verify_task2(prompt, str(wrong))

    def test_malformed_prompt_rejected(self):
        """@Pure: verify_task2 returns False for malformed prompts."""
        assert not verify_task2("garbage", "42")
        assert not verify_task2("", "0")
        assert not verify_task2("n=5;?0;", "not_a_number")

    def test_no_swaps_identity(self, rng):
        """@Pure: depth=0 means no swaps, item stays at its original position."""
        # gen_task2 with depth=0: no swaps, query_item's position is itself
        prompt, _, answer = gen_task2(0, rng)
        parts = prompt.rstrip(";").split(";")
        query = [p for p in parts if p.startswith("?")][0]
        query_item = int(query[1:])
        assert answer == query_item


# ---------------------------------------------------------------------------
# Task 3: Single-hop recall
# ---------------------------------------------------------------------------

class TestTask3:
    """Tests for single-hop recall (control task)."""

    def test_roundtrip(self):
        """@Idempotent: gen_task3 -> verify_task3, 100 iters."""
        rng = random.Random(789)
        for _ in range(100):
            prompt, answer_str, answer = gen_task3(1, rng)
            assert verify_task3(prompt, answer_str), \
                f"Task3 failed: {prompt} -> {answer_str}"

    def test_answer_in_range(self, rng):
        """@Pure: answer is always in [0, 96] (mod 97)."""
        for _ in range(100):
            _, _, answer = gen_task3(1, rng)
            assert 0 <= answer < MOD

    def test_depth_ignored(self, rng):
        """@Pure: depth parameter is ignored (always 1-hop)."""
        p1, _, a1 = gen_task3(1, rng)
        p2, _, a2 = gen_task3(99, rng)
        # Both should produce valid single-hop problems
        assert verify_task1(p1, str(a1)) or verify_task3(p1, str(a1))
        assert verify_task1(p2, str(a2)) or verify_task3(p2, str(a2))

    def test_wrong_answer_rejected(self, rng):
        """@Pure: verify_task3 rejects incorrect answers."""
        prompt, _, answer = gen_task3(1, rng)
        wrong = (answer + 1) % MOD
        assert not verify_task3(prompt, str(wrong))


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

class TestBatchGeneration:
    """Tests for sample_batch and generate_eval_set."""

    def test_batch_shapes(self, vocab, rng):
        """@Pure: sample_batch returns correct tensor shapes."""
        input_ids, labels, task_ids = sample_batch(8, 128, vocab, rng=rng)
        assert input_ids.shape == (8, 128)
        assert labels.shape == (8, 128)
        assert task_ids.shape == (8,)

    def test_task_ids_in_range(self, vocab, rng):
        """@Pure: task_ids are in [1, 3]."""
        _, _, task_ids = sample_batch(32, 128, vocab, rng=rng)
        assert (task_ids >= 1).all() and (task_ids <= 3).all()

    def test_labels_have_answer_positions(self, vocab, rng):
        """@Pure: at least some labels are non-negative (answer positions)."""
        _, labels, _ = sample_batch(16, 128, vocab, rng=rng)
        assert (labels >= 0).any(), "No answer positions in labels"

    def test_labels_use_ignore_index(self, vocab, rng):
        """@Pure: non-answer positions have label -100."""
        _, labels, _ = sample_batch(16, 128, vocab, rng=rng)
        assert (labels == -100).any(), "No ignore positions in labels"

    def test_input_ids_in_vocab_range(self, vocab, rng):
        """@Pure: all input_ids are in [0, vocab_size-1]."""
        input_ids, _, _ = sample_batch(16, 128, vocab, rng=rng)
        assert (input_ids >= 0).all() and (input_ids < vocab.VOCAB_SIZE).all()

    def test_batch_deterministic_with_seed(self, vocab):
        """@Pure: same seed produces same batch."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        ids1, labels1, _ = sample_batch(4, 128, vocab, rng=rng1)
        ids2, labels2, _ = sample_batch(4, 128, vocab, rng=rng2)
        assert torch.equal(ids1, ids2)
        assert torch.equal(labels1, labels2)

    def test_task_mix_distribution(self, vocab):
        """@Pure: task mix roughly follows configured distribution over many samples."""
        rng = random.Random(42)
        _, _, task_ids = sample_batch(1000, 128, vocab, rng=rng)
        counts = {t: (task_ids == t).sum().item() for t in [1, 2, 3]}
        # Task 1: 45%, Task 2: 45%, Task 3: 10%
        # With 1000 samples, allow ±10% tolerance
        assert 350 < counts[1] < 550, f"Task 1 count {counts[1]} outside expected range"
        assert 350 < counts[2] < 550, f"Task 2 count {counts[2]} outside expected range"
        assert 50 < counts[3] < 150, f"Task 3 count {counts[3]} outside expected range"

    def test_eval_set_structure(self, vocab, rng):
        """@Pure: generate_eval_set returns properly structured examples."""
        examples = generate_eval_set(5, [2, 4, 8], vocab, 128, rng=rng)
        assert len(examples) == 3 * 3 * 5  # 3 tasks * 3 depths * 5 per
        for ex in examples:
            assert "prompt" in ex
            assert "answer" in ex
            assert "answer_str" in ex
            assert "task_id" in ex
            assert "depth" in ex
            assert "input_ids" in ex
            assert "labels" in ex
            assert ex["input_ids"].shape == (128,)
            assert ex["labels"].shape == (128,)

    def test_eval_set_task_coverage(self, vocab, rng):
        """@Pure: eval set covers all tasks and depths."""
        examples = generate_eval_set(3, [2, 5, 8], vocab, 128, rng=rng)
        task_ids = {ex["task_id"] for ex in examples}
        depths = {ex["depth"] for ex in examples}
        assert task_ids == {1, 2, 3}
        assert depths == {2, 5, 8}


# ---------------------------------------------------------------------------
# Task registry tests
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    """Tests for the task generator/verifier registry."""

    def test_generators_match_verifiers(self):
        """@Pure: every task has both a generator and verifier."""
        assert set(TASK_GENERATORS.keys()) == set(TASK_VERIFIERS.keys())

    def test_task_mix_sums_to_one(self):
        """@Pure: task mix probabilities sum to 1.0."""
        assert abs(sum(TASK_MIX.values()) - 1.0) < 1e-6

    def test_task_mix_covers_all_tasks(self):
        """@Pure: task mix covers all registered tasks."""
        assert set(TASK_MIX.keys()) == set(TASK_GENERATORS.keys())
