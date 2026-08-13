"""Text data pipeline for the WRAP MVP — TinyStories with GPT-2 BPE tokenization.

Provides the same interface as workspace-poc/data.py's sample_batch() so the
training loop in train_text.py can use it as a drop-in replacement:

    input_ids, labels, task_ids = sample_text_batch(...)

Key differences from the synthetic data pipeline:
  - Real text (TinyStories) instead of generated arithmetic
  - BPE tokenization (tiktoken/GPT-2, 50257 vocab) instead of char-level (128)
  - Labels at every position (standard LM training) instead of answer-only
  - No task_ids (all examples are "text", task_ids=0)
  - Sequences packed from a token stream (no per-example padding needed
    since we concatenate documents and slice into fixed-length chunks)

The dataset is pre-tokenized once at startup and held in memory as a
1D token tensor. Batches are sliced from random offsets — this is the
standard "packed" approach used by most LM training loops.
"""

import os
import torch
import tiktoken
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """GPT-2 BPE tokenizer wrapper with the same interface as data.Vocab."""

    PAD = 0  # We'll use token 0 as padding (GPT-2 token 0 is '!')
    BOS = 50256  # GPT-2 <|endoftext|>
    EOS = 50256  # Same as BOS for GPT-2
    VOCAB_SIZE = 50257

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.stoi = {}  # Not used for BPE, but kept for interface compat
        self.itos = {}

    def encode(self, text: str) -> list:
        """Encode text to token IDs. Adds BOS at start."""
        tokens = self.enc.encode(text, disallowed_special=())
        return [self.BOS] + tokens

    def decode(self, ids: list) -> str:
        """Decode token IDs back to text."""
        # Strip BOS/EOS/PAD
        ids = [i for i in ids if i != self.BOS and i != self.EOS and i != self.PAD]
        return self.enc.decode(ids)

    def __len__(self):
        return self.VOCAB_SIZE


# ---------------------------------------------------------------------------
# Dataset: pre-tokenize and hold in memory
# ---------------------------------------------------------------------------

class TextDataset:
    """Pre-tokenized text dataset held in memory as a 1D token tensor.

    Tokenizes the raw text file once at init, concatenating all documents
    with EOS separators, then stores as a flat tensor for efficient slicing.
    """

    def __init__(self, path: str, tokenizer: BPETokenizer, max_tokens: int = None):
        self.tokenizer = tokenizer
        self.path = path

        cache_path = path + ".tokens.pt"
        if os.path.exists(cache_path):
            print(f"Loading pre-tokenized cache: {cache_path}", flush=True)
            self.tokens = torch.load(cache_path, weights_only=True)
            print(f"  {len(self.tokens)} tokens loaded", flush=True)
        else:
            print(f"Tokenizing {path}...", flush=True)
            self.tokens = self._tokenize_file(path, tokenizer)
            print(f"  {len(self.tokens)} tokens, saving cache to {cache_path}", flush=True)
            torch.save(self.tokens, cache_path)

        if max_tokens is not None and len(self.tokens) > max_tokens:
            self.tokens = self.tokens[:max_tokens]
            print(f"  Truncated to {len(self.tokens)} tokens", flush=True)

    @staticmethod
    def _tokenize_file(path: str, tokenizer: BPETokenizer) -> torch.Tensor:
        """Read text file, tokenize line by line, concatenate with EOS."""
        all_tokens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line is a story/document — encode and add EOS separator
                tokens = tokenizer.enc.encode(line, disallowed_special=())
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.EOS)
        return torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return len(self.tokens)


# ---------------------------------------------------------------------------
# Batch sampling — same interface as data.sample_batch
# ---------------------------------------------------------------------------

def sample_text_batch(
    batch_size: int,
    seq_len: int,
    dataset: TextDataset,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of text sequences for next-token prediction.

    Returns:
        input_ids: (B, T) token ids
        labels: (B, T) labels for next-token prediction (shifted by 1).
                Positions where input is PAD/BOS/EOS get -100 (ignore).
        task_ids: (B,) all zeros (text task)
    """
    n_tokens = len(dataset.tokens)
    max_start = n_tokens - seq_len - 1

    input_ids = torch.full((batch_size, seq_len), dataset.tokenizer.PAD, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    task_ids = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # Random start position
        if rng is not None:
            start = int(torch.randint(0, max_start, (1,), generator=rng).item())
        else:
            start = int(torch.randint(0, max_start, (1,)).item())

        # Extract sequence
        chunk = dataset.tokens[start : start + seq_len]
        input_ids[b, : len(chunk)] = chunk

        # Labels: predict next token at every position
        # label[t] = input_ids[t+1] for t in [0, seq_len-2]
        # The last position has no next token, so label stays -100
        next_chunk = dataset.tokens[start + 1 : start + seq_len + 1]
        labels[b, : len(next_chunk)] = next_chunk

    return input_ids, labels, task_ids


# ---------------------------------------------------------------------------
# Evaluation: compute perplexity on a held-out set
# ---------------------------------------------------------------------------

def make_eval_batches(
    dataset: TextDataset,
    seq_len: int,
    n_batches: int = 10,
    batch_size: int = 8,
) -> list:
    """Create fixed eval batches evenly spaced through the dataset.

    Returns list of (input_ids, labels) tuples.
    """
    n_tokens = len(dataset.tokens)
    step = (n_tokens - seq_len - 1) // (n_batches * batch_size)
    batches = []
    pos = 0
    for _ in range(n_batches):
        input_ids = torch.full((batch_size, seq_len), dataset.tokenizer.PAD, dtype=torch.long)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        for b in range(batch_size):
            chunk = dataset.tokens[pos : pos + seq_len]
            input_ids[b, : len(chunk)] = chunk
            next_chunk = dataset.tokens[pos + 1 : pos + seq_len + 1]
            labels[b, : len(next_chunk)] = next_chunk
            pos += step
        batches.append((input_ids, labels))
    return batches
