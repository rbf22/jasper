"""Data pipeline for the text latent-memory model.

Parses tiny challenges into (prompt, answer) pairs and provides batch sampling
with BPE tokenization. Each challenge line has the format:

    <story and question>? <answer>

We split on the last '?' to separate the prompt (everything up to and including
the '?') from the answer (everything after).
"""

import os
import random
from typing import List, Tuple, Optional

import torch
import tiktoken


class ChallengeDataset:
    """Loads and tokenizes tiny challenges into (prompt, answer) pairs."""

    def __init__(
        self,
        train_path: str,
        valid_path: Optional[str] = None,
        max_prompt_len: int = 256,
        max_answer_len: int = 32,
    ):
        self.enc = tiktoken.get_encoding("gpt2")
        self.bos_id = 50256
        self.eos_id = 50256
        self.pad_id = 0
        self.vocab_size = 50257
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len

        self.train_examples = self._load_and_parse(train_path)
        self.valid_examples = self._load_and_parse(valid_path) if valid_path else []

        print(f"Loaded {len(self.train_examples)} train, {len(self.valid_examples)} valid examples")

    def _load_and_parse(self, path: str) -> List[Tuple[List[int], List[int]]]:
        """Parse file into (prompt_token_ids, answer_token_ids) pairs."""
        if not path or not os.path.exists(path):
            return []
        examples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if "?" not in line:
                    continue
                # Split on last '?' — everything before + '?' is the prompt
                idx = line.rfind("?")
                prompt_text = line[:idx + 1]
                answer_text = line[idx + 1:].strip()

                # Tokenize: prompt gets BOS prefix, answer gets BOS prefix + EOS suffix
                prompt_ids = [self.bos_id] + self.enc.encode(prompt_text, disallowed_special=())
                answer_ids = [self.bos_id] + self.enc.encode(answer_text, disallowed_special=()) + [self.eos_id]

                # Truncate
                prompt_ids = prompt_ids[:self.max_prompt_len]
                answer_ids = answer_ids[:self.max_answer_len + 1]  # +1 for BOS

                if len(answer_ids) < 2:  # need at least BOS + one token
                    continue

                examples.append((prompt_ids, answer_ids))
        return examples

    def sample_batch(
        self,
        batch_size: int,
        split: str = "train",
        rng: Optional[random.Random] = None,
        fixed_prompt_len: Optional[int] = None,
        fixed_answer_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of (prompt_ids, prompt_mask, decoder_input, answer_targets, answer_mask).

        Returns:
            prompt_ids: (B, T_prompt) — padded prompt token IDs
            prompt_mask: (B, T_prompt) — 1 for valid prompt tokens
            decoder_input: (B, T_ans) — answer tokens shifted right (BOS prepended, last token dropped)
            answer_targets: (B, T_ans) — target answer tokens (first token dropped)
            answer_mask: (B, T_ans) — 1 for valid answer tokens

        If fixed_prompt_len/fixed_answer_len are provided, all batches are padded
        to those exact lengths. This avoids triggering new kernel compilations
        for every new sequence length on Tenstorrent hardware.
        """
        if rng is None:
            rng = random.Random()
        examples = self.train_examples if split == "train" else self.valid_examples

        # Sample batch
        batch = rng.sample(examples, min(batch_size, len(examples)))

        # Use fixed lengths if provided, otherwise per-batch max
        if fixed_prompt_len is not None:
            max_prompt = fixed_prompt_len
        else:
            max_prompt = max(len(p) for p, _ in batch)
        if fixed_answer_len is not None:
            max_answer = fixed_answer_len
        else:
            max_answer = max(len(a) for _, a in batch)

        prompt_ids = torch.full((len(batch), max_prompt), self.pad_id, dtype=torch.long)
        prompt_mask = torch.zeros(len(batch), max_prompt, dtype=torch.bool)
        answer_full = torch.full((len(batch), max_answer), self.pad_id, dtype=torch.long)
        answer_mask = torch.zeros(len(batch), max_answer, dtype=torch.bool)

        for i, (p_ids, a_ids) in enumerate(batch):
            prompt_ids[i, :len(p_ids)] = torch.tensor(p_ids)
            prompt_mask[i, :len(p_ids)] = True
            answer_full[i, :len(a_ids)] = torch.tensor(a_ids)
            answer_mask[i, :len(a_ids)] = True

        # decoder_input = answer_full[:, :-1] (BOS + tokens except last)
        # answer_targets = answer_full[:, 1:] (tokens except BOS)
        decoder_input = answer_full[:, :-1]
        answer_targets = answer_full[:, 1:]
        ans_mask = answer_mask[:, 1:]  # align mask with targets

        return prompt_ids, prompt_mask, decoder_input, answer_targets, ans_mask


def exact_match_accuracy(
    generated: torch.Tensor,
    targets: torch.Tensor,
    eos_id: int = 50256,
    pad_id: int = 0,
) -> float:
    """Compute exact match accuracy: generated sequence matches target up to EOS."""
    correct = 0
    total = generated.shape[0]
    for i in range(total):
        # Find EOS in targets
        target_seq = []
        for t in targets[i].tolist():
            if t == eos_id:
                break
            if t != pad_id:
                target_seq.append(t)

        gen_seq = []
        for g in generated[i].tolist():
            if g == eos_id:
                break
            if g != pad_id:
                gen_seq.append(g)

        if gen_seq == target_seq:
            correct += 1
    return correct / total
