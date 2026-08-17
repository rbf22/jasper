"""Text latent-memory reasoning model for the WRAP MVP.

Architecture:
  1. Encode the prompt once with a small transformer encoder
  2. Initialize fixed-size memory slots via cross-attention from prompt
  3. Recur K times over memory (learned transitions, gated writes)
  4. Decode the answer autoregressively via cross-attention to memory

Key properties:
  - Fixed memory size (N slots × d_model) regardless of prompt length
  - No growing KV cache — the decoder attends to fixed memory + short answer
  - Encode-once, decode-once — the prompt is never re-processed
  - Learned transitions (no exact arithmetic — text values are semantic)

This is the text adaptation of the register-file model from workspace-poc.
The POC proved that fixed-size latent memory with recurrent refinement
can solve multi-hop reasoning. This model tests whether the same
architecture works when values are continuous semantic states and
operations are learned.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextLatentMemoryConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    n_heads: int = 4
    n_slots: int = 16
    max_reasoning_steps: int = 6
    expand: int = 4
    dropout: float = 0.1
    max_answer_len: int = 32
    pad_token_id: int = 0


class TextLatentMemoryTransition(nn.Module):
    """One recurrent reasoning step over memory slots."""

    def __init__(self, config: TextLatentMemoryConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.expand, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model * config.expand, config.d_model, bias=False),
        )
        self.mlp_norm = nn.RMSNorm(config.d_model)
        # Gate: per-slot, per-dimension write gate
        self.gate = nn.Linear(config.d_model, config.d_model, bias=True)
        self.output_norm = nn.RMSNorm(config.d_model)

    def forward(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (updated_memory, gate) where gate is (B, S, d_model) in [0, 1]."""
        normed = self.attn_norm(memory)
        attended, _ = self.self_attn(normed, normed, normed, need_weights=False)
        hidden = self.attn_norm(memory + attended)
        proposal = self.mlp(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        updated = self.output_norm(memory + gate * proposal)
        return updated, gate


class TextLatentMemoryModel(nn.Module):
    """Encode-once, recurrent-memory, decode-once model for text reasoning."""

    def __init__(self, config: TextLatentMemoryConfig):
        super().__init__()
        self.config = config
        self.n_slots = config.n_slots

        # Token embedding (shared between encoder and decoder)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.prompt_pos_embedding = nn.Embedding(512, config.d_model)
        self.answer_pos_embedding = nn.Embedding(config.max_answer_len + 2, config.d_model)

        # Encoder: small transformer to encode the prompt
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.expand,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
        self.encoder_norm = nn.RMSNorm(config.d_model)

        # Memory initialization: cross-attention from learnable slot queries to encoded prompt
        self.slot_queries = nn.Parameter(torch.randn(config.n_slots, config.d_model) * 0.02)
        self.memory_init = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.memory_norm = nn.RMSNorm(config.d_model)

        # Recurrent transition
        self.transition = TextLatentMemoryTransition(config)

        # Decoder: autoregressive with cross-attention to memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.expand,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
        self.decoder_norm = nn.RMSNorm(config.d_model)

        # Output projection (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self.apply(self._init_weights)
        # Initialize gate to start open
        nn.init.zeros_(self.transition.gate.weight)
        nn.init.constant_(self.transition.gate.bias, 4.0)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode the prompt once. Returns (B, T, d_model)."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.prompt_pos_embedding(pos)
        # TransformerEncoder doesn't support key_padding_mask directly in all versions
        # Use src_key_padding_mask
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        return self.encoder_norm(hidden)

    def initialize_memory(self, encoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Initialize memory slots via cross-attention from learnable queries."""
        B = encoded.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        memory, _ = self.memory_init(
            queries, encoded, encoded,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        return self.memory_norm(queries + memory)

    def reason(self, memory: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Run K recurrent reasoning steps over memory."""
        states = [memory]
        gates = []
        for _ in range(self.config.max_reasoning_steps):
            memory, gate = self.transition(memory)
            states.append(memory)
            gates.append(gate)
        return memory, states, gates

    def decode_answer(
        self,
        memory: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode answer autoregressively via cross-attention to memory.

        Args:
            memory: (B, S, d_model) — final memory state after reasoning
            answer_ids: (B, T_ans) — answer token IDs (teacher-forced)
            answer_mask: (B, T_ans) — 1 for valid tokens, 0 for padding

        Returns: (B, T_ans, vocab_size) — logits for each answer position
        """
        B, T_ans = answer_ids.shape
        pos = torch.arange(T_ans, device=answer_ids.device).unsqueeze(0)
        ans_emb = self.token_embedding(answer_ids) + self.answer_pos_embedding(pos)

        # Causal mask for autoregressive decoding
        causal_mask = torch.triu(
            torch.full((T_ans, T_ans), float('-inf'), device=answer_ids.device),
            diagonal=1,
        )
        # Padding mask
        ans_key_padding = ~answer_mask.bool()

        decoded = self.decoder(
            ans_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=ans_key_padding,
        )
        decoded = self.decoder_norm(decoded)
        return self.lm_head(decoded)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode → init memory → reason → decode.

        answer_ids should be shifted right (BOS prepended) for teacher forcing.
        """
        encoded = self.encode_prompt(prompt_ids, prompt_mask)
        memory = self.initialize_memory(encoded, prompt_mask)
        memory, memory_states, gates = self.reason(memory)
        logits = self.decode_answer(memory, answer_ids, answer_mask)
        return {
            "logits": logits,
            "memory_states": memory_states,
            "gates": gates,
            "final_memory": memory,
        }

    def generate(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        max_new_tokens: int = 32,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
    ) -> torch.Tensor:
        """Greedy generation from the model."""
        B = prompt_ids.shape[0]
        device = prompt_ids.device
        encoded = self.encode_prompt(prompt_ids, prompt_mask)
        memory = self.initialize_memory(encoded, prompt_mask)
        memory, _, _ = self.reason(memory)

        # Start with BOS
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            answer_mask = torch.ones_like(generated, dtype=torch.bool)
            logits = self.decode_answer(memory, generated, answer_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = next_token.masked_fill(finished, self.config.pad_token_id)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

        return generated[:, 1:]  # strip BOS

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        answer_targets: torch.Tensor,
        answer_mask: torch.Tensor,
        memory_reg_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Cross-entropy loss on answer tokens.

        answer_targets: (B, T_ans) — the target token IDs (not shifted)
        answer_mask: (B, T_ans) — 1 for valid tokens
        """
        logits = outputs["logits"]  # (B, T_ans, V)
        # Shift: predict token t+1 from token t
        # logits[:, :-1] predicts answer_targets[:, 1:]
        # But we already shifted in forward (answer_ids is shifted right)
        # So logits[:, t] predicts answer_targets[:, t]
        V = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            answer_targets.reshape(-1),
            reduction='none',
        ).reshape(answer_targets.shape)
        loss = (loss * answer_mask.float()).sum() / answer_mask.float().sum().clamp_min(1)

        total = loss
        if memory_reg_weight > 0:
            # Encourage memory diversity (prevent slot collapse)
            memory = outputs["final_memory"]  # (B, S, d)
            mem_norm = F.normalize(memory, dim=-1)
            sim = torch.bmm(mem_norm, mem_norm.transpose(1, 2))  # (B, S, S)
            off_diag = sim - torch.eye(sim.size(1), device=sim.device).unsqueeze(0)
            reg = (off_diag ** 2).sum() / (sim.size(0) * sim.size(1) * sim.size(2))
            total = total + memory_reg_weight * reg

        return {"loss": total, "answer_loss": loss}
