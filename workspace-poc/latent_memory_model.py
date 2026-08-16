from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MOD, Vocab
from model import GatedBlock, ModelConfig, RMSNorm, SSMLayer


@dataclass
class LatentMemoryConfig:
    vocab_size: int = Vocab.VOCAB_SIZE
    d_model: int = 128
    n_encoder_layers: int = 4
    n_heads: int = 4
    n_slots: int = 16
    max_reasoning_steps: int = 7
    d_state: int = 32
    expand: int = 2
    dropout: float = 0.0
    detach_memory_steps: bool = True
    memory_update_scale: float = 0.1


class LatentMemoryTransition(nn.Module):
    def __init__(self, config: LatentMemoryConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.attention_norm = RMSNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.expand, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model * config.expand, config.d_model, bias=False),
        )
        self.gate = nn.Linear(config.d_model, config.d_model, bias=True)
        self.output_norm = RMSNorm(config.d_model)
        self.update_scale = config.memory_update_scale

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(memory)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        hidden = self.attention_norm(memory + attended)
        proposal = self.mlp(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return self.output_norm(memory + self.update_scale * gate * proposal)


class LatentMemoryReasoner(nn.Module):
    def __init__(self, config: LatentMemoryConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = ModelConfig(
            d_model=config.d_model,
            n_layers=config.n_encoder_layers,
            vocab_size=config.vocab_size,
            d_state=config.d_state,
            expand=config.expand,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_gradient_checkpointing=False,
        )
        self.encoder_layers = nn.ModuleList(
            GatedBlock(SSMLayer(encoder_config), config.d_model, use_checkpoint=False)
            for _ in range(config.n_encoder_layers)
        )
        self.encoder_norm = RMSNorm(config.d_model)
        self.slot_identity = nn.Parameter(torch.randn(config.n_slots, config.d_model) * 0.02)
        self.memory_writer = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.memory_init_norm = RMSNorm(config.d_model)
        self.transition = LatentMemoryTransition(config)
        self.value_probe = nn.Linear(config.d_model, MOD + 1, bias=False)
        self.source_probe = nn.Linear(config.d_model, config.n_slots + 1, bias=False)
        self.operator_probe = nn.Linear(config.d_model, 4, bias=False)
        self.operand_probe = nn.Linear(config.d_model, MOD, bias=False)
        self.memory_reader = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids)
        for layer in self.encoder_layers:
            hidden = layer(hidden)
        return self.encoder_norm(hidden)

    def initialize_memory(
        self, encoded: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = encoded.shape[0]
        queries = self.slot_identity.unsqueeze(0).expand(batch_size, -1, -1)
        written, _ = self.memory_writer(
            queries,
            encoded,
            encoded,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        return self.memory_init_norm(queries + written)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reasoning_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode(input_ids)
        memory = self.initialize_memory(encoded, attention_mask)
        states = [memory]
        steps = self.config.max_reasoning_steps if reasoning_steps is None else reasoning_steps
        for _ in range(steps):
            transition_input = memory.detach() if self.config.detach_memory_steps else memory
            memory = self.transition(transition_input)
            states.append(memory)

        memory_states = torch.stack(states, dim=1)
        memory_logits = self.value_probe(memory_states)
        source_logits = self.source_probe(memory_states)
        operator_logits = self.operator_probe(memory_states)
        operand_logits = self.operand_probe(memory_states)
        query_indices = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        query = encoded[batch_indices, query_indices].unsqueeze(1)
        _, query_attention = self.memory_reader(
            query, memory, memory, need_weights=True, average_attn_weights=True
        )
        query_attention = query_attention.squeeze(1)
        answer_logits = torch.bmm(
            query_attention.unsqueeze(1), memory_logits[:, -1, :, :MOD]
        ).squeeze(1)
        return {
            "answer_logits": answer_logits,
            "memory_logits": memory_logits,
            "memory_states": memory_states,
            "source_logits": source_logits,
            "operator_logits": operator_logits,
            "operand_logits": operand_logits,
            "query_attention": query_attention,
        }

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        memory_targets: torch.Tensor,
        memory_target_mask: torch.Tensor,
        answer_targets: torch.Tensor,
        query_targets: torch.Tensor,
        source_targets: torch.Tensor,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
        memory_weight: float = 1.0,
        routing_weight: float = 0.25,
        metadata_weight: float = 0.25,
    ) -> Dict[str, torch.Tensor]:
        available_steps = outputs["memory_logits"].shape[1]
        targets = memory_targets[:, :available_steps]
        expanded_mask = memory_target_mask[:, None, :].expand_as(targets)
        answer_loss = F.cross_entropy(outputs["answer_logits"], answer_targets)
        memory_loss = F.cross_entropy(outputs["memory_logits"][expanded_mask], targets[expanded_mask])
        routing_loss = F.nll_loss(outputs["query_attention"].clamp_min(1e-8).log(), query_targets)
        metadata_targets = (
            source_targets[:, None, :].expand(-1, available_steps, -1),
            operator_targets[:, None, :].expand(-1, available_steps, -1),
            operand_targets[:, None, :].expand(-1, available_steps, -1),
        )
        metadata_losses = [
            F.cross_entropy(outputs[name][expanded_mask], target[expanded_mask])
            for name, target in zip(
                ("source_logits", "operator_logits", "operand_logits"), metadata_targets
            )
        ]
        metadata_loss = sum(metadata_losses) / len(metadata_losses)
        total = (
            answer_loss
            + memory_weight * memory_loss
            + routing_weight * routing_loss
            + metadata_weight * metadata_loss
        )
        return {
            "loss": total,
            "answer_loss": answer_loss,
            "memory_loss": memory_loss,
            "routing_loss": routing_loss,
            "metadata_loss": metadata_loss,
        }


@dataclass
class OracleLatentMemoryConfig:
    d_model: int = 128
    n_slots: int = 16
    max_reasoning_steps: int = 7
    expand: int = 4
    detach_memory_steps: bool = True
    memory_update_scale: float = 1.0
    use_value_channel: bool = True
    value_hidden: int = 256


class OracleMemoryTransition(nn.Module):
    """Transition with an explicit value-logit channel for modular arithmetic.

    When use_value_channel is True, the transition reads the source slot's
    value logits, computes new value logits through a dedicated arithmetic
    head, and writes them back alongside the latent state update.  This
    separates *value propagation* (which must be exact) from *latent
    context* (which can be approximate).
    """

    def __init__(self, config: OracleLatentMemoryConfig):
        super().__init__()
        self.config = config
        hidden_size = config.d_model * config.expand
        self.input_norm = RMSNorm(config.d_model)
        self.transition = nn.Sequential(
            nn.Linear(config.d_model * 3, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, config.d_model, bias=False),
        )
        self.gate = nn.Linear(config.d_model * 3, config.d_model, bias=True)
        self.output_norm = RMSNorm(config.d_model)
        self.update_scale = config.memory_update_scale
        self.n_slots = config.n_slots
        self.use_value_channel = config.use_value_channel

        if self.use_value_channel:
            value_dim = MOD + 1
            # Read value logits from a slot's latent state.
            self.value_read = nn.Linear(config.d_model, value_dim, bias=False)
            # Compute new value logits from (source_value_logits, operator_emb, operand_emb).
            self.value_compute = nn.Sequential(
                nn.Linear(value_dim + config.d_model * 2, config.value_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(config.value_hidden, value_dim, bias=False),
            )
            # Write computed value logits back into latent state.
            self.value_write = nn.Linear(value_dim, config.d_model, bias=False)

    def forward(
        self,
        memory: torch.Tensor,
        metadata: torch.Tensor,
        source_targets: torch.Tensor,
        operator_metadata: torch.Tensor,
        operand_metadata: torch.Tensor,
    ) -> torch.Tensor:
        source_indices = source_targets.clamp_max(self.n_slots - 1)
        batch_indices = torch.arange(memory.shape[0], device=memory.device)[:, None]
        source_memory = memory[batch_indices, source_indices]
        transition_input = torch.cat(
            (self.input_norm(memory), self.input_norm(source_memory), metadata), dim=-1
        )
        proposal = self.transition(transition_input)
        gate = torch.sigmoid(self.gate(transition_input))
        updated = self.output_norm(memory + self.update_scale * gate * proposal)
        dependent = source_targets.lt(self.n_slots).unsqueeze(-1)

        if self.use_value_channel:
            source_value_logits = self.value_read(source_memory)
            compute_input = torch.cat(
                (source_value_logits, operator_metadata, operand_metadata), dim=-1
            )
            new_value_logits = self.value_compute(compute_input)
            value_update = self.value_write(new_value_logits)
            updated = updated + self.update_scale * value_update

        return torch.where(dependent, updated, memory)


class OracleLatentMemoryReasoner(nn.Module):
    def __init__(self, config: OracleLatentMemoryConfig):
        super().__init__()
        self.config = config
        self.slot_embedding = nn.Embedding(config.n_slots, config.d_model)
        self.source_embedding = nn.Embedding(config.n_slots + 1, config.d_model)
        self.operator_embedding = nn.Embedding(4, config.d_model)
        self.operand_embedding = nn.Embedding(MOD, config.d_model)
        self.value_embedding = nn.Embedding(MOD + 1, config.d_model)
        self.metadata_norm = RMSNorm(config.d_model)
        self.memory_norm = RMSNorm(config.d_model)
        self.transition = OracleMemoryTransition(config)
        self.value_probe = nn.Linear(config.d_model, MOD + 1, bias=False)
        self.apply(LatentMemoryReasoner._init_weights)

    def metadata(
        self,
        source_targets: torch.Tensor,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
    ) -> torch.Tensor:
        slot_indices = torch.arange(self.config.n_slots, device=source_targets.device)
        slot_indices = slot_indices.unsqueeze(0).expand(source_targets.shape[0], -1)
        metadata = (
            self.slot_embedding(slot_indices)
            + self.source_embedding(source_targets)
            + self.operator_embedding(operator_targets)
            + self.operand_embedding(operand_targets)
        )
        return self.metadata_norm(metadata)

    def operator_operand_metadata(
        self,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.operator_embedding(operator_targets),
            self.operand_embedding(operand_targets),
        )

    def forward(
        self,
        initial_values: torch.Tensor,
        source_targets: torch.Tensor,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
        query_targets: torch.Tensor,
        reasoning_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        metadata = self.metadata(source_targets, operator_targets, operand_targets)
        op_meta, operand_meta = self.operator_operand_metadata(operator_targets, operand_targets)
        memory = self.memory_norm(metadata + self.value_embedding(initial_values))
        states = [memory]
        steps = self.config.max_reasoning_steps if reasoning_steps is None else reasoning_steps
        for _ in range(steps):
            if self.config.detach_memory_steps:
                transition_input = memory.detach()
                step_metadata = metadata.detach()
                step_op = op_meta.detach()
                step_operand = operand_meta.detach()
            else:
                transition_input = memory
                step_metadata = metadata
                step_op = op_meta
                step_operand = operand_meta
            memory = self.transition(
                transition_input, step_metadata, source_targets, step_op, step_operand
            )
            states.append(memory)
        memory_states = torch.stack(states, dim=1)
        memory_logits = self.value_probe(memory_states)
        batch_indices = torch.arange(memory.shape[0], device=memory.device)
        answer_logits = memory_logits[batch_indices, -1, query_targets, :MOD]
        return {
            "answer_logits": answer_logits,
            "memory_logits": memory_logits,
            "memory_states": memory_states,
        }

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        memory_targets: torch.Tensor,
        memory_target_mask: torch.Tensor,
        answer_targets: torch.Tensor,
        memory_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        available_steps = min(outputs["memory_logits"].shape[1], memory_targets.shape[1])
        targets = memory_targets[:, :available_steps]
        logits = outputs["memory_logits"][:, :available_steps]
        expanded_mask = memory_target_mask[:, None, :].expand_as(targets)
        answer_loss = F.cross_entropy(outputs["answer_logits"], answer_targets)
        memory_loss = F.cross_entropy(logits[expanded_mask], targets[expanded_mask])
        total = answer_loss + memory_weight * memory_loss
        return {"loss": total, "answer_loss": answer_loss, "memory_loss": memory_loss}
