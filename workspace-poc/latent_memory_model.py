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
    use_explicit_arithmetic: bool = False
    value_hidden: int = 256


def _build_circulant_shift(operand: int, n: int = MOD, subtract: bool = False) -> torch.Tensor:
    """Build an n×n circulant matrix for addition/subtraction by operand (mod n).

    For addition: result[i] = src[(i - operand) % n], so one_hot(v) → one_hot((v+operand)%n).
    For subtraction: result[i] = src[(i + operand) % n], so one_hot(v) → one_hot((v-operand)%n).
    """
    mat = torch.zeros(n, n)
    for i in range(n):
        if subtract:
            mat[i, (i + operand) % n] = 1.0
        else:
            mat[i, (i - operand) % n] = 1.0
    return mat


def _build_mul_permutation(operand: int, n: int = MOD) -> torch.Tensor:
    """Build an n×n permutation matrix for multiplication by operand (mod n).

    result[i] = src[j] where i = (j * operand) % n, so one_hot(v) → one_hot((v*operand)%n).
    """
    mat = torch.zeros(n, n)
    for j in range(n):
        mat[(j * operand) % n, j] = 1.0
    return mat


class OracleMemoryTransition(nn.Module):
    """Transition with an explicit value-logit channel for modular arithmetic.

    When use_value_channel is True, the transition reads the source slot's
    value logits, computes new value logits through a dedicated arithmetic
    head, and writes them back alongside the latent state update.  This
    separates *value propagation* (which must be exact) from *latent
    context* (which can be approximate).

    When use_explicit_arithmetic is True, the arithmetic is computed exactly
    using circulant/permutation matrices instead of being learned.  This tests
    whether the routing/propagation works when arithmetic is given for free.
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
        self.use_explicit_arithmetic = config.use_explicit_arithmetic

        if self.use_value_channel or self.use_explicit_arithmetic:
            value_dim = MOD + 1
            # Read value logits from a slot's latent state.
            self.value_read = nn.Linear(config.d_model, value_dim, bias=False)
            # Write computed value logits back into latent state.
            self.value_write = nn.Linear(value_dim, config.d_model, bias=False)

        if self.use_value_channel and not self.use_explicit_arithmetic:
            # Compute new value logits from (source_value_logits, operator_emb, operand_emb).
            self.value_compute = nn.Sequential(
                nn.Linear(value_dim + config.d_model * 2, config.value_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(config.value_hidden, value_dim, bias=False),
            )

        if self.use_explicit_arithmetic:
            # Pre-compute circulant/permutation matrices for all operands.
            # Shape: (MOD, MOD, MOD) — indexed by operand value.
            self.register_buffer("add_matrices", torch.stack([_build_circulant_shift(k) for k in range(MOD)]))
            self.register_buffer("sub_matrices", torch.stack([_build_circulant_shift(k, subtract=True) for k in range(MOD)]))
            self.register_buffer("mul_matrices", torch.stack([_build_mul_permutation(k) for k in range(MOD)]))

    def explicit_arithmetic(
        self,
        source_value_logits: torch.Tensor,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exact modular arithmetic using pre-computed matrices."""
        # source_value_logits: (B, S, MOD+1) — use only first MOD logits.
        src = source_value_logits[..., :MOD]  # (B, S, MOD)
        # Softmax to get a distribution, then apply the matrix.
        src_dist = F.softmax(src, dim=-1)  # (B, S, MOD)

        # Gather the right matrix for each slot based on operator and operand.
        # operator_targets: (B, S) — 0=none, 1=add, 2=sub, 3=mul
        # operand_targets: (B, S) — 0..MOD-1
        B, S = operator_targets.shape
        result = torch.zeros_like(src_dist)

        for op_id, matrices in [(1, self.add_matrices), (2, self.sub_matrices), (3, self.mul_matrices)]:
            mask = operator_targets == op_id  # (B, S)
            if not mask.any():
                continue
            # For each (b, s) where mask is true, apply matrices[operand_targets[b,s]] to src_dist[b,s]
            indices = mask.nonzero()  # (N, 2)
            for idx in indices:
                b, s = idx
                operand = operand_targets[b, s].item()
                result[b, s] = matrices[operand] @ src_dist[b, s]

        # Append the "unknown" logit (index MOD) — set to large negative for known values.
        unknown_logit = source_value_logits[..., MOD:MOD+1]  # (B, S, 1)
        return torch.cat([result, unknown_logit], dim=-1)

    def forward(
        self,
        memory: torch.Tensor,
        metadata: torch.Tensor,
        source_targets: torch.Tensor,
        operator_metadata: torch.Tensor,
        operand_metadata: torch.Tensor,
        operator_targets: Optional[torch.Tensor] = None,
        operand_targets: Optional[torch.Tensor] = None,
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

        if self.use_explicit_arithmetic and operator_targets is not None and operand_targets is not None:
            source_value_logits = self.value_read(source_memory)
            new_value_logits = self.explicit_arithmetic(
                source_value_logits, operator_targets, operand_targets
            )
            value_update = self.value_write(new_value_logits)
            updated = updated + self.update_scale * value_update
        elif self.use_value_channel:
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
                transition_input,
                step_metadata,
                source_targets,
                step_op,
                step_operand,
                operator_targets=operator_targets if self.config.use_explicit_arithmetic else None,
                operand_targets=operand_targets if self.config.use_explicit_arithmetic else None,
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


# ---------------------------------------------------------------------------
# Register-file oracle model
#
# The key insight from the ablation: routing values through a latent bottleneck
# causes compounding information loss.  Instead, keep a parallel value-logit
# register file that never gets compressed.  The latent state only controls
# *when* to update a slot and provides routing context.
#
#   value_logits: (B, S, MOD+1)  — parallel register file, lossless
#   latent:       (B, S, d_model) — routing/context controller only
#
# Each step:
#   1. Gather source value logits directly (no projection).
#   2. Apply exact arithmetic (circulant/permutation matrices).
#   3. Compute a gate from the latent controller.
#   4. Write the result directly to the value register.
#   5. Update the latent state for context.
# ---------------------------------------------------------------------------


@dataclass
class RegisterFileConfig:
    d_model: int = 128
    n_slots: int = 16
    max_reasoning_steps: int = 7
    expand: int = 4
    detach_latent_steps: bool = True
    value_init_scale: float = 10.0


class RegisterFileTransition(nn.Module):
    """Latent controller that decides when to update each value register."""

    def __init__(self, config: RegisterFileConfig):
        super().__init__()
        self.config = config
        self.n_slots = config.n_slots
        hidden = config.d_model * config.expand

        # The controller sees: current latent, source slot latent, metadata.
        self.input_norm = RMSNorm(config.d_model)
        self.controller = nn.Sequential(
            nn.Linear(config.d_model * 3, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, config.d_model, bias=False),
        )
        self.gate_head = nn.Linear(config.d_model * 3, 1, bias=True)
        self.output_norm = RMSNorm(config.d_model)

    def forward(
        self,
        latent: torch.Tensor,
        metadata: torch.Tensor,
        source_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (updated_latent, gate) where gate is (B, S, 1) in [0, 1]."""
        source_indices = source_targets.clamp_max(self.n_slots - 1)
        batch_indices = torch.arange(latent.shape[0], device=latent.device)[:, None]
        source_latent = latent[batch_indices, source_indices]
        ctrl_input = torch.cat(
            (self.input_norm(latent), self.input_norm(source_latent), metadata), dim=-1
        )
        proposal = self.controller(ctrl_input)
        gate = torch.sigmoid(self.gate_head(ctrl_input))  # (B, S, 1)
        updated_latent = self.output_norm(latent + proposal)
        return updated_latent, gate


class RegisterFileReasoner(nn.Module):
    """Oracle model with a lossless parallel value-logit register file."""

    def __init__(self, config: RegisterFileConfig):
        super().__init__()
        self.config = config
        self.n_slots = config.n_slots

        # Latent controller embeddings.
        self.slot_embedding = nn.Embedding(config.n_slots, config.d_model)
        self.source_embedding = nn.Embedding(config.n_slots + 1, config.d_model)
        self.operator_embedding = nn.Embedding(4, config.d_model)
        self.operand_embedding = nn.Embedding(MOD, config.d_model)
        self.value_init = nn.Linear(MOD + 1, config.d_model, bias=False)
        self.metadata_norm = RMSNorm(config.d_model)
        self.latent_norm = RMSNorm(config.d_model)
        self.transition = RegisterFileTransition(config)

        # Pre-compute exact arithmetic matrices.
        self.register_buffer("add_matrices", torch.stack([_build_circulant_shift(k) for k in range(MOD)]))
        self.register_buffer("sub_matrices", torch.stack([_build_circulant_shift(k, subtract=True) for k in range(MOD)]))
        self.register_buffer("mul_matrices", torch.stack([_build_mul_permutation(k) for k in range(MOD)]))

        self.apply(LatentMemoryReasoner._init_weights)
        # Re-initialize gate to start open (~0.98). Must be after apply()
        # which zeroes all linear biases.
        nn.init.zeros_(self.transition.gate_head.weight)
        nn.init.constant_(self.transition.gate_head.bias, 4.0)

    def _init_value_logits(self, initial_values: torch.Tensor) -> torch.Tensor:
        """Create one-hot value logits for each slot. Unknown = index MOD."""
        B, S = initial_values.shape
        value_logits = torch.full(
            (B, S, MOD + 1), -self.config.value_init_scale, device=initial_values.device
        )
        # Scatter one-hot: set the correct value index to +scale, rest stay negative.
        value_logits.scatter_(2, initial_values.unsqueeze(-1), self.config.value_init_scale)
        return value_logits

    def _exact_arithmetic(
        self,
        source_value_logits: torch.Tensor,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exact modular arithmetic to value logits.

        Args:
            source_value_logits: (B, S, MOD+1)
            operator_targets: (B, S) — 0=none/direct, 1=add, 2=sub, 3=mul
            operand_targets: (B, S) — 0..MOD-1

        Returns: (B, S, MOD+1) new value logits.
        """
        # Work on the first MOD logits (the value space).
        src = source_value_logits[..., :MOD]  # (B, S, MOD)
        result = src.clone()

        for op_id, matrices in [(1, self.add_matrices), (2, self.sub_matrices), (3, self.mul_matrices)]:
            mask = operator_targets == op_id  # (B, S)
            if not mask.any():
                continue
            # Vectorized: for each slot where mask is true, apply matrices[operand] @ src
            indices = mask.nonzero()
            for idx in indices:
                b, s = idx
                operand = operand_targets[b, s].item()
                result[b, s] = matrices[operand] @ src[b, s]

        # Keep the unknown logit unchanged.
        unknown = source_value_logits[..., MOD:MOD + 1]
        return torch.cat([result, unknown], dim=-1)

    def _metadata(
        self,
        source_targets: torch.Tensor,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
    ) -> torch.Tensor:
        slot_ids = torch.arange(self.n_slots, device=source_targets.device)
        slot_ids = slot_ids.unsqueeze(0).expand(source_targets.shape[0], -1)
        meta = (
            self.slot_embedding(slot_ids)
            + self.source_embedding(source_targets)
            + self.operator_embedding(operator_targets)
            + self.operand_embedding(operand_targets)
        )
        return self.metadata_norm(meta)

    def forward(
        self,
        initial_values: torch.Tensor,
        source_targets: torch.Tensor,
        operator_targets: torch.Tensor,
        operand_targets: torch.Tensor,
        query_targets: torch.Tensor,
        reasoning_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        B, S = initial_values.shape
        device = initial_values.device

        # Initialize the parallel value register (lossless, never compressed).
        value_logits = self._init_value_logits(initial_values)  # (B, S, MOD+1)

        # Initialize the latent controller from value info + metadata.
        metadata = self._metadata(source_targets, operator_targets, operand_targets)
        latent = self.latent_norm(metadata + self.value_init(F.one_hot(initial_values, MOD + 1).float()))

        # Record states for supervision.
        value_states = [value_logits]
        latent_states = [latent]

        steps = self.config.max_reasoning_steps if reasoning_steps is None else reasoning_steps
        for _ in range(steps):
            if self.config.detach_latent_steps:
                step_latent = latent.detach()
                step_metadata = metadata.detach()
            else:
                step_latent = latent
                step_metadata = metadata

            # 1. Controller decides gate and updates latent.
            latent, gate = self.transition(step_latent, step_metadata, source_targets)

            # 2. Gather source value logits directly (no projection!).
            source_indices = source_targets.clamp_max(self.n_slots - 1)
            batch_indices = torch.arange(B, device=device)[:, None]
            src_values = value_logits[batch_indices, source_indices]  # (B, S, MOD+1)

            # 3. Apply exact arithmetic.
            new_values = self._exact_arithmetic(src_values, operator_targets, operand_targets)

            # 4. Gated write to the value register.
            #    Only write if the source is actually known (argmax != MOD).
            source_known = (src_values.argmax(dim=-1) < MOD)  # (B, S)
            dependent = source_targets.lt(self.n_slots)  # (B, S)
            write_mask = (dependent & source_known).unsqueeze(-1) * gate  # (B, S, 1)
            value_logits = value_logits + write_mask * (new_values - value_logits)

            value_states.append(value_logits)
            latent_states.append(latent)

        value_states_stack = torch.stack(value_states, dim=1)  # (B, T, S, MOD+1)
        latent_states_stack = torch.stack(latent_states, dim=1)  # (B, T, S, d_model)

        # Answer read directly from the value register — no probe needed!
        batch_indices = torch.arange(B, device=device)
        answer_logits = value_states_stack[batch_indices, -1, query_targets, :MOD]

        return {
            "answer_logits": answer_logits,
            "value_logits": value_states_stack,  # (B, T, S, MOD+1)
            "latent_states": latent_states_stack,  # (B, T, S, d_model)
        }

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        memory_targets: torch.Tensor,
        memory_target_mask: torch.Tensor,
        answer_targets: torch.Tensor,
        memory_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        available_steps = min(outputs["value_logits"].shape[1], memory_targets.shape[1])
        targets = memory_targets[:, :available_steps]  # (B, T, S)
        logits = outputs["value_logits"][:, :available_steps]  # (B, T, S, MOD+1)
        # Supervise all slots, but mask out unassigned ones.
        expanded_mask = memory_target_mask[:, None, :].expand_as(targets)
        answer_loss = F.cross_entropy(outputs["answer_logits"], answer_targets)
        memory_loss = F.cross_entropy(logits[expanded_mask], targets[expanded_mask])
        total = answer_loss + memory_weight * memory_loss
        return {"loss": total, "answer_loss": answer_loss, "memory_loss": memory_loss}
