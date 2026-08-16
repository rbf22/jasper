import random

import torch

from data import MOD, Vocab, sample_task1_latent_batch, task1_memory_trace
from latent_memory_model import (
    LatentMemoryConfig,
    LatentMemoryReasoner,
    OracleLatentMemoryConfig,
    OracleLatentMemoryReasoner,
)


def small_config(detach_memory_steps=True):
    return LatentMemoryConfig(
        d_model=32,
        n_encoder_layers=1,
        n_heads=4,
        n_slots=16,
        max_reasoning_steps=3,
        d_state=8,
        expand=2,
        detach_memory_steps=detach_memory_steps,
    )


def test_task1_memory_trace_resolves_one_dependency_per_step():
    states, query_index = task1_memory_trace("a=14;b=a+95;c=b*2;?c;")
    assert query_index == 2
    assert states[0][:3] == [14, MOD, MOD]
    assert states[1][:3] == [14, 12, MOD]
    assert states[2][:3] == [14, 12, 24]


def test_task1_latent_batch_contains_prompt_only_and_step_targets():
    vocab = Vocab()
    (
        inputs,
        mask,
        targets,
        target_mask,
        answers,
        query_targets,
        steps,
        source_targets,
        operator_targets,
        operand_targets,
    ) = sample_task1_latent_batch(
        4,
        128,
        vocab,
        depth_range=(4, 4),
        max_reasoning_steps=7,
        rng=random.Random(7),
    )
    assert inputs.shape == (4, 128)
    assert mask.shape == inputs.shape
    assert targets.shape == (4, 8, 16)
    assert target_mask.shape == (4, 16)
    assert answers.shape == (4,)
    assert query_targets.eq(3).all()
    assert steps.eq(3).all()
    for row, row_mask in zip(inputs, mask):
        assert vocab.decode(row[row_mask].tolist()).endswith(";")


def test_latent_memory_forward_has_fixed_state_size():
    vocab = Vocab()
    (
        inputs,
        mask,
        targets,
        target_mask,
        answers,
        query_targets,
        _,
        source_targets,
        operator_targets,
        operand_targets,
    ) = sample_task1_latent_batch(
        2,
        64,
        vocab,
        depth_range=(2, 3),
        max_reasoning_steps=3,
        rng=random.Random(11),
    )
    model = LatentMemoryReasoner(small_config())
    outputs = model(inputs, mask)
    assert outputs["answer_logits"].shape == (2, MOD)
    assert outputs["memory_logits"].shape == (2, 4, 16, MOD + 1)
    assert outputs["memory_states"].shape == (2, 4, 16, 32)
    losses = model.loss(
        outputs,
        targets[:, :4],
        target_mask,
        answers,
        query_targets,
        source_targets,
        operator_targets,
        operand_targets,
    )
    assert torch.isfinite(losses["loss"])


def test_memory_initialization_is_per_example():
    vocab = Vocab()
    inputs, mask, _, _, _, _, _, _, _, _ = sample_task1_latent_batch(
        2,
        64,
        vocab,
        depth_range=(2, 2),
        max_reasoning_steps=3,
        rng=random.Random(19),
    )
    model = LatentMemoryReasoner(small_config()).eval()
    with torch.no_grad():
        memory = model.initialize_memory(model.encode(inputs), mask)
    assert not torch.allclose(memory[0], memory[1])


def test_detached_transition_does_not_backpropagate_final_state_to_writer():
    vocab = Vocab()
    inputs, mask, _, _, _, _, _, _, _, _ = sample_task1_latent_batch(
        2,
        64,
        vocab,
        depth_range=(2, 2),
        max_reasoning_steps=3,
        rng=random.Random(23),
    )
    model = LatentMemoryReasoner(small_config(detach_memory_steps=True))
    outputs = model(inputs, mask)
    outputs["memory_states"][:, -1].square().mean().backward()
    writer_grad = model.memory_writer.in_proj_weight.grad
    transition_grad = model.transition.attention.in_proj_weight.grad
    assert writer_grad is None or writer_grad.abs().sum().item() == 0
    assert transition_grad is not None and transition_grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Oracle latent-memory model tests
# ---------------------------------------------------------------------------


def oracle_config(detach_memory_steps=True):
    return OracleLatentMemoryConfig(
        d_model=32,
        n_slots=16,
        max_reasoning_steps=3,
        expand=2,
        detach_memory_steps=detach_memory_steps,
    )


def oracle_batch(batch_size=4, depth_range=(2, 2), seed=101, max_reasoning_steps=3):
    vocab = Vocab()
    (
        _,
        _,
        memory_targets,
        memory_mask,
        answers,
        query_targets,
        _,
        source_targets,
        operator_targets,
        operand_targets,
    ) = sample_task1_latent_batch(
        batch_size,
        64,
        vocab,
        depth_range=depth_range,
        max_reasoning_steps=max_reasoning_steps,
        rng=random.Random(seed),
    )
    return {
        "initial_values": memory_targets[:, 0],
        "memory_targets": memory_targets,
        "memory_mask": memory_mask,
        "answers": answers,
        "query_targets": query_targets,
        "source_targets": source_targets,
        "operator_targets": operator_targets,
        "operand_targets": operand_targets,
    }


def test_oracle_forward_shapes_are_fixed():
    batch = oracle_batch()
    model = OracleLatentMemoryReasoner(oracle_config())
    outputs = model(
        batch["initial_values"],
        batch["source_targets"],
        batch["operator_targets"],
        batch["operand_targets"],
        batch["query_targets"],
    )
    assert outputs["answer_logits"].shape == (4, MOD)
    assert outputs["memory_logits"].shape == (4, 4, 16, MOD + 1)
    assert outputs["memory_states"].shape == (4, 4, 16, 32)


def test_oracle_initial_memory_contains_direct_constants():
    """After a few training steps, the value probe should read direct constants from initial memory."""
    torch.manual_seed(0)
    batch = oracle_batch(batch_size=2, depth_range=(1, 1), seed=55)
    config = oracle_config()
    config.max_reasoning_steps = 0
    model = OracleLatentMemoryReasoner(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(100):
        train_batch = oracle_batch(batch_size=64, depth_range=(1, 1), seed=random.randint(0, 10**9))
        outputs = model(
            train_batch["initial_values"],
            train_batch["source_targets"],
            train_batch["operator_targets"],
            train_batch["operand_targets"],
            train_batch["query_targets"],
            reasoning_steps=0,
        )
        losses = model.loss(
            outputs, train_batch["memory_targets"], train_batch["memory_mask"], train_batch["answers"]
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["initial_values"],
            batch["source_targets"],
            batch["operator_targets"],
            batch["operand_targets"],
            batch["query_targets"],
            reasoning_steps=0,
        )
    initial_logits = outputs["memory_logits"][:, 0]
    predicted = initial_logits.argmax(dim=-1)
    known = batch["memory_mask"]
    correct = predicted[known].eq(batch["memory_targets"][:, 0][known])
    assert correct.float().mean().item() > 0.5


def test_oracle_loss_is_finite_and_backprops():
    batch = oracle_batch()
    model = OracleLatentMemoryReasoner(oracle_config())
    outputs = model(
        batch["initial_values"],
        batch["source_targets"],
        batch["operator_targets"],
        batch["operand_targets"],
        batch["query_targets"],
    )
    losses = model.loss(
        outputs,
        batch["memory_targets"],
        batch["memory_mask"],
        batch["answers"],
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    transition_grad = model.transition.transition[0].weight.grad
    assert transition_grad is not None and transition_grad.abs().sum().item() > 0


def test_oracle_detached_transition_blocks_gradient_to_embeddings():
    batch = oracle_batch()
    model = OracleLatentMemoryReasoner(oracle_config(detach_memory_steps=True))
    outputs = model(
        batch["initial_values"],
        batch["source_targets"],
        batch["operator_targets"],
        batch["operand_targets"],
        batch["query_targets"],
    )
    # Loss only on the final memory state — should not reach embeddings.
    outputs["memory_states"][:, -1].square().mean().backward()
    value_grad = model.value_embedding.weight.grad
    slot_grad = model.slot_embedding.weight.grad
    assert value_grad is None or value_grad.abs().sum().item() < 1e-6
    assert slot_grad is None or slot_grad.abs().sum().item() < 1e-6


def test_oracle_one_step_resolves_addition():
    """A single latent step should be able to learn b = a + k (mod 97)."""
    torch.manual_seed(0)
    config = oracle_config()
    config.max_reasoning_steps = 1
    model = OracleLatentMemoryReasoner(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    vocab = Vocab()
    for _ in range(300):
        batch = oracle_batch(batch_size=64, depth_range=(1, 1), seed=random.randint(0, 10**9))
        outputs = model(
            batch["initial_values"],
            batch["source_targets"],
            batch["operator_targets"],
            batch["operand_targets"],
            batch["query_targets"],
        )
        losses = model.loss(outputs, batch["memory_targets"], batch["memory_mask"], batch["answers"])
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
    batch = oracle_batch(batch_size=128, depth_range=(1, 1), seed=999)
    with torch.no_grad():
        outputs = model(
            batch["initial_values"],
            batch["source_targets"],
            batch["operator_targets"],
            batch["operand_targets"],
            batch["query_targets"],
        )
    accuracy = outputs["answer_logits"].argmax(-1).eq(batch["answers"]).float().mean()
    assert accuracy.item() > 0.8, f"one-step addition accuracy too low: {accuracy.item():.3f}"


def test_oracle_multi_step_chain_runs_without_divergence():
    """Multi-step chains should train without NaN/divergence. Full ablation run separately."""
    torch.manual_seed(0)
    config = oracle_config()
    config.d_model = 64
    config.expand = 4
    config.max_reasoning_steps = 7
    config.detach_memory_steps = False
    model = OracleLatentMemoryReasoner(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(200):
        batch = oracle_batch(
            batch_size=64, depth_range=(2, 4), seed=random.randint(0, 10**9), max_reasoning_steps=7
        )
        outputs = model(
            batch["initial_values"],
            batch["source_targets"],
            batch["operator_targets"],
            batch["operand_targets"],
            batch["query_targets"],
        )
        losses = model.loss(outputs, batch["memory_targets"], batch["memory_mask"], batch["answers"])
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        assert torch.isfinite(losses["loss"]), f"loss diverged at step {_}"
    # Verify the model produces finite outputs on depth-4 examples.
    batch = oracle_batch(batch_size=64, depth_range=(4, 4), seed=777, max_reasoning_steps=7)
    with torch.no_grad():
        outputs = model(
            batch["initial_values"],
            batch["source_targets"],
            batch["operator_targets"],
            batch["operand_targets"],
            batch["query_targets"],
        )
    assert torch.isfinite(outputs["answer_logits"]).all()
