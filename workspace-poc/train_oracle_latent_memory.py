import argparse
import random
from pathlib import Path

import torch

from data import Vocab, sample_task1_latent_batch
from latent_memory_model import OracleLatentMemoryConfig, OracleLatentMemoryReasoner


def depth_limit(step: int) -> int:
    if step < 1000:
        return 2
    if step < 3000:
        return 4
    return 8


def sample_batch(batch_size, seq_len, vocab, depth_range, config, rng):
    batch = sample_task1_latent_batch(
        batch_size,
        seq_len,
        vocab,
        depth_range=depth_range,
        max_reasoning_steps=config.max_reasoning_steps,
        rng=rng,
    )
    (
        _,
        _,
        memory_targets,
        memory_mask,
        answers,
        query_targets,
        reasoning_steps,
        source_targets,
        operator_targets,
        operand_targets,
    ) = batch
    return {
        "memory_targets": memory_targets,
        "memory_mask": memory_mask,
        "answers": answers,
        "query_targets": query_targets,
        "reasoning_steps": reasoning_steps,
        "source_targets": source_targets,
        "operator_targets": operator_targets,
        "operand_targets": operand_targets,
    }


def move(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def forward(model, batch):
    return model(
        batch["memory_targets"][:, 0],
        batch["source_targets"],
        batch["operator_targets"],
        batch["operand_targets"],
        batch["query_targets"],
    )


def evaluate(model, vocab, batch_size, seq_len, batches, device, seed):
    model.eval()
    totals = {depth: [0, 0] for depth in (2, 4, 6, 8)}
    memory_correct = 0
    memory_total = 0
    with torch.no_grad():
        for depth in totals:
            rng = random.Random(seed + depth)
            for _ in range(batches):
                batch = move(
                    sample_batch(
                        batch_size,
                        seq_len,
                        vocab,
                        (depth, depth),
                        model.config,
                        rng,
                    ),
                    device,
                )
                outputs = forward(model, batch)
                predictions = outputs["answer_logits"].argmax(dim=-1)
                totals[depth][0] += predictions.eq(batch["answers"]).sum().item()
                totals[depth][1] += batch["answers"].numel()
                memory_predictions = outputs["memory_logits"].argmax(dim=-1)
                expanded_mask = batch["memory_mask"][:, None, :].expand_as(batch["memory_targets"])
                memory_correct += memory_predictions[expanded_mask].eq(
                    batch["memory_targets"][expanded_mask]
                ).sum().item()
                memory_total += expanded_mask.sum().item()
    model.train()
    return totals, memory_correct / max(memory_total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/oracle-latent-memory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    vocab = Vocab()
    config = OracleLatentMemoryConfig(d_model=args.d_model, expand=args.expand)
    model = OracleLatentMemoryReasoner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        maximum_depth = depth_limit(step)
        batch = move(
            sample_batch(
                args.batch_size,
                args.seq_len,
                vocab,
                (2, maximum_depth),
                config,
                rng,
            ),
            device,
        )
        outputs = forward(model, batch)
        losses = model.loss(
            outputs,
            batch["memory_targets"],
            batch["memory_mask"],
            batch["answers"],
            memory_weight=args.memory_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            answer_accuracy = outputs["answer_logits"].argmax(-1).eq(batch["answers"]).float().mean()
            expanded_mask = batch["memory_mask"][:, None, :].expand_as(batch["memory_targets"])
            memory_accuracy = outputs["memory_logits"].argmax(-1)[expanded_mask].eq(
                batch["memory_targets"][expanded_mask]
            ).float().mean()
            print(
                f"{step:6d} loss={losses['loss'].item():.4f} "
                f"answer={losses['answer_loss'].item():.4f}/{answer_accuracy.item():.3f} "
                f"memory={losses['memory_loss'].item():.4f}/{memory_accuracy.item():.3f} "
                f"grad={float(grad_norm):.3f} depth=2-{maximum_depth}",
                flush=True,
            )

        if step > 0 and step % args.eval_interval == 0:
            totals, memory_accuracy = evaluate(
                model,
                vocab,
                args.batch_size,
                args.seq_len,
                args.eval_batches,
                device,
                args.seed + step,
            )
            depth_results = " ".join(
                f"d{depth}={correct / total:.3f}" for depth, (correct, total) in totals.items()
            )
            print(f"eval {depth_results} memory={memory_accuracy:.3f}", flush=True)
            torch.save(
                {"step": step, "config": config, "model": model.state_dict()},
                checkpoint_dir / f"oracle_latent_memory_step{step}.pt",
            )

    torch.save(
        {"step": args.steps, "config": config, "model": model.state_dict()},
        checkpoint_dir / "oracle_latent_memory_final.pt",
    )


if __name__ == "__main__":
    main()
