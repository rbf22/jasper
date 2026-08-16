import argparse
import random
from pathlib import Path

import torch

from data import MOD, Vocab, sample_task1_latent_batch
from latent_memory_model import LatentMemoryConfig, LatentMemoryReasoner


def depth_limit(step: int) -> int:
    if step < 1000:
        return 2
    if step < 2500:
        return 4
    return 8


def evaluate(
    model: LatentMemoryReasoner,
    vocab: Vocab,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: torch.device,
    seed: int,
):
    model.eval()
    totals = {depth: [0, 0] for depth in (2, 4, 6, 8)}
    memory_correct = 0
    memory_total = 0
    with torch.no_grad():
        for depth in totals:
            rng = random.Random(seed + depth)
            for _ in range(batches):
                inputs, mask, memory_targets, memory_mask, answers, _, _, _, _, _ = sample_task1_latent_batch(
                    batch_size,
                    seq_len,
                    vocab,
                    depth_range=(depth, depth),
                    max_reasoning_steps=model.config.max_reasoning_steps,
                    rng=rng,
                )
                outputs = model(inputs.to(device), mask.to(device))
                predictions = outputs["answer_logits"].argmax(dim=-1).cpu()
                totals[depth][0] += predictions.eq(answers).sum().item()
                totals[depth][1] += answers.numel()
                memory_predictions = outputs["memory_logits"].argmax(dim=-1).cpu()
                expanded_mask = memory_mask[:, None, :].expand_as(memory_targets)
                memory_correct += memory_predictions[expanded_mask].eq(memory_targets[expanded_mask]).sum().item()
                memory_total += expanded_mask.sum().item()
    model.train()
    return totals, memory_correct / max(memory_total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/latent-memory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    vocab = Vocab()
    config = LatentMemoryConfig(
        d_model=args.d_model,
        n_encoder_layers=args.encoder_layers,
        max_reasoning_steps=7,
    )
    model = LatentMemoryReasoner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        maximum_depth = depth_limit(step)
        (
            inputs,
            mask,
            memory_targets,
            memory_mask,
            answers,
            query_targets,
            _,
            source_targets,
            operator_targets,
            operand_targets,
        ) = sample_task1_latent_batch(
            args.batch_size,
            args.seq_len,
            vocab,
            depth_range=(2, maximum_depth),
            max_reasoning_steps=config.max_reasoning_steps,
            rng=rng,
        )
        outputs = model(inputs.to(device), mask.to(device))
        losses = model.loss(
            outputs,
            memory_targets.to(device),
            memory_mask.to(device),
            answers.to(device),
            query_targets.to(device),
            source_targets.to(device),
            operator_targets.to(device),
            operand_targets.to(device),
            memory_weight=args.memory_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            answer_accuracy = outputs["answer_logits"].argmax(-1).eq(answers.to(device)).float().mean()
            expanded_mask = memory_mask[:, None, :].expand_as(memory_targets).to(device)
            memory_predictions = outputs["memory_logits"].argmax(-1)
            memory_accuracy = memory_predictions[expanded_mask].eq(memory_targets.to(device)[expanded_mask]).float().mean()
            print(
                f"{step:6d} loss={losses['loss'].item():.4f} "
                f"answer={losses['answer_loss'].item():.4f}/{answer_accuracy.item():.3f} "
                f"memory={losses['memory_loss'].item():.4f}/{memory_accuracy.item():.3f} "
                f"routing={losses['routing_loss'].item():.4f} "
                f"metadata={losses['metadata_loss'].item():.4f} depth=2-{maximum_depth}",
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
                {"step": step, "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                checkpoint_dir / f"latent_memory_step{step}.pt",
            )

    torch.save(
        {"step": args.steps, "config": config, "model": model.state_dict()},
        checkpoint_dir / "latent_memory_final.pt",
    )


if __name__ == "__main__":
    main()
