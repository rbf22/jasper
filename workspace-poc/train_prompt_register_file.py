import argparse
import random
from pathlib import Path

import torch

from data import Vocab, sample_task1_latent_batch
from latent_memory_model import PromptRegisterFileConfig, PromptRegisterFileReasoner


def depth_limit(step: int, max_depth: int = 8) -> int:
    if step < 500:
        return 2
    if step < 1500:
        return 4
    if step < 3000:
        return 6
    return max_depth


def sample_batch(batch_size, seq_len, vocab, depth_range, config, rng):
    batch = sample_task1_latent_batch(
        batch_size, seq_len, vocab,
        depth_range=depth_range,
        max_reasoning_steps=config.max_reasoning_steps,
        rng=rng,
    )
    inputs, mask, mem_targets, mem_mask, answers, query_targets, \
        _, source_t, op_t, operand_t = batch
    return {
        "inputs": inputs, "mask": mask,
        "memory_targets": mem_targets, "memory_mask": mem_mask,
        "answers": answers, "query_targets": query_targets,
        "source_targets": source_t, "operator_targets": op_t,
        "operand_targets": operand_t,
    }


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def forward(model, batch):
    return model(batch["inputs"], batch["mask"])


def evaluate(model, vocab, batch_size, seq_len, batches, device, seed):
    model.eval()
    totals = {depth: [0, 0] for depth in (2, 4, 6, 8)}
    parse_acc = {"source": [0, 0], "operator": [0, 0], "operand": [0, 0], "query": [0, 0]}
    with torch.no_grad():
        for depth in totals:
            rng = random.Random(seed + depth)
            for _ in range(batches):
                batch = move(
                    sample_batch(batch_size, seq_len, vocab, (depth, depth), model.config, rng),
                    device,
                )
                outputs = forward(model, batch)
                preds = outputs["answer_logits"].argmax(dim=-1)
                totals[depth][0] += preds.eq(batch["answers"]).sum().item()
                totals[depth][1] += batch["answers"].numel()

                mask = batch["memory_mask"]
                parse_acc["source"][0] += outputs["source_logits"][mask].argmax(-1).eq(
                    batch["source_targets"][mask]
                ).sum().item()
                parse_acc["operator"][0] += outputs["operator_logits"][mask].argmax(-1).eq(
                    batch["operator_targets"][mask]
                ).sum().item()
                parse_acc["operand"][0] += outputs["operand_logits"][mask].argmax(-1).eq(
                    batch["operand_targets"][mask]
                ).sum().item()
                parse_acc["query"][0] += outputs["query_logits"].argmax(-1).eq(
                    batch["query_targets"]
                ).sum().item()
                parse_acc["source"][1] += mask.sum().item()
                parse_acc["operator"][1] += mask.sum().item()
                parse_acc["operand"][1] += mask.sum().item()
                parse_acc["query"][1] += batch["query_targets"].numel()
    model.train()
    return totals, {k: v[0] / max(v[1], 1) for k, v in parse_acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-encoder-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--parse-weight", type=float, default=0.5)
    parser.add_argument("--query-weight", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/prompt-register-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-detach", action="store_true")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-reasoning-steps", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    vocab = Vocab()
    config = PromptRegisterFileConfig(
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_heads=args.n_heads,
        expand=args.expand,
        max_reasoning_steps=args.max_reasoning_steps,
        detach_latent_steps=not args.no_detach,
    )
    model = PromptRegisterFileReasoner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        max_d = depth_limit(step, args.max_depth)
        batch = move(
            sample_batch(args.batch_size, args.seq_len, vocab, (2, max_d), config, rng),
            device,
        )
        outputs = forward(model, batch)
        losses = model.loss(
            outputs,
            batch["memory_targets"], batch["memory_mask"],
            batch["answers"], batch["query_targets"],
            batch["source_targets"], batch["operator_targets"], batch["operand_targets"],
            memory_weight=args.memory_weight,
            parse_weight=args.parse_weight,
            query_weight=args.query_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            ans_acc = outputs["answer_logits"].argmax(-1).eq(batch["answers"]).float().mean()
            mask = batch["memory_mask"]
            src_acc = outputs["source_logits"][mask].argmax(-1).eq(batch["source_targets"][mask]).float().mean()
            op_acc = outputs["operator_logits"][mask].argmax(-1).eq(batch["operator_targets"][mask]).float().mean()
            q_acc = outputs["query_logits"].argmax(-1).eq(batch["query_targets"]).float().mean()
            print(
                f"{step:6d} loss={losses['loss'].item():.4f} "
                f"ans={losses['answer_loss'].item():.3f}/{ans_acc.item():.3f} "
                f"src={losses['source_loss'].item():.3f}/{src_acc.item():.3f} "
                f"op={losses['operator_loss'].item():.3f}/{op_acc.item():.3f} "
                f"q={losses['query_loss'].item():.3f}/{q_acc.item():.3f} "
                f"grad={float(grad_norm):.3f} d=2-{max_d}",
                flush=True,
            )

        if step > 0 and step % args.eval_interval == 0:
            totals, parse = evaluate(
                model, vocab, args.batch_size, args.seq_len,
                args.eval_batches, device, args.seed + step,
            )
            depth_str = " ".join(f"d{d}={c/t:.3f}" for d, (c, t) in totals.items())
            parse_str = f"src={parse['source']:.3f} op={parse['operator']:.3f} q={parse['query']:.3f}"
            print(f"eval {depth_str} {parse_str}", flush=True)
            torch.save(
                {"step": step, "config": config, "model": model.state_dict()},
                checkpoint_dir / f"prompt_rf_step{step}.pt",
            )

    torch.save(
        {"step": args.steps, "config": config, "model": model.state_dict()},
        checkpoint_dir / "prompt_rf_final.pt",
    )


if __name__ == "__main__":
    main()
