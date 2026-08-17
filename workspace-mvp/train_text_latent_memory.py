#!/usr/bin/env python3
"""Train the text latent-memory model on tiny challenges.

CPU-runnable for quick validation. No Tenstorrent hardware required.

Usage:
    # Quick test (100 steps)
    /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_text_latent_memory.py \
        --steps 100 --batch-size 32 --d-model 128 --eval-interval 50

    # Full training
    /home/rfenwick/Documents/jasper/.tt-venv/bin/python train_text_latent_memory.py \
        --steps 10000 --batch-size 64 --d-model 256 --eval-interval 500
"""

import argparse
import random
import time
from pathlib import Path

import torch

from text_latent_memory_model import TextLatentMemoryConfig, TextLatentMemoryModel
from challenge_data import ChallengeDataset, exact_match_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--max-reasoning-steps", type=int, default=6)
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-answer-len", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/text_latent_memory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-reg", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    # Load data
    data_dir = Path(__file__).parent / "data"
    dataset = ChallengeDataset(
        train_path=str(data_dir / "tiny_challenges_train.txt"),
        valid_path=str(data_dir / "tiny_challenges_valid.txt"),
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
    )

    config = TextLatentMemoryConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        n_slots=args.n_slots,
        max_reasoning_steps=args.max_reasoning_steps,
        expand=args.expand,
        max_answer_len=args.max_answer_len,
        pad_token_id=dataset.pad_id,
    )
    model = TextLatentMemoryModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / args.warmup_steps
        return args.lr * (1.0 - step / args.steps)

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        eval_rng = random.Random(args.seed + 99999)
        with torch.no_grad():
            for _ in range(args.eval_batches):
                prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
                    dataset.sample_batch(args.eval_batch_size, "valid", eval_rng)
                prompt_ids = prompt_ids.to(device)
                prompt_mask = prompt_mask.to(device)
                dec_input = dec_input.to(device)
                ans_targets = ans_targets.to(device)
                ans_mask = ans_mask.to(device)

                outputs = model(prompt_ids, prompt_mask, dec_input, ans_mask)
                losses = model.loss(outputs, ans_targets, ans_mask, memory_reg_weight=0)
                total_loss += losses["answer_loss"].item() * ans_mask.float().sum().item()

                # Generate for exact match
                generated = model.generate(
                    prompt_ids, prompt_mask,
                    max_new_tokens=args.max_answer_len,
                    bos_token_id=dataset.bos_id,
                    eos_token_id=dataset.eos_id,
                )
                # Pad generated to match targets length
                gen_len = generated.shape[1]
                tgt_len = ans_targets.shape[1]
                if gen_len < tgt_len:
                    generated = torch.cat([
                        generated,
                        torch.full((generated.shape[0], tgt_len - gen_len), dataset.pad_id,
                                   dtype=generated.dtype, device=device),
                    ], dim=1)
                elif gen_len > tgt_len:
                    generated = generated[:, :tgt_len]

                total_correct += int(exact_match_accuracy(
                    generated, ans_targets, dataset.eos_id, dataset.pad_id,
                ) * args.eval_batch_size)
                total_examples += args.eval_batch_size

        model.train()
        avg_loss = total_loss / max(total_examples * args.max_answer_len, 1)
        accuracy = total_correct / total_examples
        return avg_loss, accuracy

    # Training loop
    start_time = time.time()
    for step in range(args.steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        prompt_ids, prompt_mask, dec_input, ans_targets, ans_mask = \
            dataset.sample_batch(args.batch_size, "train", rng)
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        dec_input = dec_input.to(device)
        ans_targets = ans_targets.to(device)
        ans_mask = ans_mask.to(device)

        outputs = model(prompt_ids, prompt_mask, dec_input, ans_mask)
        losses = model.loss(outputs, ans_targets, ans_mask, memory_reg_weight=args.memory_reg)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            ans_acc = (outputs["logits"].argmax(-1) == ans_targets).float() * ans_mask.float()
            token_acc = ans_acc.sum() / ans_mask.float().sum().clamp_min(1)
            print(
                f"{step:6d} loss={losses['loss'].item():.4f} "
                f"tok_acc={token_acc.item():.3f} "
                f"grad={float(grad_norm):.3f} "
                f"lr={lr:.2e} "
                f"t={elapsed:.0f}s",
                flush=True,
            )

        if step > 0 and step % args.eval_interval == 0:
            val_loss, val_acc = evaluate()
            print(f"eval step={step} val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)
            torch.save(
                {"step": step, "config": config, "model": model.state_dict()},
                checkpoint_dir / f"text_lm_step{step}.pt",
            )

    # Final eval
    val_loss, val_acc = evaluate()
    print(f"final eval val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)
    torch.save(
        {"step": args.steps, "config": config, "model": model.state_dict()},
        checkpoint_dir / "text_lm_final.pt",
    )


if __name__ == "__main__":
    main()
