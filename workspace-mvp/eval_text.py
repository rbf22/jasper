#!/usr/bin/env python3
"""Evaluate a text-trained checkpoint — perplexity and generation.

Usage:
    # Perplexity only
    TT_VISIBLE_DEVICES=1 python eval_text.py \
        --checkpoint checkpoints/cell_text_step500.pt \
        --config configs/text_cell_c.yaml --device 0

    # Generate text
    TT_VISIBLE_DEVICES=1 python eval_text.py \
        --checkpoint checkpoints/cell_text_step500.pt \
        --config configs/text_cell_c.yaml --device 0 \
        --generate --prompt "Once upon a time"
"""

import os
import sys
import argparse
import torch
import ttnn

MVP_DIR = os.path.dirname(os.path.abspath(__file__))
POC_DIR = os.path.realpath(os.path.join(MVP_DIR, "..", "workspace-poc"))
if not os.path.isdir(POC_DIR):
    POC_DIR = os.path.realpath(os.path.join(MVP_DIR, "..", "workspace-poc"))
sys.path.insert(0, POC_DIR)
sys.path.insert(0, MVP_DIR)

from train_ttnn import load_config, build_model_config, _safe_deallocate
from model_ttnn import TTWRAPModel
from text_data import BPETokenizer, TextDataset, make_eval_batches


def setup_mesh_graph():
    if "TT_MESH_GRAPH_DESC_PATH" in os.environ:
        return
    from pathlib import Path
    candidates = [
        "/home/rfenwick/tt-boltz/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto",
        "/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto",
    ]
    for c in candidates:
        if Path(c).is_file():
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = c
            return


def compute_perplexity(model, eval_batches, device, k_value=None):
    """Compute average perplexity on eval batches."""
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in eval_batches:
        with torch.no_grad():
            logits_tt = model.forward(input_ids, k_value=k_value)
            logits = ttnn.to_torch(logits_tt)
            # REVIEWED: Deallocate device logits after host transfer
            _safe_deallocate(logits_tt)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, :-1]
            valid_mask = (shift_labels != -100)
            n_valid = valid_mask.sum().item()
            if n_valid == 0:
                # REVIEWED: Still need to clear caches even if skipping loss
                model.clear_caches()
                continue
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
                shift_labels.clamp(min=0).reshape(-1),
                reduction="none",
            )
            loss = (loss * valid_mask.reshape(-1)).sum().item()
            total_loss += loss
            total_tokens += n_valid
            # REVIEWED: Clear model caches after eval forward (no backward ran)
            model.clear_caches()
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def generate_text(model, tokenizer, prompt, device, max_new_tokens=100,
                  k_value=None, temperature=1.0):
    """Generate text autoregressively from a prompt."""
    tokens = tokenizer.encode(prompt)
    generated = list(tokens)

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long)
        with torch.no_grad():
            logits_tt = model.forward(input_ids, k_value=k_value)
            logits = ttnn.to_torch(logits_tt)  # (1, T, V)
            # REVIEWED: Deallocate device logits after host transfer
            _safe_deallocate(logits_tt)

        # Get logits for last position
        next_logits = logits[0, -1, :].float() / temperature
        probs = torch.softmax(next_logits, dim=-1)

        # Sample
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)

        # REVIEWED: Clear model caches after each generation step
        model.clear_caches()

        # Stop at EOS
        if next_token == tokenizer.EOS:
            break

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Evaluate text-trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--generate", action="store_true", help="Generate text samples")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()

    setup_mesh_graph()

    cfg = load_config(args.config)
    model_config = build_model_config(cfg)

    device = ttnn.open_device(device_id=args.device)
    model = TTWRAPModel(model_config, device)

    # Load checkpoint
    model.load_checkpoint(args.checkpoint, device=device)
    print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    tokenizer = BPETokenizer()

    # Perplexity
    valid_path = cfg.get("valid_data", "data/tinystories_valid.txt")
    valid_dataset = TextDataset(valid_path, tokenizer)
    seq_len = cfg.get("seq_len", 512)
    eval_batches = make_eval_batches(valid_dataset, seq_len=seq_len, n_batches=10, batch_size=8)

    k_value = model_config.k_inference if model_config.recurrent_core else None
    avg_loss, ppl = compute_perplexity(model, eval_batches, device, k_value=k_value)
    print(f"\nPerplexity: {ppl:.2f} (loss: {avg_loss:.4f})", flush=True)

    # Generation
    if args.generate:
        print(f"\n--- Generation samples ---", flush=True)
        for i in range(args.n_samples):
            text = generate_text(
                model, tokenizer, args.prompt, device,
                max_new_tokens=args.max_tokens, k_value=k_value,
                temperature=args.temperature,
            )
            print(f"\nSample {i+1}:", flush=True)
            print(text, flush=True)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
