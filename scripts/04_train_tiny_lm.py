"""Train and sample from a tiny decoder-only Transformer on ticket text.

This educational model is deliberately tiny and is not expected to produce a
useful support agent.  Its purpose is to make causal masking, next-token loss,
overfitting, sampling, and checkpoint metadata observable on a CPU laptop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.torch_text import (  # noqa: E402
    Vocabulary,
    load_ticket_examples,
    resolve_device,
    seed_everything,
    split_examples,
)
from llm_engineering_lab.transformer import (  # noqa: E402
    LanguageModelDataset,
    LMTrainingConfig,
    TinyCausalTransformer,
    TinyTransformerConfig,
    build_lm_token_stream,
    fit_tiny_lm,
    load_tiny_lm_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=ROOT / "data" / "customer_support_tickets.csv"
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "artifacts" / "tiny_lm.pt"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=6000)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feed-forward-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="결제가 두 번 처리되어 환불을 요청합니다")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size > args.context_length:
        raise ValueError("--block-size cannot exceed --context-length")
    if args.max_new_tokens >= args.context_length:
        raise ValueError("--max-new-tokens must be smaller than --context-length")
    seed_everything(args.seed)
    device = resolve_device(args.device)

    examples = load_ticket_examples(
        args.csv,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    train_examples, validation_examples = split_examples(
        examples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    vocabulary = Vocabulary.build(
        (example.text for example in train_examples),
        min_frequency=args.min_frequency,
        max_size=args.max_vocab_size,
    )
    train_stream = build_lm_token_stream(
        (example.text for example in train_examples), vocabulary
    )
    validation_stream = build_lm_token_stream(
        (example.text for example in validation_examples), vocabulary
    )
    train_dataset = LanguageModelDataset(
        train_stream,
        block_size=args.block_size,
        pad_id=vocabulary.pad_id,
    )
    validation_dataset = LanguageModelDataset(
        validation_stream,
        block_size=args.block_size,
        pad_id=vocabulary.pad_id,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TinyCausalTransformer(
        TinyTransformerConfig(
            vocab_size=len(vocabulary),
            pad_id=vocabulary.pad_id,
            max_seq_length=args.context_length,
            model_dim=args.model_dim,
            num_heads=args.heads,
            num_layers=args.layers,
            feed_forward_dim=args.feed_forward_dim,
            dropout=args.dropout,
        )
    )
    training_config = LMTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
    )
    print(
        f"device={device} train_windows={len(train_dataset)} "
        f"validation_windows={len(validation_dataset)} vocab={len(vocabulary)} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    history = fit_tiny_lm(
        model,
        train_loader,
        validation_loader=validation_loader,
        config=training_config,
        device=device,
        checkpoint_path=args.checkpoint,
        checkpoint_extra={
            "vocabulary": vocabulary.to_dict(),
            "source_csv": str(args.csv),
            "block_size": args.block_size,
            "seed": args.seed,
        },
    )
    for record in history:
        train = record["train"]
        valid = record["validation"]
        print(
            f"epoch={record['epoch']:02d} train_loss={train['loss']:.4f} "
            f"train_ppl={train['perplexity']:.2f} valid_loss={valid['loss']:.4f} "
            f"valid_ppl={valid['perplexity']:.2f}"
        )
    print(f"best checkpoint: {args.checkpoint}")

    best_model, payload = load_tiny_lm_checkpoint(args.checkpoint, map_location=device)
    best_model.to(device)
    saved_vocabulary = Vocabulary.from_dict(payload["extra"]["vocabulary"])
    prompt_limit = best_model.config.max_seq_length - args.max_new_tokens
    prompt_ids = saved_vocabulary.encode(
        args.prompt, add_bos=True, max_length=prompt_limit
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = best_model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=saved_vocabulary.eos_id,
        use_cache=True,
    )
    print(f"prompt: {args.prompt}")
    print(
        f"generated (학습용 tiny LM): {saved_vocabulary.decode(generated[0].tolist())}"
    )


if __name__ == "__main__":
    main()


# TODO(연습): block_size/stride를 바꾸고 validation perplexity와 속도를 기록하세요.
# TODO(현업): data provenance와 git commit hash를 checkpoint metadata에 저장하세요.
