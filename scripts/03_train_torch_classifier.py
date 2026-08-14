"""Train a fast PyTorch customer-support ticket classifier.

Run from the project root::

    python scripts/03_train_torch_classifier.py --epochs 6 --device auto

The saved checkpoint includes vocabulary and label metadata, not just weights.
This makes the artifact independently usable by a later inference service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.torch_text import (  # noqa: E402
    ClassifierTrainingConfig,
    MeanPoolTextClassifier,
    TextClassificationDataset,
    TextClassifierConfig,
    Vocabulary,
    fit_classifier,
    load_classifier_checkpoint,
    load_ticket_examples,
    make_dataloader,
    resolve_device,
    seed_everything,
    split_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=ROOT / "data" / "customer_support_tickets.csv"
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "artifacts" / "ticket_classifier.pt"
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=8000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predict",
        action="append",
        default=[],
        help="Classify this text after training; may be supplied more than once.",
    )
    return parser.parse_args()


@torch.inference_mode()
def predict_texts(
    model: MeanPoolTextClassifier,
    texts: list[str],
    vocabulary: Vocabulary,
    id_to_label: dict[int, str],
    *,
    max_length: int,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    encoded = [
        vocabulary.encode(text, add_bos=True, add_eos=True, max_length=max_length)
        for text in texts
    ]
    width = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(texts), width), vocabulary.pad_id, dtype=torch.long)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids)
    attention_mask = input_ids.ne(vocabulary.pad_id)
    probabilities = (
        model(input_ids.to(device), attention_mask.to(device)).softmax(dim=-1).cpu()
    )
    results: list[dict[str, object]] = []
    for text, scores in zip(texts, probabilities, strict=True):
        class_id = int(scores.argmax())
        results.append(
            {
                "text": text,
                "prediction": id_to_label[class_id],
                "confidence": round(float(scores[class_id]), 4),
            }
        )
    return results


def main() -> None:
    args = parse_args()
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
    labels = sorted({example.label for example in examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    vocabulary = Vocabulary.build(
        (example.text for example in train_examples),
        min_frequency=args.min_frequency,
        max_size=args.max_vocab_size,
    )

    train_dataset = TextClassificationDataset(
        train_examples,
        vocabulary,
        label_to_id,
        max_length=args.max_length,
    )
    validation_dataset = TextClassificationDataset(
        validation_examples,
        vocabulary,
        label_to_id,
        max_length=args.max_length,
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = make_dataloader(validation_dataset, batch_size=args.batch_size)

    model = MeanPoolTextClassifier(
        TextClassifierConfig(
            vocab_size=len(vocabulary),
            num_classes=len(labels),
            pad_id=vocabulary.pad_id,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    )
    training_config = ClassifierTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
    )
    print(
        f"device={device} rows={len(examples)} train={len(train_dataset)} "
        f"validation={len(validation_dataset)} vocab={len(vocabulary)} labels={labels}"
    )
    history = fit_classifier(
        model,
        train_loader,
        validation_loader,
        config=training_config,
        device=device,
        checkpoint_path=args.checkpoint,
        checkpoint_extra={
            "vocabulary": vocabulary.to_dict(),
            "label_to_id": label_to_id,
            "max_length": args.max_length,
            "source_csv": str(args.csv),
            "seed": args.seed,
        },
    )
    for record in history:
        train = record["train"]
        valid = record["validation"]
        print(
            f"epoch={record['epoch']:02d} "
            f"train_loss={train['loss']:.4f} train_f1={train['macro_f1']:.3f} "
            f"valid_loss={valid['loss']:.4f} valid_f1={valid['macro_f1']:.3f}"
        )
    print(f"best checkpoint: {args.checkpoint}")

    if args.predict:
        best_model, payload = load_classifier_checkpoint(
            args.checkpoint, map_location=device
        )
        best_model.to(device)
        checkpoint_extra = payload["extra"]
        saved_vocabulary = Vocabulary.from_dict(checkpoint_extra["vocabulary"])
        id_to_label = {
            int(class_id): label
            for label, class_id in checkpoint_extra["label_to_id"].items()
        }
        predictions = predict_texts(
            best_model,
            args.predict,
            saved_vocabulary,
            id_to_label,
            max_length=int(checkpoint_extra["max_length"]),
            device=device,
        )
        print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# TODO(연습): class-weight 또는 Focal Loss를 추가해 불균형 데이터에서 비교하세요.
# TODO(현업): validation threshold와 calibration error를 추적해 confidence를 검증하세요.
