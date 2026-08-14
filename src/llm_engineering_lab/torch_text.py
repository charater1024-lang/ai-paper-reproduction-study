"""PyTorch building blocks for a small, production-shaped text classifier.

The module deliberately keeps preprocessing explicit: a regular-expression
tokenizer, a serializable vocabulary, a Dataset/Collator pair, and plain
training loops.  They are small enough to step through in a debugger while
still mirroring the pieces used in larger LLM systems.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .acceleration import resolve_device

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
_TOKEN_PATTERN = re.compile(
    # 작은 한국어 데이터에서는 어절 전체가 대부분 OOV가 되므로 한글은 음절 단위로
    # 나눈다. 영어 단어와 숫자는 의미 있는 덩어리로 유지해 혼합 입력도 관찰한다.
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?|[가-힣]|[^\w\s]",
    flags=re.UNICODE,
)


def seed_everything(seed: int = 42) -> None:
    """Seed Python and PyTorch without forcing slow deterministic kernels."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True, slots=True)
class RegexTokenizer:
    """A transparent tokenizer suitable for Korean/English support messages."""

    lowercase: bool = True

    def __call__(self, text: str) -> list[str]:
        normalized = text.strip()
        if self.lowercase:
            normalized = normalized.lower()
        return _TOKEN_PATTERN.findall(normalized)


class Vocabulary:
    """Token/id mapping with stable special-token ids and JSON serialization."""

    def __init__(self, tokens: Sequence[str], *, lowercase: bool = True) -> None:
        ordered: list[str] = []
        seen: set[str] = set()
        for token in (*SPECIAL_TOKENS, *tokens):
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        self._id_to_token = ordered
        self._token_to_id = {token: index for index, token in enumerate(ordered)}
        self.tokenizer = RegexTokenizer(lowercase=lowercase)

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        min_frequency: int = 1,
        max_size: int | None = None,
        lowercase: bool = True,
    ) -> Vocabulary:
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if max_size is not None and max_size < len(SPECIAL_TOKENS):
            raise ValueError(f"max_size must be >= {len(SPECIAL_TOKENS)}")

        tokenizer = RegexTokenizer(lowercase=lowercase)
        counts = Counter(token for text in texts for token in tokenizer(text))
        # Frequency first, then lexical order: repeated runs create identical ids.
        candidates = sorted(counts, key=lambda token: (-counts[token], token))
        candidates = [token for token in candidates if counts[token] >= min_frequency]
        if max_size is not None:
            candidates = candidates[: max_size - len(SPECIAL_TOKENS)]
        return cls(candidates, lowercase=lowercase)

    def __len__(self) -> int:
        return len(self._id_to_token)

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<eos>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        ids = [
            self._token_to_id.get(token, self.unk_id) for token in self.tokenizer(text)
        ]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None:
            if max_length < 1:
                raise ValueError("max_length must be positive")
            ids = ids[:max_length]
            # Preserve an explicitly requested EOS marker after truncation.
            if add_eos and ids:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        specials = set(SPECIAL_TOKENS)
        for token_id in ids:
            if not 0 <= int(token_id) < len(self):
                token = "<unk>"
            else:
                token = self._id_to_token[int(token_id)]
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self._id_to_token,
            "lowercase": self.tokenizer.lowercase,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Vocabulary:
        tokens = [str(token) for token in payload["tokens"]]
        return cls(tokens, lowercase=bool(payload.get("lowercase", True)))

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> Vocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class TicketExample:
    text: str
    label: str
    group_id: str | None = None


def _find_column(
    fieldnames: Sequence[str], requested: str | None, candidates: Sequence[str]
) -> str:
    by_lower = {name.strip().lower(): name for name in fieldnames}
    if requested is not None:
        match = by_lower.get(requested.strip().lower())
        if match is None:
            raise ValueError(
                f"CSV column {requested!r} not found; available={list(fieldnames)}"
            )
        return match
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError(
        f"Could not infer a column from {list(fieldnames)}; tried={list(candidates)}"
    )


def load_ticket_examples(
    csv_path: str | Path,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
) -> list[TicketExample]:
    """Read common ticket CSV schemas without depending on pandas or ``data.py``."""

    source = Path(csv_path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {source}")
        text_key = _find_column(
            reader.fieldnames,
            text_column,
            ("text", "ticket_text", "message", "description", "content", "query"),
        )
        label_key = _find_column(
            reader.fieldnames,
            label_column,
            ("label", "category", "intent", "target", "issue_type"),
        )
        group_key = next(
            (
                name
                for name in reader.fieldnames
                if name.strip().lower()
                in {"case_id", "group_id", "conversation_id", "thread_id"}
            ),
            None,
        )
        examples = [
            TicketExample(
                text=(row.get(text_key) or "").strip(),
                label=(row.get(label_key) or "").strip(),
                group_id=(row.get(group_key) or "").strip() or None
                if group_key
                else None,
            )
            for row in reader
        ]
    examples = [example for example in examples if example.text and example.label]
    if not examples:
        raise ValueError(f"No non-empty text/label rows found in {source}")
    return examples


def split_examples(
    examples: Sequence[TicketExample],
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[TicketExample], list[TicketExample]]:
    """Make a deterministic, approximately stratified train/validation split."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(examples) < 2:
        raise ValueError("At least two examples are required for a split")

    generator = random.Random(seed)
    # Keep duplicated messages from one customer case on the same side of the
    # split.  Without this, validation metrics can be deceptively optimistic.
    case_groups: dict[str, list[TicketExample]] = defaultdict(list)
    for index, example in enumerate(examples):
        key = example.group_id or f"__ungrouped_{index}"
        case_groups[key].append(example)

    grouped: dict[str, list[list[TicketExample]]] = defaultdict(list)
    for group_id, group in case_groups.items():
        labels = {example.label for example in group}
        if len(labels) != 1:
            raise ValueError(f"All examples in group {group_id!r} must share a label")
        grouped[next(iter(labels))].append(group)

    train: list[TicketExample] = []
    validation: list[TicketExample] = []
    for label in sorted(grouped):
        label_groups = list(grouped[label])
        generator.shuffle(label_groups)
        valid_count = (
            0
            if len(label_groups) == 1
            else max(1, round(len(label_groups) * validation_ratio))
        )
        valid_count = min(valid_count, len(label_groups) - 1)
        validation.extend(
            example for group in label_groups[:valid_count] for example in group
        )
        train.extend(
            example for group in label_groups[valid_count:] for example in group
        )

    if not validation:
        if len(case_groups) < 2:
            raise ValueError(
                "At least two independent groups are required for a group-safe split"
            )
        # If every label has only one case, a stratified split is impossible.
        # Move one complete case rather than leaking near-duplicate rows.
        selected_group = min(case_groups.values(), key=len)
        selected_ids = {id(example) for example in selected_group}
        validation.extend(selected_group)
        train = [example for example in train if id(example) not in selected_ids]
    generator.shuffle(train)
    generator.shuffle(validation)
    return train, validation


class TextClassificationDataset(Dataset[dict[str, Tensor]]):
    """Encode examples lazily enough to inspect, eagerly enough to train quickly."""

    def __init__(
        self,
        examples: Sequence[TicketExample],
        vocabulary: Vocabulary,
        label_to_id: Mapping[str, int],
        *,
        max_length: int = 64,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS/EOS")
        self.examples = list(examples)
        self.vocabulary = vocabulary
        self.label_to_id = dict(label_to_id)
        self.max_length = max_length
        missing = sorted(
            {example.label for example in self.examples} - self.label_to_id.keys()
        )
        if missing:
            raise ValueError(f"Labels missing from label_to_id: {missing}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self.examples[index]
        token_ids = self.vocabulary.encode(
            example.text,
            add_bos=True,
            add_eos=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(self.label_to_id[example.label], dtype=torch.long),
        }


@dataclass(frozen=True, slots=True)
class ClassificationCollator:
    """Pad a variable-length batch and produce a 1/0 attention mask."""

    pad_id: int

    def __call__(self, items: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
        if not items:
            raise ValueError("Cannot collate an empty batch")
        input_ids = pad_sequence(
            [item["input_ids"] for item in items],
            batch_first=True,
            padding_value=self.pad_id,
        )
        labels = torch.stack([item["labels"] for item in items])
        attention_mask = input_ids.ne(self.pad_id)
        assert input_ids.shape == attention_mask.shape
        assert labels.shape == (len(items),)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def make_dataloader(
    dataset: TextClassificationDataset,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader[dict[str, Tensor]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ClassificationCollator(dataset.vocabulary.pad_id),
        generator=generator,
        num_workers=0,
    )


@dataclass(frozen=True, slots=True)
class TextClassifierConfig:
    vocab_size: int
    num_classes: int
    pad_id: int
    embedding_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.15

    def __post_init__(self) -> None:
        if self.vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("vocab_size is unexpectedly small")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class MeanPoolTextClassifier(nn.Module):
    """Embedding -> masked mean pooling -> MLP classifier."""

    def __init__(self, config: TextClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_id,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, time], got {input_ids.shape}"
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_id)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask and input_ids must have the same shape")
        embeddings = self.embedding(input_ids)
        weights = attention_mask.to(embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        assert logits.shape == (input_ids.shape[0], self.config.num_classes)
        return logits


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _metrics_from_predictions(
    *,
    total_loss: float,
    predictions: Sequence[int],
    targets: Sequence[int],
    num_classes: int,
) -> ClassificationMetrics:
    if not targets:
        raise ValueError("Cannot compute metrics for an empty loader")
    pairs = list(zip(predictions, targets, strict=True))
    correct = sum(prediction == target for prediction, target in pairs)
    f1_scores: list[float] = []
    for class_id in range(num_classes):
        true_positive = sum(p == class_id and y == class_id for p, y in pairs)
        false_positive = sum(p == class_id and y != class_id for p, y in pairs)
        false_negative = sum(p != class_id and y == class_id for p, y in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return ClassificationMetrics(
        loss=total_loss / len(targets),
        accuracy=correct / len(targets),
        macro_f1=sum(f1_scores) / num_classes,
    )


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_classifier_epoch(
    model: MeanPoolTextClassifier,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
) -> ClassificationMetrics:
    """Run one training epoch, including gradient clipping and macro-F1."""

    model.train()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_size = batch["labels"].numel()
        total_loss += loss.item() * batch_size
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        targets.extend(batch["labels"].detach().cpu().tolist())
    return _metrics_from_predictions(
        total_loss=total_loss,
        predictions=predictions,
        targets=targets,
        num_classes=model.config.num_classes,
    )


@torch.inference_mode()
def evaluate_classifier(
    model: MeanPoolTextClassifier,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
) -> ClassificationMetrics:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"])
        batch_size = batch["labels"].numel()
        total_loss += loss.item() * batch_size
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(batch["labels"].cpu().tolist())
    return _metrics_from_predictions(
        total_loss=total_loss,
        predictions=predictions,
        targets=targets,
        num_classes=model.config.num_classes,
    )


@dataclass(frozen=True, slots=True)
class ClassifierTrainingConfig:
    epochs: int = 6
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0


def fit_classifier(
    model: MeanPoolTextClassifier,
    train_loader: DataLoader[dict[str, Tensor]],
    validation_loader: DataLoader[dict[str, Tensor]],
    *,
    config: ClassifierTrainingConfig | None = None,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train and save the checkpoint with the best validation macro-F1."""

    config = config or ClassifierTrainingConfig()
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    selected_device = device or resolve_device()
    model.to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_f1 = float("-inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            device=selected_device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_metrics = evaluate_classifier(
            model, validation_loader, device=selected_device
        )
        record = {
            "epoch": epoch,
            "train": train_metrics.to_dict(),
            "validation": validation_metrics.to_dict(),
        }
        history.append(record)

        if checkpoint_path is not None and validation_metrics.macro_f1 > best_f1:
            best_f1 = validation_metrics.macro_f1
            destination = Path(checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model.config),
                    "training_config": asdict(config),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": record,
                    "extra": dict(checkpoint_extra or {}),
                },
                destination,
            )
    return history


def load_classifier_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MeanPoolTextClassifier, dict[str, Any]]:
    """Restore a classifier and return the untouched checkpoint metadata."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0 does not expose ``weights_only``.
        payload = torch.load(path, map_location=map_location)
    model = MeanPoolTextClassifier(TextClassifierConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model, payload


# TODO(연습): mean pooling을 1D-CNN 또는 양방향 GRU로 교체하고 같은 split에서 비교하세요.
# TODO(현업): label별 precision/recall과 confusion matrix를 checkpoint metadata에 추가하세요.
