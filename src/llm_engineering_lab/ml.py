"""A production-shaped scikit-learn baseline for ticket routing.

The module keeps preprocessing inside one ``Pipeline``.  Consequently TF-IDF
vocabulary and category encoders are fitted on training data only, and the same
transformations are automatically reused after model persistence.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import (
    DatasetSplit,
    TicketSchema,
    stratified_group_split,
    validate_ticket_frame,
)

MODEL_FORMAT_VERSION = 1
_WHITESPACE_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_LONG_NUMBER_RE = re.compile(r"(?<!\w)\d{5,}(?!\w)")


def normalize_ticket_text(text: str) -> str:
    """Normalize harmless variation and mask common high-cardinality identifiers."""

    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    normalized = _EMAIL_RE.sub(" <email> ", normalized)
    normalized = _LONG_NUMBER_RE.sub(" <number> ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and reproducibility controls for the baseline."""

    seed: int = 42
    validation_size: float = 0.2
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (2, 5)
    min_df: int = 1
    regularization_c: float = 4.0
    max_iter: int = 1_000
    class_weight: str | None = "balanced"

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1")
        if self.max_features <= 0 or self.min_df <= 0 or self.max_iter <= 0:
            raise ValueError("feature and iteration limits must be positive")
        if self.ngram_range[0] <= 0 or self.ngram_range[0] > self.ngram_range[1]:
            raise ValueError("ngram_range must contain increasing positive values")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Metrics that retain label order for a readable confusion matrix."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    labels: tuple[str, ...]
    per_class: dict[str, ClassMetrics]
    confusion_matrix: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["labels"] = list(self.labels)
        result["confusion_matrix"] = [list(row) for row in self.confusion_matrix]
        return result


@dataclass(slots=True)
class TrainingResult:
    """Model plus the exact holdout evidence used to judge it."""

    model: Pipeline
    report: EvaluationReport
    errors: pd.DataFrame
    split: DatasetSplit = field(repr=False)

    @property
    def train_rows(self) -> int:
        return len(self.split.train)

    @property
    def validation_rows(self) -> int:
        return len(self.split.validation)


class TicketClassifierTrainer:
    """Orchestrate leakage-safe splitting, fitting, and error analysis."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        schema: TicketSchema | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.schema = schema or TicketSchema()

    def train(self, frame: pd.DataFrame) -> TrainingResult:
        clean = validate_ticket_frame(frame, self.schema, expected_labels=None)
        split = stratified_group_split(
            clean,
            validation_size=self.config.validation_size,
            seed=self.config.seed,
            schema=self.schema,
        )
        model = build_model(self.config, self.schema)
        train_features = split.train.loc[:, self.schema.model_feature_columns]
        model.fit(train_features, split.train[self.schema.label])

        validation_features = split.validation.loc[:, self.schema.model_feature_columns]
        report = evaluate_model(
            model, validation_features, split.validation[self.schema.label]
        )
        errors = build_error_table(model, split.validation, self.schema)
        return TrainingResult(model=model, report=report, errors=errors, split=split)


def build_model(
    config: TrainingConfig | None = None,
    schema: TicketSchema | None = None,
) -> Pipeline:
    """Build an unfitted TF-IDF + categorical metadata + logistic model."""

    config = config or TrainingConfig()
    schema = schema or TicketSchema()
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "text_tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_ticket_text,
                    analyzer="char_wb",
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
                schema.text,
            ),
            (
                "metadata",
                OneHotEncoder(handle_unknown="ignore"),
                [schema.channel, schema.priority, schema.customer_tier],
            ),
        ],
        sparse_threshold=1.0,
        # One-hot metadata otherwise has a larger vector norm than the text.
        # A small weight keeps it useful without letting channel dominate intent.
        transformer_weights={"text_tfidf": 1.0, "metadata": 0.05},
    )
    classifier = LogisticRegression(
        C=config.regularization_c,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        random_state=config.seed,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame | Sequence[str],
    labels: Sequence[str] | pd.Series,
) -> EvaluationReport:
    """Evaluate a fitted model with class-aware metrics, not accuracy alone."""

    feature_frame = _coerce_feature_frame(features)
    y_true = pd.Series(labels, dtype="string").reset_index(drop=True)
    if len(feature_frame) != len(y_true):
        raise ValueError("features and labels must have the same number of rows")
    y_pred = model.predict(feature_frame)
    label_order = tuple(sorted(set(y_true.tolist()) | set(map(str, y_pred))))
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(label_order),
        zero_division=0,
    )
    per_class = {
        label: ClassMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(per_class_f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(label_order)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=list(label_order))
    return EvaluationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        labels=label_order,
        per_class=per_class,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
    )


def build_error_table(
    model: Pipeline,
    validation_frame: pd.DataFrame,
    schema: TicketSchema | None = None,
) -> pd.DataFrame:
    """Return only wrong predictions, sorted by the model's confidence."""

    schema = schema or TicketSchema()
    required = set(schema.model_feature_columns) | {
        schema.ticket_id,
        schema.case_id,
        schema.label,
    }
    missing = sorted(required - set(validation_frame.columns))
    if missing:
        raise ValueError(f"error analysis frame is missing columns: {missing}")

    features = validation_frame.loc[:, schema.model_feature_columns]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.named_steps["classifier"].classes_)
    ordered = np.argsort(probabilities, axis=1)
    best_indices = ordered[:, -1]
    second_indices = ordered[:, -2] if len(classes) > 1 else ordered[:, -1]

    analysis = validation_frame[
        [
            schema.ticket_id,
            schema.case_id,
            schema.text,
            schema.channel,
            schema.priority,
            schema.label,
        ]
    ].copy()
    analysis = analysis.rename(columns={schema.label: "actual_label"})
    analysis["predicted_label"] = predictions
    analysis["confidence"] = probabilities[np.arange(len(analysis)), best_indices]
    analysis["second_choice"] = classes[second_indices]
    analysis["second_probability"] = probabilities[
        np.arange(len(analysis)), second_indices
    ]
    analysis["confidence_margin"] = (
        analysis["confidence"] - analysis["second_probability"]
    )
    analysis = analysis.loc[analysis["actual_label"].ne(analysis["predicted_label"])]
    return analysis.sort_values("confidence", ascending=False).reset_index(drop=True)


def predict_tickets(
    model: Pipeline,
    tickets: pd.DataFrame | Sequence[str] | str,
) -> pd.DataFrame:
    """Predict one or many tickets and expose calibrated-looking confidence cues.

    Logistic-regression probabilities are useful for ranking uncertainty, but
    they are not guaranteed to be calibrated.  A production system should tune
    a human-review threshold on representative validation data.
    """

    features = _coerce_feature_frame(tickets)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.named_steps["classifier"].classes_)
    output = pd.DataFrame(
        {
            "text": features["text"].tolist(),
            "predicted_label": predictions,
            "confidence": probabilities.max(axis=1),
        }
    )
    for index, label in enumerate(classes):
        output[f"prob_{label}"] = probabilities[:, index]
    return output


def save_model(
    model: Pipeline,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted pipeline and lightweight provenance as one joblib file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "classes_"):
        raise ValueError("only a fitted ticket-classification pipeline can be saved")
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "labels": [str(label) for label in classifier.classes_],
        "metadata": metadata or {},
        "model": model,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(destination)
    return destination


def load_model(path: str | Path) -> Pipeline:
    """Load a model artifact created here. Never load untrusted joblib files."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model artifact not found: {source}")
    payload = joblib.load(source)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != MODEL_FORMAT_VERSION
    ):
        raise ValueError("unsupported or malformed model artifact")
    model = payload.get("model")
    if not isinstance(model, Pipeline):
        raise ValueError("artifact does not contain an sklearn Pipeline")
    return model


def write_training_report(
    result: TrainingResult,
    path: str | Path,
    config: TrainingConfig,
) -> Path:
    """Write metrics and split provenance as UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "split": {
            "train_rows": result.train_rows,
            "validation_rows": result.validation_rows,
            "train_cases": int(result.split.train["case_id"].nunique()),
            "validation_cases": int(result.split.validation["case_id"].nunique()),
        },
        "metrics": result.report.to_dict(),
        "error_count": len(result.errors),
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def _coerce_feature_frame(
    tickets: pd.DataFrame | Sequence[str] | str,
) -> pd.DataFrame:
    """Adapt text-only demo input to the richer production feature contract."""

    schema = TicketSchema()
    if isinstance(tickets, pd.DataFrame):
        features = tickets.copy()
        if schema.text not in features:
            raise ValueError("feature frame must contain a text column")
    else:
        texts = [tickets] if isinstance(tickets, str) else list(tickets)
        features = pd.DataFrame({schema.text: texts})

    for column in (schema.channel, schema.priority, schema.customer_tier):
        if column not in features:
            features[column] = "unknown"
    for column in schema.model_feature_columns:
        features[column] = features[column].fillna("unknown").astype("string")
    return features.loc[:, schema.model_feature_columns].reset_index(drop=True)
