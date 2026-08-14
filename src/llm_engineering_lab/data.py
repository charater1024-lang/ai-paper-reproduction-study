"""Loading, validating, profiling, and splitting support-ticket data.

The important production lesson in this module is that a random *row* split is
not always safe.  Two rows can be paraphrases from the same customer case.  We
therefore split on ``case_id`` and keep every case on only one side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import pandas as pd

KNOWN_LABELS: Final[frozenset[str]] = frozenset(
    {"account_access", "billing", "cancellation", "delivery", "technical_issue"}
)


@dataclass(frozen=True, slots=True)
class TicketSchema:
    """Column names used by the example dataset and training pipeline."""

    ticket_id: str = "ticket_id"
    case_id: str = "case_id"
    created_at: str = "created_at"
    channel: str = "channel"
    priority: str = "priority"
    customer_tier: str = "customer_tier"
    text: str = "text"
    label: str = "label"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.ticket_id,
            self.case_id,
            self.created_at,
            self.channel,
            self.priority,
            self.customer_tier,
            self.text,
            self.label,
        )

    @property
    def model_feature_columns(self) -> tuple[str, ...]:
        return (self.text, self.channel, self.priority, self.customer_tier)


class FrameReader(Protocol):
    """Small interface that makes file input replaceable in tests or services."""

    def read(self, path: Path) -> pd.DataFrame: ...


class CsvFrameReader:
    """Read UTF-8 CSV data without silently treating IDs as numbers."""

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype="string", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """JSON-friendly facts useful for a quick data-contract check."""

    row_count: int
    case_count: int
    label_counts: dict[str, int]
    channel_counts: dict[str, int]
    priority_counts: dict[str, int]
    duplicate_ticket_ids: int
    missing_cells: int
    mean_text_chars: float
    created_at_min: str
    created_at_max: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Train/validation frames whose case groups do not overlap."""

    train: pd.DataFrame
    validation: pd.DataFrame

    def assert_no_group_leakage(self, case_column: str = "case_id") -> None:
        overlap = set(self.train[case_column]) & set(self.validation[case_column])
        if overlap:
            raise AssertionError(f"case leakage detected: {sorted(overlap)[:5]}")


class TicketDataRepository:
    """Repository that owns I/O and the support-ticket data contract."""

    def __init__(
        self,
        reader: FrameReader | None = None,
        schema: TicketSchema | None = None,
        expected_labels: frozenset[str] | None = KNOWN_LABELS,
    ) -> None:
        self.reader = reader or CsvFrameReader()
        self.schema = schema or TicketSchema()
        self.expected_labels = expected_labels

    def load(self, path: str | Path) -> pd.DataFrame:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"ticket dataset not found: {source}")
        frame = self.reader.read(source)
        return validate_ticket_frame(frame, self.schema, self.expected_labels)


def validate_ticket_frame(
    frame: pd.DataFrame,
    schema: TicketSchema | None = None,
    expected_labels: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Return a cleaned copy or fail early with an actionable schema error."""

    schema = schema or TicketSchema()
    missing_columns = sorted(set(schema.required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("ticket dataset is empty")

    clean = frame.copy()
    string_columns = [
        column for column in schema.required_columns if column != schema.created_at
    ]
    for column in string_columns:
        clean[column] = clean[column].astype("string").str.strip()

    missing_by_column = {
        column: int(clean[column].isna().sum() + clean[column].eq("").sum())
        for column in string_columns
    }
    created_at_as_text = clean[schema.created_at].astype("string").str.strip()
    missing_by_column[schema.created_at] = int(
        created_at_as_text.isna().sum() + created_at_as_text.eq("").sum()
    )
    missing_by_column = {
        key: value for key, value in missing_by_column.items() if value
    }
    if missing_by_column:
        raise ValueError(f"missing or blank values: {missing_by_column}")

    duplicate_ids = clean.loc[
        clean[schema.ticket_id].duplicated(keep=False), schema.ticket_id
    ].tolist()
    if duplicate_ids:
        raise ValueError(
            f"duplicate ticket_id values: {sorted(set(duplicate_ids))[:5]}"
        )

    case_label_counts = clean.groupby(schema.case_id)[schema.label].nunique()
    mixed_cases = case_label_counts[case_label_counts > 1].index.tolist()
    if mixed_cases:
        raise ValueError(f"one case_id maps to multiple labels: {mixed_cases[:5]}")

    if expected_labels is not None:
        actual_labels = frozenset(clean[schema.label].unique().tolist())
        if actual_labels != expected_labels:
            missing = sorted(expected_labels - actual_labels)
            unexpected = sorted(actual_labels - expected_labels)
            raise ValueError(
                f"label contract mismatch; missing={missing}, unexpected={unexpected}"
            )

    try:
        clean[schema.created_at] = pd.to_datetime(
            clean[schema.created_at], utc=True, errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {schema.created_at} timestamp: {error}") from error
    if clean[schema.created_at].isna().any():
        raise ValueError(f"invalid {schema.created_at} timestamp: missing value")

    return clean.reset_index(drop=True)


def load_ticket_data(path: str | Path) -> pd.DataFrame:
    """Convenience API for loading the bundled dataset with its strict contract."""

    return TicketDataRepository().load(path)


def dataset_profile(
    frame: pd.DataFrame, schema: TicketSchema | None = None
) -> DatasetProfile:
    """Compute compact EDA statistics without modifying the input frame."""

    schema = schema or TicketSchema()
    clean = validate_ticket_frame(frame, schema, expected_labels=None)
    missing_cells = int(clean[list(schema.required_columns)].isna().sum().sum())
    return DatasetProfile(
        row_count=len(clean),
        case_count=int(clean[schema.case_id].nunique()),
        label_counts=_value_counts(clean[schema.label]),
        channel_counts=_value_counts(clean[schema.channel]),
        priority_counts=_value_counts(clean[schema.priority]),
        duplicate_ticket_ids=int(clean[schema.ticket_id].duplicated().sum()),
        missing_cells=missing_cells,
        mean_text_chars=round(float(clean[schema.text].str.len().mean()), 2),
        created_at_min=clean[schema.created_at].min().isoformat(),
        created_at_max=clean[schema.created_at].max().isoformat(),
    )


def stratified_group_split(
    frame: pd.DataFrame,
    validation_size: float = 0.2,
    seed: int = 42,
    schema: TicketSchema | None = None,
) -> DatasetSplit:
    """Split by case within each label to prevent paraphrase leakage.

    Each ``case_id`` must belong to exactly one class and each class must have at
    least two cases.  This explicit implementation is intentionally readable so
    learners can inspect the rule instead of treating splitting as a black box.
    """

    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1")
    schema = schema or TicketSchema()
    clean = validate_ticket_frame(frame, schema, expected_labels=None)
    cases = clean[[schema.case_id, schema.label]].drop_duplicates()
    rng = np.random.default_rng(seed)
    validation_cases: set[str] = set()

    for label in sorted(cases[schema.label].unique().tolist()):
        label_cases = np.array(
            sorted(cases.loc[cases[schema.label].eq(label), schema.case_id].tolist()),
            dtype=object,
        )
        if len(label_cases) < 2:
            raise ValueError(f"label {label!r} needs at least two case groups")
        rng.shuffle(label_cases)
        count = min(
            len(label_cases) - 1, max(1, round(len(label_cases) * validation_size))
        )
        validation_cases.update(str(case_id) for case_id in label_cases[:count])

    is_validation = clean[schema.case_id].isin(validation_cases)
    split = DatasetSplit(
        train=clean.loc[~is_validation].reset_index(drop=True),
        validation=clean.loc[is_validation].reset_index(drop=True),
    )
    split.assert_no_group_leakage(schema.case_id)
    return split


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}
