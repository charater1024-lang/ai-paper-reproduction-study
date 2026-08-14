"""Project 02 starter: build a leakage-safe intent-classification project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class IntentProjectConfig:
    """Record data, split seed, and the inference example."""

    data_path: Path
    seed: int = 42
    prediction_text: str = "카드에서 같은 금액이 두 번 결제됐어요"

    def __post_init__(self) -> None:
        if not self.prediction_text.strip():
            raise ValueError("prediction_text는 비어 있을 수 없습니다.")


@dataclass(slots=True)
class BaselineTrainingResult:
    """A small result contract to make metrics and leakage checks inspectable."""

    model: object
    metrics: dict[str, float]
    errors: pd.DataFrame
    train_case_ids: frozenset[str]
    validation_case_ids: frozenset[str]

    def assert_no_group_leakage(self) -> None:
        overlap = self.train_case_ids.intersection(self.validation_case_ids)
        if overlap:
            raise AssertionError(f"case_id leakage: {sorted(overlap)}")


def train_baseline(frame: pd.DataFrame, seed: int) -> BaselineTrainingResult:
    """Fit and evaluate one reproducible character TF-IDF baseline.

    TODO 2: split unique ``case_id`` groups while preserving label balance.
    TODO 3: compose character TF-IDF and logistic regression in one Pipeline.
    TODO 4: compute accuracy, macro-F1, per-class metrics, and error rows.
    """

    raise NotImplementedError("TODO 2~4: train_baseline을 구현하세요.")


class IntentClassificationProject:
    """Mirror load -> split/train -> diagnose -> infer from the solution."""

    def __init__(self, config: IntentProjectConfig) -> None:
        self.config = config
        self.training_result: BaselineTrainingResult | None = None

    def load_frame(self) -> pd.DataFrame:
        """Load the CSV and enforce its tabular schema before splitting."""

        # TODO 1 extension: validate empty values and one-label-per-case_id.
        frame = pd.read_csv(self.config.data_path)
        required = {"case_id", "text", "label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"필수 column 누락: {sorted(missing)}")
        return frame

    def train(self, frame: pd.DataFrame) -> BaselineTrainingResult:
        result = train_baseline(frame, seed=self.config.seed)
        result.assert_no_group_leakage()
        self.training_result = result
        return result

    @staticmethod
    def print_metrics(result: BaselineTrainingResult) -> None:
        for name, value in sorted(result.metrics.items()):
            print(f"{name}={value:.3f}")

    @staticmethod
    def print_errors(result: BaselineTrainingResult, limit: int = 5) -> None:
        print("\n오분류 예시")
        print(result.errors.head(limit).to_string(index=False))

    def predict_new_text(self, result: BaselineTrainingResult) -> pd.DataFrame:
        """Use the fitted Pipeline so training and inference transform identically."""

        # TODO 5: build the expected input columns and include confidence.
        raise NotImplementedError("TODO 5: predict_new_text를 구현하세요.")

    def run(self) -> BaselineTrainingResult:
        frame = self.load_frame()
        result = self.train(frame)
        self.print_metrics(result)
        self.print_errors(result)
        print(self.predict_new_text(result).to_string(index=False))
        return result


def main() -> None:
    config = IntentProjectConfig(
        data_path=ROOT / "data/customer_support_tickets.csv",
    )
    try:
        IntentClassificationProject(config).run()
    except NotImplementedError as error:
        print(f"연습 대기: {error}")


if __name__ == "__main__":
    main()
