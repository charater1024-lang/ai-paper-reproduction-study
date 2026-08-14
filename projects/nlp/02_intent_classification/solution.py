"""Train a leakage-safe TF-IDF intent classifier and inspect its mistakes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.data import load_ticket_data  # noqa: E402
from llm_engineering_lab.ml import (  # noqa: E402
    TicketClassifierTrainer,
    TrainingConfig,
    TrainingResult,
    predict_tickets,
)


@dataclass(frozen=True, slots=True)
class IntentProjectConfig:
    """Inputs that should be recorded with a classification experiment."""

    data_path: Path
    seed: int = 42
    prediction_text: str = "카드에서 같은 금액이 두 번 결제됐어요"

    def __post_init__(self) -> None:
        if not self.prediction_text.strip():
            raise ValueError("prediction_text는 비어 있을 수 없습니다.")


class IntentClassificationProject:
    """Orchestrate loading, group-safe splitting, training, and error analysis."""

    def __init__(self, config: IntentProjectConfig) -> None:
        self.config = config
        self.training_result: TrainingResult | None = None

    def load_frame(self) -> pd.DataFrame:
        """Load the validated local dataset before any train/validation split."""

        frame = load_ticket_data(self.config.data_path)
        required = {"case_id", "text", "label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"필수 column 누락: {sorted(missing)}")
        return frame

    def train(self, frame: pd.DataFrame) -> TrainingResult:
        """Fit preprocessing and classifier only through the training pipeline."""

        training_config = TrainingConfig(seed=self.config.seed)
        result = TicketClassifierTrainer(training_config).train(frame)
        result.split.assert_no_group_leakage()
        self.training_result = result
        return result

    @staticmethod
    def print_metrics(result: TrainingResult) -> None:
        """Print aggregate and per-class metrics instead of accuracy alone."""

        print(f"train={result.train_rows}, validation={result.validation_rows}")
        print(
            f"accuracy={result.report.accuracy:.3f}, "
            f"macro_f1={result.report.macro_f1:.3f}, "
            f"weighted_f1={result.report.weighted_f1:.3f}"
        )
        print("\n클래스별 지표")
        for label, metric in result.report.per_class.items():
            print(
                f"  {label:<18} precision={metric.precision:.3f} "
                f"recall={metric.recall:.3f} "
                f"f1={metric.f1:.3f} n={metric.support}"
            )

    @staticmethod
    def print_errors(result: TrainingResult, limit: int = 5) -> None:
        """Expose validation failures so the metric can be interpreted."""

        print(f"\n오분류 수: {len(result.errors)}")
        if result.errors.empty:
            print("없음")
            return
        print(result.errors.head(limit).to_string(index=False))

    def predict_new_text(self, result: TrainingResult) -> pd.DataFrame:
        """Run inference through the fitted pipeline used during validation."""

        return predict_tickets(result.model, self.config.prediction_text)

    def run(self) -> TrainingResult:
        frame = self.load_frame()
        result = self.train(frame)
        self.print_metrics(result)
        self.print_errors(result)
        print("\n새 문장 예측")
        print(self.predict_new_text(result).to_string(index=False))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/customer_support_tickets.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", default="카드에서 같은 금액이 두 번 결제됐어요")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IntentProjectConfig(
        data_path=args.data,
        seed=args.seed,
        prediction_text=args.predict,
    )
    IntentClassificationProject(config).run()


if __name__ == "__main__":
    main()
