"""Train, evaluate, persist, and optionally query the scikit-learn baseline.

Run from the repository root:
    python scripts/02_train_ml_baseline.py
    python scripts/02_train_ml_baseline.py --predict "결제가 두 번 됐어요"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.data import load_ticket_data  # noqa: E402
from llm_engineering_lab.ml import (  # noqa: E402
    TicketClassifierTrainer,
    TrainingConfig,
    predict_tickets,
    save_model,
    write_training_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "customer_support_tickets.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "ml" / "ticket_classifier.joblib",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts" / "ml" / "metrics.json",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=ROOT / "artifacts" / "ml" / "errors.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument(
        "--c", type=float, default=4.0, help="inverse regularization strength"
    )
    parser.add_argument(
        "--predict",
        action="append",
        default=[],
        metavar="TEXT",
        help="ticket text to classify after training; repeat for multiple texts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        seed=args.seed,
        validation_size=args.validation_size,
        max_features=args.max_features,
        regularization_c=args.c,
    )
    frame = load_ticket_data(args.data)
    result = TicketClassifierTrainer(config).train(frame)

    save_model(
        result.model,
        args.model,
        metadata={"source_data": str(args.data), "training_config": asdict(config)},
    )
    write_training_report(result, args.report, config)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    result.errors.to_csv(args.errors, index=False, encoding="utf-8-sig")

    print(f"split: train={result.train_rows} validation={result.validation_rows}")
    print(
        f"accuracy={result.report.accuracy:.3f} "
        f"macro_f1={result.report.macro_f1:.3f} "
        f"weighted_f1={result.report.weighted_f1:.3f}"
    )
    class_rows = [
        {"label": label, **asdict(metrics)}
        for label, metrics in result.report.per_class.items()
    ]
    print("\nper-class metrics")
    print(
        pd.DataFrame(class_rows).to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print("\nconfusion matrix (rows=actual, columns=predicted)")
    print(
        pd.DataFrame(
            result.report.confusion_matrix,
            index=result.report.labels,
            columns=result.report.labels,
        ).to_string()
    )
    print(f"\nmodel: {args.model.resolve()}")
    print(f"report: {args.report.resolve()}")
    print(f"errors ({len(result.errors)}): {args.errors.resolve()}")

    if args.predict:
        print("\npredictions")
        prediction_frame = predict_tickets(result.model, args.predict)
        print(
            prediction_frame.to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            )
        )

    # TODO(learner): choose a confidence threshold and route uncertain tickets to humans.


if __name__ == "__main__":
    main()
