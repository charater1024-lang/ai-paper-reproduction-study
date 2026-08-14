"""Inspect the bundled ticket dataset before training.

Run from the repository root:
    python scripts/01_inspect_data.py
    python scripts/01_inspect_data.py --sample 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.data import dataset_profile, load_ticket_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "customer_support_tickets.csv",
        help="UTF-8 support-ticket CSV",
    )
    parser.add_argument("--sample", type=int, default=3, help="number of example rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample < 0:
        raise SystemExit("--sample must be zero or greater")

    frame = load_ticket_data(args.data)
    profile = dataset_profile(frame)
    print(f"dataset: {args.data.resolve()}")
    print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))

    print("\nlabel x channel (row counts)")
    print(frame.groupby(["label", "channel"]).size().unstack(fill_value=0).to_string())

    if args.sample:
        print("\nexamples")
        columns = ["ticket_id", "case_id", "channel", "priority", "text", "label"]
        print(
            frame.loc[:, columns]
            .sample(min(args.sample, len(frame)), random_state=42)
            .to_string(index=False)
        )

    # TODO(learner): add text-length quantiles and inspect the longest 1% of tickets.


if __name__ == "__main__":
    main()
