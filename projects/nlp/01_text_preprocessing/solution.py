"""Inspect normalization, privacy masking, tokenization, and term frequency."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.nlp_basics import (  # noqa: E402
    corpus_term_frequencies,
    mask_pii,
    normalize_text,
    tokenize,
    top_terms,
)


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Reproducible inputs for the preprocessing portfolio project."""

    data_path: Path
    top_n: int = 15
    sample_suffix: str = "연락처 Student@example.com / 010-1234-5678"

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n은 양수여야 합니다.")


@dataclass(frozen=True, slots=True)
class PreprocessingReport:
    """Structured evidence produced by one preprocessing run."""

    original: str
    masked: str
    normalized: str
    tokens: tuple[str, ...]
    document_count: int
    frequent_terms: tuple[tuple[str, int], ...]


class TextPreprocessingProject:
    """Make every text-cleaning boundary explicit and independently testable."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def load_documents(self) -> list[str]:
        """Load only non-empty text fields from the local CSV contract."""

        with self.config.data_path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            if rows.fieldnames is None or "text" not in rows.fieldnames:
                raise ValueError("CSV에 `text` column이 필요합니다.")
            documents = [str(row["text"]) for row in rows if row.get("text")]

        if not documents:
            raise ValueError(f"텍스트가 없습니다: {self.config.data_path}")
        return documents

    @staticmethod
    def preprocess_one(text: str) -> tuple[str, str, tuple[str, ...]]:
        """Mask first, normalize second, and tokenize the safe representation."""

        masked = mask_pii(text)
        normalized = normalize_text(masked)
        tokens = tuple(tokenize(normalized))
        return masked, normalized, tokens

    def build_report(self, documents: Sequence[str]) -> PreprocessingReport:
        """Analyze a sample and aggregate corpus-level term frequencies."""

        if not documents:
            raise ValueError("documents는 비어 있을 수 없습니다.")

        original = f"{documents[0]} {self.config.sample_suffix}"
        masked, normalized, tokens = self.preprocess_one(original)
        frequencies = corpus_term_frequencies(documents, mask_sensitive_data=True)
        frequent_terms = tuple(top_terms(frequencies, limit=self.config.top_n))

        report = PreprocessingReport(
            original=original,
            masked=masked,
            normalized=normalized,
            tokens=tokens,
            document_count=len(documents),
            frequent_terms=frequent_terms,
        )
        self.validate_report(report)
        return report

    @staticmethod
    def validate_report(report: PreprocessingReport) -> None:
        """Fail early if raw email or phone values survive the privacy boundary."""

        forbidden_values = ("Student@example.com", "010-1234-5678")
        if any(value in report.masked for value in forbidden_values):
            raise AssertionError("PII masking contract failed")
        if not report.tokens:
            raise AssertionError("tokenization produced an empty sequence")

    @staticmethod
    def print_report(report: PreprocessingReport) -> None:
        print("[원문]", report.original)
        print("[마스킹]", report.masked)
        print("[정규화]", report.normalized)
        print("[토큰]", list(report.tokens))
        print(f"\n문서 수: {report.document_count}")
        print("상위 토큰:")
        for token, count in report.frequent_terms:
            print(f"  {token:<16} {count:>3}")

    def run(self) -> PreprocessingReport:
        documents = self.load_documents()
        report = self.build_report(documents)
        self.print_report(report)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/customer_support_tickets.csv",
    )
    parser.add_argument("--limit", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessingConfig(data_path=args.data, top_n=args.limit)
    TextPreprocessingProject(config).run()


if __name__ == "__main__":
    main()
