"""Project 01 starter: implement an auditable preprocessing pipeline."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def prepare_text(text: str) -> list[str]:
    """Normalize, mask sensitive values, and tokenize one document.

    TODO 1: apply Unicode NFKC, lowercase, and whitespace normalization.
    TODO 2: mask email, Korean phone, and long order identifiers.
    TODO 3: extract Korean, English, number, and mask tokens in order.
    """

    raise NotImplementedError("TODO 1~3: prepare_text를 구현하세요.")


def count_terms(documents: Iterable[str]) -> Counter[str]:
    """Call ``prepare_text`` once per document and aggregate a Counter."""

    # TODO 4: support any iterable without converting it to a list first.
    raise NotImplementedError("TODO 4: count_terms를 구현하세요.")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Record every input that changes a preprocessing experiment."""

    data_path: Path
    top_n: int = 15
    sample_suffix: str = "연락처 Student@example.com / 010-1234-5678"

    def __post_init__(self) -> None:
        # TODO 5: reject a non-positive ``top_n``.
        if self.top_n <= 0:
            raise ValueError("top_n은 양수여야 합니다.")


@dataclass(frozen=True, slots=True)
class PreprocessingReport:
    """Keep intermediate values as portfolio evidence, not only final counts."""

    original: str
    masked: str
    normalized: str
    tokens: tuple[str, ...]
    document_count: int
    frequent_terms: tuple[tuple[str, int], ...]


class TextPreprocessingProject:
    """Mirror the solution's load -> transform -> validate -> report flow."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def load_documents(self) -> list[str]:
        """Load non-empty values from the CSV ``text`` column."""

        # TODO 6: validate the header and fail if no documents are present.
        with self.config.data_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise ValueError("CSV에 `text` column이 필요합니다.")
            documents = [str(row["text"]) for row in reader if row.get("text")]
        if not documents:
            raise ValueError("전처리할 문서가 없습니다.")
        return documents

    @staticmethod
    def preprocess_one(text: str) -> tuple[str, str, tuple[str, ...]]:
        """Expose masked, normalized, and tokenized intermediate values."""

        # TODO 6: reuse the same rules as prepare_text without hiding stages.
        raise NotImplementedError("TODO 6: preprocess_one을 구현하세요.")

    def build_report(self, documents: Sequence[str]) -> PreprocessingReport:
        """Create sample evidence and deterministic top-frequency terms."""

        # TODO 7: aggregate counts and sort by (-count, token).
        raise NotImplementedError("TODO 7: build_report를 구현하세요.")

    @staticmethod
    def validate_report(report: PreprocessingReport) -> None:
        """Reject reports that leak the injected email or phone value."""

        forbidden = ("Student@example.com", "010-1234-5678")
        if any(value in report.masked for value in forbidden):
            raise AssertionError("PII masking contract failed")
        if not report.tokens:
            raise AssertionError("tokenization produced no tokens")

    @staticmethod
    def print_report(report: PreprocessingReport) -> None:
        print("[원문]", report.original)
        print("[마스킹]", report.masked)
        print("[정규화]", report.normalized)
        print("[토큰]", list(report.tokens))
        print(f"문서 수: {report.document_count}")
        for token, count in report.frequent_terms:
            print(f"  {token:<16} {count:>3}")

    def run(self) -> PreprocessingReport:
        documents = self.load_documents()
        report = self.build_report(documents)
        self.validate_report(report)
        self.print_report(report)
        return report


def main() -> None:
    config = PreprocessingConfig(
        data_path=ROOT / "data/customer_support_tickets.csv",
    )
    try:
        TextPreprocessingProject(config).run()
    except NotImplementedError as error:
        print(f"연습 대기: {error}")


if __name__ == "__main__":
    main()
