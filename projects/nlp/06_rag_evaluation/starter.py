"""Project 06 starter: evaluate retrieval and abstention separately."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class EvaluationProjectConfig:
    knowledge_path: Path
    cases_path: Path
    top_k: int = 3
    min_score: float = 0.01
    show_failures: bool = False
    threshold_candidates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20)

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k는 양수여야 합니다.")
        if not self.threshold_candidates:
            raise ValueError("threshold_candidates는 비어 있을 수 없습니다.")


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Return ``1 / rank`` for the first relevant result, or zero."""

    # TODO 1: ranks start at one and only the first relevant result matters.
    raise NotImplementedError("TODO 1: reciprocal_rank를 구현하세요.")


def evaluate(
    search: Callable[[str, int], Sequence[str]],
    cases: Sequence[dict[str, object]],
    top_k: int = 3,
) -> dict[str, float]:
    """Compute answerable ranking and no-answer abstention metrics.

    TODO 2: validate unique IDs, types, and empty/non-empty relevance lists.
    TODO 3: compute answerable hit@k, Recall@k, and MRR@k.
    TODO 4: compute coverage and no-answer abstention with separate denominators.
    """

    raise NotImplementedError("TODO 2~4: evaluate를 구현하세요.")


class RetrievalEvaluationProject:
    """Mirror load -> evaluate -> sweep -> failure-analysis boundaries."""

    def __init__(self, config: EvaluationProjectConfig) -> None:
        self.config = config

    def load_cases(self) -> list[dict[str, object]]:
        """TODO 5: parse JSONL and validate answerable/no-answer coverage."""

        raise NotImplementedError("TODO 5: load_cases를 구현하세요.")

    def build_retriever(self, min_score: float) -> object:
        """TODO 6: fit a fresh retriever for an isolated threshold trial."""

        raise NotImplementedError("TODO 6: build_retriever를 구현하세요.")

    def evaluate(
        self,
        retriever: object,
        cases: list[dict[str, object]],
    ) -> dict[str, float]:
        """Adapt retriever results to IDs and call the pure metric function."""

        # TODO 7: create a search adapter without coupling metrics to result types.
        raise NotImplementedError("TODO 7: evaluate adapter를 구현하세요.")

    def threshold_sweep(
        self,
        cases: list[dict[str, object]],
    ) -> list[tuple[float, dict[str, float]]]:
        reports: list[tuple[float, dict[str, float]]] = []
        for threshold in self.config.threshold_candidates:
            retriever = self.build_retriever(threshold)
            reports.append((threshold, self.evaluate(retriever, cases)))
        return reports

    def find_failures(
        self,
        retriever: object,
        cases: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """TODO 8: return misses and no-answer false positives by query ID."""

        raise NotImplementedError("TODO 8: find_failures를 구현하세요.")

    @staticmethod
    def compact(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    @staticmethod
    def print_report(report: dict[str, float]) -> None:
        for name, value in sorted(report.items()):
            print(f"{name}={RetrievalEvaluationProject.compact(value)}")

    def print_sweep(
        self,
        sweep: list[tuple[float, dict[str, float]]],
    ) -> None:
        for threshold, report in sweep:
            recall = report.get("recall_at_k", float("nan"))
            print(f"threshold={threshold:.2f} recall@k={recall:.3f}")

    def run(self) -> dict[str, float]:
        cases = self.load_cases()
        retriever = self.build_retriever(self.config.min_score)
        report = self.evaluate(retriever, cases)
        self.print_report(report)
        self.print_sweep(self.threshold_sweep(cases))
        if self.config.show_failures:
            print(self.find_failures(retriever, cases))
        return report


def main() -> None:
    config = EvaluationProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
        cases_path=ROOT / "data/practice/rag_queries.jsonl",
    )
    try:
        RetrievalEvaluationProject(config).run()
    except NotImplementedError as error:
        print(f"연습 대기: {error}")


if __name__ == "__main__":
    main()
