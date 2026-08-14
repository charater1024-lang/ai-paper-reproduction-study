"""Measure retrieval ranking and no-answer behaviour on a fixed JSONL set."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.rag_evaluation import (  # noqa: E402
    EvaluationCase,
    RetrievalEvaluationReport,
    evaluate_tfidf_retriever,
    load_evaluation_cases,
)
from llm_engineering_lab.retrieval import (  # noqa: E402
    TfidfRetriever,
    load_knowledge_base,
)


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


class RetrievalEvaluationProject:
    """Separate answerable ranking quality from no-answer abstention."""

    def __init__(self, config: EvaluationProjectConfig) -> None:
        self.config = config

    def load_cases(self) -> list[EvaluationCase]:
        """Load the fixed benchmark before tuning a retrieval threshold."""

        cases = load_evaluation_cases(self.config.cases_path)
        if not any(case.is_answerable for case in cases):
            raise ValueError("answerable evaluation case가 필요합니다.")
        if not any(not case.is_answerable for case in cases):
            raise ValueError("no-answer evaluation case가 필요합니다.")
        return cases

    def build_retriever(self, min_score: float) -> TfidfRetriever:
        """Build a fresh retriever so every threshold candidate is isolated."""

        articles = load_knowledge_base(self.config.knowledge_path)
        return TfidfRetriever(min_score=min_score).fit(articles)

    def evaluate(
        self,
        retriever: TfidfRetriever,
        cases: list[EvaluationCase],
    ) -> RetrievalEvaluationReport:
        """Compute ranking and abstention metrics at one operating point."""

        report = evaluate_tfidf_retriever(
            retriever,
            cases,
            top_k=self.config.top_k,
        )
        if report.total_cases != len(cases):
            raise AssertionError("evaluation dropped one or more cases")
        return report

    def threshold_sweep(
        self,
        cases: list[EvaluationCase],
    ) -> list[tuple[float, RetrievalEvaluationReport]]:
        """Expose the retrieval-versus-abstention trade-off."""

        results: list[tuple[float, RetrievalEvaluationReport]] = []
        for threshold in self.config.threshold_candidates:
            retriever = self.build_retriever(threshold)
            results.append((threshold, self.evaluate(retriever, cases)))
        return results

    def find_failures(
        self,
        retriever: TfidfRetriever,
        cases: list[EvaluationCase],
    ) -> list[tuple[EvaluationCase, list[str]]]:
        """Return concrete misses and false evidence, not only aggregate scores."""

        failures: list[tuple[EvaluationCase, list[str]]] = []
        for case in cases:
            results = retriever.search(case.query, top_k=self.config.top_k)
            retrieved_ids = [result.article.id for result in results]
            missed_answer = case.is_answerable and not set(
                case.relevant_ids
            ).intersection(retrieved_ids)
            false_evidence = not case.is_answerable and bool(retrieved_ids)
            if missed_answer or false_evidence:
                failures.append((case, retrieved_ids))
        return failures

    @staticmethod
    def compact(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    def print_report(self, report: RetrievalEvaluationReport) -> None:
        print(
            f"cases={report.total_cases} "
            f"(answerable={report.answerable_cases}, "
            f"no-answer={report.no_answer_cases})"
        )
        print(
            f"hit@{self.config.top_k}={self.compact(report.hit_rate_at_k)}, "
            f"recall@{self.config.top_k}={self.compact(report.recall_at_k)}, "
            f"MRR@{self.config.top_k}={self.compact(report.mrr_at_k)}"
        )
        print(
            f"coverage={report.coverage:.3f}, "
            "no-answer abstention accuracy="
            f"{self.compact(report.abstention_accuracy)}"
        )

    def print_sweep(
        self,
        sweep: list[tuple[float, RetrievalEvaluationReport]],
    ) -> None:
        print("\nthreshold sweep")
        print("min_score  recall@k  MRR@k  coverage  no-answer-accuracy")
        for threshold, report in sweep:
            print(
                f"{threshold:>9.2f}  "
                f"{self.compact(report.recall_at_k):>8}  "
                f"{self.compact(report.mrr_at_k):>5}  "
                f"{report.coverage:>8.3f}  "
                f"{self.compact(report.abstention_accuracy):>18}"
            )

    def run(self) -> RetrievalEvaluationReport:
        cases = self.load_cases()
        retriever = self.build_retriever(self.config.min_score)
        report = self.evaluate(retriever, cases)
        self.print_report(report)
        self.print_sweep(self.threshold_sweep(cases))

        if self.config.show_failures:
            print("\n실패 사례")
            failures = self.find_failures(retriever, cases)
            if not failures:
                print("없음")
            for case, retrieved_ids in failures:
                print(
                    f"{case.query_id}: expected={list(case.relevant_ids)} "
                    f"retrieved={retrieved_ids} | {case.query}"
                )

        if asdict(report)["top_k"] != self.config.top_k:
            raise AssertionError("serialized report changed top_k")
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--show-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
        cases_path=ROOT / "data/practice/rag_queries.jsonl",
        top_k=args.top_k,
        min_score=args.min_score,
        show_failures=args.show_failures,
    )
    RetrievalEvaluationProject(config).run()


if __name__ == "__main__":
    main()
