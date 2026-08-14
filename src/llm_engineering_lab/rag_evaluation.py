"""Offline retrieval evaluation for the practice RAG knowledge base.

The module deliberately evaluates retrieval only.  It therefore stays useful before an
LLM is connected and does not require network access or an API key.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .retrieval import Retriever, SearchResult


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One retrieval evaluation query.

    An empty ``relevant_ids`` tuple marks a no-answer query: the desired behaviour is to
    return no search results rather than attach an unrelated source.
    """

    query_id: str
    query: str
    relevant_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        query_id = _non_empty_string(self.query_id, "query_id")
        query = _non_empty_string(self.query, "query")
        if not isinstance(self.relevant_ids, tuple):
            raise ValueError("relevant_ids must be a tuple of strings")

        relevant_ids = tuple(
            _non_empty_string(value, f"relevant_ids[{index}]")
            for index, value in enumerate(self.relevant_ids)
        )
        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError("relevant_ids must not contain duplicates")

        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "relevant_ids", relevant_ids)

    @property
    def is_answerable(self) -> bool:
        return bool(self.relevant_ids)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> EvaluationCase:
        """Validate and convert one decoded JSON object."""

        raw_relevant_ids = raw.get("relevant_ids")
        if not isinstance(raw_relevant_ids, list):
            raise ValueError("relevant_ids must be a JSON array of strings")
        return cls(
            query_id=_non_empty_string(raw.get("query_id"), "query_id"),
            query=_non_empty_string(raw.get("query"), "query"),
            relevant_ids=tuple(
                _non_empty_string(value, f"relevant_ids[{index}]")
                for index, value in enumerate(raw_relevant_ids)
            ),
        )


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Aggregate retrieval metrics at one ``top_k`` value.

    ``coverage`` is the fraction of all cases for which at least one result was returned.
    ``no_answer_coverage`` is the same fraction restricted to no-answer cases, so it is
    also the false-positive rate and complements ``abstention_accuracy``.

    Metrics whose denominator is zero are ``None`` instead of a misleading zero.
    """

    top_k: int
    total_cases: int
    answerable_cases: int
    no_answer_cases: int
    answerable_hits: int
    correct_abstentions: int
    hit_rate_at_k: float | None
    recall_at_k: float | None
    mrr_at_k: float | None
    abstention_accuracy: float | None
    coverage: float
    answerable_coverage: float | None
    no_answer_coverage: float | None

    @property
    def mean_reciprocal_rank(self) -> float | None:
        """Readable alias for ``mrr_at_k``."""

        return self.mrr_at_k


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load validated RAG evaluation cases from a JSONL file.

    Blank lines are ignored.  Errors include the source path and line number, and duplicate
    ``query_id`` values are rejected because they make per-query regression tracking
    ambiguous.
    """

    source = Path(path)
    cases: list[EvaluationCase] = []
    seen_query_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("each line must contain a JSON object")
                case = EvaluationCase.from_dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid evaluation case: {error}"
                ) from error

            if case.query_id in seen_query_ids:
                raise ValueError(
                    f"{source}:{line_number}: duplicate query_id {case.query_id!r}"
                )
            seen_query_ids.add(case.query_id)
            cases.append(case)

    return cases


def evaluate_retrieval[ResultT](
    cases: Iterable[EvaluationCase],
    search: Callable[[str, int], Iterable[ResultT]],
    get_result_id: Callable[[ResultT], str],
    *,
    top_k: int = 3,
) -> RetrievalEvaluationReport:
    """Evaluate a search callable against answerable and no-answer cases.

    The search callable receives ``(query, top_k)``.  At most its first ``top_k`` results
    are scored, even if an implementation accidentally returns more.  Recall is macro
    averaged across answerable cases and MRR uses the first relevant result's rank.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    materialized = tuple(cases)
    if not materialized:
        raise ValueError("at least one evaluation case is required")
    query_ids = [case.query_id for case in materialized]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("evaluation case query_id values must be unique")

    answerable_cases = 0
    no_answer_cases = 0
    answerable_hits = 0
    correct_abstentions = 0
    covered_cases = 0
    covered_answerable_cases = 0
    covered_no_answer_cases = 0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0

    for case in materialized:
        retrieved_ids = [
            _non_empty_string(get_result_id(result), "retrieved result id")
            for result in islice(search(case.query, top_k), top_k)
        ]
        is_covered = bool(retrieved_ids)
        covered_cases += is_covered

        if not case.is_answerable:
            no_answer_cases += 1
            covered_no_answer_cases += is_covered
            correct_abstentions += not is_covered
            continue

        answerable_cases += 1
        covered_answerable_cases += is_covered
        relevant_ids = set(case.relevant_ids)
        retrieved_id_set = set(retrieved_ids)
        matched_ids = relevant_ids & retrieved_id_set
        is_hit = bool(matched_ids)
        answerable_hits += is_hit
        recall_sum += len(matched_ids) / len(relevant_ids)

        first_relevant_rank = next(
            (
                rank
                for rank, retrieved_id in enumerate(retrieved_ids, start=1)
                if retrieved_id in relevant_ids
            ),
            None,
        )
        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1 / first_relevant_rank

    total_cases = len(materialized)
    return RetrievalEvaluationReport(
        top_k=top_k,
        total_cases=total_cases,
        answerable_cases=answerable_cases,
        no_answer_cases=no_answer_cases,
        answerable_hits=answerable_hits,
        correct_abstentions=correct_abstentions,
        hit_rate_at_k=_ratio(answerable_hits, answerable_cases),
        recall_at_k=_ratio(recall_sum, answerable_cases),
        mrr_at_k=_ratio(reciprocal_rank_sum, answerable_cases),
        abstention_accuracy=_ratio(correct_abstentions, no_answer_cases),
        coverage=covered_cases / total_cases,
        answerable_coverage=_ratio(covered_answerable_cases, answerable_cases),
        no_answer_coverage=_ratio(covered_no_answer_cases, no_answer_cases),
    )


def _ratio(numerator: float | int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def search_result_article_id(result: SearchResult) -> str:
    """Extract the knowledge-base ID from a standard TF-IDF search result."""

    return result.article.id


def evaluate_tfidf_retriever(
    retriever: Retriever,
    cases: Iterable[EvaluationCase],
    *,
    top_k: int = 3,
) -> RetrievalEvaluationReport:
    """Evaluate ``TfidfRetriever.search`` (or another standard ``Retriever``)."""

    return evaluate_retrieval(
        cases,
        retriever.search,
        search_result_article_id,
        top_k=top_k,
    )


__all__ = [
    "EvaluationCase",
    "RetrievalEvaluationReport",
    "evaluate_retrieval",
    "evaluate_tfidf_retriever",
    "load_evaluation_cases",
    "search_result_article_id",
]
