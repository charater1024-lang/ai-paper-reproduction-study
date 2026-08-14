import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from llm_engineering_lab.rag_evaluation import (
    EvaluationCase,
    evaluate_retrieval,
    evaluate_tfidf_retriever,
    load_evaluation_cases,
)
from llm_engineering_lab.retrieval import TfidfRetriever, load_knowledge_base

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StubResult:
    document_id: str


def test_hand_calculated_retrieval_metrics() -> None:
    cases = [
        EvaluationCase("q-1", "first", ("a", "b")),
        EvaluationCase("q-2", "second", ("c",)),
        EvaluationCase("q-3", "unknown-empty", ()),
        EvaluationCase("q-4", "unknown-covered", ()),
    ]
    results_by_query = {
        "first": [StubResult("x"), StubResult("a"), StubResult("z")],
        "second": [StubResult("z")],
        "unknown-empty": [],
        "unknown-covered": [StubResult("x")],
    }

    report = evaluate_retrieval(
        cases,
        lambda query, top_k: results_by_query[query][:top_k],
        lambda result: result.document_id,
        top_k=3,
    )

    assert report.total_cases == 4
    assert report.answerable_cases == 2
    assert report.no_answer_cases == 2
    assert report.answerable_hits == 1
    assert report.correct_abstentions == 1
    assert report.hit_rate_at_k == pytest.approx(1 / 2)
    assert report.recall_at_k == pytest.approx(1 / 4)
    assert report.mrr_at_k == pytest.approx(1 / 4)
    assert report.mean_reciprocal_rank == report.mrr_at_k
    assert report.abstention_accuracy == pytest.approx(1 / 2)
    assert report.coverage == pytest.approx(3 / 4)
    assert report.answerable_coverage == 1.0
    assert report.no_answer_coverage == pytest.approx(1 / 2)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {"query_id": "same", "query": "one", "relevant_ids": ["a"]},
                {"query_id": "same", "query": "two", "relevant_ids": ["b"]},
            ],
            "duplicate query_id",
        ),
        ([{"query_id": "q", "query": "   ", "relevant_ids": []}], "query must"),
        (
            [{"query_id": "q", "query": "valid", "relevant_ids": "kb-001"}],
            "JSON array",
        ),
        (
            [{"query_id": "q", "query": "valid", "relevant_ids": [1]}],
            "non-empty string",
        ),
    ],
)
def test_loader_rejects_invalid_cases(
    tmp_path: Path,
    records: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_cases(path)


def test_practice_queries_work_with_tfidf_retriever() -> None:
    cases = load_evaluation_cases(ROOT / "data" / "practice" / "rag_queries.jsonl")
    articles = load_knowledge_base(ROOT / "data" / "raw" / "knowledge_base.jsonl")
    retriever = TfidfRetriever().fit(articles)

    report = evaluate_tfidf_retriever(retriever, cases, top_k=3)

    assert report.total_cases == 13
    assert report.answerable_cases == 12
    assert report.no_answer_cases == 1
    assert report.hit_rate_at_k is not None and report.hit_rate_at_k >= 0.9
    assert report.recall_at_k is not None and report.recall_at_k >= 0.9
    assert report.mrr_at_k is not None and report.mrr_at_k >= 0.8
    assert report.answerable_coverage == 1.0
    assert report.abstention_accuracy is not None
    assert report.no_answer_coverage is not None
    assert report.abstention_accuracy + report.no_answer_coverage == pytest.approx(1.0)
