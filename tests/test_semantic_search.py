from pathlib import Path

import numpy as np
import pytest

from llm_engineering_lab.retrieval import KnowledgeArticle, load_knowledge_base
from llm_engineering_lab.semantic_search import DenseSemanticRetriever

ROOT = Path(__file__).resolve().parents[1]


def _small_corpus() -> list[KnowledgeArticle]:
    return [
        KnowledgeArticle("billing", "중복 결제", "결제", "카드 승인과 환불 처리"),
        KnowledgeArticle("login", "로그인 실패", "계정", "비밀번호 재설정"),
    ]


def test_small_corpus_adjusts_components_and_exposes_normalized_embeddings() -> None:
    retriever = DenseSemanticRetriever(n_components=128).fit(_small_corpus())

    assert retriever.embedding_shape == (2, 2)
    assert retriever.embedding_diagnostics.effective_components == 2
    assert retriever.embedding_diagnostics.requested_components == 128
    assert retriever.embedding_diagnostics.vocabulary_size > 2
    np.testing.assert_allclose(retriever.embedding_norms, (1.0, 1.0), atol=1e-12)

    exported = retriever.document_embeddings
    exported[0, 0] = 99.0
    assert retriever.document_embeddings[0, 0] != 99.0


def test_fit_and_search_are_deterministic() -> None:
    first = DenseSemanticRetriever(random_state=7).fit(_small_corpus())
    second = DenseSemanticRetriever(random_state=7).fit(_small_corpus())

    np.testing.assert_allclose(
        first.document_embeddings, second.document_embeddings, atol=1e-12
    )
    first_results = first.search("카드가 두 번 결제됐어요", top_k=2)
    second_results = second.search("카드가 두 번 결제됐어요", top_k=2)
    assert [(item.article.id, item.rank) for item in first_results] == [
        (item.article.id, item.rank) for item in second_results
    ]
    np.testing.assert_allclose(
        [item.score for item in first_results],
        [item.score for item in second_results],
        atol=1e-12,
    )


def test_duplicate_and_blank_ids_are_rejected() -> None:
    duplicate = [
        KnowledgeArticle("same", "첫 문서", "테스트", "첫 내용"),
        KnowledgeArticle(" same ", "둘째 문서", "테스트", "둘째 내용"),
    ]
    with pytest.raises(ValueError, match="unique"):
        DenseSemanticRetriever().fit(duplicate)
    with pytest.raises(ValueError, match="blank"):
        DenseSemanticRetriever().fit([KnowledgeArticle(" ", "제목", "테스트", "내용")])


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected(query: str) -> None:
    retriever = DenseSemanticRetriever().fit(_small_corpus())
    with pytest.raises(ValueError, match="query"):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    retriever = DenseSemanticRetriever().fit(_small_corpus())
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("결제", top_k=top_k)


def test_one_document_corpus_has_one_dense_component() -> None:
    article = KnowledgeArticle("one", "환불", "결제", "환불을 신청하세요")
    retriever = DenseSemanticRetriever(n_components=50).fit([article])

    assert retriever.embedding_shape == (1, 1)
    np.testing.assert_allclose(retriever.embedding_norms, (1.0,), atol=1e-12)
    assert retriever.search("환불", top_k=3)[0].article.id == "one"


def test_knowledge_base_search_finds_duplicate_payment_and_429() -> None:
    articles = load_knowledge_base(ROOT / "data" / "raw" / "knowledge_base.jsonl")
    retriever = DenseSemanticRetriever().fit(articles)

    payment_results = retriever.search("카드가 이중으로 결제됐어요", top_k=3)
    rate_limit_results = retriever.search("API 요청에서 429 오류가 나요", top_k=3)

    assert payment_results[0].article.id == "kb-003"
    assert rate_limit_results[0].article.id == "kb-010"
    assert all(
        -1.0 <= result.score <= 1.0 for result in payment_results + rate_limit_results
    )
