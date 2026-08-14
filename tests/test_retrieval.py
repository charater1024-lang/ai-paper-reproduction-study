from pathlib import Path

from llm_engineering_lab.retrieval import (
    KnowledgeArticle,
    TfidfRetriever,
    build_grounded_prompt,
    load_knowledge_base,
    recall_at_k,
)

ROOT = Path(__file__).resolve().parents[1]


def test_search_finds_duplicate_payment_article() -> None:
    articles = load_knowledge_base(ROOT / "data" / "raw" / "knowledge_base.jsonl")
    results = (
        TfidfRetriever().fit(articles).search("카드가 이중으로 결제됐어요", top_k=3)
    )
    assert results
    assert results[0].article.id == "kb-003"
    assert results[0].score > 0


def test_exact_error_code_keyword_is_ranked_first() -> None:
    articles = load_knowledge_base(ROOT / "data" / "raw" / "knowledge_base.jsonl")
    results = (
        TfidfRetriever().fit(articles).search("API 요청에서 429 오류가 나요", top_k=1)
    )
    assert results[0].article.id == "kb-010"


def test_empty_query_returns_no_results() -> None:
    article = KnowledgeArticle("1", "로그인", "account", "비밀번호를 재설정하세요")
    assert TfidfRetriever().fit([article]).search("   ") == []


def test_prompt_contains_sources_and_question() -> None:
    article = KnowledgeArticle("kb-x", "제목", "test", "검증된 내용")
    results = TfidfRetriever(min_score=0).fit([article]).search("검증", 1)
    prompt = build_grounded_prompt("무엇을 해야 하나요?", results)
    assert "kb-x" in prompt
    assert "무엇을 해야 하나요?" in prompt
    assert "추측하지 말고" in prompt


def test_recall_at_k() -> None:
    articles = [
        KnowledgeArticle("billing", "중복 결제", "결제", "카드 환불 처리"),
        KnowledgeArticle("login", "로그인", "계정", "비밀번호 재설정"),
    ]
    retriever = TfidfRetriever(min_score=0).fit(articles)
    score = recall_at_k(retriever, [("카드 중복 결제", "billing")], top_k=1)
    assert score == 1.0
