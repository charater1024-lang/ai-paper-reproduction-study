"""작고 투명한 TF-IDF 검색기와 RAG 프롬프트 조립 도구.

외부 벡터 DB나 임베딩 API 없이 검색 → 순위화 → 컨텍스트 조립의 핵심을
관찰하기 위한 모듈이다. 한국어 형태소 분석기 설치 부담을 피하려고 문자 n-gram을
사용한다. 실무에서는 같은 인터페이스 뒤에 임베딩/하이브리드 검색을 붙여 비교한다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True, slots=True)
class KnowledgeArticle:
    id: str
    title: str
    category: str
    content: str
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> KnowledgeArticle:
        required = ("id", "title", "category", "content")
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(f"지식 문서 필수 필드가 비었습니다: {missing}")
        return cls(
            id=str(raw["id"]),
            title=str(raw["title"]),
            category=str(raw["category"]),
            content=str(raw["content"]),
            keywords=tuple(str(value) for value in raw.get("keywords", [])),
        )

    @property
    def searchable_text(self) -> str:
        return " ".join((self.title, self.category, *self.keywords, self.content))


@dataclass(frozen=True, slots=True)
class SearchResult:
    article: KnowledgeArticle
    score: float
    rank: int


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...


class TfidfRetriever:
    """문자 n-gram 기반 검색기.

    ``fit``과 ``search``를 분리해 오프라인 인덱싱과 온라인 조회의 경계를 익힌다.
    """

    def __init__(self, min_score: float = 0.01) -> None:
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
            max_features=20_000,
        )
        self.articles: list[KnowledgeArticle] = []
        self.document_matrix: csr_matrix | None = None

    @property
    def is_fitted(self) -> bool:
        return self.document_matrix is not None and bool(self.articles)

    def fit(self, articles: Sequence[KnowledgeArticle]) -> TfidfRetriever:
        if not articles:
            raise ValueError("검색 인덱스를 만들 문서가 없습니다.")
        ids = [article.id for article in articles]
        if len(ids) != len(set(ids)):
            raise ValueError("지식 문서 id는 고유해야 합니다.")
        self.articles = list(articles)
        self.document_matrix = self.vectorizer.fit_transform(
            article.searchable_text for article in self.articles
        ).tocsr()
        return self

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self.is_fitted:
            raise RuntimeError("검색 전에 fit()을 호출하세요.")
        if top_k <= 0:
            raise ValueError("top_k는 양수여야 합니다.")
        clean_query = query.strip()
        if not clean_query:
            return []

        assert self.document_matrix is not None
        query_vector = self.vectorizer.transform([clean_query])
        # TF-IDF 벡터는 L2 정규화되어 있어 내적이 cosine similarity와 같다.
        scores = (self.document_matrix @ query_vector.T).toarray().ravel()
        # 오류 코드처럼 짧고 정확한 식별자는 문자 유사도만으로 희석되기 쉽다.
        # 작은 lexical bonus를 섞어 dense+keyword 하이브리드 검색의 아이디어를 재현한다.
        query_lower = clean_query.lower()
        for index, article in enumerate(self.articles):
            for keyword in article.keywords:
                normalized_keyword = keyword.strip().lower()
                # '카드'처럼 흔한 단어는 오히려 오탐을 늘린다. 401/429/5xx처럼
                # 숫자가 포함된 오류 식별자만 강한 exact-match 신호로 사용한다.
                if (
                    normalized_keyword
                    and re.search(r"\d", normalized_keyword)
                    and normalized_keyword in query_lower
                ):
                    scores[index] += 0.35
        ordered_indices = np.argsort(scores)[::-1][: min(top_k, len(self.articles))]
        return [
            SearchResult(
                article=self.articles[index], score=float(scores[index]), rank=rank
            )
            for rank, index in enumerate(ordered_indices, start=1)
            if scores[index] >= self.min_score
        ]

    def save(self, path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("저장 전에 fit()을 호출하세요.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "min_score": self.min_score,
                "articles": [asdict(article) for article in self.articles],
                "vectorizer": self.vectorizer,
                "document_matrix": self.document_matrix,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TfidfRetriever:
        """신뢰할 수 있는 로컬 파일만 로드한다(joblib은 pickle 기반)."""

        payload = joblib.load(path)
        retriever = cls(min_score=float(payload["min_score"]))
        retriever.articles = [
            KnowledgeArticle.from_dict(item) for item in payload["articles"]
        ]
        retriever.vectorizer = payload["vectorizer"]
        retriever.document_matrix = payload["document_matrix"]
        return retriever


def load_knowledge_base(path: Path) -> list[KnowledgeArticle]:
    """JSONL 지식 문서를 읽고 스키마/id 중복을 검증한다."""

    articles: list[KnowledgeArticle] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                articles.append(KnowledgeArticle.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number} 문서 오류: {error}") from error
    ids = [article.id for article in articles]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}에 중복 id가 있습니다.")
    return articles


def format_context(results: Iterable[SearchResult], max_chars: int = 2_000) -> str:
    """검색 결과를 출처가 보존된 제한 길이 컨텍스트로 만든다."""

    chunks: list[str] = []
    used = 0
    for result in results:
        chunk = (
            f"[출처 {result.article.id} | {result.article.title} | "
            f"score={result.score:.3f}]\n{result.article.content}"
        )
        if used + len(chunk) > max_chars:
            remaining = max_chars - used
            if remaining > 80:
                chunks.append(chunk[:remaining].rstrip() + "…")
            break
        chunks.append(chunk)
        used += len(chunk) + 2
    return "\n\n".join(chunks)


def build_grounded_prompt(question: str, results: Sequence[SearchResult]) -> str:
    """LLM 공급자와 무관한 근거 기반 답변 프롬프트를 만든다."""

    context = format_context(results)
    return f"""당신은 고객지원 엔지니어입니다.
아래 검색 근거만 사용해 답하세요. 근거가 부족하면 추측하지 말고 추가 정보를 요청하세요.
답변 끝에는 사용한 출처 id를 괄호로 표시하세요. 개인정보나 API 키를 요청하지 마세요.

<context>
{context or "관련 근거 없음"}
</context>

사용자 질문: {question.strip()}
답변:"""


def recall_at_k(
    retriever: Retriever, cases: Iterable[tuple[str, str]], top_k: int = 3
) -> float:
    """(질문, 정답 문서 id) 평가 사례의 recall@k를 계산한다."""

    materialized = list(cases)
    if not materialized:
        raise ValueError("평가 사례가 없습니다.")
    hits = sum(
        expected_id in {result.article.id for result in retriever.search(query, top_k)}
        for query, expected_id in materialized
    )
    return hits / len(materialized)
