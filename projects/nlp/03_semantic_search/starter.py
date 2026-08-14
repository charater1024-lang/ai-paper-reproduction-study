"""Project 03 starter: implement and audit a dense semantic-search index."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class SearchProjectConfig:
    knowledge_path: Path
    query: str = "카드가 이중으로 결제됐어요"
    top_k: int = 3
    components: int = 64

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query는 비어 있을 수 없습니다.")
        if self.top_k <= 0 or self.components <= 0:
            raise ValueError("top_k와 components는 양수여야 합니다.")


@dataclass(frozen=True, slots=True)
class ArticleRecord:
    article_id: str
    title: str
    body: str


class DenseSearchIndex:
    """Own fitted transforms, document embeddings, and ID alignment."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.document_ids: tuple[str, ...] = ()
        self.document_embeddings: np.ndarray | None = None
        self.vectorizer: object | None = None
        self.reducer: object | None = None

    def fit(
        self,
        document_ids: Sequence[str],
        documents: Sequence[str],
    ) -> DenseSearchIndex:
        """Build character TF-IDF -> safe TruncatedSVD -> L2 normalization.

        TODO 1: fit a character n-gram TF-IDF matrix ``X``.
        TODO 2: choose ``min(requested, rows - 1, columns - 1)`` dimensions.
        TODO 3: normalize dense rows and preserve their source-ID order.
        """

        raise NotImplementedError("TODO 1~3: DenseSearchIndex.fit을 구현하세요.")

    def embed_query(self, query: str) -> np.ndarray:
        """Transform a query with the already fitted document pipeline."""

        # TODO 4: validate fitted state; never fit a new vectorizer on the query.
        raise NotImplementedError("TODO 4: embed_query를 구현하세요.")

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return deterministic descending cosine scores with source IDs."""

        # TODO 5: cosine rank; use document order to break equal-score ties.
        raise NotImplementedError("TODO 5: search를 구현하세요.")


class SemanticSearchProject:
    """Mirror the solution's article -> index -> diagnostics -> ranking flow."""

    def __init__(self, config: SearchProjectConfig) -> None:
        self.config = config
        self.index = DenseSearchIndex(config.components)

    def load_articles(self) -> list[ArticleRecord]:
        articles: list[ArticleRecord] = []
        with self.config.knowledge_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                articles.append(
                    ArticleRecord(
                        article_id=str(row["id"]),
                        title=str(row["title"]),
                        body=str(row["content"]),
                    )
                )
        ids = [article.article_id for article in articles]
        if not articles or len(set(ids)) != len(ids):
            raise ValueError("article은 비어 있지 않고 ID가 고유해야 합니다.")
        return articles

    def build_index(self, articles: list[ArticleRecord]) -> None:
        ids = [article.article_id for article in articles]
        texts = [f"{article.title}\n{article.body}" for article in articles]
        self.index.fit(ids, texts)

    def print_diagnostics(self) -> None:
        embeddings = self.index.document_embeddings
        if embeddings is None:
            raise RuntimeError("build_index를 먼저 호출하세요.")
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"shape={embeddings.shape}, norm=[{norms.min():.3f}, {norms.max():.3f}]")

    def embed_query(self) -> np.ndarray:
        return self.index.embed_query(self.config.query)

    def search(self) -> list[tuple[str, float]]:
        return self.index.search(self.config.query, self.config.top_k)

    def run(self) -> list[tuple[str, float]]:
        articles = self.load_articles()
        self.build_index(articles)
        self.print_diagnostics()
        print(f"query_shape={self.embed_query().shape}")
        results = self.search()
        for rank, (article_id, score) in enumerate(results, start=1):
            print(f"{rank}. [{article_id}] cosine={score:.3f}")
        return results


def main() -> None:
    config = SearchProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
    )
    try:
        SemanticSearchProject(config).run()
    except NotImplementedError as error:
        print(f"연습 대기: {error}")


if __name__ == "__main__":
    main()
