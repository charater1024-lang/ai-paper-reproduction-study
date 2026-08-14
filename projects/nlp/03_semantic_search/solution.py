"""Fit a local dense semantic index and inspect vector-search results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.retrieval import (  # noqa: E402
    KnowledgeArticle,
    SearchResult,
    load_knowledge_base,
)
from llm_engineering_lab.semantic_search import (  # noqa: E402
    DenseSemanticRetriever,
    EmbeddingDiagnostics,
)


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


class SemanticSearchProject:
    """Show the full article-to-vector-to-ranking lifecycle."""

    def __init__(self, config: SearchProjectConfig) -> None:
        self.config = config
        self.retriever = DenseSemanticRetriever(n_components=config.components)

    def load_articles(self) -> list[KnowledgeArticle]:
        """Load the stable source ID, title, and body contract."""

        articles = load_knowledge_base(self.config.knowledge_path)
        if len({article.id for article in articles}) != len(articles):
            raise ValueError("knowledge article ID는 고유해야 합니다.")
        return articles

    def build_index(self, articles: list[KnowledgeArticle]) -> EmbeddingDiagnostics:
        """Fit TF-IDF, safe-sized SVD, and normalized dense embeddings."""

        self.retriever.fit(articles)
        diagnostics = self.retriever.embedding_diagnostics
        if diagnostics.shape[0] != len(articles):
            raise AssertionError("embedding row count does not match article count")
        if not all(np.isclose(norm, 1.0) for norm in diagnostics.norms):
            raise AssertionError("document embeddings must be L2-normalized")
        return diagnostics

    @staticmethod
    def print_diagnostics(diagnostics: EmbeddingDiagnostics) -> None:
        print(
            f"documents={diagnostics.shape[0]}, "
            f"dimensions={diagnostics.shape[1]}, "
            f"vocabulary={diagnostics.vocabulary_size}"
        )
        print(
            f"requested_components={diagnostics.requested_components}, "
            f"effective_components={diagnostics.effective_components}"
        )
        print(
            f"embedding_norm: min={min(diagnostics.norms):.3f}, max={max(diagnostics.norms):.3f}"
        )

    def embed_query(self) -> np.ndarray:
        """Transform a query through exactly the fitted document pipeline."""

        vector = self.retriever.embed_query(self.config.query)
        if vector.ndim != 1:
            raise AssertionError("one query must produce one embedding vector")
        return vector

    def search(self) -> list[SearchResult]:
        """Rank cosine scores with deterministic top-k ordering."""

        results = self.retriever.search(self.config.query, top_k=self.config.top_k)
        scores = [result.score for result in results]
        if scores != sorted(scores, reverse=True):
            raise AssertionError("search results must be sorted by descending score")
        return results

    def run(self) -> list[SearchResult]:
        articles = self.load_articles()
        diagnostics = self.build_index(articles)
        self.print_diagnostics(diagnostics)
        query_vector = self.embed_query()
        print(
            f"query_shape={query_vector.shape}, query_norm={np.linalg.norm(query_vector):.3f}"
        )
        print(f"\n질문: {self.config.query}")
        results = self.search()
        for result in results:
            print(
                f"{result.rank}. [{result.article.id}] "
                f"{result.article.title} cosine={result.score:.3f}"
            )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="카드가 이중으로 결제됐어요")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--components", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SearchProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
        query=args.query,
        top_k=args.top_k,
        components=args.components,
    )
    SemanticSearchProject(config).run()


if __name__ == "__main__":
    main()
