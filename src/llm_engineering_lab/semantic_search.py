"""Local dense semantic search built from TF-IDF and latent semantic analysis.

The retriever deliberately uses only scikit-learn.  Character n-grams work
reasonably well for Korean without a morphological analyzer, while
``TruncatedSVD`` turns the sparse TF-IDF matrix into small dense embeddings.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .retrieval import KnowledgeArticle, SearchResult

FloatArray = NDArray[np.float64]
_WHITESPACE_RE = re.compile(r"\s+")
_HTTP_STATUS_RE = re.compile(r"(?<!\d)[1-5]\d{2}(?!\d)")


def normalize_semantic_text(text: str) -> str:
    """Normalize text and retain short HTTP status codes as strong features.

    Korean character n-grams handle inflection-like surface variation without
    an external tokenizer.  A status code is appended once more because a
    three-character identifier such as ``429`` should not be drowned out by a
    much longer support article.
    """

    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    status_codes = _HTTP_STATUS_RE.findall(normalized)
    if status_codes:
        normalized = f"{normalized} {' '.join(status_codes)}"
    return normalized


@dataclass(frozen=True, slots=True)
class EmbeddingDiagnostics:
    """Small, immutable summary for inspecting the learned dense space."""

    shape: tuple[int, int]
    norms: tuple[float, ...]
    vocabulary_size: int
    requested_components: int
    effective_components: int


class DenseSemanticRetriever:
    """TF-IDF -> TruncatedSVD -> cosine-similarity document retriever.

    ``n_components`` is an upper bound.  During ``fit`` it is reduced to the
    smaller of the corpus size and TF-IDF vocabulary size, so one- and
    two-document practice corpora remain valid.
    """

    def __init__(
        self,
        n_components: int = 128,
        min_score: float = 0.01,
        random_state: int = 42,
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int):
            raise ValueError("n_components must be a positive integer")
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")

        self.n_components = n_components
        self.min_score = float(min_score)
        self.random_state = random_state
        self.articles: list[KnowledgeArticle] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.reducer: TruncatedSVD | None = None
        self._document_embeddings: FloatArray | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            bool(self.articles)
            and self.vectorizer is not None
            and self.reducer is not None
            and self._document_embeddings is not None
        )

    @property
    def document_embeddings(self) -> FloatArray:
        """Return a defensive copy of the normalized document embeddings."""

        embeddings = self._require_embeddings()
        return embeddings.copy()

    @property
    def embedding_shape(self) -> tuple[int, int]:
        """Return ``(document_count, effective_components)`` after fitting."""

        shape = self._require_embeddings().shape
        return int(shape[0]), int(shape[1])

    @property
    def embedding_norms(self) -> tuple[float, ...]:
        """Return each document embedding's L2 norm for an observable pipeline."""

        norms = np.linalg.norm(self._require_embeddings(), axis=1)
        return tuple(float(value) for value in norms)

    @property
    def embedding_diagnostics(self) -> EmbeddingDiagnostics:
        """Describe dimensions and normalization without exposing mutable state."""

        vectorizer, reducer = self._require_pipeline()
        return EmbeddingDiagnostics(
            shape=self.embedding_shape,
            norms=self.embedding_norms,
            vocabulary_size=len(vectorizer.vocabulary_),
            requested_components=self.n_components,
            effective_components=int(reducer.n_components),
        )

    def fit(self, articles: Sequence[KnowledgeArticle]) -> DenseSemanticRetriever:
        materialized = list(articles)
        if not materialized:
            raise ValueError("at least one article is required")

        normalized_ids = [article.id.strip() for article in materialized]
        if any(not article_id for article_id in normalized_ids):
            raise ValueError("article ids must not be blank")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("article ids must be unique")

        texts = [article.searchable_text for article in materialized]
        if any(not text.strip() for text in texts):
            raise ValueError("article searchable text must not be blank")

        vectorizer = TfidfVectorizer(
            preprocessor=normalize_semantic_text,
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
            max_features=20_000,
            lowercase=False,
            dtype=np.float64,
        )
        term_matrix = vectorizer.fit_transform(texts)
        effective_components = min(
            self.n_components,
            int(term_matrix.shape[0]),
            int(term_matrix.shape[1]),
        )
        reducer = TruncatedSVD(
            n_components=effective_components,
            algorithm="randomized",
            n_iter=7,
            random_state=self.random_state,
        )
        # A one-document corpus has zero feature variance.  Its embedding is
        # still well-defined, but sklearn's diagnostic ratio emits a warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            dense_matrix = reducer.fit_transform(term_matrix)
        embeddings = np.asarray(normalize(dense_matrix, norm="l2"), dtype=np.float64)
        embeddings.setflags(write=False)

        # Commit fitted state only after the complete pipeline succeeds.
        self.articles = materialized
        self.vectorizer = vectorizer
        self.reducer = reducer
        self._document_embeddings = embeddings
        return self

    def embed_query(self, query: str) -> FloatArray:
        """Project one non-empty query into the fitted normalized dense space."""

        vectorizer, reducer = self._require_pipeline()
        clean_query = self._validate_query(query)
        sparse_query = vectorizer.transform([clean_query])
        dense_query = reducer.transform(sparse_query)
        normalized_query = normalize(dense_query, norm="l2")
        return np.asarray(normalized_query[0], dtype=np.float64)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Rank articles by cosine similarity with deterministic tie ordering."""

        embeddings = self._require_embeddings()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_embedding = self.embed_query(query)
        if not np.any(query_embedding):
            return []

        scores = np.clip(embeddings @ query_embedding, -1.0, 1.0)
        ordered_indices = np.argsort(-scores, kind="stable")
        results: list[SearchResult] = []
        for index in ordered_indices[: min(top_k, len(self.articles))]:
            score = float(scores[index])
            if score < self.min_score:
                break
            results.append(
                SearchResult(
                    article=self.articles[int(index)],
                    score=score,
                    rank=len(results) + 1,
                )
            )
        return results

    def _require_embeddings(self) -> FloatArray:
        if not self.is_fitted or self._document_embeddings is None:
            raise RuntimeError("call fit() before inspecting embeddings or searching")
        return self._document_embeddings

    def _require_pipeline(self) -> tuple[TfidfVectorizer, TruncatedSVD]:
        if not self.is_fitted or self.vectorizer is None or self.reducer is None:
            raise RuntimeError("call fit() before embedding a query")
        return self.vectorizer, self.reducer

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be blank")
        return clean_query
