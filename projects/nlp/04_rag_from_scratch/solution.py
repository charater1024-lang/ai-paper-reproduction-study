"""Run a transparent TF-IDF retrieve-then-prompt RAG baseline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.retrieval import (  # noqa: E402
    SearchResult,
    TfidfRetriever,
    build_grounded_prompt,
    format_context,
    load_knowledge_base,
)


@dataclass(frozen=True, slots=True)
class RagProjectConfig:
    knowledge_path: Path
    query: str = "카드에서 같은 금액이 두 번 결제됐어요"
    top_k: int = 3
    max_context_chars: int = 1_200
    show_prompt: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query는 비어 있을 수 없습니다.")
        if self.top_k <= 0 or self.max_context_chars <= 0:
            raise ValueError("top_k와 max_context_chars는 양수여야 합니다.")


class RagFromScratchProject:
    """Keep retrieval, context budgeting, and prompt grounding observable."""

    def __init__(self, config: RagProjectConfig) -> None:
        self.config = config
        self.retriever: TfidfRetriever | None = None

    def build_retriever(self) -> TfidfRetriever:
        """Fit the sparse index once instead of rebuilding it for each query."""

        articles = load_knowledge_base(self.config.knowledge_path)
        retriever = TfidfRetriever().fit(articles)
        self.retriever = retriever
        return retriever

    def retrieve(self, retriever: TfidfRetriever) -> list[SearchResult]:
        """Return ranked evidence before any prompt or answer is constructed."""

        results = retriever.search(self.config.query, top_k=self.config.top_k)
        if len(results) > self.config.top_k:
            raise AssertionError("retriever exceeded top_k")
        return results

    def build_context(self, results: list[SearchResult]) -> str:
        """Apply an explicit character budget while preserving source IDs."""

        context = format_context(results, max_chars=self.config.max_context_chars)
        if len(context) > self.config.max_context_chars:
            raise AssertionError("context exceeded its character budget")
        if results and not all(result.article.id in context for result in results):
            raise AssertionError("context must retain every included source ID")
        return context

    def build_prompt(self, results: list[SearchResult]) -> str:
        """Require evidence-only answers, citations, and abstention."""

        prompt = build_grounded_prompt(self.config.query, results)
        if self.config.query not in prompt:
            raise AssertionError("grounded prompt lost the user query")
        return prompt

    @staticmethod
    def print_ranked_results(results: list[SearchResult]) -> None:
        for result in results:
            print(
                f"{result.rank}. [{result.article.id}] "
                f"{result.article.title} score={result.score:.3f}"
            )

    def run(self) -> tuple[list[SearchResult], str, str]:
        retriever = self.build_retriever()
        results = self.retrieve(retriever)
        context = self.build_context(results)
        prompt = self.build_prompt(results)

        print(f"질문: {self.config.query}")
        self.print_ranked_results(results)
        print(f"\ncontext: {len(context)}/{self.config.max_context_chars} chars")
        print(context or "관련 근거 없음")
        if self.config.show_prompt:
            print("\n--- grounded prompt ---")
            print(prompt)
        return results, context, prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="카드에서 같은 금액이 두 번 결제됐어요")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=1_200)
    parser.add_argument("--show-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RagProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
        query=args.query,
        top_k=args.top_k,
        max_context_chars=args.max_chars,
        show_prompt=args.show_prompt,
    )
    RagFromScratchProject(config).run()


if __name__ == "__main__":
    main()
