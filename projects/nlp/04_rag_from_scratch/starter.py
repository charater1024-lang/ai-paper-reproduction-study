"""Project 04 starter: connect retrieval, context, and grounded prompting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


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


def retrieve(retriever: object, question: str, top_k: int) -> list[object]:
    """Return ranked evidence from an already-fitted retriever.

    TODO 2: validate the question, call ``search``, and enforce ``top_k``.
    Keeping fitting outside this function prevents rebuilding the index per query.
    """

    raise NotImplementedError("TODO 2: retrieve를 구현하세요.")


def format_context(results: list[object], max_chars: int) -> str:
    """Keep source IDs while adding complete evidence blocks under a budget."""

    # TODO 3: append blocks atomically so a header is never separated from content.
    raise NotImplementedError("TODO 3: format_context를 구현하세요.")


def build_prompt(question: str, context: str) -> str:
    """Require evidence-only claims, source citations, and abstention."""

    # TODO 4: state what the model must do when context is empty or insufficient.
    raise NotImplementedError("TODO 4: build_prompt를 구현하세요.")


class RagFromScratchProject:
    """Mirror the observable retrieve -> context -> prompt solution flow."""

    def __init__(self, config: RagProjectConfig) -> None:
        self.config = config
        self.retriever: object | None = None

    def build_retriever(self) -> object:
        """Load local JSONL and fit one character TF-IDF index."""

        # TODO 1: load articles, validate unique IDs, then fit the retriever once.
        raise NotImplementedError("TODO 1: build_retriever를 구현하세요.")

    def retrieve(self, retriever: object) -> list[object]:
        return retrieve(retriever, self.config.query, self.config.top_k)

    def build_context(self, results: list[object]) -> str:
        context = format_context(results, self.config.max_context_chars)
        if len(context) > self.config.max_context_chars:
            raise AssertionError("context character budget exceeded")
        return context

    def build_prompt(self, results: list[object]) -> str:
        context = self.build_context(results)
        prompt = build_prompt(self.config.query, context)
        if self.config.query not in prompt:
            raise AssertionError("prompt lost the query")
        return prompt

    @staticmethod
    def print_ranked_results(results: list[object]) -> None:
        """TODO 5: print rank, source ID, title, and retrieval score."""

        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result!r}")

    def run(self) -> tuple[list[object], str, str]:
        retriever = self.build_retriever()
        results = self.retrieve(retriever)
        context = self.build_context(results)
        prompt = self.build_prompt(results)
        self.print_ranked_results(results)
        print(f"context={len(context)}/{self.config.max_context_chars} chars")
        if self.config.show_prompt:
            print(prompt)
        return results, context, prompt


def main() -> None:
    config = RagProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
    )
    try:
        RagFromScratchProject(config).run()
    except NotImplementedError as error:
        print(f"연습 대기: {error}")


if __name__ == "__main__":
    main()
