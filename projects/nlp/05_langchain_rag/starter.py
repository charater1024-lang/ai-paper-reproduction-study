"""Project 05 starter: build an inspectable LangChain v1 RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class LangChainProjectConfig:
    knowledge_path: Path
    query: str = "429 오류는 어떻게 안전하게 재시도하나요?"
    top_k: int = 3
    min_score: float = 0.08
    show_prompt: bool = False
    use_openai: bool = False
    model_name: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query는 비어 있을 수 없습니다.")
        if self.top_k <= 0:
            raise ValueError("top_k는 양수여야 합니다.")
        if self.use_openai and not self.model_name:
            raise ValueError("OpenAI 사용 시 model_name이 필요합니다.")


def build_documents(knowledge_path: Path) -> list[object]:
    """Map JSONL articles to chunks with stable provenance metadata.

    TODO 1: create ``Document(page_content, metadata, id)`` values.
    TODO 2: split them and retain ``source_id``, ``start_index``, ``chunk_id``.
    """

    raise NotImplementedError("TODO 1~2: build_documents를 구현하세요.")


def build_retriever(documents: list[object], top_k: int) -> object:
    """Create embeddings, an in-memory vector store, and a retriever."""

    # TODO 3: index the chunks once and configure the requested result count.
    raise NotImplementedError("TODO 3: build_retriever를 구현하세요.")


def build_chain(retriever: object, chat_model: object) -> object:
    """Compose retrieval, formatting, prompt, model, and output parser."""

    # TODO 4: make the Runnable input/output mapping explicit.
    raise NotImplementedError("TODO 4: build_chain을 구현하세요.")


@dataclass(frozen=True, slots=True)
class StarterLangChainRag:
    """Retain chunks and retriever instead of returning an opaque chain only."""

    chunks: tuple[object, ...]
    retriever: object


class LangChainRagProject:
    """Mirror solution boundaries while keeping framework objects inspectable."""

    def __init__(self, config: LangChainProjectConfig) -> None:
        self.config = config

    def build_pipeline(self) -> StarterLangChainRag:
        documents = build_documents(self.config.knowledge_path)
        if not documents:
            raise AssertionError("text splitting produced no chunks")
        retriever = build_retriever(documents, self.config.top_k)
        return StarterLangChainRag(tuple(documents), retriever)

    def retrieve(self, rag: StarterLangChainRag) -> list[object]:
        """TODO 5: search and verify source_id/chunk_id on every result."""

        raise NotImplementedError("TODO 5: retrieve를 구현하세요.")

    def build_optional_model(self) -> object | None:
        """TODO 6: create ChatOpenAI only when explicit opt-in is enabled."""

        if not self.config.use_openai:
            return None
        raise NotImplementedError("TODO 6: 선택형 OpenAI model을 구현하세요.")

    def answer(self, rag: StarterLangChainRag) -> object:
        """TODO 7: run local fallback or inject the optional chat model."""

        raise NotImplementedError("TODO 7: answer를 구현하세요.")

    @staticmethod
    def print_evidence(results: list[object]) -> None:
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result!r}")

    def run(self) -> object:
        rag = self.build_pipeline()
        results = self.retrieve(rag)
        self.print_evidence(results)
        response = self.answer(rag)
        print(response)
        return response


def main() -> None:
    config = LangChainProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
        model_name=os.getenv("OPENAI_CHAT_MODEL"),
    )
    try:
        LangChainRagProject(config).run()
    except NotImplementedError as error:
        print(f"연습 대기: {error}")


if __name__ == "__main__":
    main()
