"""Run an offline LangChain RAG pipeline or optionally inject ChatOpenAI."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.langchain_rag import (  # noqa: E402
    LangChainRAG,
    LangChainRAGResponse,
    ScoredDocument,
    create_openai_model,
)


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


class LangChainRagProject:
    """Expose framework composition without hiding retrieval evidence."""

    def __init__(self, config: LangChainProjectConfig) -> None:
        self.config = config

    def build_pipeline(self) -> LangChainRAG:
        """Create documents, chunks, embeddings, vector store, and retriever."""

        rag = LangChainRAG.from_jsonl(
            self.config.knowledge_path,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
        )
        if not rag.chunks:
            raise AssertionError("text splitting produced no chunks")
        return rag

    def retrieve(self, rag: LangChainRAG) -> list[ScoredDocument]:
        """Inspect source IDs and chunk IDs before calling any model."""

        results = rag.search(self.config.query)
        for result in results:
            metadata = result.document.metadata
            if "source_id" not in metadata or "chunk_id" not in metadata:
                raise AssertionError("chunk provenance metadata is incomplete")
        return results

    def build_optional_model(self) -> object | None:
        """Keep network-backed generation an explicit opt-in dependency."""

        if not self.config.use_openai:
            return None
        return create_openai_model(self.config.model_name)

    def answer(self, rag: LangChainRAG) -> LangChainRAGResponse:
        """Use local extractive fallback unless a chat model was injected."""

        model = self.build_optional_model()
        response = rag.ask(self.config.query, model=model)
        if response.question != self.config.query:
            raise AssertionError("response question differs from the request")
        return response

    @staticmethod
    def print_evidence(results: list[ScoredDocument]) -> None:
        for rank, result in enumerate(results, start=1):
            metadata = result.document.metadata
            print(
                f"{rank}. [{metadata['source_id']}] "
                f"score={result.score:.3f} chunk={metadata['chunk_id']}"
            )

    def run(self) -> LangChainRAGResponse:
        rag = self.build_pipeline()
        print(f"chunks={len(rag.chunks)}, embedding={type(rag.embeddings).__name__}")
        print(f"질문: {self.config.query}\n")
        results = self.retrieve(rag)
        self.print_evidence(results)

        if self.config.show_prompt:
            print("\n--- prompt preview (API 호출 없음) ---")
            print(rag.prompt_preview(self.config.query))

        response = self.answer(rag)
        print("\n--- answer ---")
        print(response.answer)
        print(f"sources={list(response.sources)}, abstained={response.abstained}")
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="429 오류는 어떻게 안전하게 재시도하나요?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_CHAT_MODEL"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LangChainProjectConfig(
        knowledge_path=ROOT / "data/raw/knowledge_base.jsonl",
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        show_prompt=args.show_prompt,
        use_openai=args.openai,
        model_name=args.model,
    )
    LangChainRagProject(config).run()


if __name__ == "__main__":
    main()
