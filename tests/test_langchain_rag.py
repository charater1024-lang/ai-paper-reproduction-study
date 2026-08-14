from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from llm_engineering_lab.langchain_rag import (
    DEFAULT_ABSTENTION,
    LangChainRAG,
    LocalHashEmbeddings,
    build_lcel_rag_chain,
    format_documents,
    load_jsonl_documents,
    split_documents,
)

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "raw" / "knowledge_base.jsonl"


def test_local_hash_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = LocalHashEmbeddings(dimensions=128)
    first = np.asarray(embeddings.embed_query("카드 중복 결제"))
    second = np.asarray(embeddings.embed_query("카드 중복 결제"))
    assert first.shape == (128,)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_load_and_split_preserve_unique_source_metadata() -> None:
    documents = load_jsonl_documents(KB_PATH)
    chunks = split_documents(documents, chunk_size=120, chunk_overlap=20)
    assert len(documents) == 12
    assert len(chunks) >= len(documents)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(chunk.metadata["source_id"].startswith("kb-") for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_offline_pipeline_retrieves_evidence_and_cites_source() -> None:
    rag = LangChainRAG.from_jsonl(KB_PATH, min_score=0.05)
    response = rag.ask("같은 주문이 카드에 두 번 결제됐어요")
    assert not response.abstained
    assert response.sources[0] == "kb-003"
    assert "kb-003" in response.answer
    assert "중복 결제" in response.context


def test_pipeline_abstains_when_no_chunk_passes_threshold() -> None:
    rag = LangChainRAG.from_jsonl(KB_PATH, min_score=0.2)
    response = rag.ask("양자 얽힘 실험 장비 예약")
    assert response.abstained
    assert response.answer == DEFAULT_ABSTENTION
    assert response.sources == ()


def test_lcel_chain_connects_retriever_prompt_and_chat_model() -> None:
    rag = LangChainRAG.from_jsonl(KB_PATH)
    model = FakeListChatModel(responses=["429는 지수 백오프로 재시도하세요. [kb-010]"])
    chain = build_lcel_rag_chain(rag.retriever, model)
    answer = chain.invoke("429 오류는 어떻게 재시도하나요?")
    assert answer.endswith("[kb-010]")


def test_context_budget_and_input_validation() -> None:
    documents = [Document(page_content="가" * 100, metadata={"source_id": "x"})]
    assert len(format_documents(documents, max_chars=60)) <= 60
    with pytest.raises(ValueError, match="question"):
        LangChainRAG.from_jsonl(KB_PATH).ask("  ")
