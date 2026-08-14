"""LangChain v1 building blocks for a small, inspectable two-step RAG pipeline.

The default path is deliberately offline: a stateless character hashing embedder
and LangChain's in-memory vector store are enough to practise ``Document``
metadata, splitting, retrieval, prompt composition, and abstention.  A real chat
model can be injected without changing the indexing and retrieval code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import HashingVectorizer

from .retrieval import load_knowledge_base

DEFAULT_ABSTENTION = "관련 근거를 찾지 못했습니다. 질문을 더 구체적으로 작성해 주세요."

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 근거 중심 고객지원 도우미입니다. 아래 <context> 안의 내용만 사용해 "
            "답하세요. 근거가 없으면 모른다고 답하고 추측하지 마세요. 답변 끝에는 사용한 "
            "source_id를 대괄호로 표시하세요.\n\n<context>\n{context}\n</context>",
        ),
        ("human", "{question}"),
    ]
)


class LocalHashEmbeddings(Embeddings):
    """Deterministic, download-free character n-gram embeddings for exercises.

    This is a lexical baseline, not a pretrained semantic model.  It implements
    LangChain's standard ``Embeddings`` interface so learners can later swap in
    OpenAI, Ollama, or Hugging Face embeddings without changing the vector store.
    """

    def __init__(self, dimensions: int = 1_024) -> None:
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._vectorizer.transform(str(text) for text in texts)
        return matrix.toarray().astype(float, copy=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    document: Document
    score: float

    @property
    def source_id(self) -> str:
        return str(self.document.metadata.get("source_id", "unknown"))


@dataclass(frozen=True, slots=True)
class LangChainRAGResponse:
    question: str
    answer: str
    sources: tuple[str, ...]
    context: str
    scores: tuple[float, ...]
    abstained: bool


def load_jsonl_documents(path: str | Path) -> list[Document]:
    """Convert the repository's validated knowledge articles to Documents."""

    source = Path(path)
    documents: list[Document] = []
    for article in load_knowledge_base(source):
        keyword_text = ", ".join(article.keywords)
        page_content = f"제목: {article.title}\n{article.content}"
        if keyword_text:
            page_content += f"\n키워드: {keyword_text}"
        documents.append(
            Document(
                id=article.id,
                page_content=page_content,
                metadata={
                    "source": str(source),
                    "source_id": article.id,
                    "title": article.title,
                    "category": article.category,
                },
            )
        )
    return documents


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = 360,
    chunk_overlap: int = 60,
) -> list[Document]:
    """Split documents while assigning stable, unique chunk metadata."""

    if not documents:
        raise ValueError("documents must not be empty")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    raw_chunks = splitter.split_documents(list(documents))
    source_counts: dict[str, int] = {}
    chunks: list[Document] = []
    for raw_chunk in raw_chunks:
        metadata = dict(raw_chunk.metadata)
        source_id = str(metadata.get("source_id", raw_chunk.id or "unknown"))
        chunk_index = source_counts.get(source_id, 0)
        source_counts[source_id] = chunk_index + 1
        chunk_id = f"{source_id}:chunk-{chunk_index:03d}"
        metadata.update({"source_id": source_id, "chunk_id": chunk_id})
        chunks.append(
            Document(
                id=chunk_id, page_content=raw_chunk.page_content, metadata=metadata
            )
        )
    return chunks


def format_documents(documents: Sequence[Document], max_chars: int = 4_000) -> str:
    """Format retrieved chunks with visible source boundaries and a size budget."""

    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    blocks: list[str] = []
    used = 0
    for document in documents:
        source_id = document.metadata.get("source_id", "unknown")
        chunk_id = document.metadata.get("chunk_id", document.id or "unknown")
        title = document.metadata.get("title", "제목 없음")
        block = (
            f"[source_id={source_id} | chunk_id={chunk_id} | title={title}]\n"
            f"{document.page_content.strip()}"
        )
        separator_size = 2 if blocks else 0
        remaining = max_chars - used - separator_size
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining >= 40:
                blocks.append(block[: remaining - 1].rstrip() + "…")
            break
        blocks.append(block)
        used += len(block) + separator_size
    return "\n\n".join(blocks)


def build_generation_chain(model: Runnable[Any, Any]) -> Runnable[Any, str]:
    """Create the prompt -> chat model -> text parser portion of a RAG chain."""

    return RAG_PROMPT | model | StrOutputParser()


def build_lcel_rag_chain(
    retriever: Runnable[str, list[Document]],
    model: Runnable[Any, Any],
) -> Runnable[str, str]:
    """Compose the canonical two-step retrieval chain with LangChain Runnables."""

    return (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | model
        | StrOutputParser()
    )


class LangChainRAG:
    """Small two-step RAG service exposing retrieval evidence before generation."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        embeddings: Embeddings | None = None,
        chunk_size: int = 360,
        chunk_overlap: int = 60,
        top_k: int = 3,
        min_score: float = 0.08,
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            raise ValueError("min_score must be a number between -1 and 1")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        self.top_k = top_k
        self.min_score = min_score
        self.embeddings = embeddings or LocalHashEmbeddings()
        self.chunks = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.vector_store.add_documents(documents=self.chunks)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs: Any) -> LangChainRAG:
        return cls(load_jsonl_documents(path), **kwargs)

    def search(self, question: str, top_k: int | None = None) -> list[ScoredDocument]:
        clean_question = question.strip()
        if not clean_question:
            return []
        requested_k = self.top_k if top_k is None else top_k
        if (
            isinstance(requested_k, bool)
            or not isinstance(requested_k, int)
            or requested_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")
        pairs = self.vector_store.similarity_search_with_score(
            query=clean_question,
            k=requested_k,
        )
        return [ScoredDocument(document, float(score)) for document, score in pairs]

    def prompt_preview(self, question: str) -> str:
        """Render the exact prompt messages without calling a model provider."""

        scored = self.search(question)
        documents = [item.document for item in scored if item.score >= self.min_score]
        context = format_documents(documents)
        return RAG_PROMPT.invoke(
            {"question": question.strip(), "context": context}
        ).to_string()

    def ask(
        self,
        question: str,
        *,
        model: Runnable[Any, Any] | None = None,
    ) -> LangChainRAGResponse:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question must not be blank")
        scored = self.search(clean_question)
        accepted = [item for item in scored if item.score >= self.min_score]
        if not accepted:
            return LangChainRAGResponse(
                question=clean_question,
                answer=DEFAULT_ABSTENTION,
                sources=(),
                context="",
                scores=tuple(item.score for item in scored),
                abstained=True,
            )

        documents = [item.document for item in accepted]
        context = format_documents(documents)
        sources = tuple(dict.fromkeys(item.source_id for item in accepted))
        if model is None:
            # A deterministic extractive fallback keeps the complete pipeline
            # runnable without pretending that a local lexical baseline is an LLM.
            lead = documents[0].page_content.strip()
            answer = f"검색된 최상위 근거:\n{lead}\n\n출처: [{sources[0]}]"
        else:
            answer = build_generation_chain(model).invoke(
                {"question": clean_question, "context": context}
            )
        return LangChainRAGResponse(
            question=clean_question,
            answer=answer,
            sources=sources,
            context=context,
            scores=tuple(item.score for item in accepted),
            abstained=False,
        )


def create_openai_model(model_name: str) -> Any:
    """Create an optional ChatOpenAI model with an actionable missing-extra error."""

    if not model_name.strip():
        raise ValueError("model_name must not be blank")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise ImportError(
            'OpenAI 연동에는 `python -m pip install -e ".[openai]"`가 필요합니다.'
        ) from error
    return ChatOpenAI(model=model_name.strip(), temperature=0)
