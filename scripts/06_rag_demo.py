"""TF-IDF 검색과 RAG 프롬프트 조립을 관찰하는 CLI.

예: python scripts/06_rag_demo.py --query "카드가 두 번 결제됐어요" --show-prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.retrieval import (  # noqa: E402
    TfidfRetriever,
    build_grounded_prompt,
    load_knowledge_base,
    recall_at_k,
)

EVAL_CASES = (
    ("비밀번호를 여러 번 틀려서 계정이 막혔어요", "kb-002"),
    ("카드에서 동일한 금액이 두 번 빠져나갔습니다", "kb-003"),
    ("API 호출이 너무 많다는 429 오류", "kb-010"),
    ("배송 추적이 이틀째 그대로예요", "kb-006"),
    ("내 개인정보를 내려받고 계정을 지우고 싶어요", "kb-012"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="중복 결제 환불은 얼마나 걸리나요?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def print_results(
    retriever: TfidfRetriever, query: str, top_k: int, show_prompt: bool
) -> None:
    results = retriever.search(query, top_k=top_k)
    print(f"\n질문: {query}")
    if not results:
        print("관련 문서를 찾지 못했습니다.")
        return
    for result in results:
        print(
            f"{result.rank}. [{result.article.id}] {result.article.title} score={result.score:.3f}"
        )
        print(f"   {result.article.content}")
    if show_prompt:
        print("\n--- LLM에 전달할 grounded prompt ---")
        print(build_grounded_prompt(query, results))


def main() -> None:
    args = parse_args()
    kb_path = ROOT / "data" / "raw" / "knowledge_base.jsonl"
    retriever = TfidfRetriever().fit(load_knowledge_base(kb_path))
    print(
        f"지식 문서 {len(retriever.articles)}개, 특징 {len(retriever.vectorizer.vocabulary_):,}개"
    )
    if args.evaluate:
        print(
            f"고정 평가셋 recall@{args.top_k}: {recall_at_k(retriever, EVAL_CASES, args.top_k):.3f}"
        )
    print_results(retriever, args.query, args.top_k, args.show_prompt)

    if args.interactive:
        print("\n종료하려면 빈 줄을 입력하세요.")
        while query := input("query> ").strip():
            print_results(retriever, query, args.top_k, args.show_prompt)


if __name__ == "__main__":
    main()
