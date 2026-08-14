"""Prompt for one AI field, then open its exercise/solution notebook pair."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FIELDS = {
    "1": ("vision", "컴퓨터 비전"),
    "2": ("nlp_llm", "NLP · LLM"),
    "3": ("generative", "생성모델"),
    "4": ("reinforcement_learning", "강화학습 · 에이전트"),
    "5": ("graph_recommendation", "그래프 · 추천"),
    "6": ("self_supervised_multimodal", "자기지도 · 멀티모달"),
    "7": ("distillation_compression", "지식 증류 · 모델 경량화"),
}
ALIASES = {field_id: field_id for field_id, _ in FIELDS.values()}
ALIASES.update({key: field_id for key, (field_id, _) in FIELDS.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", nargs="?", help="1~7 또는 field id")
    parser.add_argument("number", nargs="?", help="논문 번호 00~09")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    selected = args.field
    if selected is None:
        print("\n분야별 유명 논문 실습")
        for key, (_, title) in FIELDS.items():
            print(f"  {key}: {title}")
        selected = input("\n분야 번호를 입력하세요 [기본 1]: ").strip() or "1"
    field_id = ALIASES.get(selected)
    if field_id is None:
        raise SystemExit(f"[ERROR] 알 수 없는 분야: {selected}")
    launcher = Path(__file__).with_name("start_paired_lab.py")
    command = [sys.executable, str(launcher), "--field", field_id]
    if args.number:
        command.append(args.number)
    if args.smoke_test:
        command.append("--smoke-test")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
