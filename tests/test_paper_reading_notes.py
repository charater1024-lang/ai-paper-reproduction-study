"""Structural and link checks for the Korean 70-paper reading notes."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "docs" / "paper_reading_notes"
TRACK = ROOT / "notebooks" / "field_reproductions"
FILES = {
    "VISION.md": "vision",
    "NLP_LLM.md": "nlp_llm",
    "GENERATIVE.md": "generative",
    "REINFORCEMENT_LEARNING.md": "reinforcement_learning",
    "GRAPH_RECOMMENDATION.md": "graph_recommendation",
    "SELF_SUPERVISED_MULTIMODAL.md": "self_supervised_multimodal",
    "DISTILLATION_COMPRESSION.md": "distillation_compression",
}


def markdown_links(text: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)


def test_seven_reading_notes_cover_ten_numbered_papers() -> None:
    for filename in FILES:
        text = (NOTES / filename).read_text(encoding="utf-8")
        numbers = re.findall(r"^## (\d{2})\.", text, flags=re.MULTILINE)
        zero_based = [f"{number:02d}" for number in range(10)]
        one_based = [f"{number:02d}" for number in range(1, 11)]
        assert numbers in (zero_based, one_based), filename
        assert "�" not in text
        circled = all(text.count(marker) >= 10 for marker in "①②③④⑤⑥⑦⑧")
        numbered = all(
            len(re.findall(rf"^\s*{number}\. \*\*", text, flags=re.MULTILINE)) >= 10
            for number in range(1, 9)
        )
        assert circled or numbered, filename


def test_each_note_links_all_exercise_solution_pairs() -> None:
    for filename, field_id in FILES.items():
        text = (NOTES / filename).read_text(encoding="utf-8")
        exercise_names = sorted(
            path.name
            for path in (TRACK / field_id / "exercises").glob("[0-9][0-9]_*.ipynb")
        )
        solution_names = sorted(
            path.name
            for path in (TRACK / field_id / "solutions").glob("[0-9][0-9]_*.ipynb")
        )
        assert len(exercise_names) == len(solution_names) == 10
        for notebook_name in exercise_names:
            assert f"/exercises/{notebook_name}" in text
            assert f"/solutions/{notebook_name}" in text


def test_all_local_markdown_links_resolve() -> None:
    for path in [
        NOTES / "README.md",
        NOTES / "RECENT_TOP_TIER.md",
        *(NOTES / name for name in FILES),
    ]:
        for target in markdown_links(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            clean_target = target.split("#", 1)[0]
            assert (path.parent / clean_target).resolve().exists(), (path.name, target)
