"""Data model and cell helpers for field-specific paper reproduction notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent


def clean(text: str) -> str:
    return dedent(text).strip()


@dataclass(frozen=True)
class CellSpec:
    cell_type: str
    exercise: str
    solution: str
    tags: tuple[str, ...] = ()


def markdown(source: str, *tags: str) -> CellSpec:
    text = clean(source)
    return CellSpec("markdown", text, text, tags)


def paired_markdown(exercise: str, solution: str, *tags: str) -> CellSpec:
    return CellSpec("markdown", clean(exercise), clean(solution), tags)


def code(exercise: str, solution: str, *tags: str) -> CellSpec:
    return CellSpec("code", clean(exercise), clean(solution), tags)


def shared_code(source: str, *tags: str) -> CellSpec:
    text = clean(source)
    return CellSpec("code", text, text, tags)


@dataclass(frozen=True)
class FieldPaperSpec:
    field_id: str
    field_title: str
    number: int
    slug: str
    short_title: str
    paper_title: str
    authors: str
    year: int
    primary_url: str
    venue: str
    difficulty: str
    expected_minutes: int
    dataset_file: str
    prerequisites: str
    reproduction_goal: str
    original_scale: str
    mappings: tuple[tuple[str, str, str], ...]
    cells: tuple[CellSpec, ...] = field(default_factory=tuple)

    @property
    def filename(self) -> str:
        return f"{self.number:02d}_{self.slug}.ipynb"

    @property
    def display_title(self) -> str:
        return f"{self.number:02d}. {self.short_title}"


FIELD_ORDER = (
    "vision",
    "nlp_llm",
    "generative",
    "reinforcement_learning",
    "graph_recommendation",
    "self_supervised_multimodal",
    "distillation_compression",
)
