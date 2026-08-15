"""Regression tests for API notes in the 20-paper and 70-paper tracks."""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


@lru_cache(maxsize=1)
def _modules():
    """Load script-style builders with their tools directory on ``sys.path``."""

    tools_path = str(TOOLS)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    api_notes = importlib.import_module("notebook_api_explanations")
    field_builder = importlib.import_module("build_field_reproductions")
    paper_builder = importlib.import_module("build_paper_reproductions")
    return api_notes, field_builder, paper_builder


@lru_cache(maxsize=1)
def _built_pairs():
    """Build every pair in memory; this test must never rewrite learner files."""

    _, field_builder, paper_builder = _modules()
    pairs = []
    for track, builder in (
        ("field70", field_builder),
        ("paper20", paper_builder),
    ):
        for spec in builder.load_specs():
            pairs.append(
                (
                    track,
                    spec,
                    builder.build_notebook(spec, "exercise"),
                    builder.build_notebook(spec, "solution"),
                )
            )
    return tuple(pairs)


def _code_contexts(notebook: nbformat.NotebookNode):
    """Return each code cell together with its immediately preceding Markdown."""

    contexts = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert index > 0, "a learner code cell cannot be the first notebook cell"
        markdown = notebook.cells[index - 1]
        assert markdown.cell_type == "markdown", (
            f"{cell.id}: code must be immediately preceded by Markdown"
        )
        contexts.append((markdown, cell))
    return contexts


def _api_block(markdown: str) -> str:
    api_notes, _, _ = _modules()
    start_marker = api_notes.API_EXPLANATIONS_START
    end_marker = api_notes.API_EXPLANATIONS_END
    start = markdown.find(start_marker)
    if start < 0:
        assert end_marker not in markdown
        return ""
    assert markdown.count(start_marker) == 1
    assert markdown.count(end_marker) == 1
    end = markdown.index(end_marker, start) + len(end_marker)
    return markdown[start:end].strip()


def test_all_track_code_has_an_immediately_preceding_explanation() -> None:
    field_count = 0
    paper_count = 0

    for track, spec, exercise, solution in _built_pairs():
        exercise_contexts = _code_contexts(exercise)
        solution_contexts = _code_contexts(solution)
        assert len(exercise_contexts) == len(solution_contexts)
        if track == "field70":
            field_count += len(solution_contexts)
            expected_tags = [
                spec.cells[index - 1].tags
                for index, cell in enumerate(spec.cells)
                if cell.cell_type == "code"
            ]
            actual_tags = [
                tuple(markdown.metadata.get("tags", ()))
                for markdown, _ in solution_contexts
            ]
            assert actual_tags == expected_tags
        else:
            paper_count += len(solution_contexts)
            assert all(
                "stage-contract" in markdown.metadata.get("tags", ())
                for markdown, _ in solution_contexts
            )

    assert field_count == 424
    assert paper_count == 107


def test_track_api_notes_use_solution_code_and_match_between_roles() -> None:
    api_notes, _, _ = _modules()
    matched_cell_count = 0

    for _, _, exercise, solution in _built_pairs():
        exercise_contexts = _code_contexts(exercise)
        solution_contexts = _code_contexts(solution)
        seen: set[str] = set()
        for (exercise_markdown, _), (solution_markdown, solution_code) in zip(
            exercise_contexts,
            solution_contexts,
            strict=True,
        ):
            expected = api_notes.render_api_notes(solution_code.source, seen).strip()
            exercise_block = _api_block(exercise_markdown.source)
            solution_block = _api_block(solution_markdown.source)
            assert exercise_block == solution_block == expected
            if expected:
                matched_cell_count += 1

    assert matched_cell_count > 0


def test_api_annotation_does_not_change_or_reveal_role_code() -> None:
    for track, spec, exercise, solution in _built_pairs():
        exercise_sources = [cell.source for _, cell in _code_contexts(exercise)]
        solution_sources = [cell.source for _, cell in _code_contexts(solution)]
        spec_code = [cell for cell in spec.cells if cell.cell_type == "code"]
        assert solution_sources == [cell.solution for cell in spec_code]

        if track == "field70":
            assert exercise_sources == [cell.exercise for cell in spec_code]
        else:
            for raw_cell, exercise_source, solution_source in zip(
                spec_code,
                exercise_sources,
                solution_sources,
                strict=True,
            ):
                if raw_cell.exercise == raw_cell.solution:
                    assert exercise_source == solution_source
                    continue
                assert exercise_source != solution_source
                assert "TODO" in exercise_source

        for (markdown, _), solution_source in zip(
            _code_contexts(exercise),
            solution_sources,
            strict=True,
        ):
            api_block = _api_block(markdown.source)
            assert solution_source.strip() not in api_block


def test_track_notes_cover_numerical_checks_no_grad_and_training_lifecycle() -> None:
    rendered_blocks = {
        _api_block(markdown.source)
        for _, _, _, solution in _built_pairs()
        for markdown, _ in _code_contexts(solution)
        if _api_block(markdown.source)
    }
    expected_explanations = {
        "torch.allclose(input, other,": ("허용 오차", "bool"),
        "torch.no_grad()": ("autograd", "model.eval()", "역할이 다르"),
        "model.train(mode=True)": ("학습 모드",),
        "model.eval()": ("평가 모드",),
        "optimizer.zero_grad(set_to_none=True)": ("gradient", "초기화"),
        "loss.backward()": ("역전파", ".grad"),
        "optimizer.step(closure=None)": ("파라미터", "갱신"),
    }
    for signature, meaning_keywords in expected_explanations.items():
        matching_block = next(
            (block for block in rendered_blocks if signature in block),
            None,
        )
        assert matching_block is not None, f"missing API note for {signature}"
        for keyword in meaning_keywords:
            assert keyword.lower() in matching_block.lower()


def test_track_notes_render_equations_symbols_and_shapes_for_math_apis() -> None:
    rendered = "\n".join(
        _api_block(markdown.source)
        for _, _, _, solution in _built_pairs()
        for markdown, _ in _code_contexts(solution)
        if _api_block(markdown.source)
    )

    expectations = {
        "torch.optim.Adam(params, lr=1e-3, ...)": r"m_t&=\beta_1",
        "F.cross_entropy(input, target, ...)": r"\mathcal{L}_{CE}",
        "F.mse_loss(input, target, reduction='mean')": r"\mathcal{L}_{MSE}",
        "torch.softmax(input, dim, dtype=None)": r"p_i=\frac{e^{z_i}}",
        "nn.Linear(in_features, out_features, bias=True)": r"y=xW^{\mathsf T}+b",
    }
    for signature, equation in expectations.items():
        assert signature in rendered
        assert equation in rendered

    assert "##### 관련 수식과 코드 연결" in rendered
    assert "**기호:**" in rendered
    assert "**수식 → 코드:**" in rendered
    assert "**직관:**" in rendered
    assert "**shape 확인:**" in rendered
