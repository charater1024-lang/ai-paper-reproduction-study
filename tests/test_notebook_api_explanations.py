"""Tests for the first-use API explanation layer."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

from tools.notebook_api_explanations import (
    API_EXPLANATIONS_END,
    API_EXPLANATIONS_START,
    API_FORMULAS,
    API_REGISTRY,
    annotate_notebook,
    annotate_pair,
    append_api_notes,
    detect_api_keys,
    render_api_notes,
)


def _generated_block(markdown: str) -> str:
    start = markdown.index(API_EXPLANATIONS_START)
    end = markdown.index(API_EXPLANATIONS_END) + len(API_EXPLANATIONS_END)
    return markdown[start:end]


def _markdown_before_code(notebook: nbformat.NotebookNode) -> list[str]:
    notes: list[str] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if index == 0 or notebook.cells[index - 1].cell_type != "markdown":
            notes.append("")
            continue
        notes.append(notebook.cells[index - 1].source)
    return notes


def test_registry_has_broad_structured_coverage() -> None:
    required = {
        "np.allclose",
        "np.isclose",
        "torch.no_grad",
        "torch.inference_mode",
        "Optimizer.zero_grad",
        "Tensor.backward",
        "Optimizer.step",
        "torch.optim.AdamW",
        "Tensor.reshape",
        "Tensor.permute",
        "nn.ConvTranspose2d",
        "F.binary_cross_entropy_with_logits",
        "DataLoader",
        "np.random.default_rng",
        "pd.read_csv",
        "train_test_split",
        "Estimator.predict_proba",
        "ACCELERATOR.move",
        "dataclass",
        "perf_counter",
    }

    assert len(API_REGISTRY) >= 70
    assert required <= API_REGISTRY.keys()
    for entry in API_REGISTRY.values():
        assert entry.signature
        assert entry.role
        assert entry.inputs_return
        assert entry.reason
        assert entry.caution
        assert entry.aliases


def test_formula_registry_covers_high_value_mathematical_apis() -> None:
    required = {
        "np.allclose",
        "torch.optim.Adam",
        "torch.softmax",
        "nn.Linear",
        "nn.Conv2d",
        "nn.GRU",
        "nn.LayerNorm",
        "F.cross_entropy",
        "F.binary_cross_entropy_with_logits",
        "F.mse_loss",
        "F.kl_div",
        "F.normalize",
        "F.cosine_similarity",
    }

    assert len(API_FORMULAS) >= 40
    assert required <= API_FORMULAS.keys()
    assert API_FORMULAS.keys() <= API_REGISTRY.keys()
    for note in API_FORMULAS.values():
        assert note.equation
        assert note.symbols
        assert note.code_bridge
        assert note.intuition
        assert not any(
            character in note.equation for character in ("\x08", "\x0c", "\r", "\t")
        )


def test_detect_api_keys_uses_ast_aliases_methods_and_source_order() -> None:
    source = """
import numpy as numerical
import torch
import torch.nn as layers
from torch.nn import functional as func
from dataclasses import dataclass

@dataclass
class Batch:
    values: object

close = numerical.allclose(left, right)
projector = layers.Linear(4, 2)
loss = func.cross_entropy(logits, targets)
reshaped = logits.reshape(2, 3)
with torch.no_grad():
    prediction = projector(reshaped)
dtype = torch.float32
"""

    assert detect_api_keys(source) == [
        "dataclass",
        "np.allclose",
        "nn.Linear",
        "F.cross_entropy",
        "Tensor.reshape",
        "torch.no_grad",
        "torch.float32",
    ]


def test_render_api_notes_only_explains_notebook_first_use() -> None:
    seen: set[str] = set()

    first = render_api_notes(
        "np.allclose(first, second)\nnp.isclose(first, second)",
        seen,
    )
    second = render_api_notes(
        "np.allclose(first, second)\nwith torch.no_grad():\n    pass",
        seen,
    )
    third = render_api_notes("print('중요 API 없음')", seen)

    assert first.count("#### `np.allclose") == 1
    assert first.count("#### `np.isclose") == 1
    assert "np.allclose" not in second
    assert "torch.no_grad" in second
    assert third == ""
    assert {"np.allclose", "np.isclose", "torch.no_grad"} <= seen


def test_allclose_note_explains_tolerance_and_exact_equality_difference() -> None:
    notes = render_api_notes("np.allclose(actual, expected)", set())

    assert "abs(a-b) <= atol + rtol*abs(b)" in notes
    assert "Python `bool` 하나" in notes
    assert "`==`와 달리" in notes
    assert "비대칭" in notes
    assert "1보다 매우 작은 값" in notes
    assert "numpy.org/doc/stable/reference/generated/numpy.allclose.html" in notes
    assert "##### 관련 수식과 코드 연결" in notes
    assert r"\lvert a_i-b_i\rvert" in notes
    assert "**기호:**" in notes
    assert "**수식 → 코드:**" in notes
    assert "**직관:**" in notes
    assert "**shape 확인:**" in notes
    assert notes.count("$$") == 2


def test_loss_note_renders_equation_and_tensor_contract() -> None:
    notes = render_api_notes("loss = F.cross_entropy(logits, labels)", set())

    assert r"\mathcal{L}_{CE}" in notes
    assert r"\sum_{c=1}^{C}" in notes
    assert "logits `[N,C,...]`" in notes
    assert "log_softmax" in notes
    assert notes.count("$$") == 2


def test_every_formula_renders_as_balanced_block_math() -> None:
    for key, entry in API_REGISTRY.items():
        if key not in API_FORMULAS:
            continue
        alias = entry.aliases[0]
        notes = render_api_notes(f"result = {alias}(value)", set())
        if not notes:
            notes = render_api_notes(f"result = value{alias}(other)", set())
        assert notes.count("$$") == 2, key
        assert "\n$$\n" in notes, key
        assert API_FORMULAS[key].equation in notes, key


def test_shared_equation_is_rendered_once_and_each_api_is_individually_collapsible() -> (
    None
):
    notes = render_api_notes(
        "linear = nn.Linear(4, 2)\noutput = F.linear(values, weight)",
        set(),
    )

    assert notes.count(r"y=xW^{\mathsf T}+b") == 1
    assert "같은 수식 계약" in notes
    assert "API 2개 · 수식 1개" in notes
    assert notes.count("<details>") == 3
    assert notes.count("</details>") == 3


def test_nested_details_and_latex_survive_notebook_html_rendering() -> None:
    notes = render_api_notes(
        "loss = F.cross_entropy(logits, labels)",
        set(),
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(notes)])

    html, _ = HTMLExporter().from_notebook_node(notebook)

    assert html.count("<details>") == 2
    assert html.count("</details>") == 2
    assert r"\mathcal{L}_{CE}" in html
    assert "수식 → 코드" in html


def test_append_api_notes_replaces_block_idempotently_and_can_remove_it() -> None:
    seen: set[str] = set()
    code = "np.allclose(actual, expected)"

    once = append_api_notes("### 검증", code, seen)
    twice = append_api_notes(once, code, seen)
    removed = append_api_notes(once, "print('done')", set())

    assert twice == once
    assert once.count(API_EXPLANATIONS_START) == 1
    assert once.count(API_EXPLANATIONS_END) == 1
    assert removed == "### 검증"


def test_annotate_notebook_mutates_in_place_and_inserts_deterministic_cell() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("### 준비"),
            nbformat.v4.new_code_cell("values = np.asarray([1.0, 2.0])"),
            nbformat.v4.new_code_cell("assert np.allclose(values, values)"),
            nbformat.v4.new_code_cell("assert np.allclose(values, values)"),
        ]
    )

    returned = annotate_notebook(notebook)
    first_result = copy.deepcopy(notebook)
    annotate_notebook(notebook)

    assert returned is notebook
    assert notebook == first_result
    assert notebook.cells[2].cell_type == "markdown"
    assert notebook.cells[2].id == "api-note-002"
    assert notebook.cells[2].metadata["api_explanations_inserted"] is True
    assert "np.asarray" in notebook.cells[0].source
    assert (
        sum(
            cell.source.count("#### `np.allclose")
            for cell in notebook.cells
            if cell.cell_type == "markdown"
        )
        == 1
    )
    nbformat.validate(notebook)


def test_annotate_pair_uses_solution_code_and_writes_identical_api_blocks() -> None:
    exercise = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("### 직접 구현"),
            nbformat.v4.new_code_cell("raise NotImplementedError('TODO')"),
        ]
    )
    solution = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("### 정답 구현"),
            nbformat.v4.new_code_cell(
                "assert np.allclose(actual, expected)\n"
                "with torch.no_grad():\n"
                "    prediction = model(batch)"
            ),
        ]
    )

    returned_exercise, returned_solution = annotate_pair(exercise, solution)
    exercise_note = _markdown_before_code(exercise)[0]
    solution_note = _markdown_before_code(solution)[0]

    assert returned_exercise is exercise
    assert returned_solution is solution
    assert _generated_block(exercise_note) == _generated_block(solution_note)
    assert "np.allclose" in exercise_note
    assert "torch.no_grad" in exercise_note
    assert "actual" not in exercise_note
    assert "expected" not in exercise_note


def test_legacy_pair_keeps_lesson_titles_after_api_notes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tools_path = str(Path("tools").resolve())
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    legacy = importlib.import_module("build_legacy_pairs")
    exercise_dir = tmp_path / "exercises"
    solution_dir = tmp_path / "solutions"
    monkeypatch.setattr(legacy, "EXERCISES", exercise_dir)
    monkeypatch.setattr(legacy, "SOLUTIONS", solution_dir)

    source = Path("notebooks/00_environment_and_jupyterlab.ipynb")
    exercise_path, _, target_count = legacy.build_pair(source)
    exercise = nbformat.read(exercise_path, as_version=4)
    todo_titles = [
        cell.source.splitlines()[0]
        for cell in exercise.cells
        if cell.cell_type == "code" and cell.source.startswith("# TODO")
    ]

    assert len(todo_titles) == target_count == 3
    assert all("np.allclose" not in title for title in todo_titles)
    assert all("torch." not in title for title in todo_titles)
    assert any(
        API_EXPLANATIONS_START in cell.source
        for cell in exercise.cells
        if cell.cell_type == "markdown"
    )


def test_explanation_sources_keep_physical_lines_readable() -> None:
    paths = (
        Path("tools/notebook_api_explanations.py"),
        Path("tests/test_notebook_api_explanations.py"),
    )
    failures = [
        f"{path}:{line_number} has {len(line)} characters"
        for path in paths
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if len(line) > 100
    ]

    assert not failures, "\n".join(failures)
