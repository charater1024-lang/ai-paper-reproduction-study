"""Validate Korean function/API/formula explanations in every learner notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nbformat
from notebook_api_explanations import render_api_notes

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


@dataclass(frozen=True, slots=True)
class Pair:
    exercise: Path
    solution: Path


def source_notebooks() -> list[Path]:
    return sorted(
        path
        for path in NOTEBOOKS.glob("[0-9][0-9]_*.ipynb")
        if int(path.name[:2]) <= 12
    )


def paired_notebooks() -> list[Pair]:
    pairs: list[Pair] = []
    roots = [NOTEBOOKS, NOTEBOOKS / "paper_reproductions"]
    for root in roots:
        for solution in sorted((root / "solutions").glob("*.ipynb")):
            pairs.append(Pair(root / "exercises" / solution.name, solution))
    field_root = NOTEBOOKS / "field_reproductions"
    for solution in sorted(field_root.glob("*/solutions/*.ipynb")):
        exercise = solution.parents[1] / "exercises" / solution.name
        pairs.append(Pair(exercise, solution))
    return pairs


def code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def previous_markdown(
    notebook: nbformat.NotebookNode,
    code_ordinal: int,
) -> str:
    seen_codes = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if seen_codes == code_ordinal:
            if index == 0 or notebook.cells[index - 1].cell_type != "markdown":
                return ""
            return str(notebook.cells[index - 1].source)
        seen_codes += 1
    raise IndexError(code_ordinal)


def expected_notes_by_code(
    reference: nbformat.NotebookNode,
) -> list[str]:
    seen: set[str] = set()
    return [render_api_notes(cell.source, seen) for cell in code_cells(reference)]


def note_structure_failures(note: str, label: str) -> list[str]:
    """Check generated HTML folding and MathJax delimiters."""

    failures: list[str] = []
    formula_sections = note.count("##### 관련 수식과 코드 연결")
    math_delimiters = note.count("\n$$\n")
    if math_delimiters != formula_sections * 2:
        failures.append(
            f"{label}: 수식 구분자 불균형 "
            f"(sections={formula_sections}, delimiters={math_delimiters})"
        )
    if note.count("<details>") != note.count("</details>"):
        failures.append(f"{label}: details 태그 불균형")
    if any(character in note for character in ("\x08", "\x0c", "\r", "\t")):
        failures.append(f"{label}: LaTeX를 훼손할 제어문자 발견")
    return failures


def validate_one(path: Path) -> tuple[int, int, list[str]]:
    notebook = nbformat.read(path, as_version=4)
    failures: list[str] = []
    noted_cells = 0
    entries = 0
    for ordinal, expected in enumerate(expected_notes_by_code(notebook)):
        if not expected:
            continue
        label = f"{path.relative_to(ROOT)} code {ordinal + 1}"
        noted_cells += 1
        entries += expected.count("\n#### `")
        failures.extend(note_structure_failures(expected, label))
        actual = previous_markdown(notebook, ordinal)
        if expected not in actual:
            failures.append(f"{label}: API 해설 누락")
    return noted_cells, entries, failures


def validate_pair(pair: Pair) -> tuple[int, int, list[str]]:
    if not pair.exercise.is_file():
        return 0, 0, [f"missing exercise: {pair.exercise.relative_to(ROOT)}"]
    exercise = nbformat.read(pair.exercise, as_version=4)
    solution = nbformat.read(pair.solution, as_version=4)
    exercise_codes = code_cells(exercise)
    solution_codes = code_cells(solution)
    if len(exercise_codes) != len(solution_codes):
        return 0, 0, [f"code count mismatch: {pair.solution.relative_to(ROOT)}"]

    failures: list[str] = []
    noted_cells = 0
    entries = 0
    for ordinal, expected in enumerate(expected_notes_by_code(solution)):
        if not expected:
            continue
        label = f"{pair.solution.relative_to(ROOT)} code {ordinal + 1}"
        noted_cells += 1
        entries += expected.count("\n#### `")
        failures.extend(note_structure_failures(expected, label))
        exercise_note = previous_markdown(exercise, ordinal)
        solution_note = previous_markdown(solution, ordinal)
        if expected not in exercise_note:
            failures.append(
                f"{pair.exercise.relative_to(ROOT)} code {ordinal + 1}: "
                "solution 기준 API 해설 누락"
            )
        if expected not in solution_note:
            failures.append(
                f"{pair.solution.relative_to(ROOT)} code {ordinal + 1}: API 해설 누락"
            )
    return noted_cells, entries, failures


def main() -> None:
    originals = source_notebooks()
    pairs = paired_notebooks()
    if len(originals) != 13:
        raise AssertionError(f"expected 13 original notebooks, got {len(originals)}")
    if len(pairs) != 113:
        raise AssertionError(f"expected 113 notebook pairs, got {len(pairs)}")

    noted_cells = 0
    entries = 0
    formulas = 0
    failures: list[str] = []
    for path in originals:
        notebook = nbformat.read(path, as_version=4)
        formulas += sum(
            note.count("##### 관련 수식과 코드 연결")
            for note in expected_notes_by_code(notebook)
        )
        cell_count, bullet_count, errors = validate_one(path)
        noted_cells += cell_count
        entries += bullet_count
        failures.extend(errors)
    for pair in pairs:
        solution = nbformat.read(pair.solution, as_version=4)
        formulas += sum(
            note.count("##### 관련 수식과 코드 연결")
            for note in expected_notes_by_code(solution)
        )
        cell_count, bullet_count, errors = validate_pair(pair)
        noted_cells += cell_count
        entries += bullet_count
        failures.extend(errors)

    if failures:
        for failure in failures[:40]:
            print(f"FAIL: {failure}")
        if len(failures) > 40:
            print(f"... and {len(failures) - 40} more failures")
        raise SystemExit(1)
    print(
        "PASS: 239 canonical notebooks; "
        f"API-note cells={noted_cells}; detailed entries={entries}; "
        f"formula sections={formulas}"
    )


if __name__ == "__main__":
    main()
