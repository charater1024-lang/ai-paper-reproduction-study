"""Add Korean function/API notes without replacing learner-written code.

The generated paper tracks receive the same notes from their builders.  This
utility exists for the hand-authored 00--12 notebooks and for already opened
base exercise notebooks, where rebuilding a pair could erase a learner's work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from notebook_api_explanations import annotate_notebook, annotate_pair

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXERCISES = NOTEBOOKS / "exercises"
SOLUTIONS = NOTEBOOKS / "solutions"


def source_notebooks() -> list[Path]:
    """Return 00--04 generated inputs and 05--12 notebook-maintained inputs."""

    paths = [
        path
        for path in NOTEBOOKS.glob("[0-9][0-9]_*.ipynb")
        if int(path.name[:2]) <= 12
    ]
    if len(paths) != 13:
        raise AssertionError(f"00~12 원본 노트북은 13개여야 합니다: {len(paths)}")
    return sorted(paths)


def paired_notebooks() -> list[tuple[Path, Path]]:
    """Return all 23 base exercise/solution pairs."""

    solution_paths = sorted(SOLUTIONS.glob("[0-9][0-9]_*.ipynb"))
    pairs = [(EXERCISES / path.name, path) for path in solution_paths]
    missing = [str(exercise) for exercise, _ in pairs if not exercise.is_file()]
    if missing:
        raise FileNotFoundError(f"실습 counterpart가 없습니다: {missing}")
    if len(pairs) != 23:
        raise AssertionError(f"기본 페어는 23개여야 합니다: {len(pairs)}")
    return pairs


def code_state(notebook: nbformat.NotebookNode) -> list[tuple[object, object, object]]:
    """Capture code, outputs, and counters so annotation cannot erase answers."""

    return [
        (
            cell.source,
            cell.get("outputs", []),
            cell.get("execution_count"),
        )
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]


def update_notebook(path: Path, *, check: bool) -> bool:
    notebook = nbformat.read(path, as_version=4)
    before = nbformat.writes(notebook)
    protected_code = code_state(notebook)
    annotate_notebook(notebook)
    if code_state(notebook) != protected_code:
        raise AssertionError(f"annotation changed learner code or output: {path}")
    nbformat.validate(notebook)
    changed = nbformat.writes(notebook) != before
    if changed and not check:
        nbformat.write(notebook, path)
    return changed


def update_pair(exercise_path: Path, solution_path: Path, *, check: bool) -> bool:
    exercise = nbformat.read(exercise_path, as_version=4)
    solution = nbformat.read(solution_path, as_version=4)
    before_exercise = nbformat.writes(exercise)
    before_solution = nbformat.writes(solution)
    protected_exercise = code_state(exercise)
    protected_solution = code_state(solution)
    annotate_pair(exercise, solution)
    if code_state(exercise) != protected_exercise:
        raise AssertionError(f"annotation changed learner answer: {exercise_path}")
    if code_state(solution) != protected_solution:
        raise AssertionError(f"annotation changed solution code: {solution_path}")
    nbformat.validate(exercise)
    nbformat.validate(solution)
    changed = (
        nbformat.writes(exercise) != before_exercise
        or nbformat.writes(solution) != before_solution
    )
    if changed and not check:
        nbformat.write(exercise, exercise_path)
        nbformat.write(solution, solution_path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale notebooks without writing files",
    )
    args = parser.parse_args()

    changed: list[Path] = []
    for path in source_notebooks():
        if update_notebook(path, check=args.check):
            changed.append(path)
    for exercise_path, solution_path in paired_notebooks():
        if update_pair(exercise_path, solution_path, check=args.check):
            changed.extend((exercise_path, solution_path))

    action = "STALE" if args.check else "UPDATED"
    for path in changed:
        print(f"{action}: {path.relative_to(ROOT)}")
    if args.check and changed:
        raise SystemExit(
            "API 해설이 최신이 아닙니다. tools/apply_api_explanations.py를 실행하세요."
        )
    print(f"PASS: checked 13 originals and 23 base pairs; changed={len(changed)}")


if __name__ == "__main__":
    main()
