"""Check exercise/solution notebook pairs and optionally execute every solution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "notebooks" / "exercises"
SOLUTIONS = ROOT / "notebooks" / "solutions"
EXPECTED_NUMBERS = [f"{number:02d}" for number in range(23)]
ROLE_MARKER = "<!-- paired-role-banner -->"
ARTIFACT_FILENAMES = (
    "notebook_text_classifier.pt",
    "notebook_10_tiny_lm.pt",
    "rnn_sequence_classifier.pt",
    "ai_lab16_best_tiny_transformer.pt",
    "tiny_shapes_cnn.pt",
    "ai_lab21_gpu_resume.pt",
)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def normalized_artifact_references(source: str) -> str:
    for filename in ARTIFACT_FILENAMES:
        stem, suffix = filename.rsplit(".", maxsplit=1)
        source = source.replace(f"{stem}_exercise.{suffix}", filename)
        source = source.replace(f"{stem}_solution.{suffix}", filename)
    return source


def normalized_code(source: str) -> str:
    source = normalized_artifact_references(source)
    return "\n".join(
        line
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def validate_pair(exercise_path: Path, solution_path: Path) -> tuple[int, int, int]:
    exercise = nbformat.read(exercise_path, as_version=4)
    solution = nbformat.read(solution_path, as_version=4)
    nbformat.validate(exercise)
    nbformat.validate(solution)

    if len(exercise.cells) != len(solution.cells):
        raise AssertionError(f"{exercise_path.name}: 실습/정답 셀 수가 다릅니다")
    exercise_types = [cell.cell_type for cell in exercise.cells]
    solution_types = [cell.cell_type for cell in solution.cells]
    if exercise_types != solution_types:
        raise AssertionError(f"{exercise_path.name}: 실습/정답 셀 종류 순서가 다릅니다")
    if [cell.id for cell in exercise.cells] != [cell.id for cell in solution.cells]:
        raise AssertionError(f"{exercise_path.name}: 실습/정답 셀 ID 순서가 다릅니다")

    exercise_meta = exercise.metadata.get("paired_learning", {})
    solution_meta = solution.metadata.get("paired_learning", {})
    if (
        exercise_meta.get("role") != "exercise"
        or solution_meta.get("role") != "solution"
    ):
        raise AssertionError(
            f"{exercise_path.name}: paired role metadata가 잘못되었습니다"
        )
    if (
        ROLE_MARKER not in exercise.cells[0].source
        or "🟧" not in exercise.cells[0].source
    ):
        raise AssertionError(f"{exercise_path.name}: 실습 역할 배너가 없습니다")
    if (
        ROLE_MARKER not in solution.cells[0].source
        or "🟩" not in solution.cells[0].source
    ):
        raise AssertionError(f"{exercise_path.name}: 정답 역할 배너가 없습니다")

    for index, (exercise_cell, solution_cell) in enumerate(
        zip(exercise.cells[1:], solution.cells[1:], strict=True), start=1
    ):
        if exercise_cell.cell_type == "markdown" and normalized_artifact_references(
            exercise_cell.source
        ) != normalized_artifact_references(solution_cell.source):
            raise AssertionError(
                f"{exercise_path.name}: Markdown 셀 {index}가 서로 다릅니다"
            )

    changed_code_cells = 0
    todo_count = 0
    for index, (exercise_cell, solution_cell) in enumerate(
        zip(exercise.cells, solution.cells, strict=True)
    ):
        if exercise_cell.cell_type != "code":
            continue
        todo_count += exercise_cell.source.count("TODO")
        if normalized_code(exercise_cell.source) != normalized_code(
            solution_cell.source
        ):
            changed_code_cells += 1
            scaffold_markers = ("TODO", "NotImplementedError", "...")
            if not any(marker in exercise_cell.source for marker in scaffold_markers):
                raise AssertionError(
                    f"{exercise_path.name}: 변경된 실습 셀 {index}에 scaffold 표시가 없습니다"
                )

    if any(
        cell.cell_type == "code" and "NotImplementedError" in cell.source
        for cell in solution.cells
    ):
        raise AssertionError(
            f"{exercise_path.name}: 정답에 NotImplementedError가 남았습니다"
        )

    if changed_code_cells == 0 or todo_count == 0:
        raise AssertionError(f"{exercise_path.name}: 숨긴 코드 또는 TODO가 없습니다")
    if exercise_meta.get("hidden_cell_count") != changed_code_cells:
        raise AssertionError(
            f"{exercise_path.name}: 실습 hidden_cell_count가 실제와 다릅니다"
        )
    if solution_meta.get("hidden_cell_count") != changed_code_cells:
        raise AssertionError(
            f"{exercise_path.name}: 정답 hidden_cell_count가 실제와 다릅니다"
        )

    return len(exercise.cells), changed_code_cells, todo_count


def execute_solution(path: Path, timeout: int) -> None:
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        # Use the active interpreter through PATH; the named user kernelspec may point to a
        # previous workspace location after a OneDrive move.
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute(cwd=str(ROOT))


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-solutions", action="store_true")
    parser.add_argument(
        "--timeout", type=int, default=300, help="코드 셀당 제한 시간(초)"
    )
    parser.add_argument("--only", help="예: 16 (특정 번호만 검사)")
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    interpreter_directory = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_directory, os.environ.get("PATH", "")]
    )

    exercise_paths = sorted(EXERCISES.glob("[0-9][0-9]_*.ipynb"))
    solution_paths = sorted(SOLUTIONS.glob("[0-9][0-9]_*.ipynb"))
    if args.only:
        selected_number = args.only.strip().zfill(2)
        if selected_number not in EXPECTED_NUMBERS:
            raise AssertionError(f"--only는 00~22 중 하나여야 합니다: {args.only}")
        exercise_paths = [
            path for path in exercise_paths if path.name[:2] == selected_number
        ]
        solution_paths = [
            path for path in solution_paths if path.name[:2] == selected_number
        ]

    exercise_names = [path.name for path in exercise_paths]
    solution_names = [path.name for path in solution_paths]
    if exercise_names != solution_names:
        missing_solutions = sorted(set(exercise_names) - set(solution_names))
        missing_exercises = sorted(set(solution_names) - set(exercise_names))
        raise AssertionError(
            f"짝이 맞지 않습니다. solution 없음={missing_solutions}, exercise 없음={missing_exercises}"
        )
    if not exercise_paths:
        raise AssertionError("검사할 paired notebook이 없습니다")
    if not args.only:
        actual_numbers = [path.name[:2] for path in exercise_paths]
        if actual_numbers != EXPECTED_NUMBERS:
            raise AssertionError(
                f"노트북 번호가 00~22 연속이 아닙니다: {actual_numbers}"
            )
    elif len(exercise_paths) != 1:
        raise AssertionError(f"--only {selected_number}는 정확히 한 쌍이어야 합니다")

    for exercise_path, solution_path in zip(
        exercise_paths, solution_paths, strict=True
    ):
        cells, changed, todos = validate_pair(exercise_path, solution_path)
        if args.execute_solutions:
            execute_solution(solution_path, args.timeout)
        mode = "EXECUTED" if args.execute_solutions else "VALID"
        print(
            f"{mode:8} {exercise_path.name} "
            f"(cells={cells}, hidden={changed}, TODO={todos})"
        )

    print(f"PASS: {len(exercise_paths)} exercise/solution pairs")


if __name__ == "__main__":
    main()
