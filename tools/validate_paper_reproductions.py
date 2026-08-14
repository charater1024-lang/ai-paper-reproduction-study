"""Validate and optionally execute the 20 paired paper-reproduction notebooks."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "notebooks" / "paper_reproductions"
EXERCISES = TRACK / "exercises"
SOLUTIONS = TRACK / "solutions"
EXPECTED_NUMBERS = [f"{number:02d}" for number in range(20)]
ROLE_MARKER = "<!-- paper-paired-role-banner -->"
FORBIDDEN_SOURCE = (
    "%pip",
    "!pip",
    "!wget",
    "!curl",
    "requests.get(",
    "urlretrieve(",
    "download=True",
)
PAPER_LOCATION_MARKERS = (
    "§",
    "Eq.",
    "식 ",
    "Figure",
    "Fig.",
    "그림",
    "Algorithm",
    "Proposition",
    "Theorem",
)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def read_notebook(path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def validate_clean_cells(
    path: Path, notebook: nbformat.NotebookNode, role: str
) -> None:
    ids = [cell.id for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{path.name}: duplicate cell id")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"{path.name}: checked-in output at cell {index}")
        try:
            ast.parse(cell.source)
        except SyntaxError as error:
            raise AssertionError(
                f"{path.name}: invalid Python at cell {index}: {error}"
            ) from error
        for forbidden in FORBIDDEN_SOURCE:
            if forbidden in cell.source:
                raise AssertionError(
                    f"{path.name}: forbidden network/install source {forbidden!r}"
                )
        if role == "solution" and "NotImplementedError" in cell.source:
            raise AssertionError(f"{path.name}: unfinished solution at cell {index}")


def validate_pair(exercise_path: Path, solution_path: Path) -> tuple[int, int, int]:
    exercise = read_notebook(exercise_path)
    solution = read_notebook(solution_path)
    validate_clean_cells(exercise_path, exercise, "exercise")
    validate_clean_cells(solution_path, solution, "solution")

    if len(exercise.cells) != len(solution.cells):
        raise AssertionError(f"{exercise_path.name}: pair cell counts differ")
    if [cell.cell_type for cell in exercise.cells] != [
        cell.cell_type for cell in solution.cells
    ]:
        raise AssertionError(f"{exercise_path.name}: pair cell types differ")
    if [cell.id for cell in exercise.cells] != [cell.id for cell in solution.cells]:
        raise AssertionError(f"{exercise_path.name}: stable cell ids differ")

    exercise_meta = exercise.metadata.get("paper_reproduction", {})
    solution_meta = solution.metadata.get("paper_reproduction", {})
    if (
        exercise_meta.get("role") != "exercise"
        or solution_meta.get("role") != "solution"
    ):
        raise AssertionError(f"{exercise_path.name}: invalid role metadata")
    if exercise_meta.get("scope") != "hardware-aware-mini-reproduction":
        raise AssertionError(f"{exercise_path.name}: missing mini-reproduction scope")
    for key in (
        "number",
        "paper_title",
        "year",
        "primary_url",
        "expected_minutes",
        "scope",
        "device_policy",
        "device_override_env",
    ):
        if not exercise_meta.get(key) or exercise_meta.get(key) != solution_meta.get(
            key
        ):
            raise AssertionError(f"{exercise_path.name}: inconsistent metadata {key}")
    if (
        ROLE_MARKER not in exercise.cells[0].source
        or "실습 코드" not in exercise.cells[0].source
    ):
        raise AssertionError(f"{exercise_path.name}: missing exercise role banner")
    if (
        exercise_meta["device_policy"] != "auto:cuda-mps-cpu"
        or exercise_meta["device_override_env"] != "AI_LAB_DEVICE"
    ):
        raise AssertionError(f"{exercise_path.name}: invalid device policy")
    if (
        ROLE_MARKER not in solution.cells[0].source
        or "정답 코드" not in solution.cells[0].source
    ):
        raise AssertionError(f"{solution_path.name}: missing solution role banner")
    if "논문 ↔ 노트북 지도" not in exercise.cells[2].source:
        raise AssertionError(f"{exercise_path.name}: missing paper mapping table")
    if re.search(r"셀\s*\d", exercise.cells[2].source):
        raise AssertionError(
            f"{exercise_path.name}: map uses a fragile numeric cell reference; "
            "name the function, class, or task instead"
        )
    table_rows = [
        line.strip()
        for line in exercise.cells[2].source.splitlines()
        if line.strip().startswith("|") and "---" not in line
    ]
    if len(table_rows) < 2:
        raise AssertionError(
            f"{exercise_path.name}: paper mapping table has no data rows"
        )
    for row in table_rows[1:]:
        paper_location = row.strip("|").split("|", maxsplit=1)[0].strip()
        if not any(marker in paper_location for marker in PAPER_LOCATION_MARKERS):
            raise AssertionError(
                f"{exercise_path.name}: mapping must start with an exact paper location, "
                f"got {paper_location!r}"
            )

    changed = 0
    todo_count = 0
    for index, (exercise_cell, solution_cell) in enumerate(
        zip(exercise.cells, solution.cells, strict=True)
    ):
        if exercise_cell.cell_type == "markdown":
            tags = set(exercise_cell.metadata.get("tags", []))
            role_specific = index in {0, len(exercise.cells) - 1} or bool(
                tags & {"interpretation", "reflection", "solution-explanation"}
            )
            if not role_specific and exercise_cell.source != solution_cell.source:
                raise AssertionError(
                    f"{exercise_path.name}: shared markdown differs at cell {index}"
                )
            continue
        if exercise_cell.source != solution_cell.source:
            changed += 1
            todo_count += exercise_cell.source.count("TODO")
            if "TODO" not in exercise_cell.source:
                raise AssertionError(
                    f"{exercise_path.name}: changed cell {index} lacks TODO marker"
                )

    if changed < 3 or todo_count < 3:
        raise AssertionError(
            f"{exercise_path.name}: need at least three implementation tasks"
        )
    if changed != exercise_meta.get("hidden_cell_count"):
        raise AssertionError(
            f"{exercise_path.name}: exercise hidden_cell_count mismatch"
        )
    if changed != solution_meta.get("hidden_cell_count"):
        raise AssertionError(
            f"{solution_path.name}: solution hidden_cell_count mismatch"
        )
    return len(exercise.cells), changed, todo_count


def execute_solution(path: Path, timeout: int) -> tuple[str, float]:
    from nbclient import NotebookClient

    notebook = read_notebook(path)
    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=timeout,
        # The repository-local python3 spec launches `python`; PATH is prepended in main so it
        # always resolves to the active .venv even when a stale user kernelspec exists.
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute(cwd=str(ROOT))
    return path.name, time.perf_counter() - started


def selected_pairs(only: str | None) -> list[tuple[Path, Path]]:
    exercise_paths = sorted(EXERCISES.glob("[0-9][0-9]_*.ipynb"))
    solution_paths = sorted(SOLUTIONS.glob("[0-9][0-9]_*.ipynb"))
    if only is not None:
        number = only.strip().zfill(2)
        if number not in EXPECTED_NUMBERS:
            raise AssertionError(f"--only must be 00..19, got {only}")
        exercise_paths = [path for path in exercise_paths if path.name[:2] == number]
        solution_paths = [path for path in solution_paths if path.name[:2] == number]
    if [path.name for path in exercise_paths] != [path.name for path in solution_paths]:
        raise AssertionError("exercise/solution filenames do not match")
    if not exercise_paths:
        raise AssertionError("no paper-reproduction pairs found")
    if only is None and [path.name[:2] for path in exercise_paths] != EXPECTED_NUMBERS:
        raise AssertionError("paper notebook numbers must be contiguous 00..19")
    return list(zip(exercise_paths, solution_paths, strict=True))


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-solutions", action="store_true")
    parser.add_argument("--only", help="one paper number, for example 09")
    parser.add_argument(
        "--timeout", type=int, default=240, help="timeout per code cell in seconds"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="parallel kernels while executing solutions"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    validation_state = ROOT / "artifacts" / "paper_validation_runtime"
    runtime_directory = validation_state / "runtime"
    ipython_directory = validation_state / "ipython"
    runtime_directory.mkdir(parents=True, exist_ok=True)
    ipython_directory.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_directory)
    os.environ["IPYTHONDIR"] = str(ipython_directory)
    interpreter_directory = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_directory, os.environ.get("PATH", "")]
    )
    pairs = selected_pairs(args.only)
    for exercise_path, solution_path in pairs:
        cells, changed, todos = validate_pair(exercise_path, solution_path)
        print(
            f"VALID    {exercise_path.name} (cells={cells}, tasks={changed}, TODO={todos})"
        )

    if args.execute_solutions:
        solutions = [solution for _, solution in pairs]
        if args.jobs == 1:
            for solution in solutions:
                name, elapsed = execute_solution(solution, args.timeout)
                print(f"EXECUTED {name} ({elapsed:.1f}s)")
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(execute_solution, path, args.timeout): path
                    for path in solutions
                }
                for future in as_completed(futures):
                    name, elapsed = future.result()
                    print(f"EXECUTED {name} ({elapsed:.1f}s)")

    mode = "validated and executed" if args.execute_solutions else "validated"
    print(f"PASS: {mode} {len(pairs)} paper exercise/solution pairs")


if __name__ == "__main__":
    main()
