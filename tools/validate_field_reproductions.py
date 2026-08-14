"""Validate datasets and 70 field-paper pairs; optionally execute all solutions."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nbformat
from field_curriculum.common import FIELD_ORDER

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "notebooks" / "field_reproductions"
DATA = ROOT / "data" / "field_curriculum"
ROLE_MARKER = "<!-- field-paper-role-banner -->"
FORBIDDEN = (
    "%pip",
    "!pip",
    "!wget",
    "!curl",
    "requests.get(",
    "urlretrieve(",
    "load_dataset(",
    "download=True",
)
PAPER_LOCATION_MARKERS = (
    "§",
    "Eq.",
    "Equation",
    "식 ",
    "Figure",
    "Fig.",
    "그림",
    "Algorithm",
    "Appendix",
    "Table",
    "Section",
    "Methods",
    "Main text",
    "Proposition",
    "Theorem",
)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_datasets() -> None:
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("license") != "CC0-1.0" or len(manifest.get("files", [])) != 7:
        raise AssertionError("field dataset manifest must describe seven CC0 files")
    for entry in manifest["files"]:
        path = DATA / entry["path"]
        if not path.is_file():
            raise AssertionError(f"missing field dataset: {path}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise AssertionError(f"field dataset checksum mismatch: {path.name}")
    print("DATA     seven local synthetic datasets (manifest + SHA-256 valid)")


def read(path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def validate_code(path: Path, notebook: nbformat.NotebookNode, role: str) -> None:
    ids = [cell.id for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{path}: duplicate cell ids")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise AssertionError(f"{path}: committed output at cell {index}")
        try:
            ast.parse(cell.source)
        except SyntaxError as error:
            raise AssertionError(
                f"{path}: invalid Python at cell {index}: {error}"
            ) from error
        for marker in FORBIDDEN:
            if marker in cell.source:
                raise AssertionError(
                    f"{path}: forbidden network/install marker {marker!r}"
                )
        if role == "solution" and "NotImplementedError" in cell.source:
            raise AssertionError(f"{path}: unfinished solution at cell {index}")


def validate_pair(exercise_path: Path, solution_path: Path) -> tuple[int, int, int]:
    exercise = read(exercise_path)
    solution = read(solution_path)
    validate_code(exercise_path, exercise, "exercise")
    validate_code(solution_path, solution, "solution")
    if len(exercise.cells) != len(solution.cells):
        raise AssertionError(f"{exercise_path}: pair cell count differs")
    if [cell.id for cell in exercise.cells] != [cell.id for cell in solution.cells]:
        raise AssertionError(f"{exercise_path}: pair cell ids differ")
    if [cell.cell_type for cell in exercise.cells] != [
        cell.cell_type for cell in solution.cells
    ]:
        raise AssertionError(f"{exercise_path}: pair cell types differ")
    em = exercise.metadata.get("field_reproduction", {})
    sm = solution.metadata.get("field_reproduction", {})
    if em.get("role") != "exercise" or sm.get("role") != "solution":
        raise AssertionError(f"{exercise_path}: invalid pair roles")
    for key in (
        "field_id",
        "field_title",
        "number",
        "paper_title",
        "year",
        "primary_url",
        "dataset_file",
        "scope",
        "device_policy",
        "device_override_env",
    ):
        if not em.get(key) or em.get(key) != sm.get(key):
            raise AssertionError(f"{exercise_path}: inconsistent metadata {key}")
    if em["scope"] != "hardware-aware-mini-reproduction":
        raise AssertionError(f"{exercise_path}: invalid scope")
    if (
        em["device_policy"] != "auto:cuda-mps-cpu"
        or em["device_override_env"] != "AI_LAB_DEVICE"
    ):
        raise AssertionError(f"{exercise_path}: invalid device policy")
    dataset_path = ROOT / em["dataset_file"]
    if not dataset_path.is_file() or DATA not in dataset_path.parents:
        raise AssertionError(
            f"{exercise_path}: dataset missing/outside field data directory"
        )
    if (
        ROLE_MARKER not in exercise.cells[0].source
        or "실습본" not in exercise.cells[0].source
    ):
        raise AssertionError(f"{exercise_path}: missing exercise banner")
    if (
        ROLE_MARKER not in solution.cells[0].source
        or "정답·해설본" not in solution.cells[0].source
    ):
        raise AssertionError(f"{solution_path}: missing solution banner")
    if "원 논문 ↔ 실습 지도" not in exercise.cells[2].source:
        raise AssertionError(f"{exercise_path}: missing paper map")
    table_rows = [
        line.strip()
        for line in exercise.cells[2].source.splitlines()
        if line.strip().startswith("|") and "---" not in line
    ]
    if len(table_rows) < 2:
        raise AssertionError(f"{exercise_path}: paper map has no data rows")
    for row in table_rows[1:]:
        paper_location = row.strip("|").split("|", maxsplit=1)[0].strip()
        if not any(marker in paper_location for marker in PAPER_LOCATION_MARKERS):
            raise AssertionError(
                f"{exercise_path}: mapping must start with an exact paper location, "
                f"got {paper_location!r}"
            )
    changed = 0
    todos = 0
    for index, (exercise_cell, solution_cell) in enumerate(
        zip(exercise.cells, solution.cells, strict=True)
    ):
        if (
            exercise_cell.cell_type != "code"
            or exercise_cell.source == solution_cell.source
        ):
            continue
        changed += 1
        todos += exercise_cell.source.count("TODO")
        if "TODO" not in exercise_cell.source:
            raise AssertionError(
                f"{exercise_path}: changed code cell {index} lacks TODO"
            )
    if changed < 3 or todos < 3:
        raise AssertionError(
            f"{exercise_path}: requires at least three implementation tasks"
        )
    if em.get("hidden_cell_count") != changed or sm.get("hidden_cell_count") != changed:
        raise AssertionError(f"{exercise_path}: hidden task metadata mismatch")
    return len(exercise.cells), changed, todos


def pairs(field: str | None, only: str | None) -> list[tuple[Path, Path]]:
    fields = (field,) if field else FIELD_ORDER
    selected: list[tuple[Path, Path]] = []
    for field_id in fields:
        exercise_dir = TRACK / field_id / "exercises"
        solution_dir = TRACK / field_id / "solutions"
        exercises = sorted(exercise_dir.glob("[0-9][0-9]_*.ipynb"))
        solutions = sorted(solution_dir.glob("[0-9][0-9]_*.ipynb"))
        if only is not None:
            number = only.strip().zfill(2)
            exercises = [path for path in exercises if path.name[:2] == number]
            solutions = [path for path in solutions if path.name[:2] == number]
        if [path.name for path in exercises] != [path.name for path in solutions]:
            raise AssertionError(f"{field_id}: exercise/solution filenames differ")
        expected = 1 if only is not None else 10
        if len(exercises) != expected:
            raise AssertionError(
                f"{field_id}: expected {expected} pairs, found {len(exercises)}"
            )
        if only is None and [path.name[:2] for path in exercises] != [
            f"{n:02d}" for n in range(10)
        ]:
            raise AssertionError(f"{field_id}: numbers must be 00..09")
        selected.extend(zip(exercises, solutions, strict=True))
    return selected


def execute_solution(path: Path, timeout: int) -> tuple[str, float]:
    from nbclient import NotebookClient

    notebook = read(path)
    started = time.perf_counter()
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    ).execute(cwd=str(ROOT))
    return str(path.relative_to(TRACK)), time.perf_counter() - started


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=FIELD_ORDER)
    parser.add_argument("--only", help="paper number 00..09")
    parser.add_argument("--execute-solutions", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.only is not None and not args.field:
        raise SystemExit("--only requires --field")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    state = ROOT / "artifacts" / "field_validation_runtime"
    runtime = state / "runtime"
    ipython = state / "ipython"
    runtime.mkdir(parents=True, exist_ok=True)
    ipython.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(ipython)
    interpreter = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join([interpreter, os.environ.get("PATH", "")])

    validate_datasets()
    selected = pairs(args.field, args.only)
    for exercise, solution in selected:
        cells, tasks, todos = validate_pair(exercise, solution)
        print(
            f"VALID    {exercise.relative_to(TRACK)} "
            f"(cells={cells}, tasks={tasks}, TODO={todos})"
        )
    if args.execute_solutions:
        solutions = [solution for _, solution in selected]
        if args.jobs == 1:
            for path in solutions:
                name, elapsed = execute_solution(path, args.timeout)
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
    verb = "validated and executed" if args.execute_solutions else "validated"
    print(f"PASS: {verb} {len(selected)} field-paper pairs")


if __name__ == "__main__":
    main()
