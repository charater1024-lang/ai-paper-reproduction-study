"""Validate every curriculum notebook and optionally execute it in a real kernel."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def validate_notebook(path: Path, *, execute: bool, timeout: int) -> tuple[int, int]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    code_count = sum(cell.cell_type == "code" for cell in notebook.cells)
    todo_count = sum(cell.source.count("TODO") for cell in notebook.cells)
    assert_count = sum(cell.source.count("assert") for cell in notebook.cells)
    if code_count == 0 or todo_count == 0 or assert_count == 0:
        raise AssertionError(f"{path.name}: code, TODO, assert 학습 요소를 확인하세요")

    if execute:
        # 원본 노트북에는 실행 번호와 출력을 기록하지 않는다.
        client = NotebookClient(
            notebook,
            timeout=timeout,
            # Resolve the standard python3 spec through the active .venv PATH instead of a
            # possibly stale user kernelspec left behind when the workspace was moved.
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(cwd=str(ROOT))

    return len(notebook.cells), code_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="실제 Jupyter 커널로 전체 실행"
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="코드 셀당 제한 시간(초)"
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    interpreter_directory = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_directory, os.environ.get("PATH", "")]
    )
    paths = sorted(NOTEBOOK_DIR.glob("[0-1][0-9]_*.ipynb"))
    expected_numbers = [f"{number:02d}" for number in range(13)]
    actual_numbers = [path.name[:2] for path in paths]
    if actual_numbers != expected_numbers:
        raise AssertionError(f"노트북 번호가 00~12가 아닙니다: {actual_numbers}")

    for path in paths:
        cell_count, code_count = validate_notebook(
            path,
            execute=args.execute,
            timeout=args.timeout,
        )
        mode = "EXECUTED" if args.execute else "VALID"
        print(f"{mode:8} {path.name} (cells={cell_count}, code={code_count})")

    print(f"PASS: {len(paths)} notebooks")


if __name__ == "__main__":
    main()
