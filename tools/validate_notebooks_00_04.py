"""Validate and smoke-execute the introductory curriculum notebooks."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook_paths = sorted((ROOT / "notebooks").glob("0[0-4]_*.ipynb"))
    if len(notebook_paths) != 5:
        raise AssertionError(f"expected 5 notebooks, found {len(notebook_paths)}")

    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for path in notebook_paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            if not 15 <= len(notebook.cells) <= 30:
                raise AssertionError(f"{path.name}: cell count is out of range")
            markdown_count = sum(
                cell.cell_type == "markdown" for cell in notebook.cells
            )
            code_count = sum(cell.cell_type == "code" for cell in notebook.cells)
            todo_count = sum(cell.source.count("TODO") for cell in notebook.cells)
            assert_count = sum(cell.source.count("assert") for cell in notebook.cells)
            output_count = sum(len(cell.get("outputs", [])) for cell in notebook.cells)
            if min(markdown_count, code_count, todo_count, assert_count) < 1:
                raise AssertionError(
                    f"{path.name}: required learning elements are missing"
                )
            if output_count:
                raise AssertionError(
                    f"{path.name}: generated notebook must not save outputs"
                )

            # Dataclasses resolve postponed annotations through sys.modules.
            # ``__main__`` is registered and closely matches a notebook kernel.
            namespace = {"__name__": "__main__"}
            captured = io.StringIO()
            for index, cell in enumerate(notebook.cells, start=1):
                if cell.cell_type != "code":
                    continue
                try:
                    with contextlib.redirect_stdout(captured):
                        exec(
                            compile(cell.source, f"{path.name}:cell-{index}", "exec"),
                            namespace,
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"{path.name}: code cell {index} failed"
                    ) from error

            print(
                f"PASS {path.name}: cells={len(notebook.cells)}, "
                f"markdown={markdown_count}, code={code_count}, "
                f"TODO={todo_count}, assert={assert_count}, outputs={output_count}"
            )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
