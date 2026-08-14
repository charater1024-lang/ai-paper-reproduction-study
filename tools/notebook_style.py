"""Formatting helpers for generated portfolio notebooks.

The curriculum source files contain Python code inside strings.  Formatting the
generated notebook keeps the learner-facing code readable without asking every
spec author to hand-maintain whitespace that a formatter can determine safely.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

NOTEBOOK_LINE_LENGTH = 88


def format_notebooks(paths: Iterable[Path]) -> None:
    """Format code cells in generated notebooks with Ruff.

    Ruff understands the notebook JSON format and changes only Python code-cell
    source.  Markdown, cell IDs, metadata, TODO markers, and outputs are kept.
    The Windows installer includes Ruff through ``requirements-dev.txt``.
    """

    notebook_paths = [Path(path).resolve() for path in paths]
    if not notebook_paths:
        return

    command = [
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--line-length",
        str(NOTEBOOK_LINE_LENGTH),
        *map(str, notebook_paths),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        details = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            "Generated notebook formatting failed. Install requirements-dev.txt "
            f"and retry.\n{details}"
        )
