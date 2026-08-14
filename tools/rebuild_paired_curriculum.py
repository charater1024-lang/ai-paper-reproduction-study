"""Rebuild, normalize and strictly validate all 00-22 exercise/solution pairs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from notebook_style import format_notebooks

ROOT = Path(__file__).resolve().parents[1]
BUILD_STEPS = (
    "tools/build_legacy_pairs.py",
    "tools/build_paired_13_14.py",
    "tools/build_paired_15_16.py",
    "tools/build_paired_17_18.py",
    "tools/normalize_pair_banners.py",
)
VALIDATOR = "tools/validate_paired_notebooks.py"


def original_notebooks() -> list[Path]:
    """Return the hand-authored 00-12 notebooks used by the legacy pair builder."""

    return sorted((ROOT / "notebooks").glob("[0-1][0-9]_*.ipynb"))


def paired_notebooks() -> list[Path]:
    """Return learner-facing exercise and solution notebooks."""

    notebooks_root = ROOT / "notebooks"
    return [
        *sorted((notebooks_root / "exercises").glob("*.ipynb")),
        *sorted((notebooks_root / "solutions").glob("*.ipynb")),
    ]


def main() -> None:
    print("\n=== format original notebooks ===", flush=True)
    format_notebooks(original_notebooks())

    for relative_path in BUILD_STEPS:
        print(f"\n=== {relative_path} ===", flush=True)
        subprocess.run([sys.executable, relative_path], cwd=ROOT, check=True)

    print("\n=== format paired notebooks ===", flush=True)
    format_notebooks(paired_notebooks())

    print(f"\n=== {VALIDATOR} ===", flush=True)
    subprocess.run([sys.executable, VALIDATOR], cwd=ROOT, check=True)
    print("\nPASS: paired curriculum 00~22 rebuilt and validated")


if __name__ == "__main__":
    main()
