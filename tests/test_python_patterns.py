import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "00_python_engineering_patterns.py"
)
SPEC = importlib.util.spec_from_file_location("python_patterns", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pipeline_normalizes_and_redacts() -> None:
    pipeline = MODULE.Pipeline([MODULE.NormalizeWhitespace(), MODULE.RedactSecrets()])
    result = pipeline(MODULE.Ticket("1", "  api_key=secret   노출 "))
    assert result.text == "[REDACTED] 노출"


def test_batched_preserves_remainder() -> None:
    assert list(MODULE.batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_batched_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        list(MODULE.batched([1], 0))
