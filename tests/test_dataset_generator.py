import importlib.util
import sys
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "00_generate_data.py"
SPEC = importlib.util.spec_from_file_location("dataset_generator", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_generated_dataset_is_balanced_grouped_and_reproducible() -> None:
    first = MODULE.build_rows(seed=7)
    second = MODULE.build_rows(seed=7)
    assert first == second
    assert len(first) == 100
    assert len({row["ticket_id"] for row in first}) == 100
    assert len({row["case_id"] for row in first}) == 50
    assert set(Counter(row["label"] for row in first).values()) == {20}
    assert set(Counter(row["case_id"] for row in first).values()) == {2}
