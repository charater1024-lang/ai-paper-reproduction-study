"""Tests for readable helper scripts used by the Windows batch launchers."""

from __future__ import annotations

from tools.check_windows_python import is_supported_runtime
from tools.verify_cuda_runtime import normalized_cuda_version


def test_windows_runtime_contract_requires_python_312_64_bit() -> None:
    assert is_supported_runtime((3, 12), 64)
    assert not is_supported_runtime((3, 11), 64)
    assert not is_supported_runtime((3, 12), 32)


def test_cuda_runtime_version_matches_wheel_suffix_format() -> None:
    assert normalized_cuda_version("13.0") == "130"
    assert normalized_cuda_version("12.6") == "126"
    assert normalized_cuda_version(None) == ""
