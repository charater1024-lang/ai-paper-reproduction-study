"""Shared device selection and safe acceleration helpers for every lab.

The notebooks use one small contract instead of spelling out CUDA checks in every
experiment.  ``AI_LAB_DEVICE`` can be set to ``auto`` (default), ``cuda``, ``cuda:0``,
``mps``, or ``cpu``.  ``AI_LAB_AMP`` accepts ``auto`` (default), ``1``, or ``0``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DEVICE_ENV = "AI_LAB_DEVICE"
AMP_ENV = "AI_LAB_AMP"
CPU_THREADS_ENV = "AI_LAB_CPU_THREADS"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str | None = None) -> torch.device:
    """Choose CUDA, Apple MPS, or CPU and validate explicit requests."""

    choice = (requested or os.getenv(DEVICE_ENV, "auto")).strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{DEVICE_ENV}={choice!r} was requested, but this PyTorch build cannot use CUDA. "
            "Run Install_GPU_PyTorch.cmd or set AI_LAB_DEVICE=cpu."
        )
    if device.type == "cuda" and device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"detected device count={torch.cuda.device_count()}."
            )
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError(
            f"{DEVICE_ENV}=mps was requested, but Apple MPS is unavailable on this system."
        )
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported {DEVICE_ENV} value: {choice!r}")
    return device


def _resolve_amp(device: torch.device, requested: str | None) -> bool:
    value = (requested or os.getenv(AMP_ENV, "auto")).strip().lower()
    if value == "auto":
        return device.type == "cuda"
    if value in _TRUE_VALUES:
        if device.type != "cuda":
            raise RuntimeError(f"{AMP_ENV}=1 currently requires a CUDA device.")
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{AMP_ENV} must be auto, 1, or 0; got {value!r}")


def _move(value: Any, device: torch.device, non_blocking: bool) -> Any:
    if isinstance(value, (torch.Tensor, nn.Module)):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move(item, device, non_blocking)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_move(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [_move(item, device, non_blocking) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LabAccelerator:
    """The selected device plus helpers shared by scripts and notebooks."""

    device: torch.device
    amp_enabled: bool

    @property
    def pin_memory(self) -> bool:
        return self.device.type == "cuda"

    @property
    def non_blocking(self) -> bool:
        return self.device.type == "cuda"

    @property
    def name(self) -> str:
        if self.device.type == "cuda":
            index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            return torch.cuda.get_device_name(index)
        if self.device.type == "mps":
            return "Apple Metal (MPS)"
        return "CPU"

    def move(self, *values: Any) -> Any:
        """Move tensors/modules, including nested mappings and sequences, to this device."""

        moved = tuple(_move(value, self.device, self.non_blocking) for value in values)
        return moved[0] if len(moved) == 1 else moved

    def autocast(self):
        """Return a CUDA AMP context, or a no-op context on other backends."""

        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def grad_scaler(self) -> torch.amp.GradScaler:
        """Create a scaler that is active only when CUDA AMP is enabled."""

        return torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def summary(self) -> str:
        thread_text = (
            f" | CPU threads={torch.get_num_threads()}"
            if self.device.type == "cpu"
            else ""
        )
        return (
            f"device={self.device} ({self.name}) | "
            f"AMP policy={'enabled' if self.amp_enabled else 'disabled'} | "
            f"override: {DEVICE_ENV}=cpu/cuda/mps{thread_text}"
        )


def get_accelerator(
    requested: str | None = None,
    *,
    amp: str | None = None,
    optimize: bool = True,
) -> LabAccelerator:
    """Build the common lab accelerator and enable safe fixed-shape speed defaults."""

    device = resolve_device(requested)
    amp_enabled = _resolve_amp(device, amp)
    if optimize and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif optimize and device.type == "cpu":
        default_threads = min(4, os.cpu_count() or 1)
        raw_threads = os.getenv(CPU_THREADS_ENV, str(default_threads)).strip()
        try:
            cpu_threads = int(raw_threads)
        except ValueError as error:
            raise ValueError(f"{CPU_THREADS_ENV} must be a positive integer") from error
        if cpu_threads < 1:
            raise ValueError(f"{CPU_THREADS_ENV} must be a positive integer")
        torch.set_num_threads(cpu_threads)
    return LabAccelerator(device=device, amp_enabled=amp_enabled)
