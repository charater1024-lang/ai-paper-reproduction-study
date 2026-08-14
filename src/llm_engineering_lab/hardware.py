"""Hardware-aware defaults for laptop-safe ML/DL experiments."""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def _windows_memory_gb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.total_physical / 1024**3, 1)


@dataclass(frozen=True)
class HardwareProfile:
    cpu_threads: int
    ram_gb: float | None
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float | None
    torch_version: str
    cuda_runtime: str | None

    @classmethod
    def detect(cls) -> HardwareProfile:
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        vram_gb = None
        if cuda_available:
            vram_gb = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
        return cls(
            cpu_threads=os.cpu_count() or 1,
            ram_gb=_windows_memory_gb(),
            cuda_available=cuda_available,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            torch_version=torch.__version__,
            cuda_runtime=torch.version.cuda,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )


@dataclass(frozen=True)
class LabTrainingProfile:
    device: str
    use_amp: bool
    dataloader_workers: int
    text_batch_size: int
    vision_batch_size: int
    sequence_length: int
    transformer_width: int
    transformer_layers: int
    gradient_accumulation_steps: int


def recommend_training_profile(
    hardware: HardwareProfile | None = None,
) -> LabTrainingProfile:
    """Return conservative defaults that leave room for Jupyter and the browser."""

    hardware = hardware or HardwareProfile.detect()
    if hardware.cuda_available and (hardware.vram_gb or 0) >= 12:
        return LabTrainingProfile(
            device="cuda",
            use_amp=True,
            dataloader_workers=0,  # Windows/Jupyter spawn safety
            text_batch_size=64,
            vision_batch_size=256,
            sequence_length=256,
            transformer_width=256,
            transformer_layers=4,
            gradient_accumulation_steps=2,
        )
    if hardware.cuda_available:
        return LabTrainingProfile(
            device="cuda",
            use_amp=True,
            dataloader_workers=0,
            text_batch_size=32,
            vision_batch_size=96,
            sequence_length=128,
            transformer_width=128,
            transformer_layers=3,
            gradient_accumulation_steps=4,
        )
    return LabTrainingProfile(
        device="cpu",
        use_amp=False,
        dataloader_workers=0,
        text_batch_size=16,
        vision_batch_size=32,
        sequence_length=96,
        transformer_width=96,
        transformer_layers=2,
        gradient_accumulation_steps=4,
    )
