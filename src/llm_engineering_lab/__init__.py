"""현업형 ML/LLM 엔지니어링 실습 패키지."""

from .acceleration import LabAccelerator, get_accelerator, resolve_device
from .config import ProjectPaths, seed_everything

__all__ = [
    "LabAccelerator",
    "ProjectPaths",
    "get_accelerator",
    "resolve_device",
    "seed_everything",
]
__version__ = "0.1.0"
