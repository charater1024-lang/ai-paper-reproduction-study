"""프로젝트 전역 설정과 재현성 유틸리티.

실무에서는 경로, 난수 시드, 장치 선택을 여러 스크립트에 복사하지 않고
이처럼 한곳에서 관리한다. 환경 변수로 덮어쓸 수 있게 만드는 것도 중요하다.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

DEFAULT_SEED: Final[int] = 42


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """현재 파일 위치를 기준으로 계산한 프로젝트 경로 모음."""

    root: Path
    data: Path
    artifacts: Path

    @classmethod
    def discover(cls) -> ProjectPaths:
        override = os.getenv("LLM_LAB_ROOT")
        root = (
            Path(override).expanduser().resolve()
            if override
            else Path(__file__).parents[2]
        )
        return cls(root=root, data=root / "data", artifacts=root / "artifacts")

    def ensure_output_dirs(self) -> None:
        self.artifacts.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Python/NumPy/PyTorch 난수를 가능한 범위에서 고정한다.

    ``deterministic=True``는 디버깅에 유리하지만 GPU 학습 속도를 낮출 수 있다.
    PyTorch가 설치되지 않은 데이터 전처리 환경에서도 이 함수는 동작한다.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
