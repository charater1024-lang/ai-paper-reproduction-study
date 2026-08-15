"""Build the introductory JupyterLab curriculum notebooks with nbformat.

Run this file from the project root after ``nbformat`` is available in the
active Python environment.  The generated notebooks intentionally contain no
saved outputs so that learners reproduce every result in their own kernel.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from notebook_api_explanations import annotate_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str):
    return new_markdown_cell(dedent(source).strip())


def code(source: str):
    return new_code_cell(dedent(source).strip())


def merge_markdown_into_previous(cells: list, markers: tuple[str, ...]) -> list:
    """Keep detailed prose while grouping closely related sections into fewer cells."""

    for marker in markers:
        current_index = next(
            index
            for index, cell in enumerate(cells)
            if cell.cell_type == "markdown" and marker in cell.source
        )
        previous_index = next(
            index
            for index in range(current_index - 1, -1, -1)
            if cells[index].cell_type == "markdown"
        )
        cells[previous_index].source += "\n\n" + cells[current_index].source
        del cells[current_index]
    return cells


def base_notebook(cells: list) -> nbformat.NotebookNode:
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (LLM Engineering Lab)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10+",
            },
        },
    )


COMMON_BOOTSTRAP = '''
from pathlib import Path
import sys


def find_project_root(start: Path | None = None) -> Path:
    """현재 폴더부터 위로 올라가며 프로젝트 루트를 찾습니다."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("pyproject.toml과 src/가 있는 프로젝트 폴더를 찾지 못했습니다.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("project:", PROJECT_ROOT)
print("python :", sys.executable)
'''


def notebook_00():
    cells = [
        md("""
        # 00. JupyterLab 환경과 재현 가능한 실행 습관

        이 노트북은 앞으로 사용할 **프로젝트 경로, Python 커널, 패키지 버전, 난수 시드**를 점검합니다.
        AI 엔지니어의 실험은 “한 번 성공한 코드”보다 **다시 실행해도 같은 근거를 남기는 코드**가 중요합니다.

        학습 목표:

        - Jupyter의 셀과 커널 상태를 구분한다.
        - 현재 노트북이 올바른 `.venv` 커널에서 실행되는지 확인한다.
        - 프로젝트의 `src/`, `data/`, `artifacts/` 경로를 안전하게 찾는다.
        - Python/NumPy/PyTorch 난수를 고정하고 재현성을 확인한다.
        """),
        md("""
        ## 권장 실행 순서

        1. JupyterLab에서 이 파일을 연다.
        2. 우측 상단 커널이 `Python (LLM Engineering Lab)`인지 확인한다.
        3. `Kernel → Restart Kernel and Run All Cells`로 전체 실행한다.
        4. 오류가 없으면 `01`부터 번호 순서대로 진행한다.

        노트북은 설명과 관찰에, `src/`는 재사용 가능한 구현에, `tests/`는 자동 검증에 사용합니다.
        """),
        code(COMMON_BOOTSTRAP),
        md("""
        ## 커널과 패키지 상태 확인

        JupyterLab 서버를 실행한 Python과 **현재 노트북 커널의 Python은 다를 수 있습니다.**
        아래의 `sys.executable`이 프로젝트의 `.venv\\Scripts\\python.exe`를 가리키는지 확인하세요.
        """),
        code("""
        import platform
        import importlib.metadata as metadata

        packages = ["numpy", "pandas", "scikit-learn", "torch"]
        print("Python   :", platform.python_version())
        print("OS       :", platform.platform())
        for package in packages:
            try:
                print(f"{package:12}: {metadata.version(package)}")
            except metadata.PackageNotFoundError:
                print(f"{package:12}: 설치되지 않음")
        """),
        md("""
        ## 셀 상태(state)를 직접 관찰하기

        커널은 셀 실행이 끝나도 메모리의 변수를 유지합니다. 아래 두 셀을 순서대로 실행한 뒤,
        두 번째 셀만 여러 번 실행해 보세요. 실행 순서에 따라 값이 달라지는 것이 노트북의 대표적인 함정입니다.
        """),
        code("""
        experiment_state = {"run_count": 0, "losses": []}
        experiment_state
        """),
        code("""
        experiment_state["run_count"] += 1
        experiment_state["losses"].append(round(1 / experiment_state["run_count"], 3))
        experiment_state
        """),
        md("""
        셀 번호가 `[1]`, `[2]`, `[7]`처럼 뒤섞였거나 변수가 어디서 만들어졌는지 불분명하면 커널을 재시작하세요.
        제출하거나 공유하기 전에는 반드시 **Restart Kernel and Run All Cells**로 검증합니다.
        """),
        md("""
        ## 난수와 재현성

        모델 초기값, 데이터 셔플, 샘플링에는 난수가 사용됩니다. 같은 시드를 사용하면 디버깅과 실험 비교가 쉬워집니다.
        완전 결정론적 GPU 연산은 속도를 낮출 수 있으므로 목적에 맞게 선택해야 합니다.
        """),
        code("""
        import random
        import numpy as np

        from llm_engineering_lab.config import seed_everything

        seed_everything(42)
        python_sample = [random.random() for _ in range(3)]
        numpy_sample = np.random.random(3)

        print("Python:", python_sample)
        print("NumPy :", numpy_sample)
        """),
        code("""
        # 같은 시드로 되돌리면 같은 수열이 다시 나와야 합니다.
        seed_everything(42)
        assert python_sample == [random.random() for _ in range(3)]
        assert np.allclose(numpy_sample, np.random.random(3))
        print("재현성 검사 통과")
        """),
        md("""
        ## 프로젝트 경로와 패키지 import

        현재 작업 폴더(`cwd`)를 문자열로 하드코딩하지 않고 프로젝트 루트를 기준으로 경로를 만듭니다.
        그러면 JupyterLab을 프로젝트 루트 또는 `notebooks/`에서 열어도 같은 코드가 동작합니다.
        """),
        code("""
        from llm_engineering_lab.config import ProjectPaths

        paths = ProjectPaths.discover()
        assert paths.root == PROJECT_ROOT
        assert paths.data.is_dir()

        print("data     :", paths.data)
        print("artifacts:", paths.artifacts)
        """),
        code("""
        from llm_engineering_lab.data import load_ticket_data

        tickets = load_ticket_data(paths.data / "customer_support_tickets.csv")
        print("shape:", tickets.shape)
        tickets.head(3)
        """),
        md("""
        ## TODO: 커널 건강 상태 함수 따라 치기

        아래 함수는 실험 시작 전에 자주 확인할 정보를 한 딕셔너리로 모읍니다.
        구현을 지운 뒤 주석만 보고 다시 작성해 보세요.
        """),
        code("""
        def kernel_health_report(project_root: Path) -> dict[str, object]:
            # TODO 1: Python 실행 파일, 작업 폴더, 프로젝트 존재 여부를 반환하세요.
            return {
                "python": sys.executable,
                "cwd": str(Path.cwd()),
                "project_exists": project_root.is_dir(),
                "data_exists": (project_root / "data").is_dir(),
            }


        health = kernel_health_report(PROJECT_ROOT)
        assert health["project_exists"] is True
        assert health["data_exists"] is True
        assert str(PROJECT_ROOT) in str(paths.root)
        health
        """),
        md("""
        ## 실험 노트북 운영 규칙

        - 원본 데이터는 덮어쓰지 않고 가공 결과를 별도 변수/파일에 저장합니다.
        - 긴 학습은 함수나 `scripts/`로 옮기고 노트북에서는 작은 설정으로 검증합니다.
        - 지표와 그림만 보지 말고 시드, 데이터 분할, 하이퍼파라미터도 기록합니다.
        - 비밀키를 셀에 직접 쓰지 않습니다. 환경 변수나 비밀 관리 도구를 사용합니다.
        - 셀을 반복 실행해도 파일이나 상태가 망가지지 않는 **멱등성**을 지향합니다.
        """),
        code("""
        # 마지막 셀: 최소 환경 검증
        required_paths = [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "tests",
        ]
        assert all(path.exists() for path in required_paths)
        assert len(tickets) > 0
        print("✅ 00 노트북 준비 완료 — 다음은 01 중급 Python 패턴입니다.")
        """),
    ]
    return base_notebook(cells)


def notebook_01():
    cells = [
        md("""
        # 01. AI 엔지니어를 위한 중급 Python 패턴

        ML 코드는 실험 단계에서는 짧지만, 운영 단계에서는 설정·데이터·모델·오류를 명확히 구분해야 합니다.
        이 노트북에서는 **class, dataclass, Protocol, 제네릭, iterator/generator, decorator,
        context manager, 예외, typing**을 작은 ML 예제로 익힙니다.
        """),
        code(COMMON_BOOTSTRAP),
        code("""
        from __future__ import annotations

        from collections.abc import Callable, Generator, Iterable, Iterator
        from contextlib import contextmanager
        from dataclasses import asdict, dataclass, field
        from functools import wraps
        from time import perf_counter
        from typing import Generic, Protocol, TypeVar, get_type_hints, runtime_checkable

        import numpy as np
        """),
        md("""
        ## 1. `dataclass`: 실험 설정을 값 객체로 만들기

        여러 인자를 딕셔너리로 흩어 두는 대신 타입과 검증 규칙을 가진 객체로 묶습니다.
        `frozen=True`는 학습 도중 설정이 우연히 바뀌는 일을 줄이고, `slots=True`는 허용되지 않은 속성 추가를 막습니다.
        """),
        code("""
        @dataclass(frozen=True, slots=True)
        class ExperimentConfig:
            learning_rate: float = 1e-3
            batch_size: int = 32
            epochs: int = 5
            labels: tuple[str, ...] = ("billing", "delivery")
            tags: dict[str, str] = field(default_factory=dict)

            def __post_init__(self) -> None:
                if self.learning_rate <= 0:
                    raise ValueError("learning_rate must be positive")
                if self.batch_size <= 0 or self.epochs <= 0:
                    raise ValueError("batch_size and epochs must be positive")


        config = ExperimentConfig(tags={"owner": "learner", "stage": "dev"})
        assert config.batch_size == 32
        asdict(config)
        """),
        code("""
        # TODO 1: 잘못된 설정을 조용히 허용하지 않는 이유를 확인하세요.
        try:
            ExperimentConfig(learning_rate=-0.1)
        except ValueError as error:
            print("예상한 오류:", error)
        else:
            raise AssertionError("잘못된 learning_rate가 거부되어야 합니다.")
        """),
        md("""
        ## 2. `Protocol`: 상속보다 필요한 동작을 표현하기

        입력 객체가 특정 부모 클래스를 상속했는지가 아니라, 필요한 메서드를 제공하는지를 타입으로 표현합니다.
        임베딩 공급자를 로컬 모델에서 API 모델로 바꿀 때도 호출 코드는 그대로 둘 수 있습니다.
        """),
        code("""
        @runtime_checkable
        class TextEncoder(Protocol):
            dimension: int

            def encode(self, texts: list[str]) -> np.ndarray:
                ...


        class LengthEncoder:
            dimension = 2

            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[len(text), len(text.split())] for text in texts], dtype=float)


        class CharacterCountEncoder:
            dimension = 3

            def encode(self, texts: list[str]) -> np.ndarray:
                vowels = set("aeiouAEIOU")
                return np.array([
                    [len(text), sum(ch.isdigit() for ch in text), sum(ch in vowels for ch in text)]
                    for text in texts
                ], dtype=float)
        """),
        code("""
        def build_feature_matrix(encoder: TextEncoder, texts: list[str]) -> np.ndarray:
            matrix = encoder.encode(texts)
            if matrix.shape != (len(texts), encoder.dimension):
                raise ValueError("encoder가 약속한 shape과 실제 shape이 다릅니다.")
            return matrix


        sample_texts = ["refund 123", "delivery delayed"]
        features = build_feature_matrix(LengthEncoder(), sample_texts)
        assert features.shape == (2, 2)
        assert isinstance(LengthEncoder(), TextEncoder)
        features
        """),
        md("""
        ## 3. 제네릭(`Generic`): 자료형을 잃지 않는 컨테이너

        같은 저장소·배치 코드를 여러 타입에 재사용하면서도 편집기와 타입 검사기가 원소 타입을 추적할 수 있습니다.
        """),
        code("""
        T = TypeVar("T")


        class Stack(Generic[T]):
            def __init__(self) -> None:
                self._items: list[T] = []

            def push(self, item: T) -> None:
                self._items.append(item)

            def pop(self) -> T:
                if not self._items:
                    raise IndexError("empty stack")
                return self._items.pop()

            def __len__(self) -> int:
                return len(self._items)


        experiment_stack = Stack[ExperimentConfig]()
        experiment_stack.push(config)
        assert experiment_stack.pop().tags["stage"] == "dev"
        """),
        md("""
        ## 4. Iterator: 상태를 가진 배치 순회

        iterator는 `__iter__`와 `__next__`를 구현하고, 데이터가 끝나면 `StopIteration`을 발생시킵니다.
        작은 클래스는 epoch별 셔플이나 커서 상태를 명시적으로 관찰할 때 유용합니다.
        """),
        code("""
        class BatchIterator(Iterator[list[T]], Generic[T]):
            def __init__(self, items: Iterable[T], batch_size: int) -> None:
                if batch_size <= 0:
                    raise ValueError("batch_size must be positive")
                self.items = list(items)
                self.batch_size = batch_size
                self.cursor = 0

            def __iter__(self) -> BatchIterator[T]:
                return self

            def __next__(self) -> list[T]:
                if self.cursor >= len(self.items):
                    raise StopIteration
                batch = self.items[self.cursor : self.cursor + self.batch_size]
                self.cursor += self.batch_size
                return batch


        batches = list(BatchIterator(range(7), batch_size=3))
        assert batches == [[0, 1, 2], [3, 4, 5], [6]]
        batches
        """),
        md("""
        ## 5. Generator: 필요한 순간에 배치를 만들기

        `yield`를 사용하면 전체 결과를 메모리에 만들지 않고 한 배치씩 제공합니다.
        대규모 로그 전처리나 스트리밍 데이터에 적합합니다.
        """),
        code("""
        def iter_batches(items: Iterable[T], batch_size: int) -> Generator[list[T], None, None]:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            batch: list[T] = []
            for item in items:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch


        lazy_batches = iter_batches((value * value for value in range(8)), 3)
        first = next(lazy_batches)
        rest = list(lazy_batches)
        assert first == [0, 1, 4]
        assert rest[-1] == [36, 49]
        first, rest
        """),
        md("""
        ## 6. Decorator: 계측 코드를 핵심 로직과 분리하기

        운영 ML 시스템에서는 실행 시간, 호출 횟수, 오류율을 반복 측정합니다.
        decorator는 함수의 핵심 로직을 바꾸지 않고 공통 동작을 감쌉니다.
        """),
        code("""
        def count_calls(function: Callable[..., T]) -> Callable[..., T]:
            @wraps(function)
            def wrapper(*args, **kwargs):
                wrapper.calls += 1
                return function(*args, **kwargs)

            wrapper.calls = 0
            return wrapper


        @count_calls
        def normalize_score(value: float) -> float:
            # TODO 2: [0, 1] 범위로 자르는 코드를 직접 다시 작성해 보세요.
            return min(1.0, max(0.0, float(value)))


        assert normalize_score(-0.5) == 0.0
        assert normalize_score(1.7) == 1.0
        assert normalize_score.calls == 2
        print("호출 횟수:", normalize_score.calls)
        """),
        md("""
        ## 7. Context manager: 자원의 시작과 정리를 한 쌍으로 묶기

        파일, 데이터베이스 연결, mixed precision 범위, 시간 측정처럼 시작/종료가 반드시 짝을 이뤄야 할 때 사용합니다.
        `finally` 덕분에 내부에서 예외가 나도 정리 코드가 실행됩니다.
        """),
        code("""
        @contextmanager
        def timer(name: str):
            started = perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (perf_counter() - started) * 1_000
                print(f"{name}: {elapsed_ms:.3f} ms")


        with timer("matrix multiplication"):
            result = np.eye(100) @ np.ones((100, 4))

        assert result.shape == (100, 4)
        """),
        md("""
        ## 8. 예외: 실패 원인을 데이터 계약으로 표현하기

        `except Exception: pass`로 오류를 숨기지 않습니다. 복구 가능한 오류만 구체적으로 처리하고,
        호출자가 행동을 선택할 수 있도록 맥락이 있는 사용자 정의 예외를 제공합니다.
        """),
        code('''
        class LabelParseError(ValueError):
            """레이블 문자열이 계약을 위반했을 때 발생합니다."""


        def parse_label(raw: str, allowed: set[str]) -> str:
            normalized = raw.strip().lower()
            if not normalized:
                raise LabelParseError("레이블이 비어 있습니다.")
            if normalized not in allowed:
                raise LabelParseError(f"지원하지 않는 레이블: {normalized!r}")
            return normalized


        assert parse_label(" Billing ", {"billing", "delivery"}) == "billing"
        try:
            parse_label("unknown", {"billing", "delivery"})
        except LabelParseError as error:
            print("검증 실패를 정상적으로 포착:", error)
        '''),
        md("""
        ## 9. Typing은 실행을 대신하지 않고 의도를 기록한다

        타입 힌트는 문서화·자동완성·정적 검사를 돕지만 런타임 검증을 자동 수행하지 않습니다.
        외부 입력에는 앞의 명시적 검증이 여전히 필요합니다.
        """),
        code("""
        hints = get_type_hints(build_feature_matrix)
        assert "encoder" in hints and "return" in hints
        hints
        """),
        md("""
        ## 10. 미니 파이프라인으로 패턴 연결하기

        Protocol 기반 encoder, generator 배치, 예외 기반 shape 계약을 한 번에 연결합니다.
        구현을 바꾸더라도 `encode_dataset`의 인터페이스는 유지됩니다.
        """),
        code("""
        def encode_dataset(
            texts: list[str],
            encoder: TextEncoder,
            batch_size: int,
        ) -> np.ndarray:
            encoded_batches = [
                build_feature_matrix(encoder, batch)
                for batch in iter_batches(texts, batch_size)
            ]
            if not encoded_batches:
                return np.empty((0, encoder.dimension), dtype=float)
            return np.vstack(encoded_batches)


        corpus = ["card charged twice", "package delayed", "login failed", "cancel plan"]
        matrix = encode_dataset(corpus, CharacterCountEncoder(), batch_size=2)
        assert matrix.shape == (4, 3)
        assert np.isfinite(matrix).all()
        matrix
        """),
        md("""
        ## 추가 연습

        - TODO 3: `BatchIterator`에 `drop_last: bool` 옵션을 추가하세요.
        - TODO 4: `timer`가 측정값을 리스트에도 기록하도록 확장하세요.
        - TODO 5: `TextEncoder` 계약을 일부러 위반하는 클래스를 만들고 shape 오류를 확인하세요.
        - TODO 6: `ExperimentConfig`를 JSON으로 저장하고 다시 복원해 동등성을 검사하세요.

        다음 노트북에서는 이 Python 도구를 바탕으로 NumPy 텐서 계산을 다룹니다.
        """),
    ]
    return base_notebook(cells)


def notebook_02():
    cells = [
        md("""
        # 02. NumPy: 벡터화, broadcasting, 수치 안정성, 선형대수

        딥러닝 프레임워크의 핵심 연산은 NumPy와 같은 배열 사고방식에서 출발합니다.
        이 노트북에서는 shape·axis·dtype을 추적하면서 **벡터화, broadcasting, 난수 생성기,
        안정적인 softmax/cross-entropy, 행렬곱, cosine similarity, 최소제곱, SVD**를 연습합니다.
        """),
        code(COMMON_BOOTSTRAP),
        code("""
        from time import perf_counter

        import numpy as np

        np.set_printoptions(precision=4, suppress=True)
        """),
        md("""
        ## 1. 배열의 기본 계약: shape, dtype, ndim

        AI 코드에서 가장 흔한 오류는 값 자체보다 shape의 의미를 잃는 것입니다.
        아래 배열은 `(batch, features)` 구조입니다.
        """),
        code("""
        features = np.array([
            [2.0, 0.5, 1.0],
            [1.0, 1.5, 0.0],
            [3.0, 0.2, 1.0],
            [0.5, 2.0, 0.0],
        ], dtype=np.float32)

        assert features.shape == (4, 3)
        assert features.ndim == 2
        assert features.dtype == np.float32
        print("shape:", features.shape, "dtype:", features.dtype)
        features
        """),
        md("""
        ## 2. Python loop와 벡터화

        벡터화는 연산 의도를 배열 단위로 표현해 C/BLAS 수준 구현을 활용합니다.
        성능 측정은 환경에 따라 달라지므로 속도를 assert하지 않고 결과의 동등성만 검증합니다.
        """),
        code("""
        values = np.linspace(-3, 3, 200_000, dtype=np.float64)

        started = perf_counter()
        loop_result = np.array([value * value + 2 * value + 1 for value in values])
        loop_ms = (perf_counter() - started) * 1_000

        started = perf_counter()
        vectorized_result = values**2 + 2 * values + 1
        vectorized_ms = (perf_counter() - started) * 1_000

        assert np.allclose(loop_result, vectorized_result)
        print(f"loop={loop_ms:.2f} ms, vectorized={vectorized_ms:.2f} ms")
        """),
        md("""
        ## 3. Broadcasting

        `(batch, features)` 배열에서 `(features,)` 평균을 빼면 NumPy가 마지막 축을 맞춰 확장합니다.
        실제 복사가 아니라 호환 가능한 shape 규칙이 적용됩니다.
        """),
        code("""
        feature_mean = features.mean(axis=0)       # (features,)
        centered = features - feature_mean         # (batch, features) - (features,)

        assert feature_mean.shape == (3,)
        assert centered.shape == features.shape
        assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-6)
        centered
        """),
        md("""
        ## 4. `axis`와 `keepdims`

        `axis=0`은 batch를 모으고 feature별 통계를, `axis=1`은 feature를 모으고 샘플별 통계를 냅니다.
        `keepdims=True`는 축을 길이 1로 유지해 후속 broadcasting을 명확하게 합니다.
        """),
        code("""
        row_sums = features.sum(axis=1, keepdims=True)
        normalized_rows = features / np.where(row_sums == 0, 1, row_sums)

        assert row_sums.shape == (4, 1)
        assert np.allclose(normalized_rows.sum(axis=1), 1.0)
        normalized_rows
        """),
        md("""
        ## 5. Boolean mask와 fancy indexing

        조건을 배열로 만들고 필요한 샘플만 선택합니다. 원본 순서를 보존하지만 대부분 새 배열을 만듭니다.
        """),
        code("""
        confidence = np.array([0.92, 0.41, 0.77, 0.33])
        review_mask = confidence < 0.6
        review_rows = features[review_mask]

        assert review_mask.tolist() == [False, True, False, True]
        assert review_rows.shape == (2, 3)
        review_rows
        """),
        md("""
        ## 6. 전역 난수 대신 `Generator` 사용

        함수 내부에 독립적인 `Generator`를 전달하면 실험 간 난수 상태가 덜 얽힙니다.
        동일한 seed로 만든 두 generator는 같은 수열을 생성합니다.
        """),
        code("""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        sample_a = rng_a.normal(loc=0.0, scale=1.0, size=(2, 4))
        sample_b = rng_b.normal(loc=0.0, scale=1.0, size=(2, 4))

        assert np.array_equal(sample_a, sample_b)
        sample_a
        """),
        md("""
        ## 7. 수치적으로 안정적인 softmax

        `exp(1000)`은 overflow합니다. 각 행의 최댓값을 먼저 빼도 softmax 결과는 같고 지수 범위는 안전해집니다.
        """),
        code("""
        def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
            logits = np.asarray(logits, dtype=np.float64)
            shifted = logits - logits.max(axis=axis, keepdims=True)
            exponentials = np.exp(shifted)
            return exponentials / exponentials.sum(axis=axis, keepdims=True)


        large_logits = np.array([[1_000.0, 1_001.0, 999.0], [-1_000.0, -999.0, -998.0]])
        probabilities = stable_softmax(large_logits)

        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        probabilities
        """),
        md("""
        ## 8. log-sum-exp와 cross-entropy

        정답 클래스의 `-log(probability)`가 cross-entropy입니다. 확률을 먼저 계산한 뒤 log를 취하는 대신
        logits 공간의 log-sum-exp를 사용하면 극단적인 값에서도 안정적입니다.
        """),
        code("""
        def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
            maximum = values.max(axis=axis, keepdims=True)
            shifted_sum = np.exp(values - maximum).sum(
                axis=axis,
                keepdims=True,
            )
            return (maximum + np.log(shifted_sum)).squeeze(axis)


        def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
            row_indices = np.arange(len(logits))
            losses = logsumexp(logits, axis=1) - logits[row_indices, targets]
            return float(losses.mean())


        targets = np.array([1, 2])
        loss = cross_entropy_from_logits(large_logits, targets)
        expected = -np.log(probabilities[np.arange(2), targets]).mean()
        assert np.isclose(loss, expected)
        print("cross-entropy:", round(loss, 6))
        """),
        md("""
        ## 9. 행렬곱: batch 입력과 선형 계층

        선형 계층은 `X @ W + b`입니다. `(batch, input_dim) @ (input_dim, output_dim)`의 결과는
        `(batch, output_dim)`이 됩니다.
        """),
        code("""
        rng = np.random.default_rng(7)
        weights = rng.normal(0, 0.1, size=(3, 2))
        bias = np.zeros(2)
        logits = features @ weights + bias

        assert logits.shape == (4, 2)
        logits
        """),
        md("""
        ## 10. Cosine similarity

        임베딩 검색은 방향이 비슷한 벡터를 찾습니다. 0 벡터의 norm은 작은 epsilon으로 보호합니다.
        """),
        code("""
        def cosine_similarity(
            query: np.ndarray,
            documents: np.ndarray,
            eps: float = 1e-12,
        ) -> np.ndarray:
            query = np.asarray(query, dtype=float)
            documents = np.asarray(documents, dtype=float)
            query_norm = max(float(np.linalg.norm(query)), eps)
            document_norms = np.maximum(np.linalg.norm(documents, axis=1), eps)
            return (documents @ query) / (document_norms * query_norm)


        query = np.array([1.0, 0.0, 1.0])
        similarities = cosine_similarity(query, features)
        assert similarities.shape == (4,)
        assert np.all((-1 <= similarities) & (similarities <= 1))
        similarities
        """),
        md("""
        ## 11. 최소제곱으로 선형 회귀 풀기

        역행렬을 직접 계산하기보다 `np.linalg.lstsq`를 사용합니다. rank가 부족한 경우에도 더 안정적입니다.
        """),
        code("""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 1.5 * x + 0.7
        design = np.column_stack([x, np.ones_like(x)])

        coefficients, residuals, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
        slope, intercept = coefficients

        assert np.isclose(slope, 1.5)
        assert np.isclose(intercept, 0.7)
        print({"slope": slope, "intercept": intercept, "rank": rank})
        """),
        md("""
        ## 12. SVD와 저차원 표현

        중심화한 행렬을 SVD로 분해하면 주성분 방향을 얻을 수 있습니다.
        여기서는 첫 두 방향으로 투영하고 shape과 복원 오차를 확인합니다.
        """),
        code("""
        centered64 = features.astype(np.float64) - features.mean(axis=0)
        u, singular_values, vt = np.linalg.svd(centered64, full_matrices=False)
        projected_2d = centered64 @ vt[:2].T
        reconstructed = projected_2d @ vt[:2]

        assert projected_2d.shape == (4, 2)
        assert np.linalg.norm(centered64 - reconstructed) >= 0
        print("singular values:", singular_values)
        projected_2d
        """),
        md("""
        ## 13. View와 copy

        slice는 원본 메모리를 공유할 수 있습니다. 전처리 중 원본을 보존해야 한다면 `.copy()`를 명시하세요.
        """),
        code("""
        original = np.arange(8)
        view = original[2:5]
        independent = original[2:5].copy()

        view[0] = 999
        independent[1] = -1

        assert original[2] == 999       # view 수정이 원본에 반영됨
        assert original[3] != -1        # copy 수정은 원본과 독립적
        original, view, independent
        """),
        md("""
        ## TODO 1: feature 표준화 함수

        구현을 지운 뒤 `mean=0`, `std=1`이 되도록 다시 작성하세요. 분산이 0인 열은 0으로 유지해야 합니다.
        """),
        code("""
        def standardize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
            # TODO: 열별 평균과 표준편차를 사용해 표준화하세요.
            matrix = np.asarray(matrix, dtype=np.float64)
            mean = matrix.mean(axis=0, keepdims=True)
            std = matrix.std(axis=0, keepdims=True)
            safe_std = np.where(std < eps, 1.0, std)
            return (matrix - mean) / safe_std


        standardized = standardize(features)
        assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-7)
        assert np.allclose(standardized.std(axis=0), 1.0, atol=1e-7)
        standardized
        """),
        md("""
        ## TODO 2: batch query-document 유사도

        loop 없이 `(queries, dim) @ (documents, dim).T`를 사용해 모든 조합의 cosine similarity를 계산하세요.
        """),
        code("""
        def pairwise_cosine(
            queries: np.ndarray,
            documents: np.ndarray,
            eps: float = 1e-12,
        ) -> np.ndarray:
            # TODO: 각 행을 L2 정규화한 뒤 행렬곱하세요.
            query_norms = np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), eps)
            document_norms = np.maximum(np.linalg.norm(documents, axis=1, keepdims=True), eps)
            return (queries / query_norms) @ (documents / document_norms).T


        pairwise = pairwise_cosine(features[:2], features)
        assert pairwise.shape == (2, 4)
        assert np.allclose(np.diag(pairwise[:, :2]), 1.0)
        pairwise
        """),
        md("""
        ## 정리

        - 항상 `(batch, sequence, hidden)`처럼 각 축의 의미를 말로 적습니다.
        - 벡터화는 속도뿐 아니라 코드의 수학적 의도를 선명하게 합니다.
        - softmax/log 연산은 overflow와 underflow를 고려합니다.
        - 역행렬보다 `solve`, `lstsq`, SVD 같은 안정적인 도구를 우선합니다.
        - 다음 노트북에서는 같은 원칙을 표 형태 데이터와 데이터 누수 문제에 적용합니다.
        """),
    ]
    cells = merge_markdown_into_previous(
        cells,
        (
            "## 4. `axis`와 `keepdims`",
            "## 6. 전역 난수 대신",
            "## 10. Cosine similarity",
            "## 12. SVD와 저차원 표현",
        ),
    )
    return base_notebook(cells)


def notebook_03():
    cells = [
        md("""
        # 03. pandas EDA와 누수 없는 데이터 분할

        모델링 전에 데이터 계약, 결측, 중복, 그룹 구조, 레이블 분포를 확인합니다.
        이 프로젝트의 두 문장은 같은 고객 사건(`case_id`)의 표현 변형일 수 있으므로 행 단위 무작위 분할은 누수를 만듭니다.
        """),
        code(COMMON_BOOTSTRAP),
        code("""
        import numpy as np
        import pandas as pd

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None

        from llm_engineering_lab.data import (
            TicketSchema,
            dataset_profile,
            load_ticket_data,
            stratified_group_split,
            validate_ticket_frame,
        )

        DATA_PATH = PROJECT_ROOT / "data" / "customer_support_tickets.csv"
        tickets = load_ticket_data(DATA_PATH)
        schema = TicketSchema()
        print("shape:", tickets.shape)
        tickets.head()
        """),
        md("""
        ## 1. 스키마와 dtype

        ID는 숫자 계산 대상이 아니므로 문자열로, 시간은 timezone을 포함한 datetime으로 유지합니다.
        `info()`는 결측과 메모리 사용량을 함께 보는 빠른 출발점입니다.
        """),
        code("""
        tickets.info()

        assert set(schema.required_columns).issubset(tickets.columns)
        assert pd.api.types.is_datetime64_any_dtype(tickets[schema.created_at])
        """),
        md("""
        ## 2. 재사용 가능한 프로파일

        노트북의 일회성 출력만 보지 않고, 운영 파이프라인에서도 호출할 수 있는 `dataset_profile`을 사용합니다.
        """),
        code("""
        profile = dataset_profile(tickets)
        profile.to_dict()
        """),
        md("""
        ## 3. 결측값: 비율과 의미를 함께 보기

        실제 파일은 엄격한 검증을 통과했으므로 결측이 없습니다. 작은 복사본에 결측을 주입해 탐지와 처리 차이를 연습합니다.
        결측을 무조건 제거하기 전에 “수집 실패인가, 해당 없음인가”를 구분해야 합니다.
        """),
        code("""
        dirty = tickets.head(8).copy()
        dirty.loc[1, "channel"] = pd.NA
        dirty.loc[3, "text"] = ""

        missing_report = pd.DataFrame({
            "na_count": dirty.isna().sum(),
            "blank_count": dirty.select_dtypes(include=["string", "object"])
                                .apply(
                                    lambda column: column.astype("string")
                                    .str.strip()
                                    .eq("")
                                    .sum()
                                ),
        }).fillna(0).astype(int)
        missing_report[missing_report.sum(axis=1) > 0]
        """),
        code("""
        # 범주형 결측은 unknown으로 표시할 수 있지만, 핵심 텍스트 결측은 보통 학습에서 제외합니다.
        cleaned_demo = dirty.copy()
        cleaned_demo["channel"] = cleaned_demo["channel"].fillna("unknown")
        cleaned_demo = cleaned_demo[cleaned_demo["text"].fillna("").str.strip().ne("")]

        assert cleaned_demo["channel"].isna().sum() == 0
        assert cleaned_demo["text"].str.strip().ne("").all()
        cleaned_demo[["ticket_id", "channel", "text"]]
        """),
        md("""
        ## 4. 중복: 행 중복과 의미 중복은 다르다

        `ticket_id` 중복은 데이터 계약 오류입니다. 반면 동일 `case_id`의 여러 표현은 의도된 그룹일 수 있습니다.
        """),
        code("""
        duplicated_demo = pd.concat([tickets.head(3), tickets.iloc[[0]]], ignore_index=True)
        duplicate_ids = duplicated_demo.loc[
            duplicated_demo["ticket_id"].duplicated(keep=False), "ticket_id"
        ]

        assert duplicate_ids.nunique() == 1
        duplicated_demo[duplicated_demo["ticket_id"].duplicated(keep=False)]
        """),
        code("""
        deduplicated = duplicated_demo.drop_duplicates(subset="ticket_id", keep="first")
        assert len(deduplicated) == len(duplicated_demo) - 1

        case_sizes = tickets.groupby("case_id", observed=True).size()
        print("case당 행 수 분포:")
        print(case_sizes.value_counts().sort_index())
        """),
        md("""
        ## 5. GroupBy와 교차표

        평균만 보지 말고 표본 수, 범주 분포, 그룹별 편향을 함께 확인합니다.
        """),
        code("""
        label_summary = (
            tickets.assign(text_chars=tickets["text"].str.len())
            .groupby("label", observed=True)
            .agg(
                rows=("ticket_id", "size"),
                cases=("case_id", "nunique"),
                mean_text_chars=("text_chars", "mean"),
                high_priority_rate=("priority", lambda values: values.eq("high").mean()),
            )
            .sort_values("rows", ascending=False)
        )
        label_summary
        """),
        code("""
        channel_by_label = pd.crosstab(
            tickets["label"],
            tickets["channel"],
            normalize="index",
        ).round(3)

        assert np.allclose(channel_by_label.sum(axis=1), 1.0)
        channel_by_label
        """),
        md("""
        ## 6. 벡터화된 feature engineering

        원본 컬럼을 덮어쓰지 않고 파생 변수를 새 DataFrame에 추가합니다.
        텍스트 길이는 신호가 될 수 있지만 레이블 생성 이후에만 알 수 있는 정보는 feature로 쓰면 안 됩니다.
        """),
        code("""
        enriched = tickets.assign(
            text_chars=tickets["text"].str.len(),
            word_count=tickets["text"].str.split().str.len(),
            hour_utc=tickets["created_at"].dt.hour,
            day_of_week=tickets["created_at"].dt.day_name(),
            is_high_priority=tickets["priority"].eq("high").astype("int8"),
        )

        assert enriched["text_chars"].gt(0).all()
        enriched[["text", "text_chars", "word_count", "hour_utc"]].head()
        """),
        code("""
        daily_counts = (
            tickets.set_index("created_at")
            .resample("7D")["ticket_id"]
            .count()
            .rename("tickets")
        )
        daily_counts.head()
        """),
        md("""
        ## 7. 시각화: 먼저 질문을 정한다

        “레이블 불균형이 있는가?”를 확인하기 위한 막대그래프입니다.
        `matplotlib`이 없다면 같은 집계표를 출력해 노트북 전체 흐름은 유지합니다.
        """),
        code("""
        label_counts = tickets["label"].value_counts().sort_values()

        if plt is None:
            print("matplotlib 미설치 — 집계표로 대체합니다.")
            print(label_counts)
        else:
            ax = label_counts.plot.barh(figsize=(8, 4), color="#4C78A8")
            ax.set(title="Tickets per label", xlabel="rows", ylabel="label")
            ax.grid(axis="x", alpha=0.2)
            plt.tight_layout()
            plt.show()
        """),
        md("""
        ## 8. 데이터 계약을 실패시켜 보기

        검증 함수는 필수 열, 빈 값, ID 중복, 한 case의 복수 label, timestamp를 조기에 거부합니다.
        학습 중간의 모호한 오류보다 입력 경계에서 명확히 실패하는 편이 안전합니다.
        """),
        code("""
        invalid = tickets.head(10).copy()
        invalid.loc[1, "ticket_id"] = invalid.loc[0, "ticket_id"]

        try:
            validate_ticket_frame(invalid, expected_labels=None)
        except ValueError as error:
            assert "duplicate ticket_id" in str(error)
            print("예상한 계약 오류:", error)
        else:
            raise AssertionError("중복 ID가 거부되어야 합니다.")
        """),
        md("""
        ## 9. 행 단위 무작위 분할이 만드는 group leakage

        같은 `case_id`의 두 표현이 train/validation에 갈라지면 모델이 사건 고유 표현을 기억해 성능을 과대평가할 수 있습니다.
        """),
        code("""
        from sklearn.model_selection import train_test_split

        row_train, row_validation = train_test_split(
            tickets,
            test_size=0.2,
            random_state=42,
            stratify=tickets["label"],
        )
        leaked_cases = set(row_train["case_id"]) & set(row_validation["case_id"])

        print("행 분할 train/validation 중복 case 수:", len(leaked_cases))
        print("예시:", sorted(leaked_cases)[:5])
        """),
        md("""
        ## 10. 레이블별 group 분할

        프로젝트 함수는 각 레이블 안에서 `case_id`를 셔플하고 case 전체를 한쪽에 배치합니다.
        행 수가 아니라 그룹 수를 기준으로 분리한다는 점이 핵심입니다.
        """),
        code("""
        split = stratified_group_split(tickets, validation_size=0.2, seed=42)
        train_cases = set(split.train["case_id"])
        validation_cases = set(split.validation["case_id"])

        split.assert_no_group_leakage()
        assert train_cases.isdisjoint(validation_cases)
        assert set(split.train["label"]) == set(split.validation["label"])

        print("train     :", split.train.shape, "cases=", len(train_cases))
        print("validation:", split.validation.shape, "cases=", len(validation_cases))
        """),
        code("""
        split_distribution = pd.concat(
            {
                "train": split.train["label"].value_counts(normalize=True),
                "validation": split.validation["label"].value_counts(normalize=True),
            },
            axis=1,
        ).fillna(0).sort_index()

        split_distribution["absolute_gap"] = (
            split_distribution["train"] - split_distribution["validation"]
        ).abs()
        split_distribution
        """),
        code("""
        if plt is None:
            print(split_distribution[["train", "validation"]])
        else:
            ax = split_distribution[["train", "validation"]].plot.bar(figsize=(9, 4))
            ax.set(title="Label proportion after group split", ylabel="proportion", xlabel="label")
            ax.legend(title="split")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.show()
        """),
        md("""
        ## TODO: 재사용 가능한 품질 보고서

        구현을 지운 뒤 요구사항만 보고 다시 작성하세요. 반환값은 테스트·로그에 넣기 쉬운 dict여야 합니다.
        """),
        code("""
        def quality_report(frame: pd.DataFrame) -> dict[str, int | float]:
            # TODO 1: 행 수, case 수, 결측 셀 수, ticket_id 중복 수, 평균 텍스트 길이를 계산하세요.
            return {
                "rows": len(frame),
                "cases": int(frame["case_id"].nunique()),
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_ticket_ids": int(frame["ticket_id"].duplicated().sum()),
                "mean_text_chars": float(frame["text"].str.len().mean()),
            }


        report = quality_report(tickets)
        assert report["rows"] == len(tickets)
        assert report["cases"] == tickets["case_id"].nunique()
        assert report["duplicate_ticket_ids"] == 0
        report
        """),
        md("""
        ## 추가 연습

        - TODO 2: `channel × priority` 조합별 건수와 평균 텍스트 길이를 계산하세요.
        - TODO 3: 매 레이블의 validation case가 최소 1개인지 assert를 작성하세요.
        - TODO 4: `created_at` 이후에 발생한 사건만 사용하는 시간 기반 분할을 설계하고 group 누수를 검사하세요.
        - TODO 5: 데이터 드리프트를 감지하기 위해 월별 channel 분포를 비교하세요.

        다음 노트북에서는 이 group split을 고정한 채 scikit-learn 기준 모델을 학습합니다.
        """),
    ]
    cells = merge_markdown_into_previous(
        cells,
        ("## 4. 중복", "## 6. 벡터화된 feature engineering"),
    )
    return base_notebook(cells)


def notebook_04():
    cells = [
        md("""
        # 04. scikit-learn 텍스트 분류 기준 모델

        복잡한 신경망 전에 빠르고 설명 가능한 기준선을 만듭니다.
        이 노트북은 **정규화 → TF-IDF/One-Hot → ColumnTransformer → Pipeline → Logistic Regression →
        metrics/confusion matrix/error analysis**를 연결하고, group leakage가 평가를 어떻게 왜곡하는지 확인합니다.
        """),
        code(COMMON_BOOTSTRAP),
        code("""
        import numpy as np
        import pandas as pd

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import classification_report, ConfusionMatrixDisplay
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder

        from llm_engineering_lab.data import TicketSchema, load_ticket_data, stratified_group_split
        from llm_engineering_lab.ml import (
            TrainingConfig,
            build_error_table,
            build_model,
            evaluate_model,
            normalize_ticket_text,
            predict_tickets,
        )

        tickets = load_ticket_data(PROJECT_ROOT / "data" / "customer_support_tickets.csv")
        schema = TicketSchema()
        split = stratified_group_split(tickets, validation_size=0.2, seed=42)
        print("train:", split.train.shape, "validation:", split.validation.shape)
        """),
        md("""
        ## 1. 텍스트 정규화

        Unicode 표현과 공백을 통일하고 이메일·긴 숫자처럼 샘플마다 달라지는 식별자를 마스킹합니다.
        레이블 정보를 암시하는 후처리나 전체 데이터 통계를 사용하면 안 됩니다.
        """),
        code("""
        normalization_examples = [
            "  카드 결제가   두 번 됐어요  ",
            "User.Test@example.com 계정의 주문 번호는 123456 입니다",
            "ＡＰＩ 오류가 발생했습니다",  # 전각 문자
        ]

        normalized = [normalize_ticket_text(text) for text in normalization_examples]
        assert "<email>" in normalized[1]
        assert "<number>" in normalized[1]
        normalized
        """),
        md("""
        ## 2. TF-IDF를 작은 문장으로 해부하기

        TF-IDF는 한 문서 안에서 자주 나오지만 전체 문서에는 흔하지 않은 항목을 강조합니다.
        한국어 형태소 분석기 없이도 문자 n-gram은 오탈자와 조사 변화에 비교적 강한 기준선이 됩니다.
        """),
        code("""
        tiny_corpus = [
            "카드 결제가 두 번 됐어요",
            "카드 결제를 취소해 주세요",
            "배송이 너무 늦어요",
        ]
        tiny_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        tiny_matrix = tiny_vectorizer.fit_transform(tiny_corpus)

        print("sparse shape:", tiny_matrix.shape)
        print("non-zero values:", tiny_matrix.nnz)
        assert tiny_matrix.shape[0] == 3
        """),
        code("""
        feature_names = tiny_vectorizer.get_feature_names_out()
        first_row = tiny_matrix[0].toarray().ravel()
        top_indices = np.argsort(first_row)[-8:][::-1]
        pd.DataFrame({
            "ngram": feature_names[top_indices],
            "tfidf": first_row[top_indices],
        })
        """),
        md("""
        ## 3. 범주형 metadata와 unknown 처리

        `OneHotEncoder(handle_unknown="ignore")`는 추론 시 처음 보는 channel이 들어와도 전체 요청을 실패시키지 않습니다.
        unknown이 모두 0으로 표현된다는 운영 의미는 별도로 모니터링해야 합니다.
        """),
        code("""
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_channels = pd.DataFrame({"channel": ["web", "email", "chat"]})
        encoded_train = encoder.fit_transform(train_channels)
        encoded_new = encoder.transform(pd.DataFrame({"channel": ["phone"]}))

        assert encoded_train.shape == (3, 3)
        assert np.all(encoded_new == 0)
        pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out())
        """),
        md("""
        ## 4. `Pipeline`으로 전처리와 모델을 묶기

        `fit`은 train에만 호출합니다. validation에 `fit_transform`을 다시 호출하면
        vocabulary와 category 공간이 달라지고 누수가 생깁니다.
        Pipeline은 저장·추론 때도 동일한 변환을 재사용하게 해 줍니다.
        """),
        code("""
        config = TrainingConfig(
            seed=42,
            validation_size=0.2,
            max_features=20_000,
            ngram_range=(2, 5),
            regularization_c=4.0,
        )
        model = build_model(config, schema)
        print(model)
        """),
        code("""
        train_x = split.train.loc[:, schema.model_feature_columns]
        train_y = split.train[schema.label]
        validation_x = split.validation.loc[:, schema.model_feature_columns]
        validation_y = split.validation[schema.label]

        model.fit(train_x, train_y)
        predictions = model.predict(validation_x)

        assert len(predictions) == len(validation_y)
        assert set(predictions).issubset(set(train_y))
        print("학습 완료")
        """),
        md("""
        ## 5. Pipeline 내부 shape 확인

        전처리 결과는 대부분 0인 sparse matrix입니다. 대규모 vocabulary를 dense 배열로 바꾸면 메모리가 급증합니다.
        """),
        code("""
        preprocessor = model.named_steps["preprocessing"]
        transformed_train = preprocessor.transform(train_x)

        print("type :", type(transformed_train).__name__)
        print("shape:", transformed_train.shape)
        print("nnz  :", transformed_train.nnz)
        density = transformed_train.nnz / (transformed_train.shape[0] * transformed_train.shape[1])
        print("density:", round(density, 6))
        assert density < 0.5
        """),
        md("""
        ## 6. 확률과 예측

        가장 높은 확률은 불확실성 순위를 정하는 신호이지, 자동으로 보정된 확률을 뜻하지는 않습니다.
        클래스 순서는 `classifier.classes_`와 함께 확인합니다.
        """),
        code("""
        classifier = model.named_steps["classifier"]
        probabilities = model.predict_proba(validation_x)

        probability_table = pd.DataFrame(probabilities, columns=classifier.classes_)
        probability_table.insert(0, "actual", validation_y.reset_index(drop=True))
        probability_table.insert(1, "predicted", predictions)

        assert np.allclose(probabilities.sum(axis=1), 1.0)
        probability_table.head()
        """),
        md("""
        ## 7. Accuracy만 보지 않는 평가

        Macro F1은 각 클래스를 동일한 비중으로 평균내어 소수 클래스 성능 저하를 드러냅니다.
        지원 건수(`support`)와 클래스별 precision/recall을 함께 봅니다.
        """),
        code("""
        report = evaluate_model(model, validation_x, validation_y)
        print({
            "accuracy": round(report.accuracy, 4),
            "macro_f1": round(report.macro_f1, 4),
            "weighted_f1": round(report.weighted_f1, 4),
        })

        pd.DataFrame({
            label: {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "support": metrics.support,
            }
            for label, metrics in report.per_class.items()
        }).T
        """),
        code("""
        # sklearn의 텍스트 보고서도 비교해 봅니다.
        print(classification_report(validation_y, predictions, zero_division=0))
        """),
        md("""
        ## 8. Confusion matrix

        행은 실제 레이블, 열은 예측 레이블입니다. 어떤 클래스 쌍이 반복해서 혼동되는지 확인합니다.
        """),
        code("""
        confusion = pd.DataFrame(
            report.confusion_matrix,
            index=pd.Index(report.labels, name="actual"),
            columns=pd.Index(report.labels, name="predicted"),
        )
        confusion
        """),
        code("""
        if plt is None:
            print(confusion)
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(
                confusion_matrix=np.asarray(report.confusion_matrix),
                display_labels=report.labels,
            ).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
            ax.set_title("Group-split validation confusion matrix")
            plt.tight_layout()
            plt.show()
        """),
        md("""
        ## 9. 오류 분석 표

        틀린 샘플을 confidence가 높은 순으로 보면 모델이 확신하며 틀린 데이터·레이블 문제를 먼저 찾을 수 있습니다.
        데이터가 작고 쉬우면 오류가 0개일 수 있으므로 low-margin 정답도 함께 관찰합니다.
        """),
        code("""
        errors = build_error_table(model, split.validation, schema)

        if errors.empty:
            margins = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
            uncertain = split.validation[["ticket_id", "case_id", "text", "label"]].copy()
            uncertain["predicted_label"] = predictions
            uncertain["confidence"] = probabilities.max(axis=1)
            uncertain["margin"] = margins
            print("오분류가 없어 margin이 작은 정답 사례를 표시합니다.")
            analysis_table = uncertain.sort_values("margin").head(10)
        else:
            analysis_table = errors.head(10)

        analysis_table
        """),
        md("""
        ## 10. Leakage 실험: 행 분할과 group 분할 비교

        행 무작위 분할에서는 같은 case의 변형 문장이 양쪽에 들어갈 수 있습니다.
        성능 차이가 항상 크게 나타나는 것은 아니지만, 평가 가정 자체가 깨졌다는 사실이 중요합니다.
        """),
        code("""
        row_train, row_validation = train_test_split(
            tickets,
            test_size=0.2,
            random_state=42,
            stratify=tickets["label"],
        )
        overlap = set(row_train["case_id"]) & set(row_validation["case_id"])

        row_model = build_model(config, schema)
        row_model.fit(row_train.loc[:, schema.model_feature_columns], row_train["label"])
        row_report = evaluate_model(
            row_model,
            row_validation.loc[:, schema.model_feature_columns],
            row_validation["label"],
        )

        print("leaked case count:", len(overlap))
        """),
        code("""
        comparison = pd.DataFrame([
            {
                "split": "random rows (leaky)",
                "accuracy": row_report.accuracy,
                "macro_f1": row_report.macro_f1,
                "overlapping_cases": len(overlap),
            },
            {
                "split": "stratified groups",
                "accuracy": report.accuracy,
                "macro_f1": report.macro_f1,
                "overlapping_cases": 0,
            },
        ]).set_index("split")

        assert comparison.loc["stratified groups", "overlapping_cases"] == 0
        comparison
        """),
        md("""
        ## 11. Feature leakage가 없는지 vocabulary 확인

        TF-IDF vocabulary는 train에서만 학습되었습니다. validation 전용 토큰을 만들어 변환해도 vocabulary 크기는 변하지 않습니다.
        """),
        code("""
        text_vectorizer = preprocessor.named_transformers_["text_tfidf"]
        vocabulary_size_before = len(text_vectorizer.vocabulary_)

        _ = model.predict(pd.DataFrame({
            "text": ["validationonlytoken_xyz"],
            "channel": ["unknown"],
            "priority": ["unknown"],
            "customer_tier": ["unknown"],
        }))

        assert len(text_vectorizer.vocabulary_) == vocabulary_size_before
        assert "validationonlytoken_xyz" not in text_vectorizer.vocabulary_
        print("vocabulary size:", vocabulary_size_before)
        """),
        md("""
        ## 12. Logistic Regression의 feature 가중치

        양의 계수가 큰 feature는 해당 클래스로 log-odds를 밀어 올립니다.
        상관관계 기반 설명이지 인과 설명은 아니며, 문자 n-gram은 조각 단위로 읽어야 합니다.
        """),
        code("""
        all_feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_

        top_features: dict[str, list[tuple[str, float]]] = {}
        for class_index, label in enumerate(classifier.classes_):
            top_indices = np.argsort(coefficients[class_index])[-8:][::-1]
            top_features[str(label)] = [
                (str(all_feature_names[index]), round(float(coefficients[class_index, index]), 4))
                for index in top_indices
            ]

        pd.DataFrame(top_features)
        """),
        md("""
        ## 13. 새 문의 추론

        학습과 동일한 Pipeline을 통과시키므로 정규화, TF-IDF, metadata 인코딩 순서를 수동으로 맞출 필요가 없습니다.
        """),
        code("""
        new_tickets = pd.DataFrame([
            {
                "text": "카드 결제가 두 번 처리됐어요",
                "channel": "chat",
                "priority": "high",
                "customer_tier": "pro",
            },
            {
                "text": "로그인이 계속 실패하고 복구 코드도 안 와요",
                "channel": "web",
                "priority": "medium",
                "customer_tier": "free",
            },
            {
                "text": "배송 조회가 며칠째 같은 위치예요",
                "channel": "email",
                "priority": "low",
                "customer_tier": "business",
            },
        ])

        inference = predict_tickets(model, new_tickets)
        assert len(inference) == 3
        inference
        """),
        md("""
        ## 14. TODO: 사람 검토 threshold

        운영에서는 confidence가 낮은 예측을 자동 처리하지 않고 사람에게 보낼 수 있습니다.
        threshold는 비용과 대표 validation 데이터에 맞춰 조정해야 하며 0.7은 단지 실습값입니다.
        """),
        code("""
        def route_for_review(
            prediction_frame: pd.DataFrame,
            threshold: float = 0.7,
        ) -> pd.DataFrame:
            # TODO 1: confidence가 threshold보다 낮으면 needs_review=True로 표시하세요.
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold must be in [0, 1]")
            output = prediction_frame.copy()
            output["needs_review"] = output["confidence"].lt(threshold)
            return output


        routed = route_for_review(inference, threshold=0.7)
        assert routed["needs_review"].dtype == bool
        routed[["text", "predicted_label", "confidence", "needs_review"]]
        """),
        md("""
        ## 직접 바꿔 볼 실험

        - TODO 2: `ngram_range=(1, 3)`과 `(3, 6)`의 macro F1, feature 수, 학습 시간을 비교하세요.
        - TODO 3: metadata 가중치를 0 또는 1로 바꾸고 channel 편향을 분석하세요.
        - TODO 4: 가장 margin이 작은 문장의 일부 단어를 바꿔 확률 변화를 기록하세요.
        - TODO 5: `C`를 0.1, 1, 10으로 바꾸고 과소/과적합 경향을 비교하세요.
        - TODO 6: 오류 유형을 `ambiguous`, `label_issue`, `missing_context`로 직접 태깅하세요.

        기준 모델이 확보되면 이후 PyTorch 모델은 단순 정확도뿐 아니라 비용·지연·유지보수 측면에서도 이 기준선과 비교해야 합니다.
        """),
    ]
    cells = merge_markdown_into_previous(
        cells,
        (
            "## 2. TF-IDF를 작은 문장으로 해부하기",
            "## 4. `Pipeline`으로 전처리와 모델을 묶기",
            "## 6. 확률과 예측",
            "## 8. Confusion matrix",
            "## 10. Leakage 실험",
            "## 12. Logistic Regression의 feature 가중치",
            "## 14. TODO: 사람 검토 threshold",
        ),
    )
    return base_notebook(cells)


def write_notebooks() -> list[tuple[Path, int]]:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_environment_and_jupyterlab.ipynb": notebook_00(),
        "01_intermediate_python_for_ai.ipynb": notebook_01(),
        "02_numpy_for_ml.ipynb": notebook_02(),
        "03_pandas_eda_and_group_split.ipynb": notebook_03(),
        "04_sklearn_text_baseline.ipynb": notebook_04(),
    }

    written: list[tuple[Path, int]] = []
    for filename, notebook in notebooks.items():
        destination = NOTEBOOK_DIR / filename
        annotate_notebook(notebook)
        nbformat.validate(notebook)
        nbformat.write(notebook, destination)
        written.append((destination, len(notebook.cells)))
    return written


if __name__ == "__main__":
    for path, cell_count in write_notebooks():
        print(f"{path.relative_to(ROOT)}: {cell_count} cells")
