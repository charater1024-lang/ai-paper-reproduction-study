"""LLM/ML 코드베이스에서 자주 쓰는 중급 Python 패턴 실습.

다루는 내용:
- dataclass로 데이터 계약 표현
- Protocol과 제네릭으로 교체 가능한 컴포넌트 설계
- generator로 큰 입력을 스트리밍
- decorator와 context manager로 지연/실험 메타데이터 기록
- 예외를 데이터 경계에서 일찍 발생시키기

실행: python scripts/00_python_engineering_patterns.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Protocol, TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Ticket:
    """파이프라인 단계 사이에서 공유하는 불변 데이터 계약."""

    ticket_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ticket_id.strip():
            raise ValueError("ticket_id는 비어 있을 수 없습니다.")
        if not self.text.strip():
            raise ValueError("text는 비어 있을 수 없습니다.")


class Step(Protocol[T]):
    """상속 없이도 구현할 수 있는 파이프라인 단계 인터페이스."""

    def __call__(self, value: T) -> T: ...


class NormalizeWhitespace:
    def __call__(self, ticket: Ticket) -> Ticket:
        return Ticket(ticket.ticket_id, " ".join(ticket.text.split()), ticket.metadata)


@dataclass(slots=True)
class RedactSecrets:
    """단순 예제용 마스커. 실제 환경에서는 검증된 DLP/PII 탐지기를 사용한다."""

    secret_markers: tuple[str, ...] = ("sk-", "api_key=")

    def __call__(self, ticket: Ticket) -> Ticket:
        words = ticket.text.split()
        safe_words = [
            "[REDACTED]"
            if any(marker in word.lower() for marker in self.secret_markers)
            else word
            for word in words
        ]
        return Ticket(ticket.ticket_id, " ".join(safe_words), ticket.metadata)


@dataclass(slots=True)
class Pipeline:
    steps: list[Step[Ticket]]

    def __call__(self, ticket: Ticket) -> Ticket:
        for step in self.steps:
            ticket = step(ticket)
        return ticket


def timed[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """원래 함수의 타입/메타데이터를 보존하는 지연 시간 decorator."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1_000
            print(f"metric function={function.__name__} latency_ms={elapsed_ms:.3f}")

    return wrapper


def batched[T](items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """전체 입력을 메모리에 올리지 않고 고정 크기 batch를 만든다."""

    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 합니다.")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@contextmanager
def experiment_run(name: str, output_dir: Path) -> Iterator[dict[str, object]]:
    """성공/실패 여부와 실행 시간을 JSON으로 남기는 context manager."""

    started = time.perf_counter()
    record: dict[str, object] = {
        "name": name,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
    }
    try:
        yield record
    except Exception as error:
        record.update(status="failed", error=type(error).__name__)
        raise
    else:
        record["status"] = "completed"
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@timed
def preprocess_batch(pipeline: Pipeline, tickets: list[Ticket]) -> list[Ticket]:
    return [pipeline(ticket) for ticket in tickets]


def main() -> None:
    tickets = (
        Ticket("T-001", "  로그인에   실패합니다  ", {"channel": "web"}),
        Ticket("T-002", "api_key=sk-example 키가 노출됐어요", {"channel": "email"}),
        Ticket("T-003", "결제가 두 번 되었어요", {"channel": "chat"}),
    )
    pipeline = Pipeline([NormalizeWhitespace(), RedactSecrets()])

    with experiment_run("python_patterns_demo", ROOT / "artifacts") as run:
        processed: list[Ticket] = []
        for batch_index, batch in enumerate(batched(tickets, batch_size=2), start=1):
            print(f"batch={batch_index} size={len(batch)}")
            processed.extend(preprocess_batch(pipeline, batch))
        run["processed_count"] = len(processed)
        run["redacted_count"] = sum("[REDACTED]" in ticket.text for ticket in processed)

    print(
        json.dumps(
            [asdict(ticket) for ticket in processed], ensure_ascii=False, indent=2
        )
    )
    print(
        "\nTODO: Step[Ticket]을 구현하는 UnicodeNormalizer를 추가하고 테스트를 작성하세요."
    )


if __name__ == "__main__":
    main()
