"""재현 가능한 합성 고객지원 티켓 CSV를 만든다.

실제 개인정보 없이도 그룹 누수, 클래스 분류, 오류 분석을 연습하도록 각 case마다
같은 의미의 문장 두 개를 만든다. 기본 파일이 이미 있으면 보존하며, 다시 만들려면
``--force``를 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDNAMES = (
    "ticket_id",
    "case_id",
    "created_at",
    "channel",
    "priority",
    "customer_tier",
    "text",
    "label",
)

SCENARIOS: dict[str, tuple[str, ...]] = {
    "account_access": (
        "비밀번호 재설정 메일이 오지 않아요",
        "로그인 실패가 반복되어 계정이 잠겼어요",
        "새 휴대전화에서 이중 인증 코드를 받을 수 없어요",
        "회사 이메일을 바꾼 뒤 기존 계정에 접근하지 못해요",
        "소셜 로그인 버튼을 누르면 다시 로그인 화면으로 돌아와요",
        "인증 앱을 삭제해서 복구 코드가 필요해요",
        "비밀번호를 바꿨는데 이전 비밀번호만 동작해요",
        "초대받은 팀 워크스페이스에 들어갈 수 없어요",
        "로그인 세션이 몇 분마다 자동으로 풀려요",
        "탈퇴를 취소했는데 계정이 비활성 상태예요",
    ),
    "billing": (
        "같은 카드 요금이 두 번 중복 결제됐어요",
        "무료 체험 중인데 예상하지 못한 청구가 발생했어요",
        "결제 영수증의 회사명을 수정하고 싶어요",
        "카드를 변경했는데 다음 결제에 반영되지 않았어요",
        "환불 승인 후 열흘이 지나도 입금되지 않았어요",
        "부가세가 포함된 세금계산서가 필요해요",
        "월간 요금제를 결제했는데 연간 금액이 청구됐어요",
        "카드 승인은 됐지만 서비스 요금제는 그대로예요",
        "해외 결제 수수료가 별도로 붙은 이유가 궁금해요",
        "팀 좌석을 줄였는데 청구 금액이 줄지 않았어요",
    ),
    "cancellation": (
        "다음 갱신 전에 구독을 해지하고 싶어요",
        "구독 취소 버튼이 설정 화면에 보이지 않아요",
        "해지하면 남은 결제 기간을 계속 쓸 수 있는지 궁금해요",
        "실수로 시작한 연간 구독을 취소하고 싶어요",
        "팀 요금제를 종료하면 멤버 데이터가 어떻게 되는지 궁금해요",
        "체험 기간 종료 전에 자동 갱신을 끄고 싶어요",
        "이미 해지했는데 다음 달 갱신 안내 메일이 왔어요",
        "모바일에서 구매한 구독을 웹에서 취소할 수 없어요",
        "서비스 탈퇴와 유료 구독 해지를 함께 처리하고 싶어요",
        "일시 중지한 구독을 완전히 종료하고 싶어요",
    ),
    "delivery": (
        "배송 조회가 이틀째 같은 위치에서 멈춰 있어요",
        "출고 전에 배송지 주소를 변경하고 싶어요",
        "주문한 상품 일부만 도착했어요",
        "배송 완료라고 표시되지만 물건을 받지 못했어요",
        "예정 배송일이 지났는데 운송장이 갱신되지 않아요",
        "해외 배송 통관에 필요한 정보를 알고 싶어요",
        "잘못된 상품이 배송되어 교환하고 싶어요",
        "배송 중 파손된 제품을 받았어요",
        "여러 주문을 한 번에 묶어서 배송받고 싶어요",
        "부재중 반송된 상품의 재배송을 요청하고 싶어요",
    ),
    "technical_issue": (
        "파일 업로드 요청에서 500 오류가 발생해요",
        "API 호출이 429 오류로 계속 거절돼요",
        "Authorization 헤더를 넣어도 401 응답이 와요",
        "앱을 실행하면 빈 화면 뒤에 바로 종료돼요",
        "웹훅이 같은 이벤트를 여러 번 전송해요",
        "대용량 파일 처리 중 연결 시간이 초과돼요",
        "SDK를 업데이트한 뒤 응답 파싱에 실패해요",
        "한국어 입력이 깨진 문자로 저장돼요",
        "검색 결과가 갱신된 데이터를 반영하지 않아요",
        "특정 브라우저에서만 버튼 클릭이 동작하지 않아요",
    ),
}

# 일부 고객이 선택한 카테고리 제목이나 상담원이 붙인 제목을 흉내 낸다. 모든 행에
# 넣으면 분류가 지나치게 쉬워지므로 클래스별 4개 case에만 적용해 오류 분석도 남긴다.
TOPIC_HINTS = {
    "account_access": "계정과 로그인 접근 관련 문의입니다",
    "billing": "결제와 청구 내역 관련 문의입니다",
    "cancellation": "구독 해지와 갱신 중단 관련 요청입니다",
    "delivery": "주문 상품의 배송 관련 문의입니다",
    "technical_issue": "앱과 API의 기술 오류 관련 문의입니다",
}


def build_rows(seed: int = 42) -> list[dict[str, str]]:
    """seed가 같으면 내용과 순서가 완전히 같은 100행을 반환한다."""

    rng = random.Random(seed)
    started_at = datetime(2026, 1, 3, 0, 12, tzinfo=UTC)
    rows: list[dict[str, str]] = []
    ticket_number = 1
    for label_index, (label, scenarios) in enumerate(SCENARIOS.items()):
        prefix = label.split("_")[0][:2].upper()
        for case_index, scenario in enumerate(scenarios, start=1):
            case_id = f"C-{prefix}-{case_index:03d}"
            topic_hint = f"{TOPIC_HINTS[label]}. " if case_index <= 4 else ""
            paraphrases = (
                f"{topic_hint}{scenario}. 확인하고 해결 방법을 알려주세요.",
                f"{topic_hint}{scenario} 빠르게 처리하려면 무엇을 해야 하나요?",
            )
            for variant, text in enumerate(paraphrases):
                created_at = started_at + timedelta(
                    days=label_index * 28 + case_index * 2,
                    hours=variant * 5 + rng.randrange(4),
                    minutes=rng.randrange(60),
                )
                rows.append(
                    {
                        "ticket_id": f"T-{ticket_number:04d}",
                        "case_id": case_id,
                        "created_at": created_at.isoformat(),
                        "channel": rng.choice(("web", "email", "chat", "phone")),
                        "priority": rng.choice(("low", "medium", "medium", "high")),
                        "customer_tier": rng.choice(("free", "pro", "business")),
                        "text": text,
                        "label": label,
                    }
                )
                ticket_number += 1
    rng.shuffle(rows)
    return rows


def write_rows(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def existing_summary(path: Path) -> tuple[int, Counter[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), Counter(row.get("label", "") for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "customer_support_tickets.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="기존 파일을 다시 생성")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        count, labels = existing_summary(args.output)
        print(f"기존 데이터 보존: {args.output} (rows={count}, labels={dict(labels)})")
        print("재생성하려면 --force를 추가하세요.")
        return
    rows = build_rows(args.seed)
    write_rows(rows, args.output)
    print(f"생성 완료: {args.output} (rows={len(rows)}, seed={args.seed})")
    print(f"라벨 분포: {dict(sorted(Counter(row['label'] for row in rows).items()))}")


if __name__ == "__main__":
    main()
