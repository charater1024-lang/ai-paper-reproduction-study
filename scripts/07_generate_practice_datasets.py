"""Generate small, reproducible datasets for the paired AI engineering labs.

The generated files are intentionally small enough for a laptop while still
containing realistic issues such as class imbalance, groups, drift and noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_sensor_timeseries(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    steps = 288
    for device_number in range(12):
        phase = rng.uniform(0, 2 * np.pi)
        drift = rng.normal(0.0, 0.002)
        device = f"sensor-{device_number:02d}"
        for step in range(steps):
            seasonal = np.sin(2 * np.pi * step / 48 + phase)
            load = np.clip(0.55 + 0.3 * seasonal + rng.normal(0, 0.06), 0, 1)
            anomaly = int((device_number + step) % 173 == 0 or rng.random() < 0.012)
            temperature = (
                35.0
                + 8.0 * load
                + drift * step
                + rng.normal(0, 0.55)
                + anomaly * rng.uniform(5, 9)
            )
            vibration = 0.18 + 0.42 * load + rng.normal(0, 0.035) + anomaly * 0.35
            pressure = 101.3 - 1.8 * load + rng.normal(0, 0.18)
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(minutes=5 * step),
                    "device_id": device,
                    "load": round(float(load), 5),
                    "temperature": round(float(temperature), 5),
                    "vibration": round(float(vibration), 5),
                    "pressure": round(float(pressure), 5),
                    "anomaly": anomaly,
                }
            )

    frame = pd.DataFrame(rows)
    frame["next_temperature"] = frame.groupby("device_id")["temperature"].shift(-1)
    return frame.dropna().reset_index(drop=True)


def make_tabular_risk(rng: np.random.Generator, size: int = 1_500) -> pd.DataFrame:
    industries = np.array(["commerce", "finance", "gaming", "health", "education"])
    regions = np.array(["seoul", "busan", "daejeon", "incheon", "global"])
    tiers = np.array(["free", "starter", "pro", "enterprise"])
    frame = pd.DataFrame(
        {
            "account_id": [f"acct-{index:05d}" for index in range(size)],
            "industry": rng.choice(
                industries, size=size, p=[0.30, 0.16, 0.18, 0.16, 0.20]
            ),
            "region": rng.choice(regions, size=size, p=[0.42, 0.15, 0.10, 0.12, 0.21]),
            "tier": rng.choice(tiers, size=size, p=[0.34, 0.32, 0.25, 0.09]),
            "tenure_months": rng.integers(1, 73, size=size),
            "monthly_spend": np.round(rng.lognormal(3.5, 0.75, size=size), 2),
            "weekly_sessions": rng.poisson(8, size=size),
            "support_tickets_90d": rng.poisson(1.4, size=size),
            "latency_ms": np.round(rng.gamma(4, 38, size=size), 2),
        }
    )
    logit = (
        -3.2
        - 0.025 * frame["tenure_months"]
        - 0.09 * frame["weekly_sessions"]
        + 0.42 * frame["support_tickets_90d"]
        + 0.0035 * frame["latency_ms"]
        + 0.75 * (frame["tier"] == "free")
        + 0.45 * (frame["industry"] == "gaming")
    )
    probability = 1 / (1 + np.exp(-logit))
    frame["risk_label"] = (rng.random(size) < probability).astype(int)
    missing = rng.random(size) < 0.035
    frame.loc[missing, "monthly_spend"] = np.nan
    return frame


def make_shape_images(
    rng: np.random.Generator, samples_per_class: int = 400
) -> dict[str, np.ndarray]:
    image_size = 28
    images: list[np.ndarray] = []
    labels: list[int] = []
    yy, xx = np.mgrid[:image_size, :image_size]
    for label in range(3):
        for _ in range(samples_per_class):
            cx, cy = rng.integers(9, 20, size=2)
            radius = int(rng.integers(4, 8))
            if label == 0:  # circle
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            elif label == 1:  # square
                mask = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
            else:  # cross
                width = int(rng.integers(1, 3))
                mask = (
                    (np.abs(xx - cx) <= width) & (np.abs(yy - cy) <= radius + 2)
                ) | ((np.abs(yy - cy) <= width) & (np.abs(xx - cx) <= radius + 2))
            image = rng.normal(18, 12, size=(image_size, image_size))
            image[mask] += rng.uniform(170, 225)
            images.append(np.clip(image, 0, 255).astype(np.uint8))
            labels.append(label)
    order = rng.permutation(len(images))
    return {
        "images": np.stack(images)[order],
        "labels": np.asarray(labels, dtype=np.int64)[order],
        "class_names": np.asarray(["circle", "square", "cross"]),
    }


def write_rag_queries(path: Path) -> None:
    examples = [
        ("비밀번호 재설정 메일이 도착하지 않아요", ["kb-001"]),
        ("로그인을 여러 번 실패해서 계정이 잠겼습니다", ["kb-002", "kb-001"]),
        ("같은 주문이 카드에 두 번 결제됐어요", ["kb-003"]),
        ("구독을 취소하면 남은 기간도 바로 못 쓰나요?", ["kb-004"]),
        ("환불 승인 후 카드 내역에 언제 반영되나요?", ["kb-005"]),
        ("운송장 조회가 이틀째 그대로예요", ["kb-006"]),
        ("이미 출고된 주문의 배송지를 바꾸고 싶어요", ["kb-007"]),
        ("받은 제품이 깨져 있는데 어떤 사진이 필요한가요?", ["kb-008"]),
        ("API 요청에서 401이 발생합니다", ["kb-009"]),
        ("429 응답을 안전하게 재시도하는 방법", ["kb-010"]),
        ("5xx 장애 문의 전에 무엇을 기록해야 하나요?", ["kb-011"]),
        ("개인정보를 내려받고 계정을 삭제하고 싶어요", ["kb-012"]),
        ("회사 주소가 어디인가요?", []),
    ]
    with path.open("w", encoding="utf-8") as stream:
        for index, (query, relevant_ids) in enumerate(examples, start=1):
            stream.write(
                json.dumps(
                    {
                        "query_id": f"q-{index:03d}",
                        "query": query,
                        "relevant_ids": relevant_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "practice")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sensor_path = output_dir / "sensor_timeseries.csv"
    tabular_path = output_dir / "tabular_risk.csv"
    image_path = output_dir / "image_shapes.npz"
    query_path = output_dir / "rag_queries.jsonl"

    make_sensor_timeseries(rng).to_csv(sensor_path, index=False, encoding="utf-8")
    make_tabular_risk(rng).to_csv(tabular_path, index=False, encoding="utf-8")
    np.savez_compressed(image_path, **make_shape_images(rng))
    write_rag_queries(query_path)

    for path in (sensor_path, tabular_path, image_path, query_path):
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"created {display_path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
