import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "practice"


def test_sensor_and_tabular_datasets_have_learning_edge_cases() -> None:
    sensors = pd.read_csv(DATA / "sensor_timeseries.csv")
    tabular = pd.read_csv(DATA / "tabular_risk.csv")

    assert sensors.shape == (3_444, 8)
    assert sensors["device_id"].nunique() == 12
    assert 0 < sensors["anomaly"].mean() < 0.05
    assert sensors.isna().sum().sum() == 0

    assert tabular.shape == (1_500, 10)
    assert 0 < tabular["risk_label"].mean() < 0.15
    assert tabular["monthly_spend"].isna().any()


def test_image_and_rag_evaluation_datasets() -> None:
    arrays = np.load(DATA / "image_shapes.npz")
    assert arrays["images"].shape == (1_200, 28, 28)
    assert arrays["images"].dtype == np.uint8
    assert np.array_equal(np.bincount(arrays["labels"]), np.array([400, 400, 400]))

    records = [
        json.loads(line)
        for line in (DATA / "rag_queries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 13
    assert any(not record["relevant_ids"] for record in records)
