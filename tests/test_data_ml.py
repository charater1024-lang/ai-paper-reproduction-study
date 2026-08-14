from pathlib import Path

import pandas as pd
import pytest

from llm_engineering_lab.data import (
    KNOWN_LABELS,
    dataset_profile,
    load_ticket_data,
    stratified_group_split,
    validate_ticket_frame,
)
from llm_engineering_lab.ml import (
    TicketClassifierTrainer,
    TrainingConfig,
    load_model,
    predict_tickets,
    save_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_support_tickets.csv"


@pytest.fixture(scope="module")
def ticket_frame() -> pd.DataFrame:
    return load_ticket_data(DATA_PATH)


@pytest.fixture(scope="module")
def training_result(ticket_frame: pd.DataFrame):
    return TicketClassifierTrainer(TrainingConfig(seed=17)).train(ticket_frame)


def test_dataset_contract_and_profile(ticket_frame: pd.DataFrame) -> None:
    profile = dataset_profile(ticket_frame)
    assert profile.row_count == 100
    assert profile.case_count == 50
    assert set(profile.label_counts) == KNOWN_LABELS
    assert set(profile.label_counts.values()) == {20}
    assert profile.duplicate_ticket_ids == 0
    assert profile.missing_cells == 0


def test_split_is_reproducible_stratified_and_group_safe(
    ticket_frame: pd.DataFrame,
) -> None:
    first = stratified_group_split(ticket_frame, validation_size=0.2, seed=7)
    second = stratified_group_split(ticket_frame, validation_size=0.2, seed=7)

    first.assert_no_group_leakage()
    assert (
        first.validation["ticket_id"].tolist()
        == second.validation["ticket_id"].tolist()
    )
    assert set(first.train["label"]) == KNOWN_LABELS
    assert set(first.validation["label"]) == KNOWN_LABELS
    assert len(first.train) == 80
    assert len(first.validation) == 20


def test_validation_rejects_case_with_mixed_labels(ticket_frame: pd.DataFrame) -> None:
    case_id = ticket_frame.loc[0, "case_id"]
    broken = ticket_frame.loc[ticket_frame["case_id"].eq(case_id)].copy()
    replacement = next(
        label for label in KNOWN_LABELS if label != broken.iloc[0]["label"]
    )
    broken.loc[broken.index[1], "label"] = replacement
    with pytest.raises(ValueError, match="multiple labels"):
        validate_ticket_frame(broken, expected_labels=None)


def test_baseline_trains_and_reports_class_metrics(training_result) -> None:
    assert training_result.report.macro_f1 >= 0.70
    assert set(training_result.report.per_class) == KNOWN_LABELS
    assert len(training_result.report.confusion_matrix) == len(KNOWN_LABELS)
    training_result.split.assert_no_group_leakage()


def test_text_only_prediction_and_model_round_trip(
    training_result, tmp_path: Path
) -> None:
    texts = ["카드 요금이 두 번 결제됐어요", "파일 업로드에서 500 오류가 납니다"]
    before = predict_tickets(training_result.model, texts)
    path = save_model(training_result.model, tmp_path / "ticket.joblib", {"test": True})
    restored = load_model(path)
    after = predict_tickets(restored, texts)

    assert before["predicted_label"].tolist() == ["billing", "technical_issue"]
    pd.testing.assert_frame_equal(before, after)
