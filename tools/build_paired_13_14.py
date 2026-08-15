"""Build paired exercise/solution notebooks for data, RNN, CNN and tabular labs.

The generated notebooks intentionally share the same markdown cells and cell order.
Only the implementation inside selected code cells differs.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from notebook_api_explanations import annotate_pair

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "notebooks" / "exercises"
SOLUTIONS = ROOT / "notebooks" / "solutions"


def clean(text: str) -> str:
    return dedent(text).strip("\n")


class PairNotebook:
    def __init__(self, title: str) -> None:
        self.title = title
        self.exercise_cells: list = []
        self.solution_cells: list = []

    def md(self, source: str) -> None:
        source = clean(source)
        cell_id = f"cell-{len(self.exercise_cells) + 1:02d}"
        self.exercise_cells.append(nbf.v4.new_markdown_cell(source, id=cell_id))
        self.solution_cells.append(nbf.v4.new_markdown_cell(source, id=cell_id))

    def code(self, solution: str, exercise: str | None = None) -> None:
        solution = clean(solution)
        exercise = clean(exercise if exercise is not None else solution)
        cell_id = f"cell-{len(self.exercise_cells) + 1:02d}"
        self.exercise_cells.append(nbf.v4.new_code_cell(exercise, id=cell_id))
        self.solution_cells.append(nbf.v4.new_code_cell(solution, id=cell_id))

    def write(self, filename: str) -> None:
        metadata = {
            "kernelspec": {
                "display_name": "Python (AI Engineering Lab)",
                "language": "python",
                "name": "ai-engineering-lab",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "paired_notebook": {"exercise": True, "solution": True},
        }
        notebooks = {
            "exercise": nbf.v4.new_notebook(
                cells=self.exercise_cells,
                metadata={**metadata, "role": "exercise"},
            ),
            "solution": nbf.v4.new_notebook(
                cells=self.solution_cells,
                metadata={**metadata, "role": "solution"},
            ),
        }
        annotate_pair(notebooks["exercise"], notebooks["solution"])

        for folder, role in (
            (EXERCISES, "exercise"),
            (SOLUTIONS, "solution"),
        ):
            folder.mkdir(parents=True, exist_ok=True)
            nb = notebooks[role]
            nbf.validate(nb)
            nbf.write(nb, folder / filename)


def common_path_cell() -> str:
    return """
    from pathlib import Path

    def find_project_root(start: Path | None = None) -> Path:
        start = (start or Path.cwd()).resolve()
        for candidate in (start, *start.parents):
            if (candidate / "data" / "practice").exists():
                return candidate
        raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다. Start_JupyterLab.cmd로 실행해 주세요.")

    ROOT = find_project_root()
    DATA = ROOT / "data" / "practice"
    ARTIFACTS = ROOT / "artifacts" / "practice"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("project root:", ROOT)
    """


def build_13() -> None:
    nb = PairNotebook("13_data_modalities_and_pipelines")
    nb.md("""
    # 13. 데이터 모달리티와 재사용 가능한 파이프라인

    같은 모델링 문제라도 **표형 데이터, 시계열, 텍스트, 이미지**는 샘플의 의미와
    올바른 분할 방식이 다릅니다. 이 노트북에서는 외부 다운로드 없이 제공된 소형
    데이터로 스키마 검사 → 분할 → 변환 → Dataset/DataLoader까지 연결합니다.

    연습본은 `TODO`에서 멈추도록 설계했습니다. 오른쪽 정답본의 같은 번호 셀을
    참고하되, 먼저 직접 구현하고 `assert`로 계약을 확인하세요.
    """)
    nb.md("""
    ## 학습 목표

    - 행(row), 시간 창(window), 문장, 이미지가 각각 무엇을 한 샘플로 보는지 설명한다.
    - 무작위 분할이 시계열/그룹 데이터에서 누수를 만드는 이유를 이해한다.
    - `dataclass` 스키마와 상태를 가진 전처리 클래스를 작성한다.
    - 길이가 다른 텍스트를 `collate_fn`으로 padding한다.
    - NumPy/Pandas와 PyTorch 사이의 dtype·shape 경계를 검사한다.
    """)
    nb.code("""
    import random
    from collections import Counter
    from dataclasses import dataclass
    
    import numpy as np
    import pandas as pd
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("torch:", torch.__version__)
    """)
    nb.code(common_path_cell())
    nb.md("""
    ## 1) 네 종류의 데이터를 한 번에 관찰하기

    파일 전체를 곧바로 모델에 넣지 말고, 먼저 행 수·열·dtype·결측·레이블 분포를
    확인합니다. 텍스트 CSV는 원문 인코딩 상태와 무관하게 `label`과 문자열 길이를
    이용할 수 있습니다.
    """)
    nb.code("""
    sensor = pd.read_csv(DATA / "sensor_timeseries.csv", parse_dates=["timestamp"])
    tabular = pd.read_csv(DATA / "tabular_risk.csv")
    tickets = pd.read_csv(ROOT / "data" / "customer_support_tickets.csv")
    shapes = np.load(DATA / "image_shapes.npz")
    
    print("sensor", sensor.shape, sensor.dtypes.to_dict())
    print("tabular", tabular.shape, "risk rate=", round(tabular.risk_label.mean(), 3))
    print("text", tickets.shape, tickets.label.value_counts().to_dict())
    print(
        "image",
        shapes["images"].shape,
        shapes["labels"].shape,
        shapes["class_names"].tolist(),
    )
    """)
    nb.md("""
    ### 모달리티별 샘플 단위

    | 모달리티 | 한 샘플 | 보존할 구조 | 대표적인 누수 |
    |---|---|---|---|
    | 표형 | 계정 한 행 | 열의 의미·범주 | 같은 계정/미래 집계가 양쪽 분할에 존재 |
    | 시계열 | 일정 길이의 시간 창 | 시간 순서·장비 그룹 | 미래 값을 학습 통계에 사용 |
    | 텍스트 | 문장/문서 | 토큰 순서·길이 | 중복 문서 또는 정답 문구 |
    | 이미지 | 채널×높이×너비 | 공간적 이웃 | 원본과 증강본이 서로 다른 분할에 존재 |
    """)
    nb.md("""
    ## 2) `dataclass`로 스키마 계약 만들기

    현업 파이프라인에서는 컬럼명이 조용히 바뀌는 것이 가장 위험한 실패 중 하나입니다.
    설정과 검증 규칙을 클래스로 묶으면 테스트와 재사용이 쉬워집니다.
    """)
    nb.code(
        solution="""
        @dataclass(frozen=True)
        class FrameSchema:
            required: tuple[str, ...]
            target: str
            group: str | None = None
            time: str | None = None
        
            def validate(self, frame: pd.DataFrame) -> None:
                missing = sorted(set(self.required) - set(frame.columns))
                if missing:
                    raise ValueError(f"missing columns: {missing}")
                if frame[self.target].isna().any():
                    raise ValueError(f"target {self.target!r} contains missing values")
                if self.time and not pd.api.types.is_datetime64_any_dtype(frame[self.time]):
                    raise TypeError(f"{self.time!r} must be datetime dtype")
        
        
        SENSOR_SCHEMA = FrameSchema(
            required=(
                "timestamp",
                "device_id",
                "load",
                "temperature",
                "vibration",
                "pressure",
                "anomaly",
            ),
            target="anomaly",
            group="device_id",
            time="timestamp",
        )
        SENSOR_SCHEMA.validate(sensor)
        print(SENSOR_SCHEMA)
        """,
        exercise="""
        @dataclass(frozen=True)
        class FrameSchema:
            required: tuple[str, ...]
            target: str
            group: str | None = None
            time: str | None = None
        
            def validate(self, frame: pd.DataFrame) -> None:
                # TODO 13-1: 필수 컬럼, target 결측, time dtype을 검사하세요.
                raise NotImplementedError("오른쪽 정답본을 보기 전에 스키마 검증을 구현하세요.")
        
        
        SENSOR_SCHEMA = FrameSchema(
            required=(
                "timestamp",
                "device_id",
                "load",
                "temperature",
                "vibration",
                "pressure",
                "anomaly",
            ),
            target="anomaly",
            group="device_id",
            time="timestamp",
        )
        SENSOR_SCHEMA.validate(sensor)
        """,
    )
    nb.code("""
    assert SENSOR_SCHEMA.target in sensor
    assert sensor["device_id"].nunique() >= 2
    assert (
        sensor["timestamp"].is_monotonic_increasing is False
    )  # 장비별 블록이므로 전체 정렬은 아님
    print("schema contract: OK")
    """)
    nb.md("""
    ## 3) 장비별 시간 분할

    각 장비의 앞 70%만 학습, 다음 15%를 검증, 마지막 15%를 테스트로 사용합니다.
    이 방식은 모든 장비를 경험하면서도 미래가 과거의 통계에 섞이지 않게 합니다.
    """)
    nb.code(
        solution="""
        def chronological_group_split(
            frame: pd.DataFrame,
            group_col: str,
            time_col: str,
            train_ratio: float = 0.70,
            valid_ratio: float = 0.15,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            train_parts, valid_parts, test_parts = [], [], []
            for _, group in frame.sort_values(time_col).groupby(group_col, sort=False):
                n = len(group)
                train_end = int(n * train_ratio)
                valid_end = int(n * (train_ratio + valid_ratio))
                train_parts.append(group.iloc[:train_end])
                valid_parts.append(group.iloc[train_end:valid_end])
                test_parts.append(group.iloc[valid_end:])
            return tuple(
                pd.concat(parts, ignore_index=True)
                for parts in (train_parts, valid_parts, test_parts)
            )
        
        
        sensor_train, sensor_valid, sensor_test = chronological_group_split(
            sensor, "device_id", "timestamp"
        )
        print(len(sensor_train), len(sensor_valid), len(sensor_test))
        """,
        exercise="""
        def chronological_group_split(
            frame: pd.DataFrame,
            group_col: str,
            time_col: str,
            train_ratio: float = 0.70,
            valid_ratio: float = 0.15,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            # TODO 13-2: group별 시간 정렬 후 iloc으로 70/15/15를 나누세요.
            raise NotImplementedError("group-aware chronological split을 구현하세요.")
        
        
        sensor_train, sensor_valid, sensor_test = chronological_group_split(
            sensor, "device_id", "timestamp"
        )
        """,
    )
    nb.code("""
    for device in sensor.device_id.unique():
        tr = sensor_train.query("device_id == @device")
        va = sensor_valid.query("device_id == @device")
        te = sensor_test.query("device_id == @device")
        assert tr.timestamp.max() < va.timestamp.min() < te.timestamp.min()
    assert len(sensor_train) + len(sensor_valid) + len(sensor_test) == len(sensor)
    print("temporal leakage check: OK")
    """)
    nb.md("""
    ## 4) 상태를 가진 표형 전처리기

    평균·표준편차와 범주 목록은 **train에만 fit**합니다. 검증/테스트에는 저장된 상태를
    그대로 적용합니다. 이 작은 클래스는 `sklearn Pipeline`의 `fit/transform` 계약과 같습니다.
    """)
    nb.code(
        solution="""
        @dataclass
        class TabularEncoder:
            numeric: list[str]
            categorical: list[str]
            means_: pd.Series | None = None
            stds_: pd.Series | None = None
            categories_: dict[str, list[str]] | None = None
        
            def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
                self.means_ = frame[self.numeric].mean()
                self.stds_ = frame[self.numeric].std().replace(0, 1.0)
                self.categories_ = {
                    col: sorted(frame[col].astype(str).unique().tolist())
                    for col in self.categorical
                }
                return self
        
            def transform(self, frame: pd.DataFrame) -> np.ndarray:
                if self.means_ is None or self.stds_ is None or self.categories_ is None:
                    raise RuntimeError("fit must be called before transform")
                filled = frame[self.numeric].fillna(self.means_)
                numeric = ((filled - self.means_) / self.stds_).to_numpy(np.float32)
                blocks = [numeric]
                for col in self.categorical:
                    values = frame[col].astype(str).to_numpy()
                    blocks.append(
                        np.stack(
                            [(values == category) for category in self.categories_[col]], axis=1
                        )
                    )
                return np.concatenate(blocks, axis=1).astype(np.float32)
        
        
        numeric_cols = [
            "tenure_months",
            "monthly_spend",
            "weekly_sessions",
            "support_tickets_90d",
            "latency_ms",
        ]
        categorical_cols = ["industry", "region", "tier"]
        cut = int(len(tabular) * 0.8)
        table_train, table_test = tabular.iloc[:cut], tabular.iloc[cut:]
        encoder = TabularEncoder(numeric_cols, categorical_cols).fit(table_train)
        X_table_train = encoder.transform(table_train)
        X_table_test = encoder.transform(table_test)
        print(X_table_train.shape, X_table_test.shape, X_table_train.dtype, X_table_test.dtype)
        """,
        exercise="""
        @dataclass
        class TabularEncoder:
            numeric: list[str]
            categorical: list[str]
            means_: pd.Series | None = None
            stds_: pd.Series | None = None
            categories_: dict[str, list[str]] | None = None
        
            def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
                # TODO 13-3a: train의 평균, 표준편차, 범주 목록을 self에 저장하세요.
                raise NotImplementedError
        
            def transform(self, frame: pd.DataFrame) -> np.ndarray:
                # TODO 13-3b: 결측은 train 평균으로 채우고 숫자 표준화 + 범주 one-hot을
                # float32 배열로 합치세요.
                raise NotImplementedError
        
        
        numeric_cols = [
            "tenure_months",
            "monthly_spend",
            "weekly_sessions",
            "support_tickets_90d",
            "latency_ms",
        ]
        categorical_cols = ["industry", "region", "tier"]
        cut = int(len(tabular) * 0.8)
        table_train, table_test = tabular.iloc[:cut], tabular.iloc[cut:]
        encoder = TabularEncoder(numeric_cols, categorical_cols).fit(table_train)
        X_table_train = encoder.transform(table_train)
        X_table_test = encoder.transform(table_test)
        """,
    )
    nb.code("""
    assert X_table_train.dtype == np.float32
    assert X_table_train.shape[1] == X_table_test.shape[1]
    assert np.isfinite(X_table_test).all()
    print("tabular feature contract: OK, feature_dim=", X_table_train.shape[1])
    """)
    nb.md("""
    ## 5) 시계열을 겹치는 window로 변환

    `history=12`이면 5분 간격 센서 1시간이 입력 한 개입니다. window가 장비 경계를
    넘지 않도록 장비별로 생성하고, 입력은 `(samples, time, features)`가 됩니다.
    """)
    nb.code(
        solution="""
        SENSOR_FEATURES = ["load", "temperature", "vibration", "pressure"]
        
        
        def make_windows(frame: pd.DataFrame, history: int = 12, stride: int = 4):
            xs, ys = [], []
            for _, group in frame.sort_values("timestamp").groupby("device_id", sort=False):
                values = group[SENSOR_FEATURES].to_numpy(np.float32)
                labels = group["anomaly"].to_numpy(np.float32)
                for end in range(history, len(group) + 1, stride):
                    xs.append(values[end - history : end])
                    ys.append(labels[end - history : end].max())
            return np.stack(xs), np.asarray(ys, dtype=np.float32)
        
        
        X_seq, y_seq = make_windows(sensor_train)
        print(X_seq.shape, y_seq.shape, "positive windows=", int(y_seq.sum()))
        """,
        exercise="""
        SENSOR_FEATURES = ["load", "temperature", "vibration", "pressure"]
        
        
        def make_windows(frame: pd.DataFrame, history: int = 12, stride: int = 4):
            # TODO 13-4: 장비별로 (history, features) window를 만들고
            # window 안 anomaly의 max를 target으로 반환하세요.
            raise NotImplementedError
        
        
        X_seq, y_seq = make_windows(sensor_train)
        """,
    )
    nb.code("""
    assert X_seq.ndim == 3 and X_seq.shape[1:] == (12, 4)
    assert len(X_seq) == len(y_seq)
    assert set(np.unique(y_seq)).issubset({0.0, 1.0})
    print("sequence window contract: OK")
    """)
    nb.md("""
    ## 6) 텍스트 vocabulary와 동적 padding

    여기서는 인코딩이 안정적인 짧은 운영 문장을 사용합니다. 실제 프로젝트에서는
    정규화/토크나이저 버전과 vocabulary를 artifact로 저장해야 재현할 수 있습니다.
    """)
    nb.code("""
    text_samples = [
        ("payment failed after card renewal", 0),
        ("cancel my subscription before renewal", 1),
        ("login verification code never arrived", 2),
        ("invoice amount is incorrect", 0),
        ("please close the account", 1),
        ("password reset link expired", 2),
    ]
    
    
    def tokenize(text: str) -> list[str]:
        return text.lower().replace("?", "").split()
    
    
    counts = Counter(token for text, _ in text_samples for token in tokenize(text))
    stoi = {
        "<pad>": 0,
        "<unk>": 1,
        **{token: i + 2 for i, token in enumerate(sorted(counts))},
    }
    print("vocab size:", len(stoi), "sample:", tokenize(text_samples[0][0]))
    """)
    nb.code(
        solution="""
        class TextDataset(Dataset):
            def __init__(self, samples, vocabulary):
                self.samples = samples
                self.vocabulary = vocabulary
        
            def __len__(self):
                return len(self.samples)
        
            def __getitem__(self, index):
                text, label = self.samples[index]
                ids = [self.vocabulary.get(token, 1) for token in tokenize(text)]
                return torch.tensor(ids, dtype=torch.long), torch.tensor(
                    label, dtype=torch.long
                )
        
        
        def pad_collate(batch):
            sequences, labels = zip(*batch)
            lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
            padded = pad_sequence(sequences, batch_first=True, padding_value=0)
            return padded, lengths, torch.stack(labels)
        
        
        text_loader = DataLoader(
            TextDataset(text_samples, stoi), batch_size=4, collate_fn=pad_collate
        )
        token_ids, lengths, labels = next(iter(text_loader))
        print(token_ids.shape, lengths.tolist(), labels.tolist())
        """,
        exercise="""
        class TextDataset(Dataset):
            def __init__(self, samples, vocabulary):
                self.samples = samples
                self.vocabulary = vocabulary
        
            def __len__(self):
                # TODO 13-5a: 샘플 수를 반환하세요.
                raise NotImplementedError
        
            def __getitem__(self, index):
                # TODO 13-5b: token id LongTensor와 label LongTensor를 반환하세요.
                raise NotImplementedError
        
        
        def pad_collate(batch):
            # TODO 13-5c: pad_sequence와 원래 lengths를 반환하세요.
            raise NotImplementedError
        
        
        text_loader = DataLoader(
            TextDataset(text_samples, stoi), batch_size=4, collate_fn=pad_collate
        )
        token_ids, lengths, labels = next(iter(text_loader))
        """,
    )
    nb.code("""
    assert token_ids.dtype == torch.long and labels.dtype == torch.long
    assert token_ids.shape[0] == len(lengths) == len(labels)
    assert torch.all((token_ids != 0).sum(dim=1) == lengths)
    print("text batch contract: OK")
    """)
    nb.md("""
    ## 7) 이미지 Dataset과 train 전용 증강

    저장 형식은 `(N, H, W)` uint8입니다. 모델 경계에서는 `(C, H, W)` float32와
    `[0, 1]` 범위로 바꿉니다. 확률적 증강은 train Dataset에만 적용해야 합니다.
    """)
    nb.code(
        solution="""
        class ShapeDataset(Dataset):
            def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
                self.images = images
                self.labels = labels
                self.augment = augment
        
            def __len__(self):
                return len(self.labels)
        
            def __getitem__(self, index):
                image = (
                    torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0) / 255.0
                )
                if self.augment and torch.rand(()) < 0.5:
                    image = torch.flip(image, dims=[2])
                label = torch.tensor(self.labels[index], dtype=torch.long)
                return image, label
        
        
        image_loader = DataLoader(
            ShapeDataset(shapes["images"][:64], shapes["labels"][:64], augment=True),
            batch_size=16,
            shuffle=True,
        )
        image_batch, image_labels = next(iter(image_loader))
        print(
            image_batch.shape,
            image_labels.shape,
            image_batch.min().item(),
            image_batch.max().item(),
        )
        """,
        exercise="""
        class ShapeDataset(Dataset):
            def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
                self.images = images
                self.labels = labels
                self.augment = augment
        
            def __len__(self):
                return len(self.labels)
        
            def __getitem__(self, index):
                # TODO 13-6: (H,W) uint8을 (1,H,W) float32 [0,1]로 바꾸고
                # augment=True일 때 50% 확률 수평 flip을 적용하세요.
                raise NotImplementedError
        
        
        image_loader = DataLoader(
            ShapeDataset(shapes["images"][:64], shapes["labels"][:64], augment=True),
            batch_size=16,
            shuffle=True,
        )
        image_batch, image_labels = next(iter(image_loader))
        """,
    )
    nb.code("""
    assert image_batch.shape == (16, 1, 28, 28)
    assert image_batch.dtype == torch.float32
    assert 0.0 <= image_batch.min() <= image_batch.max() <= 1.0
    print("image batch contract: OK")
    """)
    nb.md("""
    ## 실무 점검표

    1. split을 먼저 하고, 통계 fit은 train에서만 했는가?
    2. 샘플 단위와 그룹 경계가 명확한가?
    3. 모델 직전 dtype과 shape를 `assert`했는가?
    4. padding 값과 실제 길이를 함께 전달하는가?
    5. 전처리 상태·클래스 순서·seed를 artifact로 남기는가?
    """)
    nb.code("""
    modality_shapes = {
        "tabular": tuple(X_table_train.shape),
        "timeseries": tuple(X_seq.shape),
        "text": tuple(token_ids.shape),
        "image": tuple(image_batch.shape),
    }
    for name, shape in modality_shapes.items():
        print(f"{name:10s} -> {shape}")
    assert all(len(shape) >= 2 for shape in modality_shapes.values())
    """)
    nb.md("""
    ## 마무리

    핵심은 파일 형식이 아니라 **샘플의 의미, 분할 경계, fit 상태, 모델 입력 계약**입니다.
    다음 실습에서는 여기서 만든 시계열 원칙을 RNN/GRU/LSTM 분류와 예측에 적용합니다.
    """)
    nb.write("13_data_modalities_and_pipelines.ipynb")


def build_14() -> None:
    nb = PairNotebook("14_rnn_sequence_modeling")
    nb.md("""
    # 14. RNN/GRU/LSTM 시퀀스 모델링

    센서 시간 창을 이용해 **이상 징후 분류**와 **다음 온도 예측**을 수행합니다.
    vanilla RNN, GRU, LSTM의 인터페이스를 같은 클래스로 비교하고, 가변 길이 padding,
    `pack_padded_sequence`, hidden state shape, gradient clipping, checkpoint까지 연습합니다.
    """)
    nb.md("""
    ## 실행 규모

    기본 설정은 CPU에서도 수 분 이내, RTX 3080에서는 더 빠르게 끝나도록 작은 모델과
    적은 epoch를 사용합니다. GPU 메모리보다 데이터 전송 시간이 더 큰 규모이므로
    `device` 코드는 연습하되, CPU가 반드시 열등한 선택은 아닙니다.
    """)
    nb.code("""
    import copy
    import random
    from dataclasses import dataclass
    
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import f1_score, mean_absolute_error
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
    from torch.utils.data import DataLoader, Dataset
    from llm_engineering_lab.acceleration import get_accelerator
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    ACCELERATOR = get_accelerator()
    device = ACCELERATOR.device
    print(ACCELERATOR.summary(), "torch:", torch.__version__)
    """)
    nb.code(common_path_cell())
    nb.md("""
    ## 1) 시간 순서 보존과 train 통계

    장비별 앞 75%를 train, 뒤 25%를 test로 두고, 표준화 통계는 train 구간만으로
    계산합니다. target window 안에 anomaly가 하나라도 있으면 양성으로 정의합니다.
    """)
    nb.code("""
    frame = pd.read_csv(DATA / "sensor_timeseries.csv", parse_dates=["timestamp"])
    FEATURES = ["load", "temperature", "vibration", "pressure"]
    
    train_parts, test_parts = [], []
    for _, group in frame.sort_values("timestamp").groupby("device_id", sort=False):
        cut = int(len(group) * 0.75)
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    train_frame = pd.concat(train_parts, ignore_index=True)
    test_frame = pd.concat(test_parts, ignore_index=True)
    means = train_frame[FEATURES].mean()
    stds = train_frame[FEATURES].std().replace(0, 1.0)
    print(train_frame.shape, test_frame.shape, "raw anomaly rate=", frame.anomaly.mean())
    """)
    nb.code(
        solution="""
        def build_variable_sequences(frame, means, stds, max_history=12, stride=3):
            sequences, labels = [], []
            for _, group in frame.sort_values("timestamp").groupby("device_id", sort=False):
                values = ((group[FEATURES] - means) / stds).to_numpy(np.float32)
                anomaly = group["anomaly"].to_numpy(np.float32)
                for end in range(max_history, len(group) + 1, stride):
                    length = 6 + ((end // stride) % 7)  # 6..12의 가변 길이
                    sequences.append(torch.tensor(values[end - length : end]))
                    labels.append(float(anomaly[end - length : end].max()))
            return sequences, torch.tensor(labels, dtype=torch.float32)
        
        
        train_sequences, train_labels = build_variable_sequences(train_frame, means, stds)
        test_sequences, test_labels = build_variable_sequences(test_frame, means, stds)
        print(
            len(train_sequences),
            len(test_sequences),
            "positive train windows=",
            int(train_labels.sum()),
        )
        """,
        exercise="""
        def build_variable_sequences(frame, means, stds, max_history=12, stride=3):
            # TODO 14-1: train 통계로 표준화하고, 장비 경계를 넘지 않는
            # 길이 6~12의 Tensor 목록과 window anomaly max label을 만드세요.
            raise NotImplementedError
        
        
        train_sequences, train_labels = build_variable_sequences(train_frame, means, stds)
        test_sequences, test_labels = build_variable_sequences(test_frame, means, stds)
        """,
    )
    nb.md("""
    ## 2) padding과 원래 길이

    padding은 배치 텐서를 직사각형으로 만들 뿐 실제 관측값이 아닙니다. 원래 길이를
    보존한 뒤 packing하거나 mask해야 모델이 padding을 신호로 학습하지 않습니다.
    """)
    nb.code(
        solution="""
        class SequenceDataset(Dataset):
            def __init__(self, sequences, labels):
                self.sequences, self.labels = sequences, labels
        
            def __len__(self):
                return len(self.labels)
        
            def __getitem__(self, index):
                return self.sequences[index], self.labels[index]
        
        
        def sequence_collate(batch):
            sequences, labels = zip(*batch)
            lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
            padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
            return padded, lengths, torch.stack(labels)
        
        
        train_loader = DataLoader(
            SequenceDataset(train_sequences, train_labels),
            batch_size=64,
            shuffle=True,
            collate_fn=sequence_collate,
        )
        test_loader = DataLoader(
            SequenceDataset(test_sequences, test_labels),
            batch_size=128,
            shuffle=False,
            collate_fn=sequence_collate,
        )
        batch_x, batch_lengths, batch_y = next(iter(train_loader))
        print(
            batch_x.shape,
            batch_lengths.shape,
            batch_y.shape,
            batch_lengths.min().item(),
            batch_lengths.max().item(),
        )
        """,
        exercise="""
        class SequenceDataset(Dataset):
            def __init__(self, sequences, labels):
                self.sequences, self.labels = sequences, labels
        
            def __len__(self):
                # TODO 14-2a
                raise NotImplementedError
        
            def __getitem__(self, index):
                # TODO 14-2b
                raise NotImplementedError
        
        
        def sequence_collate(batch):
            # TODO 14-2c: batch_first padding, lengths, stacked labels를 반환하세요.
            raise NotImplementedError
        
        
        train_loader = DataLoader(
            SequenceDataset(train_sequences, train_labels),
            batch_size=64,
            shuffle=True,
            collate_fn=sequence_collate,
        )
        test_loader = DataLoader(
            SequenceDataset(test_sequences, test_labels),
            batch_size=128,
            shuffle=False,
            collate_fn=sequence_collate,
        )
        batch_x, batch_lengths, batch_y = next(iter(train_loader))
        """,
    )
    nb.code("""
    assert batch_x.ndim == 3 and batch_x.shape[2] == len(FEATURES)
    assert int(batch_lengths.max()) == batch_x.shape[1]
    assert torch.all((batch_x.abs().sum(dim=2) != 0).sum(dim=1) <= batch_lengths)
    print("padded batch contract: OK")
    """)
    nb.md("""
    ## 3) 하나의 wrapper로 RNN/GRU/LSTM 비교

    세 모듈은 공통적으로 `(output, hidden)`을 반환합니다. LSTM만 hidden이 `(h, c)`
    튜플이며, 다층/양방향 모델의 `h` shape는 `(layers × directions, batch, hidden)`입니다.
    """)
    nb.code(
        solution="""
        class SequenceClassifier(nn.Module):
            def __init__(
                self, cell: str, input_dim: int, hidden_dim: int = 24, num_layers: int = 1
            ):
                super().__init__()
                cells = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
                if cell not in cells:
                    raise ValueError(f"unknown cell: {cell}")
                self.cell_name = cell
                self.recurrent = cells[cell](
                    input_dim, hidden_dim, num_layers=num_layers, batch_first=True
                )
                self.head = nn.Linear(hidden_dim, 1)
        
            def forward(self, x, lengths, return_hidden=False):
                packed = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, hidden = self.recurrent(packed)
                h = hidden[0] if isinstance(hidden, tuple) else hidden
                logits = self.head(h[-1]).squeeze(1)
                return (logits, hidden) if return_hidden else logits
        """,
        exercise="""
        class SequenceClassifier(nn.Module):
            def __init__(
                self, cell: str, input_dim: int, hidden_dim: int = 24, num_layers: int = 1
            ):
                super().__init__()
                # TODO 14-3a: cell 문자열로 nn.RNN/GRU/LSTM을 선택하고 head를 만드세요.
                raise NotImplementedError
        
            def forward(self, x, lengths, return_hidden=False):
                # TODO 14-3b: pack → recurrent → 마지막 layer h → linear 순서로 구현하세요.
                # LSTM의 hidden tuple도 처리해야 합니다.
                raise NotImplementedError
        """,
    )
    nb.code("""
    for cell in ("RNN", "GRU", "LSTM"):
        probe = SequenceClassifier(cell, len(FEATURES), hidden_dim=16)
        logits, hidden = probe(batch_x[:5], batch_lengths[:5], return_hidden=True)
        if isinstance(hidden, tuple):
            hidden_shapes = [tuple(item.shape) for item in hidden]
        else:
            hidden_shapes = tuple(hidden.shape)
        print(cell, "logits", tuple(logits.shape), "hidden", hidden_shapes)
        assert logits.shape == (5,)
    """)
    nb.md("""
    ## 4) gradient clipping을 포함한 학습 루프

    긴 시퀀스의 반복 곱은 gradient exploding을 일으킬 수 있습니다. 역전파 후
    optimizer step 전에 `clip_grad_norm_`을 호출하고, 반환되는 clipping 전 norm을 기록합니다.
    """)
    nb.code(
        solution="""
        @dataclass
        class EpochStats:
            loss: float
            grad_norm: float
        
        
        def train_epoch(model, loader, optimizer, loss_fn, max_norm=1.0):
            model.train()
            total_loss, max_grad = 0.0, 0.0
            for x, lengths, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x, lengths), y)
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
                total_loss += loss.item() * len(y)
                max_grad = max(max_grad, float(grad_norm))
            return EpochStats(total_loss / len(loader.dataset), max_grad)
        
        
        @torch.inference_mode()
        def evaluate_classifier(model, loader):
            model.eval()
            probabilities, labels = [], []
            for x, lengths, y in loader:
                probabilities.append(model(x.to(device), lengths).sigmoid().cpu())
                labels.append(y)
            probabilities = torch.cat(probabilities).numpy()
            labels = torch.cat(labels).numpy().astype(int)
            return (
                f1_score(labels, probabilities >= 0.5, zero_division=0),
                probabilities,
                labels,
            )
        """,
        exercise="""
        @dataclass
        class EpochStats:
            loss: float
            grad_norm: float
        
        
        def train_epoch(model, loader, optimizer, loss_fn, max_norm=1.0):
            # TODO 14-4a: forward/backward → clip_grad_norm_ → optimizer.step을 구현하세요.
            raise NotImplementedError
        
        
        @torch.inference_mode()
        def evaluate_classifier(model, loader):
            # TODO 14-4b: sigmoid 확률, F1, labels를 반환하세요.
            raise NotImplementedError
        """,
    )
    nb.code("""
    positive = float(train_labels.sum())
    negative = float(len(train_labels) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], device=device)
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("pos_weight:", round(pos_weight.item(), 2))
    """)
    nb.code(
        solution="""
        comparison = {}
        trained_models = {}
        for cell in ("RNN", "GRU", "LSTM"):
            torch.manual_seed(SEED)
            model = SequenceClassifier(cell, len(FEATURES)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            history = []
            for epoch in range(3):
                stats = train_epoch(model, train_loader, optimizer, classification_loss)
                history.append(stats.loss)
            f1, probabilities, labels = evaluate_classifier(model, test_loader)
            comparison[cell] = {"final_loss": history[-1], "test_f1": f1}
            trained_models[cell] = model
            print(cell, comparison[cell])
        """,
        exercise="""
        comparison = {}
        trained_models = {}
        # TODO 14-5: RNN/GRU/LSTM 각각 3 epoch 학습하고 loss와 test F1을 저장하세요.
        # 힌트: model → Adam → train_epoch → evaluate_classifier 순서입니다.
        raise NotImplementedError
        """,
    )
    nb.code("""
    result_table = pd.DataFrame(comparison).T.sort_values("test_f1", ascending=False)
    print(result_table)
    assert set(result_table.index) == {"RNN", "GRU", "LSTM"}
    assert np.isfinite(result_table.to_numpy()).all()
    """)
    nb.md("""
    ### 비교를 해석할 때 주의할 점

    작은 데이터에서 한 번의 seed로 얻은 순위는 모델의 보편적 우열이 아닙니다.
    RNN은 단순하지만 장기 의존성에 취약하고, GRU는 gate가 적어 가벼우며, LSTM은
    cell state를 별도로 유지합니다. 반복 실험의 평균·편차와 latency를 함께 비교하세요.
    """)
    nb.md("""
    ## 5) padding 값 불변성 확인

    packing이 올바르면 실제 길이 뒤의 padding 값을 크게 바꿔도 logits는 변하지 않습니다.
    이 테스트는 mask/length 누락 버그를 매우 잘 찾아냅니다.
    """)
    nb.code("""
    best_name = result_table.index[0]
    best_model = trained_models[best_name].eval()
    x = batch_x[:8].to(device)
    lengths = batch_lengths[:8]
    changed_padding = x.clone()
    for row, length in enumerate(lengths.tolist()):
        changed_padding[row, length:] = 999.0
    with torch.inference_mode():
        original_logits = best_model(x, lengths)
        changed_logits = best_model(changed_padding, lengths)
    max_difference = (original_logits - changed_logits).abs().max().item()
    print("max padding-induced difference:", max_difference)
    assert max_difference < 1e-5
    """)
    nb.md("""
    ## 6) checkpoint 저장과 복원

    최소 checkpoint에는 모델 종류/차원, `state_dict`, 전처리 통계, feature 순서가 있어야
    합니다. optimizer 상태와 epoch까지 저장하면 중단된 학습도 이어갈 수 있습니다.
    """)
    nb.code(
        solution="""
        checkpoint_path = ARTIFACTS / "rnn_sequence_classifier.pt"
        checkpoint = {
            "cell": best_name,
            "input_dim": len(FEATURES),
            "hidden_dim": 24,
            "state_dict": best_model.state_dict(),
            "features": FEATURES,
            "means": means.to_dict(),
            "stds": stds.to_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
        restored = SequenceClassifier(
            loaded["cell"], loaded["input_dim"], loaded["hidden_dim"]
        ).to(device)
        restored.load_state_dict(loaded["state_dict"])
        restored.eval()
        with torch.inference_mode():
            restored_logits = restored(x, lengths)
        assert torch.allclose(original_logits, restored_logits)
        print("checkpoint restored:", checkpoint_path)
        """,
        exercise="""
        checkpoint_path = ARTIFACTS / "rnn_sequence_classifier.pt"
        # TODO 14-6: model 설정, state_dict, features, train 통계를 저장하고
        # 새 SequenceClassifier에 복원한 뒤 logits가 같은지 확인하세요.
        raise NotImplementedError
        """,
    )
    nb.md("""
    ## 7) 다음 온도 forecasting

    분류와 달리 연속값을 예측하므로 MSE로 학습하고 MAE로 평가합니다. 각 window 직후의
    `next_temperature`를 target으로 하며, target 역시 train 평균/표준편차로 정규화합니다.
    """)
    nb.code(
        solution="""
        def make_forecast_samples(frame, feature_means, feature_stds, history=12, stride=3):
            xs, ys = [], []
            target_mean = float(train_frame["next_temperature"].mean())
            target_std = float(train_frame["next_temperature"].std())
            for _, group in frame.sort_values("timestamp").groupby("device_id", sort=False):
                values = ((group[FEATURES] - feature_means) / feature_stds).to_numpy(np.float32)
                targets = group["next_temperature"].to_numpy(np.float32)
                for end in range(history, len(group), stride):
                    xs.append(torch.tensor(values[end - history : end]))
                    ys.append((targets[end - 1] - target_mean) / target_std)
            return (
                torch.stack(xs),
                torch.tensor(ys, dtype=torch.float32),
                target_mean,
                target_std,
            )
        
        
        forecast_train_x, forecast_train_y, target_mean, target_std = make_forecast_samples(
            train_frame, means, stds
        )
        forecast_test_x, forecast_test_y, _, _ = make_forecast_samples(test_frame, means, stds)
        print(forecast_train_x.shape, forecast_train_y.shape)
        """,
        exercise="""
        def make_forecast_samples(frame, feature_means, feature_stds, history=12, stride=3):
            # TODO 14-7: 고정 길이 입력 window와 정규화한 next_temperature target을 만드세요.
            # target 통계도 train_frame에서만 계산해야 합니다.
            raise NotImplementedError
        
        
        forecast_train_x, forecast_train_y, target_mean, target_std = make_forecast_samples(
            train_frame, means, stds
        )
        forecast_test_x, forecast_test_y, _, _ = make_forecast_samples(test_frame, means, stds)
        """,
    )
    nb.code(
        solution="""
        class GRUForecaster(nn.Module):
            def __init__(self, input_dim, hidden_dim=24):
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
                self.head = nn.Linear(hidden_dim, 1)
        
            def forward(self, x):
                _, hidden = self.gru(x)
                return self.head(hidden[-1]).squeeze(1)
        
        
        forecast_model = GRUForecaster(len(FEATURES)).to(device)
        forecast_optimizer = torch.optim.Adam(forecast_model.parameters(), lr=3e-3)
        forecast_loader = DataLoader(
            torch.utils.data.TensorDataset(forecast_train_x, forecast_train_y),
            batch_size=64,
            shuffle=True,
        )
        for epoch in range(4):
            forecast_model.train()
            total = 0.0
            for bx, by in forecast_loader:
                bx, by = bx.to(device), by.to(device)
                forecast_optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(forecast_model(bx), by)
                loss.backward()
                clip_grad_norm_(forecast_model.parameters(), 1.0)
                forecast_optimizer.step()
                total += loss.item() * len(by)
            print(
                f"forecast epoch {epoch + 1}: loss={total / len(forecast_loader.dataset):.4f}"
            )
        """,
        exercise="""
        class GRUForecaster(nn.Module):
            def __init__(self, input_dim, hidden_dim=24):
                super().__init__()
                # TODO 14-8a: GRU와 회귀 head를 만드세요.
                raise NotImplementedError
        
            def forward(self, x):
                # TODO 14-8b: 마지막 hidden으로 연속값 하나를 예측하세요.
                raise NotImplementedError
        
        
        # TODO 14-8c: MSE, Adam, gradient clipping으로 4 epoch 학습하세요.
        raise NotImplementedError
        """,
    )
    nb.code("""
    forecast_model.eval()
    with torch.inference_mode():
        normalized_prediction = forecast_model(forecast_test_x.to(device)).cpu().numpy()
    predicted_temperature = normalized_prediction * target_std + target_mean
    actual_temperature = forecast_test_y.numpy() * target_std + target_mean
    forecast_mae = mean_absolute_error(actual_temperature, predicted_temperature)
    print("test MAE (°C):", round(forecast_mae, 3))
    for actual, predicted in list(zip(actual_temperature, predicted_temperature))[:5]:
        print(f"actual={actual:6.2f}  predicted={predicted:6.2f}")
    assert np.isfinite(forecast_mae)
    """)
    nb.md("""
    ## hidden state shape 기억법

    - 입력(`batch_first=True`): `(batch, time, input_dim)`
    - output: `(batch, time, directions × hidden_dim)`
    - h: `(layers × directions, batch, hidden_dim)`
    - LSTM: `(h, c)` 두 텐서를 반환
    - packed 입력에서도 hidden은 각 샘플의 **실제 마지막 timestep**을 반영
    """)
    nb.code("""
    summary = {
        "classification_models": list(comparison),
        "best_cell": best_name,
        "padding_invariance": max_difference,
        "forecast_mae": float(forecast_mae),
        "checkpoint": str(checkpoint_path),
    }
    print(summary)
    assert checkpoint_path.exists()
    assert summary["padding_invariance"] < 1e-5
    """)
    nb.md("""
    ## 확장 과제

    1. `num_layers=2`, dropout, bidirectional을 추가하고 hidden shape를 기록하세요.
    2. 분류 threshold를 validation F1로 선택하고 test에는 한 번만 적용하세요.
    3. forecasting을 한 시점이 아니라 다음 6개 시점 출력으로 바꾸세요.
    4. gradient norm을 epoch별로 기록하여 clipping 전후 안정성을 비교하세요.
    """)
    nb.md("""
    ## 마무리

    시퀀스 모델의 핵심은 셀 이름보다 **시간 분할, 길이 전달, hidden 선택, gradient 안정성,
    재현 가능한 checkpoint**입니다. Transformer에서도 padding mask와 시간 누수 원칙은 그대로 이어집니다.
    """)
    nb.write("14_rnn_sequence_modeling.ipynb")


def build_19() -> None:
    nb = PairNotebook("19_cnn_image_classification")
    nb.md("""
    # 19. CNN 이미지 분류: Dataset부터 confusion matrix까지

    외부 다운로드 없이 생성된 28×28 도형 이미지를 분류합니다. 직접 작성한 Dataset,
    train 전용 augmentation, 소형 CNN, 학습/평가 루프, confusion matrix, checkpoint를
    하나의 재현 가능한 파이프라인으로 연결합니다.
    """)
    nb.md("""
    ## 학습 목표

    - `(N,H,W)` uint8을 `(N,C,H,W)` float32로 변환한다.
    - split 이후 train에만 무작위 augmentation을 적용한다.
    - convolution/pooling 뒤 tensor shape를 추론한다.
    - accuracy뿐 아니라 class별 confusion matrix를 해석한다.
    - 모델 설정과 class 이름을 checkpoint에 함께 저장한다.
    """)
    nb.code("""
    import random
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import confusion_matrix
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from llm_engineering_lab.acceleration import get_accelerator
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ACCELERATOR = get_accelerator()
    DEVICE = ACCELERATOR.device
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    SCALER = ACCELERATOR.grad_scaler()
    print(ACCELERATOR.summary())
    """)
    nb.code(common_path_cell())
    nb.md("""
    ## 1) 데이터와 stratified split

    원본/증강본이 양쪽 분할에 섞이지 않도록 **원본 index를 먼저 분할**합니다.
    클래스마다 70/15/15 비율을 사용합니다.
    """)
    nb.code("""
    raw = np.load(DATA / "image_shapes.npz")
    images, labels, class_names = raw["images"], raw["labels"], raw["class_names"].tolist()
    print(images.shape, images.dtype, labels.shape, class_names)
    print("class counts:", np.bincount(labels).tolist())
    """)
    nb.code(
        solution="""
        def stratified_indices(labels, seed=42):
            rng = np.random.default_rng(seed)
            train, valid, test = [], [], []
            for label in np.unique(labels):
                index = np.flatnonzero(labels == label)
                rng.shuffle(index)
                a, b = int(0.70 * len(index)), int(0.85 * len(index))
                train.extend(index[:a])
                valid.extend(index[a:b])
                test.extend(index[b:])
            return tuple(np.asarray(part, dtype=np.int64) for part in (train, valid, test))
        
        
        train_idx, valid_idx, test_idx = stratified_indices(labels)
        print(len(train_idx), len(valid_idx), len(test_idx))
        """,
        exercise="""
        def stratified_indices(labels, seed=42):
            # TODO 19-1: 클래스별 index를 shuffle한 뒤 70/15/15로 나누어 합치세요.
            raise NotImplementedError
        
        
        train_idx, valid_idx, test_idx = stratified_indices(labels)
        """,
    )
    nb.code("""
    assert not (
        set(train_idx) & set(valid_idx)
        | set(train_idx) & set(test_idx)
        | set(valid_idx) & set(test_idx)
    )
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(labels)
    for part in (train_idx, valid_idx, test_idx):
        assert len(np.unique(labels[part])) == len(class_names)
    print("split contract: OK")
    """)
    nb.md("""
    ## 2) custom Dataset과 augmentation

    수평/수직 flip, 약한 Gaussian noise를 직접 구현합니다. 이 합성 도형에서는 방향이
    class 의미를 바꾸지 않지만, 숫자 6/9처럼 방향이 label인 문제에는 이런 증강이 위험합니다.
    """)
    nb.code(
        solution="""
        class ShapesDataset(Dataset):
            def __init__(self, images, labels, indices, augment=False):
                self.images, self.labels = images, labels
                self.indices = np.asarray(indices)
                self.augment = augment
        
            def __len__(self):
                return len(self.indices)
        
            def __getitem__(self, item):
                index = int(self.indices[item])
                image = (
                    torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0) / 255.0
                )
                if self.augment:
                    if torch.rand(()) < 0.5:
                        image = torch.flip(image, [2])
                    if torch.rand(()) < 0.5:
                        image = torch.flip(image, [1])
                    image = (image + torch.randn_like(image) * 0.025).clamp(0, 1)
                return image, torch.tensor(self.labels[index], dtype=torch.long)
        """,
        exercise="""
        class ShapesDataset(Dataset):
            def __init__(self, images, labels, indices, augment=False):
                self.images, self.labels = images, labels
                self.indices = np.asarray(indices)
                self.augment = augment
        
            def __len__(self):
                # TODO 19-2a
                raise NotImplementedError
        
            def __getitem__(self, item):
                # TODO 19-2b: index lookup, float 변환, C축 추가, train 증강을 구현하세요.
                raise NotImplementedError
        """,
    )
    nb.code("""
    train_loader = DataLoader(
        ShapesDataset(images, labels, train_idx, True),
        batch_size=64,
        shuffle=True,
        pin_memory=ACCELERATOR.pin_memory,
    )
    valid_loader = DataLoader(
        ShapesDataset(images, labels, valid_idx),
        batch_size=128,
        pin_memory=ACCELERATOR.pin_memory,
    )
    test_loader = DataLoader(
        ShapesDataset(images, labels, test_idx),
        batch_size=128,
        pin_memory=ACCELERATOR.pin_memory,
    )
    bx, by = next(iter(train_loader))
    print(bx.shape, bx.dtype, by.shape, bx.min().item(), bx.max().item())
    assert bx.shape[1:] == (1, 28, 28) and bx.dtype == torch.float32
    """)
    nb.md("""
    ## 3) 작은 CNN

    3×3 convolution은 지역 패턴을 찾고, pooling은 해상도를 줄이며 수용 영역을 넓힙니다.
    `AdaptiveAvgPool2d(1)`을 사용하면 flatten 차원을 손으로 계산하는 오류가 줄어듭니다.
    """)
    nb.code(
        solution="""
        class TinyCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 24, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(24, num_classes)
        
            def forward(self, x):
                features = self.features(x).flatten(1)
                return self.classifier(features)
        
        
        model = ACCELERATOR.move(TinyCNN(len(class_names)))
        probe = model(ACCELERATOR.move(bx[:4]))
        print(model, "\\nlogits:", probe.shape)
        """,
        exercise="""
        class TinyCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # TODO 19-3a: Conv-ReLU-Pool 블록 2개, Conv, AdaptiveAvgPool을 만드세요.
                raise NotImplementedError
        
            def forward(self, x):
                # TODO 19-3b: features → flatten(1) → classifier
                raise NotImplementedError
        
        
        model = ACCELERATOR.move(TinyCNN(len(class_names)))
        probe = model(ACCELERATOR.move(bx[:4]))
        """,
    )
    nb.code("""
    assert probe.shape == (4, len(class_names))
    parameter_count = sum(p.numel() for p in model.parameters())
    print("parameters:", parameter_count)
    assert parameter_count < 20_000
    """)
    nb.md("""
    ## 4) 학습과 validation checkpoint 선택

    validation accuracy가 개선될 때만 state를 메모리에 보관합니다. test는 모델 선택에
    사용하지 않고 마지막에 한 번 평가합니다.
    """)
    nb.code(
        solution="""
        def train_cnn_epoch(model, loader, optimizer):
            model.train()
            total_loss = 0.0
            for x, y in loader:
                x, y = ACCELERATOR.move(x, y)
                optimizer.zero_grad(set_to_none=True)
                with ACCELERATOR.autocast():
                    loss = nn.functional.cross_entropy(model(x), y)
                SCALER.scale(loss).backward()
                SCALER.step(optimizer)
                SCALER.update()
                total_loss += loss.item() * len(y)
            return total_loss / len(loader.dataset)
        
        
        @torch.inference_mode()
        def predict_cnn(model, loader):
            model.eval()
            predictions, targets = [], []
            for x, y in loader:
                predictions.append(model(ACCELERATOR.move(x)).argmax(1).cpu())
                targets.append(y)
            return torch.cat(predictions).numpy(), torch.cat(targets).numpy()
        """,
        exercise="""
        def train_cnn_epoch(model, loader, optimizer):
            # TODO 19-4a: cross entropy 학습 한 epoch을 구현하세요.
            raise NotImplementedError
        
        
        @torch.inference_mode()
        def predict_cnn(model, loader):
            # TODO 19-4b: argmax 예측과 target NumPy 배열을 반환하세요.
            raise NotImplementedError
        """,
    )
    nb.code(
        solution="""
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
        best_accuracy, best_state = -1.0, None
        for epoch in range(4):
            loss = train_cnn_epoch(model, train_loader, optimizer)
            valid_pred, valid_true = predict_cnn(model, valid_loader)
            accuracy = float((valid_pred == valid_true).mean())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(f"epoch={epoch + 1} loss={loss:.4f} valid_accuracy={accuracy:.3f}")
        model.load_state_dict(best_state)
        """,
        exercise="""
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
        # TODO 19-5: 4 epoch 학습하고 최고 validation accuracy의 state_dict를 복원하세요.
        raise NotImplementedError
        """,
    )
    nb.md("""
    ## 5) confusion matrix

    행은 실제 class, 열은 예측 class입니다. 전체 accuracy가 같아도 특정 class 쌍이
    반복적으로 혼동되는지 확인하면 데이터/모델 개선 방향을 찾을 수 있습니다.
    """)
    nb.code("""
    test_pred, test_true = predict_cnn(model, test_loader)
    matrix = confusion_matrix(test_true, test_pred, labels=np.arange(len(class_names)))
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"true_{x}" for x in class_names],
        columns=[f"pred_{x}" for x in class_names],
    )
    test_accuracy = float((test_pred == test_true).mean())
    print(matrix_frame)
    print("test accuracy:", round(test_accuracy, 3))
    assert matrix.sum() == len(test_idx) and np.isfinite(test_accuracy)
    """)
    nb.code("""
    per_class_recall = matrix.diagonal() / matrix.sum(axis=1).clip(min=1)
    for name, recall in zip(class_names, per_class_recall):
        print(f"{name:>8s}: recall={recall:.3f}")
    weakest = class_names[int(per_class_recall.argmin())]
    print("가장 어려운 class:", weakest)
    """)
    nb.md("""
    ## 6) checkpoint와 단일 이미지 추론

    배포 시 class index 순서를 잃으면 모델이 맞는 index를 내도 잘못된 이름으로 표시됩니다.
    따라서 `class_names`와 입력 정규화 정보를 모델과 함께 저장합니다.
    """)
    nb.code(
        solution="""
        checkpoint_path = ARTIFACTS / "tiny_shapes_cnn.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "input_shape": [1, 28, 28],
                "scale": 255.0,
            },
            checkpoint_path,
        )
        payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        restored = ACCELERATOR.move(TinyCNN(len(payload["class_names"])))
        restored.load_state_dict(payload["state_dict"])
        restored.eval()
        sample_image, sample_label = test_loader.dataset[0]
        with torch.inference_mode():
            sample_index = (
                restored(ACCELERATOR.move(sample_image.unsqueeze(0))).argmax(1).item()
            )
        print(
            "actual:", class_names[sample_label.item()], "predicted:", class_names[sample_index]
        )
        """,
        exercise="""
        checkpoint_path = ARTIFACTS / "tiny_shapes_cnn.pt"
        # TODO 19-6: state_dict, class_names, input shape/scale을 저장하고 새 모델로 복원하세요.
        raise NotImplementedError
        """,
    )
    nb.code("""
    assert checkpoint_path.exists()
    assert payload["class_names"] == class_names
    with torch.inference_mode():
        sample_batch = ACCELERATOR.move(sample_image.unsqueeze(0))
        before = model(sample_batch)
        after = restored(sample_batch)
    assert torch.allclose(before, after)
    print("checkpoint contract: OK")
    """)
    nb.md("""
    ## 오류 분석 과제

    test에서 틀린 index 5개를 찾아 원본 픽셀 통계(평균/최댓값)와 실제·예측 class를
    출력하세요. 그 뒤 noise 크기, 채널 수, epoch 중 하나만 바꾸어 confusion matrix가
    어떻게 달라지는지 비교하세요.
    """)
    nb.code("""
    mistakes = np.flatnonzero(test_pred != test_true)[:5]
    for position in mistakes:
        original_index = test_idx[position]
        print(
            {
                "dataset_index": int(original_index),
                "pixel_mean": round(float(images[original_index].mean()), 2),
                "actual": class_names[int(test_true[position])],
                "predicted": class_names[int(test_pred[position])],
            }
        )
    print("mistake count:", int((test_pred != test_true).sum()))
    """)
    nb.md("""
    ## 마무리

    이미지 파이프라인의 재현성은 모델만이 아니라 split index, train-only augmentation,
    dtype/range, class 순서까지 포함합니다. 실제 대형 이미지에서는 파일을 지연 로딩하고
    DataLoader worker/pinned memory를 측정해 조정하세요.
    """)
    nb.write("19_cnn_image_classification.ipynb")


def build_20() -> None:
    nb = PairNotebook("20_tabular_mlp_and_anomaly_detection")
    nb.md("""
    # 20. 표형 MLP와 비지도 센서 이상 탐지

    희소한 계정 위험 레이블을 MLP로 분류하고 threshold를 validation에서 선택합니다.
    이어서 센서 정상 구간만으로 autoencoder를 학습해 reconstruction error 기반 이상 탐지를
    구성합니다. 지도/비지도 접근의 데이터 요구 사항과 평가 차이를 비교합니다.
    """)
    nb.md("""
    ## 학습 목표

    - train 통계만 사용하는 숫자 표준화와 범주 one-hot 파이프라인을 만든다.
    - `pos_weight`로 class imbalance를 손실에 반영한다.
    - 0.5 고정 대신 validation F1로 threshold를 고른다.
    - 정상 데이터만으로 autoencoder를 fit하고 validation error로 임계값을 정한다.
    - accuracy가 희소 이벤트에서 왜 위험한 지표인지 설명한다.
    """)
    nb.code("""
    import random
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from llm_engineering_lab.acceleration import get_accelerator
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    ACCELERATOR = get_accelerator()
    device = ACCELERATOR.device
    print(ACCELERATOR.summary())
    """)
    nb.code(common_path_cell())
    nb.md("""
    ## 1) 희소한 계정 risk 데이터

    accuracy만 보면 모든 샘플을 정상으로 예측하는 모델도 높아 보일 수 있습니다.
    positive 비율과 naive baseline을 먼저 계산한 뒤 precision/recall/F1을 사용합니다.
    """)
    nb.code("""
    frame = pd.read_csv(DATA / "tabular_risk.csv")
    target = "risk_label"
    identifier = "account_id"
    print(frame.shape, frame.dtypes.to_dict())
    positive_rate = frame[target].mean()
    print(
        "positive rate:",
        round(positive_rate, 4),
        "all-negative accuracy:",
        round(1 - positive_rate, 4),
    )
    """)
    nb.code(
        solution="""
        train_frame, temp_frame = train_test_split(
            frame, test_size=0.30, random_state=SEED, stratify=frame[target]
        )
        valid_frame, test_frame = train_test_split(
            temp_frame, test_size=0.50, random_state=SEED, stratify=temp_frame[target]
        )
        numeric = [
            "tenure_months",
            "monthly_spend",
            "weekly_sessions",
            "support_tickets_90d",
            "latency_ms",
        ]
        categorical = ["industry", "region", "tier"]
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ]
        )
        X_train = preprocessor.fit_transform(train_frame).astype(np.float32)
        X_valid = preprocessor.transform(valid_frame).astype(np.float32)
        X_test = preprocessor.transform(test_frame).astype(np.float32)
        y_train = train_frame[target].to_numpy(np.float32)
        y_valid = valid_frame[target].to_numpy(np.int64)
        y_test = test_frame[target].to_numpy(np.int64)
        print(X_train.shape, X_valid.shape, X_test.shape)
        """,
        exercise="""
        # TODO 20-1: stratify를 사용해 70/15/15로 나누고,
        # train에만 fit하는 median imputer + StandardScaler + OneHotEncoder
        # ColumnTransformer를 만드세요.
        # 결과 배열은 float32여야 합니다.
        raise NotImplementedError
        """,
    )
    nb.code("""
    assert X_train.dtype == X_valid.dtype == X_test.dtype == np.float32
    assert X_train.shape[1] == X_valid.shape[1] == X_test.shape[1]
    assert (
        np.isfinite(X_train).all()
        and np.isfinite(X_valid).all()
        and np.isfinite(X_test).all()
    )
    assert abs(y_train.mean() - y_test.mean()) < 0.03
    print("preprocessing contract: OK")
    """)
    nb.md("""
    ## 2) class imbalance를 고려한 MLP

    `BCEWithLogitsLoss(pos_weight=negative/positive)`는 양성 오류를 더 크게 벌점화합니다.
    확률을 먼저 sigmoid한 뒤 BCE를 적용하지 말고, 수치적으로 안정적인 logits 손실을 사용합니다.
    """)
    nb.code(
        solution="""
        class RiskMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim=32):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
        
            def forward(self, x):
                return self.network(x).squeeze(1)
        
        
        model = RiskMLP(X_train.shape[1]).to(device)
        positives = float(y_train.sum())
        pos_weight = torch.tensor([(len(y_train) - positives) / positives], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        print("pos_weight:", pos_weight.item())
        """,
        exercise="""
        class RiskMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim=32):
                super().__init__()
                # TODO 20-2a: Linear/ReLU/Dropout을 포함한 binary logits MLP를 만드세요.
                raise NotImplementedError
        
            def forward(self, x):
                raise NotImplementedError
        
        
        # TODO 20-2b: negative/positive pos_weight와 BCEWithLogitsLoss, AdamW를 만드세요.
        raise NotImplementedError
        """,
    )
    nb.code("""
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
    )
    print(
        "batches:",
        len(train_loader),
        "parameters:",
        sum(p.numel() for p in model.parameters()),
    )
    """)
    nb.code(
        solution="""
        for epoch in range(10):
            model.train()
            total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
            if epoch in {0, 4, 9}:
                print(f"epoch={epoch + 1:02d} loss={total / len(train_loader.dataset):.4f}")
        
        
        @torch.inference_mode()
        def predict_probability(model, features):
            model.eval()
            x = torch.from_numpy(features).to(device)
            return model(x).sigmoid().cpu().numpy()
        
        
        valid_probability = predict_probability(model, X_valid)
        """,
        exercise="""
        # TODO 20-3a: MLP를 10 epoch 학습하세요.
        raise NotImplementedError
        
        
        @torch.inference_mode()
        def predict_probability(model, features):
            # TODO 20-3b: sigmoid 확률 NumPy 배열을 반환하세요.
            raise NotImplementedError
        
        
        valid_probability = predict_probability(model, X_valid)
        """,
    )
    nb.md("""
    ## 3) threshold는 validation에서 선택

    손실의 class weight와 의사결정 threshold는 별개입니다. 운영 비용에 따라 precision과
    recall의 선호가 달라집니다. 여기서는 단순화를 위해 validation F1 최대값을 선택합니다.
    """)
    nb.code(
        solution="""
        thresholds = np.linspace(0.10, 0.90, 81)
        valid_f1 = np.array(
            [
                f1_score(y_valid, valid_probability >= threshold, zero_division=0)
                for threshold in thresholds
            ]
        )
        best_threshold = float(thresholds[valid_f1.argmax()])
        print("best threshold:", best_threshold, "validation F1:", valid_f1.max())
        """,
        exercise="""
        # TODO 20-4: 0.10~0.90 threshold를 탐색해 validation F1 최대 threshold를 고르세요.
        raise NotImplementedError
        """,
    )
    nb.code("""
    test_probability = predict_probability(model, X_test)
    test_prediction = test_probability >= best_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_prediction, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, test_probability)
    supervised_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }
    print(supervised_metrics)
    assert all(np.isfinite(value) for value in supervised_metrics.values())
    """)
    nb.md("""
    ## 4) 정상 센서만 학습하는 autoencoder

    레이블이 부족할 때는 정상 패턴을 압축·복원하도록 학습하고 복원 오차가 큰 샘플을
    이상으로 간주할 수 있습니다. train에는 시간상 앞부분의 정상 행만 사용합니다.
    """)
    nb.code("""
    sensor = pd.read_csv(DATA / "sensor_timeseries.csv", parse_dates=["timestamp"])
    sensor_features = ["load", "temperature", "vibration", "pressure"]
    ordered = sensor.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    train_parts, valid_parts, test_parts = [], [], []
    for _, group in ordered.groupby("device_id", sort=False):
        a, b = int(len(group) * 0.60), int(len(group) * 0.80)
        train_parts.append(group.iloc[:a])
        valid_parts.append(group.iloc[a:b])
        test_parts.append(group.iloc[b:])
    ae_train = pd.concat(train_parts)
    ae_valid = pd.concat(valid_parts)
    ae_test = pd.concat(test_parts)
    ae_means = ae_train[sensor_features].mean()
    ae_stds = ae_train[sensor_features].std().replace(0, 1)
    normal_train = ae_train.query("anomaly == 0")
    print(
        len(normal_train),
        len(ae_valid),
        len(ae_test),
        "test anomalies:",
        int(ae_test.anomaly.sum()),
    )
    """)
    nb.code(
        solution="""
        def sensor_matrix(frame):
            return ((frame[sensor_features] - ae_means) / ae_stds).to_numpy(np.float32)
        
        
        ae_train_x = sensor_matrix(normal_train)
        ae_valid_x = sensor_matrix(ae_valid)
        ae_test_x = sensor_matrix(ae_test)
        
        
        class AutoEncoder(nn.Module):
            def __init__(self, input_dim, latent_dim=2):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, input_dim)
                )
        
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        
        autoencoder = AutoEncoder(len(sensor_features)).to(device)
        ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=3e-3)
        ae_loader = DataLoader(
            TensorDataset(torch.from_numpy(ae_train_x)), batch_size=128, shuffle=True
        )
        """,
        exercise="""
        def sensor_matrix(frame):
            # TODO 20-5a: ae_train 통계로 sensor_features를 표준화하세요.
            raise NotImplementedError
        
        
        ae_train_x = sensor_matrix(normal_train)
        ae_valid_x = sensor_matrix(ae_valid)
        ae_test_x = sensor_matrix(ae_test)
        
        
        class AutoEncoder(nn.Module):
            def __init__(self, input_dim, latent_dim=2):
                super().__init__()
                # TODO 20-5b: input→8→latent→8→input 구조를 만드세요.
                raise NotImplementedError
        
            def forward(self, x):
                raise NotImplementedError
        
        
        # TODO 20-5c: 모델, Adam, 정상 train DataLoader를 만드세요.
        raise NotImplementedError
        """,
    )
    nb.code(
        solution="""
        for epoch in range(10):
            autoencoder.train()
            total = 0.0
            for (x,) in ae_loader:
                x = x.to(device)
                ae_optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(autoencoder(x), x)
                loss.backward()
                ae_optimizer.step()
                total += loss.item() * len(x)
            if epoch in {0, 4, 9}:
                print(f"AE epoch={epoch + 1:02d} loss={total / len(ae_loader.dataset):.4f}")
        
        
        @torch.inference_mode()
        def reconstruction_error(features):
            autoencoder.eval()
            x = torch.from_numpy(features).to(device)
            return ((autoencoder(x) - x) ** 2).mean(dim=1).cpu().numpy()
        
        
        valid_error = reconstruction_error(ae_valid_x)
        test_error = reconstruction_error(ae_test_x)
        """,
        exercise="""
        # TODO 20-6a: 정상 데이터로 autoencoder를 10 epoch MSE 학습하세요.
        raise NotImplementedError
        
        
        @torch.inference_mode()
        def reconstruction_error(features):
            # TODO 20-6b: 샘플별 feature 평균 제곱 복원 오차를 반환하세요.
            raise NotImplementedError
        
        
        valid_error = reconstruction_error(ae_valid_x)
        test_error = reconstruction_error(ae_test_x)
        """,
    )
    nb.md("""
    ## 5) 정상 validation 오차로 anomaly threshold 설정

    threshold 선택에도 test label을 사용하면 평가가 낙관적으로 변합니다. validation 중
    정상 샘플 오차의 99% 분위수를 사용하고 test label은 마지막 평가에만 사용합니다.
    """)
    nb.code(
        solution="""
        valid_normal_error = valid_error[ae_valid.anomaly.to_numpy() == 0]
        anomaly_threshold = float(np.quantile(valid_normal_error, 0.99))
        anomaly_prediction = test_error >= anomaly_threshold
        anomaly_true = ae_test.anomaly.to_numpy()
        ae_precision, ae_recall, ae_f1, _ = precision_recall_fscore_support(
            anomaly_true, anomaly_prediction, average="binary", zero_division=0
        )
        ae_auc = roc_auc_score(anomaly_true, test_error)
        unsupervised_metrics = {
            "threshold": anomaly_threshold,
            "precision": ae_precision,
            "recall": ae_recall,
            "f1": ae_f1,
            "roc_auc": ae_auc,
        }
        print(unsupervised_metrics)
        """,
        exercise="""
        # TODO 20-7: validation 정상 error의 99% 분위수로 threshold를 정하고
        # test precision/recall/F1/ROC-AUC를 계산하세요.
        raise NotImplementedError
        """,
    )
    nb.code("""
    assert anomaly_threshold > 0
    assert all(np.isfinite(v) for v in unsupervised_metrics.values())
    print(
        "detected/actual anomalies:", int(anomaly_prediction.sum()), int(anomaly_true.sum())
    )
    """)
    nb.md("""
    ## 지도 분류와 비지도 이상 탐지 비교

    | 방식 | 학습 데이터 | score | 장점 | 주의점 |
    |---|---|---|---|---|
    | MLP | 정상+위험 label | risk probability | 목표 label에 직접 최적화 | label 비용, 분포 변화 |
    | Autoencoder | 주로 정상 | reconstruction error | 이상 label 없이 시작 | 잘 복원되는 이상, threshold 민감도 |

    두 결과의 수치는 데이터셋과 target 정의가 달라 직접적인 승패 비교가 아닙니다.
    """)
    nb.code("""
    comparison = pd.DataFrame(
        [
            {"task": "account risk / supervised MLP", **supervised_metrics},
            {
                "task": "sensor anomaly / autoencoder",
                **{k: v for k, v in unsupervised_metrics.items() if k != "threshold"},
            },
        ]
    ).set_index("task")
    print(comparison)
    """)
    nb.md("""
    ## 운영 threshold 실험

    false negative 비용이 크면 threshold를 낮춰 recall을 높이고, 검토 인력이 제한되면
    precision 또는 상위 K alert를 최적화할 수 있습니다. 여러 threshold의 alert 수와
    precision/recall을 표로 만들어 운영팀과 합의하세요.
    """)
    nb.code("""
    operating_points = []
    for threshold in (0.3, 0.5, best_threshold, 0.7):
        prediction = test_probability >= threshold
        p, r, score, _ = precision_recall_fscore_support(
            y_test, prediction, average="binary", zero_division=0
        )
        operating_points.append(
            {
                "threshold": round(float(threshold), 3),
                "alerts": int(prediction.sum()),
                "precision": p,
                "recall": r,
                "f1": score,
            }
        )
    print(pd.DataFrame(operating_points).drop_duplicates("threshold"))
    """)
    nb.md("""
    ## 마무리

    불균형 문제에서는 모델 구조보다 split, train-only fit, 손실 가중치, validation threshold,
    운영 비용의 정의가 성능을 좌우합니다. autoencoder는 좋은 시작점이지만 시계열 문맥까지
    필요하다면 sequence autoencoder나 forecasting residual로 확장해 보세요.
    """)
    nb.write("20_tabular_mlp_and_anomaly_detection.ipynb")


def main() -> None:
    build_13()
    build_14()
    build_19()
    build_20()
    for path in sorted(EXERCISES.glob("*.ipynb")):
        if path.name in {
            "13_data_modalities_and_pipelines.ipynb",
            "14_rnn_sequence_modeling.ipynb",
            "19_cnn_image_classification.ipynb",
            "20_tabular_mlp_and_anomaly_detection.ipynb",
        }:
            nb = nbf.read(path, as_version=4)
            print(f"built {path.relative_to(ROOT)} ({len(nb.cells)} cells)")


def validate_all() -> None:
    filenames = [
        "13_data_modalities_and_pipelines.ipynb",
        "14_rnn_sequence_modeling.ipynb",
        "19_cnn_image_classification.ipynb",
        "20_tabular_mlp_and_anomaly_detection.ipynb",
    ]
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for filename in filenames:
            exercise = nbf.read(EXERCISES / filename, as_version=4)
            solution = nbf.read(SOLUTIONS / filename, as_version=4)
            nbf.validate(exercise)
            nbf.validate(solution)
            assert 24 <= len(exercise.cells) <= 34
            assert len(exercise.cells) == len(solution.cells)
            for index, (left, right) in enumerate(
                zip(exercise.cells, solution.cells, strict=True), start=1
            ):
                assert left.cell_type == right.cell_type, (filename, index, "cell type")
                if left.cell_type == "markdown":
                    assert left.source == right.source, (
                        filename,
                        index,
                        "markdown mismatch",
                    )
            assert any(
                "TODO" in cell.source or "NotImplementedError" in cell.source
                for cell in exercise.cells
                if cell.cell_type == "code"
            )

            # dataclass resolves annotations through sys.modules[cls.__module__].
            # Reusing __main__ keeps the lightweight cell runner faithful enough
            # without starting a second kernel process for every notebook.
            namespace = {"__name__": "__main__"}
            started = time.perf_counter()
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                for index, cell in enumerate(solution.cells, start=1):
                    if cell.cell_type != "code":
                        continue
                    try:
                        exec(
                            compile(cell.source, f"{filename}:cell-{index}", "exec"),
                            namespace,
                        )
                    except Exception as exc:
                        preview = (
                            cell.source.splitlines()[0] if cell.source else "<empty>"
                        )
                        recent_output = captured.getvalue()[-3000:]
                        failure_message = (
                            f"{filename} cell {index} failed: {preview}\n"
                            f"Recent output:\n{recent_output}"
                        )
                        raise RuntimeError(failure_message) from exc
            elapsed = time.perf_counter() - started
            print(
                f"validated+executed {filename}: {len(solution.cells)} cells, {elapsed:.1f}s"
            )
    finally:
        os.chdir(previous_cwd)
        for checkpoint_path in (
            ROOT / "artifacts" / "practice" / "rnn_sequence_classifier.pt",
            ROOT / "artifacts" / "practice" / "tiny_shapes_cnn.pt",
        ):
            checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
    validate_all()
