"""Self-supervised and multimodal portfolio paper specifications."""

from __future__ import annotations

from .common import CellSpec, FieldPaperSpec, clean
from .portfolio_stage import enrich_field_specs


def _cell(
    cell_type: str,
    exercise: str,
    solution: str,
    *tags: str,
) -> CellSpec:
    """Build one paired cell while keeping source blocks readable."""
    return CellSpec(
        cell_type=cell_type,
        exercise=clean(exercise),
        solution=clean(solution),
        tags=tuple(tags),
    )


SPECS: tuple[FieldPaperSpec, ...] = (
    # 00 ? CPC: 잠재공간에서 미래 예측
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=0,
        slug="cpc",
        short_title="CPC: 잠재공간에서 미래 예측",
        paper_title="Representation Learning with Contrastive Predictive Coding",
        authors="Aaron van den Oord, Yazhe Li, Oriol Vinyals",
        year=2018,
        primary_url="https://arxiv.org/abs/1807.03748",
        venue="arXiv",
        difficulty="중급",
        expected_minutes=65,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="GRU, mutual information, contrastive learning",
        reproduction_goal=(
            "encoder와 autoregressive context로 미래 latent를 batch negatives 중 식별하는 I"
            "nfoNCE를 학습합니다."
        ),
        original_scale="speech·image·text·RL 대규모 실험 대신 240개 길이-8 합성 시퀀스를 사용합니다.",
        mappings=(
            (
                "§2.1, Eq. (1)",
                "관측 x_t→latent z_t encoder",
                "latent shape",
            ),
            (
                "§2.1, Eq. (2)",
                "과거 latent→context c_t",
                "미래 누설 없는 prefix",
            ),
            (
                "§2.2, Eq. (3)",
                "bilinear density-ratio score",
                "N×N logits",
            ),
            (
                "§2.3, Eq. (4) 및 Figure 1",
                "InfoNCE로 positive future 식별",
                "loss·top-1",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{N}=-\mathbb E\!\left[\log \frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}
\right]$$

- **기호 정의:** c_t는 과거 context, x_{t+k}는 positive 미래, X는 batch 후보 집합입니다.
- **수식의 역할:** 미래 latent를 batch negative 중 식별해 관측 간 공유 정보를 보존합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §2.1, Eq. (1)을 구현합니다.
- **입출력 shape:** 시퀀스 [B, T, 4] → latent/context [B, T, D] → score [B, B]
- **평가:** InfoNCE 감소와 미래 식별 top-1 accuracy
- **원문 대비 한계:** speech·image·text·RL 대규모 실험 대신 240개 길이-8 합성 시퀀스를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{N}=-\mathbb E\!\left[\log \frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}
\right]$$

- **기호 정의:** c_t는 과거 context, x_{t+k}는 positive 미래, X는 batch 후보 집합입니다.
- **수식의 역할:** 미래 latent를 batch negative 중 식별해 관측 간 공유 정보를 보존합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §2.1, Eq. (1)을 구현합니다.
- **입출력 shape:** 시퀀스 [B, T, 4] → latent/context [B, T, D] → score [B, B]
- **평가:** InfoNCE 감소와 미래 식별 top-1 accuracy
- **원문 대비 한계:** speech·image·text·RL 대규모 실험 대신 240개 길이-8 합성 시퀀스를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "markdown",
                r"""
## 1. Figure 1: `x → z → c → 미래 z`
같은 batch의 다른 시퀀스 미래 latent가 negative 역할을 합니다.
""",
                r"""
## 1. Figure 1: `x → z → c → 미래 z`
같은 batch의 다른 시퀀스 미래 latent가 negative 역할을 합니다.
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 00-1: Eq. (1) encoder와 Eq. (2) autoregressive context를 정의하세요.
class CPCModel(nn.Module):
    def __init__(self, input_dim=4, latent=12):
        super().__init__()
        raise NotImplementedError("TODO 00-1")

    def encode_context(self, x: Tensor, time: int):
        raise NotImplementedError("TODO 00-1")
""",
                r"""
class CPCModel(nn.Module):
    def __init__(self, input_dim=4, latent=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent), nn.ReLU())
        self.ar = nn.GRU(latent, latent, batch_first=True)
        self.predict = nn.Linear(latent, latent, bias=False)

    def encode_context(self, x: Tensor, time: int):
        z = self.encoder(x)
        _, hidden = self.ar(z[:, : time + 1])
        return z, self.predict(hidden[-1])


cpc = CPCModel()
z, context = cpc.encode_context(sequences[:64], 3)
assert z.shape == (64, 8, 12) and context.shape == (64, 12)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 00-2: Eq. (3)–(4)의 in-batch InfoNCE를 구현하세요.
def cpc_loss(model: nn.Module, batch: Tensor, time=3, horizon=1):
    raise NotImplementedError("TODO 00-2")
""",
                r"""
def cpc_loss(model: nn.Module, batch: Tensor, time=3, horizon=1):
    z, context = model.encode_context(batch, time)
    future = z[:, time + horizon]
    logits = F.normalize(context, dim=1) @ F.normalize(future, dim=1).T / 0.12
    target = torch.arange(len(batch), device=logits.device)
    return F.cross_entropy(logits, target), logits


probe_loss, probe_logits = cpc_loss(cpc, sequences[:64])
assert probe_logits.shape == (64, 64) and torch.isfinite(probe_loss)
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 00-3: InfoNCE를 학습하고 positive-future top-1을 시각화하세요.
raise NotImplementedError("TODO 00-3")
""",
                r"""
cpc = ACCELERATOR.move(cpc)
batch = ACCELERATOR.move(sequences[train_idx[:96]])
optimizer = torch.optim.Adam(cpc.parameters(), lr=0.025)
losses = []
for _ in range(120):
    loss, _ = cpc_loss(cpc, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    test_sequences = ACCELERATOR.move(sequences[test_idx])
    _, logits = cpc_loss(cpc, test_sequences)
    accuracy = float(
        (logits.argmax(1) == torch.arange(len(test_idx), device=logits.device))
        .float()
        .mean()
    )
assert losses[-1] < losses[0] * 0.75 and accuracy > 1 / len(test_idx)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("InfoNCE")
ax[1].imshow(logits.detach().cpu(), cmap="magma")
ax[1].set_title("future logits")
plt.tight_layout()
plt.show()
print({"future_top1": round(accuracy, 3), "chance": round(1 / len(test_idx), 3)})
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class CpcPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    temperature: float = 0.2
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    return batch[:, :-1], batch[:, -1]


class CpcPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(sequences.shape[-1], config.hidden_dim)
        self.context = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.predictor = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, prefix: Tensor, future: Tensor) -> Tensor:
        prefix_latent = self.encoder(prefix)
        _, state = self.context(prefix_latent)
        prediction = F.normalize(self.predictor(state[-1]), dim=1)
        target = F.normalize(self.encoder(future), dim=1)
        return prediction @ target.T / self.config.temperature

    def compute_loss(self, logits: Tensor) -> Tensor:
        target = torch.arange(len(logits), device=logits.device)
        return F.cross_entropy(logits, target)

    def training_step(self, prefix: Tensor, future: Tensor) -> Tensor:
        return self.compute_loss(self(prefix, future))
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(
    model: CpcPortfolioModel,
    prefix: Tensor,
    future: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(prefix, future)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: CpcPortfolioModel,
    prefix: Tensor,
    future: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        logits = model(prefix, future)
        target = torch.arange(len(logits), device=logits.device)
        accuracy = (logits.argmax(1) == target).float().mean()
    return {"future_top1": float(accuracy.cpu())}


portfolio_train_index = train_idx[:64]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_prefix, portfolio_train_future = prepare_views(
    sequences[portfolio_train_index]
)
portfolio_test_prefix, portfolio_test_future = prepare_views(
    sequences[portfolio_test_index]
)
(
    portfolio_train_prefix,
    portfolio_train_future,
    portfolio_test_prefix,
    portfolio_test_future,
) = ACCELERATOR.move(
    portfolio_train_prefix,
    portfolio_train_future,
    portfolio_test_prefix,
    portfolio_test_future,
)
portfolio_model = ACCELERATOR.move(CpcPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_train_prefix,
    portfolio_train_future,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_test_prefix,
    portfolio_test_future,
)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 01 ? MoCo: momentum dictionary와 queue
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=1,
        slug="moco",
        short_title="MoCo: momentum dictionary와 queue",
        paper_title="Momentum Contrast for Unsupervised Visual Representation Learning",
        authors="Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick",
        year=2020,
        primary_url="https://arxiv.org/abs/1911.05722",
        venue="CVPR",
        difficulty="중급",
        expected_minutes=70,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="contrastive loss, EMA, queue",
        reproduction_goal=(
            "query/key encoder, momentum update, FIFO negative queue를 갖는 MoCo 한 학"
            "습 단계를 구현합니다."
        ),
        original_scale="ImageNet ResNet-50·65K queue 대신 8×8 이미지·64개 queue를 사용합니다.",
        mappings=(
            (
                "Algorithm 1, lines 1–3 및 §3.1",
                "두 stochastic view→query/key encoder",
                "view 차이·unit-norm embedding shape",
            ),
            (
                "§3.1, Eq. (1)",
                "query-positive/queue-negative InfoNCE",
                "logit shape·target 0",
            ),
            (
                "§3.2, Eq. (2)",
                "key encoder momentum update",
                "EMA 수치",
            ),
            (
                "§3.2, dictionary as a queue",
                "enqueue/dequeue",
                "고정 길이·unit norm",
            ),
            (
                "Algorithm 1",
                "stop-gradient key branch 학습",
                "loss·positive similarity",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_q=-\log\frac{\exp(q\cdot k_+/\tau)}{\exp(q\cdot k_+/\tau)+\sum_{k_-}\exp(q
\cdot k_-/\tau)}$$

- **기호 정의:** q는 query, k+는 momentum positive, k-는 queue negative, τ는 온도입니다.
- **수식의 역할:** EMA key encoder와 FIFO queue로 크고 일관된 dictionary를 유지합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1, Eq. (1)을 구현합니다.
- **입출력 shape:** 두 image view [B, 1, 8, 8] → q/k [B, D] → logit [B, 1+Q]
- **평가:** contrastive loss, positive-vs-queue cosine, queue 길이와 norm
- **원문 대비 한계:** ImageNet ResNet-50·65K queue 대신 8×8 이미지·64개 queue를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_q=-\log\frac{\exp(q\cdot k_+/\tau)}{\exp(q\cdot k_+/\tau)+\sum_{k_-}\exp(q
\cdot k_-/\tau)}$$

- **기호 정의:** q는 query, k+는 momentum positive, k-는 queue negative, τ는 온도입니다.
- **수식의 역할:** EMA key encoder와 FIFO queue로 크고 일관된 dictionary를 유지합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1, Eq. (1)을 구현합니다.
- **입출력 shape:** 두 image view [B, 1, 8, 8] → q/k [B, D] → logit [B, 1+Q]
- **평가:** contrastive loss, positive-vs-queue cosine, queue 길이와 norm
- **원문 대비 한계:** ImageNet ResNet-50·65K queue 대신 8×8 이미지·64개 queue를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "code",
                r"""
def image_view(x: Tensor, scale: float):
    return (x * scale + 0.035 * torch.randn_like(x)).clamp(0, 1)


class TinyEncoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
""",
                r"""
def image_view(x: Tensor, scale: float):
    return (x * scale + 0.035 * torch.randn_like(x)).clamp(0, 1)


class TinyEncoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 01-1: Eq. (2)의 key-encoder EMA update를 구현하세요.
@torch.no_grad()
def momentum_update(query: nn.Module, key: nn.Module, momentum: float):
    raise NotImplementedError("TODO 01-1")
""",
                r"""
@torch.no_grad()
def momentum_update(query: nn.Module, key: nn.Module, momentum: float):
    for q, k in zip(query.parameters(), key.parameters()):
        k.data.mul_(momentum).add_(q.data, alpha=1 - momentum)


query_encoder = TinyEncoder()
key_encoder = deepcopy(query_encoder)
before = next(key_encoder.parameters()).clone()
next(query_encoder.parameters()).data.add_(0.1)
momentum_update(query_encoder, key_encoder, 0.9)
assert torch.allclose(
    next(key_encoder.parameters()),
    0.9 * before + 0.1 * next(query_encoder.parameters()),
    atol=1e-6,
)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 01-2: Eq. (1)의 positive 1개 + queue negatives logits/loss를 구현하세요.
def moco_objective(q: Tensor, k: Tensor, queue: Tensor, temperature=0.2):
    raise NotImplementedError("TODO 01-2")
""",
                r"""
def moco_objective(q: Tensor, k: Tensor, queue: Tensor, temperature=0.2):
    positive = (q * k).sum(1, keepdim=True)
    negative = q @ queue.T
    logits = torch.cat([positive, negative], 1) / temperature
    return F.cross_entropy(
        logits, torch.zeros(len(q), dtype=torch.long, device=logits.device)
    ), logits


queue = F.normalize(torch.randn(64, 16), dim=1)
sample = images[:24]
q = query_encoder(image_view(sample, 0.9))
k = key_encoder(image_view(sample, 1.1)).detach()
loss, logits = moco_objective(q, k, queue)
assert logits.shape == (24, 65) and torch.isfinite(loss)
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 01-3: FIFO queue와 Algorithm 1 학습을 연결하세요.
raise NotImplementedError("TODO 01-3")
""",
                r"""
query_encoder, key_encoder, queue = ACCELERATOR.move(query_encoder, key_encoder, queue)
optimizer = torch.optim.Adam(query_encoder.parameters(), lr=0.02)
losses = []
batch = ACCELERATOR.move(images[train_idx[:96]])
for step in range(100):
    ids = torch.arange(step * 24, (step + 1) * 24, device=batch.device) % len(batch)
    x = batch[ids]
    q = query_encoder(image_view(x, 0.9))
    with torch.no_grad():
        momentum_update(query_encoder, key_encoder, 0.97)
        k = key_encoder(image_view(x, 1.1))
    loss, _ = moco_objective(q, k, queue)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
    queue = torch.cat([k, queue], 0)[:64].detach()
with torch.no_grad():
    test_images = ACCELERATOR.move(images[test_idx])
    q = query_encoder(test_images)
    k = key_encoder(test_images)
    positive = float((q * k).sum(1).mean())
    negative = float((q @ queue.T).mean())
assert (
    queue.shape == (64, 16)
    and torch.allclose(
        queue.norm(dim=1), torch.ones(64, device=queue.device), atol=1e-5
    )
    and positive > negative
)
plt.plot(losses)
plt.title("MoCo InfoNCE")
plt.xlabel("step")
plt.show()
print({"positive": round(positive, 3), "queue_negative": round(negative, 3)})
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class MocoPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    temperature: float = 0.2
    momentum: float = 0.9
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    first = (batch + 0.03 * torch.randn_like(batch)).clamp(0.0, 1.0)
    second = torch.flip(batch, dims=[3])
    return first, second


class MocoPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.online = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.hidden_dim),
        )
        self.target = deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        self.register_buffer(
            "queue", F.normalize(torch.randn(48, config.hidden_dim), dim=1)
        )
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))

    def forward(self, first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
        online = F.normalize(self.online(first), dim=1)
        with torch.no_grad():
            target = F.normalize(self.target(second), dim=1)
        return online, target

    def compute_loss(self, online: Tensor, target: Tensor) -> Tensor:

        positive = (online * target).sum(dim=1, keepdim=True)
        negative = online @ self.queue.T
        logits = torch.cat([positive, negative], dim=1) / self.config.temperature
        labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def update_target(self) -> None:
        with torch.no_grad():
            for target, online in zip(
                self.target.parameters(),
                self.online.parameters(),
                strict=True,
            ):
                target.mul_(self.config.momentum)
                target.add_(online, alpha=1.0 - self.config.momentum)

    @torch.no_grad()
    def enqueue(self, keys: Tensor) -> None:
        keys = keys.detach()
        capacity = len(self.queue)
        if len(keys) >= capacity:
            self.queue.copy_(keys[-capacity:])
            self.queue_pointer.zero_()
            return
        pointer = int(self.queue_pointer)
        first_count = min(capacity - pointer, len(keys))
        self.queue[pointer : pointer + first_count] = keys[:first_count]
        remaining = len(keys) - first_count
        if remaining:
            self.queue[:remaining] = keys[first_count:]
        self.queue_pointer.fill_((pointer + len(keys)) % capacity)

    def training_step(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        first, second = prepare_views(batch)
        online, target = self(first, second)
        return self.compute_loss(online, target), target
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(model: MocoPortfolioModel, batch: Tensor) -> list[float]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=model.config.learning_rate)
    history = []
    initial_queue = model.queue.clone()
    initial_pointer = int(model.queue_pointer)
    for _ in range(model.config.steps):
        loss, keys = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target()
        model.enqueue(keys)
        history.append(float(loss.detach().cpu()))
    assert not torch.allclose(initial_queue, model.queue)
    assert int(model.queue_pointer) != initial_pointer
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: MocoPortfolioModel,
    batch: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        first, second = prepare_views(batch)
        online, target = model(first, second)
        agreement = (online * target).sum(dim=1).mean()
        spread = online.std(dim=0).mean()
    return {
        "view_agreement": float(agreement.cpu()),
        "feature_spread": float(spread.cpu()),
    }


portfolio_train_index = train_idx[:32]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_images, portfolio_test_images = ACCELERATOR.move(
    images[portfolio_train_index],
    images[portfolio_test_index],
)
portfolio_model = ACCELERATOR.move(MocoPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(portfolio_model, portfolio_train_images)
portfolio_metrics = evaluate_portfolio_model(portfolio_model, portfolio_test_images)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 02 ? SimCLR: augmentation과 NT-Xent
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=2,
        slug="simclr",
        short_title="SimCLR: augmentation과 NT-Xent",
        paper_title=(
            "A Simple Framework for Contrastive Learning of Visual Representations"
        ),
        authors="Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton",
        year=2020,
        primary_url="https://arxiv.org/abs/2002.05709",
        venue="ICML",
        difficulty="중급",
        expected_minutes=70,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="cosine similarity, cross entropy, augmentation",
        reproduction_goal="두 augmentation view, encoder f, projection g, 대칭 NT-Xent를 구현합니다.",
        original_scale="ImageNet·ResNet-50·batch 4096 대신 96개 8×8 이미지와 작은 MLP를 사용합니다.",
        mappings=(
            (
                "Figure 2",
                "x→두 view→f→g",
                "view 비동일성·shape",
            ),
            (
                "§2.1, Eq. (1)",
                "L2-normalized cosine",
                "대각/범위",
            ),
            (
                "§2.1, Eq. (2)",
                "2N 대칭 NT-Xent",
                "self mask·positive index",
            ),
            (
                "Algorithm 1 및 §2.3",
                "projection 학습·h 평가",
                "same/different class cosine",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\ell_{i,j}=-\log\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(
\operatorname{sim}(z_i,z_k)/\tau)}$$

- **기호 정의:** i와 j는 같은 샘플의 두 view, z는 projection, τ는 온도입니다.
- **수식의 역할:** 강한 augmentation pair를 positive로 두고 2B-2개 view를 negative로 사용합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 Figure 2을 구현합니다.
- **입출력 shape:** 두 view [B, 1, 8, 8] → projection [2B, D] → similarity [2B, 2B]
- **평가:** 대칭 NT-Xent와 같은 클래스/다른 클래스 cosine gap
- **원문 대비 한계:** ImageNet·ResNet-50·batch 4096 대신 96개 8×8 이미지와 작은 MLP를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\ell_{i,j}=-\log\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(
\operatorname{sim}(z_i,z_k)/\tau)}$$

- **기호 정의:** i와 j는 같은 샘플의 두 view, z는 projection, τ는 온도입니다.
- **수식의 역할:** 강한 augmentation pair를 positive로 두고 2B-2개 view를 negative로 사용합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 Figure 2을 구현합니다.
- **입출력 shape:** 두 view [B, 1, 8, 8] → projection [2B, D] → similarity [2B, 2B]
- **평가:** 대칭 NT-Xent와 같은 클래스/다른 클래스 cosine gap
- **원문 대비 한계:** ImageNet·ResNet-50·batch 4096 대신 96개 8×8 이미지와 작은 MLP를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 02-1: 서로 독립적인 두 brightness/noise view를 만드세요.
def make_views(x: Tensor):
    raise NotImplementedError("TODO 02-1")
""",
                r"""
def make_views(x: Tensor):
    def aug():
        scale = 0.75 + 0.5 * torch.rand(len(x), 1, 1, 1, device=x.device)
        return (x * scale + 0.04 * torch.randn_like(x)).clamp(0, 1)

    return aug(), aug()


a, b = make_views(images[:96])
assert a.shape == b.shape == (96, 1, 8, 8) and not torch.equal(a, b)
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 02-2: Eq. (1)–(2)의 2N NT-Xent를 구현하세요.
def nt_xent(z1: Tensor, z2: Tensor, temperature=0.2):
    raise NotImplementedError("TODO 02-2")
""",
                r"""
def nt_xent(z1: Tensor, z2: Tensor, temperature=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-1e9)
    targets = (torch.arange(2 * n, device=logits.device) + n) % (2 * n)
    return F.cross_entropy(logits, targets), logits


probe, _ = nt_xent(torch.randn(8, 6), torch.randn(8, 6))
assert torch.isfinite(probe)
""",
                "task",
            ),
            # Cell 05 ? metric
            _cell(
                "code",
                r"""
# TODO 02-3: encoder f/projector g를 학습하고 h의 class cosine을 그리세요.
raise NotImplementedError("TODO 02-3")
""",
                r"""
encoder = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
projector = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
encoder, projector = ACCELERATOR.move(encoder, projector)
optimizer = torch.optim.Adam([*encoder.parameters(), *projector.parameters()], lr=0.02)
batch = ACCELERATOR.move(images[train_idx[:96]])
losses = []
for _ in range(110):
    a, b = make_views(batch)
    loss, _ = nt_xent(projector(encoder(a)), projector(encoder(b)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    h = F.normalize(encoder(ACCELERATOR.move(images[test_idx])), dim=1).cpu()
    sim = h @ h.T
    y = class_ids[test_idx]
    eye = torch.eye(len(y), dtype=torch.bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = y[:, None] != y[None, :]
    sm = float(sim[same].mean())
    dm = float(sim[diff].mean())
assert losses[-1] < losses[0] and sm > dm
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("NT-Xent")
ax[1].bar(["same", "different"], [sm, dm])
ax[1].set_title("encoder h cosine")
plt.tight_layout()
plt.show()
""",
                "metric",
            ),
            # Cell 06 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 07 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class SimclrPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def nt_xent_logits(self, *projections: Tensor):
        raise NotImplementedError("TODO P-1: 2B x 2B NT-Xent logits")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    temperature: float = 0.2
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    first = (batch + 0.04 * torch.randn_like(batch)).clamp(0.0, 1.0)
    second = torch.flip(batch, dims=[3])
    second = (second + 0.04 * torch.randn_like(second)).clamp(0.0, 1.0)
    return first, second


class SimclrPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, config.hidden_dim),
        )

    def forward(self, view: Tensor) -> Tensor:
        return F.normalize(self.projector(self.encoder(view)), dim=1)

    def nt_xent_logits(
        self,
        first_projection: Tensor,
        second_projection: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(first_projection)
        representations = torch.cat(
            [first_projection, second_projection],
            dim=0,
        )
        pair_count = 2 * batch_size
        logits = representations @ representations.T
        logits = logits / self.config.temperature
        self_mask = torch.eye(
            pair_count,
            dtype=torch.bool,
            device=logits.device,
        )
        logits = logits.masked_fill(self_mask, -torch.inf)
        anchors = torch.arange(pair_count, device=logits.device)
        positive_index = (anchors + batch_size) % pair_count

        expected_positive = torch.cat(
            [anchors[batch_size:], anchors[:batch_size]],
        )
        assert logits.shape == (pair_count, pair_count)
        assert torch.isneginf(logits.diagonal()).all()
        assert torch.equal(positive_index, expected_positive)
        return logits, positive_index

    def compute_loss(self, first: Tensor, second: Tensor) -> Tensor:
        first_projection = self(first)
        second_projection = self(second)
        logits, positive_index = self.nt_xent_logits(
            first_projection,
            second_projection,
        )
        return F.cross_entropy(logits, positive_index)

    def training_step(self, batch: Tensor) -> Tensor:
        first, second = prepare_views(batch)
        return self.compute_loss(first, second)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(model: SimclrPortfolioModel, batch: Tensor) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 10 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: SimclrPortfolioModel,
    batch: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        first, second = prepare_views(batch)
        logits, positive_index = model.nt_xent_logits(
            model(first),
            model(second),
        )
        accuracy = (logits.argmax(1) == positive_index).float().mean()
        loss = F.cross_entropy(logits, positive_index)
    return {
        "nt_xent_pair_top1": float(accuracy.cpu()),
        "nt_xent_loss": float(loss.cpu()),
    }


portfolio_train_index = train_idx[:64]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_images, portfolio_test_images = ACCELERATOR.move(
    images[portfolio_train_index],
    images[portfolio_test_index],
)
portfolio_model = ACCELERATOR.move(SimclrPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(portfolio_model, portfolio_train_images)
portfolio_metrics = evaluate_portfolio_model(portfolio_model, portfolio_test_images)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 03 ? BYOL: negative 없이 bootstrap
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=3,
        slug="byol",
        short_title="BYOL: negative 없이 bootstrap",
        paper_title=(
            "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
        ),
        authors="Jean-Bastien Grill et al.",
        year=2020,
        primary_url="https://arxiv.org/abs/2006.07733",
        venue="NeurIPS",
        difficulty="중급",
        expected_minutes=70,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="EMA, stop-gradient, normalized regression",
        reproduction_goal="비대칭 online predictor와 stop-gradient target EMA로 두 view 표현을 회귀합니다.",
        original_scale="ImageNet·ResNet 대신 96개 8×8 이미지와 MLP를 사용하며 최종 성능 재현은 범위 밖입니다.",
        mappings=(
            (
                "§3.1, Figure 2",
                "online encoder/projector/predictor와 target",
                "비대칭 구조",
            ),
            (
                "Eq. (1)",
                "target parameter EMA",
                "수치 update",
            ),
            (
                "Eq. (2)",
                "normalized prediction regression",
                "2−2 cosine 동치",
            ),
            (
                "Eq. (3)",
                "symmetric loss·target stop-gradient",
                "loss·feature std",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{\theta,\xi}=\left\|\bar q_\theta(z_\theta)-\operatorname{sg}(\bar z_\xi')
\right\|_2^2,\quad\xi\leftarrow\tau\xi+(1-\tau)\theta$$

- **기호 정의:** θ는 online, ξ는 target, q는 predictor, sg는 stop-gradient입니다.
- **수식의 역할:** negative 없이 비대칭 predictor와 느린 EMA target을 bootstrap 대상으로 씁니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1, Figure 2을 구현합니다.
- **입출력 shape:** 두 view [B, 1, 8, 8] → online prediction/target [B, D]
- **평가:** normalized regression loss, target gradient 없음, representation 표준편차
- **원문 대비 한계:** ImageNet·ResNet 대신 96개 8×8 이미지와 MLP를 사용하며 최종 성능 재현은 범위 밖입니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{\theta,\xi}=\left\|\bar q_\theta(z_\theta)-\operatorname{sg}(\bar z_\xi')
\right\|_2^2,\quad\xi\leftarrow\tau\xi+(1-\tau)\theta$$

- **기호 정의:** θ는 online, ξ는 target, q는 predictor, sg는 stop-gradient입니다.
- **수식의 역할:** negative 없이 비대칭 predictor와 느린 EMA target을 bootstrap 대상으로 씁니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1, Figure 2을 구현합니다.
- **입출력 shape:** 두 view [B, 1, 8, 8] → online prediction/target [B, D]
- **평가:** normalized regression loss, target gradient 없음, representation 표준편차
- **원문 대비 한계:** ImageNet·ResNet 대신 96개 8×8 이미지와 MLP를 사용하며 최종 성능 재현은 범위 밖입니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "code",
                r"""
def byol_view(x):
    scale = 0.8 + 0.4 * torch.rand(len(x), 1, 1, 1, device=x.device)
    return (x * scale + 0.035 * torch.randn_like(x)).clamp(0, 1)


def backbone():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
""",
                r"""
def byol_view(x):
    scale = 0.8 + 0.4 * torch.rand(len(x), 1, 1, 1, device=x.device)
    return (x * scale + 0.035 * torch.randn_like(x)).clamp(0, 1)


def backbone():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 03-1: Eq. (1)의 target EMA를 구현하세요.
@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, tau: float):
    raise NotImplementedError("TODO 03-1")
""",
                r"""
@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, tau: float):
    for o, t in zip(online.parameters(), target.parameters()):
        t.data.mul_(tau).add_(o.data, alpha=1 - tau)


online = backbone()
target = deepcopy(online)
old = next(target.parameters()).clone()
next(online.parameters()).data.add_(0.2)
ema_update(online, target, 0.9)
assert torch.allclose(
    next(target.parameters()), 0.9 * old + 0.1 * next(online.parameters()), atol=1e-6
)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 03-2: Eq. (2)의 normalized MSE와 양방향 loss를 구현하세요.
def normalized_mse(pred, target):
    raise NotImplementedError("TODO 03-2")
""",
                r"""
def normalized_mse(pred, target):
    return 2 - 2 * (F.normalize(pred, dim=1) * F.normalize(target, dim=1)).sum(1).mean()


predictor = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 16))
x = images[:16]
v1, v2 = byol_view(x), byol_view(x)
test_loss = normalized_mse(predictor(online(v1)), target(v2).detach())
assert 0 <= float(test_loss) <= 4
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 03-3: symmetric BYOL update와 target EMA를 반복하고 collapse 지표를 그리세요.
raise NotImplementedError("TODO 03-3")
""",
                r"""
online, target, predictor = ACCELERATOR.move(online, target, predictor)
optimizer = torch.optim.Adam([*online.parameters(), *predictor.parameters()], lr=0.012)
batch = ACCELERATOR.move(images[train_idx[:96]])
losses = []
stds = []
for _ in range(120):
    v1, v2 = byol_view(batch), byol_view(batch)
    p1, p2 = predictor(online(v1)), predictor(online(v2))
    with torch.no_grad():
        t1, t2 = target(v1), target(v2)
    loss = normalized_mse(p1, t2) + normalized_mse(p2, t1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema_update(online, target, 0.97)
    losses.append(float(loss.detach().cpu()))
    stds.append(float(F.normalize(online(batch), dim=1).std(0).mean().detach().cpu()))
assert (
    losses[-1] < losses[0]
    and stds[-1] > 1e-3
    and all(p.grad is None for p in target.parameters())
)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("BYOL loss")
ax[1].plot(stds)
ax[1].set_title("feature std (collapse check)")
plt.tight_layout()
plt.show()
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class ByolPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    momentum: float = 0.9
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    first = (batch + 0.03 * torch.randn_like(batch)).clamp(0.0, 1.0)
    second = torch.flip(batch, dims=[3])
    return first, second


class ByolPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.online = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.hidden_dim),
        )
        self.target = deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()
        self.predictor = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        first: Tensor,
        second: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        online_first = F.normalize(self.online(first), dim=1)
        online_second = F.normalize(self.online(second), dim=1)
        with torch.no_grad():
            target_first = F.normalize(self.target(first), dim=1)
            target_second = F.normalize(self.target(second), dim=1)
        assert not target_first.requires_grad
        assert not target_second.requires_grad
        return online_first, online_second, target_first, target_second

    def compute_loss(
        self,
        online_first: Tensor,
        online_second: Tensor,
        target_first: Tensor,
        target_second: Tensor,
    ) -> Tensor:
        prediction_first = F.normalize(self.predictor(online_first), dim=1)
        prediction_second = F.normalize(self.predictor(online_second), dim=1)
        first_to_second = 2.0 - 2.0 * (
            prediction_first * target_second.detach()
        ).sum(dim=1)
        second_to_first = 2.0 - 2.0 * (
            prediction_second * target_first.detach()
        ).sum(dim=1)
        return 0.5 * (first_to_second.mean() + second_to_first.mean())

    def update_target(self) -> None:
        with torch.no_grad():
            for target, online in zip(
                self.target.parameters(),
                self.online.parameters(),
                strict=True,
            ):
                target.mul_(self.config.momentum)
                target.add_(online, alpha=1.0 - self.config.momentum)

    def training_step(self, batch: Tensor) -> Tensor:
        first, second = prepare_views(batch)
        representations = self(first, second)
        return self.compute_loss(*representations)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(model: ByolPortfolioModel, batch: Tensor) -> list[float]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        target_before = [
            parameter.detach().clone()
            for parameter in model.target.parameters()
        ]
        loss = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        assert all(
            parameter.grad is None
            for parameter in model.target.parameters()
        )
        optimizer.step()
        assert all(
            torch.equal(before, after)
            for before, after in zip(
                target_before,
                model.target.parameters(),
                strict=True,
            )
        )
        model.update_target()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: ByolPortfolioModel,
    batch: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        first, second = prepare_views(batch)
        representations = model(first, second)
        regression = model.compute_loss(*representations)
        online_first, online_second, _, _ = representations
        online_views = torch.cat([online_first, online_second], dim=0)
        agreement = 1.0 - regression / 2.0
        spread = online_views.std(dim=0).mean()
    return {
        "symmetric_view_agreement": float(agreement.cpu()),
        "symmetric_regression_loss": float(regression.cpu()),
        "online_feature_spread": float(spread.cpu()),
    }


portfolio_train_index = train_idx[:64]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_images, portfolio_test_images = ACCELERATOR.move(
    images[portfolio_train_index],
    images[portfolio_test_index],
)
portfolio_model = ACCELERATOR.move(ByolPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(portfolio_model, portfolio_train_images)
portfolio_metrics = evaluate_portfolio_model(portfolio_model, portfolio_test_images)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 04 ? DINO: label 없는 self-distillation
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=4,
        slug="dino",
        short_title="DINO: label 없는 self-distillation",
        paper_title="Emerging Properties in Self-Supervised Vision Transformers",
        authors="Mathilde Caron et al.",
        year=2021,
        primary_url="https://arxiv.org/abs/2104.14294",
        venue="ICCV",
        difficulty="중급",
        expected_minutes=70,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="knowledge distillation, EMA, temperature softmax",
        reproduction_goal=(
            "student/teacher, sharpening, centering, EMA를 결합한 DINO self-distillat"
            "ion을 구현합니다."
        ),
        original_scale="ImageNet ViT multi-crop 대신 8×8 이미지의 두 global view와 MLP를 사용합니다.",
        mappings=(
            (
                "§3, Figure 2",
                "동일 image의 다른 view student/teacher",
                "cross-view loss",
            ),
            (
                "Eq. (1)–(2)",
                "temperature softmax와 cross entropy",
                "확률 합",
            ),
            (
                "Eq. (3)",
                "teacher parameter EMA",
                "stop-gradient",
            ),
            (
                "§3.2 및 Figure 3",
                "centering+sharpening collapse 방지",
                "entropy·center 변화",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L=\tfrac12[H(P_t(x_2),P_s(x_1))+H(P_t(x_1),P_s(x_2))]$$

$$P_t(x)=\operatorname{softmax}((g_{\theta_t}(x)-c)/\tau_t)$$

- **기호 정의:** P_s/P_t는 student/teacher 분포, c는 center, τ_t는 teacher 온도입니다.
- **수식의 역할:** centering·sharpening EMA teacher를 반대 view의 student에 대칭 증류합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3, Figure 2을 구현합니다.
- **입출력 shape:** 두 image view [B, 1, 8, 8] → prototype 확률 [B, K]
- **평가:** cross-view CE, teacher entropy, center 및 EMA update
- **원문 대비 한계:** ImageNet ViT multi-crop 대신 8×8 이미지의 두 global view와 MLP를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L=\tfrac12[H(P_t(x_2),P_s(x_1))+H(P_t(x_1),P_s(x_2))]$$

$$P_t(x)=\operatorname{softmax}((g_{\theta_t}(x)-c)/\tau_t)$$

- **기호 정의:** P_s/P_t는 student/teacher 분포, c는 center, τ_t는 teacher 온도입니다.
- **수식의 역할:** centering·sharpening EMA teacher를 반대 view의 student에 대칭 증류합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3, Figure 2을 구현합니다.
- **입출력 shape:** 두 image view [B, 1, 8, 8] → prototype 확률 [B, K]
- **평가:** cross-view CE, teacher entropy, center 및 EMA update
- **원문 대비 한계:** ImageNet ViT multi-crop 대신 8×8 이미지의 두 global view와 MLP를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "code",
                r"""
def dino_net(out=24):
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 48), nn.GELU(), nn.Linear(48, out))


def dino_view(x):
    return (
        x * (0.75 + 0.5 * torch.rand(len(x), 1, 1, 1, device=x.device))
        + 0.03 * torch.randn_like(x)
    ).clamp(0, 1)
""",
                r"""
def dino_net(out=24):
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 48), nn.GELU(), nn.Linear(48, out))


def dino_view(x):
    return (
        x * (0.75 + 0.5 * torch.rand(len(x), 1, 1, 1, device=x.device))
        + 0.03 * torch.randn_like(x)
    ).clamp(0, 1)
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 04-1: centering과 서로 다른 temperature의 student/teacher 분포를 만드세요.
def dino_distributions(
    student_logits, teacher_logits, center, student_t=0.1, teacher_t=0.04
):
    raise NotImplementedError("TODO 04-1")
""",
                r"""
def dino_distributions(
    student_logits, teacher_logits, center, student_t=0.1, teacher_t=0.04
):
    return F.log_softmax(student_logits / student_t, dim=1), F.softmax(
        (teacher_logits - center) / teacher_t, dim=1
    ).detach()


s_log, t_prob = dino_distributions(
    torch.randn(8, 24), torch.randn(8, 24), torch.zeros(1, 24)
)
assert torch.allclose(t_prob.sum(1), torch.ones(8)) and not t_prob.requires_grad
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 04-2: cross-view distillation CE와 teacher EMA를 구현하세요.
def distill_ce(student_log, teacher_prob):
    raise NotImplementedError("TODO 04-2")


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    raise NotImplementedError("TODO 04-2")
""",
                r"""
def distill_ce(student_log, teacher_prob):
    return -(teacher_prob * student_log).sum(1).mean()


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for s, t in zip(student.parameters(), teacher.parameters()):
        t.data.mul_(momentum).add_(s.data, alpha=1 - momentum)


assert torch.isfinite(distill_ce(s_log, t_prob))
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 04-3: DINO update와 center EMA를 학습하고 teacher entropy를 추적하세요.
raise NotImplementedError("TODO 04-3")
""",
                r"""
student = dino_net()
teacher = deepcopy(student)
student, teacher = ACCELERATOR.move(student, teacher)
optimizer = torch.optim.Adam(student.parameters(), lr=0.008)
center = torch.zeros(1, 24, device=DEVICE)
batch = ACCELERATOR.move(images[train_idx[:96]])
losses = []
entropies = []
for _ in range(120):
    v1, v2 = dino_view(batch), dino_view(batch)
    sl1, sl2 = student(v1), student(v2)
    with torch.no_grad():
        tl1, tl2 = teacher(v1), teacher(v2)
    s1, t2 = dino_distributions(sl1, tl2, center)
    s2, t1 = dino_distributions(sl2, tl1, center)
    loss = (distill_ce(s1, t2) + distill_ce(s2, t1)) / 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    update_teacher(student, teacher, 0.97)
    with torch.no_grad():
        center = 0.9 * center + 0.1 * torch.cat([tl1, tl2]).mean(0, keepdim=True)
        entropy = -(t1 * t1.clamp_min(1e-8).log()).sum(1).mean()
    losses.append(float(loss.detach().cpu()))
    entropies.append(float(entropy.detach().cpu()))
assert (
    torch.isfinite(center).all()
    and all(p.grad is None for p in teacher.parameters())
    and np.isfinite(losses).all()
)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("DINO loss")
ax[1].plot(entropies)
ax[1].set_title("teacher entropy")
plt.tight_layout()
plt.show()
print(
    {
        "center_norm": round(float(center.norm()), 3),
        "final_entropy": round(entropies[-1], 3),
    }
)
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class DinoPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    student_temperature: float = 0.2
    teacher_temperature: float = 0.07
    momentum: float = 0.9
    center_momentum: float = 0.9
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    first = (batch + 0.03 * torch.randn_like(batch)).clamp(0.0, 1.0)
    second = torch.flip(batch, dims=[3])
    return first, second


class DinoPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.online = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.hidden_dim),
        )
        self.target = deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, config.hidden_dim))

    def forward(
        self,
        first: Tensor,
        second: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        student_first = self.online(first)
        student_second = self.online(second)
        with torch.no_grad():
            teacher_first = self.target(first)
            teacher_second = self.target(second)
        return student_first, student_second, teacher_first, teacher_second

    def compute_loss(
        self,
        student_first: Tensor,
        student_second: Tensor,
        teacher_first: Tensor,
        teacher_second: Tensor,
    ) -> Tensor:
        student_first_log = F.log_softmax(
            student_first / self.config.student_temperature,
            dim=1,
        )
        student_second_log = F.log_softmax(
            student_second / self.config.student_temperature,
            dim=1,
        )
        teacher_first_prob = F.softmax(
            (teacher_first - self.center) / self.config.teacher_temperature,
            dim=1,
        ).detach()
        teacher_second_prob = F.softmax(
            (teacher_second - self.center) / self.config.teacher_temperature,
            dim=1,
        ).detach()
        first_to_second = -(teacher_second_prob * student_first_log).sum(dim=1)
        second_to_first = -(teacher_first_prob * student_second_log).sum(dim=1)
        return 0.5 * (first_to_second.mean() + second_to_first.mean())

    @torch.no_grad()
    def update_center(self, teacher_first: Tensor, teacher_second: Tensor) -> None:
        batch_center = torch.cat([teacher_first, teacher_second]).mean(0, keepdim=True)
        momentum = self.config.center_momentum
        self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)

    def update_target(self) -> None:
        with torch.no_grad():
            for target, online in zip(
                self.target.parameters(),
                self.online.parameters(),
                strict=True,
            ):
                target.mul_(self.config.momentum)
                target.add_(online, alpha=1.0 - self.config.momentum)

    def training_step(self, batch: Tensor) -> Tensor:
        first, second = prepare_views(batch)
        outputs = self(first, second)
        loss = self.compute_loss(*outputs)
        self.update_center(outputs[2], outputs[3])
        return loss
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(model: DinoPortfolioModel, batch: Tensor) -> list[float]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: DinoPortfolioModel,
    batch: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        first, second = prepare_views(batch)
        student_first, student_second, teacher_first, teacher_second = model(
            first,
            second,
        )
        symmetric_loss = model.compute_loss(
            student_first,
            student_second,
            teacher_first,
            teacher_second,
        )
        normalized_student = F.normalize(student_first, dim=1)
        normalized_teacher = F.normalize(teacher_second, dim=1)
        agreement = (normalized_student * normalized_teacher).sum(dim=1).mean()
        spread = normalized_student.std(dim=0).mean()
    return {
        "symmetric_cross_view_loss": float(symmetric_loss.cpu()),
        "view_agreement": float(agreement.cpu()),
        "feature_spread": float(spread.cpu()),
    }


portfolio_train_index = train_idx[:64]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_images, portfolio_test_images = ACCELERATOR.move(
    images[portfolio_train_index],
    images[portfolio_test_index],
)
portfolio_model = ACCELERATOR.move(DinoPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(portfolio_model, portfolio_train_images)
portfolio_metrics = evaluate_portfolio_model(portfolio_model, portfolio_test_images)
assert np.isfinite(portfolio_history).all()
assert portfolio_model.config.student_temperature != (
    portfolio_model.config.teacher_temperature
)
assert all(not parameter.requires_grad for parameter in portfolio_model.target.parameters())
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 05 ? MAE: 75%를 가리고 복원하기
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=5,
        slug="mae",
        short_title="MAE: 75%를 가리고 복원하기",
        paper_title="Masked Autoencoders Are Scalable Vision Learners",
        authors=(
            "Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross"
            " Girshick"
        ),
        year=2022,
        primary_url="https://arxiv.org/abs/2111.06377",
        venue="CVPR",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="ViT patch, masking, autoencoder",
        reproduction_goal=(
            "patchify, 75% random mask, visible-only encoder, lightweight decoder"
            "의 masked-pixel MSE를 구현합니다."
        ),
        original_scale=(
            "ImageNet ViT-L/H와 800 epochs 대신 8×8 grayscale·16 patch·작은 attention "
            "decoder를 사용합니다."
        ),
        mappings=(
            (
                "§3 및 Figure 1",
                "image를 non-overlapping patch token으로 변환·복원",
                "patchify/unpatchify round-trip·patch shape",
            ),
            (
                "§3.1 및 Figure 1",
                "random patch masking 75%",
                "visible/masked index",
            ),
            (
                "§3.2",
                "mask token 없는 visible-only encoder",
                "encoder token 수",
            ),
            (
                "§3.3",
                "mask token을 더한 lightweight decoder",
                "원래 patch 순서",
            ),
            (
                "§3.4",
                "masked patch pixel MSE",
                "loss·reconstruction",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{MAE}=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}\left\|x_i-\hat x_i\right\|_2^2$$

- **기호 정의:** M은 가려진 patch index, x_i와 x̂_i는 원본·복원 patch입니다.
- **수식의 역할:** encoder에는 visible patch만 넣고 작은 decoder가 masked patch를 복원합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1 및 Figure 1을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8] → patch [B, 16, 4] → 복원 [B, 16, 4]
- **평가:** masked-only MSE와 원본/복원 시각화
- **원문 대비 한계:** ImageNet ViT-L/H와 800 epochs 대신 8×8 grayscale·16 patch·작은 attention decoder를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{MAE}=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}\left\|x_i-\hat x_i\right\|_2^2$$

- **기호 정의:** M은 가려진 patch index, x_i와 x̂_i는 원본·복원 patch입니다.
- **수식의 역할:** encoder에는 visible patch만 넣고 작은 decoder가 masked patch를 복원합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1 및 Figure 1을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8] → patch [B, 16, 4] → 복원 [B, 16, 4]
- **평가:** masked-only MSE와 원본/복원 시각화
- **원문 대비 한계:** ImageNet ViT-L/H와 800 epochs 대신 8×8 grayscale·16 patch·작은 attention decoder를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 05-1: 8×8 image를 2×2 patch 16개로 바꾸고 역변환하세요.
def patchify(x: Tensor, patch=2):
    raise NotImplementedError("TODO 05-1")


def unpatchify(patches: Tensor, patch=2):
    raise NotImplementedError("TODO 05-1")
""",
                r"""
def patchify(x: Tensor, patch=2):
    return (
        x.unfold(2, patch, patch)
        .unfold(3, patch, patch)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(len(x), -1, patch * patch)
    )


def unpatchify(patches: Tensor, patch=2):
    side = int(patches.shape[1] ** 0.5)
    return (
        patches.reshape(len(patches), side, side, 1, patch, patch)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(len(patches), 1, side * patch, side * patch)
    )


p = patchify(images[:8])
assert p.shape == (8, 16, 4) and torch.allclose(unpatchify(p), images[:8])
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 05-2: sample마다 75% random masked index와 25% visible index를 만드세요.
def random_mask(batch: int, n_patches: int, ratio=0.75):
    raise NotImplementedError("TODO 05-2")
""",
                r"""
def random_mask(batch: int, n_patches: int, ratio=0.75):
    order = torch.stack([torch.randperm(n_patches) for _ in range(batch)])
    keep = int(n_patches * (1 - ratio))
    return order[:, :keep], order[:, keep:]


visible, masked = random_mask(8, 16)
assert visible.shape == (8, 4) and masked.shape == (8, 12)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 05-3: visible-only encoder와 full-token decoder를 정의하세요.
class TinyMAE(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("TODO 05-3")

    def forward(self, patches, visible, masked):
        raise NotImplementedError("TODO 05-3")
""",
                r"""
class TinySelfAttention(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, hidden):
        batch, tokens, dim = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch,
            tokens,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        weights = scores.softmax(dim=-1)
        attended = weights @ value
        merged = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.output(merged), weights


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim=16, heads=2, ff_dim=32):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = TinySelfAttention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, hidden):
        attended, weights = self.attention(self.attention_norm(hidden))
        hidden = hidden + attended
        return hidden + self.ff(self.ff_norm(hidden)), weights


class TinyMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 16)
        self.position = nn.Parameter(torch.randn(1, 16, 16) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 16))
        self.decoder = TinyTransformerBlock(dim=16, heads=2, ff_dim=32)
        self.head = nn.Linear(16, 4)

    def forward(self, patches, visible, masked):
        visible = visible.to(patches.device)
        b = torch.arange(len(patches), device=patches.device)[:, None]
        visible_latent = (
            self.encoder(patches[b, visible]) + self.position[:, visible[0]]
            if torch.equal(visible, visible[:1].expand_as(visible))
            else self.encoder(patches[b, visible])
            + self.position.expand(len(patches), -1, -1)[b, visible]
        )
        full = self.mask_token.expand(len(patches), patches.shape[1], -1).clone()
        full[b, visible] = visible_latent
        decoded, _ = self.decoder(full + self.position)
        prediction = self.head(decoded)
        return prediction, visible_latent


mae = TinyMAE()
patches = patchify(images[:8])
pred, latent = mae(patches, visible, masked)
assert pred.shape == patches.shape and latent.shape[1] == 4
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 05-4: masked patch에만 MSE를 적용하고 복원을 그리세요.
raise NotImplementedError("TODO 05-4")
""",
                r"""
mae = ACCELERATOR.move(mae)
optimizer = torch.optim.Adam(mae.parameters(), lr=0.008)
patches = ACCELERATOR.move(patchify(images[train_idx[:96]]))
losses = []
for _ in range(100):
    visible, masked = random_mask(len(patches), 16)
    visible, masked = ACCELERATOR.move(visible, masked)
    pred, _ = mae(patches, visible, masked)
    b = torch.arange(len(patches), device=DEVICE)[:, None]
    loss = F.mse_loss(pred[b, masked], patches[b, masked])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
visible, masked = random_mask(8, 16)
source = patchify(images[test_idx[:8]])
source_device, visible_device, masked_device = ACCELERATOR.move(source, visible, masked)
pred, _ = mae(source_device, visible_device, masked_device)
recon = source_device.clone()
b = torch.arange(8, device=DEVICE)[:, None]
recon[b, masked_device] = pred[b, masked_device].detach()
recon = recon.cpu()
assert losses[-1] < losses[0]
fig, ax = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    ax[0, i].imshow(images[test_idx[i], 0], cmap="gray")
    ax[1, i].imshow(unpatchify(recon)[i, 0], cmap="gray")
    ax[0, i].axis("off")
    ax[1, i].axis("off")
ax[0, 0].set_ylabel("input")
ax[1, 0].set_ylabel("MAE")
plt.tight_layout()
plt.show()
print(f"masked MSE {losses[0]:.4f} → {losses[-1]:.4f}")
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(
    batch: Tensor,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class MaePortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 20
    mask_ratio: float = 0.75
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(
    batch: Tensor,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    patches = patchify(batch)
    patch_count = patches.shape[1]
    masked_count = round(patch_count * mask_ratio)
    if not 0 < masked_count < patch_count:
        raise ValueError("mask_ratio must leave visible and masked patches")
    noise = torch.rand(
        len(batch),
        patch_count,
        generator=generator,
        device=batch.device,
    )
    mask = noise.argsort(dim=1)[:, :masked_count]
    assert mask.shape == (len(batch), masked_count)
    return patches, mask


class MaePortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(4, config.hidden_dim)
        self.context = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.decoder = nn.Linear(config.hidden_dim, 4)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.position = nn.Parameter(torch.randn(1, 16, config.hidden_dim) * 0.02)

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        visible = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        visible.scatter_(1, mask, False)
        visible_count = patches.shape[1] - mask.shape[1]
        visible_index = visible.float().argsort(dim=1, descending=True)[:, :visible_count]
        patch_index = visible_index.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        visible_patches = patches.gather(1, patch_index)
        position_index = visible_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_dim
        )
        visible_position = self.position.expand(len(patches), -1, -1).gather(
            1, position_index
        )
        visible_hidden = self.encoder(visible_patches) + visible_position

        # MAE §3.3: mask token을 masked 위치에 놓고, visible latent를
        # 원래 patch index에 scatter한 뒤 lightweight decoder로 복원합니다.
        decoder_tokens = self.mask_token.expand(len(patches), patches.shape[1], -1)
        decoder_tokens = decoder_tokens + self.position
        decoder_tokens = decoder_tokens.scatter(1, position_index, visible_hidden)
        visible_context = visible_hidden.mean(dim=1, keepdim=True)
        decoded = F.gelu(self.context(decoder_tokens + visible_context))
        return self.decoder(decoded)

    def compute_loss(self, prediction: Tensor, patches: Tensor, mask: Tensor) -> Tensor:
        batch = torch.arange(len(patches), device=patches.device)[:, None]
        return F.mse_loss(prediction[batch, mask], patches[batch, mask])

    def training_step(self, patches: Tensor, mask: Tensor) -> Tensor:
        prediction = self(patches, mask)
        return self.compute_loss(prediction, patches, mask)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(model: MaePortfolioModel, batch: Tensor) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    generator = torch.Generator(device=batch.device).manual_seed(31)
    audit_generator = torch.Generator(device=batch.device).manual_seed(31)
    history = []
    previous_mask = None
    for _ in range(model.config.steps):
        patches, mask = prepare_views(
            batch,
            model.config.mask_ratio,
            generator,
        )
        _, repeated_mask = prepare_views(
            batch,
            model.config.mask_ratio,
            audit_generator,
        )
        expected_count = round(patches.shape[1] * model.config.mask_ratio)
        assert mask.shape[1] == expected_count
        assert torch.equal(mask, repeated_mask)
        if previous_mask is not None:
            assert not torch.equal(mask, previous_mask)
        previous_mask = mask.clone()
        loss = model.training_step(patches, mask)
        optimizer.zero_grad()
        loss.backward()
        assert model.mask_token.grad is not None
        assert float(model.mask_token.grad.abs().sum()) > 0.0
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: MaePortfolioModel, batch: Tensor
) -> dict[str, float]:
    with torch.no_grad():
        evaluation_generator = torch.Generator(
            device=batch.device,
        ).manual_seed(1031)
        patches, mask = prepare_views(
            batch,
            model.config.mask_ratio,
            evaluation_generator,
        )
        prediction = model(patches, mask)
        masked_mse = model.compute_loss(prediction, patches, mask)
        batch_index = torch.arange(len(patches), device=patches.device)[:, None]
        masked_prediction = prediction[batch_index, mask]
        position_variation = masked_prediction.std(dim=1).mean()
    return {
        "masked_patch_mse": float(masked_mse.cpu()),
        "masked_position_variation": float(position_variation.cpu()),
        "masked_patch_count": float(mask.shape[1]),
    }


portfolio_train_index = train_idx[:64]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_images, portfolio_test_images = ACCELERATOR.move(
    images[portfolio_train_index],
    images[portfolio_test_index],
)
portfolio_model = ACCELERATOR.move(MaePortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(portfolio_model, portfolio_train_images)
portfolio_metrics = evaluate_portfolio_model(portfolio_model, portfolio_test_images)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["masked_position_variation"] > 0.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 06 ? CLIP: 언어로 zero-shot 분류
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=6,
        slug="clip",
        short_title="CLIP: 언어로 zero-shot 분류",
        paper_title=(
            "Learning Transferable Visual Models From Natural Language Supervision"
        ),
        authors="Alec Radford et al.",
        year=2021,
        primary_url="https://arxiv.org/abs/2103.00020",
        venue="ICML",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="dual encoder, contrastive loss, zero-shot classification",
        reproduction_goal=(
            "image/text dual encoder와 symmetric contrastive loss를 학습해 retrieval·z"
            "ero-shot 분류를 수행합니다."
        ),
        original_scale=(
            "4억 web pairs와 ResNet/ViT 대신 240개 합성 pair와 MLP/mean text encoder를 사용합"
            "니다."
        ),
        mappings=(
            (
                "Figure 1, step (1)",
                "image/text dual encoder",
                "L2 norm·similarity matrix",
            ),
            (
                "Figure 3 pseudocode",
                "learned temperature와 대칭 CE",
                "image↔text loss",
            ),
            (
                "Figure 1, step (2)",
                "class text prototype",
                "class×embedding shape",
            ),
            (
                "Figure 1, step (3), §3.1",
                "zero-shot prediction",
                "class prediction accuracy",
            ),
            (
                "Figure 3 pseudocode 및 §2.2",
                "paired image↔text cosine retrieval",
                "retrieval@1·similarity matrix",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L=\tfrac12\left[CE(S,\operatorname{diag})+CE(S^\top,\operatorname{diag})\right],
\quad S=\exp(t)I_eT_e^\top$$

- **기호 정의:** I_e/T_e는 정규화 image/text embedding, t는 logit scale입니다.
- **수식의 역할:** paired image-text를 같은 공간에 맞추는 대칭 dual-encoder 목적입니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 Figure 1, step (1)을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8], token [B, L] → similarity [B, B]
- **평가:** image↔text Recall@1과 class text prototype zero-shot accuracy
- **원문 대비 한계:** 4억 web pairs와 ResNet/ViT 대신 240개 합성 pair와 MLP/mean text encoder를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L=\tfrac12\left[CE(S,\operatorname{diag})+CE(S^\top,\operatorname{diag})\right],
\quad S=\exp(t)I_eT_e^\top$$

- **기호 정의:** I_e/T_e는 정규화 image/text embedding, t는 logit scale입니다.
- **수식의 역할:** paired image-text를 같은 공간에 맞추는 대칭 dual-encoder 목적입니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 Figure 1, step (1)을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8], token [B, L] → similarity [B, B]
- **평가:** image↔text Recall@1과 class text prototype zero-shot accuracy
- **원문 대비 한계:** 4억 web pairs와 ResNet/ViT 대신 240개 합성 pair와 MLP/mean text encoder를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 06-1: image/text를 같은 unit sphere로 보내는 dual encoder를 정의하세요.
class TinyCLIP(nn.Module):
    def __init__(self, vocab, dim=16):
        super().__init__()
        raise NotImplementedError("TODO 06-1")

    def encode_image(self, x):
        raise NotImplementedError("TODO 06-1")

    def encode_text(self, t):
        raise NotImplementedError("TODO 06-1")
""",
                r"""
class TinyCLIP(nn.Module):
    def __init__(self, vocab, dim=16):
        super().__init__()
        self.image = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, dim)
        )
        self.word = nn.Embedding(vocab, 24)
        self.text = nn.Linear(24, dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / 0.2), dtype=torch.float32)
        )

    def encode_image(self, x):
        return F.normalize(self.image(x), dim=1)

    def encode_text(self, t):
        return F.normalize(self.text(self.word(t).mean(1)), dim=1)


clip = TinyCLIP(int(tokens.max()) + 1)
assert torch.allclose(
    clip.encode_image(images[:8]).norm(dim=1), torch.ones(8), atol=1e-5
)
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 06-2: Figure 3의 image→text / text→image CE 평균을 구현하세요.
def clip_loss(model, x, t):
    raise NotImplementedError("TODO 06-2")
""",
                r"""
def clip_loss(model, x, t):
    logits = (
        model.logit_scale.exp().clamp(max=100)
        * model.encode_image(x)
        @ model.encode_text(t).T
    )
    target = torch.arange(len(x), device=logits.device)
    return (
        F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)
    ) / 2, logits


loss, logits = clip_loss(clip, images[:24], tokens[:24])
assert logits.shape == (24, 24) and torch.isfinite(loss)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 06-3: paired batch를 contrastive pre-train하세요.
raise NotImplementedError("TODO 06-3")
""",
                r"""
clip = ACCELERATOR.move(clip)
optimizer = torch.optim.Adam(clip.parameters(), lr=0.012)
losses = []
x, t = ACCELERATOR.move(images[train_idx[:96]], tokens[train_idx[:96]])
for _ in range(140):
    loss, _ = clip_loss(clip, x, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
assert losses[-1] < losses[0] * 0.7
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 06-4: class text prototype으로 zero-shot 분류와 retrieval@1을 평가하세요.
raise NotImplementedError("TODO 06-4")
""",
                r"""
classes = class_ids.unique(sorted=True)
class_tokens = torch.stack([tokens[torch.where(class_ids == c)[0][0]] for c in classes])
with torch.no_grad():
    test_images, test_tokens, class_tokens_device, test_labels = ACCELERATOR.move(
        images[test_idx], tokens[test_idx], class_tokens, class_ids[test_idx]
    )
    zi = clip.encode_image(test_images)
    zt = clip.encode_text(test_tokens)
    retrieval = float(
        (zi @ zt.T)
        .argmax(1)
        .eq(torch.arange(len(test_idx), device=DEVICE))
        .float()
        .mean()
    )
    prototypes = clip.encode_text(class_tokens_device)
    prediction = (zi @ prototypes.T).argmax(1)
    zero_shot = float((prediction == test_labels).float().mean())
assert zero_shot > 1 / len(classes) and retrieval >= 1 / len(test_idx)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("CLIP loss")
ax[1].imshow((zi @ prototypes.T).cpu().numpy(), cmap="magma")
ax[1].set_title("image × class prompt")
plt.tight_layout()
plt.show()
print({"retrieval@1": round(retrieval, 3), "zero_shot_accuracy": round(zero_shot, 3)})
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class ClipPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def encode_image(self, image_batch: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: normalized image encoder")

    def encode_text(self, text_batch: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: normalized text encoder")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    temperature: float = 0.15
    learning_rate: float = 0.02
    steps: int = 40


def prepare_views(
    image_batch: Tensor,
    text_batch: Tensor,
) -> tuple[Tensor, Tensor]:
    return image_batch, text_batch


class ClipPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        vocabulary = int(tokens.max()) + 1
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.hidden_dim),
        )
        self.token_embedding = nn.Embedding(vocabulary, config.hidden_dim)
        self.text_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / config.temperature))
        )

    def encode_image(self, image_batch: Tensor) -> Tensor:
        return F.normalize(self.image_encoder(image_batch), dim=1)

    def encode_text(self, text_batch: Tensor) -> Tensor:
        text_hidden = self.token_embedding(text_batch).mean(dim=1)
        return F.normalize(self.text_projection(text_hidden), dim=1)

    def forward(self, image_batch: Tensor, text_batch: Tensor) -> Tensor:
        image_embedding = self.encode_image(image_batch)
        text_embedding = self.encode_text(text_batch)
        scale = self.logit_scale.clamp(max=4.6052).exp()
        return scale * (image_embedding @ text_embedding.T)

    def compute_loss(self, similarity: Tensor) -> Tensor:
        target = torch.arange(len(similarity), device=similarity.device)
        return 0.5 * (
            F.cross_entropy(similarity, target) + F.cross_entropy(similarity.T, target)
        )

    def training_step(self, image_batch: Tensor, text_batch: Tensor) -> Tensor:
        return self.compute_loss(self(image_batch, text_batch))
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(
    model: ClipPortfolioModel,
    image_batch: Tensor,
    text_batch: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    initial_logit_scale = model.logit_scale.detach().clone()
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(image_batch, text_batch)
        optimizer.zero_grad()
        loss.backward()
        assert model.logit_scale.grad is not None
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=4.6052)
        history.append(float(loss.detach().cpu()))
    assert not torch.allclose(initial_logit_scale, model.logit_scale.detach())
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: ClipPortfolioModel,
    image_batch: Tensor,
    text_batch: Tensor,
    class_text_batch: Tensor,
    image_labels: Tensor,
    class_labels: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        image_embedding = model.encode_image(image_batch)
        text_embedding = model.encode_text(text_batch)
        similarity = image_embedding @ text_embedding.T
        target = torch.arange(len(similarity), device=similarity.device)
        recall = (similarity.argmax(1) == target).float().mean()
        class_prototypes = model.encode_text(class_text_batch)
        class_scores = image_embedding @ class_prototypes.T
        predicted_labels = class_labels[class_scores.argmax(dim=1)]
        zero_shot_accuracy = (predicted_labels == image_labels).float().mean()
    return {
        "heldout_image_to_text_recall1": float(recall.cpu()),
        "heldout_zero_shot_accuracy": float(zero_shot_accuracy.cpu()),
        "learned_temperature": float(model.logit_scale.exp().reciprocal().cpu()),
    }


portfolio_train_index = train_idx[:96]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_class_labels = class_ids[portfolio_train_index].unique(sorted=True)
portfolio_class_tokens = torch.stack(
    [
        tokens[
            portfolio_train_index[
                torch.where(
                    class_ids[portfolio_train_index] == class_label
                )[0][0]
            ]
        ]
        for class_label in portfolio_class_labels
    ]
)
portfolio_train_images, portfolio_train_tokens = prepare_views(
    images[portfolio_train_index],
    tokens[portfolio_train_index],
)
(
    portfolio_train_images,
    portfolio_train_tokens,
    portfolio_test_images,
    portfolio_test_tokens,
    portfolio_class_tokens,
    portfolio_test_labels,
    portfolio_class_labels,
) = ACCELERATOR.move(
    portfolio_train_images,
    portfolio_train_tokens,
    images[portfolio_test_index],
    tokens[portfolio_test_index],
    portfolio_class_tokens,
    class_ids[portfolio_test_index],
    portfolio_class_labels,
)
portfolio_model = ACCELERATOR.move(ClipPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_train_images,
    portfolio_train_tokens,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_test_images,
    portfolio_test_tokens,
    portfolio_class_tokens,
    portfolio_test_labels,
    portfolio_class_labels,
)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["heldout_image_to_text_recall1"] >= 1 / len(test_idx)
assert portfolio_metrics["heldout_zero_shot_accuracy"] > 1 / len(
    portfolio_class_labels
)
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 07 ? ALIGN: noisy alt-text를 규모로 이기기
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=7,
        slug="align",
        short_title="ALIGN: noisy alt-text를 규모로 이기기",
        paper_title=(
            "Scaling Up Visual and Vision-Language Representation Learning With N"
            "oisy Text Supervision"
        ),
        authors="Chao Jia et al.",
        year=2021,
        primary_url="https://arxiv.org/abs/2102.05918",
        venue="ICML",
        difficulty="중급",
        expected_minutes=65,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="CLIP, noisy labels, normalized softmax",
        reproduction_goal=(
            "일부 caption을 의도적으로 오염시키고 ALIGN의 단순 dual-encoder 대칭 normalized-softmax"
            "를 학습합니다."
        ),
        original_scale="18억 noisy image-alt-text pairs 대신 240개 중 30%를 결정적으로 오염시킵니다.",
        mappings=(
            (
                "§2.1",
                "minimal filtering의 noisy pairs",
                "오염 index·비율",
            ),
            (
                "§2.2, Eq. (1)",
                "image-to-text normalized softmax",
                "row CE",
            ),
            (
                "§2.2, Eq. (2)",
                "text-to-image symmetric loss",
                "column CE",
            ),
            (
                "Figure 1",
                "image/text dual encoder",
                "L2-normalized embedding shape",
            ),
            (
                "§3.2 retrieval evaluation",
                "dual-encoder nearest-neighbor retrieval",
                "clean test Recall@1",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L=\tfrac12\left[-\log p(y_i\mid x_i)-\log p(x_i\mid y_i)\right]$$

- **기호 정의:** x_i/y_i는 noisy image-alt-text pair이며 batch의 다른 pair가 negative입니다.
- **수식의 역할:** 복잡한 fusion 없이 normalized-softmax와 데이터 규모로 noisy supervision을 견딥니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §2.1을 구현합니다.
- **입출력 shape:** image feature [B, 64], token [B, L] → normalized embedding과 [B, B] score
- **평가:** clean test Recall@1, pair noise 비율과 shuffled-caption 비교
- **원문 대비 한계:** 18억 noisy image-alt-text pairs 대신 240개 중 30%를 결정적으로 오염시킵니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L=\tfrac12\left[-\log p(y_i\mid x_i)-\log p(x_i\mid y_i)\right]$$

- **기호 정의:** x_i/y_i는 noisy image-alt-text pair이며 batch의 다른 pair가 negative입니다.
- **수식의 역할:** 복잡한 fusion 없이 normalized-softmax와 데이터 규모로 noisy supervision을 견딥니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §2.1을 구현합니다.
- **입출력 shape:** image feature [B, 64], token [B, L] → normalized embedding과 [B, B] score
- **평가:** clean test Recall@1, pair noise 비율과 shuffled-caption 비교
- **원문 대비 한계:** 18억 noisy image-alt-text pairs 대신 240개 중 30%를 결정적으로 오염시킵니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 07-1: 30% caption을 다른 sample caption으로 바꾸는 deterministic noisy corpus를 만드세요.
def noisy_captions(text: Tensor, rate=0.3, seed=5):
    raise NotImplementedError("TODO 07-1")
""",
                r"""
def noisy_captions(text: Tensor, rate=0.3, seed=5):
    rng = np.random.default_rng(seed)
    out = text.clone()
    count = int(len(text) * rate)
    idx = torch.tensor(rng.choice(len(text), count, replace=False))
    out[idx] = out[idx.roll(1)]
    return out, idx


noisy, corrupted = noisy_captions(tokens[train_idx])
assert len(corrupted) == int(len(train_idx) * 0.3) and not torch.equal(
    noisy, tokens[train_idx]
)
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 07-2: Eq. (1)–(2)의 normalized-softmax dual encoder와 loss를 구현하세요.
class TinyALIGN(nn.Module):
    def __init__(self, vocab, dim=16):
        super().__init__()
        raise NotImplementedError("TODO 07-2")

    def forward(self, x, t):
        raise NotImplementedError("TODO 07-2")
""",
                r"""
class TinyALIGN(nn.Module):
    def __init__(self, vocab, dim=16):
        super().__init__()
        self.image = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, dim)
        )
        self.word = nn.Embedding(vocab, 20)
        self.text = nn.Linear(20, dim)
        self.temperature = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, t):
        zi = F.normalize(self.image(x), dim=1)
        zt = F.normalize(self.text(self.word(t).mean(1)), dim=1)
        return zi @ zt.T / self.temperature.clamp(0.03, 1)


align = TinyALIGN(int(tokens.max()) + 1)
logits = align(images[:16], tokens[:16])
target = torch.arange(16)
loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
assert logits.shape == (16, 16) and torch.isfinite(loss)
""",
                "task",
            ),
            # Cell 05 ? metric
            _cell(
                "code",
                r"""
# TODO 07-3: noisy pairs로 학습하고 clean test retrieval을 평가·시각화하세요.
raise NotImplementedError("TODO 07-3")
""",
                r"""
align = ACCELERATOR.move(align)
x, noisy_device = ACCELERATOR.move(images[train_idx], noisy)
optimizer = torch.optim.Adam(align.parameters(), lr=0.012)
losses = []
for _ in range(130):
    logits = align(x, noisy_device)
    target = torch.arange(len(x), device=logits.device)
    loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    test_images, test_tokens = ACCELERATOR.move(images[test_idx], tokens[test_idx])
    test_logits = align(test_images, test_tokens)
    recall = float(
        (
            test_logits.argmax(1)
            == torch.arange(len(test_idx), device=test_logits.device)
        )
        .float()
        .mean()
    )
assert losses[-1] < losses[0] and recall >= 1 / len(test_idx)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("noisy ALIGN loss")
ax[1].imshow(test_logits.detach().cpu(), cmap="viridis")
ax[1].set_title("clean test logits")
plt.tight_layout()
plt.show()
print(
    {
        "noise_rate": len(corrupted) / len(train_idx),
        "clean_retrieval@1": round(recall, 3),
    }
)
""",
                "metric",
            ),
            # Cell 06 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 07 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(
    image_batch: Tensor,
    text_batch: Tensor,
    apply_noise: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class AlignPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    temperature: float = 0.15
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(
    image_batch: Tensor,
    text_batch: Tensor,
    apply_noise: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    if apply_noise:
        prepared_text, corrupted = noisy_captions(
            text_batch,
            rate=0.3,
            seed=5,
        )
    else:
        prepared_text = text_batch
        corrupted = torch.empty(0, dtype=torch.long)
    return image_batch, prepared_text, corrupted


class AlignPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        vocabulary = int(tokens.max()) + 1
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.hidden_dim),
        )
        self.token_embedding = nn.Embedding(vocabulary, config.hidden_dim)
        self.text_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, image_batch: Tensor, text_batch: Tensor) -> Tensor:
        image_embedding = F.normalize(self.image_encoder(image_batch), dim=1)
        text_hidden = self.token_embedding(text_batch).mean(dim=1)
        text_embedding = F.normalize(self.text_projection(text_hidden), dim=1)
        return image_embedding @ text_embedding.T / self.config.temperature

    def compute_loss(self, similarity: Tensor) -> Tensor:
        target = torch.arange(len(similarity), device=similarity.device)
        return 0.5 * (
            F.cross_entropy(similarity, target) + F.cross_entropy(similarity.T, target)
        )

    def training_step(self, image_batch: Tensor, text_batch: Tensor) -> Tensor:
        return self.compute_loss(self(image_batch, text_batch))
""",
                "portfolio-model",
                "task",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(
    model: AlignPortfolioModel,
    image_batch: Tensor,
    text_batch: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(image_batch, text_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 10 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: AlignPortfolioModel,
    image_batch: Tensor,
    clean_text_batch: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        _, noisy_text_batch, corrupted = prepare_views(
            image_batch,
            clean_text_batch,
            apply_noise=True,
        )
        clean_similarity = model(image_batch, clean_text_batch)
        noisy_similarity = model(image_batch, noisy_text_batch)
        target = torch.arange(
            len(clean_similarity),
            device=clean_similarity.device,
        )
        clean_recall = (
            clean_similarity.argmax(1) == target
        ).float().mean()
        noisy_recall = (
            noisy_similarity.argmax(1) == target
        ).float().mean()
    return {
        "clean_image_to_text_recall1": float(clean_recall.cpu()),
        "noisy_image_to_text_recall1": float(noisy_recall.cpu()),
        "caption_noise_rate": len(corrupted) / len(clean_text_batch),
    }


portfolio_train_index = train_idx[:96]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
(
    portfolio_train_images,
    portfolio_noisy_tokens,
    portfolio_corrupted,
) = prepare_views(
    images[portfolio_train_index],
    tokens[portfolio_train_index],
    apply_noise=True,
)
assert len(portfolio_corrupted) == int(len(portfolio_train_index) * 0.3)
portfolio_train_images, portfolio_noisy_tokens = ACCELERATOR.move(
    portfolio_train_images,
    portfolio_noisy_tokens,
)
portfolio_test_images, portfolio_clean_test_tokens = ACCELERATOR.move(
    images[test_idx],
    tokens[test_idx],
)
portfolio_model = ACCELERATOR.move(AlignPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_train_images,
    portfolio_noisy_tokens,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_test_images,
    portfolio_clean_test_tokens,
)
assert np.isfinite(portfolio_history).all()
expected_noise_rate = int(len(test_idx) * 0.3) / len(test_idx)
assert portfolio_metrics["caption_noise_rate"] == expected_noise_rate
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 08 ? BLIP: 이해·생성과 CapFilt
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=8,
        slug="blip",
        short_title="BLIP: 이해·생성과 CapFilt",
        paper_title=(
            "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-L"
            "anguage Understanding and Generation"
        ),
        authors="Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi",
        year=2022,
        primary_url="https://arxiv.org/abs/2201.12086",
        venue="ICML",
        difficulty="고급",
        expected_minutes=85,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="CLIP, image-text matching, language modeling",
        reproduction_goal="MED의 ITC·ITM·LM 세 목적과 CapFilt의 생성/필터 단계를 작은 paired corpus에서 재현합니다.",
        original_scale=(
            "1.29억 web pairs·ViT/BERT 대신 240개 pair와 작은 shared text embedding을 사용합"
            "니다."
        ),
        mappings=(
            (
                "§3.1, Figure 2",
                "ITC·ITM·LM 세 동작 모드",
                "세 loss 유한값",
            ),
            (
                "§3.1 Image-Text Contrastive",
                "unimodal alignment",
                "대칭 CE",
            ),
            (
                "§3.1 Image-Text Matching/LM",
                "negative match·causal token prediction",
                "binary/next-token loss",
            ),
            (
                "§3.2, Figure 3",
                "captioner 생성+filter 제거 CapFilt",
                "정제 전후 noise",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{BLIP}=\mathcal L_{ITC}+\mathcal L_{ITM}+\mathcal L_{LM}$$

- **기호 정의:** ITC는 정렬, ITM은 pair 일치, LM은 다음 token 생성 목적입니다.
- **수식의 역할:** 하나의 모델에 이해와 생성을 결합하고 생성·필터 과정으로 caption을 정제합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1, Figure 2을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8], token [B, L] → contrastive [B, B], LM [B, L-1, V]
- **평가:** 세 loss, retrieval, clean/noisy pair filtering precision
- **원문 대비 한계:** 1.29억 web pairs·ViT/BERT 대신 240개 pair와 작은 shared text embedding을 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$\mathcal L_{BLIP}=\mathcal L_{ITC}+\mathcal L_{ITM}+\mathcal L_{LM}$$

- **기호 정의:** ITC는 정렬, ITM은 pair 일치, LM은 다음 token 생성 목적입니다.
- **수식의 역할:** 하나의 모델에 이해와 생성을 결합하고 생성·필터 과정으로 caption을 정제합니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §3.1, Figure 2을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8], token [B, L] → contrastive [B, B], LM [B, L-1, V]
- **평가:** 세 loss, retrieval, clean/noisy pair filtering precision
- **원문 대비 한계:** 1.29억 web pairs·ViT/BERT 대신 240개 pair와 작은 shared text embedding을 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "code",
                r"""
image_features = torch.tensor(raw["image_features"], dtype=torch.float32)
class_image_features = torch.tensor(raw["class_image_features"], dtype=torch.float32)
class_tokens = torch.stack(
    [tokens[torch.where(class_ids == c)[0][0]] for c in class_ids.unique(sorted=True)]
)
""",
                r"""
image_features = torch.tensor(raw["image_features"], dtype=torch.float32)
class_image_features = torch.tensor(raw["class_image_features"], dtype=torch.float32)
class_tokens = torch.stack(
    [tokens[torch.where(class_ids == c)[0][0]] for c in class_ids.unique(sorted=True)]
)
""",
                "data",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 08-1: 35% noisy web caption과 nearest-prototype captioner를 만드세요.
def make_web_and_synthetic(text, image_feat, rate=0.35):
    raise NotImplementedError("TODO 08-1")
""",
                r"""
def make_web_and_synthetic(text, image_feat, rate=0.35):
    web = text.clone()
    count = int(len(text) * rate)
    bad = torch.arange(count)
    web[bad] = web[bad.roll(1)]
    nearest = (
        F.normalize(image_feat, dim=1) @ F.normalize(class_image_features, dim=1).T
    ).argmax(1)
    synthetic = class_tokens[nearest]
    return web, synthetic, bad


web, synthetic, bad = make_web_and_synthetic(tokens, image_features)
assert len(bad) == int(len(tokens) * 0.35) and synthetic.shape == tokens.shape
""",
                "data",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 08-2: captioner가 생성한 caption과 일치하는 web caption만 남기는 CapFilt를 구현하세요.
def capfilt(web: Tensor, synthetic: Tensor):
    raise NotImplementedError("TODO 08-2")
""",
                r"""
def capfilt(web: Tensor, synthetic: Tensor):
    keep = (web == synthetic).float().mean(1) >= 0.5
    cleaned = web.clone()
    cleaned[~keep] = synthetic[~keep]
    return cleaned, keep


cleaned, keep = capfilt(web, synthetic)
before = float((web != tokens).any(1).float().mean())
after = float((cleaned != tokens).any(1).float().mean())
assert keep.dtype == torch.bool and after < before
""",
                "task",
            ),
            # Cell 06 ? task
            _cell(
                "code",
                r"""
# TODO 08-3: shared text embedding을 쓰는 ITC·ITM·LM 세 loss를 구현하세요.
class TinyBLIP(nn.Module):
    def __init__(self, vocab, dim=16):
        super().__init__()
        raise NotImplementedError("TODO 08-3")

    def losses(self, x, t):
        raise NotImplementedError("TODO 08-3")
""",
                r"""
class TinyBLIP(nn.Module):
    def __init__(self, vocab, dim=16):
        super().__init__()
        self.vocab = vocab
        self.image = nn.Sequential(nn.Flatten(), nn.Linear(64, dim))
        self.word = nn.Embedding(vocab, dim)
        self.itm = nn.Linear(2 * dim, 1)
        self.lm = nn.Linear(2 * dim, vocab)

    def losses(self, x, t):
        zi = F.normalize(self.image(x), dim=1)
        zt = F.normalize(self.word(t).mean(1), dim=1)
        target = torch.arange(len(x), device=x.device)
        sim = zi @ zt.T / 0.15
        itc = (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target)) / 2
        neg = zt.roll(1, 0)
        match_logits = torch.cat(
            [self.itm(torch.cat([zi, zt], 1)), self.itm(torch.cat([zi, neg], 1))]
        )
        itm_target = torch.cat(
            [
                torch.ones(len(x), 1, device=x.device),
                torch.zeros(len(x), 1, device=x.device),
            ]
        )
        itm = F.binary_cross_entropy_with_logits(match_logits, itm_target)
        context = zi[:, None, :].expand(-1, t.shape[1] - 1, -1)
        lm_logits = self.lm(torch.cat([self.word(t[:, :-1]), context], 2))
        lm = F.cross_entropy(lm_logits.reshape(-1, self.vocab), t[:, 1:].reshape(-1))
        return itc, itm, lm, sim


blip = TinyBLIP(int(tokens.max()) + 1)
parts = blip.losses(images[:24], cleaned[:24])
assert all(torch.isfinite(v) for v in parts[:3])
""",
                "task",
            ),
            # Cell 07 ? metric
            _cell(
                "code",
                r"""
# TODO 08-4: 세 목적을 공동 학습하고 정제 전후와 retrieval을 보고하세요.
raise NotImplementedError("TODO 08-4")
""",
                r"""
blip = ACCELERATOR.move(blip)
idx = train_idx[:96]
train_images, train_cleaned = ACCELERATOR.move(images[idx], cleaned[idx])
optimizer = torch.optim.Adam(blip.parameters(), lr=0.012)
losses = []
for _ in range(130):
    itc, itm, lm, _ = blip.losses(train_images, train_cleaned)
    loss = itc + itm + 0.5 * lm
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    test_images, test_cleaned = ACCELERATOR.move(images[test_idx], cleaned[test_idx])
    *_, sim = blip.losses(test_images, test_cleaned)
    retrieval = float(
        (sim.argmax(1) == torch.arange(len(test_idx), device=sim.device)).float().mean()
    )
assert losses[-1] < losses[0] and after < before
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("ITC+ITM+LM")
ax[1].bar(["web", "CapFilt"], [before, after])
ax[1].set_title("caption noise rate")
plt.tight_layout()
plt.show()
print(
    {"noise_before": before, "noise_after": after, "retrieval@1": round(retrieval, 3)}
)
""",
                "metric",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class BlipPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(image_batch: Tensor, text_batch: Tensor) -> tuple[Tensor, Tensor]:
    return image_batch, text_batch


class BlipPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.vocabulary = int(tokens.max()) + 1
        self.image_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(64, config.hidden_dim)
        )
        self.token_embedding = nn.Embedding(self.vocabulary, config.hidden_dim)
        self.matching_head = nn.Linear(config.hidden_dim * 2, 1)
        self.language_model = nn.GRU(
            config.hidden_dim, config.hidden_dim, batch_first=True
        )
        self.token_head = nn.Linear(config.hidden_dim, self.vocabulary)

    def forward(self, image_batch: Tensor, text_batch: Tensor) -> dict[str, Tensor]:
        image_embedding = self.image_encoder(image_batch)
        token_hidden = self.token_embedding(text_batch)
        text_embedding = token_hidden.mean(dim=1)
        contrastive = F.normalize(image_embedding, dim=1)
        contrastive = contrastive @ F.normalize(text_embedding, dim=1).T
        positive_pair = torch.cat([image_embedding, text_embedding], dim=1)
        negative_pair = torch.cat(
            [image_embedding, text_embedding.roll(shifts=1, dims=0)], dim=1
        )
        matching = self.matching_head(torch.cat([positive_pair, negative_pair], dim=0))
        # BLIP §3.1 LM mode: image representation이 decoder initial state를
        # condition하므로 caption token 예측이 image에 따라 달라집니다.
        lm_hidden, _ = self.language_model(
            token_hidden[:, :-1], image_embedding.unsqueeze(0)
        )
        token_logits = self.token_head(lm_hidden)
        return {"contrastive": contrastive, "matching": matching, "lm": token_logits}

    def compute_loss(self, outputs: dict[str, Tensor], text_batch: Tensor) -> Tensor:
        target = torch.arange(len(text_batch), device=text_batch.device)
        contrastive = 0.5 * (
            F.cross_entropy(outputs["contrastive"], target)
            + F.cross_entropy(outputs["contrastive"].T, target)
        )
        matching = F.binary_cross_entropy_with_logits(
            outputs["matching"],
            torch.cat(
                [
                    torch.ones(len(text_batch), 1, device=text_batch.device),
                    torch.zeros(len(text_batch), 1, device=text_batch.device),
                ],
                dim=0,
            ),
        )
        language = F.cross_entropy(
            outputs["lm"].reshape(-1, self.vocabulary),
            text_batch[:, 1:].reshape(-1),
        )
        return contrastive + matching + language

    def training_step(self, image_batch: Tensor, text_batch: Tensor) -> Tensor:
        return self.compute_loss(self(image_batch, text_batch), text_batch)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 10 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(
    model: BlipPortfolioModel,
    image_batch: Tensor,
    text_batch: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(image_batch, text_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 12 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 13 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: BlipPortfolioModel,
    image_batch: Tensor,
    text_batch: Tensor,
    keep_mask: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        outputs = model(image_batch, text_batch)
        similarity = outputs["contrastive"]
        target = torch.arange(len(similarity), device=similarity.device)
        recall = (similarity.argmax(1) == target).float().mean()
        rolled_outputs = model(image_batch.roll(shifts=1, dims=0), text_batch)
        conditioning_gap = (outputs["lm"] - rolled_outputs["lm"]).abs().mean()
        positive_logit, negative_logit = outputs["matching"].chunk(2, dim=0)
        matching_margin = positive_logit.mean() - negative_logit.mean()
        retained_count = int(keep_mask.sum().item())
        replaced_count = int((~keep_mask).sum().item())
    return {
        "retrieval_recall1": float(recall.cpu()),
        "lm_image_conditioning_gap": float(conditioning_gap.cpu()),
        "itm_positive_negative_margin": float(matching_margin.cpu()),
        "capfilt_retained_count": float(retained_count),
        "capfilt_replaced_count": float(replaced_count),
    }


portfolio_train_index = train_idx[:96]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_images, portfolio_cleaned_tokens = prepare_views(
    images[portfolio_train_index],
    cleaned[portfolio_train_index],
)
portfolio_train_images, portfolio_cleaned_tokens = ACCELERATOR.move(
    portfolio_train_images,
    portfolio_cleaned_tokens,
)
(
    portfolio_test_images,
    portfolio_test_cleaned_tokens,
    portfolio_test_keep,
) = ACCELERATOR.move(
    images[portfolio_test_index],
    cleaned[portfolio_test_index],
    keep[portfolio_test_index],
)
portfolio_model = ACCELERATOR.move(BlipPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_train_images,
    portfolio_cleaned_tokens,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_test_images,
    portfolio_test_cleaned_tokens,
    portfolio_test_keep,
)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["lm_image_conditioning_gap"] > 0.0
assert (
    portfolio_metrics["capfilt_retained_count"]
    + portfolio_metrics["capfilt_replaced_count"]
    == len(portfolio_test_index)
)
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 09 ? Flamingo: frozen LM에 시각 cross-attention
    FieldPaperSpec(
        field_id="self_supervised_multimodal",
        field_title="자기지도 학습 · 멀티모달",
        number=9,
        slug="flamingo",
        short_title="Flamingo: frozen LM에 시각 cross-attention",
        paper_title="Flamingo: a Visual Language Model for Few-Shot Learning",
        authors="Jean-Baptiste Alayrac et al.",
        year=2022,
        primary_url="https://arxiv.org/abs/2204.14198",
        venue="NeurIPS",
        difficulty="고급",
        expected_minutes=90,
        dataset_file="data/field_curriculum/multimodal_pairs.npz",
        prerequisites="cross-attention, language modeling, few-shot prompting",
        reproduction_goal=(
            "Perceiver Resampler와 tanh-gated cross-attention으로 image token을 text "
            "stream에 주입합니다."
        ),
        original_scale=(
            "Chinchilla 기반 3B–80B 모델·43M interleaved documents 대신 8×8 image-capti"
            "on pair를 사용합니다."
        ),
        mappings=(
            (
                "§2.1 및 Figure 3",
                "Perceiver Resampler의 고정 visual tokens",
                "latent shape",
            ),
            (
                "§2.2 및 Figure 4",
                "frozen LM 사이 gated cross-attention",
                "gate=0 identity",
            ),
            (
                "Eq. (1)",
                "interleaved next-token likelihood",
                "causal CE",
            ),
            (
                "§2.3",
                "vision/LM freeze, bridge만 학습",
                "trainable parameter audit·conditioning gap",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                r"""
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(17)

DATA_PATH = Path("data/field_curriculum/multimodal_pairs.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
images = torch.tensor(raw["images"], dtype=torch.float32)
tokens = torch.tensor(raw["tokens"], dtype=torch.long)
class_ids = torch.tensor(raw["class_ids"], dtype=torch.long)
sequences = torch.tensor(raw["sequences"], dtype=torch.float32)
train_idx = torch.tensor(raw["train_idx"], dtype=torch.long)
test_idx = torch.tensor(raw["test_idx"], dtype=torch.long)
assert len(images) == len(tokens) == len(class_ids) == len(sequences)
assert images.shape[1:] == (1, 8, 8) and tokens.ndim == 2 and sequences.ndim == 3
print(
    f"pairs={len(images)}, classes={class_ids.unique().numel()}, "
    f"image={tuple(images.shape[1:])}, tokens={tokens.shape[1]}"
)
print(ACCELERATOR.summary())
print("dataset/index preprocessing stays on CPU, while neural training uses DEVICE")
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$p(y_\ell\mid y_{<\ell},x)=\operatorname{softmax}\!\left(W(h_\ell+\tanh(g)
\operatorname{XAttn}(h_\ell,V_x))\right)$$

- **기호 정의:** h는 frozen LM state, V_x는 resampled visual token, g는 학습 gate입니다.
- **수식의 역할:** 고정 LM 사이에 0으로 초기화한 gated cross-attention을 삽입해 시각 조건을 줍니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §2.1 및 Figure 3을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8], token [B, L] → visual latent [B, M, D] → LM logit
- **평가:** next-token CE와 correct/shuffled image conditioning gap
- **원문 대비 한계:** Chinchilla 기반 3B–80B 모델·43M interleaved documents 대신 8×8 image-caption pair를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                r"""
## 포트폴리오 해설: 표현, 목적함수, 업데이트

$$p(y_\ell\mid y_{<\ell},x)=\operatorname{softmax}\!\left(W(h_\ell+\tanh(g)
\operatorname{XAttn}(h_\ell,V_x))\right)$$

- **기호 정의:** h는 frozen LM state, V_x는 resampled visual token, g는 학습 gate입니다.
- **수식의 역할:** 고정 LM 사이에 0으로 초기화한 gated cross-attention을 삽입해 시각 조건을 줍니다.
- **구현 이유:** representation 붕괴·modality 누설·잘못된 positive 연결을
  코드에서 확인할 수 있도록 핵심 수식을 view 준비와 loss 경계로 분리했습니다.
- **Task/코드 대응:** Task P의 `prepare_views`, 논문 전용 `nn.Module`의
  `forward`, `compute_loss`, `training_step`이 §2.1 및 Figure 3을 구현합니다.
- **입출력 shape:** image [B, 1, 8, 8], token [B, L] → visual latent [B, M, D] → LM logit
- **평가:** next-token CE와 correct/shuffled image conditioning gap
- **원문 대비 한계:** Chinchilla 기반 3B–80B 모델·43M interleaved documents 대신 8×8 image-caption pair를 사용합니다.

구현은 `로컬 데이터 통계 → view/modality 준비 → model → objective/update → 평가`
순서입니다. 이 순서는 데이터 누설과 stop-gradient 경계를 코드에서 찾기 쉽게 합니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 09-1: image patch를 고정 개수 latent로 압축하는 Perceiver Resampler를 구현하세요.
class PerceiverResampler(nn.Module):
    def __init__(self, dim=16, n_latents=4):
        super().__init__()
        raise NotImplementedError("TODO 09-1")

    def forward(self, image):
        raise NotImplementedError("TODO 09-1")
""",
                r"""
class ExplicitCrossAttention(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def split_heads(self, tensor):
        batch, tokens, dim = tensor.shape
        return tensor.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, tensor):
        batch, _, tokens, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(
            batch,
            tokens,
            self.heads * self.head_dim,
        )

    def forward(self, query, key, value):
        query_heads = self.split_heads(self.query(query))
        key_heads = self.split_heads(self.key(key))
        value_heads = self.split_heads(self.value(value))
        scores = query_heads @ key_heads.transpose(-2, -1)
        weights = (scores / self.head_dim**0.5).softmax(dim=-1)
        attended = weights @ value_heads
        return self.output(self.merge_heads(attended)), weights


class PerceiverResampler(nn.Module):
    def __init__(self, dim=16, n_latents=4):
        super().__init__()
        self.patch = nn.Linear(4, dim)
        self.latents = nn.Parameter(torch.randn(1, n_latents, dim) * 0.02)
        self.attn = ExplicitCrossAttention(dim, heads=2)

    def forward(self, image):
        patch = (
            image.unfold(2, 2, 2)
            .unfold(3, 2, 2)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(len(image), 16, 4)
        )
        kv = self.patch(patch)
        q = self.latents.expand(len(image), -1, -1)
        return self.attn(q, kv, kv)[0]


resampler = PerceiverResampler()
visual = resampler(images[:8])
assert visual.shape == (8, 4, 16)
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 09-2: Figure 4의 tanh-gated cross-attention residual을 구현하세요.
class GatedCrossAttention(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        raise NotImplementedError("TODO 09-2")

    def forward(self, text, visual):
        raise NotImplementedError("TODO 09-2")
""",
                r"""
class GatedCrossAttention(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attn = ExplicitCrossAttention(dim, heads=2)
        self.gate = nn.Parameter(torch.zeros(()))
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim)
        )
        self.ff_gate = nn.Parameter(torch.zeros(()))

    def forward(self, text, visual):
        attended = self.attn(self.attention_norm(text), visual, visual)[0]
        h = text + torch.tanh(self.gate) * attended
        return h + torch.tanh(self.ff_gate) * self.ff(h)


cross = GatedCrossAttention()
probe = torch.randn(8, 5, 16)
assert torch.allclose(cross(probe, visual), probe, atol=1e-7)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 09-3: frozen token backbone과 trainable visual bridge를 연결해 Eq. (1)을 학습하세요.
raise NotImplementedError("TODO 09-3")
""",
                r"""
vocab = int(tokens.max()) + 1
token_backbone = nn.Embedding(vocab, 16)
token_backbone.weight.requires_grad_(False)
head = nn.Linear(16, vocab)
token_backbone, resampler, cross, head = ACCELERATOR.move(
    token_backbone, resampler, cross, head
)
parameters = [*resampler.parameters(), *cross.parameters(), *head.parameters()]
optimizer = torch.optim.Adam(parameters, lr=0.012)
idx = train_idx[:96]
losses = []
train_images, train_tokens = ACCELERATOR.move(images[idx], tokens[idx])
for _ in range(140):
    t = train_tokens
    text = token_backbone(t[:, :-1])
    visual = resampler(train_images)
    logits = head(cross(text, visual))
    loss = F.cross_entropy(logits.reshape(-1, vocab), t[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
assert (
    losses[-1] < losses[0]
    and token_backbone.weight.grad is None
    and abs(float(torch.tanh(cross.gate))) > 1e-4
)
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 09-4: 올바른 image와 shuffled image의 next-token loss를 비교하세요.
raise NotImplementedError("TODO 09-4")
""",
                r"""
def conditioned_loss(image, text):
    with torch.no_grad():
        h = token_backbone(text[:, :-1])
        logits = head(cross(h, resampler(image)))
        return float(
            F.cross_entropy(logits.reshape(-1, vocab), text[:, 1:].reshape(-1))
            .detach()
            .cpu()
        )


x, t = ACCELERATOR.move(images[test_idx], tokens[test_idx])
correct = conditioned_loss(x, t)
shuffled = conditioned_loss(x.roll(1, 0), t)
assert np.isfinite([correct, shuffled]).all()
plt.plot(losses)
plt.title("Flamingo bridge LM loss")
plt.xlabel("step")
plt.show()
print(
    {
        "correct_image_loss": round(correct, 3),
        "shuffled_image_loss": round(shuffled, 3),
        "visual_conditioning_gap": round(shuffled - correct, 3),
    }
)
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                r"""
### Task P1 · modality/view 준비와 핵심 모델

image·text·sequence의 shape를 먼저 고정하고 augmentation 또는 modality
경계를 명시한 뒤 논문 전용 `nn.Module`을 구성합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 modality 준비, 핵심 module, objective와 평가를 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_views(batch: Tensor) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: modality/view preparation")


class FlamingoPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific modules")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward equation")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update boundary")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    visual_tokens: int = 4
    learning_rate: float = 0.02
    steps: int = 14


def prepare_views(image_batch: Tensor, text_batch: Tensor) -> tuple[Tensor, Tensor]:
    return image_batch, text_batch


class FlamingoPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.vocabulary = int(tokens.max()) + 1
        self.visual_projection = nn.Linear(4, config.hidden_dim)
        self.resampler_queries = nn.Parameter(
            torch.randn(config.visual_tokens, config.hidden_dim) * 0.02
        )
        self.token_embedding = nn.Embedding(self.vocabulary, config.hidden_dim)
        self.language_model = nn.GRU(
            config.hidden_dim, config.hidden_dim, batch_first=True
        )
        self.resampler_attention = ExplicitCrossAttention(
            config.hidden_dim,
            heads=2,
        )
        self.gated_cross_attention = ExplicitCrossAttention(
            config.hidden_dim,
            heads=2,
        )
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.token_head = nn.Linear(config.hidden_dim, self.vocabulary)
        frozen_modules = (
            self.visual_projection,
            self.token_embedding,
            self.language_model,
            self.token_head,
        )
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        trainable_names = [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]
        allowed_prefixes = (
            "resampler_queries",
            "resampler_attention",
            "gated_cross_attention",
            "gate",
        )
        assert trainable_names
        assert all(name.startswith(allowed_prefixes) for name in trainable_names)

    def resample_visual(self, image_batch: Tensor) -> Tensor:
        patch = image_batch.unfold(2, 2, 2).unfold(3, 2, 2)
        patch = patch.permute(0, 2, 3, 1, 4, 5).reshape(len(image_batch), 16, 4)
        visual = self.visual_projection(patch)
        queries = self.resampler_queries[None].expand(len(image_batch), -1, -1)
        latent, _ = self.resampler_attention(queries, visual, visual)
        return latent

    def forward(self, image_batch: Tensor, text_batch: Tensor) -> Tensor:
        visual = self.resample_visual(image_batch)
        token_hidden = self.token_embedding(text_batch[:, :-1])
        language_hidden, _ = self.language_model(token_hidden)
        conditioned, _ = self.gated_cross_attention(
            language_hidden,
            visual,
            visual,
        )
        fused = language_hidden + torch.tanh(self.gate) * conditioned
        return self.token_head(fused)

    def compute_loss(self, logits: Tensor, text_batch: Tensor) -> Tensor:
        return F.cross_entropy(
            logits.reshape(-1, self.vocabulary),
            text_batch[:, 1:].reshape(-1),
        )

    def training_step(self, image_batch: Tensor, text_batch: Tensor) -> Tensor:
        return self.compute_loss(self(image_batch, text_batch), text_batch)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                r"""
### Task P2 · objective, stop-gradient, update

positive/negative, online/target 또는 masked/visible 경계를 확인하면서
`fit_portfolio_model`의 gradient와 EMA update 순서를 추적합니다.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization")
""",
                r"""
def fit_portfolio_model(
    model: FlamingoPortfolioModel,
    image_batch: Tensor,
    text_batch: Tensor,
) -> list[float]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    allowed_prefixes = (
        "resampler_queries",
        "resampler_attention",
        "gated_cross_attention",
        "gate",
    )
    assert trainable
    assert all(name.startswith(allowed_prefixes) for name, _ in trainable)
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in trainable],
        lr=model.config.learning_rate,
    )
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(image_batch, text_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                r"""
### Task P3 · 표현 품질 평가

학습 loss와 retrieval·reconstruction·conditioning 지표를 분리해 계산합니다.
작은 로컬 pair의 수치는 원 논문의 downstream benchmark와 직접 비교하지 않습니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: FlamingoPortfolioModel,
    image_batch: Tensor,
    text_batch: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        correct = model.training_step(image_batch, text_batch)
        shuffled = model.training_step(image_batch.roll(1, 0), text_batch)
    return {
        "language_loss": float(correct.cpu()),
        "conditioning_gap": float((shuffled - correct).cpu()),
    }


portfolio_train_index = train_idx[:48]
portfolio_test_index = test_idx
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_images, portfolio_train_tokens = prepare_views(
    images[portfolio_train_index],
    tokens[portfolio_train_index],
)
portfolio_test_images, portfolio_test_tokens = prepare_views(
    images[portfolio_test_index],
    tokens[portfolio_test_index],
)
(
    portfolio_train_images,
    portfolio_train_tokens,
    portfolio_test_images,
    portfolio_test_tokens,
) = ACCELERATOR.move(
    portfolio_train_images,
    portfolio_train_tokens,
    portfolio_test_images,
    portfolio_test_tokens,
)
portfolio_model = ACCELERATOR.move(FlamingoPortfolioModel(PortfolioConfig()))
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_train_images,
    portfolio_train_tokens,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_test_images,
    portfolio_test_tokens,
)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
)

SPECS = enrich_field_specs(SPECS)

assert len(SPECS) == 10
assert tuple(spec.number for spec in SPECS) == tuple(range(10))
assert len({spec.slug for spec in SPECS}) == 10
