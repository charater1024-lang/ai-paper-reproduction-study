"""Distillation and compression portfolio paper specifications."""

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
    # 00 ? Knowledge Distillation
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=0,
        slug="knowledge_distillation",
        short_title="Knowledge Distillation",
        paper_title="Distilling the Knowledge in a Neural Network",
        authors="Geoffrey Hinton, Oriol Vinyals, Jeff Dean",
        year=2015,
        primary_url=(
            "https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/"
        ),
        venue="NeurIPS 2014 Deep Learning Workshop paper (published 2015)",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal="temperature로 부드러워진 teacher 분포와 hard label을 함께 사용해 작은 student를 학습한다.",
        original_scale=(
            "원 논문은 ensemble과 대형 음성·영상 모델을 다뤘다. 실습은 32차원 합성 분류의 linear student로 da"
            "rk knowledge만 분리해 본다."
        ),
        mappings=(
            (
                "§2, Eq. (1): temperature softmax",
                "Task 1",
                "온도 증가 시 entropy 변화",
            ),
            (
                "§2, Eq. (2)–(4): distillation gradient",
                "Task 2",
                "T²가 포함된 KL loss",
            ),
            (
                "§2: soft targets + correct labels",
                "Task 3",
                "KD+CE student 학습",
            ),
            (
                "§3.1: specialist/teacher transfer",
                "Task 3",
                "teacher agreement와 loss plot",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
정답 class만이 아니라 teacher가 다른 class에 나눠 준 작은 확률, 즉 **dark knowledge**를 student가 배우는지 확인합니다.
""",
                r"""
## 미니 재현
정답 class만이 아니라 teacher가 다른 class에 나눠 준 작은 확률, 즉 **dark knowledge**를 student가 배우는지 확인합니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L=\alpha T^2D_{KL}(p_T^{teacher}\Vert p_T^{student})+(1-\alpha)CE(y,p_1^{student})$$

- **기호 정의:** T는 temperature, α는 soft/hard loss 비율, p_T는 온도 T의 softmax입니다.
- **수식의 역할:** teacher의 class 간 상대 확률인 dark knowledge와 정답 label을 함께 전달합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2, Eq. (1): temperature softmax의 흐름을 명시합니다.
- **입출력 shape:** 특징 [B, 32] → student logit [B, 4], teacher logit [B, 4]
- **평가:** KD loss, test accuracy, teacher/student KL divergence
- **원문 대비 한계:** 원 논문은 ensemble과 대형 음성·영상 모델을 다뤘다. 실습은 32차원 합성 분류의 linear student로 dark
knowledge만 분리해 본다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L=\alpha T^2D_{KL}(p_T^{teacher}\Vert p_T^{student})+(1-\alpha)CE(y,p_1^{student})$$

- **기호 정의:** T는 temperature, α는 soft/hard loss 비율, p_T는 온도 T의 softmax입니다.
- **수식의 역할:** teacher의 class 간 상대 확률인 dark knowledge와 정답 label을 함께 전달합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2, Eq. (1): temperature softmax의 흐름을 명시합니다.
- **입출력 shape:** 특징 [B, 32] → student logit [B, 4], teacher logit [B, 4]
- **평가:** KD loss, test accuracy, teacher/student KL divergence
- **원문 대비 한계:** 원 논문은 ensemble과 대형 음성·영상 모델을 다뤘다. 실습은 32차원 합성 분류의 linear student로 dark
knowledge만 분리해 본다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, paper-equation
            _cell(
                "code",
                r"""
# TODO 1: temperature T를 적용한 soft target을 구현하고 entropy 변화를 확인하세요.
def soft_targets(logits, temperature):
    raise NotImplementedError
""",
                r"""
def soft_targets(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)


def entropy(prob):
    return -(prob * prob.clamp_min(1e-9).log()).sum(-1).mean()


p1, p4 = soft_targets(teacher_logits, 1.0), soft_targets(teacher_logits, 4.0)
assert torch.allclose(p4.sum(-1), torch.ones(len(p4)), atol=1e-6)
assert entropy(p4) > entropy(p1)
print("entropy T=1/T=4:", float(entropy(p1)), float(entropy(p4)))
""",
                "todo",
                "paper-equation",
            ),
            # Cell 05 ? todo, loss
            _cell(
                "code",
                r"""
# TODO 2: Eq. (2)–(4)에 맞춰 T²-scaled KL과 hard CE를 섞은 loss를 구현하세요.
def kd_loss(student, teacher, target, temperature=4.0, alpha=0.7):
    raise NotImplementedError
""",
                r"""
def kd_loss(student, teacher, target, temperature=4.0, alpha=0.7):
    soft = (
        F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
    hard = F.cross_entropy(student, target)
    return alpha * soft + (1 - alpha) * hard


probe = torch.zeros(8, 4, requires_grad=True)
loss_probe = kd_loss(probe, teacher_logits[:8], labels[:8])
loss_probe.backward()
assert torch.isfinite(loss_probe) and probe.grad.abs().sum() > 0
""",
                "todo",
                "loss",
            ),
            # Cell 06 ? todo, training, metric, visualization
            _cell(
                "code",
                r"""
# TODO 3: 작은 student를 학습하고 teacher KL·정확도·loss를 기록하세요.
raise NotImplementedError
""",
                r"""
student = nn.Linear(32, 4)
optimizer = torch.optim.Adam(student.parameters(), lr=0.04)
losses = []
before = F.kl_div(
    F.log_softmax(student(features[test_idx]), -1),
    F.softmax(teacher_logits[test_idx], -1),
    reduction="batchmean",
).item()
for _ in range(60):
    optimizer.zero_grad()
    logits = student(features[train_idx])
    loss = kd_loss(logits, teacher_logits[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    predicted = student(features[test_idx])
    after = F.kl_div(
        F.log_softmax(predicted, -1),
        F.softmax(teacher_logits[test_idx], -1),
        reduction="batchmean",
    ).item()
    accuracy = (predicted.argmax(-1) == labels[test_idx]).float().mean().item()
assert losses[-1] < losses[0] and after < before and accuracy > 0.85
plt.plot(losses)
plt.title(f"KD student: acc={accuracy:.1%}")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()
print(f"teacher KL {before:.3f}→{after:.3f}, accuracy={accuracy:.1%}")
""",
                "todo",
                "training",
                "metric",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class KnowledgeDistillationPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    temperature: float = 3.0
    soft_weight: float = 0.7
    learning_rate: float = 0.03
    steps: int = 18


def prepare_batch() -> tuple[Tensor, Tensor, Tensor]:
    return features[train_idx], labels[train_idx], teacher_logits[train_idx]


class KnowledgeDistillationPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.student = nn.Sequential(
            nn.Linear(features.shape[1], config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, teacher_logits.shape[1]),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.student(inputs)

    def compute_loss(
        self,
        student_logits: Tensor,
        targets: Tensor,
        teacher: Tensor,
    ) -> Tensor:
        temperature = self.config.temperature
        soft = (
            F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher / temperature, dim=1),
                reduction="batchmean",
            )
            * temperature**2
        )
        hard = F.cross_entropy(student_logits, targets)
        weight = self.config.soft_weight
        return weight * soft + (1.0 - weight) * hard

    def training_step(self, inputs: Tensor, targets: Tensor, teacher: Tensor) -> Tensor:
        return self.compute_loss(self(inputs), targets, teacher)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: KnowledgeDistillationPortfolioModel) -> list[float]:
    inputs, targets, teacher = prepare_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, targets, teacher)
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
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(
    model: KnowledgeDistillationPortfolioModel,
) -> dict[str, float]:
    with torch.no_grad():
        logits = model(features[test_idx])
        accuracy = (logits.argmax(1) == labels[test_idx]).float().mean()
        divergence = F.kl_div(
            F.log_softmax(logits, dim=1),
            F.softmax(teacher_logits[test_idx], dim=1),
            reduction="batchmean",
        )
    return {
        "test_accuracy": float(accuracy.cpu()),
        "teacher_kl": float(divergence.cpu()),
    }


portfolio_model = KnowledgeDistillationPortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 01 ? FitNets
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=1,
        slug="fitnets",
        short_title="FitNets",
        paper_title="FitNets: Hints for Thin Deep Nets",
        authors="Adriana Romero et al.",
        year=2015,
        primary_url="https://arxiv.org/abs/1412.6550",
        venue="ICLR 2015",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal="teacher 중간 표현을 regressor로 맞춘 뒤 출력 distillation을 수행하는 2단계 학습을 구현한다.",
        original_scale=(
            "원 논문은 CIFAR/SVHN의 깊고 얇은 CNN을 사용한다. 실습은 32차원 MLP hidden hint로 같은 훈련 순"
            "서를 검증한다."
        ),
        mappings=(
            (
                "§3.1, Eq. (2): hint-based training",
                "Task 1–2",
                "guided→hint regressor shape/MSE",
            ),
            (
                "Figure 1: two-stage training",
                "Task 2–3",
                "hint 선학습 뒤 같은 sequence의 teacher logit으로 분류",
            ),
            (
                "§3.2, Eq. (3): knowledge distillation",
                "Task 3",
                "soft output loss",
            ),
            (
                "§4: thin/deep student experiments",
                "Task 3",
                "loss와 accuracy plot",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
최종 출력만 흉내 내기 어려운 얇은 student에게 teacher의 **중간 표현을 먼저 힌트**로 줍니다.
""",
                r"""
## 미니 재현
최종 출력만 흉내 내기 어려운 얇은 student에게 teacher의 **중간 표현을 먼저 힌트**로 줍니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L_{hint}=\frac12\left\|u_h(\mathbf x,W_{Hint})-r(v_g(\mathbf x,W_{Guided}),W_r)
\right\|_2^2$$

- **기호 정의:** u_h는 teacher hint, v_g는 student guided layer, r은 shape regressor입니다.
- **수식의 역할:** 얇은 student가 최종 logit 전에 teacher의 중간 표현을 따라가도록 단계별로 지도합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §3.1, Eq. (2): hint-based training의 흐름을 명시합니다.
- **입출력 shape:** token embedding [B, 6, 16] → guided [B, 6, H] → regressed hint [B, 6, 24]
- **평가:** hint MSE, task CE, test accuracy
- **원문 대비 한계:** 원 논문은 CIFAR/SVHN의 깊고 얇은 CNN을 사용한다. 실습은 32차원 MLP hidden hint로 같은 훈련 순서를 검증한다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L_{hint}=\frac12\left\|u_h(\mathbf x,W_{Hint})-r(v_g(\mathbf x,W_{Guided}),W_r)
\right\|_2^2$$

- **기호 정의:** u_h는 teacher hint, v_g는 student guided layer, r은 shape regressor입니다.
- **수식의 역할:** 얇은 student가 최종 logit 전에 teacher의 중간 표현을 따라가도록 단계별로 지도합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §3.1, Eq. (2): hint-based training의 흐름을 명시합니다.
- **입출력 shape:** token embedding [B, 6, 16] → guided [B, 6, H] → regressed hint [B, 6, 24]
- **평가:** hint MSE, task CE, test accuracy
- **원문 대비 한계:** 원 논문은 CIFAR/SVHN의 깊고 얇은 CNN을 사용한다. 실습은 32차원 MLP hidden hint로 같은 훈련 순서를 검증한다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, architecture
            _cell(
                "code",
                r"""
# TODO 1: 12차원 guided layer와 24차원 teacher hint를 잇는 regressor를 구성하세요.
raise NotImplementedError
""",
                r"""
class FitStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.guided = nn.Linear(32, 12)
        self.regressor = nn.Linear(12, 24)
        self.head = nn.Linear(12, 4)

    def forward(self, x):
        h = F.relu(self.guided(x))
        return self.head(h), self.regressor(h)


student = FitStudent()
teacher_hint = torch.cat([features[:, :16], features[:, 16:24].square()], dim=1)
logits, hint = student(features[:5])
assert logits.shape == (5, 4) and hint.shape == teacher_hint[:5].shape
""",
                "todo",
                "architecture",
            ),
            # Cell 05 ? todo, hint-training
            _cell(
                "code",
                r"""
# TODO 2: regressor와 guided layer만 사용해 hint reconstruction을 먼저 학습하세요.
raise NotImplementedError
""",
                r"""
hint_optimizer = torch.optim.Adam(
    list(student.guided.parameters()) + list(student.regressor.parameters()), lr=0.025
)
hint_losses = []
for _ in range(50):
    hint_optimizer.zero_grad()
    _, predicted_hint = student(features[train_idx])
    loss = F.mse_loss(predicted_hint, teacher_hint[train_idx])
    loss.backward()
    hint_optimizer.step()
    hint_losses.append(float(loss.detach()))
assert hint_losses[-1] < hint_losses[0] * 0.55
guided_before = student.guided.weight.detach().clone()
""",
                "todo",
                "hint-training",
            ),
            # Cell 06 ? todo, training, visualization
            _cell(
                "code",
                r"""
# TODO 3: hint 단계 뒤 출력 KD+CE로 전체 student를 미세조정하세요.
raise NotImplementedError
""",
                r"""
optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
task_losses = []
for _ in range(50):
    optimizer.zero_grad()
    logits, _ = student(features[train_idx])
    loss = 0.65 * F.kl_div(
        F.log_softmax(logits / 3, -1),
        F.softmax(teacher_logits[train_idx] / 3, -1),
        reduction="batchmean",
    ) * 9 + 0.35 * F.cross_entropy(logits, labels[train_idx])
    loss.backward()
    optimizer.step()
    task_losses.append(float(loss.detach()))
with torch.no_grad():
    accuracy = (
        (student(features[test_idx])[0].argmax(-1) == labels[test_idx])
        .float()
        .mean()
        .item()
    )
assert (
    task_losses[-1] < task_losses[0]
    and accuracy > 0.8
    and not torch.equal(student.guided.weight, guided_before)
)
plt.plot(hint_losses, label="hint")
plt.plot(task_losses, label="task KD")
plt.legend()
plt.show()
print(f"student accuracy={accuracy:.1%}")
""",
                "todo",
                "training",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class FitnetsPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    hint_weight: float = 1.0
    temperature: float = 2.0
    learning_rate: float = 0.025
    hint_steps: int = 8
    distillation_steps: int = 8


def prepare_batch() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    inputs = token_embeddings[:192]
    targets, predictions = sequence_teacher_targets(inputs)
    hints = teacher_hidden[:192]
    return inputs, targets, hints, predictions


def sequence_teacher_targets(inputs: Tensor) -> tuple[Tensor, Tensor]:
    '''Create labels and soft logits from the same sequence seen by the student.'''
    class_count = teacher_logits.shape[-1]
    pooled = inputs.mean(dim=1)
    predictions = 4.0 * pooled[:, :class_count]
    return predictions.argmax(dim=1), predictions


class FitnetsPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.guided = nn.Linear(token_embeddings.shape[-1], config.hidden_dim)
        self.regressor = nn.Linear(config.hidden_dim, teacher_hidden.shape[-1])
        self.classifier = nn.Linear(config.hidden_dim, teacher_logits.shape[-1])

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        guided = F.relu(self.guided(inputs))
        regressed_hint = self.regressor(guided)
        logits = self.classifier(guided.mean(dim=1))
        return logits, regressed_hint

    def compute_hint_loss(
        self,
        outputs: tuple[Tensor, Tensor],
        teacher_hint: Tensor,
    ) -> Tensor:
        _, regressed_hint = outputs
        return self.config.hint_weight * F.mse_loss(regressed_hint, teacher_hint)

    def compute_distillation_loss(
        self,
        outputs: tuple[Tensor, Tensor],
        targets: Tensor,
        teacher_prediction: Tensor,
    ) -> Tensor:
        logits, _ = outputs
        temperature = self.config.temperature
        soft_targets = torch.softmax(teacher_prediction / temperature, dim=1)
        soft_loss = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            soft_targets,
            reduction="batchmean",
        )
        return F.cross_entropy(logits, targets) + temperature**2 * soft_loss

    def training_step(
        self,
        inputs: Tensor,
        target: Tensor,
        teacher_target: Tensor,
        phase: str,
    ) -> Tensor:
        outputs = self(inputs)
        if phase == "hint":
            return self.compute_hint_loss(outputs, teacher_target)
        if phase == "distillation":
            return self.compute_distillation_loss(outputs, target, teacher_target)
        raise ValueError(f"unknown FitNets phase: {phase}")
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: FitnetsPortfolioModel) -> list[float]:
    inputs, targets, hints, teacher_predictions = prepare_batch()
    hint_parameters = list(model.guided.parameters()) + list(model.regressor.parameters())
    hint_optimizer = torch.optim.Adam(
        hint_parameters,
        lr=model.config.learning_rate,
    )
    history = []
    for _ in range(model.config.hint_steps):
        loss = model.training_step(inputs, targets, hints, "hint")
        hint_optimizer.zero_grad()
        loss.backward()
        hint_optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.regressor.requires_grad_(False)
    distillation_optimizer = torch.optim.Adam(
        list(model.guided.parameters()) + list(model.classifier.parameters()),
        lr=model.config.learning_rate,
    )
    for _ in range(model.config.distillation_steps):
        loss = model.training_step(
            inputs,
            targets,
            teacher_predictions,
            "distillation",
        )
        distillation_optimizer.zero_grad()
        loss.backward()
        distillation_optimizer.step()
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
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(model: FitnetsPortfolioModel) -> dict[str, float]:
    with torch.no_grad():
        test_inputs = token_embeddings[192:256]
        test_targets, teacher_predictions = sequence_teacher_targets(test_inputs)
        logits, hint = model(test_inputs)
        accuracy = (logits.argmax(1) == test_targets).float().mean()
        hint_mse = F.mse_loss(hint, teacher_hidden[192:256])
        temperature = model.config.temperature
        teacher_probability = torch.softmax(
            teacher_predictions / temperature,
            dim=1,
        )
        prediction_kl = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            teacher_probability,
            reduction="batchmean",
        )
    return {
        "test_accuracy": float(accuracy.cpu()),
        "hint_mse": float(hint_mse.cpu()),
        "teacher_student_kl": float(prediction_kl.cpu()),
        "two_stage_updates": float(
            model.config.hint_steps + model.config.distillation_steps
        ),
    }


portfolio_model = FitnetsPortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["test_accuracy"] > 0.35
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 02 ? Attention Transfer
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=2,
        slug="attention_transfer",
        short_title="Attention Transfer",
        paper_title=(
            "Paying More Attention to Attention: Improving the Performance of Con"
            "volutional Neural Networks via Attention Transfer"
        ),
        authors="Sergey Zagoruyko, Nikos Komodakis",
        year=2017,
        primary_url="https://openreview.net/forum?id=Sks9_ajex",
        venue="ICLR 2017",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "CNN feature를 spatial attention map으로 요약하고 teacher와 student의 정규화된 map"
            "을 맞춘다."
        ),
        original_scale=(
            "원 논문은 Wide ResNet teacher/student와 CIFAR/ImageNet을 사용한다. 실습은 16×16 합"
            "성 영상의 작은 CNN이다."
        ),
        mappings=(
            (
                "§2.1, Eq. (1)–(2): activation attention",
                "Task 1",
                "channel power 합",
            ),
            (
                "§2.1, Eq. (3): vector normalization",
                "Task 1",
                "scale invariance",
            ),
            (
                "§2.2, Eq. (4): attention transfer loss",
                "Task 2",
                "teacher/student map distance",
            ),
            (
                "§3: classification + AT experiments",
                "Task 3",
                "joint loss와 accuracy",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
채널 수가 달라도 feature가 **어디를 보고 있는지**를 2차원 attention map으로 비교할 수 있습니다.
""",
                r"""
## 미니 재현
채널 수가 달라도 feature가 **어디를 보고 있는지**를 2차원 attention map으로 비교할 수 있습니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L_{AT}=\frac{\beta}{2}\sum_j\left\|\frac{Q_S^j}{\|Q_S^j\|_2}-\frac{Q_T^j}{\|Q_T^j
\|_2}\right\|_2^2$$

- **기호 정의:** Q는 channel activation을 합한 spatial attention map, β는 전달 강도입니다.
- **수식의 역할:** 채널 수가 다른 teacher/student도 어디를 보는지라는 공간 정보로 정렬합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2.1, Eq. (1)–(2): activation attention의 흐름을 명시합니다.
- **입출력 shape:** image [B, 1, 16, 16] → feature [B, C, H, W] → attention [B, H·W]
- **평가:** 정규화 attention MSE와 분류 accuracy
- **원문 대비 한계:** 원 논문은 Wide ResNet teacher/student와 CIFAR/ImageNet을 사용한다. 실습은 16×16 합성 영상의 작은 CNN이다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L_{AT}=\frac{\beta}{2}\sum_j\left\|\frac{Q_S^j}{\|Q_S^j\|_2}-\frac{Q_T^j}{\|Q_T^j
\|_2}\right\|_2^2$$

- **기호 정의:** Q는 channel activation을 합한 spatial attention map, β는 전달 강도입니다.
- **수식의 역할:** 채널 수가 다른 teacher/student도 어디를 보는지라는 공간 정보로 정렬합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2.1, Eq. (1)–(2): activation attention의 흐름을 명시합니다.
- **입출력 shape:** image [B, 1, 16, 16] → feature [B, C, H, W] → attention [B, H·W]
- **평가:** 정규화 attention MSE와 분류 accuracy
- **원문 대비 한계:** 원 논문은 Wide ResNet teacher/student와 CIFAR/ImageNet을 사용한다. 실습은 16×16 합성 영상의 작은 CNN이다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, paper-equation
            _cell(
                "code",
                r"""
# TODO 1: Eq. (2)–(3)의 channel-power attention과 L2 normalization을 구현하세요.
def attention_map(feature, power=2):
    raise NotImplementedError
""",
                r"""
def attention_map(feature, power=2):
    spatial = feature.abs().pow(power).sum(1).flatten(1)
    return F.normalize(spatial, p=2, dim=1)


probe = torch.rand(4, 5, 8, 8)
assert attention_map(probe).shape == (4, 64)
assert torch.allclose(attention_map(probe), attention_map(probe * 7), atol=1e-5)
""",
                "todo",
                "paper-equation",
            ),
            # Cell 05 ? todo, attention-loss
            _cell(
                "code",
                r"""
# TODO 2: 고정 teacher feature와 student feature의 attention distance를 계산하세요.
raise NotImplementedError
""",
                r"""
smooth = F.avg_pool2d(images, 3, 1, 1)
edge_x = F.pad(images[:, :, :, 1:] - images[:, :, :, :-1], (0, 1, 0, 0))
teacher_feature = torch.cat([images, smooth, edge_x], 1)


class ATStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, padding=1)
        self.head = nn.Linear(6 * 16 * 16, 4)

    def forward(self, x):
        feature = F.relu(self.conv(x))
        return self.head(feature.flatten(1)), feature


student = ATStudent()
_, student_feature = student(images[:8])
at_loss = F.mse_loss(attention_map(student_feature), attention_map(teacher_feature[:8]))
assert torch.isfinite(at_loss) and at_loss > 0
""",
                "todo",
                "attention-loss",
            ),
            # Cell 06 ? todo, training, visualization
            _cell(
                "code",
                r"""
# TODO 3: classification+attention loss로 student를 학습하고 두 loss를 그리세요.
raise NotImplementedError
""",
                r"""
optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
losses = []
attentions = []
for _ in range(45):
    optimizer.zero_grad()
    logits, feature = student(images[image_train_idx])
    ce = F.cross_entropy(logits, image_labels[image_train_idx])
    at = F.mse_loss(
        attention_map(feature), attention_map(teacher_feature[image_train_idx])
    )
    loss = ce + 25 * at
    loss.backward()
    optimizer.step()
    losses.append(float(ce.detach()))
    attentions.append(float(at.detach()))
with torch.no_grad():
    accuracy = (
        (student(images[image_test_idx])[0].argmax(-1) == image_labels[image_test_idx])
        .float()
        .mean()
        .item()
    )
assert losses[-1] < losses[0] and attentions[-1] < attentions[0] and accuracy > 0.7
plt.plot(losses, label="CE")
plt.plot(attentions, label="AT")
plt.legend()
plt.show()
print(f"accuracy={accuracy:.1%}")
""",
                "todo",
                "training",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class AttentionTransferPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_channels: int = 8
    teacher_channels: int = 16
    attention_weight: float = 2.0
    learning_rate: float = 0.02
    teacher_steps: int = 8
    steps: int = 16


def prepare_batch() -> tuple[Tensor, Tensor]:
    batch = images[image_train_idx[:128]]
    target = image_labels[image_train_idx[:128]]
    return batch, target


class AttentionTeacher(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(channels, 4)
        self.requires_grad_(False)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        feature_map = self.features(inputs)
        logits = self.classifier(feature_map.mean(dim=(2, 3)))
        return logits, feature_map


class AttentionTransferPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.teacher = AttentionTeacher(config.teacher_channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, config.hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(config.hidden_channels, 4)

    @staticmethod
    def spatial_attention(feature_map: Tensor) -> Tensor:
        attention = feature_map.square().sum(dim=1).flatten(1)
        return F.normalize(attention, dim=1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        feature_map = self.features(inputs)
        logits = self.classifier(feature_map.mean(dim=(2, 3)))
        student_map = self.spatial_attention(feature_map)
        with torch.no_grad():
            _, teacher_feature = self.teacher(inputs)
            teacher_map = self.spatial_attention(teacher_feature)
        return logits, student_map, teacher_map

    def compute_loss(
        self,
        outputs: tuple[Tensor, Tensor, Tensor],
        targets: Tensor,
    ) -> Tensor:
        logits, student_map, teacher_map = outputs
        classification = F.cross_entropy(logits, targets)
        transfer = F.mse_loss(student_map, teacher_map)
        return classification + self.config.attention_weight * transfer

    def training_step(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return self.compute_loss(self(inputs), targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: AttentionTransferPortfolioModel) -> list[float]:
    inputs, targets = prepare_batch()
    model.teacher.requires_grad_(True)
    teacher_optimizer = torch.optim.Adam(
        model.teacher.parameters(),
        lr=model.config.learning_rate,
    )
    history = []
    for _ in range(model.config.teacher_steps):
        teacher_logits_local, _ = model.teacher(inputs)
        teacher_loss = F.cross_entropy(teacher_logits_local, targets)
        teacher_optimizer.zero_grad()
        teacher_loss.backward()
        teacher_optimizer.step()
        history.append(float(teacher_loss.detach().cpu()))
    model.teacher.requires_grad_(False)
    frozen_before = [parameter.clone() for parameter in model.teacher.parameters()]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=model.config.learning_rate)
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    frozen_after = list(model.teacher.parameters())
    assert all(torch.equal(before, after) for before, after in zip(frozen_before, frozen_after))
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(
    model: AttentionTransferPortfolioModel,
) -> dict[str, float]:
    with torch.no_grad():
        logits, student_map, teacher_map = model(images[image_test_idx])
        accuracy = (logits.argmax(1) == image_labels[image_test_idx]).float().mean()
        transfer_error = F.mse_loss(student_map, teacher_map)
    teacher_trainable = sum(parameter.requires_grad for parameter in model.teacher.parameters())
    return {
        "test_accuracy": float(accuracy.cpu()),
        "teacher_student_attention_mse": float(transfer_error.cpu()),
        "teacher_trainable_tensors": float(teacher_trainable),
    }


portfolio_model = AttentionTransferPortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 03 ? TinyBERT
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=3,
        slug="tinybert",
        short_title="TinyBERT",
        paper_title="TinyBERT: Distilling BERT for Natural Language Understanding",
        authors="Xiaoqi Jiao et al.",
        year=2020,
        primary_url="https://aclanthology.org/2020.findings-emnlp.372/",
        venue="Findings of EMNLP 2020",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "Transformer student가 teacher의 hidden state, attention matrix, predic"
            "tion을 함께 모방하도록 학습한다."
        ),
        original_scale=(
            "원 논문은 BERT-Base를 4/6-layer TinyBERT로 2단계 증류한다. 실습은 저장된 6-token teach"
            "er tensor와 작은 projection이다."
        ),
        mappings=(
            (
                "Figure 1: general/task-specific distillation",
                "Task 3",
                "같은 sequence target으로 general/task-specific 2단계 loss",
            ),
            (
                "§3.1, Eq. (4)–(5): embedding/hidden loss",
                "Task 1",
                "regressor MSE",
            ),
            (
                "§3.1, Eq. (6): attention loss",
                "Task 2",
                "attention matrix MSE",
            ),
            (
                "§3.1, Eq. (7)–(8): prediction/total loss",
                "Task 3",
                "loss 감소",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
TinyBERT는 logits 하나만 복사하지 않고 Transformer의 hidden과 attention까지 여러 층에서 맞춥니다.
""",
                r"""
## 미니 재현
TinyBERT는 logits 하나만 복사하지 않고 Transformer의 hidden과 attention까지 여러 층에서 맞춥니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L_{model}=\sum_m(\mathcal L_{attn}^m+\mathcal L_{hidn}^m)+\mathcal L_{embd}+
\mathcal L_{pred}$$

- **기호 정의:** m은 student layer, L_attn/L_hidn/L_pred는 attention·hidden·prediction 손실입니다.
- **수식의 역할:** Transformer 내부의 여러 수준을 동시에 맞춰 작은 BERT의 표현 경로를 보존합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  Figure 1: general/task-specific distillation의 흐름을 명시합니다.
- **입출력 shape:** token embedding [B, 6, 16] → hidden [B, 6, 24], attention [B, 6, 6]
- **평가:** hidden/attention/logit loss와 분류 accuracy
- **원문 대비 한계:** 원 논문은 BERT-Base를 4/6-layer TinyBERT로 2단계 증류한다. 실습은 저장된 6-token teacher
tensor와 작은 projection이다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\mathcal L_{model}=\sum_m(\mathcal L_{attn}^m+\mathcal L_{hidn}^m)+\mathcal L_{embd}+
\mathcal L_{pred}$$

- **기호 정의:** m은 student layer, L_attn/L_hidn/L_pred는 attention·hidden·prediction 손실입니다.
- **수식의 역할:** Transformer 내부의 여러 수준을 동시에 맞춰 작은 BERT의 표현 경로를 보존합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  Figure 1: general/task-specific distillation의 흐름을 명시합니다.
- **입출력 shape:** token embedding [B, 6, 16] → hidden [B, 6, 24], attention [B, 6, 6]
- **평가:** hidden/attention/logit loss와 분류 accuracy
- **원문 대비 한계:** 원 논문은 BERT-Base를 4/6-layer TinyBERT로 2단계 증류한다. 실습은 저장된 6-token teacher
tensor와 작은 projection이다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, hidden-distillation
            _cell(
                "code",
                r"""
# TODO 1: 16→12 student hidden과 12→24 regressor를 만들고 teacher hidden loss를 정의하세요.
raise NotImplementedError
""",
                r"""
class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16, 12)
        self.regressor = nn.Linear(12, 24)
        self.q = nn.Linear(12, 12)
        self.k = nn.Linear(12, 12)
        self.head = nn.Linear(12, 4)

    def forward(self, x):
        hidden = torch.tanh(self.embed(x))
        q, k = self.q(hidden), self.k(hidden)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(12), -1)
        return hidden, self.regressor(hidden), attention, self.head(hidden.mean(1))


student = TinyStudent()
hidden, regressed, _, _ = student(token_embeddings[:4])
hidden_loss = F.mse_loss(regressed, teacher_hidden[:4])
assert hidden.shape == (4, 6, 12) and regressed.shape == (4, 6, 24) and hidden_loss > 0
""",
                "todo",
                "hidden-distillation",
            ),
            # Cell 05 ? todo, attention-distillation
            _cell(
                "code",
                r"""
# TODO 2: student attention과 teacher attention의 Eq. (6) MSE를 계산하세요.
raise NotImplementedError
""",
                r"""
_, _, student_attention, _ = student(token_embeddings[:8])
attention_loss = F.mse_loss(student_attention, teacher_attention[:8])
assert student_attention.shape == (8, 6, 6)
assert torch.allclose(student_attention.sum(-1), torch.ones(8, 6), atol=1e-6)
assert attention_loss > 0
""",
                "todo",
                "attention-distillation",
            ),
            # Cell 06 ? todo, training, visualization
            _cell(
                "code",
                r"""
# TODO 3: hidden+attention+prediction loss로 student를 학습하세요.
raise NotImplementedError
""",
                r"""
sequence_labels = token_embeddings.mean(1)[:, :4].argmax(-1)
teacher_prediction = F.one_hot(sequence_labels, 4).float() * 4
optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
losses = []
for _ in range(55):
    optimizer.zero_grad()
    _, regressed, attention, prediction = student(token_embeddings)
    loss = (
        F.mse_loss(regressed, teacher_hidden)
        + 8 * F.mse_loss(attention, teacher_attention)
        + F.kl_div(
            F.log_softmax(prediction / 2, -1),
            F.softmax(teacher_prediction / 2, -1),
            reduction="batchmean",
        )
        * 4
    )
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    accuracy = (
        (student(token_embeddings)[3].argmax(-1) == sequence_labels)
        .float()
        .mean()
        .item()
    )
assert losses[-1] < losses[0] * 0.55 and accuracy > 0.65
plt.plot(losses)
plt.title(f"TinyBERT tensor distillation, acc={accuracy:.1%}")
plt.show()
print(f"loss {losses[0]:.3f}→{losses[-1]:.3f}")
""",
                "todo",
                "training",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class TinybertPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    temperature: float = 2.0
    learning_rate: float = 0.02
    general_steps: int = 8
    task_steps: int = 8


def sequence_teacher_targets(inputs: Tensor) -> tuple[Tensor, Tensor]:
    '''Derive task labels and dark-knowledge logits from each input sequence.'''
    class_count = teacher_logits.shape[-1]
    pooled = inputs.mean(dim=1)
    predictions = 4.0 * pooled[:, :class_count]
    return predictions.argmax(dim=1), predictions


def prepare_batch() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    inputs = token_embeddings[:192]
    targets, prediction_targets = sequence_teacher_targets(inputs)
    return (
        inputs,
        targets,
        prediction_targets,
        teacher_hidden[:192],
        teacher_attention[:192],
    )


class TinybertPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        input_dim = token_embeddings.shape[-1]
        self.query = nn.Linear(input_dim, config.hidden_dim)
        self.key = nn.Linear(input_dim, config.hidden_dim)
        self.value = nn.Linear(input_dim, config.hidden_dim)
        self.hidden_regressor = nn.Linear(config.hidden_dim, teacher_hidden.shape[-1])
        self.classifier = nn.Linear(config.hidden_dim, teacher_logits.shape[-1])

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        scale = self.config.hidden_dim**0.5
        attention = torch.softmax(query @ key.transpose(1, 2) / scale, dim=-1)
        hidden = attention @ value
        regressed = self.hidden_regressor(hidden)
        logits = self.classifier(hidden.mean(dim=1))
        return logits, regressed, attention

    def compute_loss(
        self,
        outputs: tuple[Tensor, Tensor, Tensor],
        targets: Tensor,
        prediction_target: Tensor,
        hidden_target: Tensor,
        attention_target: Tensor,
        phase: str,
    ) -> Tensor:
        logits, hidden, attention = outputs
        hidden_loss = F.mse_loss(hidden, hidden_target)
        attention_loss = F.mse_loss(attention, attention_target)
        intermediate = hidden_loss + 4.0 * attention_loss
        if phase == "general":
            return intermediate
        if phase != "task_specific":
            raise ValueError(f"unknown TinyBERT phase: {phase}")
        temperature = self.config.temperature
        teacher_probability = torch.softmax(prediction_target / temperature, dim=1)
        soft_prediction = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            teacher_probability,
            reduction="batchmean",
        )
        prediction = F.cross_entropy(logits, targets) + temperature**2 * soft_prediction
        return prediction + intermediate

    def training_step(
        self,
        inputs: Tensor,
        targets: Tensor,
        prediction_target: Tensor,
        hidden_target: Tensor,
        attention_target: Tensor,
        phase: str,
    ) -> Tensor:
        outputs = self(inputs)
        return self.compute_loss(
            outputs,
            targets,
            prediction_target,
            hidden_target,
            attention_target,
            phase,
        )
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: TinybertPortfolioModel) -> list[float]:
    batch = prepare_batch()
    history = []
    intermediate_parameters = [
        *model.query.parameters(),
        *model.key.parameters(),
        *model.value.parameters(),
        *model.hidden_regressor.parameters(),
    ]
    general_optimizer = torch.optim.Adam(
        intermediate_parameters,
        lr=model.config.learning_rate,
    )
    for _ in range(model.config.general_steps):
        loss = model.training_step(*batch, phase="general")
        general_optimizer.zero_grad()
        loss.backward()
        general_optimizer.step()
        history.append(float(loss.detach().cpu()))
    task_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.config.learning_rate,
    )
    for _ in range(model.config.task_steps):
        loss = model.training_step(*batch, phase="task_specific")
        task_optimizer.zero_grad()
        loss.backward()
        task_optimizer.step()
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
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(model: TinybertPortfolioModel) -> dict[str, float]:
    with torch.no_grad():
        test_inputs = token_embeddings[192:256]
        test_targets, prediction_targets = sequence_teacher_targets(test_inputs)
        logits, hidden, attention = model(test_inputs)
        accuracy = (logits.argmax(1) == test_targets).float().mean()
        hidden_mse = F.mse_loss(hidden, teacher_hidden[192:256])
        attention_mse = F.mse_loss(attention, teacher_attention[192:256])
        temperature = model.config.temperature
        prediction_kl = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            torch.softmax(prediction_targets / temperature, dim=1),
            reduction="batchmean",
        )
    return {
        "test_accuracy": float(accuracy.cpu()),
        "prediction_kl": float(prediction_kl.cpu()),
        "hidden_mse": float(hidden_mse.cpu()),
        "attention_mse": float(attention_mse.cpu()),
        "two_stage_updates": float(
            model.config.general_steps + model.config.task_steps
        ),
    }


portfolio_model = TinybertPortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["test_accuracy"] > 0.35
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 04 ? Deep Compression
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=4,
        slug="deep_compression",
        short_title="Deep Compression",
        paper_title=(
            "Deep Compression: Compressing Deep Neural Networks with Pruning, Tra"
            "ined Quantization and Huffman Coding"
        ),
        authors="Song Han, Huizi Mao, William J. Dally",
        year=2016,
        primary_url="https://arxiv.org/abs/1510.00149",
        venue="ICLR 2016",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "magnitude pruning→weight-sharing quantization→entropy coding의 세 단계를 "
            "작은 weight tensor에 적용한다."
        ),
        original_scale=(
            "원 논문은 AlexNet/VGG를 재학습하고 실제 저장·에너지를 측정했다. 실습은 압축 지표와 reconstruction "
            "error만 계산한다."
        ),
        mappings=(
            (
                "Figure 1: three-stage pipeline",
                "Task 1–3",
                "단계별 크기",
            ),
            (
                "§3: network pruning",
                "Task 1",
                "sparsity/mask",
            ),
            (
                "§4, Figure 2: trained quantization",
                "Task 2",
                "고정 assignment 아래 centroid gradient 재학습",
            ),
            (
                "§5: Huffman coding",
                "Task 3",
                "entropy lower-bound bit estimate",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
프루닝은 0을 늘리고, 양자화는 남은 값을 centroid index로 바꾸며, entropy coding은 자주 나오는 index에 짧은 코드를 줍니다.
""",
                r"""
## 미니 재현
프루닝은 0을 늘리고, 양자화는 남은 값을 centroid index로 바꾸며, entropy coding은 자주 나오는 index에 짧은 코드를 줍니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\hat W=\operatorname{Huffman}(\operatorname{Quantize}(W\odot\mathbf1_{|W|>\tau}))$$

- **기호 정의:** τ는 pruning 임계값, Quantize는 centroid 공유, Huffman은 entropy coding입니다.
- **수식의 역할:** 0 제거, weight sharing, 빈도 기반 부호화를 순서대로 적용해 저장량을 줄입니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  Figure 1: three-stage pipeline의 흐름을 명시합니다.
- **입출력 shape:** dense weight [64, 32] → sparse mask → centroid index → 추정 bit 수
- **평가:** sparsity, centroid 재학습 전후 loss, entropy 기반 압축률
- **원문 대비 한계:** 원 논문은 AlexNet/VGG를 재학습하고 실제 저장·에너지를 측정했다. 실습은 압축 지표와 reconstruction error만 계산한다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\hat W=\operatorname{Huffman}(\operatorname{Quantize}(W\odot\mathbf1_{|W|>\tau}))$$

- **기호 정의:** τ는 pruning 임계값, Quantize는 centroid 공유, Huffman은 entropy coding입니다.
- **수식의 역할:** 0 제거, weight sharing, 빈도 기반 부호화를 순서대로 적용해 저장량을 줄입니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  Figure 1: three-stage pipeline의 흐름을 명시합니다.
- **입출력 shape:** dense weight [64, 32] → sparse mask → centroid index → 추정 bit 수
- **평가:** sparsity, centroid 재학습 전후 loss, entropy 기반 압축률
- **원문 대비 한계:** 원 논문은 AlexNet/VGG를 재학습하고 실제 저장·에너지를 측정했다. 실습은 압축 지표와 reconstruction error만 계산한다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, pruning
            _cell(
                "code",
                r"""
# TODO 1: 절댓값이 작은 70% weight를 0으로 만드는 magnitude pruning을 구현하세요.
def magnitude_prune(weight, sparsity):
    raise NotImplementedError
""",
                r"""
def magnitude_prune(weight, sparsity):
    threshold = torch.quantile(weight.abs().flatten(), sparsity)
    mask = weight.abs() > threshold
    return weight * mask, mask


pruned, mask = magnitude_prune(dense_weights, 0.70)
actual_sparsity = (~mask).float().mean().item()
assert 0.68 < actual_sparsity < 0.72 and torch.equal(pruned == 0, ~mask)
""",
                "todo",
                "pruning",
            ),
            # Cell 05 ? todo, quantization
            _cell(
                "code",
                r"""
# TODO 2: nonzero weight를 8개 centroid로 공유하는 1D k-means quantization을 구현하세요.
raise NotImplementedError
""",
                r"""
values = pruned[mask]
centroids = torch.linspace(values.min(), values.max(), 8)
for _ in range(8):
    assignment = (values[:, None] - centroids[None, :]).abs().argmin(1)
    centroids = torch.stack(
        [
            values[assignment == i].mean() if (assignment == i).any() else centroids[i]
            for i in range(8)
        ]
    )
quantized = pruned.clone()
quantized[mask] = centroids[assignment]
quant_error = F.mse_loss(quantized, dense_weights).item()
assert (
    len(torch.unique(quantized)) <= 9
    and quant_error < dense_weights.square().mean().item()
)
""",
                "todo",
                "quantization",
            ),
            # Cell 06 ? todo, coding, visualization
            _cell(
                "code",
                r"""
# TODO 3: index 빈도의 entropy로 Huffman 평균 bit 하한과 압축률을 추정하세요.
raise NotImplementedError
""",
                r"""
symbols = torch.zeros_like(pruned, dtype=torch.long)
symbols[mask] = assignment + 1
counts = torch.bincount(symbols.flatten(), minlength=9).float()
probabilities = counts / counts.sum()
entropy = (
    -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log2())
    .sum()
    .item()
)
estimated_bits = entropy * symbols.numel() + 32 * len(centroids)
original_bits = 32 * dense_weights.numel()
compression = original_bits / estimated_bits
assert entropy < math.log2(9) and compression > 4
plt.bar(
    ["dense", "pruned", "shared"],
    [dense_weights.numel(), int(mask.sum()), len(torch.unique(quantized))],
)
plt.ylabel("stored values / symbols")
plt.show()
print(
    f"sparsity={actual_sparsity:.1%}, entropy={entropy:.2f} bit, "
    f"estimated compression={compression:.1f}x"
)
""",
                "todo",
                "coding",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class DeepCompressionPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    keep_ratio: float = 0.3
    cluster_count: int = 6
    learning_rate: float = 0.025
    steps: int = 16
    centroid_steps: int = 8


def prepare_batch() -> tuple[Tensor, Tensor]:
    return features[train_idx], labels[train_idx]


class DeepCompressionPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(features.shape[1], 48)
        self.classifier = nn.Linear(48, teacher_logits.shape[1])
        self.register_buffer("mask", torch.ones_like(self.linear.weight))
        self.centroids = nn.Parameter(torch.zeros(config.cluster_count))
        self.register_buffer(
            "assignments",
            torch.zeros_like(self.linear.weight, dtype=torch.long),
        )
        self.register_buffer("sharing_active", torch.tensor(False))
        self.pruned_train_loss = float("nan")
        self.quantized_loss_before = float("nan")
        self.quantized_loss_after = float("nan")
        self.centroid_update_norm = 0.0
        self.sharing_optimizer_fresh = 0

    def effective_weight(self) -> Tensor:
        if bool(self.sharing_active):
            shared = self.centroids[self.assignments]
            return shared * self.mask
        return self.linear.weight * self.mask

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.relu(F.linear(inputs, self.effective_weight(), self.linear.bias))
        return self.classifier(hidden)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)

    def training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.compute_loss(self(inputs), targets)

    def prune(self) -> None:
        with torch.no_grad():
            threshold = torch.quantile(
                self.linear.weight.abs(),
                1.0 - self.config.keep_ratio,
            )
            self.mask.copy_((self.linear.weight.abs() >= threshold).float())
            self.linear.weight.mul_(self.mask)

    def apply_weight_sharing(self) -> None:
        with torch.no_grad():
            survivor = self.linear.weight[self.mask.bool()]
            fractions = torch.linspace(0.0, 1.0, self.config.cluster_count)
            centroids = torch.quantile(survivor, fractions.to(survivor.device))
            for _ in range(6):
                distance = (survivor[:, None] - centroids[None, :]).abs()
                survivor_assignment = distance.argmin(dim=1)
                for index in range(self.config.cluster_count):
                    members = survivor[survivor_assignment == index]
                    if members.numel():
                        centroids[index] = members.mean()
            full_assignment = torch.zeros_like(self.assignments)
            distance = (survivor[:, None] - centroids[None, :]).abs()
            full_assignment[self.mask.bool()] = distance.argmin(dim=1)
            self.centroids.copy_(centroids)
            self.assignments.copy_(full_assignment)
            self.sharing_active.fill_(True)

    def coding_proxy(self) -> dict[str, float]:
        symbols = self.assignments[self.mask.bool()]
        counts = torch.bincount(symbols, minlength=self.config.cluster_count).float()
        probability = counts[counts > 0] / counts.sum().clamp_min(1.0)
        entropy = -(probability * probability.log2()).sum()
        index_bits = entropy * symbols.numel()
        compressed_bits = index_bits + 32 * self.config.cluster_count + self.mask.numel()
        dense_bits = 32 * self.mask.numel()
        return {
            "index_entropy_bits": float(entropy.cpu()),
            "estimated_compression_ratio": float((dense_bits / compressed_bits).cpu()),
        }

    def enforce_mask(self) -> None:
        with torch.no_grad():
            self.linear.weight.mul_(self.mask)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: DeepCompressionPortfolioModel) -> list[float]:
    inputs, targets = prepare_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    warmup = max(4, model.config.steps // 2)
    history = []
    for step in range(model.config.steps):
        loss = model.training_step(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        if model.linear.weight.grad is not None:
            model.linear.weight.grad.mul_(model.mask)
        optimizer.step()
        model.enforce_mask()
        history.append(float(loss.detach().cpu()))
        if step + 1 == warmup:
            model.prune()
    with torch.no_grad():
        model.pruned_train_loss = float(
            model.training_step(inputs, targets).detach().cpu()
        )
    model.apply_weight_sharing()
    with torch.no_grad():
        model.quantized_loss_before = float(
            model.training_step(inputs, targets).detach().cpu()
        )
    centroids_before = model.centroids.detach().clone()
    sharing_parameters = [
        model.centroids,
        model.linear.bias,
        *model.classifier.parameters(),
    ]
    sharing_optimizer = torch.optim.Adam(
        sharing_parameters,
        lr=model.config.learning_rate,
    )
    model.sharing_optimizer_fresh = int(len(sharing_optimizer.state) == 0)
    for _ in range(model.config.centroid_steps):
        loss = model.training_step(inputs, targets)
        sharing_optimizer.zero_grad()
        loss.backward()
        sharing_optimizer.step()
        history.append(float(loss.detach().cpu()))
    with torch.no_grad():
        model.quantized_loss_after = float(
            model.training_step(inputs, targets).detach().cpu()
        )
        model.centroid_update_norm = float(
            (model.centroids - centroids_before).norm().cpu()
        )
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(model: DeepCompressionPortfolioModel) -> dict[str, float]:
    with torch.no_grad():
        logits = model(features[test_idx])
        accuracy = (logits.argmax(1) == labels[test_idx]).float().mean()
        sparsity = 1.0 - model.mask.mean()
        shared_values = torch.unique(model.effective_weight()[model.mask.bool()]).numel()
    metrics = {
        "test_accuracy": float(accuracy.cpu()),
        "weight_sparsity": float(sparsity.cpu()),
        "shared_weight_values": float(shared_values),
        "pruned_train_loss": model.pruned_train_loss,
        "quantized_loss_before_retraining": model.quantized_loss_before,
        "quantized_loss_after_retraining": model.quantized_loss_after,
        "centroid_update_norm": model.centroid_update_norm,
        "centroid_retraining_steps": float(model.config.centroid_steps),
        "sharing_optimizer_fresh": float(model.sharing_optimizer_fresh),
    }
    metrics.update(model.coding_proxy())
    return metrics


portfolio_model = DeepCompressionPortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["centroid_update_norm"] > 0.0
assert (
    portfolio_metrics["quantized_loss_after_retraining"]
    < portfolio_metrics["quantized_loss_before_retraining"]
)
assert portfolio_metrics["estimated_compression_ratio"] > 1.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 05 ? Lottery Ticket Hypothesis
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=5,
        slug="lottery_ticket",
        short_title="Lottery Ticket Hypothesis",
        paper_title=(
            "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
        ),
        authors="Jonathan Frankle, Michael Carbin",
        year=2019,
        primary_url="https://openreview.net/forum?id=rJl-b3RcF7",
        venue="ICLR 2019",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "dense network 학습으로 mask를 찾고 surviving weight를 초기값으로 되감아 sparse subne"
            "twork를 재학습한다."
        ),
        original_scale=(
            "원 논문은 MNIST/CIFAR의 반복 magnitude pruning과 여러 seed를 비교했다. 실습은 한 번의 pru"
            "ning과 작은 MLP다."
        ),
        mappings=(
            (
                "§2: winning ticket definition",
                "Task 1",
                "초기값·mask 보존",
            ),
            (
                "§2, Figure 1: iterative pruning procedure",
                "Task 1–2",
                "train→prune→rewind",
            ),
            (
                "§3: experimental methodology",
                "Task 2",
                "동일 mask·step과 세 fresh optimizer",
            ),
            (
                "§4: winning tickets results",
                "Task 3",
                "rewind ticket/random sparse/dense 비교",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
중요한 것은 작은 weight 자체뿐 아니라, 그 연결에 대응하는 **초기값으로 되감는 것**이라는 가설을 코드로 분리합니다.
""",
                r"""
## 미니 재현
중요한 것은 작은 weight 자체뿐 아니라, 그 연결에 대응하는 **초기값으로 되감는 것**이라는 가설을 코드로 분리합니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$f(\mathbf x,m\odot\theta_0)\quad\text{with}\quad\operatorname{Train}(m\odot\theta_0)
\approx\operatorname{Train}(\theta)$$

- **기호 정의:** m은 pruning mask, θ0는 원래 초기값, m⊙θ0는 winning ticket 후보입니다.
- **수식의 역할:** 학습 후 찾은 sparse 연결을 원래 초기값으로 되감아 독립적으로 재학습합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2: winning ticket definition의 흐름을 명시합니다.
- **입출력 shape:** 특징 [B, 32] → masked hidden [B, 64] → class logit [B, 4]
- **평가:** 동일 예산의 rewind ticket/random sparse/dense 정확도와 loss
- **원문 대비 한계:** 원 논문은 MNIST/CIFAR의 반복 magnitude pruning과 여러 seed를 비교했다. 실습은 한 번의 pruning과 작은 MLP다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$f(\mathbf x,m\odot\theta_0)\quad\text{with}\quad\operatorname{Train}(m\odot\theta_0)
\approx\operatorname{Train}(\theta)$$

- **기호 정의:** m은 pruning mask, θ0는 원래 초기값, m⊙θ0는 winning ticket 후보입니다.
- **수식의 역할:** 학습 후 찾은 sparse 연결을 원래 초기값으로 되감아 독립적으로 재학습합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2: winning ticket definition의 흐름을 명시합니다.
- **입출력 shape:** 특징 [B, 32] → masked hidden [B, 64] → class logit [B, 4]
- **평가:** 동일 예산의 rewind ticket/random sparse/dense 정확도와 loss
- **원문 대비 한계:** 원 논문은 MNIST/CIFAR의 반복 magnitude pruning과 여러 seed를 비교했다. 실습은 한 번의 pruning과 작은 MLP다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, pruning
            _cell(
                "code",
                r"""
# TODO 1: dense MLP를 학습하고 hidden weight의 상위 30% magnitude mask를 만드세요.
raise NotImplementedError
""",
                r"""
class TicketNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(32, 64)
        self.out = nn.Linear(64, 4)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))


dense = TicketNet()
initial = {k: v.detach().clone() for k, v in dense.state_dict().items()}
optimizer = torch.optim.Adam(dense.parameters(), lr=0.025)
for _ in range(35):
    optimizer.zero_grad()
    loss = F.cross_entropy(dense(features[train_idx]), labels[train_idx])
    loss.backward()
    optimizer.step()
threshold = torch.quantile(dense.hidden.weight.detach().abs(), 0.70)
mask = (dense.hidden.weight.detach().abs() > threshold).float()
assert 0.28 < mask.mean().item() < 0.32
""",
                "todo",
                "pruning",
            ),
            # Cell 05 ? todo, rewinding
            _cell(
                "code",
                r"""
# TODO 2: surviving connection을 원래 초기값으로 rewind하고 mask가 유지되는 학습 함수를 구현하세요.
raise NotImplementedError
""",
                r"""
ticket = TicketNet()
ticket.load_state_dict(initial)
with torch.no_grad():
    ticket.hidden.weight.mul_(mask)


def train_masked(model, mask, steps=45):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(features[train_idx]), labels[train_idx])
        loss.backward()
        model.hidden.weight.grad.mul_(mask)
        optimizer.step()
        with torch.no_grad():
            model.hidden.weight.mul_(mask)
        history.append(float(loss.detach()))
    return history


ticket_history = train_masked(ticket, mask)
assert (ticket.hidden.weight.detach()[mask == 0] == 0).all() and ticket_history[
    -1
] < ticket_history[0]
""",
                "todo",
                "rewinding",
            ),
            # Cell 06 ? todo, metric, visualization
            _cell(
                "code",
                r"""
# TODO 3: sparse ticket의 정확도와 파라미터 비율을 기록·시각화하세요.
raise NotImplementedError
""",
                r"""
with torch.no_grad():
    ticket_accuracy = (
        (ticket(features[test_idx]).argmax(-1) == labels[test_idx])
        .float()
        .mean()
        .item()
    )
dense_count = sum(p.numel() for p in ticket.parameters())
active_count = (
    int(mask.sum())
    + ticket.hidden.bias.numel()
    + ticket.out.weight.numel()
    + ticket.out.bias.numel()
)
active_ratio = active_count / dense_count
assert ticket_accuracy > 0.75 and active_ratio < 0.45
plt.plot(ticket_history)
plt.title(
    f"rewound sparse ticket: acc={ticket_accuracy:.1%}, active={active_ratio:.1%}"
)
plt.show()
print(f"active parameters={active_ratio:.1%}, accuracy={ticket_accuracy:.1%}")
""",
                "todo",
                "metric",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class LotteryTicketPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    keep_ratio: float = 0.3
    learning_rate: float = 0.025
    mask_finding_steps: int = 8
    comparison_steps: int = 10


def prepare_batch() -> tuple[Tensor, Tensor]:
    return features[train_idx], labels[train_idx]


class LotteryTicketPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(features.shape[1], 48)
        self.classifier = nn.Linear(48, teacher_logits.shape[1])
        self._initial_state = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
        }
        self._masks = {
            name: torch.ones_like(parameter)
            for name, parameter in self.named_parameters()
        }
        self.rewind_verified = False
        self.optimizer_reset_count = 0

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(F.relu(self.linear(inputs)))

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)

    def training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.compute_loss(self(inputs), targets)

    def parameter_state(self) -> dict[str, Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
        }

    def load_parameter_state(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(state[name].to(parameter.device))

    def mask_state(self) -> dict[str, Tensor]:
        return {name: mask.detach().clone() for name, mask in self._masks.items()}

    def install_masks(self, masks: dict[str, Tensor]) -> None:
        self._masks = {
            name: masks[name].to(parameter.device).clone()
            for name, parameter in self.named_parameters()
        }
        self.enforce_mask()

    def prune_and_rewind(self) -> None:
        with torch.no_grad():
            prunable = [
                parameter.abs().flatten()
                for parameter in self.parameters()
                if parameter.ndim >= 2
            ]
            threshold = torch.quantile(torch.cat(prunable), 1.0 - self.config.keep_ratio)
            for name, parameter in self.named_parameters():
                if parameter.ndim >= 2:
                    self._masks[name] = (parameter.abs() >= threshold).float()
                initial = self._initial_state[name].to(parameter.device)
                mask = self._masks[name].to(parameter.device)
                parameter.copy_(initial * mask)
            self.rewind_verified = all(
                torch.equal(
                    parameter,
                    self._initial_state[name].to(parameter.device)
                    * self._masks[name].to(parameter.device),
                )
                for name, parameter in self.named_parameters()
            )

    def mask_gradients(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(self._masks[name].to(parameter.device))

    def enforce_mask(self) -> None:
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.mul_(self._masks[name].to(parameter.device))

    def sparsity(self) -> Tensor:
        weight_masks = [
            mask.flatten()
            for name, mask in self._masks.items()
            if name.endswith("weight")
        ]
        return 1.0 - torch.cat(weight_masks).mean()


@dataclass
class LotteryComparison:
    ticket: LotteryTicketPortfolioModel
    random_control: LotteryTicketPortfolioModel
    dense_baseline: LotteryTicketPortfolioModel
    mask_finding_history: list[float]
    histories: dict[str, list[float]]
    fresh_optimizer_count: int
    random_control_differs: bool
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def train_comparison_model(
    model: LotteryTicketPortfolioModel,
    inputs: Tensor,
    targets: Tensor,
    steps: int,
) -> tuple[list[float], int]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.config.learning_rate,
    )
    fresh_optimizer = int(len(optimizer.state) == 0)
    history = []
    for _ in range(steps):
        loss = model.training_step(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        model.mask_gradients()
        optimizer.step()
        model.enforce_mask()
        history.append(float(loss.detach().cpu()))
    model.optimizer_reset_count = fresh_optimizer
    return history, fresh_optimizer


def fit_portfolio_model(
    model: LotteryTicketPortfolioModel,
) -> LotteryComparison:
    inputs, targets = prepare_batch()
    initial_state = model.parameter_state()
    dense_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.config.learning_rate,
    )
    mask_finding_history = []
    for _ in range(model.config.mask_finding_steps):
        loss = model.training_step(inputs, targets)
        dense_optimizer.zero_grad()
        loss.backward()
        dense_optimizer.step()
        mask_finding_history.append(float(loss.detach().cpu()))
    model.prune_and_rewind()

    shared_masks = model.mask_state()
    dense_baseline = LotteryTicketPortfolioModel(model.config)
    dense_baseline.load_parameter_state(initial_state)
    random_control = LotteryTicketPortfolioModel(model.config)
    random_control.install_masks(shared_masks)
    random_control_differs = any(
        not torch.equal(
            ticket_parameter,
            control_parameter,
        )
        for ticket_parameter, control_parameter in zip(
            model.parameters(),
            random_control.parameters(),
        )
    )

    histories = {}
    fresh_optimizer_count = 0
    candidates = {
        "ticket": model,
        "random_control": random_control,
        "dense_baseline": dense_baseline,
    }
    for name, candidate in candidates.items():
        history, fresh = train_comparison_model(
            candidate,
            inputs,
            targets,
            model.config.comparison_steps,
        )
        histories[name] = history
        fresh_optimizer_count += fresh
    return LotteryComparison(
        ticket=model,
        random_control=random_control,
        dense_baseline=dense_baseline,
        mask_finding_history=mask_finding_history,
        histories=histories,
        fresh_optimizer_count=fresh_optimizer_count,
        random_control_differs=random_control_differs,
    )
""",
                "portfolio-training",
                "task",
            ),
            # Cell 11 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(
    comparison: LotteryComparison,
) -> dict[str, float]:
    accuracies = {}
    with torch.no_grad():
        candidates = {
            "ticket": comparison.ticket,
            "random_control": comparison.random_control,
            "dense_baseline": comparison.dense_baseline,
        }
        for name, candidate in candidates.items():
            logits = candidate(features[test_idx])
            accuracy = (logits.argmax(1) == labels[test_idx]).float().mean()
            accuracies[name] = float(accuracy.cpu())
        ticket_sparsity = float(comparison.ticket.sparsity().cpu())
        control_sparsity = float(comparison.random_control.sparsity().cpu())
    ticket_loss = comparison.histories["ticket"][-1]
    control_loss = comparison.histories["random_control"][-1]
    dense_loss = comparison.histories["dense_baseline"][-1]
    winning_condition = (
        accuracies["ticket"] >= accuracies["random_control"]
        and accuracies["ticket"] >= accuracies["dense_baseline"] - 0.05
        and ticket_loss <= control_loss
    )
    return {
        "ticket_test_accuracy": accuracies["ticket"],
        "random_control_test_accuracy": accuracies["random_control"],
        "dense_baseline_test_accuracy": accuracies["dense_baseline"],
        "ticket_minus_random_accuracy": (
            accuracies["ticket"] - accuracies["random_control"]
        ),
        "ticket_minus_dense_accuracy": (
            accuracies["ticket"] - accuracies["dense_baseline"]
        ),
        "ticket_final_train_loss": ticket_loss,
        "random_control_final_train_loss": control_loss,
        "dense_baseline_final_train_loss": dense_loss,
        "random_minus_ticket_train_loss": control_loss - ticket_loss,
        "ticket_weight_sparsity": ticket_sparsity,
        "control_weight_sparsity": control_sparsity,
        "matched_comparison_steps": float(
            comparison.ticket.config.comparison_steps
        ),
        "full_tensor_rewind_verified": float(comparison.ticket.rewind_verified),
        "fresh_comparison_optimizers": float(comparison.fresh_optimizer_count),
        "random_control_differs": float(comparison.random_control_differs),
        "winning_ticket_condition": float(winning_condition),
    }


portfolio_model = LotteryTicketPortfolioModel(PortfolioConfig())
portfolio_comparison = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_comparison)
all_histories = [
    portfolio_comparison.mask_finding_history,
    *portfolio_comparison.histories.values(),
]
assert all(np.isfinite(history).all() for history in all_histories)
assert portfolio_metrics["fresh_comparison_optimizers"] == 3.0
assert portfolio_metrics["random_control_differs"] == 1.0
assert abs(
    portfolio_metrics["ticket_weight_sparsity"]
    - portfolio_metrics["control_weight_sparsity"]
) < 1e-8
assert portfolio_metrics["winning_ticket_condition"] == 1.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 06 ? Integer-only Quantization
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=6,
        slug="integer_quantization",
        short_title="Integer-only Quantization",
        paper_title=(
            "Quantization and Training of Neural Networks for Efficient Integer-A"
            "rithmetic-Only Inference"
        ),
        authors="Benoit Jacob et al.",
        year=2018,
        primary_url=(
            "https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantizat"
            "ion_and_Training_CVPR_2018_paper.html"
        ),
        venue="CVPR 2018",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "affine quantization, integer matrix multiplication, fake-quantizatio"
            "n 학습을 구현한다."
        ),
        original_scale=(
            "원 논문은 MobileNet/COCO와 실제 ARM integer kernel을 평가했다. 실습은 작은 linear lay"
            "er의 수치 오차와 QAT proxy다."
        ),
        mappings=(
            (
                "§2.1, Eq. (1): r=S(q-Z)",
                "Task 1",
                "scale/zero point와 round-trip",
            ),
            (
                "§2.2, Eq. (2)–(4): quantized matmul",
                "Task 2",
                "integer accumulator",
            ),
            (
                "§2.4: zero-point and range",
                "Task 1–2",
                "train-only calibration scale 고정과 uint8 clamp",
            ),
            (
                "§3: training with simulated quantization",
                "Task 3",
                "fake-quant loss/accuracy",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
실수 r을 scale S와 zero-point Z로 정수 q에 대응시키고, 추론의 핵심 누산을 정수로 수행합니다.
""",
                r"""
## 미니 재현
실수 r을 scale S와 zero-point Z로 정수 q에 대응시키고, 추론의 핵심 누산을 정수로 수행합니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$r=S(q-Z),\quad q=\operatorname{clamp}(\operatorname{round}(r/S)+Z,q_{min},q_{max})$$

- **기호 정의:** r은 실수, q는 정수, S는 scale, Z는 zero-point입니다.
- **수식의 역할:** affine mapping으로 0을 정확히 표현하고 integer accumulator 뒤 scale을 복원합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2.1, Eq. (1): r=S(q-Z)의 흐름을 명시합니다.
- **입출력 shape:** float activation [B, 32] → uint8/int32 → quantized logit [B, 4]
- **평가:** round-trip MSE, integer matmul 상대오차, QAT accuracy
- **원문 대비 한계:** 원 논문은 MobileNet/COCO와 실제 ARM integer kernel을 평가했다. 실습은 작은 linear layer의 수치
오차와 QAT proxy다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$r=S(q-Z),\quad q=\operatorname{clamp}(\operatorname{round}(r/S)+Z,q_{min},q_{max})$$

- **기호 정의:** r은 실수, q는 정수, S는 scale, Z는 zero-point입니다.
- **수식의 역할:** affine mapping으로 0을 정확히 표현하고 integer accumulator 뒤 scale을 복원합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2.1, Eq. (1): r=S(q-Z)의 흐름을 명시합니다.
- **입출력 shape:** float activation [B, 32] → uint8/int32 → quantized logit [B, 4]
- **평가:** round-trip MSE, integer matmul 상대오차, QAT accuracy
- **원문 대비 한계:** 원 논문은 MobileNet/COCO와 실제 ARM integer kernel을 평가했다. 실습은 작은 linear layer의 수치
오차와 QAT proxy다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, quantization
            _cell(
                "code",
                r"""
# TODO 1: uint8 affine quantize/dequantize를 구현하세요.
def affine_quantize(x, bits=8):
    raise NotImplementedError
""",
                r"""
def affine_quantize(x, bits=8):
    qmin, qmax = 0, 2**bits - 1
    scale = (x.max() - x.min()).clamp_min(1e-8) / (qmax - qmin)
    zero = torch.round(torch.tensor(qmin, device=x.device) - x.min() / scale).clamp(
        qmin, qmax
    )
    q = torch.round(x / scale + zero).clamp(qmin, qmax).to(torch.int32)
    return q, scale, zero


def dequantize(q, scale, zero):
    return scale * (q.float() - zero)


q, s, z = affine_quantize(calibration)
roundtrip = dequantize(q, s, z)
assert q.min() >= 0 and q.max() <= 255 and F.mse_loss(roundtrip, calibration) < 2e-4
""",
                "todo",
                "quantization",
            ),
            # Cell 05 ? todo, integer-inference
            _cell(
                "code",
                r"""
# TODO 2: centered integer activation/weight를 곱하고 scale을 복원하세요.
raise NotImplementedError
""",
                r"""
# PyTorch CUDA에는 int32 GEMM kernel이 없으므로 정수 누산 검증만 CPU에서 수행합니다.
x = calibration[:16].cpu()
weight = teacher_weights.cpu()
qx, sx, zx = affine_quantize(x)
qw, sw, zw = affine_quantize(weight)
integer_acc = (qx - zx.to(torch.int32)) @ (qw - zw.to(torch.int32))
quantized_output = integer_acc.float() * (sx * sw)
float_output = x @ weight
relative_error = (quantized_output - float_output).norm() / float_output.norm()
assert integer_acc.dtype == torch.int32 and relative_error < 0.03
print(f"integer matmul relative error={relative_error:.3%} (CPU int32 backend)")
""",
                "todo",
                "integer-inference",
            ),
            # Cell 06 ? todo, qat, visualization
            _cell(
                "code",
                r"""
# TODO 3: straight-through fake quantization으로 작은 linear classifier를 학습하세요.
raise NotImplementedError
""",
                r"""
def fake_quant(x, bits=8):
    q, s, z = affine_quantize(x.detach(), bits)
    rounded = dequantize(q, s, z)
    return x + (rounded - x).detach()


model = nn.Linear(32, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
losses = []
for _ in range(50):
    optimizer.zero_grad()
    logits = F.linear(
        fake_quant(features[train_idx]), fake_quant(model.weight), model.bias
    )
    loss = F.cross_entropy(logits, labels[train_idx])
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    accuracy = (
        (
            F.linear(
                fake_quant(features[test_idx]), fake_quant(model.weight), model.bias
            ).argmax(-1)
            == labels[test_idx]
        )
        .float()
        .mean()
        .item()
    )
assert losses[-1] < losses[0] and accuracy > 0.85
plt.plot(losses)
plt.title(f"fake-quant training: acc={accuracy:.1%}")
plt.show()
""",
                "todo",
                "qat",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class IntegerQuantizationPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    bits: int = 8
    learning_rate: float = 0.025
    steps: int = 16


def prepare_batch() -> tuple[Tensor, Tensor]:
    return features[train_idx], labels[train_idx]


class AffineQuantizer(nn.Module):
    def __init__(self, bits: int):
        super().__init__()
        self.minimum = 0
        self.maximum = 2**bits - 1
        self.register_buffer("frozen_scale", torch.tensor(1.0))
        self.register_buffer(
            "frozen_zero_point",
            torch.tensor(0, dtype=torch.int32),
        )
        self.register_buffer("calibrated", torch.tensor(False))

    def parameters_for(self, values: Tensor) -> tuple[Tensor, Tensor]:
        scale = (values.max() - values.min()).clamp_min(1e-8)
        scale = scale / (self.maximum - self.minimum)
        zero = torch.round(self.minimum - values.min() / scale)
        return scale, zero.clamp(self.minimum, self.maximum)

    def calibrate(self, values: Tensor) -> None:
        scale, zero = self.parameters_for(values.detach())
        self.frozen_scale.copy_(scale)
        self.frozen_zero_point.copy_(zero.to(torch.int32))
        self.calibrated.fill_(True)

    def forward(self, values: Tensor, use_frozen: bool = False) -> Tensor:
        integer, scale, zero = self.quantize_to_int(values, use_frozen)
        restored = scale * (integer - zero)
        return values + (restored.float() - values).detach()

    def quantize_to_int(
        self,
        values: Tensor,
        use_frozen: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if use_frozen:
            if not bool(self.calibrated):
                raise RuntimeError("calibrate activation quantizer before inference")
            scale = self.frozen_scale
            zero = self.frozen_zero_point
        else:
            scale, zero = self.parameters_for(values.detach())
        integer = torch.round(values.detach() / scale + zero)
        integer = integer.clamp(self.minimum, self.maximum).to(torch.int32)
        return integer, scale, zero.to(torch.int32)


class IntegerQuantizationPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.activation_quantizer = AffineQuantizer(config.bits)
        self.weight_quantizer = AffineQuantizer(config.bits)
        self.classifier = nn.Linear(features.shape[1], teacher_logits.shape[1])

    def calibrate_activations(self, inputs: Tensor) -> None:
        self.activation_quantizer.calibrate(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        quantized_input = self.activation_quantizer(inputs, use_frozen=True)
        quantized_weight = self.weight_quantizer(self.classifier.weight)
        return F.linear(quantized_input, quantized_weight, self.classifier.bias)

    def integer_forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        input_integer, input_scale, input_zero = (
            self.activation_quantizer.quantize_to_int(
                inputs,
                use_frozen=True,
            )
        )
        weight_integer, weight_scale, weight_zero = (
            self.weight_quantizer.quantize_to_int(self.classifier.weight)
        )
        centered_input = input_integer - input_zero
        centered_weight = weight_integer - weight_zero
        source_device = centered_input.device
        if source_device.type == "cpu":
            accumulator = centered_input @ centered_weight.transpose(0, 1)
        else:
            accumulator = centered_input.cpu() @ centered_weight.cpu().transpose(0, 1)
        if self.classifier.bias is not None:
            accumulator_scale = (input_scale * weight_scale).to(accumulator.device)
            integer_bias = torch.round(
                self.classifier.bias.detach().to(accumulator.device) / accumulator_scale
            ).to(torch.int32)
            accumulator = accumulator + integer_bias
        output_scale = (input_scale * weight_scale).to(accumulator.device)
        logits = (accumulator.float() * output_scale).to(source_device)
        return logits, accumulator

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)

    def training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.compute_loss(self(inputs), targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: IntegerQuantizationPortfolioModel) -> list[float]:
    inputs, targets = prepare_batch()
    model.calibrate_activations(inputs[:128])
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, targets)
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
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(
    model: IntegerQuantizationPortfolioModel,
) -> dict[str, float]:
    with torch.no_grad():
        fake_logits = model(features[test_idx])
        integer_logits, accumulator = model.integer_forward(features[test_idx])
        accuracy = (integer_logits.argmax(1) == labels[test_idx]).float().mean()
        inference_mse = F.mse_loss(integer_logits, fake_logits)
        calibration_inputs = features[train_idx[:128]]
        quantized = model.activation_quantizer(
            calibration_inputs,
            use_frozen=True,
        )
        roundtrip = F.mse_loss(quantized, calibration_inputs)
    return {
        "test_accuracy": float(accuracy.cpu()),
        "roundtrip_mse": float(roundtrip.cpu()),
        "integer_inference_mse": float(inference_mse.cpu()),
        "integer_accumulator_verified": float(accumulator.dtype == torch.int32),
        "integer_kernel_cpu_fallback": float(features.device.type != "cpu"),
        "accumulator_peak": float(accumulator.abs().max().cpu()),
        "activation_calibrated": float(
            model.activation_quantizer.calibrated.float().cpu()
        ),
        "activation_scale": float(model.activation_quantizer.frozen_scale.cpu()),
    }


portfolio_model = IntegerQuantizationPortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["activation_calibrated"] == 1.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 07 ? MobileNetV2
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=7,
        slug="mobilenet_v2",
        short_title="MobileNetV2",
        paper_title="MobileNetV2: Inverted Residuals and Linear Bottlenecks",
        authors="Mark Sandler et al.",
        year=2018,
        primary_url=(
            "https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileN"
            "etV2_Inverted_Residuals_CVPR_2018_paper.html"
        ),
        venue="CVPR 2018",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "depthwise convolution과 expand→filter→linear project의 inverted residu"
            "al block을 구현한다."
        ),
        original_scale=(
            "원 논문은 ImageNet/COCO/VOC와 모바일 지연시간을 측정했다. 실습은 16×16 영상의 2-block class"
            "ifier다."
        ),
        mappings=(
            (
                "§3.1: depthwise separable convolution",
                "Task 1",
                "parameter 계산",
            ),
            (
                "§3.2: linear bottlenecks",
                "Task 2",
                "마지막 projection 무활성화",
            ),
            (
                "§3.3, Figure 3: inverted residual",
                "Task 2",
                "expand-depthwise-project",
            ),
            (
                "§3.4, Table 2: architecture",
                "Task 3",
                "작은 network/parameter 수",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
좁은 입력을 먼저 넓히고, 값싼 depthwise convolution으로 공간을 처리한 뒤 선형으로 다시 좁힙니다.
""",
                r"""
## 미니 재현
좁은 입력을 먼저 넓히고, 값싼 depthwise convolution으로 공간을 처리한 뒤 선형으로 다시 좁힙니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\operatorname{Block}(x)=x+P_{linear}(D_{3\times3}(E_{1\times1}(x)))$$

- **기호 정의:** E는 expand, D는 depthwise, P는 linear project이며 shape가 같을 때 residual입니다.
- **수식의 역할:** 좁은 공간이 아닌 확장 공간에서 비선형 spatial filtering을 하고 선형으로 압축합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §3.1: depthwise separable convolution의 흐름을 명시합니다.
- **입출력 shape:** image [B, 1, 16, 16] → expand/depthwise/project feature → logit [B, 4]
- **평가:** parameter 수, loss/accuracy, 선택 장치 latency
- **원문 대비 한계:** 원 논문은 ImageNet/COCO/VOC와 모바일 지연시간을 측정했다. 실습은 16×16 영상의 2-block classifier다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$\operatorname{Block}(x)=x+P_{linear}(D_{3\times3}(E_{1\times1}(x)))$$

- **기호 정의:** E는 expand, D는 depthwise, P는 linear project이며 shape가 같을 때 residual입니다.
- **수식의 역할:** 좁은 공간이 아닌 확장 공간에서 비선형 spatial filtering을 하고 선형으로 압축합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §3.1: depthwise separable convolution의 흐름을 명시합니다.
- **입출력 shape:** image [B, 1, 16, 16] → expand/depthwise/project feature → logit [B, 4]
- **평가:** parameter 수, loss/accuracy, 선택 장치 latency
- **원문 대비 한계:** 원 논문은 ImageNet/COCO/VOC와 모바일 지연시간을 측정했다. 실습은 16×16 영상의 2-block classifier다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, efficiency
            _cell(
                "code",
                r"""
# TODO 1: 표준 3×3 conv와 depthwise+pointwise의 parameter 수를 비교하세요.
raise NotImplementedError
""",
                r"""
channels = 32
standard = channels * channels * 3 * 3
separable = channels * 3 * 3 + channels * channels
assert separable < standard and separable / standard < 0.15
print("standard/separable:", standard, separable)
""",
                "todo",
                "efficiency",
            ),
            # Cell 05 ? todo, architecture
            _cell(
                "code",
                r"""
# TODO 2: expand→depthwise→linear project와 조건부 shortcut을 구현하세요.
raise NotImplementedError
""",
                r"""
class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, expand=3, stride=1):
        super().__init__()
        hidden = cin * expand
        self.use_skip = stride == 1 and cin == cout
        self.net = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False),
            nn.ReLU6(),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.ReLU6(),
            nn.Conv2d(hidden, cout, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        return x + out if self.use_skip else out


block = InvertedResidual(8, 8)
probe = torch.rand(2, 8, 8, 8)
assert block(probe).shape == probe.shape and block.net[-1].groups == 1
""",
                "todo",
                "architecture",
            ),
            # Cell 06 ? todo, training, visualization
            _cell(
                "code",
                r"""
# TODO 3: 작은 MobileNetV2를 학습하고 parameter 수·정확도를 기록하세요.
raise NotImplementedError
""",
                r"""
class MobileMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.blocks = nn.Sequential(
            InvertedResidual(8, 8, 2), InvertedResidual(8, 12, 3, 2)
        )
        self.head = nn.Linear(12, 4)

    def forward(self, x):
        return self.head(self.blocks(F.relu6(self.stem(x))).mean((2, 3)))


model = MobileMini()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
losses = []
for _ in range(40):
    optimizer.zero_grad()
    loss = F.cross_entropy(
        model(images[image_train_idx]), image_labels[image_train_idx]
    )
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    accuracy = (
        (model(images[image_test_idx]).argmax(-1) == image_labels[image_test_idx])
        .float()
        .mean()
        .item()
    )
parameters = sum(p.numel() for p in model.parameters())
assert losses[-1] < losses[0] and accuracy > 0.7 and parameters < 4000
plt.plot(losses)
plt.title(f"MobileNetV2 mini: {parameters} params, acc={accuracy:.1%}")
plt.show()
""",
                "todo",
                "training",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class MobilenetV2PortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    channels: int = 8
    learning_rate: float = 0.02
    steps: int = 14


def prepare_batch() -> tuple[Tensor, Tensor]:
    return images[image_train_idx[:160]], image_labels[image_train_idx[:160]]


class InvertedResidualPortfolioBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.relu6(self.expand(inputs))
        hidden = F.relu6(self.depthwise(hidden))
        return inputs + self.project(hidden)


class MobilenetV2PortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Conv2d(1, config.channels, 3, padding=1)
        self.block = InvertedResidualPortfolioBlock(config.channels)
        self.classifier = nn.Linear(config.channels, 4)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.relu(self.stem(inputs))
        hidden = self.block(hidden).mean(dim=(2, 3))
        return self.classifier(hidden)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)

    def training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.compute_loss(self(inputs), targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: MobilenetV2PortfolioModel) -> list[float]:
    inputs, targets = prepare_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, targets)
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
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(model: MobilenetV2PortfolioModel) -> dict[str, float]:
    with torch.no_grad():
        logits = model(images[image_test_idx])
        accuracy = (logits.argmax(1) == image_labels[image_test_idx]).float().mean()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    depthwise_groups = model.block.depthwise.groups
    return {
        "test_accuracy": float(accuracy.cpu()),
        "parameter_count": float(parameter_count),
        "depthwise_groups": float(depthwise_groups),
    }


portfolio_model = MobilenetV2PortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 08 ? ShuffleNet V2
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=8,
        slug="shufflenet_v2",
        short_title="ShuffleNet V2",
        paper_title=(
            "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
        ),
        authors="Ningning Ma et al.",
        year=2018,
        primary_url=(
            "https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-"
            "weight_CNN_Architecture_ECCV_2018_paper.html"
        ),
        venue="ECCV 2018",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "channel split·가벼운 branch·concat·channel shuffle을 구현하고 FLOPs와 실제 late"
            "ncy를 분리해 기록한다."
        ),
        original_scale=(
            "원 논문은 ARM/GPU 최적화 라이브러리에서 실제 속도를 측정했다. 실습 latency는 현재 선택 장치의 상대 관찰값일"
            " 뿐이다."
        ),
        mappings=(
            (
                "§2: G1–G4 practical guidelines",
                "Task 1",
                "균형 channel과 memory access 직관",
            ),
            (
                "§3, Figure 3(c): channel split",
                "Task 2",
                "반쪽 identity/branch",
            ),
            (
                "§3, Figure 3: concat + shuffle",
                "Task 1–2",
                "채널 permutation",
            ),
            (
                "§4: FLOPs vs measured speed",
                "Task 3",
                "선택 장치 latency를 별도 기록",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
FLOPs가 같아도 memory access·분기·element-wise 연산 때문에 속도가 다릅니다. 구조와 실제 시간을 따로 봅니다.
""",
                r"""
## 미니 재현
FLOPs가 같아도 memory access·분기·element-wise 연산 때문에 속도가 다릅니다. 구조와 실제 시간을 따로 봅니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$Y=\operatorname{Shuffle}(X_1\VertP_{1\times1}D_{3\times3}P_{1\times1}(X_2))$$

- **기호 정의:** X1/X2는 channel split, D는 depthwise, ||는 concat, Shuffle은 channel permutation입니다.
- **수식의 역할:** 절반은 identity로 두고 계산 branch 뒤 채널을 섞어 정보 교환과 메모리 효율을 얻습니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2: G1–G4 practical guidelines의 흐름을 명시합니다.
- **입출력 shape:** feature [B, C, H, W] → 두 [B, C/2, H, W] branch → shuffled output
- **평가:** parameter 수, accuracy, 실제 batch latency를 FLOPs와 별도 기록
- **원문 대비 한계:** 원 논문은 ARM/GPU 최적화 라이브러리에서 실제 속도를 측정했다. 실습 latency는 현재 선택 장치의 상대 관찰값일 뿐이다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$Y=\operatorname{Shuffle}(X_1\VertP_{1\times1}D_{3\times3}P_{1\times1}(X_2))$$

- **기호 정의:** X1/X2는 channel split, D는 depthwise, ||는 concat, Shuffle은 channel permutation입니다.
- **수식의 역할:** 절반은 identity로 두고 계산 branch 뒤 채널을 섞어 정보 교환과 메모리 효율을 얻습니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  §2: G1–G4 practical guidelines의 흐름을 명시합니다.
- **입출력 shape:** feature [B, C, H, W] → 두 [B, C/2, H, W] branch → shuffled output
- **평가:** parameter 수, accuracy, 실제 batch latency를 FLOPs와 별도 기록
- **원문 대비 한계:** 원 논문은 ARM/GPU 최적화 라이브러리에서 실제 속도를 측정했다. 실습 latency는 현재 선택 장치의 상대 관찰값일 뿐이다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, channel-shuffle
            _cell(
                "code",
                r"""
# TODO 1: [N,C,H,W]를 groups로 reshape/transposition하는 channel shuffle을 구현하세요.
def channel_shuffle(x, groups=2):
    raise NotImplementedError
""",
                r"""
def channel_shuffle(x, groups=2):
    n, c, h, w = x.shape
    assert c % groups == 0
    return (
        x.reshape(n, groups, c // groups, h, w)
        .transpose(1, 2)
        .contiguous()
        .reshape(n, c, h, w)
    )


probe = torch.arange(8.0).view(1, 8, 1, 1)
shuffled = channel_shuffle(probe, 2)
assert shuffled.flatten().tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
""",
                "todo",
                "channel-shuffle",
            ),
            # Cell 05 ? todo, architecture
            _cell(
                "code",
                r"""
# TODO 2: channel 절반은 identity, 절반은 pointwise-depthwise-pointwise branch로 처리하세요.
raise NotImplementedError
""",
                r"""
class ShuffleUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.branch = nn.Sequential(
            nn.Conv2d(half, half, 1),
            nn.ReLU(),
            nn.Conv2d(half, half, 3, padding=1, groups=half),
            nn.Conv2d(half, half, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        left, right = x.chunk(2, 1)
        return channel_shuffle(torch.cat([left, self.branch(right)], 1), 2)


unit = ShuffleUnit(8)
probe = torch.rand(2, 8, 8, 8)
assert unit(probe).shape == probe.shape and unit.branch[2].groups == 4
""",
                "todo",
                "architecture",
            ),
            # Cell 06 ? todo, latency, visualization
            _cell(
                "code",
                r"""
# TODO 3: tiny classifier를 학습하고 forward latency를 FLOPs와 별개로 기록하세요.
raise NotImplementedError
""",
                r"""
class ShuffleMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.units = nn.Sequential(ShuffleUnit(8), ShuffleUnit(8))
        self.head = nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        return self.head(F.max_pool2d(self.units(F.relu(self.stem(x))), 2).flatten(1))


model = ShuffleMini()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
losses = []
for _ in range(35):
    optimizer.zero_grad()
    loss = F.cross_entropy(
        model(images[image_train_idx]), image_labels[image_train_idx]
    )
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    for _ in range(10):
        model(images[:16])
    ACCELERATOR.synchronize()
    start = time.perf_counter()
    for _ in range(80):
        model(images[:16])
    ACCELERATOR.synchronize()
    milliseconds = (time.perf_counter() - start) * 1000 / 80
    accuracy = (
        (model(images[image_test_idx]).argmax(-1) == image_labels[image_test_idx])
        .float()
        .mean()
        .item()
    )
assert losses[-1] < losses[0] and accuracy > 0.65 and milliseconds > 0
plt.bar(
    ["parameters", "latency×100"],
    [sum(p.numel() for p in model.parameters()), milliseconds * 100],
)
plt.title("서로 다른 단위이므로 직접 비교 금지")
plt.show()
print(
    f"accuracy={accuracy:.1%}, local {DEVICE.type.upper()} "
    f"latency={milliseconds:.3f} ms/batch"
)
""",
                "todo",
                "latency",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class ShufflenetV2PortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    channels: int = 8
    learning_rate: float = 0.02
    steps: int = 14


def prepare_batch() -> tuple[Tensor, Tensor]:
    return images[image_train_idx[:160]], image_labels[image_train_idx[:160]]


def channel_shuffle(inputs: Tensor, groups: int = 2) -> Tensor:
    batch, channels, height, width = inputs.shape
    grouped = inputs.reshape(batch, groups, channels // groups, height, width)
    shuffled = grouped.transpose(1, 2).contiguous()
    return shuffled.reshape(batch, channels, height, width)


class ShufflePortfolioBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.branch = nn.Sequential(
            nn.Conv2d(half, half, 1),
            nn.ReLU(),
            nn.Conv2d(half, half, 3, padding=1, groups=half),
            nn.Conv2d(half, half, 1),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        identity, branch_input = inputs.chunk(2, dim=1)
        branch_output = self.branch(branch_input)
        return channel_shuffle(torch.cat([identity, branch_output], dim=1))


class ShufflenetV2PortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Conv2d(1, config.channels, 3, padding=1)
        self.block = ShufflePortfolioBlock(config.channels)
        self.classifier = nn.Linear(config.channels, 4)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.relu(self.stem(inputs))
        hidden = self.block(hidden).mean(dim=(2, 3))
        return self.classifier(hidden)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)

    def training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.compute_loss(self(inputs), targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: ShufflenetV2PortfolioModel) -> list[float]:
    inputs, targets = prepare_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, targets)
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
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def evaluate_portfolio_model(model: ShufflenetV2PortfolioModel) -> dict[str, float]:
    benchmark_batch = images[image_test_idx[:32]]
    model.eval()
    with torch.no_grad():
        logits = model(images[image_test_idx])
        accuracy = (logits.argmax(1) == image_labels[image_test_idx]).float().mean()
        for _ in range(3):
            model(benchmark_batch)
        if benchmark_batch.is_cuda:
            torch.cuda.synchronize(benchmark_batch.device)
        started = time.perf_counter()
        for _ in range(25):
            model(benchmark_batch)
        if benchmark_batch.is_cuda:
            torch.cuda.synchronize(benchmark_batch.device)
        latency_ms = 1000.0 * (time.perf_counter() - started) / 25
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    depthwise_groups = model.block.branch[2].groups
    return {
        "test_accuracy": float(accuracy.cpu()),
        "parameter_count": float(parameter_count),
        "depthwise_groups": float(depthwise_groups),
        "measured_batch_latency_ms": float(latency_ms),
        "measured_images_per_second": float(32_000.0 / latency_ms),
    }


portfolio_model = ShufflenetV2PortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 09 ? Once-for-All
    FieldPaperSpec(
        field_id="distillation_compression",
        field_title="지식 증류 · 모델 경량화",
        number=9,
        slug="once_for_all",
        short_title="Once-for-All",
        paper_title=(
            "Once-for-All: Train One Network and Specialize it for Efficient Deployment"
        ),
        authors="Han Cai et al.",
        year=2020,
        primary_url="https://openreview.net/forum?id=HylxE1HKwS",
        venue="ICLR 2020",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/compression_bench.npz",
        prerequisites="PyTorch tensor와 autograd, softmax·cross entropy, 선형대수 기초",
        reproduction_goal=(
            "여러 width를 공유하는 supernet을 progressive shrinking으로 학습하고 latency budget"
            "에 맞는 subnet을 선택한다."
        ),
        original_scale=(
            "원 논문은 depth·width·kernel·resolution과 실제 여러 하드웨어를 지원한다. 실습은 width와 제공"
            "된 합성 latency table만 쓴다."
        ),
        mappings=(
            (
                "Figure 1: train once, specialize",
                "Task 1–3",
                "공유 weight/subnet 선택",
            ),
            (
                "§3.1: elastic kernel/depth/width/resolution",
                "Task 1",
                "width slicing",
            ),
            (
                "§3.2: progressive shrinking",
                "Task 2",
                "큰 subnet→작은 subnet 순서",
            ),
            (
                "§3.3: specialization",
                "Task 3",
                "validation 선택 뒤 고정 subnet test 평가",
            ),
        ),
        cells=(
            # Cell 01 ? shared
            _cell(
                "markdown",
                r"""
## 미니 재현
모든 기기마다 새 모델을 훈련하지 않고 하나의 supernet weight를 잘라 여러 subnet으로 사용합니다.
""",
                r"""
## 미니 재현
모든 기기마다 새 모델을 훈련하지 않고 하나의 supernet weight를 잘라 여러 subnet으로 사용합니다.
""",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$w^*=\arg\min_w\mathcal L_{train}(N(S,w)),\quad
s^*=\arg\max_{s\in S,\,\mathrm{lat}(s)\le b}
\mathrm{Acc}(N(s,w^*))$$

- **기호 정의:** S는 subnet 공간, w는 공유 weight, b는 배포 latency budget입니다.
- **수식의 역할:** 한 supernet을 progressive shrinking으로 학습한 뒤 장치 제약에 맞춰 전문화합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  Figure 1: train once, specialize의 흐름을 명시합니다.
- **입출력 shape:** 특징 [B, 32] → width별 hidden slice → logit [B, 4]
- **평가:** width별 accuracy, 제공 latency table의 budget 만족 여부
- **원문 대비 한계:** 원 논문은 depth·width·kernel·resolution과 실제 여러 하드웨어를 지원한다. 실습은 width와 제공된 합성
latency table만 쓴다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                r"""
## 포트폴리오 해설: 정확도와 효율의 계약

$$w^*=\arg\min_w\mathcal L_{train}(N(S,w)),\quad
s^*=\arg\max_{s\in S,\,\mathrm{lat}(s)\le b}
\mathrm{Acc}(N(s,w^*))$$

- **기호 정의:** S는 subnet 공간, w는 공유 weight, b는 배포 latency budget입니다.
- **수식의 역할:** 한 supernet을 progressive shrinking으로 학습한 뒤 장치 제약에 맞춰 전문화합니다.
- **구현 이유:** 정확도만 남기면 경량화의 비용 절감 주장을 검증할 수 없으므로,
  핵심 변환과 품질·비용 측정을 독립 함수와 method로 분리했습니다.
- **Task/코드 대응:** Task P의 논문 전용 class와 `forward`, `compute_loss`,
  `training_step`, `fit_portfolio_model`, `evaluate_portfolio_model`이
  Figure 1: train once, specialize의 흐름을 명시합니다.
- **입출력 shape:** 특징 [B, 32] → width별 hidden slice → logit [B, 4]
- **평가:** width별 accuracy, 제공 latency table의 budget 만족 여부
- **원문 대비 한계:** 원 논문은 depth·width·kernel·resolution과 실제 여러 하드웨어를 지원한다. 실습은 width와 제공된 합성
latency table만 쓴다.

경량화 실험에서는 정확도만 보지 않고 parameter·sparsity·bit·latency 중 논문이
주장한 비용을 함께 기록해야 합니다. 아래 코드는 단계별 중간값을 남깁니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? setup, local-data
            _cell(
                "code",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                r"""
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

torch.manual_seed(20260814)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
torch.set_default_device(DEVICE)
print(ACCELERATOR.summary())


def locate(relative):
    for base in (Path.cwd(), *Path.cwd().parents):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


dataset_path = locate("data/field_curriculum/compression_bench.npz")
raw = np.load(dataset_path, allow_pickle=False)
features = torch.from_numpy(raw["features"]).float().to(DEVICE)
labels = torch.from_numpy(raw["labels"]).long().to(DEVICE)
teacher_logits = torch.from_numpy(raw["teacher_logits"]).float().to(DEVICE)
teacher_weights = torch.from_numpy(raw["teacher_weights"]).float().to(DEVICE)
calibration = torch.from_numpy(raw["calibration_activations"]).float().to(DEVICE)
dense_weights = torch.from_numpy(raw["dense_weights"]).float().to(DEVICE)
train_idx = torch.from_numpy(raw["train_idx"]).long().to(DEVICE)
test_idx = torch.from_numpy(raw["test_idx"]).long().to(DEVICE)
images = torch.from_numpy(raw["images"]).float().to(DEVICE)
image_labels = torch.from_numpy(raw["image_labels"]).long().to(DEVICE)
image_train_idx = torch.from_numpy(raw["image_train_idx"]).long().to(DEVICE)
image_test_idx = torch.from_numpy(raw["image_test_idx"]).long().to(DEVICE)
token_embeddings = torch.from_numpy(raw["token_embeddings"]).float().to(DEVICE)
teacher_hidden = torch.from_numpy(raw["teacher_hidden"]).float().to(DEVICE)
teacher_attention = torch.from_numpy(raw["teacher_attention"]).float().to(DEVICE)
candidate_configs = torch.from_numpy(raw["candidate_configs"]).float()
assert features.shape == (512, 32) and teacher_logits.shape == (512, 4)
assert images.shape == (320, 1, 16, 16)
assert teacher_hidden.shape == (256, 6, 24)
print(
    dataset_path.name, "features", tuple(features.shape), "images", tuple(images.shape)
)
""",
                "setup",
                "local-data",
            ),
            # Cell 04 ? todo, supernet
            _cell(
                "code",
                r"""
# TODO 1: 최대 width=24의 weight를 slicing해 8/12/16/24 subnet을 제공하세요.
raise NotImplementedError
""",
                r"""
class WidthSupernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_weight = nn.Parameter(torch.randn(24, 32) * 0.08)
        self.hidden_bias = nn.Parameter(torch.zeros(24))
        self.out_weight = nn.Parameter(torch.randn(4, 24) * 0.08)
        self.out_bias = nn.Parameter(torch.zeros(4))

    def forward(self, x, width):
        hidden = F.relu(
            F.linear(x, self.hidden_weight[:width], self.hidden_bias[:width])
        )
        return F.linear(hidden, self.out_weight[:, :width], self.out_bias)


supernet = WidthSupernet()
assert all(supernet(features[:3], w).shape == (3, 4) for w in (8, 12, 16, 24))
""",
                "todo",
                "supernet",
            ),
            # Cell 05 ? todo, progressive-shrinking
            _cell(
                "code",
                r"""
# TODO 2: 큰 width에서 시작해 작은 width를 차례로 포함하는 progressive shrinking을 수행하세요.
raise NotImplementedError
""",
                r"""
optimizer = torch.optim.Adam(supernet.parameters(), lr=0.03)
widths = (24, 16, 12, 8)
history = []
for phase in range(len(widths)):
    active = widths[: phase + 1]
    for _ in range(18):
        optimizer.zero_grad()
        loss = sum(
            F.cross_entropy(supernet(features[train_idx], w), labels[train_idx])
            for w in active
        ) / len(active)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
assert history[-1] < history[0]
with torch.no_grad():
    accuracies = {
        w: (supernet(features[test_idx], w).argmax(-1) == labels[test_idx])
        .float()
        .mean()
        .item()
        for w in widths
    }
assert min(accuracies.values()) > 0.7
""",
                "todo",
                "progressive-shrinking",
            ),
            # Cell 06 ? todo, specialization, visualization
            _cell(
                "code",
                r"""
# TODO 3: 제공된 latency table에서 budget 이하 후보 중 정확도가 가장 높은 subnet을 고르세요.
raise NotImplementedError
""",
                r"""
budget_ms = 1.3
candidates = []
for width, depth, kernel, latency in candidate_configs.tolist():
    width = int(width)
    accuracy = accuracies.get(
        width, accuracies[min(widths, key=lambda w: abs(w - width))]
    )
    candidates.append(
        {
            "width": width,
            "depth": int(depth),
            "kernel": int(kernel),
            "latency": latency,
            "accuracy": accuracy,
        }
    )
feasible = [row for row in candidates if row["latency"] <= budget_ms]
chosen = max(feasible, key=lambda row: (row["accuracy"], -row["latency"]))
assert chosen["latency"] <= budget_ms and chosen in feasible
plt.scatter([r["latency"] for r in candidates], [r["accuracy"] for r in candidates])
plt.axvline(budget_ms, color="red", linestyle="--")
plt.xlabel("synthetic latency (ms)")
plt.ylabel("accuracy")
plt.show()
print("selected subnet:", chosen)
""",
                "todo",
                "specialization",
                "visualization",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                r"""
### Task P1 · 압축 대상과 핵심 모델

tensor shape와 비용 단위를 먼저 확인한 뒤 논문 전용 class의 `forward`와
`compute_loss`에 정확도·표현 보존 계약을 명시합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 layer/pipeline과 비용 평가를 구조화해 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 16
    learning_rate: float = 0.02
    steps: int = 12


def prepare_batch() -> tuple[Tensor, ...]:
    raise NotImplementedError("TODO P-1: local batch and shape")


class OnceForAllPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, *inputs: Tensor):
        raise NotImplementedError("TODO P-1: forward")

    def compute_loss(self, *outputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: update")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    widths: tuple[int, ...] = (24, 16, 12, 8)
    latency_budget_ms: float = 1.3
    learning_rate: float = 0.025
    steps_per_phase: int = 5


def prepare_batch() -> tuple[Tensor, Tensor]:
    fit_index = train_idx[:-64]
    return features[fit_index], labels[fit_index]


def width_latency_table(widths: tuple[int, ...]) -> dict[int, float]:
    '''Collapse the fixture to the width-only search space implemented below.'''
    rows = candidate_configs.tolist()
    table = {}
    for width in widths:
        latencies = [row[3] for row in rows if int(row[0]) == width]
        if not latencies:
            raise ValueError(f"missing latency proxy for width {width}")
        table[width] = float(min(latencies))
    return table


class OnceForAllPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        maximum = max(config.widths)
        self.hidden_weight = nn.Parameter(
            torch.randn(maximum, features.shape[1]) * 0.08
        )
        self.hidden_bias = nn.Parameter(torch.zeros(maximum))
        self.output_weight = nn.Parameter(torch.randn(4, maximum) * 0.08)
        self.output_bias = nn.Parameter(torch.zeros(4))

    def forward(self, inputs: Tensor, width: int) -> Tensor:
        hidden = F.relu(
            F.linear(
                inputs,
                self.hidden_weight[:width],
                self.hidden_bias[:width],
            )
        )
        return F.linear(hidden, self.output_weight[:, :width], self.output_bias)

    def compute_loss(
        self, inputs: Tensor, targets: Tensor, widths: tuple[int, ...]
    ) -> Tensor:
        losses = [F.cross_entropy(self(inputs, width), targets) for width in widths]
        return torch.stack(losses).mean()

    def training_step(
        self,
        inputs: Tensor,
        targets: Tensor,
        widths: tuple[int, ...],
    ) -> Tensor:
        return self.compute_loss(inputs, targets, widths)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
""",
                r"""
### Task P2 · 학습·pruning·distillation update

`fit_portfolio_model`에서 optimizer, mask/EMA/quantization 경계와 update 순서를
드러냅니다. 비용 절감이 학습 경로에 미치는 영향을 함께 읽으세요.
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
def fit_portfolio_model(model: OnceForAllPortfolioModel) -> list[float]:
    inputs, targets = prepare_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for phase in range(len(model.config.widths)):
        active = model.config.widths[: phase + 1]
        for _ in range(model.config.steps_per_phase):
            loss = model.training_step(inputs, targets, active)
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
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                r"""
### Task P3 · 품질과 비용을 함께 평가

accuracy 또는 reconstruction 품질과 parameter·sparsity·bit·latency를 별도로
기록합니다. 로컬 지표는 원문 하드웨어 benchmark의 대체값이 아닙니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: quality and cost")
""",
                r"""
def select_subnet_under_budget(
    model: OnceForAllPortfolioModel,
    budget_ms: float,
    validation_inputs: Tensor,
    validation_targets: Tensor,
) -> dict[str, float]:
    latency_table = width_latency_table(model.config.widths)
    candidates = []
    with torch.no_grad():
        for width, latency in latency_table.items():
            if latency <= budget_ms:
                logits = model(validation_inputs, width)
                accuracy = (
                    (logits.argmax(1) == validation_targets).float().mean()
                )
                candidates.append(
                    {
                        "width": float(width),
                        "latency_ms": latency,
                        "validation_accuracy": float(accuracy.cpu()),
                    }
                )
    if not candidates:
        raise ValueError(f"no subnet satisfies latency budget {budget_ms}")
    return max(
        candidates,
        key=lambda item: (
            item["validation_accuracy"],
            -item["latency_ms"],
        ),
    )


def evaluate_portfolio_model(model: OnceForAllPortfolioModel) -> dict[str, float]:
    validation_index = train_idx[-64:]
    validation_inputs = features[validation_index]
    validation_targets = labels[validation_index]
    assert not torch.isin(validation_index, test_idx).any()
    budget = model.config.latency_budget_ms
    selected = select_subnet_under_budget(
        model,
        budget,
        validation_inputs,
        validation_targets,
    )
    selected_width = int(selected["width"])
    with torch.no_grad():
        test_logits = model(features[test_idx], selected_width)
        test_accuracy = (
            (test_logits.argmax(1) == labels[test_idx]).float().mean()
        )
    metrics = {
        "selected_test_accuracy": float(test_accuracy.cpu()),
        "fit_sample_count": float(len(train_idx) - len(validation_index)),
        "validation_sample_count": float(len(validation_index)),
        "test_sample_count": float(len(test_idx)),
    }
    metrics.update({f"selected_{key}": value for key, value in selected.items()})
    metrics["selection_within_budget"] = float(selected["latency_ms"] <= budget)
    return metrics


portfolio_model = OnceForAllPortfolioModel(PortfolioConfig())
portfolio_history = fit_portfolio_model(portfolio_model)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["selection_within_budget"] == 1.0
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
