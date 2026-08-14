"""Graph learning and recommendation portfolio paper specifications."""

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
    # 00 ? DeepWalk: 그래프를 문장처럼 읽기
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=0,
        slug="deepwalk",
        short_title="DeepWalk: 그래프를 문장처럼 읽기",
        paper_title="DeepWalk: Online Learning of Social Representations",
        authors="Bryan Perozzi, Rami Al-Rfou, Steven Skiena",
        year=2014,
        primary_url="https://arxiv.org/abs/1403.6652",
        venue="KDD",
        difficulty="초급",
        expected_minutes=60,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="확률, random walk, softmax, 임베딩",
        reproduction_goal="절단 random walk를 문장으로 보고 Skip-gram 문맥쌍을 만든 뒤 작은 그래프의 노드 임베딩을 학습합니다.",
        original_scale=(
            "BlogCatalog/Flickr/YouTube와 hierarchical softmax 대신 18개 노드, full-sof"
            "tmax를 사용합니다."
        ),
        mappings=(
            (
                "§3.1, §4.2 및 Algorithm 1",
                "각 정점에서 절단 random walk 생성",
                "길이·이웃 전이·seed 재현성",
            ),
            (
                "§4.2.1 및 Algorithm 2",
                "walk의 중심-문맥 쌍 생성",
                "window 경계와 self-pair 제외",
            ),
            (
                "§3.3, Eq. (2) 및 Algorithm 2 lines 3–4",
                "Skip-gram 조건부 우도 최적화",
                "초기/최종 cross-entropy",
            ),
            (
                "Figure 3",
                "walk→문맥→표현 파이프라인",
                "동일 label/상이 label cosine",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\max_{\Phi}\sum_{v\in V}\sum_{u\in\mathcal{N}_w(v)}\log p(u\mid v)$$

- **기호 정의:** v는 중심 노드, u는 walk 문맥, Φ는 노드 임베딩, N_w는 window 문맥입니다.
- **수식의 역할:** random walk를 문장으로 바꾼 뒤 Skip-gram 우도를 높여 구조적 근접성을 학습합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.1, §4.2 및 Algorithm 1을 구조화합니다.
- **입출력 shape:** 노드 id [P] → 전체 노드 logit [P, N]
- **평가:** cross-entropy 감소와 동일 클래스 노드의 평균 cosine 분리도
- **원문 대비 한계:** BlogCatalog/Flickr/YouTube와 hierarchical softmax 대신 18개 노드, full-softmax를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\max_{\Phi}\sum_{v\in V}\sum_{u\in\mathcal{N}_w(v)}\log p(u\mid v)$$

- **기호 정의:** v는 중심 노드, u는 walk 문맥, Φ는 노드 임베딩, N_w는 window 문맥입니다.
- **수식의 역할:** random walk를 문장으로 바꾼 뒤 Skip-gram 우도를 높여 구조적 근접성을 학습합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.1, §4.2 및 Algorithm 1을 구조화합니다.
- **입출력 shape:** 노드 id [P] → 전체 노드 logit [P, N]
- **평가:** cross-entropy 감소와 동일 클래스 노드의 평균 cosine 분리도
- **원문 대비 한계:** BlogCatalog/Flickr/YouTube와 hierarchical softmax 대신 18개 노드, full-softmax를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "markdown",
                r"""
## 1. Algorithm 1: 절단 random walk

DeepWalk의 핵심 전환은 **정점=단어, walk=문장**입니다. 아래 구현은 현재 정점의 이웃에서만
균등 표본을 뽑습니다. 논문의 병렬/streaming 실행은 생략하지만 표본 과정은 같습니다.
""",
                r"""
## 1. Algorithm 1: 절단 random walk

DeepWalk의 핵심 전환은 **정점=단어, walk=문장**입니다. 아래 구현은 현재 정점의 이웃에서만
균등 표본을 뽑습니다. 논문의 병렬/streaming 실행은 생략하지만 표본 과정은 같습니다.
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 00-1: seed를 받는 균등 random walk를 구현하세요.
def random_walk(adj: Tensor, start: int, length: int, seed: int) -> list[int]:
    raise NotImplementedError("TODO 00-1")


walk = random_walk(adjacency, 0, 12, 11)
""",
                r"""
def random_walk(adj: Tensor, start: int, length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    walk = [int(start)]
    while len(walk) < length:
        neighbors = torch.where(adj[walk[-1]] > 0)[0].numpy()
        if len(neighbors) == 0:
            break
        walk.append(int(rng.choice(neighbors)))
    return walk


walk = random_walk(adjacency, 0, 12, 11)
assert len(walk) == 12
assert all(adjacency[a, b] == 1 for a, b in zip(walk, walk[1:]))
assert walk == random_walk(adjacency, 0, 12, 11)
print("walk:", walk)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 00-2: Algorithm 2처럼 window 안의 (center, context) 쌍을 만드세요.
def skipgram_pairs(walks: list[list[int]], window: int = 2) -> Tensor:
    raise NotImplementedError("TODO 00-2")
""",
                r"""
def skipgram_pairs(walks: list[list[int]], window: int = 2) -> Tensor:
    pairs = []
    for seq in walks:
        for i, center in enumerate(seq):
            for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                if i != j:
                    pairs.append((center, seq[j]))
    return torch.tensor(pairs, dtype=torch.long)


walks = [
    random_walk(adjacency, s, 12, 100 * s + r)
    for s in range(len(features))
    for r in range(2)
]
pairs = skipgram_pairs(walks)
pairs = pairs[torch.randperm(len(pairs))[: min(2400, len(pairs))]]
assert pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) > len(walks)
print("walks / positive pairs:", len(walks), len(pairs))
""",
                "task",
            ),
            # Cell 06 ? task
            _cell(
                "code",
                r"""
# TODO 00-3: Eq. (2)의 Skip-gram 목적을 작은 full-softmax로 학습하세요.
raise NotImplementedError("TODO 00-3")
""",
                r"""
n, dim = len(features), 8
pairs_device = ACCELERATOR.move(pairs)
input_embed = ACCELERATOR.move(nn.Embedding(n, dim))
output_embed = ACCELERATOR.move(nn.Embedding(n, dim))
optimizer = torch.optim.Adam(
    [*input_embed.parameters(), *output_embed.parameters()], lr=0.04
)
losses = []
for step in range(55):
    logits = input_embed(pairs_device[:, 0]) @ output_embed.weight.T
    loss = F.cross_entropy(logits, pairs_device[:, 1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
node_embedding = F.normalize(input_embed.weight.detach(), dim=1).cpu()
assert losses[-1] < losses[0] * 0.9 and torch.isfinite(node_embedding).all()
print(f"Skip-gram CE: {losses[0]:.3f} → {losses[-1]:.3f}")
""",
                "task",
            ),
            # Cell 07 ? metric
            _cell(
                "code",
                r"""
# TODO 00-4: 학습곡선과 동일/상이 label cosine을 비교하세요.
raise NotImplementedError("TODO 00-4")
""",
                r"""
similarity = node_embedding @ node_embedding.T
eye = torch.eye(len(labels), dtype=torch.bool)
same = (labels[:, None] == labels[None, :]) & ~eye
different = labels[:, None] != labels[None, :]
same_mean = float(similarity[same].mean())
diff_mean = float(similarity[different].mean())
assert same_mean > diff_mean
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set(title="Skip-gram loss", xlabel="step")
ax[1].bar(["same", "different"], [same_mean, diff_mean])
ax[1].set(title="cosine by label")
plt.tight_layout()
plt.show()
print({"same_cosine": round(same_mean, 3), "different_cosine": round(diff_mean, 3)})
""",
                "metric",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    walks_per_node: int = 2
    walk_length: int = 10
    context_window: int = 2
    max_pairs: int = 720
    learning_rate: float = 0.04
    steps: int = 16


def preprocess_graph(graph: Tensor, config: PortfolioConfig) -> Tensor:
    raise NotImplementedError("TODO P-1: walk corpus and Skip-gram pairs")


class DeepwalkPortfolioModel(nn.Module):
    def __init__(self, node_count: int, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: center/context embeddings")

    def forward(self, node_ids: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: full-softmax logits")

    def compute_loss(self, pairs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: Skip-gram objective")

    def training_step(self, pairs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    walks_per_node: int = 2
    walk_length: int = 10
    context_window: int = 2
    max_pairs: int = 720
    learning_rate: float = 0.04
    steps: int = 16


def preprocess_graph(graph: Tensor, config: PortfolioConfig) -> Tensor:
    walks = [
        random_walk(
            graph,
            start=node_id,
            length=config.walk_length,
            seed=1_000 * node_id + walk_id,
        )
        for node_id in range(len(graph))
        for walk_id in range(config.walks_per_node)
    ]
    pairs = skipgram_pairs(walks, window=config.context_window)
    generator = torch.Generator().manual_seed(7)
    order = torch.randperm(len(pairs), generator=generator)
    return pairs[order[: config.max_pairs]]


class DeepwalkPortfolioModel(nn.Module):
    def __init__(self, node_count: int, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.center = nn.Embedding(node_count, config.embedding_dim)
        self.context = nn.Embedding(node_count, config.embedding_dim)

    def forward(self, node_ids: Tensor) -> Tensor:
        return self.center(node_ids) @ self.context.weight.T

    def compute_loss(self, pairs: Tensor) -> Tensor:
        return F.cross_entropy(self(pairs[:, 0]), pairs[:, 1])

    def training_step(self, pairs: Tensor) -> Tensor:
        return self.compute_loss(pairs)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 10 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: DeepwalkPortfolioModel,
    pairs: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(pairs)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 13 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(model: DeepwalkPortfolioModel) -> dict[str, float]:
    embedding = F.normalize(model.center.weight.detach(), dim=1)
    similarity = embedding @ embedding.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)
    separation = similarity.cpu()[same & ~eye].mean() - similarity.cpu()[~same].mean()
    return {"class_cosine_gap": float(separation)}


portfolio_config = PortfolioConfig()
portfolio_pairs = ACCELERATOR.move(preprocess_graph(adjacency, portfolio_config))
portfolio_model = ACCELERATOR.move(
    DeepwalkPortfolioModel(len(features), portfolio_config)
)
portfolio_history = fit_portfolio_model(portfolio_model, portfolio_pairs)
portfolio_metrics = evaluate_portfolio_model(portfolio_model)
assert len(portfolio_pairs) > len(features)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 01 ? node2vec: BFS와 DFS 사이
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=1,
        slug="node2vec",
        short_title="node2vec: BFS와 DFS 사이",
        paper_title="node2vec: Scalable Feature Learning for Networks",
        authors="Aditya Grover, Jure Leskovec",
        year=2016,
        primary_url="https://arxiv.org/abs/1607.00653",
        venue="KDD",
        difficulty="초급",
        expected_minutes=55,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="DeepWalk, 조건부 확률, 그래프 거리",
        reproduction_goal="p·q로 조절되는 2차 biased walk를 구현하고 BFS형/DFS형 탐색 통계를 비교합니다.",
        original_scale="대규모 multi-label/link prediction 대신 동일한 18노드 그래프에서 walk 통계를 재현합니다.",
        mappings=(
            (
                "§3.2.2, Eq. (2)",
                "거리별 α_pq 전이 bias",
                "return/in/out 가중치",
            ),
            (
                "Figure 2",
                "직전 정점 t를 조건으로 다음 x 선택",
                "확률 합과 허용 edge",
            ),
            (
                "§3.2.3 및 Algorithm 1",
                "2차 random walk corpus",
                "seed 재현성",
            ),
            (
                "§3.2.2 설명",
                "q에 따른 BFS/DFS trade-off",
                "start 거리·고유 정점 수",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\pi_{vx}=\alpha_{pq}(t,x)w_{vx},\quad \alpha_{pq}\in\{1/p,1,1/q\}$$

- **기호 정의:** t는 이전 노드, v는 현재 노드, x는 후보, p와 q는 귀환·탐색 bias입니다.
- **수식의 역할:** 2차 Markov 전이로 BFS형 지역 탐색과 DFS형 외곽 탐색을 연속적으로 조절합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.2.2, Eq. (2)을 구조화합니다.
- **입출력 shape:** 인접행렬 [N, N] → biased walk와 중심-문맥 쌍 [P, 2]
- **평가:** 전이확률 행합, walk 유효성, q에 따른 시작점 평균거리
- **원문 대비 한계:** 대규모 multi-label/link prediction 대신 동일한 18노드 그래프에서 walk 통계를 재현합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\pi_{vx}=\alpha_{pq}(t,x)w_{vx},\quad \alpha_{pq}\in\{1/p,1,1/q\}$$

- **기호 정의:** t는 이전 노드, v는 현재 노드, x는 후보, p와 q는 귀환·탐색 bias입니다.
- **수식의 역할:** 2차 Markov 전이로 BFS형 지역 탐색과 DFS형 외곽 탐색을 연속적으로 조절합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.2.2, Eq. (2)을 구조화합니다.
- **입출력 shape:** 인접행렬 [N, N] → biased walk와 중심-문맥 쌍 [P, 2]
- **평가:** 전이확률 행합, walk 유효성, q에 따른 시작점 평균거리
- **원문 대비 한계:** 대규모 multi-label/link prediction 대신 동일한 18노드 그래프에서 walk 통계를 재현합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "markdown",
                r"""
## 1. Eq. (2): 두 번째 차수 전이
직전 정점 `t`, 현재 정점 `v`, 후보 `x`의 최단거리 0/1/2에 따라 `1/p, 1, 1/q`를 곱합니다.
""",
                r"""
## 1. Eq. (2): 두 번째 차수 전이
직전 정점 `t`, 현재 정점 `v`, 후보 `x`의 최단거리 0/1/2에 따라 `1/p, 1, 1/q`를 곱합니다.
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 01-1: 후보별 비정규화 α_pq와 정규화 확률을 반환하세요.
def transition_probs(adj: Tensor, previous: int, current: int, p: float, q: float):
    raise NotImplementedError("TODO 01-1")
""",
                r"""
def transition_probs(adj: Tensor, previous: int, current: int, p: float, q: float):
    candidates = torch.where(adj[current] > 0)[0]
    weights = []
    for x in candidates.tolist():
        distance = 0 if x == previous else (1 if adj[previous, x] > 0 else 2)
        weights.append(1 / p if distance == 0 else (1.0 if distance == 1 else 1 / q))
    probs = torch.tensor(weights, dtype=torch.float32)
    return candidates, probs / probs.sum()


candidates, probs = transition_probs(
    adjacency, 0, int(torch.where(adjacency[0] > 0)[0][0]), 2.0, 0.5
)
assert torch.isclose(probs.sum(), torch.tensor(1.0)) and (probs > 0).all()
print(list(zip(candidates.tolist(), probs.round(decimals=3).tolist())))
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 01-2: 첫 step은 균등, 이후는 Eq. (2)를 쓰는 biased walk를 구현하세요.
def biased_walk(adj: Tensor, start: int, length: int, p: float, q: float, seed: int):
    raise NotImplementedError("TODO 01-2")
""",
                r"""
def biased_walk(adj: Tensor, start: int, length: int, p: float, q: float, seed: int):
    rng = np.random.default_rng(seed)
    walk = [start]
    while len(walk) < length:
        nbr = torch.where(adj[walk[-1]] > 0)[0]
        if len(nbr) == 0:
            break
        if len(walk) == 1:
            nxt = int(rng.choice(nbr.numpy()))
        else:
            cand, pr = transition_probs(adj, walk[-2], walk[-1], p, q)
            nxt = int(rng.choice(cand.numpy(), p=pr.numpy()))
        walk.append(nxt)
    return walk


probe = biased_walk(adjacency, 0, 20, p=1.0, q=0.5, seed=9)
assert len(probe) == 20 and all(adjacency[a, b] for a, b in zip(probe, probe[1:]))
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 01-3: q=0.5(바깥 탐색)와 q=2(BFS형)의 walk 통계를 비교·시각화하세요.
raise NotImplementedError("TODO 01-3")
""",
                r"""
def shortest_distances(adj: Tensor, start: int) -> Tensor:
    dist = torch.full((len(adj),), 999, dtype=torch.long)
    dist[start] = 0
    frontier = [start]
    while frontier:
        u = frontier.pop(0)
        for v in torch.where(adj[u] > 0)[0].tolist():
            if dist[v] == 999:
                dist[v] = dist[u] + 1
                frontier.append(v)
    return dist


distances = shortest_distances(adjacency, 0)
stats = {}
for name, q in [("DFS-like q=.5", 0.5), ("BFS-like q=2", 2.0)]:
    corpus = [biased_walk(adjacency, 0, 24, 1.0, q, 100 + r) for r in range(160)]
    stats[name] = [float(distances[torch.tensor(w)].float().mean()) for w in corpus]
means = {k: float(np.mean(v)) for k, v in stats.items()}
assert means["DFS-like q=.5"] >= means["BFS-like q=2"] - 0.05
plt.boxplot(stats.values(), tick_labels=stats.keys())
plt.ylabel("mean distance from start")
plt.xticks(rotation=10)
plt.show()
print({k: round(v, 3) for k, v in means.items()})
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    return_bias: float = 1.0
    inout_bias: float = 0.5
    walks_per_node: int = 2
    walk_length: int = 10
    context_window: int = 2
    max_pairs: int = 720
    learning_rate: float = 0.04
    steps: int = 16


def walk_context_pairs(walks: list[list[int]], window: int) -> Tensor:
    raise NotImplementedError("TODO P-1: center/context pairs")


def preprocess_graph(graph: Tensor, config: PortfolioConfig) -> Tensor:
    raise NotImplementedError("TODO P-1: p/q-biased walks")


class Node2VecPortfolioModel(nn.Module):
    def __init__(self, node_count: int, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: center/context embeddings")

    def forward(self, node_ids: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: full-softmax logits")

    def compute_loss(self, pairs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: Skip-gram objective")

    def training_step(self, pairs: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    return_bias: float = 1.0
    inout_bias: float = 0.5
    walks_per_node: int = 2
    walk_length: int = 10
    context_window: int = 2
    max_pairs: int = 720
    learning_rate: float = 0.04
    steps: int = 16


def walk_context_pairs(walks: list[list[int]], window: int) -> Tensor:
    pairs = []
    for walk in walks:
        for center_index, center in enumerate(walk):
            left = max(0, center_index - window)
            right = min(len(walk), center_index + window + 1)
            for context_index in range(left, right):
                if context_index != center_index:
                    pairs.append((center, walk[context_index]))
    return torch.tensor(pairs, dtype=torch.long)


def preprocess_graph(graph: Tensor, config: PortfolioConfig) -> Tensor:
    walks = [
        biased_walk(
            graph,
            start=node_id,
            length=config.walk_length,
            p=config.return_bias,
            q=config.inout_bias,
            seed=1_000 * node_id + walk_id,
        )
        for node_id in range(len(graph))
        for walk_id in range(config.walks_per_node)
    ]
    pairs = walk_context_pairs(walks, config.context_window)
    generator = torch.Generator().manual_seed(7)
    order = torch.randperm(len(pairs), generator=generator)
    return pairs[order[: config.max_pairs]]


class Node2VecPortfolioModel(nn.Module):
    def __init__(self, node_count: int, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.center = nn.Embedding(node_count, config.embedding_dim)
        self.context = nn.Embedding(node_count, config.embedding_dim)

    def forward(self, node_ids: Tensor) -> Tensor:
        return self.center(node_ids) @ self.context.weight.T

    def compute_loss(self, pairs: Tensor) -> Tensor:
        return F.cross_entropy(self(pairs[:, 0]), pairs[:, 1])

    def training_step(self, pairs: Tensor) -> Tensor:
        return self.compute_loss(pairs)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: Node2VecPortfolioModel,
    pairs: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(pairs)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: Node2VecPortfolioModel,
    graph: Tensor,
) -> dict[str, float]:
    embedding = F.normalize(model.center.weight.detach(), dim=1)
    similarity = embedding @ embedding.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)
    separation = similarity.cpu()[same & ~eye].mean() - similarity.cpu()[~same].mean()

    distances = shortest_distances(graph, 0)

    def mean_walk_radius(inout_bias: float) -> float:
        walks = [
            biased_walk(
                graph,
                start=0,
                length=20,
                p=model.config.return_bias,
                q=inout_bias,
                seed=10_000 + repeat,
            )
            for repeat in range(64)
        ]
        radii = [distances[torch.tensor(walk)].float().mean() for walk in walks]
        return float(torch.stack(radii).mean())

    configured_radius = mean_walk_radius(model.config.inout_bias)
    bfs_radius = mean_walk_radius(2.0)
    return {
        "class_cosine_gap": float(separation),
        "configured_walk_radius": configured_radius,
        "bfs_walk_radius": bfs_radius,
    }


portfolio_config = PortfolioConfig()
portfolio_pairs = ACCELERATOR.move(preprocess_graph(adjacency, portfolio_config))
portfolio_model = ACCELERATOR.move(
    Node2VecPortfolioModel(len(features), portfolio_config)
)
portfolio_history = fit_portfolio_model(portfolio_model, portfolio_pairs)
portfolio_metrics = evaluate_portfolio_model(portfolio_model, adjacency)
assert len(portfolio_pairs) > len(features)
assert np.isfinite(portfolio_history).all()
assert portfolio_metrics["configured_walk_radius"] >= 0.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 02 ? GCN: 정규화된 이웃 전파
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=2,
        slug="gcn",
        short_title="GCN: 정규화된 이웃 전파",
        paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
        authors="Thomas N. Kipf, Max Welling",
        year=2017,
        primary_url="https://arxiv.org/abs/1609.02907",
        venue="ICLR",
        difficulty="중급",
        expected_minutes=65,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="선형대수, PyTorch, cross entropy",
        reproduction_goal="renormalization trick과 2층 GCN을 구현해 일부 label만으로 노드를 분류합니다.",
        original_scale="Cora/Citeseer/Pubmed 대신 18개 노드의 homophilic 합성 그래프를 사용합니다.",
        mappings=(
            (
                "§2, Eq. (2)",
                "GCN layer-wise propagation",
                "shape와 유한값",
            ),
            (
                "§2.2, Eq. (8)",
                "A+I 및 대칭 degree 정규화",
                "대칭성·self-loop",
            ),
            (
                "§3.1, Eq. (9)",
                "2층 softmax 모형",
                "logit shape",
            ),
            (
                "§3.1, Eq. (10)",
                "labelled subset cross entropy",
                "train/test accuracy와 loss",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$H^{(l+1)}=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)})$$

- **기호 정의:** Ã=A+I, D̃는 Ã의 차수, H는 노드 표현, W는 학습 가능한 선형변환입니다.
- **수식의 역할:** self-loop와 대칭 정규화로 이웃 특징을 안정적으로 섞습니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §2, Eq. (2)을 구조화합니다.
- **입출력 shape:** 특징 [N, F]와 인접행렬 [N, N] → 클래스 logit [N, C]
- **평가:** labelled subset cross-entropy와 train/test accuracy
- **원문 대비 한계:** Cora/Citeseer/Pubmed 대신 18개 노드의 homophilic 합성 그래프를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$H^{(l+1)}=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)})$$

- **기호 정의:** Ã=A+I, D̃는 Ã의 차수, H는 노드 표현, W는 학습 가능한 선형변환입니다.
- **수식의 역할:** self-loop와 대칭 정규화로 이웃 특징을 안정적으로 섞습니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §2, Eq. (2)을 구조화합니다.
- **입출력 shape:** 특징 [N, F]와 인접행렬 [N, N] → 클래스 logit [N, C]
- **평가:** labelled subset cross-entropy와 train/test accuracy
- **원문 대비 한계:** Cora/Citeseer/Pubmed 대신 18개 노드의 homophilic 합성 그래프를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 02-1: Eq. (8)의 renormalization trick을 구현하세요.
def normalize_adjacency(adj: Tensor) -> Tensor:
    raise NotImplementedError("TODO 02-1")
""",
                r"""
def normalize_adjacency(adj: Tensor) -> Tensor:
    a = adj + torch.eye(len(adj), device=adj.device, dtype=adj.dtype)
    inv = a.sum(1).clamp_min(1).pow(-0.5)
    return inv[:, None] * a * inv[None, :]


a_hat = normalize_adjacency(adjacency)
assert torch.allclose(a_hat, a_hat.T) and (torch.diagonal(a_hat) > 0).all()
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 02-2: Eq. (2)와 Eq. (9)의 2층 GCN을 정의하세요.
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        raise NotImplementedError("TODO 02-2")

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        raise NotImplementedError("TODO 02-2")


class MiniGCN(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("TODO 02-2")

    def forward(self, x, a):
        raise NotImplementedError("TODO 02-2")
""",
                r"""
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.2)

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        return a @ x @ self.weight


class MiniGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = GCNLayer(features.shape[1], 12)
        self.g2 = GCNLayer(12, int(labels.max()) + 1)

    def forward(self, x, a):
        return self.g2(F.relu(self.g1(x, a)), a)


model = MiniGCN()
assert model(features, a_hat).shape == (len(features), int(labels.max()) + 1)
""",
                "task",
            ),
            # Cell 05 ? metric
            _cell(
                "code",
                r"""
# TODO 02-3: Eq. (10)처럼 train_mask 위치만으로 학습하세요.
raise NotImplementedError("TODO 02-3")
""",
                r"""
model = ACCELERATOR.move(model)
features_device, a_hat_device, labels_device = ACCELERATOR.move(features, a_hat, labels)
train_mask_device, test_mask_device = ACCELERATOR.move(train_mask, test_mask)
optimizer = torch.optim.Adam(model.parameters(), lr=0.04, weight_decay=5e-4)
losses = []
for _ in range(160):
    logits = model(features_device, a_hat_device)
    loss = F.cross_entropy(logits[train_mask_device], labels_device[train_mask_device])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    pred = model(features_device, a_hat_device).argmax(1)
    train_acc = float(
        (pred[train_mask_device] == labels_device[train_mask_device]).float().mean()
    )
    test_acc = float(
        (pred[test_mask_device] == labels_device[test_mask_device]).float().mean()
    )
assert losses[-1] < losses[0] and train_acc >= 0.8
plt.plot(losses)
plt.title(f"GCN loss · test acc={test_acc:.2f}")
plt.xlabel("step")
plt.show()
print({"train_accuracy": train_acc, "test_accuracy": test_acc})
""",
                "metric",
            ),
            # Cell 06 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 07 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    dropout: float = 0.15
    learning_rate: float = 0.03
    steps: int = 18


def preprocess_graph(graph: Tensor) -> Tensor:
    raise NotImplementedError("TODO P-1: graph preprocessing")


class GcnPortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        raise NotImplementedError("TODO P-1: two explicit GCN layers")

    def forward(self, inputs: Tensor, graph: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: paper equation")

    def compute_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        label_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(
        self,
        inputs: Tensor,
        graph: Tensor,
        targets: Tensor,
        label_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    dropout: float = 0.15
    learning_rate: float = 0.03
    steps: int = 18


def preprocess_graph(graph: Tensor) -> Tensor:
    return normalize_adjacency(graph)


class GcnPortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        self.config = config
        self.input_layer = GCNLayer(input_dim, config.hidden_dim)
        self.output_layer = GCNLayer(config.hidden_dim, class_count)

    def forward(self, inputs: Tensor, graph: Tensor) -> Tensor:
        hidden = F.relu(self.input_layer(inputs, graph))
        hidden = F.dropout(
            hidden,
            p=self.config.dropout,
            training=self.training,
        )
        return self.output_layer(hidden, graph)

    def compute_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        label_mask: Tensor,
    ) -> Tensor:
        return F.cross_entropy(outputs[label_mask], targets[label_mask])

    def training_step(
        self,
        inputs: Tensor,
        graph: Tensor,
        targets: Tensor,
        label_mask: Tensor,
    ) -> Tensor:
        outputs = self(inputs, graph)
        return self.compute_loss(outputs, targets, label_mask)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: GcnPortfolioModel,
    inputs: Tensor,
    graph: Tensor,
    targets: Tensor,
    label_mask: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    model.train()
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, graph, targets, label_mask)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: GcnPortfolioModel,
    inputs: Tensor,
    graph: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(inputs, graph).argmax(dim=1)
        accuracy = (
            (prediction[test_mask_device] == targets[test_mask_device]).float().mean()
        )
    return {"test_accuracy": float(accuracy.cpu())}


features_device, labels_device = ACCELERATOR.move(features, labels)
train_mask_device, test_mask_device = ACCELERATOR.move(train_mask, test_mask)
portfolio_config = PortfolioConfig()
portfolio_graph = ACCELERATOR.move(preprocess_graph(adjacency))
portfolio_model = ACCELERATOR.move(
    GcnPortfolioModel(
        features.shape[1],
        int(labels.max()) + 1,
        portfolio_config,
    )
)
portfolio_history = fit_portfolio_model(
    portfolio_model,
    features_device,
    portfolio_graph,
    labels_device,
    train_mask_device,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    features_device,
    portfolio_graph,
    labels_device,
)
assert np.isfinite(portfolio_history).all()
assert 0.0 <= portfolio_metrics["test_accuracy"] <= 1.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 03 ? GraphSAGE: 보지 못한 노드도 임베딩하기
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=3,
        slug="graphsage",
        short_title="GraphSAGE: 보지 못한 노드도 임베딩하기",
        paper_title="Inductive Representation Learning on Large Graphs",
        authors="William L. Hamilton, Rex Ying, Jure Leskovec",
        year=2017,
        primary_url="https://arxiv.org/abs/1706.02216",
        venue="NeurIPS",
        difficulty="중급",
        expected_minutes=65,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="GCN, permutation-invariant aggregation",
        reproduction_goal="고정 크기 이웃 표본과 mean aggregator를 구현하고 feature 기반 inductive 출력을 확인합니다.",
        original_scale="Reddit/PPI의 대규모 minibatch sampling 대신 전체 18노드와 작은 fanout을 사용합니다.",
        mappings=(
            (
                "§3.1, Neighborhood definition",
                "고정 fanout 이웃 표본",
                "개수·seed·edge 유효성",
            ),
            (
                "§3.3, Mean aggregator",
                "mean neighborhood aggregation",
                "순열 불변성",
            ),
            (
                "§3.1, Algorithm 1 lines 4–7",
                "self와 neighbor concat·정규화",
                "row norm",
            ),
            (
                "§2 및 §3.1",
                "unseen-node inductive inference",
                "feature 변경에 따른 출력",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_v^k=\sigma\!\left(W^k[h_v^{k-1}\Vert \operatorname{MEAN}_{u\in\mathcal N(v)}h_u^{k-1}]\right)$$

- **기호 정의:** v는 중심 노드, N(v)는 표본 이웃, ||는 concat, W는 공유 가중치입니다.
- **수식의 역할:** 특징 기반 mean aggregation으로 학습 때 보지 못한 노드도 계산합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.1, Neighborhood definition을 구조화합니다.
- **입출력 shape:** 특징 [N, F]와 fanout 이웃 → 정규화 표현 [N, H] → logit [N, C]
- **평가:** 분류 정확도와 입력 특징 변화에 대한 inductive 출력 변화
- **Inductive split:** optimizer는 train-induced subgraph만 사용하고,
  held-out 노드의 feature·neighbor·label은 학습 종료 뒤 추론에만 넣습니다.
- **원문 대비 한계:** Reddit/PPI의 대규모 minibatch sampling 대신 전체 18노드와 작은 fanout을 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_v^k=\sigma\!\left(W^k[h_v^{k-1}\Vert \operatorname{MEAN}_{u\in\mathcal N(v)}h_u^{k-1}]\right)$$

- **기호 정의:** v는 중심 노드, N(v)는 표본 이웃, ||는 concat, W는 공유 가중치입니다.
- **수식의 역할:** 특징 기반 mean aggregation으로 학습 때 보지 못한 노드도 계산합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.1, Neighborhood definition을 구조화합니다.
- **입출력 shape:** 특징 [N, F]와 fanout 이웃 → 정규화 표현 [N, H] → logit [N, C]
- **평가:** 분류 정확도와 입력 특징 변화에 대한 inductive 출력 변화
- **Inductive split:** optimizer는 train-induced subgraph만 사용하고,
  held-out 노드의 feature·neighbor·label은 학습 종료 뒤 추론에만 넣습니다.
- **원문 대비 한계:** Reddit/PPI의 대규모 minibatch sampling 대신 전체 18노드와 작은 fanout을 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 03-1: 각 노드에서 최대 fanout개의 이웃을 seed 기반 표본화하세요.
def sample_neighbors(adj: Tensor, fanout: int, seed: int) -> list[Tensor]:
    raise NotImplementedError("TODO 03-1")
""",
                r"""
def sample_neighbors(adj: Tensor, fanout: int, seed: int) -> list[Tensor]:
    rng = np.random.default_rng(seed)
    result = []
    for u in range(len(adj)):
        nbr = torch.where(adj[u] > 0)[0].numpy()
        chosen = (
            rng.choice(nbr, min(fanout, len(nbr)), replace=False)
            if len(nbr)
            else np.array([], dtype=int)
        )
        result.append(torch.tensor(chosen, dtype=torch.long))
    return result


sampled = sample_neighbors(adjacency, 3, 13)
assert all(len(v) <= 3 for v in sampled)
assert all(adjacency[u, v].all() for u, v in enumerate(sampled) if len(v))
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 03-2: §3.3 mean과 Algorithm 1의 concat·normalize layer를 구현하세요.
class MeanSAGE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        raise NotImplementedError("TODO 03-2")

    def forward(self, x: Tensor, neighbors: list[Tensor]) -> Tensor:
        raise NotImplementedError("TODO 03-2")
""",
                r"""
class MeanSAGE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x: Tensor, neighbors: list[Tensor]) -> Tensor:
        agg = torch.stack(
            [
                x[idx].mean(0) if len(idx) else torch.zeros_like(x[0])
                for idx in neighbors
            ]
        )
        return F.normalize(F.relu(self.linear(torch.cat([x, agg], dim=1))), dim=1)


sage = MeanSAGE(features.shape[1], 10)
h = sage(features, sampled)
assert h.shape == (len(features), 10) and torch.allclose(
    h.norm(dim=1)[h.norm(dim=1) > 0],
    torch.ones_like(h.norm(dim=1)[h.norm(dim=1) > 0]),
    atol=1e-5,
)
""",
                "task",
            ),
            # Cell 05 ? metric
            _cell(
                "code",
                r"""
# TODO 03-3: frozen SAGE 표현에 선형 분류기를 학습하고 unseen feature 민감도를 재세요.
raise NotImplementedError("TODO 03-3")
""",
                r"""
train_index = torch.where(train_mask)[0]
test_index = torch.where(test_mask)[0]
assert not torch.isin(train_index, test_index).any()
train_adjacency = adjacency[train_index][:, train_index]
train_features = features[train_index].clone()
train_labels = labels[train_index].clone()
train_sampled = sample_neighbors(train_adjacency, 3, 13)
classifier = nn.Linear(h.shape[1], int(train_labels.max()) + 1)
opt = torch.optim.Adam([*sage.parameters(), *classifier.parameters()], lr=0.04)
losses = []
for _ in range(140):
    h = sage(train_features, train_sampled)
    logits = classifier(h)
    loss = F.cross_entropy(logits, train_labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss))
with torch.no_grad():
    # New nodes enter only here, with their own features and sampled neighbors.
    base = sage(features, sampled)
    unseen_logits = classifier(base)[test_index]
    changed_features = features.clone()
    changed_features[test_index] += 0.25
    changed = sage(changed_features, sampled)
    sensitivity = float(
        (base[test_index] - changed[test_index]).norm(dim=1).mean()
    )
    test_acc = float(
        (unseen_logits.argmax(1) == labels[test_index]).float().mean()
    )
assert losses[-1] < losses[0] and sensitivity > 0
assert len(train_features) == len(train_labels) == len(train_sampled)
assert train_adjacency.shape == (len(train_index), len(train_index))
plt.plot(losses)
plt.title(f"GraphSAGE · unseen sensitivity={sensitivity:.3f}")
plt.show()
print({"test_accuracy": test_acc, "feature_sensitivity": sensitivity})
""",
                "metric",
            ),
            # Cell 06 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 07 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    fanout: int = 3
    sample_seed: int = 13
    learning_rate: float = 0.03
    steps: int = 18


def preprocess_graph(
    graph: Tensor,
    config: PortfolioConfig,
) -> list[Tensor]:
    raise NotImplementedError("TODO P-1: fixed-fanout sampling")


class GraphsagePortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        raise NotImplementedError("TODO P-1: MeanSAGE encoder and classifier")

    def encode(self, inputs: Tensor, neighbors: list[Tensor]) -> Tensor:
        raise NotImplementedError("TODO P-1: sampled mean aggregation")

    def forward(self, inputs: Tensor, neighbors: list[Tensor]) -> Tensor:
        raise NotImplementedError("TODO P-1: paper equation")

    def compute_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(
        self,
        inputs: Tensor,
        neighbors: list[Tensor],
        targets: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    fanout: int = 3
    sample_seed: int = 13
    learning_rate: float = 0.03
    steps: int = 18


def preprocess_graph(
    graph: Tensor,
    config: PortfolioConfig,
) -> list[Tensor]:
    return sample_neighbors(graph, config.fanout, config.sample_seed)


class GraphsagePortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        self.config = config
        self.encoder = MeanSAGE(input_dim, config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, class_count)

    def encode(self, inputs: Tensor, neighbors: list[Tensor]) -> Tensor:
        return self.encoder(inputs, neighbors)

    def forward(self, inputs: Tensor, neighbors: list[Tensor]) -> Tensor:
        return self.classifier(self.encode(inputs, neighbors))

    def compute_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return F.cross_entropy(outputs, targets)

    def training_step(
        self,
        inputs: Tensor,
        neighbors: list[Tensor],
        targets: Tensor,
    ) -> Tensor:
        outputs = self(inputs, neighbors)
        return self.compute_loss(outputs, targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: GraphsagePortfolioModel,
    inputs: Tensor,
    neighbors: list[Tensor],
    targets: Tensor,
) -> list[float]:
    assert len(inputs) == len(neighbors) == len(targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    model.train()
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, neighbors, targets)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: GraphsagePortfolioModel,
    inputs: Tensor,
    neighbors: list[Tensor],
    unseen_targets: Tensor,
    unseen_index: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        base_logits = model(inputs, neighbors)
        unseen_logits = base_logits[unseen_index]
        prediction = unseen_logits.argmax(dim=1)
        accuracy = (prediction == unseen_targets).float().mean()
        changed_inputs = inputs.clone()
        changed_inputs[unseen_index] += 0.25
        changed_logits = model(changed_inputs, neighbors)
        sensitivity = (
            (unseen_logits - changed_logits[unseen_index])
            .norm(dim=1)
            .mean()
        )
    return {
        "unseen_test_accuracy": float(accuracy.cpu()),
        "unseen_feature_sensitivity": float(sensitivity.cpu()),
    }


portfolio_train_index = torch.where(train_mask)[0]
portfolio_test_index = torch.where(test_mask)[0]
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_train_graph = adjacency[portfolio_train_index][
    :, portfolio_train_index
]
portfolio_train_features = features[portfolio_train_index].clone()
portfolio_train_labels = labels[portfolio_train_index].clone()
portfolio_inference_features = features.clone()
portfolio_config = PortfolioConfig()
portfolio_train_neighbors = preprocess_graph(
    portfolio_train_graph,
    portfolio_config,
)
portfolio_inference_neighbors = preprocess_graph(
    adjacency,
    portfolio_config,
)
(
    portfolio_train_features,
    portfolio_train_labels,
    portfolio_train_neighbors,
    portfolio_inference_features,
    portfolio_inference_neighbors,
    portfolio_unseen_labels,
    portfolio_test_index_device,
) = ACCELERATOR.move(
    portfolio_train_features,
    portfolio_train_labels,
    portfolio_train_neighbors,
    portfolio_inference_features,
    portfolio_inference_neighbors,
    labels[portfolio_test_index].clone(),
    portfolio_test_index,
)
portfolio_model = ACCELERATOR.move(
    GraphsagePortfolioModel(
        portfolio_train_features.shape[1],
        int(portfolio_train_labels.max()) + 1,
        portfolio_config,
    )
)
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_train_features,
    portfolio_train_neighbors,
    portfolio_train_labels,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_inference_features,
    portfolio_inference_neighbors,
    portfolio_unseen_labels,
    portfolio_test_index_device,
)
assert np.isfinite(portfolio_history).all()
assert 0.0 <= portfolio_metrics["unseen_test_accuracy"] <= 1.0
assert portfolio_metrics["unseen_feature_sensitivity"] > 0.0
assert len(portfolio_train_features) == len(portfolio_train_index)
assert len(portfolio_train_neighbors) == len(portfolio_train_index)
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 04 ? GAT: 이웃마다 다른 주의 가중치
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=4,
        slug="gat",
        short_title="GAT: 이웃마다 다른 주의 가중치",
        paper_title="Graph Attention Networks",
        authors=(
            "Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero"
            ", Pietro Liò, Yoshua Bengio"
        ),
        year=2018,
        primary_url="https://arxiv.org/abs/1710.10903",
        venue="ICLR",
        difficulty="중급",
        expected_minutes=70,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="attention, masked softmax, GCN",
        reproduction_goal="masked self-attention 계수와 single-head GAT layer를 구현해 이웃별 가중치를 관찰합니다.",
        original_scale="Cora/PPI multi-head 모형 대신 18노드 single-head 모형을 사용합니다.",
        mappings=(
            (
                "§2, Eq. (1)",
                "공유 선형변환 Wh",
                "출력 차원",
            ),
            (
                "§2, Eq. (2)",
                "LeakyReLU attention score",
                "edge별 e_ij",
            ),
            (
                "§2, Eq. (3)",
                "neighborhood masked softmax",
                "비edge 0·행합 1",
            ),
            (
                "§2, Eq. (4)",
                "attention weighted aggregation",
                "분류 loss·attention heatmap",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\alpha_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}\!\left(
\operatorname{LeakyReLU}(a^\top[Wh_i\Vert Wh_j])\right)$$

- **기호 정의:** i는 수신 노드, j는 이웃, W는 투영, a는 attention 벡터입니다.
- **수식의 역할:** masked softmax로 이웃마다 다른 데이터 의존 메시지 가중치를 만듭니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §2, Eq. (1)을 구조화합니다.
- **입출력 shape:** 특징 [N, F]와 edge mask [N, N] → 표현 [N, H], attention [N, N]
- **평가:** attention 행합·비edge 0 여부, 분류 loss와 test accuracy
- **원문 대비 한계:** Cora/PPI multi-head 모형 대신 18노드 single-head 모형을 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\alpha_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}\!\left(
\operatorname{LeakyReLU}(a^\top[Wh_i\Vert Wh_j])\right)$$

- **기호 정의:** i는 수신 노드, j는 이웃, W는 투영, a는 attention 벡터입니다.
- **수식의 역할:** masked softmax로 이웃마다 다른 데이터 의존 메시지 가중치를 만듭니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §2, Eq. (1)을 구조화합니다.
- **입출력 shape:** 특징 [N, F]와 edge mask [N, N] → 표현 [N, H], attention [N, N]
- **평가:** attention 행합·비edge 0 여부, 분류 loss와 test accuracy
- **원문 대비 한계:** Cora/PPI multi-head 모형 대신 18노드 single-head 모형을 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 04-1: Eq. (2)–(3)의 masked graph attention을 구현하세요.
def graph_attention(wh: Tensor, adj: Tensor, attn_vector: Tensor) -> Tensor:
    raise NotImplementedError("TODO 04-1")
""",
                r"""
def graph_attention(wh: Tensor, adj: Tensor, attn_vector: Tensor) -> Tensor:
    n = len(wh)
    left = wh[:, None, :].expand(n, n, -1)
    right = wh[None, :, :].expand(n, n, -1)
    score = F.leaky_relu(torch.cat([left, right], -1) @ attn_vector, negative_slope=0.2)
    mask = (adj + torch.eye(n, device=adj.device, dtype=adj.dtype)) > 0
    return torch.softmax(score.masked_fill(~mask, -1e9), dim=1)


probe_wh = features[:, :4]
probe_a = graph_attention(probe_wh, adjacency, torch.linspace(-0.2, 0.3, 8))
edge_mask = (adjacency + torch.eye(len(adjacency))) > 0
assert torch.allclose(probe_a.sum(1), torch.ones(len(adjacency)), atol=1e-5)
assert probe_a[~edge_mask].max() < 1e-7
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 04-2: Eq. (1)–(4)를 묶은 single-head GAT layer를 정의하세요.
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        raise NotImplementedError("TODO 04-2")

    def forward(self, x, adj, return_attention=False):
        raise NotImplementedError("TODO 04-2")
""",
                r"""
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.randn(2 * out_dim) * 0.1)

    def forward(self, x, adj, return_attention=False):
        wh = self.linear(x)
        alpha = graph_attention(wh, adj, self.attn)
        out = alpha @ wh
        return (out, alpha) if return_attention else out


gat = GATLayer(features.shape[1], 12)
out, alpha = gat(features, adjacency, True)
assert out.shape == (len(features), 12)
""",
                "task",
            ),
            # Cell 05 ? metric
            _cell(
                "code",
                r"""
# TODO 04-3: GAT 표현과 classifier를 semi-supervised 학습하고 attention을 그리세요.
raise NotImplementedError("TODO 04-3")
""",
                r"""
gat = ACCELERATOR.move(gat)
classifier = ACCELERATOR.move(nn.Linear(12, int(labels.max()) + 1))
features_device, adjacency_device, labels_device = ACCELERATOR.move(
    features, adjacency, labels
)
train_mask_device, test_mask_device = ACCELERATOR.move(train_mask, test_mask)
optimizer = torch.optim.Adam([*gat.parameters(), *classifier.parameters()], lr=0.035)
losses = []
for _ in range(160):
    logits = classifier(F.elu(gat(features_device, adjacency_device)))
    loss = F.cross_entropy(logits[train_mask_device], labels_device[train_mask_device])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    h, alpha = gat(features_device, adjacency_device, True)
    pred = classifier(F.elu(h)).argmax(1)
    accuracy = float(
        (pred[test_mask_device] == labels_device[test_mask_device]).float().mean()
    )
assert losses[-1] < losses[0] and torch.allclose(
    alpha.sum(1), torch.ones(len(alpha), device=alpha.device), atol=1e-5
)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("loss")
ax[1].imshow(alpha.detach().cpu(), cmap="magma")
ax[1].set_title("learned α_ij")
plt.tight_layout()
plt.show()
print("test accuracy:", round(accuracy, 3))
""",
                "metric",
            ),
            # Cell 06 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 07 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 10
    basis_count: int = 2
    learning_rate: float = 0.03
    steps: int = 14


def preprocess_graph(graph: Tensor) -> Tensor:
    raise NotImplementedError("TODO P-1: graph preprocessing")


class GatPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, inputs: Tensor, graph: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: paper equation")

    def compute_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, inputs: Tensor, graph: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    learning_rate: float = 0.03
    steps: int = 18


def preprocess_graph(graph: Tensor) -> Tensor:
    with_self = graph + torch.eye(len(graph), dtype=graph.dtype)
    degree = with_self.sum(dim=1).clamp_min(1.0)
    inverse_root = degree.pow(-0.5)
    return inverse_root[:, None] * with_self * inverse_root[None, :]


class GatPortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(features.shape[1], config.hidden_dim)
        self.attention = nn.Parameter(torch.zeros(config.hidden_dim * 2))

        self.hidden = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, int(labels.max()) + 1)
        self.last_attention = None

    def forward(self, inputs: Tensor, graph: Tensor) -> Tensor:
        hidden = self.input_projection(inputs)

        left = hidden[:, None, :].expand(-1, len(hidden), -1)
        right = hidden[None, :, :].expand(len(hidden), -1, -1)
        pair = torch.cat([left, right], dim=-1)
        scores = F.leaky_relu(pair @ self.attention, negative_slope=0.2)
        mask = graph > 0
        scores = scores.masked_fill(~mask, -1e9)
        self.last_attention = torch.softmax(scores, dim=1)
        combined = self.last_attention @ hidden

        representation = F.relu(self.hidden(combined))
        return self.classifier(representation)

    def compute_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(outputs[train_mask_device], targets[train_mask_device])

    def training_step(self, inputs: Tensor, graph: Tensor, targets: Tensor) -> Tensor:
        outputs = self(inputs, graph)
        return self.compute_loss(outputs, targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: GatPortfolioModel,
    inputs: Tensor,
    graph: Tensor,
    targets: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, graph, targets)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: GatPortfolioModel,
    inputs: Tensor,
    graph: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(inputs, graph).argmax(dim=1)
        accuracy = (
            (prediction[test_mask_device] == targets[test_mask_device]).float().mean()
        )
    return {"test_accuracy": float(accuracy.cpu())}


features_device, labels_device = ACCELERATOR.move(features, labels)
train_mask_device, test_mask_device = ACCELERATOR.move(train_mask, test_mask)
portfolio_config = PortfolioConfig()
portfolio_graph = ACCELERATOR.move(preprocess_graph(adjacency))
portfolio_model = ACCELERATOR.move(GatPortfolioModel(portfolio_config))
portfolio_history = fit_portfolio_model(
    portfolio_model,
    features_device,
    portfolio_graph,
    labels_device,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    features_device,
    portfolio_graph,
    labels_device,
)
assert np.isfinite(portfolio_history).all()
assert 0.0 <= portfolio_metrics["test_accuracy"] <= 1.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 05 ? VGAE: 확률적 그래프 잠재공간
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=5,
        slug="vgae",
        short_title="VGAE: 확률적 그래프 잠재공간",
        paper_title="Variational Graph Auto-Encoders",
        authors="Thomas N. Kipf, Max Welling",
        year=2016,
        primary_url="https://arxiv.org/abs/1611.07308",
        venue="NeurIPS Workshop on Bayesian Deep Learning",
        difficulty="중급",
        expected_minutes=70,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="VAE, GCN, binary cross entropy",
        reproduction_goal=(
            "GCN posterior encoder, reparameterization, inner-product decoder로 링크"
            "를 복원합니다."
        ),
        original_scale="Cora/Citeseer/PubMed link prediction 대신 18노드 전체 adjacency 복원을 수행합니다.",
        mappings=(
            (
                "Eq. (1)",
                "factorized Gaussian q(Z|X,A)",
                "mu/logstd shape",
            ),
            (
                "Eq. (2)",
                "inner-product Bernoulli decoder",
                "대칭 logit",
            ),
            (
                "Eq. (3)",
                "ELBO=reconstruction−KL",
                "loss 감소",
            ),
            (
                "§2, GCN encoder 설명",
                "공유 첫 층·분기 head",
                "positive/negative link score",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\mathcal L=\mathbb E_{q(Z\mid X,A)}[\log p(A\mid Z)]-D_{KL}(q(Z\mid X,A)\Vert p(Z))$$

- **기호 정의:** Z는 잠재 노드 벡터, q는 GCN posterior, p(A|Z)는 inner-product decoder입니다.
- **수식의 역할:** 재구성 우도와 Gaussian KL을 함께 최적화해 확률적 그래프 표현을 얻습니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 Eq. (1)을 구조화합니다.
- **입출력 shape:** 특징 [N, F], 인접행렬 [N, N] → μ/logσ [N, D] → edge logit [N, N]
- **평가:** ELBO 감소와 positive/negative edge score 분리
- **원문 대비 한계:** Cora/Citeseer/PubMed link prediction 대신 18노드 전체 adjacency 복원을 수행합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$\mathcal L=\mathbb E_{q(Z\mid X,A)}[\log p(A\mid Z)]-D_{KL}(q(Z\mid X,A)\Vert p(Z))$$

- **기호 정의:** Z는 잠재 노드 벡터, q는 GCN posterior, p(A|Z)는 inner-product decoder입니다.
- **수식의 역할:** 재구성 우도와 Gaussian KL을 함께 최적화해 확률적 그래프 표현을 얻습니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 Eq. (1)을 구조화합니다.
- **입출력 shape:** 특징 [N, F], 인접행렬 [N, N] → μ/logσ [N, D] → edge logit [N, N]
- **평가:** ELBO 감소와 positive/negative edge score 분리
- **원문 대비 한계:** Cora/Citeseer/PubMed link prediction 대신 18노드 전체 adjacency 복원을 수행합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 05-1: self-loop를 포함한 대칭 정규화 adjacency를 만드세요.
def norm_adj(adj):
    raise NotImplementedError("TODO 05-1")
""",
                r"""
def norm_adj(adj):
    a = adj + torch.eye(len(adj), device=adj.device, dtype=adj.dtype)
    d = a.sum(1).pow(-0.5)
    return d[:, None] * a * d[None, :]


a_hat = norm_adj(adjacency)
assert torch.allclose(a_hat, a_hat.T)
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 05-2: Eq. (1)의 GCN Gaussian encoder와 reparameterization을 구현하세요.
class VGAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden=12, latent=6):
        super().__init__()
        raise NotImplementedError("TODO 05-2")

    def forward(self, x, a):
        raise NotImplementedError("TODO 05-2")
""",
                r"""
class VGAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden=12, latent=6):
        super().__init__()
        self.shared = nn.Linear(in_dim, hidden, bias=False)
        self.mu = nn.Linear(hidden, latent, bias=False)
        self.logstd = nn.Linear(hidden, latent, bias=False)

    def forward(self, x, a):
        h = F.relu(a @ self.shared(x))
        mu = a @ self.mu(h)
        logstd = (a @ self.logstd(h)).clamp(-4, 4)
        z = mu + torch.randn_like(mu) * logstd.exp()
        return z, mu, logstd


encoder = VGAEEncoder(features.shape[1])
z, mu, logstd = encoder(features, a_hat)
assert z.shape == mu.shape == logstd.shape == (len(features), 6)
""",
                "task",
            ),
            # Cell 05 ? metric
            _cell(
                "code",
                r"""
# TODO 05-3: Eq. (2) decoder와 Eq. (3)의 negative ELBO를 구현·학습하세요.
raise NotImplementedError("TODO 05-3")
""",
                r"""
encoder = ACCELERATOR.move(encoder)
features_device, a_hat_device, target = ACCELERATOR.move(
    features, a_hat, adjacency.clone()
)
target.fill_diagonal_(0)
pos_weight = (target.numel() - target.sum()) / target.sum().clamp_min(1)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.035)
losses = []
for _ in range(180):
    z, mu, logstd = encoder(features_device, a_hat_device)
    logits = z @ z.T
    recon = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    kl = -0.5 * torch.mean(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp())
    loss = recon + 0.02 * kl
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    _, mu, _ = encoder(features_device, a_hat_device)
    probability = torch.sigmoid(mu @ mu.T)
    eye = torch.eye(len(target), dtype=torch.bool, device=target.device)
    pos = target.bool()
    neg = ~pos & ~eye
    pos_score = float(probability[pos].mean())
    neg_score = float(probability[neg].mean())
assert losses[-1] < losses[0] and pos_score > neg_score
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("negative ELBO")
ax[1].imshow(probability.detach().cpu(), cmap="viridis", vmin=0, vmax=1)
ax[1].set_title("p(A|Z)")
plt.tight_layout()
plt.show()
print({"edge_score": round(pos_score, 3), "nonedge_score": round(neg_score, 3)})
""",
                "metric",
            ),
            # Cell 06 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 07 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    learning_rate: float = 0.02
    steps: int = 12


def preprocess_graph(graph: Tensor) -> Tensor:
    raise NotImplementedError("TODO P-1: graph preprocessing")


class VgaePortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: paper-specific layers")

    def forward(self, inputs: Tensor, graph: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: paper equation")

    def compute_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(self, inputs: Tensor, graph: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    latent_dim: int = 6
    learning_rate: float = 0.03
    steps: int = 14


def preprocess_graph(graph: Tensor) -> Tensor:
    with_self = graph + torch.eye(len(graph), dtype=graph.dtype)
    degree = with_self.sum(1).clamp_min(1.0).pow(-0.5)
    return degree[:, None] * with_self * degree[None, :]


class VgaePortfolioModel(nn.Module):
    def __init__(self, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.shared = nn.Linear(features.shape[1], 12)
        self.mean = nn.Linear(12, config.latent_dim)
        self.log_variance = nn.Linear(12, config.latent_dim)

    def forward(self, inputs: Tensor, graph: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = F.relu(graph @ self.shared(inputs))
        mean = graph @ self.mean(hidden)
        log_variance = (graph @ self.log_variance(hidden)).clamp(-5.0, 5.0)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        return latent @ latent.T, mean, log_variance

    def compute_loss(self, outputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        edge_logits, mean, log_variance = outputs
        reconstruction = F.binary_cross_entropy_with_logits(
            edge_logits, adjacency_device
        )
        kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()
        return reconstruction + 0.05 * kl

    def training_step(self, inputs: Tensor, graph: Tensor) -> Tensor:
        return self.compute_loss(self(inputs, graph))
""",
                "portfolio-model",
                "task",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: VgaePortfolioModel,
    inputs: Tensor,
    graph: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, graph)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: VgaePortfolioModel,
    inputs: Tensor,
    graph: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        logits, _, _ = model(inputs, graph)
        positive = logits[adjacency_device.bool()].mean()
        negative = logits[~adjacency_device.bool()].mean()
    return {"edge_score_gap": float((positive - negative).cpu())}


features_device, adjacency_device = ACCELERATOR.move(features, adjacency)
portfolio_config = PortfolioConfig()
portfolio_graph = ACCELERATOR.move(preprocess_graph(adjacency))
portfolio_model = ACCELERATOR.move(VgaePortfolioModel(portfolio_config))
portfolio_history = fit_portfolio_model(
    portfolio_model,
    features_device,
    portfolio_graph,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    features_device,
    portfolio_graph,
)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 06 ? R-GCN: 관계 종류별 메시지
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=6,
        slug="rgcn",
        short_title="R-GCN: 관계 종류별 메시지",
        paper_title="Modeling Relational Data with Graph Convolutional Networks",
        authors=(
            "Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den B"
            "erg, Ivan Titov, Max Welling"
        ),
        year=2018,
        primary_url="https://arxiv.org/abs/1703.06103",
        venue="ESWC",
        difficulty="중급",
        expected_minutes=60,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="GCN, knowledge graph, relation types",
        reproduction_goal="관계별 weight와 degree normalization을 갖는 R-GCN propagation을 구현합니다.",
        original_scale="AIFB/MUTAG/BGS/AM와 FB15k 대신 12개 entity·3 relation의 로컬 지식그래프를 사용합니다.",
        mappings=(
            (
                "§2, Eq. (2)",
                "relation-specific neighborhood sum",
                "관계별 weight·shape",
            ),
            (
                "Eq. (2)의 c_i,r",
                "관계별 degree normalization",
                "행합/isolated 처리",
            ),
            (
                "Eq. (3)",
                "basis decomposition",
                "parameter reconstruction",
            ),
            (
                "Figure 2",
                "edge type 변화의 메시지 영향",
                "embedding displacement",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_i^{l+1}=\sigma\!\left(\sum_{r\in\mathcal R}\sum_{j\in\mathcal N_i^r}\frac{1}{c_{i
,r}}W_r^l h_j^l+W_0^lh_i^l\right)$$

- **기호 정의:** r은 관계, N_i^r은 관계별 이웃, c는 정규화 상수, W_r은 관계별 weight입니다.
- **수식의 역할:** 관계 종류마다 다른 메시지 변환을 적용해 다중 관계 그래프를 표현합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §2, Eq. (2)을 구조화합니다.
- **입출력 shape:** 관계 인접행렬 [R, N, N], 특징 [N, F] → 노드 표현 [N, H]
- **평가:** 관계별 edge reconstruction BCE와 관계 변경에 따른 표현 변화
- **원문 대비 한계:** AIFB/MUTAG/BGS/AM와 FB15k 대신 12개 entity·3 relation의 로컬 지식그래프를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_i^{l+1}=\sigma\!\left(\sum_{r\in\mathcal R}\sum_{j\in\mathcal N_i^r}\frac{1}{c_{i
,r}}W_r^l h_j^l+W_0^lh_i^l\right)$$

- **기호 정의:** r은 관계, N_i^r은 관계별 이웃, c는 정규화 상수, W_r은 관계별 weight입니다.
- **수식의 역할:** 관계 종류마다 다른 메시지 변환을 적용해 다중 관계 그래프를 표현합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §2, Eq. (2)을 구조화합니다.
- **입출력 shape:** 관계 인접행렬 [R, N, N], 특징 [N, F] → 노드 표현 [N, H]
- **평가:** 관계별 edge reconstruction BCE와 관계 변경에 따른 표현 변화
- **원문 대비 한계:** AIFB/MUTAG/BGS/AM와 FB15k 대신 12개 entity·3 relation의 로컬 지식그래프를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "code",
                r"""
relation_adjacency = torch.tensor(raw["relation_adjacency"], dtype=torch.float32)
relation_features = torch.tensor(raw["relation_features"], dtype=torch.float32)
assert relation_adjacency.ndim == 3 and relation_adjacency.shape[
    1
] == relation_adjacency.shape[2] == len(relation_features)
""",
                r"""
relation_adjacency = torch.tensor(raw["relation_adjacency"], dtype=torch.float32)
relation_features = torch.tensor(raw["relation_features"], dtype=torch.float32)
assert relation_adjacency.ndim == 3 and relation_adjacency.shape[
    1
] == relation_adjacency.shape[2] == len(relation_features)
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 06-1: 각 relation adjacency를 발신 이웃 수로 정규화하세요.
def relation_normalize(rel_adj: Tensor) -> Tensor:
    raise NotImplementedError("TODO 06-1")
""",
                r"""
def relation_normalize(rel_adj: Tensor) -> Tensor:
    degree = rel_adj.sum(2, keepdim=True).clamp_min(1)
    return rel_adj / degree


rel_norm = relation_normalize(relation_adjacency)
row_sums = rel_norm.sum(2)
assert ((row_sums == 0) | (torch.isclose(row_sums, torch.ones_like(row_sums)))).all()
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 06-2: Eq. (3)의 basis decomposition으로 relation weight를 만드세요.
def basis_weights(bases: Tensor, coefficients: Tensor) -> Tensor:
    raise NotImplementedError("TODO 06-2")
""",
                r"""
def basis_weights(bases: Tensor, coefficients: Tensor) -> Tensor:
    return torch.einsum("rb,bio->rio", coefficients, bases)


bases = torch.randn(2, relation_features.shape[1], 8) * 0.2
coeff = torch.randn(len(rel_norm), 2)
rel_weights = basis_weights(bases, coeff)
assert rel_weights.shape == (len(rel_norm), relation_features.shape[1], 8)
""",
                "task",
            ),
            # Cell 06 ? task
            _cell(
                "code",
                r"""
# TODO 06-3: Eq. (2)의 relation message와 self-loop를 합산하세요.
def rgcn_layer(x: Tensor, rel_a: Tensor, rel_w: Tensor, self_w: Tensor) -> Tensor:
    raise NotImplementedError("TODO 06-3")
""",
                r"""
def rgcn_layer(x: Tensor, rel_a: Tensor, rel_w: Tensor, self_w: Tensor) -> Tensor:
    messages = torch.stack([rel_a[r] @ x @ rel_w[r] for r in range(len(rel_a))]).sum(0)
    return F.relu(messages + x @ self_w)


self_w = torch.randn(relation_features.shape[1], 8) * 0.2
h = rgcn_layer(relation_features, rel_norm, rel_weights, self_w)
assert h.shape == (len(relation_features), 8)
""",
                "task",
            ),
            # Cell 07 ? metric
            _cell(
                "code",
                r"""
# TODO 06-4: 같은 edge를 다른 relation으로 옮겨 표현 변화를 시각화하세요.
raise NotImplementedError("TODO 06-4")
""",
                r"""
changed = relation_adjacency.clone()
edge = torch.nonzero(changed[0] > 0)[0]
u, v = edge.tolist()
changed[0, u, v] = 0
changed[1, u, v] = 1
h_changed = rgcn_layer(
    relation_features, relation_normalize(changed), rel_weights, self_w
)
delta = (h_changed - h).norm(dim=1).detach()
assert float(delta.max()) > 0
plt.bar(np.arange(len(delta)), delta.numpy())
plt.xlabel("entity")
plt.ylabel("embedding change")
plt.title("edge-type intervention")
plt.show()
print("moved edge / max change:", (u, v), float(delta.max()))
""",
                "metric",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    learning_rate: float = 0.02
    steps: int = 12


def preprocess_graph(graph: Tensor) -> Tensor:
    raise NotImplementedError("TODO P-1: graph preprocessing")


class RgcnPortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        relation_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        raise NotImplementedError("TODO P-1: bases, coefficients, and decoder")

    def forward(self, inputs: Tensor, relations: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("TODO P-1: basis R-GCN propagation")

    def compute_loss(self, edge_logits: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(
        self,
        inputs: Tensor,
        relations: Tensor,
        targets: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass

relation_adjacency = torch.tensor(raw["relation_adjacency"], dtype=torch.float32)
relation_features = torch.tensor(raw["relation_features"], dtype=torch.float32)


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 10
    basis_count: int = 2
    learning_rate: float = 0.03
    steps: int = 14


def preprocess_graph(relations: Tensor) -> Tensor:
    return relation_normalize(relations)


class RgcnPortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        relation_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        self.config = config
        self.bases = nn.Parameter(
            torch.randn(config.basis_count, input_dim, config.hidden_dim) * 0.1
        )
        self.coefficients = nn.Parameter(
            torch.randn(relation_count, config.basis_count) * 0.1
        )
        self.self_weight = nn.Parameter(
            torch.randn(input_dim, config.hidden_dim) * 0.1
        )
        self.decoders = nn.Parameter(
            torch.randn(relation_count, config.hidden_dim, config.hidden_dim) * 0.1
        )

    def forward(self, inputs: Tensor, relations: Tensor) -> tuple[Tensor, Tensor]:
        relation_weights = basis_weights(self.bases, self.coefficients)
        hidden = rgcn_layer(
            inputs,
            relations,
            relation_weights,
            self.self_weight,
        )
        edge_logits = torch.einsum("ih,rhk,jk->rij", hidden, self.decoders, hidden)
        return hidden, edge_logits

    def compute_loss(self, edge_logits: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(edge_logits, targets)

    def training_step(
        self, inputs: Tensor, relations: Tensor, targets: Tensor
    ) -> Tensor:
        _, edge_logits = self(inputs, relations)
        return self.compute_loss(edge_logits, targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 10 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: RgcnPortfolioModel,
    inputs: Tensor,
    relations: Tensor,
    targets: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    for _ in range(model.config.steps):
        loss = model.training_step(inputs, relations, targets)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 13 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: RgcnPortfolioModel,
    inputs: Tensor,
    relations: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        _, logits = model(inputs, relations)
        probabilities = torch.sigmoid(logits)
        target_mask = targets.bool()
        gap = probabilities[target_mask].mean() - probabilities[~target_mask].mean()
        relation_weights = basis_weights(model.bases, model.coefficients)
    return {
        "relation_edge_gap": float(gap.cpu()),
        "basis_weight_norm": float(relation_weights.norm().cpu()),
    }


relation_features_device = ACCELERATOR.move(relation_features)
relation_adjacency_device = ACCELERATOR.move(relation_adjacency)
portfolio_relations = ACCELERATOR.move(preprocess_graph(relation_adjacency))
portfolio_model = ACCELERATOR.move(
    RgcnPortfolioModel(
        relation_features.shape[1],
        relation_adjacency.shape[0],
        PortfolioConfig(),
    )
)
portfolio_history = fit_portfolio_model(
    portfolio_model,
    relation_features_device,
    portfolio_relations,
    relation_adjacency_device,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    relation_features_device,
    portfolio_relations,
    relation_adjacency_device,
)
assert np.isfinite(portfolio_history).all()
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 07 ? GIN: 합 집계와 WL 표현력
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=7,
        slug="gin",
        short_title="GIN: 합 집계와 WL 표현력",
        paper_title="How Powerful are Graph Neural Networks?",
        authors="Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka",
        year=2019,
        primary_url="https://openreview.net/forum?id=ryGs6iA5Km",
        venue="ICLR",
        difficulty="중급",
        expected_minutes=55,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="message passing, multiset, Weisfeiler–Lehman test",
        reproduction_goal=(
            "sum aggregator와 (1+epsilon) self term을 구현해 mean이 잃는 degree 정보를 보존함을 "
            "확인합니다."
        ),
        original_scale=(
            "9개 graph-classification benchmark 대신 한 로컬 그래프의 node multiset 표현력을 점검"
            "합니다."
        ),
        mappings=(
            (
                "§4.1, Lemma 5 및 Corollary 6",
                "injective multiset sum 직관",
                "degree 구별",
            ),
            (
                "§4.2, Eq. (4.1)",
                "GIN update",
                "epsilon self term",
            ),
            (
                "§4.3, Eq. (4.2)",
                "layer별 graph readout",
                "permutation invariance",
            ),
            (
                "Figure 2",
                "mean/max/sum 표현력 비교",
                "고유 representation 수",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_v^k=\operatorname{MLP}^k\!\left((1+\epsilon^k)h_v^{k-1}+\sum_{u\in
\mathcal N(v)}h_u^{k-1}\right)$$

- **기호 정의:** ε은 self 가중치, 합은 이웃 multiset, MLP는 injective map 근사입니다.
- **수식의 역할:** sum aggregation으로 mean이 지우는 이웃 중복도와 degree 정보를 보존합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** 두 GIN layer가 §4.2의 update를 수행하고,
  layer-wise sum readout과 graph prediction head가 §4.3을 구현합니다.
- **입출력 shape:** ego graph [N_g, F], [N_g, N_g] →
  layer states → concatenated graph readout → graph logit [1, C]
- **평가:** held-out ego-graph accuracy, readout 순열 불변성,
  sum/mean의 degree·multiset 분리
- **원문 대비 한계:** 한 로컬 graph의 ego-network를 graph sample로 사용하므로,
  원 논문의 9개 graph-classification benchmark와 직접 비교하지 않습니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_v^k=\operatorname{MLP}^k\!\left((1+\epsilon^k)h_v^{k-1}+\sum_{u\in
\mathcal N(v)}h_u^{k-1}\right)$$

- **기호 정의:** ε은 self 가중치, 합은 이웃 multiset, MLP는 injective map 근사입니다.
- **수식의 역할:** sum aggregation으로 mean이 지우는 이웃 중복도와 degree 정보를 보존합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** 두 GIN layer가 §4.2의 update를 수행하고,
  layer-wise sum readout과 graph prediction head가 §4.3을 구현합니다.
- **입출력 shape:** ego graph [N_g, F], [N_g, N_g] →
  layer states → concatenated graph readout → graph logit [1, C]
- **평가:** held-out ego-graph accuracy, readout 순열 불변성,
  sum/mean의 degree·multiset 분리
- **원문 대비 한계:** 한 로컬 graph의 ego-network를 graph sample로 사용하므로,
  원 논문의 9개 graph-classification benchmark와 직접 비교하지 않습니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? task
            _cell(
                "code",
                r"""
# TODO 07-1: Eq. (4.1)의 pre-MLP GIN aggregation을 구현하세요.
def gin_aggregate(x: Tensor, adj: Tensor, epsilon: float = 0.0) -> Tensor:
    raise NotImplementedError("TODO 07-1")
""",
                r"""
def gin_aggregate(x: Tensor, adj: Tensor, epsilon: float = 0.0) -> Tensor:
    return (1 + epsilon) * x + adj @ x


ones = torch.ones(len(adjacency), 1)
gin_counts = gin_aggregate(ones, adjacency)
assert torch.allclose(gin_counts[:, 0], adjacency.sum(1) + 1)
""",
                "task",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 07-2: mean과 sum이 상수 node feature에서 보존하는 정보를 비교하세요.
raise NotImplementedError("TODO 07-2")
""",
                r"""
degree = adjacency.sum(1, keepdim=True)
mean_neighbors = (adjacency @ ones) / degree.clamp_min(1)
active = degree[:, 0] > 0
mean_unique = len(torch.unique(mean_neighbors[active]))
sum_unique = len(torch.unique((adjacency @ ones)[active]))
assert mean_unique == 1 and sum_unique > mean_unique
print({"mean_unique": mean_unique, "sum_unique": sum_unique})
""",
                "task",
            ),
            # Cell 05 ? metric
            _cell(
                "code",
                r"""
# TODO 07-3: 3층 GIN과 Eq. (4.2)의 layer-wise sum readout을 구현하세요.
raise NotImplementedError("TODO 07-3")
""",
                r"""
mlps = nn.ModuleList(
    [
        nn.Sequential(
            nn.Linear(features.shape[1] if i == 0 else 10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )
        for i in range(3)
    ]
)
states = [features]
h = features
for mlp in mlps:
    h = mlp(gin_aggregate(h, adjacency, 0.1))
    states.append(h)
readouts = torch.stack([s.sum(0).norm() for s in states])
perm = torch.randperm(len(features))
p_adj = adjacency[perm][:, perm]
p_x = features[perm]
p_states = [p_x]
ph = p_x
for mlp in mlps:
    ph = mlp(gin_aggregate(ph, p_adj, 0.1))
    p_states.append(ph)
p_readouts = torch.stack([s.sum(0).norm() for s in p_states])
assert torch.allclose(readouts, p_readouts, atol=1e-5)
plt.plot(range(4), readouts.detach(), marker="o")
plt.xlabel("layer")
plt.ylabel("readout norm")
plt.title("GIN layer-wise readout")
plt.show()
""",
                "metric",
            ),
            # Cell 06 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 07 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    num_layers: int = 2
    learning_rate: float = 0.02
    steps: int = 12


def preprocess_graph(graph: Tensor) -> Tensor:
    raise NotImplementedError("TODO P-1: graph preprocessing")


def build_ego_graphs(
    inputs: Tensor,
    graph: Tensor,
    center_index: Tensor,
) -> list[tuple[Tensor, Tensor]]:
    raise NotImplementedError("TODO P-1: rooted graph examples")


class GinPortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        raise NotImplementedError("TODO P-1: epsilon, MLP, and classifier")

    def encode(self, inputs: Tensor, graph: Tensor) -> list[Tensor]:
        raise NotImplementedError("TODO P-1: two GIN message-passing layers")

    def graph_readout(self, inputs: Tensor, graph: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: permutation-invariant sum")

    def forward(self, inputs: Tensor, graph: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: paper equation")

    def compute_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: objective")

    def training_step(
        self,
        inputs: Tensor,
        graph: Tensor,
        targets: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")
""",
                r"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    hidden_dim: int = 12
    num_layers: int = 2
    learning_rate: float = 0.03
    steps: int = 18


def preprocess_graph(graph: Tensor) -> Tensor:
    return (graph > 0).to(graph.dtype)


def build_ego_graphs(
    inputs: Tensor,
    graph: Tensor,
    center_index: Tensor,
) -> list[tuple[Tensor, Tensor]]:
    examples = []
    for center in center_index.tolist():
        neighbors = torch.where(graph[center] > 0)[0]
        nodes = torch.cat(
            [torch.tensor([center]), neighbors],
        ).unique(sorted=True)
        ego_inputs = inputs[nodes].clone()
        ego_graph = graph[nodes][:, nodes].clone()
        examples.append((ego_inputs, ego_graph))
    return examples


class GinPortfolioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        config: PortfolioConfig,
    ):
        super().__init__()
        self.config = config
        if config.num_layers < 2:
            raise ValueError("GIN requires at least two message-passing layers")
        self.epsilons = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(0.0))
                for _ in range(config.num_layers)
            ]
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        input_dim if layer == 0 else config.hidden_dim,
                        config.hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                for layer in range(config.num_layers)
            ]
        )
        readout_dim = input_dim + config.num_layers * config.hidden_dim
        self.graph_classifier = nn.Linear(
            readout_dim,
            class_count,
        )

    def encode(self, inputs: Tensor, graph: Tensor) -> list[Tensor]:
        states = [inputs]
        hidden = inputs
        layer_parameters = zip(
            self.epsilons,
            self.mlps,
            strict=True,
        )
        for epsilon, mlp in layer_parameters:
            aggregated = gin_aggregate(hidden, graph, epsilon)
            hidden = mlp(aggregated)
            states.append(hidden)
        return states

    def graph_readout(self, inputs: Tensor, graph: Tensor) -> Tensor:
        layer_states = self.encode(inputs, graph)
        layer_readouts = [state.sum(dim=0) for state in layer_states]
        return torch.cat(layer_readouts, dim=0)

    def forward(self, inputs: Tensor, graph: Tensor) -> Tensor:
        readout = self.graph_readout(inputs, graph)
        return self.graph_classifier(readout).unsqueeze(0)

    def compute_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return F.cross_entropy(outputs, targets)

    def training_step(
        self,
        inputs: Tensor,
        graph: Tensor,
        targets: Tensor,
    ) -> Tensor:
        outputs = self(inputs, graph)
        return self.compute_loss(outputs, targets)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(
    model: nn.Module,
    graph_examples: list[tuple[Tensor, Tensor]],
    graph_targets: Tensor,
) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: GinPortfolioModel,
    graph_examples: list[tuple[Tensor, Tensor]],
    graph_targets: Tensor,
) -> list[float]:
    assert len(graph_examples) == len(graph_targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    model.train()
    for _ in range(model.config.steps):
        example_losses = []
        paired_examples = zip(
            graph_examples,
            graph_targets,
            strict=True,
        )
        for (inputs, graph), target in paired_examples:
            example_loss = model.training_step(
                inputs,
                graph,
                target.reshape(1),
            )
            example_losses.append(example_loss)
        loss = torch.stack(example_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    assert all(epsilon.grad is not None for epsilon in model.epsilons)
    assert model.graph_classifier.weight.grad is not None
    return history
""",
                "portfolio-training",
                "task",
            ),
            # Cell 10 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(
    model: nn.Module,
    graph_examples: list[tuple[Tensor, Tensor]],
    graph_targets: Tensor,
) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: GinPortfolioModel,
    graph_examples: list[tuple[Tensor, Tensor]],
    graph_targets: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        graph_logits = torch.cat(
            [model(inputs, graph) for inputs, graph in graph_examples],
            dim=0,
        )
        prediction = graph_logits.argmax(dim=1)
        accuracy = (prediction == graph_targets).float().mean()

        probe_inputs, probe_graph = graph_examples[0]
        permutation = torch.arange(
            len(probe_inputs) - 1,
            -1,
            -1,
            device=probe_inputs.device,
        )
        permuted_inputs = probe_inputs[permutation]
        permuted_graph = probe_graph[permutation][:, permutation]
        readout = model.graph_readout(probe_inputs, probe_graph)
        permuted_readout = model.graph_readout(
            permuted_inputs,
            permuted_graph,
        )
        permutation_error = (readout - permuted_readout).abs().max()
        assert torch.allclose(
            readout,
            permuted_readout,
            atol=1e-5,
        )
        assert torch.allclose(
            model(probe_inputs, probe_graph),
            model(permuted_inputs, permuted_graph),
            atol=1e-5,
        )

        layer_states = model.encode(probe_inputs, probe_graph)
        layer_norms = torch.stack(
            [state.sum(dim=0).norm() for state in layer_states]
        )

        degree_probe_graph = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            device=probe_inputs.device,
        )
        degree_probe_features = torch.ones(
            4,
            1,
            device=probe_inputs.device,
        )
        degree = degree_probe_graph.sum(dim=1, keepdim=True)
        mean_multiset = degree_probe_graph @ degree_probe_features
        mean_multiset = mean_multiset / degree.clamp_min(1.0)
        sum_multiset = gin_aggregate(
            degree_probe_features,
            degree_probe_graph,
            epsilon=0.0,
        )
        mean_unique = torch.unique(mean_multiset).numel()
        sum_unique = torch.unique(sum_multiset).numel()
        assert mean_unique == 1
        assert sum_unique > mean_unique
        assert torch.allclose(sum_multiset[:, 0], degree[:, 0] + 1.0)
    return {
        "heldout_ego_graph_accuracy": float(accuracy.cpu()),
        "readout_permutation_error": float(permutation_error.cpu()),
        "layer_readout_norm_mean": float(layer_norms.mean().cpu()),
        "layer_readout_count": float(len(layer_states)),
        "degree_multiset_unique_count": float(sum_unique),
    }


portfolio_train_index = torch.where(train_mask)[0]
portfolio_test_index = torch.where(test_mask)[0]
assert not torch.isin(portfolio_train_index, portfolio_test_index).any()
portfolio_config = PortfolioConfig()
portfolio_graph = preprocess_graph(adjacency)
portfolio_train_graphs = build_ego_graphs(
    features,
    portfolio_graph,
    portfolio_train_index,
)
portfolio_test_graphs = build_ego_graphs(
    features,
    portfolio_graph,
    portfolio_test_index,
)
portfolio_train_graphs = [
    ACCELERATOR.move(ego_inputs, ego_graph)
    for ego_inputs, ego_graph in portfolio_train_graphs
]
portfolio_test_graphs = [
    ACCELERATOR.move(ego_inputs, ego_graph)
    for ego_inputs, ego_graph in portfolio_test_graphs
]
portfolio_train_targets, portfolio_test_targets = ACCELERATOR.move(
    labels[portfolio_train_index],
    labels[portfolio_test_index],
)
portfolio_model = ACCELERATOR.move(
    GinPortfolioModel(
        features.shape[1],
        int(labels.max()) + 1,
        portfolio_config,
    )
)
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_train_graphs,
    portfolio_train_targets,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_test_graphs,
    portfolio_test_targets,
)
assert np.isfinite(portfolio_history).all()
assert 0.0 <= portfolio_metrics["heldout_ego_graph_accuracy"] <= 1.0
assert portfolio_metrics["readout_permutation_error"] <= 1e-5
assert portfolio_metrics["layer_readout_count"] == portfolio_config.num_layers + 1
expected_readout_dim = features.shape[1]
expected_readout_dim += portfolio_config.num_layers * portfolio_config.hidden_dim
assert portfolio_model.graph_classifier.in_features == expected_readout_dim
assert portfolio_metrics["degree_multiset_unique_count"] > 1.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 08 ? PinSage: random-walk 기반 추천 이웃
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=8,
        slug="pinsage",
        short_title="PinSage: random-walk 기반 추천 이웃",
        paper_title=(
            "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
        ),
        authors=(
            "Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Ha"
            "milton, Jure Leskovec"
        ),
        year=2018,
        primary_url="https://arxiv.org/abs/1806.01973",
        venue="KDD",
        difficulty="중급",
        expected_minutes=70,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="GraphSAGE, bipartite graph, ranking loss",
        reproduction_goal=(
            "user-item graph에서 item graph를 만들고 random-walk visit importance로 weig"
            "hted convolution을 수행합니다."
        ),
        original_scale="Pinterest 30억 node/180억 edge와 MapReduce 대신 6 user×9 item을 사용합니다.",
        mappings=(
            (
                "§3.2, importance-based neighborhoods",
                "random-walk visit count top-T",
                "L1 weight·top-k",
            ),
            (
                "§3.2, Algorithm 1 line 1",
                "importance pooling",
                "weighted mean",
            ),
            (
                "Algorithm 1 lines 2–3",
                "concat transform와 L2 normalize",
                "unit norm",
            ),
            (
                "§3.3, Eq. (1)",
                "max-margin ranking",
                "positive/negative score gap",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_v^{k}=\sigma\!\left(W^k[h_v^{k-1}\Vert\sum_{u\in\mathcal N(v)}\alpha_u h_u^{k-1}]\right)$$

- **기호 정의:** v는 item, α는 random-walk visit 기반 중요도, N(v)는 top-T 이웃입니다.
- **수식의 역할:** bipartite walk로 얻은 중요 이웃만 가중 집계해 웹 규모 추천을 근사합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.2, importance-based neighborhoods을 구조화합니다.
- **입출력 shape:** user-item [U, I] → item affinity [I, I] → user/item score [U, I]
- **평가:** BPR 또는 margin loss 감소와 held-out Recall@3
- **원문 대비 한계:** Pinterest 30억 node/180억 edge와 MapReduce 대신 6 user×9 item을 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$h_v^{k}=\sigma\!\left(W^k[h_v^{k-1}\Vert\sum_{u\in\mathcal N(v)}\alpha_u h_u^{k-1}]\right)$$

- **기호 정의:** v는 item, α는 random-walk visit 기반 중요도, N(v)는 top-T 이웃입니다.
- **수식의 역할:** bipartite walk로 얻은 중요 이웃만 가중 집계해 웹 규모 추천을 근사합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.2, importance-based neighborhoods을 구조화합니다.
- **입출력 shape:** user-item [U, I] → item affinity [I, I] → user/item score [U, I]
- **평가:** BPR 또는 margin loss 감소와 held-out Recall@3
- **원문 대비 한계:** Pinterest 30억 node/180억 edge와 MapReduce 대신 6 user×9 item을 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "code",
                r"""
user_item = torch.tensor(raw["user_item"], dtype=torch.float32)
item_graph = user_item.T @ user_item
item_graph.fill_diagonal_(0)
item_graph = (item_graph > 0).float()
assert user_item.ndim == 2 and item_graph.shape == (
    user_item.shape[1],
    user_item.shape[1],
)
""",
                r"""
user_item = torch.tensor(raw["user_item"], dtype=torch.float32)
item_graph = user_item.T @ user_item
item_graph.fill_diagonal_(0)
item_graph = (item_graph > 0).float()
assert user_item.ndim == 2 and item_graph.shape == (
    user_item.shape[1],
    user_item.shape[1],
)
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 08-1: item에서 시작한 짧은 walk의 visit count로 top-k 중요 이웃을 구하세요.
def importance_neighbors(adj: Tensor, start: int, walks=24, length=4, topk=3, seed=0):
    raise NotImplementedError("TODO 08-1")
""",
                r"""
def importance_neighbors(adj: Tensor, start: int, walks=24, length=4, topk=3, seed=0):
    rng = np.random.default_rng(seed)
    counts = torch.zeros(len(adj))
    for _ in range(walks):
        u = start
        for _ in range(length):
            nbr = torch.where(adj[u] > 0)[0].numpy()
            if not len(nbr):
                break
            u = int(rng.choice(nbr))
            counts[u] += 1
    counts[start] = 0
    idx = torch.topk(counts, min(topk, int((counts > 0).sum())), largest=True).indices
    weights = counts[idx] / counts[idx].sum().clamp_min(1)
    return idx, weights


idx, w = importance_neighbors(item_graph, 0)
assert len(idx) <= 3 and torch.isclose(w.sum(), torch.tensor(1.0))
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 08-2: Algorithm 1의 importance pooling convolution을 구현하세요.
class PinSageConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        raise NotImplementedError("TODO 08-2")

    def forward(self, x, adj):
        raise NotImplementedError("TODO 08-2")
""",
                r"""
class PinSageConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.neighbor = nn.Linear(dim, dim)
        self.combine = nn.Linear(2 * dim, dim)

    def forward(self, x, adj):
        if not hasattr(self, "_importance"):
            self._importance = [
                importance_neighbors(adj, u, seed=100 + u) for u in range(len(x))
            ]
        rows = []
        for u in range(len(x)):
            idx, w = self._importance[u]
            pooled = (
                (F.relu(self.neighbor(x[idx])) * w[:, None]).sum(0)
                if len(idx)
                else torch.zeros_like(x[u])
            )
            rows.append(F.relu(self.combine(torch.cat([x[u], pooled]))))
        return F.normalize(torch.stack(rows), dim=1)


item_x = torch.eye(item_graph.shape[0])
conv = PinSageConv(item_graph.shape[0])
item_z = conv(item_x, item_graph)
assert item_z.shape == item_x.shape and torch.allclose(
    item_z.norm(dim=1), torch.ones(len(item_z)), atol=1e-5
)
""",
                "task",
            ),
            # Cell 06 ? metric
            _cell(
                "code",
                r"""
# TODO 08-3: Eq. (1) margin ranking으로 positive co-view를 negative보다 가깝게 학습하세요.
raise NotImplementedError("TODO 08-3")
""",
                r"""
pos = torch.nonzero(torch.triu(item_graph, diagonal=1) > 0)
neg = torch.nonzero(
    torch.triu(1 - item_graph - torch.eye(len(item_graph)), diagonal=1) > 0
)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.03)
losses = []
for step in range(120):
    z = conv(item_x, item_graph)
    p = pos[step % len(pos)]
    n = neg[step % len(neg)]
    loss = F.relu(0.5 - (z[p[0]] * z[p[1]]).sum() + (z[n[0]] * z[n[1]]).sum())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss))
with torch.no_grad():
    z = conv(item_x, item_graph)
    sim = z @ z.T
    pos_mean = float(sim[item_graph.bool()].mean())
    non = (item_graph == 0) & ~torch.eye(len(item_graph), dtype=torch.bool)
    neg_mean = float(sim[non].mean())
assert pos_mean > neg_mean
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("margin loss")
ax[1].imshow(sim, cmap="viridis")
ax[1].set_title("item cosine")
plt.tight_layout()
plt.show()
print({"co_view_cosine": round(pos_mean, 3), "other_cosine": round(neg_mean, 3)})
""",
                "metric",
            ),
            # Cell 07 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 08 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    walk_count: int = 24
    walk_length: int = 4
    top_neighbors: int = 3
    margin: float = 0.5
    recall_k: int = 3
    learning_rate: float = 0.03
    steps: int = 24


def preprocess_graph(
    interactions: Tensor,
    config: PortfolioConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: holdout and walk importance")


class PinsagePortfolioModel(nn.Module):
    def __init__(self, item_count: int, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: weighted PinSage convolution")

    def forward(self, importance: Tensor) -> Tensor:
        raise NotImplementedError("TODO P-1: weighted concat and L2 normalize")

    def compute_loss(
        self,
        importance: Tensor,
        query: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: max-margin ranking")

    def training_step(
        self,
        importance: Tensor,
        query: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")


def sample_margin_triplets(
    item_graph: Tensor,
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: query/positive/negative samples")
""",
                r"""
from dataclasses import dataclass

@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    walk_count: int = 24
    walk_length: int = 4
    top_neighbors: int = 3
    margin: float = 0.5
    recall_k: int = 3
    learning_rate: float = 0.03
    steps: int = 24


def preprocess_graph(
    interactions: Tensor,
    config: PortfolioConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    train_interactions = interactions.clone()
    heldout = []
    for user_id, row in enumerate(train_interactions):
        positives = torch.where(row > 0)[0]
        assert len(positives) >= 2
        heldout_item = int(positives[-1])
        heldout.append(heldout_item)
        train_interactions[user_id, heldout_item] = 0

    item_graph = train_interactions.T @ train_interactions
    item_graph.fill_diagonal_(0)
    item_graph = (item_graph > 0).float()
    importance = torch.zeros_like(item_graph)
    for item_id in range(len(item_graph)):
        neighbors, weights = importance_neighbors(
            item_graph,
            item_id,
            walks=config.walk_count,
            length=config.walk_length,
            topk=config.top_neighbors,
            seed=100 + item_id,
        )
        importance[item_id, neighbors] = weights
    return train_interactions, item_graph, importance, torch.tensor(heldout)


class PinsagePortfolioModel(nn.Module):
    def __init__(self, item_count: int, config: PortfolioConfig):
        super().__init__()
        self.config = config
        self.item_embedding = nn.Parameter(
            torch.randn(item_count, config.embedding_dim) * 0.1
        )
        self.neighbor_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )
        self.combine = nn.Linear(config.embedding_dim * 2, config.embedding_dim)

    def forward(self, importance: Tensor) -> Tensor:
        neighbor_hidden = F.relu(self.neighbor_projection(self.item_embedding))
        pooled = importance @ neighbor_hidden
        combined = torch.cat([self.item_embedding, pooled], dim=1)
        return F.normalize(F.relu(self.combine(combined)), dim=1)

    def compute_loss(
        self,
        importance: Tensor,
        query: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        item_embedding = self(importance)
        positive_score = (item_embedding[query] * item_embedding[positive]).sum(1)
        negative_score = (item_embedding[query] * item_embedding[negative]).sum(1)
        return F.relu(self.config.margin - positive_score + negative_score).mean()

    def training_step(
        self,
        importance: Tensor,
        query: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        return self.compute_loss(importance, query, positive, negative)


def sample_margin_triplets(
    item_graph: Tensor,
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    queries = []
    positives = []
    negatives = []
    for query, row in enumerate(item_graph):
        observed = torch.where(row > 0)[0]
        missing = torch.where(row == 0)[0]
        missing = missing[missing != query]
        if len(observed) and len(missing):
            queries.append(torch.tensor(query))
            positives.append(observed[step % len(observed)])
            negatives.append(missing[(step + query) % len(missing)])
    return torch.stack(queries), torch.stack(positives), torch.stack(negatives)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 09 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 10 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: PinsagePortfolioModel,
    importance: Tensor,
    item_graph: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    model.train()
    for step in range(model.config.steps):
        query, positive, negative = sample_margin_triplets(item_graph, step)
        query, positive, negative = ACCELERATOR.move(query, positive, negative)
        loss = model.training_step(importance, query, positive, negative)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 12 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: PinsagePortfolioModel,
    importance: Tensor,
    train_interactions: Tensor,
    heldout: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        item_embedding = model(importance)
        interactions_device, heldout_device = ACCELERATOR.move(
            train_interactions,
            heldout,
        )
        weights = interactions_device / interactions_device.sum(1, keepdim=True)
        user_embedding = F.normalize(weights @ item_embedding, dim=1)
        scores = user_embedding @ item_embedding.T
        scores = scores.masked_fill(interactions_device.bool(), -torch.inf)
        k = min(model.config.recall_k, scores.shape[1])
        recommendations = scores.topk(k, dim=1).indices
        recall = (recommendations == heldout_device[:, None]).any(dim=1).float().mean()

        item_graph = train_interactions.T @ train_interactions
        item_graph.fill_diagonal_(0)
        item_graph = (item_graph > 0).float()
        query, positive, negative = sample_margin_triplets(item_graph, 0)
        query, positive, negative = ACCELERATOR.move(query, positive, negative)
        positive_score = (item_embedding[query] * item_embedding[positive]).sum(1)
        negative_score = (item_embedding[query] * item_embedding[negative]).sum(1)
        margin_satisfied = (
            positive_score >= negative_score + model.config.margin
        ).float().mean()
    return {
        f"recall_at_{k}": float(recall.cpu()),
        "margin_satisfaction": float(margin_satisfied.cpu()),
    }


portfolio_config = PortfolioConfig()
(
    portfolio_train_ui,
    portfolio_item_graph,
    portfolio_importance,
    portfolio_heldout,
) = preprocess_graph(user_item, portfolio_config)
portfolio_importance_device = ACCELERATOR.move(portfolio_importance)
portfolio_model = ACCELERATOR.move(
    PinsagePortfolioModel(user_item.shape[1], portfolio_config)
)
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_importance_device,
    portfolio_item_graph,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_importance_device,
    portfolio_train_ui,
    portfolio_heldout,
)
assert np.isfinite(portfolio_history).all()
assert 0.0 <= portfolio_metrics["recall_at_3"] <= 1.0
print(portfolio_metrics)
""",
                "portfolio-evaluation",
                "metric",
            ),
        ),
    ),
    # 09 ? LightGCN: 추천에는 전파만 남기기
    FieldPaperSpec(
        field_id="graph_recommendation",
        field_title="그래프 학습 · 추천 시스템",
        number=9,
        slug="lightgcn",
        short_title="LightGCN: 추천에는 전파만 남기기",
        paper_title=(
            "LightGCN: Simplifying and Powering Graph Convolution Network for Rec"
            "ommendation"
        ),
        authors=(
            "Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang"
        ),
        year=2020,
        primary_url="https://arxiv.org/abs/2002.02126",
        venue="SIGIR",
        difficulty="중급",
        expected_minutes=75,
        dataset_file="data/field_curriculum/graph_recommendation.npz",
        prerequisites="collaborative filtering, bipartite graph, BPR",
        reproduction_goal="비선형/feature transform 없이 user-item embedding을 전파하고 BPR로 학습합니다.",
        original_scale=(
            "Gowalla/Yelp2018/Amazon-Book 대신 6 user×9 item과 leave-one-out 평가를 사용합"
            "니다."
        ),
        mappings=(
            (
                "§3.1.1, Eq. (3)–(4)",
                "대칭 degree-normalized user↔item 전파",
                "한 층 shape",
            ),
            (
                "§3.1.2, Eq. (7)",
                "0..K층 weighted sum",
                "층 평균",
            ),
            (
                "Eq. (8)",
                "user-item inner-product score",
                "score matrix",
            ),
            (
                "§3.1.3, Eq. (9)",
                "BPR loss",
                "loss·Recall@3",
            ),
        ),
        cells=(
            # Cell 01 ? setup
            _cell(
                "code",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                r"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from llm_engineering_lab.acceleration import get_accelerator

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(7)

DATA_PATH = Path("data/field_curriculum/graph_recommendation.npz")
assert DATA_PATH.exists(), f"먼저 로컬 학습 데이터를 준비하세요: {DATA_PATH}"
raw = np.load(DATA_PATH, allow_pickle=False)
adjacency = torch.tensor(raw["adjacency"], dtype=torch.float32)
features = torch.tensor(raw["features"], dtype=torch.float32)
labels = torch.tensor(raw["labels"], dtype=torch.long)
train_mask = torch.tensor(raw["train_mask"], dtype=torch.bool)
test_mask = torch.tensor(raw["test_mask"], dtype=torch.bool)
assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1] == len(features)
assert torch.allclose(adjacency, adjacency.T) and not torch.diagonal(adjacency).any()
edge_count = int(adjacency.sum().item() / 2)
print(
    f"nodes={len(features)}, "
    f"edges={edge_count}, "
    f"features={features.shape[1]}"
)
print(ACCELERATOR.summary())
print(
    "graph sampling/topology preprocessing stays on CPU, "
    "while neural training uses DEVICE"
)
""",
                "setup",
            ),
            # Cell 02 ? portfolio-theory
            _cell(
                "markdown",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$e_u^{(k+1)}=\sum_{i\in\mathcal N_u}\frac{e_i^{(k)}}{\sqrt{|\mathcal N_u||\mathcal N_i|}},
\quad e_u=\sum_{k=0}^{K}\alpha_k e_u^{(k)}$$

- **기호 정의:** e는 user/item embedding, K는 전파 깊이, α는 층 결합 계수입니다.
- **수식의 역할:** feature transform과 활성화를 제거하고 협업 신호 전파 자체만 유지합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.1.1, Eq. (3)–(4)을 구조화합니다.
- **입출력 shape:** user-item [U, I] → bipartite 전파 [U+I, U+I] → score [U, I]
- **평가:** BPR loss, Recall@3, 층별 embedding norm
- **원문 대비 한계:** Gowalla/Yelp2018/Amazon-Book 대신 6 user×9 item과 leave-one-out 평가를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                r"""
## 포트폴리오 해설: 핵심 수식에서 실행 코드까지

$$e_u^{(k+1)}=\sum_{i\in\mathcal N_u}\frac{e_i^{(k)}}{\sqrt{|\mathcal N_u||\mathcal N_i|}},
\quad e_u=\sum_{k=0}^{K}\alpha_k e_u^{(k)}$$

- **기호 정의:** e는 user/item embedding, K는 전파 깊이, α는 층 결합 계수입니다.
- **수식의 역할:** feature transform과 활성화를 제거하고 협업 신호 전파 자체만 유지합니다.
- **구현 이유:** 원문 전체를 축약하더라도 위 관계가 결과를 만드는 핵심이므로,
  이 수식을 전처리·`forward`·loss 경계로 나누어 직접 구현했습니다.
- **Task/코드 대응:** Task P의 `preprocess_graph`, 논문 전용 클래스의
  `forward`, `compute_loss`, `training_step`이 §3.1.1, Eq. (3)–(4)을 구조화합니다.
- **입출력 shape:** user-item [U, I] → bipartite 전파 [U+I, U+I] → score [U, I]
- **평가:** BPR loss, Recall@3, 층별 embedding norm
- **원문 대비 한계:** Gowalla/Yelp2018/Amazon-Book 대신 6 user×9 item과 leave-one-out 평가를 사용합니다.

코드를 읽을 때는 `Config → 전처리 → Module.forward → loss → train → evaluate`
순서로 추적하세요. 각 함수는 한 가지 책임만 갖도록 분리했습니다.
""",
                "portfolio-theory",
            ),
            # Cell 03 ? shared
            _cell(
                "code",
                r"""
user_item = torch.tensor(raw["user_item"], dtype=torch.float32)
train_ui = user_item.clone()
heldout = []
for u in range(len(train_ui)):
    positive = torch.where(train_ui[u] > 0)[0]
    assert len(positive) >= 2
    item = int(positive[-1])
    heldout.append(item)
    train_ui[u, item] = 0
heldout = torch.tensor(heldout)
""",
                r"""
user_item = torch.tensor(raw["user_item"], dtype=torch.float32)
train_ui = user_item.clone()
heldout = []
for u in range(len(train_ui)):
    positive = torch.where(train_ui[u] > 0)[0]
    assert len(positive) >= 2
    item = int(positive[-1])
    heldout.append(item)
    train_ui[u, item] = 0
heldout = torch.tensor(heldout)
""",
            ),
            # Cell 04 ? task
            _cell(
                "code",
                r"""
# TODO 09-1: user-item bipartite adjacency와 대칭 정규화 행렬을 만드세요.
def bipartite_normalized(ui: Tensor) -> Tensor:
    raise NotImplementedError("TODO 09-1")
""",
                r"""
def bipartite_normalized(ui: Tensor) -> Tensor:
    nu, ni = ui.shape
    a = torch.zeros(nu + ni, nu + ni, device=ui.device, dtype=ui.dtype)
    a[:nu, nu:] = ui
    a[nu:, :nu] = ui.T
    d = a.sum(1).clamp_min(1).pow(-0.5)
    return d[:, None] * a * d[None, :]


propagation = bipartite_normalized(train_ui)
assert torch.allclose(propagation, propagation.T)
""",
                "task",
            ),
            # Cell 05 ? task
            _cell(
                "code",
                r"""
# TODO 09-2: Eq. (3),(4),(7)처럼 K회 선형 전파 후 모든 층을 평균하세요.
def lightgcn_embeddings(initial: Tensor, prop: Tensor, layers: int):
    raise NotImplementedError("TODO 09-2")
""",
                r"""
def lightgcn_embeddings(initial: Tensor, prop: Tensor, layers: int):
    states = [initial]
    h = initial
    for _ in range(layers):
        h = prop @ h
        states.append(h)
    return torch.stack(states).mean(0), states


probe = torch.randn(len(propagation), 8)
final, states = lightgcn_embeddings(probe, propagation, 2)
assert len(states) == 3 and torch.allclose(
    final, (states[0] + states[1] + states[2]) / 3
)
""",
                "task",
            ),
            # Cell 06 ? task
            _cell(
                "code",
                r"""
# TODO 09-3: Eq. (9)의 BPR loss로 초기 embedding만 학습하세요.
raise NotImplementedError("TODO 09-3")
""",
                r"""
nu, ni = train_ui.shape
propagation_device, train_ui_device, heldout_device = ACCELERATOR.move(
    propagation, train_ui, heldout
)
initial = nn.Parameter(torch.randn(nu + ni, 10, device=DEVICE) * 0.1)
optimizer = torch.optim.Adam([initial], lr=0.05)
positives = [torch.where(train_ui[u] > 0)[0] for u in range(nu)]
negatives = [torch.where(user_item[u] == 0)[0] for u in range(nu)]
losses = []
for step in range(180):
    emb, _ = lightgcn_embeddings(initial, propagation_device, 2)
    ue, ie = emb[:nu], emb[nu:]
    pos = torch.tensor(
        [int(positives[u][step % len(positives[u])]) for u in range(nu)], device=DEVICE
    )
    neg = torch.tensor(
        [int(negatives[u][step % len(negatives[u])]) for u in range(nu)], device=DEVICE
    )
    pos_score = (ue * ie[pos]).sum(1)
    neg_score = (ue * ie[neg]).sum(1)
    loss = -F.logsigmoid(pos_score - neg_score).mean() + 1e-4 * initial.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
assert losses[-1] < losses[0]
""",
                "task",
            ),
            # Cell 07 ? metric
            _cell(
                "code",
                r"""
# TODO 09-4: held-out Recall@3와 층별 embedding norm을 시각화하세요.
raise NotImplementedError("TODO 09-4")
""",
                r"""
with torch.no_grad():
    emb, states = lightgcn_embeddings(initial, propagation_device, 2)
    scores = emb[:nu] @ emb[nu:].T
    scores[train_ui_device.bool()] = -1e9
    top3 = scores.topk(3, dim=1).indices
    recall = float((top3 == heldout_device[:, None]).any(1).float().mean())
    norms = [float(s.norm(dim=1).mean()) for s in states]
assert 0 <= recall <= 1 and top3.shape == (nu, 3)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(losses)
ax[0].set_title("BPR loss")
ax[1].plot(range(3), norms, marker="o")
ax[1].set_title("layer embedding norm")
plt.tight_layout()
plt.show()
print({"Recall@3": recall, "heldout": heldout.tolist(), "top3": top3.cpu().tolist()})
""",
                "metric",
            ),
            # Cell 08 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                r"""
### Task P1 · 전처리와 핵심 모델

먼저 graph 입력의 shape와 정규화 규칙을 고정하고, 논문의 핵심 연산을
`nn.Module`의 `forward`와 `compute_loss`로 분리합니다.
""",
                "portfolio-stage",
            ),
            # Cell 09 ? portfolio-model, task
            _cell(
                "code",
                r"""
# TODO P-1: 논문의 핵심 연산을 아래 구조에 직접 구현하세요.
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    layers: int = 2
    regularization: float = 1e-4
    recall_k: int = 3
    learning_rate: float = 0.03
    steps: int = 16


def preprocess_graph(interactions: Tensor) -> Tensor:
    raise NotImplementedError("TODO P-1: normalized bipartite graph")


class LightgcnPortfolioModel(nn.Module):
    def __init__(self, user_count: int, item_count: int, config: PortfolioConfig):
        super().__init__()
        raise NotImplementedError("TODO P-1: trainable user/item embeddings")

    def forward(self, graph: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("TODO P-1: layer-wise propagation and mean")

    def compute_loss(
        self,
        graph: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: BPR objective")

    def training_step(
        self,
        graph: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        raise NotImplementedError("TODO P-1: train step")


def sample_ranking_triplets(
    interactions: Tensor,
    heldout: Tensor,
    step: int,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("TODO P-1: leakage-free BPR samples")
""",
                r"""
from dataclasses import dataclass

@dataclass(frozen=True)
class PortfolioConfig:
    embedding_dim: int = 10
    layers: int = 2
    regularization: float = 1e-4
    recall_k: int = 3
    learning_rate: float = 0.03
    steps: int = 16


def preprocess_graph(interactions: Tensor) -> Tensor:
    user_count, item_count = interactions.shape
    graph = torch.zeros(user_count + item_count, user_count + item_count)
    graph[:user_count, user_count:] = interactions
    graph[user_count:, :user_count] = interactions.T
    degree = graph.sum(1).clamp_min(1.0).pow(-0.5)
    return degree[:, None] * graph * degree[None, :]


class LightgcnPortfolioModel(nn.Module):
    def __init__(self, user_count: int, item_count: int, config: PortfolioConfig):
        super().__init__()
        self.user_count = user_count
        self.config = config
        total = user_count + item_count
        self.embedding = nn.Parameter(torch.randn(total, config.embedding_dim) * 0.1)

    def forward(self, graph: Tensor) -> tuple[Tensor, Tensor]:
        states = [self.embedding]
        for _ in range(self.config.layers):
            states.append(graph @ states[-1])
        combined = torch.stack(states).mean(dim=0)
        return combined[: self.user_count], combined[self.user_count :]

    def compute_loss(self, graph: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        user_embedding, item_embedding = self(graph)
        user_ids = torch.arange(self.user_count, device=graph.device)
        positive_score = (user_embedding * item_embedding[positive]).sum(dim=1)
        negative_score = (user_embedding * item_embedding[negative]).sum(dim=1)
        ranking = -F.logsigmoid(positive_score - negative_score).mean()
        regularizer = self.embedding.square().mean()
        return ranking + self.config.regularization * regularizer

    def training_step(
        self,
        graph: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        return self.compute_loss(graph, positive, negative)


def sample_ranking_triplets(
    interactions: Tensor,
    heldout: Tensor,
    step: int,
) -> tuple[Tensor, Tensor]:
    positive = []
    negative = []
    for user_id, row in enumerate(interactions):
        observed = torch.where(row > 0)[0]
        missing = torch.where(row == 0)[0]
        missing = missing[missing != heldout[user_id]]
        assert len(observed) and len(missing)
        positive.append(observed[step % len(observed)])
        negative.append(missing[(step + user_id) % len(missing)])
    return torch.stack(positive), torch.stack(negative)
""",
                "portfolio-model",
                "task",
            ),
            # Cell 10 ? portfolio-stage
            _cell(
                "markdown",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                r"""
### Task P2 · objective와 update

`fit_portfolio_model`은 gradient가 흐르는 parameter와 한 step의 순서를
드러냅니다. 논문의 목적함수가 코드의 어느 tensor에 적용되는지 확인하세요.
""",
                "portfolio-stage",
            ),
            # Cell 11 ? portfolio-training, task
            _cell(
                "code",
                r"""
def fit_portfolio_model(model: nn.Module, graph: Tensor) -> list[float]:
    raise NotImplementedError("TODO P-1: optimization loop")
""",
                r"""
def fit_portfolio_model(
    model: LightgcnPortfolioModel,
    graph: Tensor,
    interactions: Tensor,
    heldout: Tensor,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    history = []
    model.train()
    for step in range(model.config.steps):
        positive, negative = sample_ranking_triplets(interactions, heldout, step)
        positive, negative = ACCELERATOR.move(positive, negative)
        loss = model.training_step(graph, positive, negative)
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
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                r"""
### Task P3 · 평가와 한계 기록

loss와 별개의 논문 지표를 계산하고, 로컬 합성 graph에서 얻은 수치가
원 논문의 대규모 benchmark 결과를 대신하지 않는다는 점을 기록합니다.
""",
                "portfolio-stage",
            ),
            # Cell 13 ? portfolio-evaluation, metric
            _cell(
                "code",
                r"""
def evaluate_portfolio_model(model: nn.Module, graph: Tensor) -> dict[str, float]:
    raise NotImplementedError("TODO P-1: evaluation")
""",
                r"""
def evaluate_portfolio_model(
    model: LightgcnPortfolioModel,
    graph: Tensor,
    interactions: Tensor,
    heldout: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        user_embedding, item_embedding = model(graph)
        scores = user_embedding @ item_embedding.T
        interactions_device, heldout_device = ACCELERATOR.move(interactions, heldout)
        scores = scores.masked_fill(interactions_device.bool(), -torch.inf)
        k = min(model.config.recall_k, scores.shape[1])
        recommendations = scores.topk(k, dim=1).indices
        recall = (recommendations == heldout_device[:, None]).any(dim=1).float().mean()
    return {f"recall_at_{k}": float(recall.cpu())}


portfolio_config = PortfolioConfig()
portfolio_graph = ACCELERATOR.move(preprocess_graph(train_ui))
portfolio_model = ACCELERATOR.move(
    LightgcnPortfolioModel(train_ui.shape[0], train_ui.shape[1], portfolio_config)
)
portfolio_history = fit_portfolio_model(
    portfolio_model,
    portfolio_graph,
    train_ui,
    heldout,
)
portfolio_metrics = evaluate_portfolio_model(
    portfolio_model,
    portfolio_graph,
    train_ui,
    heldout,
)
assert np.isfinite(portfolio_history).all()
assert 0.0 <= portfolio_metrics["recall_at_3"] <= 1.0
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
