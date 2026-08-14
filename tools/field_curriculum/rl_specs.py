"""Ten portfolio-grade, offline-first reproductions of landmark RL papers."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from textwrap import dedent, fill

from .common import (
    FieldPaperSpec,
    code,
    markdown,
    paired_markdown,
    shared_code,
)

FIELD_ID = "reinforcement_learning"
FIELD_TITLE = "강화학습·에이전트"
DATASET = "data/field_curriculum/gridworld.json"


def _setup(seed: int) -> object:
    return shared_code(
        f"""
        from pathlib import Path
        import json
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        import torch.nn.functional as F
        from llm_engineering_lab.acceleration import get_accelerator

        torch.manual_seed({seed})
        np.random.seed({seed})
        ACCELERATOR = get_accelerator()
        DEVICE = ACCELERATOR.device
        # Tiny scalar/table loops may be faster on CPU even when a GPU exists;
        # AI_LAB_DEVICE=cpu remains the low-overhead override.
        DATA_PATH = Path({DATASET!r})
        assert DATA_PATH.exists(), f"준비된 로컬 데이터가 없습니다: {{DATA_PATH}}"
        config = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        required = {{
            "seed",
            "size",
            "start_state",
            "terminal_state",
            "gamma",
            "transitions",
            "bandit_means",
            "offline_action_noise",
        }}
        assert required.issubset(config), f"JSON 키 불일치: {{sorted(config)}}"
        rows = config["transitions"]
        states = torch.tensor([int(r["state"]) for r in rows], dtype=torch.long)
        actions = torch.tensor([int(r["action"]) for r in rows], dtype=torch.long)
        rewards = torch.tensor([float(r["reward"]) for r in rows], dtype=torch.float32)
        next_states = torch.tensor([int(r["next_state"]) for r in rows], dtype=torch.long)
        dones = torch.tensor([float(r["done"]) for r in rows], dtype=torch.float32)
        n_states = int(max(states.max(), next_states.max()).item()) + 1
        n_actions = int(actions.max().item()) + 1
        gamma = float(config["gamma"])
        start_state, terminal_state = int(config["start_state"]), int(config["terminal_state"])
        transition_table = {{
            (int(r["state"]), int(r["action"])): (
                int(r["next_state"]),
                float(r["reward"]),
                bool(r["done"]),
            )
            for r in rows
        }}

        def encode(s):
            return F.one_hot(torch.as_tensor(s, dtype=torch.long), n_states).float()

        def encode_device(s):
            return encode(s).to(DEVICE)

        def env_step(state, action):
            return transition_table[(int(state), int(action))]

        def value_iteration(iterations=300):
            q = torch.zeros(n_states, n_actions)
            for _ in range(iterations):
                old = q.clone()
                for (s, a), (ns, r, done) in transition_table.items():
                    q[s, a] = r if done else r + gamma * old[ns].max()
            return q

        q_star = value_iteration()
        assert torch.isfinite(q_star).all() and len(rows) > 0
        print(ACCELERATOR.summary())
        print(
            "RL environment, replay metadata, and tree search stay on CPU; "
            "neural batches use DEVICE."
        )
        print(
            f"local MDP: states={{n_states}}, actions={{n_actions}}, "
            f"transitions={{len(rows)}}, gamma={{gamma}}"
        )
        """,
        "setup",
    )


def _intro(body: str) -> object:
    return markdown(body, "paper-map")


REINFORCE = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=0,
    slug="reinforce",
    short_title="REINFORCE",
    paper_title=(
        "Simple Statistical Gradient-Following Algorithms for "
        "Connectionist Reinforcement Learning"
    ),
    authors="Ronald J. Williams",
    year=1992,
    primary_url="https://doi.org/10.1007/BF00992696",
    venue="Machine Learning",
    difficulty="초급",
    expected_minutes=50,
    dataset_file=DATASET,
    prerequisites="softmax policy, Monte Carlo return, log-derivative trick",
    reproduction_goal=(
        "로컬 bandit means에서 reward-baseline과 characteristic "
        "eligibility로 policy gradient를 재현한다."
    ),
    original_scale=(
        "논문은 여러 connectionist stochastic units와 episodic tasks를 다룬다. "
        "축소판은 가장 투명한 softmax bandit REINFORCE를 사용한다."
    ),
    mappings=(
        (
            "§3, p. 234, core REINFORCE rule",
            "reward-minus-baseline log-policy update",
            "Δw=α(r-b)∂log g/∂w를 자동미분 loss로 구현한다.",
        ),
        (
            "§3, Eq. (7)–(10)",
            "characteristic eligibility and adaptive baseline",
            "score-function gradient와 moving baseline을 계산한다.",
        ),
        (
            "§5, Eq. (11) and Theorem 2",
            "episodic eligibility accumulation",
            "discounted episode return으로 일반화되는 위치를 확인한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Williams의 핵심 규칙 $\Delta w=\alpha(r-b)\nabla_w\log\pi_w(a)$를 로컬
        Gaussian bandit에 적용합니다. baseline은 기대 gradient를 바꾸지 않고 분산을 낮춥니다."""),
        _setup(401),
        code(
            """
            # TODO: reward sequence의 discounted return-to-go를 역방향으로 계산하세요.
            def discounted_returns(reward_sequence, discount):
                raise NotImplementedError


            assert torch.allclose(
                discounted_returns(torch.tensor([1.0, 2.0]), 0.5), torch.tensor([2.0, 2.0])
            )
            """,
            """
            def discounted_returns(reward_sequence, discount):
                out = torch.empty_like(reward_sequence)
                running = 0.0
                for i in reversed(range(len(reward_sequence))):
                    running = float(reward_sequence[i]) + discount * running
                    out[i] = running
                return out


            assert torch.allclose(
                discounted_returns(torch.tensor([1.0, 2.0]), 0.5), torch.tensor([2.0, 2.0])
            )
            """,
            "implementation",
        ),
        code(
            """
            # TODO: softmax logits를 (r-baseline)*log pi(a)로 400 bandit rounds 학습하세요.
            raise NotImplementedError
            """,
            """
            # A one-vector bandit update plus a NumPy reward source is CPU-fast;
            # transferring every scalar round would cost more than the optimization.
            means = torch.tensor(config["bandit_means"], dtype=torch.float32)
            logits = nn.Parameter(torch.zeros(len(means)))
            opt = torch.optim.Adam([logits], lr=0.06)
            rng = torch.Generator().manual_seed(402)
            noise_rng = np.random.default_rng(402)
            baseline = 0.0
            reward_history = []
            prob_history = []
            for _ in range(400):
                probs = logits.softmax(0)
                action = int(torch.multinomial(probs, 1, generator=rng))
                reward = float(means[action]) + float(noise_rng.normal(0, 0.15))
                advantage = reward - baseline
                loss = -torch.log(probs[action].clamp_min(1e-8)) * advantage
                opt.zero_grad()
                loss.backward()
                opt.step()
                baseline = 0.95 * baseline + 0.05 * reward
                reward_history.append(reward)
                prob_history.append(probs.detach().cpu().numpy())
            assert np.isfinite(reward_history).all() and int(logits.argmax()) == int(
                means.argmax()
            )
            """,
            "training",
        ),
        code(
            """
            # TODO: 최적 arm 확률과 30-step reward 평균을 그리고 success metric을 검증하세요.
            raise NotImplementedError
            """,
            """
            final_probs = logits.softmax(0).detach()
            best = int(means.argmax())
            print("policy=", final_probs.round(decimals=3).tolist())
            assert float(final_probs[best]) > 0.65
            moving = np.convolve(reward_history, np.ones(30) / 30, mode="valid")
            ph = np.asarray(prob_history)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(moving)
            axes[0].set_title("30-round reward")
            axes[1].plot(ph)
            axes[1].set_title("arm probabilities")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "action과 무관한 baseline이 gradient 기대값을 바꾸지 않는 이유는?",
            "Σ_a π(a)∇logπ(a)=Σ_a∇π(a)=∇1=0이므로 action-independent baseline 항의 기대 gradient는 0입니다.",
            "reflection",
        ),
    ),
)


DQN = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=1,
    slug="dqn",
    short_title="DQN",
    paper_title="Human-level Control through Deep Reinforcement Learning",
    authors="Volodymyr Mnih et al.",
    year=2015,
    primary_url="https://www.nature.com/articles/nature14236",
    venue="Nature",
    difficulty="중급",
    expected_minutes=55,
    dataset_file=DATASET,
    prerequisites="Bellman equation, replay, target network",
    reproduction_goal="로컬 transition replay에서 one-hot DQN과 frozen target network의 TD loss를 학습한다.",
    original_scale=(
        "논문은 raw Atari frames, convolutional Q-network, 대규모 replay를 "
        "사용한다. 축소판은 동일한 target/replay mechanics를 작은 MDP에 적용한다."
    ),
    mappings=(
        (
            "Methods, ‘Deep reinforcement learning’, Eq. (1)–(2)",
            "squared TD loss and frozen target",
            "r+γmax Q_target로 online Q를 회귀한다.",
        ),
        (
            "Methods, experience replay paragraph",
            "uniform replay minibatches",
            "준비된 transition rows에서 무작위 mini-batch를 뽑는다.",
        ),
        (
            "Extended Data, Algorithm 1",
            "DQN training loop",
            "epsilon interaction 대신 offline replay를 사용하되 update와 target sync를 보존한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Methods Eq. (2)의 target $y=r+\gamma(1-d)\max_{a'}Q_{target}(s',a')$를
        구현합니다. target tensor가 gradient graph와 분리되는지도 검사합니다."""),
        _setup(411),
        code(
            """
            # TODO: terminal mask와 detach를 포함한 DQN Bellman target을 구현하세요.
            def dqn_target(r, done, next_q, discount):
                raise NotImplementedError


            probe = dqn_target(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0.0, 1.0]),
                torch.tensor([[3.0, 4.0], [9.0, 8.0]], requires_grad=True),
                0.5,
            )
            assert (
                torch.allclose(probe, torch.tensor([3.0, 2.0])) and not probe.requires_grad
            )
            """,
            """
            def dqn_target(r, done, next_q, discount):
                return (r + discount * (1 - done) * next_q.max(1).values).detach()


            probe = dqn_target(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0.0, 1.0]),
                torch.tensor([[3.0, 4.0], [9.0, 8.0]], requires_grad=True),
                0.5,
            )
            assert (
                torch.allclose(probe, torch.tensor([3.0, 2.0])) and not probe.requires_grad
            )
            """,
            "implementation",
        ),
        code(
            """
            # TODO: online/target MLP, uniform replay, 20-step target sync로 260 updates
            # 수행하세요.
            raise NotImplementedError
            """,
            """
            def qnet():
                return nn.Sequential(
                    nn.Linear(n_states, 32), nn.ReLU(), nn.Linear(32, n_actions)
                ).to(DEVICE)


            online, target = qnet(), qnet()
            target.load_state_dict(online.state_dict())
            opt = torch.optim.Adam(online.parameters(), lr=0.01)
            rng = torch.Generator().manual_seed(412)
            history = []
            for step in range(260):
                idx = torch.randint(len(rows), (min(32, len(rows)),), generator=rng)
                s, a, r, ns, d = ACCELERATOR.move(
                    states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]
                )
                pred = online(encode_device(s)).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    y = dqn_target(r, d, target(encode_device(ns)), gamma)
                loss = F.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                history.append(float(loss.detach()))
                if (step + 1) % 20 == 0:
                    target.load_state_dict(online.state_dict())
            assert np.isfinite(history).all() and np.mean(history[-20:]) < np.mean(
                history[:20]
            )
            """,
            "training",
        ),
        code(
            """
            # TODO: learned Q의 Bellman residual과 optimal greedy agreement를 계산하고 그리세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                learned = online(encode_device(torch.arange(n_states)))
                s, a, r, ns, d, q_ref = ACCELERATOR.move(
                    states, actions, rewards, next_states, dones, q_star
                )
                pred = learned[s, a]
                y = r + gamma * (1 - d) * learned[ns].max(1).values
                residual = float((pred - y).abs().mean())
                nonterminal = torch.arange(n_states, device=DEVICE) != terminal_state
                agreement = float(
                    (learned.argmax(1)[nonterminal] == q_ref.argmax(1)[nonterminal])
                    .float()
                    .mean()
                )
            print(f"Bellman MAE={residual:.4f}, greedy agreement={agreement:.3f}")
            assert math.isfinite(residual) and agreement >= 0.5
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(history)
            axes[0].set_title("replay TD loss")
            axes[1].imshow(learned.detach().cpu().T, aspect="auto", cmap="viridis")
            axes[1].set(title="Q(s,a)", xlabel="state", ylabel="action")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "target network을 매 update 동기화하지 않는 이유는?",
            (
                "회귀 목표가 online Q와 동시에 빠르게 움직이면 bootstrap feedback이 불안정해집니다. "
                "일정 기간 고정된 target은 supervised-like objective를 제공합니다."
            ),
            "reflection",
        ),
    ),
)


DOUBLE_DQN = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=2,
    slug="double_dqn",
    short_title="Double DQN",
    paper_title="Deep Reinforcement Learning with Double Q-learning",
    authors="Hado van Hasselt, Arthur Guez, and David Silver",
    year=2016,
    primary_url="https://ojs.aaai.org/index.php/AAAI/article/view/10295",
    venue="AAAI",
    difficulty="중급",
    expected_minutes=55,
    dataset_file=DATASET,
    prerequisites="DQN, maximization bias, independent estimators",
    reproduction_goal=(
        "online action selection과 target evaluation을 분리하고 noisy-value "
        "overestimation을 계량한다."
    ),
    original_scale=(
        "논문은 Atari 49 games에서 DQN과 Double DQN을 비교한다. 축소판은 논문의 noisy "
        "action-value 현상과 Double target을 로컬 MDP에서 재현한다."
    ),
    mappings=(
        (
            "§2, Eq. (3)",
            "DQN max target",
            "같은 estimator가 선택과 평가를 모두 하는 target을 비교 기준으로 둔다.",
        ),
        (
            "§2, Eq. (4); §3 Double DQN displayed target",
            "decoupled selection/evaluation",
            "argmax online, evaluate target 순서를 구현한다.",
        ),
        (
            "§2, Theorem 1 and Figure 1",
            "overestimation experiment",
            "zero-valued noisy actions에서 max estimator의 positive bias를 측정한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Double DQN target은 $a^*=\arg\max_a Q_{online}(s',a)$로 고르고
        $Q_{target}(s',a^*)$로 평가합니다. 두 noise estimator를 분리해 max bias도 직접 측정합니다."""),
        _setup(421),
        code(
            """
            # TODO: online_next로 action을 고르고 target_next로 평가하는 Double DQN target을 구현하세요.
            def double_target(r, done, online_next, target_next, discount):
                raise NotImplementedError


            y = double_target(
                torch.tensor([1.0]),
                torch.tensor([0.0]),
                torch.tensor([[1.0, 4.0]]),
                torch.tensor([[8.0, 2.0]]),
                0.5,
            )
            assert torch.allclose(y, torch.tensor([2.0]))
            """,
            """
            def double_target(r, done, online_next, target_next, discount):
                chosen = online_next.argmax(1, keepdim=True)
                evaluated = target_next.gather(1, chosen).squeeze(1)
                return (r + discount * (1 - done) * evaluated).detach()


            y = double_target(
                torch.tensor([1.0]),
                torch.tensor([0.0]),
                torch.tensor([[1.0, 4.0]]),
                torch.tensor([[8.0, 2.0]]),
                0.5,
            )
            assert torch.allclose(y, torch.tensor([2.0]))
            """,
            "implementation",
        ),
        code(
            """
            # TODO: true value=0인 8 actions에 독립 Gaussian noise를 더해 DQN/Double bias를 5000회
            # 추정하세요.
            raise NotImplementedError
            """,
            """
            rng = np.random.default_rng(422)
            online_noise = rng.normal(size=(5000, 8))
            target_noise = rng.normal(size=(5000, 8))
            dqn_est = online_noise.max(1)
            chosen = online_noise.argmax(1)
            double_est = target_noise[np.arange(len(chosen)), chosen]
            dqn_bias = float(dqn_est.mean())
            double_bias = float(double_est.mean())
            print(f"DQN bias={dqn_bias:.3f}, Double bias={double_bias:.3f}")
            assert dqn_bias > 0.8 and abs(double_bias) < 0.08
            """,
            "bias-experiment",
        ),
        code(
            """
            # TODO: 로컬 replay로 Double DQN 220 updates를 수행하고 bias histogram/loss를 그리세요.
            raise NotImplementedError
            """,
            """
            online = nn.Sequential(
                nn.Linear(n_states, 24), nn.ReLU(), nn.Linear(24, n_actions)
            ).to(DEVICE)
            target = nn.Sequential(
                nn.Linear(n_states, 24), nn.ReLU(), nn.Linear(24, n_actions)
            ).to(DEVICE)
            target.load_state_dict(online.state_dict())
            opt = torch.optim.Adam(online.parameters(), lr=0.012)
            gen = torch.Generator().manual_seed(423)
            history = []
            for step in range(220):
                idx = torch.randint(len(rows), (min(32, len(rows)),), generator=gen)
                s, a, r, ns, d = ACCELERATOR.move(
                    states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]
                )
                pred = online(encode_device(s))[torch.arange(len(idx), device=DEVICE), a]
                with torch.no_grad():
                    y = double_target(
                        r, d, online(encode_device(ns)), target(encode_device(ns)), gamma
                    )
                loss = F.mse_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                history.append(float(loss.detach()))
                if (step + 1) % 20 == 0:
                    target.load_state_dict(online.state_dict())
            assert np.isfinite(history).all()
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].hist(dqn_est, bins=35, alpha=0.6, label="max same")
            axes[0].hist(double_est, bins=35, alpha=0.6, label="double")
            axes[0].legend()
            axes[1].plot(history)
            axes[1].set_title("Double DQN TD loss")
            fig.tight_layout()
            plt.show()
            """,
            "training-evaluation",
        ),
        paired_markdown(
            "online과 target estimator가 완전히 같은 값을 내면 Double의 이득은?",
            (
                "선택과 평가 noise가 완전히 상관되면 다시 max estimator와 같아져 overestimation "
                "완화가 사라집니다. 지연 target이 두 estimator를 부분적으로 분리합니다."
            ),
            "reflection",
        ),
    ),
)


DUELING_DQN = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=3,
    slug="dueling_dqn",
    short_title="Dueling DQN",
    paper_title="Dueling Network Architectures for Deep Reinforcement Learning",
    authors="Ziyu Wang et al.",
    year=2016,
    primary_url="https://proceedings.mlr.press/v48/wangf16.html",
    venue="ICML",
    difficulty="중급",
    expected_minutes=50,
    dataset_file=DATASET,
    prerequisites="Q-learning, value/advantage decomposition",
    reproduction_goal="V와 A stream을 mean-centered aggregation으로 결합해 로컬 optimal Q table을 근사한다.",
    original_scale=(
        "논문은 Atari DQN backbone을 두 stream으로 분리한다. 축소판은 "
        "architecture/identifiability를 tabular-size MLP로 격리한다."
    ),
    mappings=(
        (
            "§3, Eq. (7)",
            "naive V+A decomposition",
            "식별 불가능한 상수 이동 문제를 관찰한다.",
        ),
        (
            "§3, Eq. (8)–(9)",
            "max/mean-centered aggregation",
            "실험에 쓰인 Q=V+A-mean(A)를 구현한다.",
        ),
        (
            "§4.1, corridor policy-evaluation experiment",
            "action-redundancy interpretation",
            "상태가치와 action-specific advantage를 따로 시각화한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (9) $Q(s,a)=V(s)+A(s,a)-\frac1{|A|}\sum_{a'}A(s,a')$를 구현합니다.
        평균 advantage가 0이므로 V가 Q의 action 평균으로 식별됩니다."""),
        _setup(431),
        code(
            """
            # TODO: shared feature 뒤 scalar V와 n_actions A stream을 가진 DuelingQ를 구현하세요.
            raise NotImplementedError
            """,
            """
            class DuelingQ(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.feature = nn.Sequential(nn.Linear(n_states, 32), nn.ReLU())
                    self.value = nn.Linear(32, 1)
                    self.advantage = nn.Linear(32, n_actions)

                def streams(self, x):
                    h = self.feature(x)
                    return self.value(h), self.advantage(h)

                def forward(self, x):
                    v, a = self.streams(x)
                    return v + a - a.mean(1, keepdim=True)


            model = DuelingQ().to(DEVICE)
            assert model(encode_device(torch.arange(n_states))).shape == (
                n_states,
                n_actions,
            )
            """,
            "implementation",
        ),
        code(
            """
            # TODO: value iteration으로 준비된 q_star를 220회 supervised-fit 하세요.
            raise NotImplementedError
            """,
            """
            opt = torch.optim.Adam(model.parameters(), lr=0.015)
            all_states = encode_device(torch.arange(n_states))
            q_ref = q_star.to(DEVICE)
            history = []
            with torch.no_grad():
                initial = float(F.mse_loss(model(all_states), q_ref))
            for _ in range(220):
                loss = F.mse_loss(model(all_states), q_ref)
                opt.zero_grad()
                loss.backward()
                opt.step()
                history.append(float(loss.detach()))
            assert history[-1] < 0.05 * initial
            """,
            "training",
        ),
        code(
            """
            # TODO: centered advantage 평균=0, Q MSE를 검증하고 V/A/Q를 그리세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                v, a = model.streams(all_states)
                centered = a - a.mean(1, keepdim=True)
                learned = v + centered
                mse = float(F.mse_loss(learned, q_ref))
            assert (
                torch.allclose(
                    centered.mean(1), torch.zeros_like(centered.mean(1)), atol=1e-6
                )
                and mse < 0.02
            )
            print(f"Q-table MSE={mse:.6f}")
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].plot(history)
            axes[0].set_title("fit loss")
            axes[1].plot(v.detach().cpu()[:, 0])
            axes[1].set_title("V(s)")
            axes[2].imshow(centered.detach().cpu().T, aspect="auto", cmap="coolwarm")
            axes[2].set_title("centered A(s,a)")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "mean subtraction이 식별 문제를 어떻게 해결하나요?",
            "V에 c를 더하고 모든 A에서 c를 빼도 Q가 같다는 자유도를 ΣA=0 제약으로 제거합니다. 따라서 V는 action 평균 Q가 됩니다.",
            "reflection",
        ),
    ),
)


PER = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=4,
    slug="prioritized_replay",
    short_title="Prioritized Replay",
    paper_title="Prioritized Experience Replay",
    authors="Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver",
    year=2016,
    primary_url="https://arxiv.org/abs/1511.05952",
    venue="ICLR",
    difficulty="중급",
    expected_minutes=55,
    dataset_file=DATASET,
    prerequisites="DQN, TD error, importance sampling",
    reproduction_goal=(
        "TD-error proportional sampling과 annealed importance weights로 "
        "로컬 transition Q-table을 학습한다."
    ),
    original_scale=(
        "논문은 Atari replay trees와 Double DQN을 사용한다. 축소판은 모든 "
        "transition을 메모리에 두고 정확한 categorical probabilities를 계산한다."
    ),
    mappings=(
        (
            "§3.2",
            "TD-error priority",
            "큰 surprise를 가진 transition을 더 자주 replay한다.",
        ),
        (
            "§3.3, Eq. (1)",
            "proportional sampling",
            "P(i)=p_i^alpha/sum p^alpha와 p=|delta|+epsilon을 구현한다.",
        ),
        (
            "§3.4 and Algorithm 1",
            "importance weights and prioritized Double DQN",
            "w_i=(N P(i))^-beta를 정규화해 weighted TD loss를 만든다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (1)로 TD error가 큰 전이를 자주 뽑되, §3.4의 importance weight로 바뀐 sampling
        distribution의 편향을 완화합니다."""),
        _setup(441),
        code(
            """
            # TODO: priorities에서 proportional probabilities와 max-normalized IS weights를
            # 계산하세요.
            def per_probabilities(priority, alpha=0.7):
                raise NotImplementedError


            def is_weights(probability, beta=0.5):
                raise NotImplementedError


            p = per_probabilities(torch.tensor([1.0, 2.0, 4.0]))
            w = is_weights(p)
            assert torch.allclose(p.sum(), torch.tensor(1.0)) and torch.allclose(
                w.max(), torch.tensor(1.0)
            )
            """,
            """
            def per_probabilities(priority, alpha=0.7):
                scaled = priority.clamp_min(1e-6).pow(alpha)
                return scaled / scaled.sum()


            def is_weights(probability, beta=0.5):
                raw = (len(probability) * probability).pow(-beta)
                return raw / raw.max()


            p = per_probabilities(torch.tensor([1.0, 2.0, 4.0]))
            w = is_weights(p)
            assert torch.allclose(p.sum(), torch.tensor(1.0)) and torch.allclose(
                w.max(), torch.tensor(1.0)
            )
            """,
            "implementation",
        ),
        code(
            """
            # TODO: learnable Q table, priority update, IS-weighted TD loss로 300 replay
            # updates를 수행하세요.
            raise NotImplementedError
            """,
            """
            # This tabular update is coupled to NumPy sampling, so it intentionally
            # stays on CPU; neural replay variants move complete batches instead.
            q = nn.Parameter(torch.zeros(n_states, n_actions))
            opt = torch.optim.Adam([q], lr=0.05)
            priority = torch.ones(len(rows))
            rng = np.random.default_rng(442)
            history = []
            counts = np.zeros(len(rows), dtype=int)
            for step in range(300):
                prob = per_probabilities(priority).detach().cpu().numpy()
                idx_np = rng.choice(
                    len(rows), size=min(16, len(rows)), replace=True, p=prob
                )
                counts += np.bincount(idx_np, minlength=len(rows))
                idx = torch.tensor(idx_np)
                pred = q[states[idx], actions[idx]]
                target = (
                    rewards[idx]
                    + gamma * (1 - dones[idx]) * q.detach()[next_states[idx]].max(1).values
                )
                td = target - pred
                weights = is_weights(
                    per_probabilities(priority)[idx], beta=0.4 + 0.6 * step / 299
                )
                loss = (weights * td.square()).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                priority[idx] = td.detach().abs() + 1e-3
                history.append(float(loss.detach()))
            assert np.isfinite(history).all() and priority.min() > 0
            """,
            "training",
        ),
        code(
            """
            # TODO: 고정 priority를 5000회 sample해 empirical frequency와 Eq. (1)의 상관을 검증하고 학습
            # metric을 그리세요.
            raise NotImplementedError
            """,
            """
            probe_priority = torch.arange(1, 7, dtype=torch.float32)
            expected = per_probabilities(probe_priority).cpu().numpy()
            draws = np.random.default_rng(443).choice(6, size=5000, p=expected)
            empirical = np.bincount(draws, minlength=6) / 5000
            corr = float(np.corrcoef(expected, empirical)[0, 1])
            q_mse = float(F.mse_loss(q.detach(), q_star))
            print(f"sampling corr={corr:.3f}, Q MSE={q_mse:.4f}")
            assert corr > 0.95 and math.isfinite(q_mse)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].bar(np.arange(6) - 0.18, expected, 0.36, label="Eq.(1)")
            axes[0].bar(np.arange(6) + 0.18, empirical, 0.36, label="empirical")
            axes[0].legend()
            axes[1].plot(history)
            axes[1].set_title("IS-weighted TD loss")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "alpha=0과 beta=1은 각각 무엇을 뜻하나요?",
            (
                "alpha=0은 uniform replay입니다. beta=1은 sampling 확률 변화에 대한 완전한 "
                "importance correction이며 논문은 학습 말기에 beta를 1로 anneal합니다."
            ),
            "reflection",
        ),
    ),
)


A3C = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=5,
    slug="a3c",
    short_title="A3C",
    paper_title="Asynchronous Methods for Deep Reinforcement Learning",
    authors="Volodymyr Mnih et al.",
    year=2016,
    primary_url="https://proceedings.mlr.press/v48/mniha16.html",
    venue="ICML",
    difficulty="고급",
    expected_minutes=65,
    dataset_file=DATASET,
    prerequisites="actor-critic, n-step return, entropy regularization",
    reproduction_goal=(
        "공유 actor-critic에 여러 seed worker rollout의 n-step advantage를 "
        "순차 적용해 A3C update를 재현한다."
    ),
    original_scale=(
        "논문은 다중 CPU thread와 Atari/Labyrinth 환경을 사용한다. 결정적 검증을 위해 "
        "worker를 순차 실행하지만 shared parameters, n-step return, entropy "
        "objective는 동일하다."
    ),
    mappings=(
        (
            "§3, n-step Q-learning paragraph",
            "bootstrapped n-step return",
            "k rewards와 terminal이 아닐 때 V(s_{t+k})를 합친다.",
        ),
        (
            "§4, ‘Asynchronous advantage actor-critic’",
            "advantage actor and value losses",
            "R_t-V(s_t)로 policy/value를 동시에 갱신한다.",
        ),
        (
            "§4 entropy equation; Supplementary Algorithm S2",
            "entropy-regularized shared-worker loop",
            "여러 actor-learners의 탐색과 shared parameter update를 축소 재현한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        §4의 advantage $R_t-V(s_t)$와 entropy bonus를 사용합니다. 실제 thread 경쟁은
        비결정적이므로 네 개 seed worker가 같은 모델에 차례로 gradient를 적용합니다."""),
        _setup(451),
        code(
            """
            # TODO: rewards와 마지막 bootstrap value에서 n-step returns를 역방향 계산하세요.
            def nstep_returns(reward_list, bootstrap, discount):
                raise NotImplementedError


            out = nstep_returns([1.0, 2.0], torch.tensor(3.0), 0.5)
            assert torch.allclose(out, torch.tensor([2.75, 3.5]))
            """,
            """
            def nstep_returns(reward_list, bootstrap, discount):
                values = []
                running = bootstrap
                for reward in reversed(reward_list):
                    running = (
                        torch.as_tensor(reward, device=bootstrap.device)
                        + discount * running
                    )
                    values.append(running)
                return torch.stack(list(reversed(values)))


            out = nstep_returns([1.0, 2.0], torch.tensor(3.0), 0.5)
            assert torch.allclose(out, torch.tensor([2.75, 3.5]))
            """,
            "implementation",
        ),
        code(
            """
            # TODO: shared actor-critic를 만들고 4 sequential workers × 100 rounds의 entropy A3C
            # updates를 수행하세요.
            raise NotImplementedError
            """,
            """
            class ActorCritic(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.body = nn.Sequential(nn.Linear(n_states, 32), nn.ReLU())
                    self.actor = nn.Linear(32, n_actions)
                    self.critic = nn.Linear(32, 1)

                def forward(self, s):
                    h = self.body(encode_device(s))
                    return self.actor(h), self.critic(h).squeeze(-1)


            model = ActorCritic().to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=0.008)
            history = []
            for round_id in range(100):
                for worker in range(4):
                    gen = torch.Generator().manual_seed(452 + 1000 * round_id + worker)
                    state = start_state
                    logps = []
                    values = []
                    entropies = []
                    rollout_rewards = []
                    done = False
                    for _ in range(2 * n_states + 2):
                        logits, value = model(torch.tensor([state]))
                        probs = logits.softmax(-1)[0]
                        action = int(
                            torch.multinomial(probs.detach().cpu(), 1, generator=gen)
                        )
                        ns, r, done = env_step(state, action)
                        logps.append(torch.log(probs[action].clamp_min(1e-8)))
                        entropies.append(-(probs * torch.log(probs.clamp_min(1e-8))).sum())
                        values.append(value[0])
                        rollout_rewards.append(r)
                        state = ns
                        if done:
                            break
                    bootstrap = (
                        torch.tensor(0.0, device=DEVICE)
                        if done
                        else model(torch.tensor([state]))[1][0].detach()
                    )
                    returns = nstep_returns(rollout_rewards, bootstrap, gamma)
                    value_tensor = torch.stack(values)
                    advantage = returns - value_tensor
                    loss = (
                        -(torch.stack(logps) * advantage.detach()).mean()
                        + 0.5 * advantage.square().mean()
                        - 0.01 * torch.stack(entropies).mean()
                    )
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    history.append(float(loss.detach()))
            assert np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: greedy policy의 optimal-action agreement와 rollout return을 평가하고
            # loss/policy를 그리세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                logits, values = model(torch.arange(n_states))
                policy = logits.softmax(1)
                nonterminal = torch.arange(n_states, device=DEVICE) != terminal_state
                agreement = float(
                    (
                        policy.argmax(1)[nonterminal]
                        == q_star.to(DEVICE).argmax(1)[nonterminal]
                    )
                    .float()
                    .mean()
                )
            state = start_state
            total = 0.0
            seen = set()
            for _ in range(2 * n_states + 2):
                action = int(policy[state].argmax())
                ns, r, done = env_step(state, action)
                total += r
                state = ns
                if done or (state, action) in seen:
                    break
                seen.add((state, action))
            print(f"greedy return={total:.3f}, optimal-action agreement={agreement:.3f}")
            assert math.isfinite(total) and agreement >= 0.5
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(history)
            axes[0].set_title("shared-worker loss")
            axes[1].imshow(policy.detach().cpu().T, aspect="auto", vmin=0, vmax=1)
            axes[1].set_title("actor policy")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "순차 worker 축소판과 실제 asynchronous A3C의 차이는?",
            (
                "실제 A3C는 서로 다른 환경 상태와 stale parameter에서 계산한 gradient가 비동기적으로 "
                "공유 파라미터에 도착합니다. 여기서는 그 decorrelation 효과 대신 서로 다른 seed "
                "rollout과 shared update 식만 보존합니다."
            ),
            "reflection",
        ),
    ),
)


DDPG = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=6,
    slug="ddpg",
    short_title="DDPG",
    paper_title="Continuous Control with Deep Reinforcement Learning",
    authors="Timothy P. Lillicrap et al.",
    year=2016,
    primary_url="https://arxiv.org/abs/1509.02971",
    venue="ICLR",
    difficulty="고급",
    expected_minutes=65,
    dataset_file=DATASET,
    prerequisites="actor-critic, deterministic policy gradient, target networks",
    reproduction_goal=(
        "로컬 상태에서 연속 action proxy를 만들고 deterministic actor, critic, "
        "soft target update를 학습한다."
    ),
    original_scale=(
        "논문은 MuJoCo 연속제어와 replay/OU noise를 사용한다. 축소판은 "
        "offline_action_noise로 만든 작은 continuous replay에서 Algorithm 1을 "
        "보존한다."
    ),
    mappings=(
        (
            "§2, Eq. (3)",
            "deterministic Bellman equation",
            "Q(s,a)=E[r+γQ(s',mu(s'))] target을 구성한다.",
        ),
        (
            "§3, Eq. (4)–(6)",
            "critic TD loss and deterministic policy gradient",
            "critic을 회귀하고 actor는 Q를 action 방향으로 최대화한다.",
        ),
        (
            "§3, Algorithm 1",
            "replay, exploration noise, soft target update",
            "prepared noise와 tau interpolation을 구현한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (6)의 deterministic policy gradient는 actor action을 critic에 넣어 $Q(s,\mu(s))$를
        최대화합니다. `offline_action_noise`로 만든 로컬 continuous replay와 soft target을 사용합니다."""),
        _setup(461),
        code(
            """
            # TODO: tanh actor, state-action critic, tau soft_update를 구현하세요.
            raise NotImplementedError
            """,
            """
            class Actor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh()
                    )

                def forward(self, s):
                    return self.net(s)


            class Critic(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(2, 48), nn.ReLU(), nn.Linear(48, 1))

                def forward(self, s, a):
                    return self.net(torch.cat([s, a], 1))


            def soft_update(target, source, tau):
                with torch.no_grad():
                    for tp, sp in zip(target.parameters(), source.parameters()):
                        tp.mul_(1 - tau).add_(sp, alpha=tau)


            actor, critic, actor_t, critic_t = ACCELERATOR.move(
                Actor(), Critic(), Actor(), Critic()
            )
            actor_t.load_state_dict(actor.state_dict())
            critic_t.load_state_dict(critic.state_dict())
            """,
            "implementation",
        ),
        code(
            """
            # TODO: q_star greedy action을 [-1,1] target으로 바꾼 noisy replay에서 Algorithm 1을
            # 320회 수행하세요.
            raise NotImplementedError
            """,
            """
            state_grid = (
                torch.arange(n_states).float()[:, None] / max(1, n_states - 1) * 2 - 1
            )
            best = q_star.argmax(1).float()
            target_action = best[:, None] / max(1, n_actions - 1) * 2 - 1
            repeat = 40
            replay_s = state_grid.repeat_interleave(repeat, 0)
            replay_target = target_action.repeat_interleave(repeat, 0)
            local_noise = torch.tensor(config["offline_action_noise"], dtype=torch.float32)
            local_noise = local_noise.repeat(
                math.ceil(len(replay_target) / len(local_noise))
            )[: len(replay_target), None]
            action_sweep = torch.linspace(-1, 1, repeat).repeat(n_states)[:, None]
            replay_a = (action_sweep + local_noise).clamp(-1, 1)
            replay_r = -(replay_a - replay_target).square()
            replay_ns = replay_s.clone()
            replay_done = torch.ones_like(replay_r)
            (
                state_grid,
                target_action,
                replay_s,
                replay_a,
                replay_r,
                replay_ns,
                replay_done,
            ) = ACCELERATOR.move(
                state_grid,
                target_action,
                replay_s,
                replay_a,
                replay_r,
                replay_ns,
                replay_done,
            )
            oc = torch.optim.Adam(critic.parameters(), lr=0.006)
            oa = torch.optim.Adam(actor.parameters(), lr=0.004)
            gen = torch.Generator().manual_seed(463)
            history = []
            for _ in range(320):
                idx = torch.randint(len(replay_s), (64,), generator=gen).to(DEVICE)
                s, a, r, ns, d = (
                    replay_s[idx],
                    replay_a[idx],
                    replay_r[idx],
                    replay_ns[idx],
                    replay_done[idx],
                )
                with torch.no_grad():
                    y = r + gamma * (1 - d) * critic_t(ns, actor_t(ns))
                lc = F.mse_loss(critic(s, a), y)
                oc.zero_grad()
                lc.backward()
                oc.step()
                la = -critic(s, actor(s)).mean()
                oa.zero_grad()
                la.backward()
                oa.step()
                soft_update(actor_t, actor, 0.02)
                soft_update(critic_t, critic, 0.02)
                history.append((float(lc.detach()), float(la.detach())))
            assert np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: actor action MSE와 soft-target distance를 계산하고 action curve/loss를 그리세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                learned_action = actor(state_grid)
                action_mse = float(F.mse_loss(learned_action, target_action))
                target_gap = sum(
                    float((p - q).square().sum())
                    for p, q in zip(actor.parameters(), actor_t.parameters())
                )
            print(f"actor target MSE={action_mse:.4f}, online-target gap={target_gap:.6f}")
            assert action_mse < 0.6 and math.isfinite(target_gap)
            h = np.asarray(history)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(target_action.detach().cpu(), label="optimal proxy")
            axes[0].plot(learned_action.detach().cpu(), label="actor")
            axes[0].legend()
            axes[1].plot(h[:, 0], label="critic")
            axes[1].plot(h[:, 1], label="actor")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "actor update에서 replay action 대신 actor(s)를 critic에 넣는 이유는?",
            (
                "deterministic policy gradient는 현재 policy가 선택하는 action에서 Q의 "
                "action gradient를 따라야 합니다. replay action은 critic 회귀에 쓰지만 "
                "actor의 결정 경로가 아닙니다."
            ),
            "reflection",
        ),
    ),
)


PPO = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=7,
    slug="ppo",
    short_title="PPO",
    paper_title="Proximal Policy Optimization Algorithms",
    authors="John Schulman et al.",
    year=2017,
    primary_url="https://arxiv.org/abs/1707.06347",
    venue="arXiv",
    difficulty="중급",
    expected_minutes=60,
    dataset_file=DATASET,
    prerequisites="policy ratio, advantage, actor-critic",
    reproduction_goal=(
        "고정 old log-prob batch에서 clipped surrogate를 여러 epoch 최적화해 로컬 "
        "optimal policy를 학습한다."
    ),
    original_scale=(
        "논문은 MuJoCo/Atari에서 parallel actors와 GAE를 사용한다. 축소판은 exact "
        "q_star advantage를 써 clipping mechanics를 격리한다."
    ),
    mappings=(
        (
            "§2, Eq. (1) and Eq. (6)",
            "policy-gradient/CPI surrogate",
            "old-policy importance ratio objective의 출발점을 계산한다.",
        ),
        (
            "§3, Eq. (7)",
            "clipped surrogate",
            "min(rA, clip(r)A)로 큰 policy step의 이득을 제한한다.",
        ),
        (
            "§5, Eq. (9), Algorithm 1",
            "combined actor/value/entropy multi-epoch update",
            "같은 rollout batch를 여러 epoch 재사용한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (7)의 $\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)$를
        구현합니다. old log-prob는 rollout 뒤 고정하고 같은 batch를 여러 epoch 사용합니다."""),
        _setup(471),
        code(
            """
            # TODO: new/old log-prob와 advantage에서 PPO clipped objective와 clip fraction을
            # 반환하세요.
            def ppo_objective(new_logp, old_logp, adv, eps=0.2):
                raise NotImplementedError


            obj, frac = ppo_objective(
                torch.log(torch.tensor([1.4, 0.8])),
                torch.zeros(2),
                torch.tensor([1.0, -1.0]),
            )
            assert obj.ndim == 0 and 0 <= frac <= 1
            """,
            """
            def ppo_objective(new_logp, old_logp, adv, eps=0.2):
                ratio = (new_logp - old_logp).exp()
                surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
                return surrogate.mean(), float(((ratio - 1).abs() > eps).float().mean())


            obj, frac = ppo_objective(
                torch.log(torch.tensor([1.4, 0.8])),
                torch.zeros(2),
                torch.tensor([1.0, -1.0]),
            )
            assert obj.ndim == 0 and 0 <= frac <= 1
            """,
            "implementation",
        ),
        code(
            """
            # TODO: actor/value table에서 batch를 old policy로 뽑고 batch당 4 epochs씩 45 PPO
            # iterations 학습하세요.
            raise NotImplementedError
            """,
            """
            # The policy/value tables contain only a few hundred scalars and every
            # iteration samples with a CPU generator, so this toy PPO fit is CPU-fast.
            policy_logits = nn.Parameter(torch.zeros(n_states, n_actions))
            values = nn.Parameter(torch.zeros(n_states))
            opt = torch.optim.Adam([policy_logits, values], lr=0.04)
            gen = torch.Generator().manual_seed(472)
            history = []
            batch_states = torch.arange(n_states).repeat_interleave(12)
            for _ in range(45):
                with torch.no_grad():
                    old_dist = policy_logits[batch_states].softmax(1)
                    batch_actions = torch.multinomial(old_dist, 1, generator=gen).squeeze(1)
                    old_logp = torch.log(
                        old_dist[torch.arange(len(batch_states)), batch_actions].clamp_min(
                            1e-8
                        )
                    )
                    returns = q_star[batch_states, batch_actions]
                    adv = returns - values[batch_states]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                for _ in range(4):
                    probs = policy_logits[batch_states].softmax(1)
                    new_logp = torch.log(
                        probs[torch.arange(len(batch_states)), batch_actions].clamp_min(
                            1e-8
                        )
                    )
                    objective, clipfrac = ppo_objective(new_logp, old_logp, adv)
                    value_loss = F.mse_loss(values[batch_states], returns)
                    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(1).mean()
                    loss = -objective + 0.5 * value_loss - 0.01 * entropy
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                history.append((float(loss.detach()), clipfrac))
            assert np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: greedy optimal-action agreement, policy entropy, clip fraction을 출력하고
            # 그리세요.
            raise NotImplementedError
            """,
            """
            probs = policy_logits.softmax(1).detach()
            nonterminal = torch.arange(n_states) != terminal_state
            agreement = float(
                (probs.argmax(1)[nonterminal] == q_star.argmax(1)[nonterminal])
                .float()
                .mean()
            )
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-8))).sum(1)).mean())
            h = np.asarray(history)
            print(
                f"agreement={agreement:.3f}, entropy={entropy:.3f}, "
                f"last clip fraction={h[-1, 1]:.3f}"
            )
            assert agreement >= 0.7 and math.isfinite(entropy)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].imshow(probs.T, aspect="auto", vmin=0, vmax=1)
            axes[0].set_title("PPO policy")
            axes[1].plot(h[:, 0], label="loss")
            axes[1].plot(h[:, 1], label="clip fraction")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "advantage가 음수일 때 clipping 방향이 달라 보이는 이유는?",
            (
                "음수 advantage action의 probability를 지나치게 낮추는 것도 surrogate 개선으로 "
                "과대평가될 수 있습니다. min 연산은 부호에 맞춰 양쪽의 과도한 ratio 변화를 제한합니다."
            ),
            "reflection",
        ),
    ),
)


SAC = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=8,
    slug="sac",
    short_title="SAC",
    paper_title=(
        "Soft Actor-Critic: Off-Policy Maximum Entropy Deep "
        "Reinforcement Learning with a Stochastic Actor"
    ),
    authors="Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine",
    year=2018,
    primary_url="https://proceedings.mlr.press/v80/haarnoja18b.html",
    venue="ICML",
    difficulty="고급",
    expected_minutes=70,
    dataset_file=DATASET,
    prerequisites="maximum-entropy RL, reparameterized Gaussian, actor-critic",
    reproduction_goal=(
        "squashed Gaussian actor와 soft actor objective를 로컬 "
        "continuous-action replay에서 학습한다."
    ),
    original_scale=(
        "논문은 MuJoCo continuous control과 twin-style off-policy "
        "updates를 평가한다. 축소판은 one-step continuous contextual control로 "
        "entropy/Q trade-off를 격리한다."
    ),
    mappings=(
        (
            "§3.2, Eq. (1)",
            "maximum-entropy objective",
            "reward와 expected policy entropy를 함께 최대화한다.",
        ),
        (
            "§4.1, Eq. (2)–(3), Eq. (7)–(8)",
            "soft Bellman backup and Q loss",
            "soft value가 들어가는 critic target 구조를 연결한다.",
        ),
        (
            "§4.2, Eq. (10)–(12), Algorithm 1",
            "KL policy update and reparameterized stochastic actor",
            "tanh Gaussian sample로 alpha log pi - Q를 최소화한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (11)–(12)의 reparameterized stochastic actor를 사용해
        $J_\pi=\mathbb E[\alpha\log\pi(a|s)-Q(s,a)]$를 최소화합니다."""),
        _setup(481),
        code(
            """
            # TODO: state에서 mean/log_std를 내고 tanh-reparameterized action과 corrected
            # log_prob를 반환하는 actor를 구현하세요.
            raise NotImplementedError
            """,
            """
            class GaussianActor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.body = nn.Sequential(nn.Linear(n_states, 32), nn.ReLU())
                    self.mean = nn.Linear(32, 1)
                    self.logstd = nn.Linear(32, 1)

                def sample(self, s):
                    h = self.body(encode_device(s))
                    mean = self.mean(h)
                    logstd = self.logstd(h).clamp(-4, 1)
                    std = logstd.exp()
                    normal = torch.distributions.Normal(mean, std)
                    u = normal.rsample()
                    a = torch.tanh(u)
                    logp = normal.log_prob(u) - torch.log(1 - a.square() + 1e-6)
                    return a, logp.sum(1), torch.tanh(mean)


            actor = GaussianActor().to(DEVICE)
            a, lp, m = actor.sample(torch.arange(n_states))
            assert (
                a.shape == (n_states, 1)
                and lp.shape == (n_states,)
                and float(a.detach().abs().max()) <= 1
            )
            """,
            "implementation",
        ),
        code(
            """
            # TODO: uniform-action local replay에서 critic MSE와 alpha*logpi-Q actor loss를 360회
            # 교대 최적화하세요.
            raise NotImplementedError
            """,
            """
            critic = nn.Sequential(
                nn.Linear(n_states + 1, 48), nn.ReLU(), nn.Linear(48, 1)
            ).to(DEVICE)
            oq = torch.optim.Adam(critic.parameters(), lr=0.006)
            op = torch.optim.Adam(actor.parameters(), lr=0.004)
            alpha = 0.08
            rng = np.random.default_rng(482)
            replay_s = torch.arange(n_states).repeat_interleave(80)
            replay_a = torch.tensor(
                rng.uniform(-1, 1, size=(len(replay_s), 1)), dtype=torch.float32
            )
            target = (q_star.argmax(1).float() / max(1, n_actions - 1) * 2 - 1)[:, None]
            replay_target = target[replay_s]
            replay_r = -(replay_a - replay_target).square()
            replay_s, replay_a, replay_r, target = ACCELERATOR.move(
                replay_s, replay_a, replay_r, target
            )
            gen = torch.Generator().manual_seed(483)
            history = []
            for _ in range(360):
                idx = torch.randint(len(replay_s), (64,), generator=gen).to(DEVICE)
                s, a, r = replay_s[idx], replay_a[idx], replay_r[idx]
                pred = critic(torch.cat([encode_device(s), a], 1))
                lq = F.mse_loss(pred, r)
                oq.zero_grad()
                lq.backward()
                oq.step()
                sample_a, logp, _ = actor.sample(s)
                lp = (
                    alpha * logp
                    - critic(torch.cat([encode_device(s), sample_a], 1)).squeeze(1)
                ).mean()
                op.zero_grad()
                lp.backward()
                op.step()
                history.append((float(lq.detach()), float(lp.detach())))
            assert np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: deterministic mean action MSE와 stochastic action std를 평가하고 soft policy를
            # 그리세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                samples = []
            for _ in range(60):
                samples.append(actor.sample(torch.arange(n_states))[0].detach())
            samples = torch.stack(samples)
            mean_action = actor.sample(torch.arange(n_states))[2].detach()
            mse = float(F.mse_loss(mean_action, target))
            diversity = float(samples.std(0).mean())
            print(f"mean-action MSE={mse:.4f}, stochastic std={diversity:.3f}")
            assert mse < 0.45 and diversity > 0
            h = np.asarray(history)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(target.detach().cpu(), label="optimal proxy")
            axes[0].plot(mean_action.detach().cpu(), label="SAC mean")
            axes[0].legend()
            axes[1].plot(h[:, 0], label="Q")
            axes[1].plot(h[:, 1], label="policy")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "SAC가 deterministic 최적 action에 정확히 collapse하지 않을 수 있는 이유는?",
            (
                "maximum-entropy objective는 reward뿐 아니라 entropy도 보상합니다. "
                "alpha>0이면 약간 낮은 Q action에도 확률을 남기는 것이 전체 soft objective에 유리할 "
                "수 있습니다."
            ),
            "reflection",
        ),
    ),
)


ALPHAZERO = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=9,
    slug="alphazero",
    short_title="AlphaZero / MCTS",
    paper_title=(
        "Mastering Chess and Shogi by Self-Play with a General "
        "Reinforcement Learning Algorithm"
    ),
    authors="David Silver et al.",
    year=2017,
    primary_url="https://arxiv.org/abs/1712.01815",
    venue="arXiv / Science",
    difficulty="고급",
    expected_minutes=75,
    dataset_file=DATASET,
    prerequisites="MCTS, policy/value network, self-play targets",
    reproduction_goal="로컬 deterministic MDP에서 PUCT visit policy와 joint policy/value loss를 구현한다.",
    original_scale=(
        "논문은 chess/shogi self-play와 대규모 residual network를 사용한다. 축소판은 "
        "exact model과 value-iteration leaf oracle로 MCTS/visit "
        "target/loss를 분리한다."
    ),
    mappings=(
        (
            "Main text, network paragraph and Eq. (1)",
            "(p,v)=f_theta(s) and joint loss",
            "visit policy cross-entropy, outcome value MSE, L2를 결합한다.",
        ),
        (
            "Main text, MCTS paragraph",
            "PUCT search and root visit counts",
            "prior P, mean Q, visit N으로 action을 고르고 pi를 만든다.",
        ),
        (
            "Main text, self-play paragraph",
            "search-improved policy targets",
            "a_t~pi_t와 terminal z의 축소판으로 모든 state search targets를 만든다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        네트워크 $(p,v)=f_\theta(s)$가 prior와 leaf value를 내고 MCTS root visit counts가
        개선된 policy $\pi$를 만듭니다. 마지막에는 Eq. (1)의 policy/value/L2 loss로 network를 학습합니다."""),
        _setup(491),
        code(
            """
            # TODO: policy/value network와 Agent를 만들고 select_action으로 root search를
            # 실행하세요. 반환된 visit count와 temperature policy를 검증하세요.
            raise NotImplementedError
            """,
            """
            class PVNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.body = nn.Sequential(nn.Linear(n_states, 32), nn.ReLU())
                    self.policy = nn.Linear(32, n_actions)
                    self.value = nn.Sequential(nn.Linear(32, 1), nn.Tanh())

                def forward(self, state):
                    hidden = self.body(encode_device(state))
                    return self.policy(hidden), self.value(hidden).squeeze(1)


            net = PVNet().to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), lr=0.015)
            portfolio_agent = AlphaZeroSearchAgent(
                net,
                opt,
                env_step,
                q_star,
                gamma,
                n_actions,
            )
            root_action, root_pi, root_visits, root_prior = (
                portfolio_agent.select_action(
                    start_state,
                    simulations=80,
                    temperature=1.0,
                )
            )
            assert root_visits.sum() == 80
            assert torch.allclose(root_pi.sum(), torch.tensor(1.0))
            assert int(root_pi.argmax()) == root_action
            assert (root_pi - root_prior).abs().sum() > 0.0
            """,
            "implementation",
        ),
        code(
            """
            # TODO: 모든 nonterminal state에서 Agent.select_action을 호출해 improved target_pi,
            # visit_counts, raw_priors를 구성하세요.
            raise NotImplementedError
            """,
            """
            train_states_cpu = torch.tensor(
                [state for state in range(n_states) if state != terminal_state]
            )
            search_results = [
                portfolio_agent.select_action(
                    int(state),
                    simulations=80,
                    temperature=1.0,
                )
                for state in train_states_cpu
            ]
            search_actions = torch.tensor([result[0] for result in search_results])
            target_pi_cpu = torch.stack([result[1] for result in search_results])
            visit_counts_cpu = torch.stack([result[2] for result in search_results])
            raw_priors_cpu = torch.stack([result[3] for result in search_results])
            target_z_cpu = q_star[train_states_cpu].max(1).values
            target_z_cpu = target_z_cpu / target_z_cpu.abs().max().clamp_min(1.0)
            improved_target_l1 = float(
                (target_pi_cpu - raw_priors_cpu).abs().mean()
            )
            assert torch.equal(visit_counts_cpu.sum(1), torch.full_like(
                visit_counts_cpu.sum(1), 80
            ))
            assert torch.equal(target_pi_cpu.argmax(1), search_actions)
            assert torch.allclose(
                target_pi_cpu.sum(1),
                torch.ones_like(target_pi_cpu.sum(1)),
            )
            assert improved_target_l1 > 0.0
            """,
            "search",
        ),
        code(
            """
            # TODO: 모든 nonterminal state의 search pi와 z=max Q*로 Eq. (1) network를 200회 학습하고
            # 시각화하세요.
            raise NotImplementedError
            """,
            """
            train_states, target_pi, target_z = ACCELERATOR.move(
                train_states_cpu,
                target_pi_cpu,
                target_z_cpu,
            )
            visit_counts, raw_priors = ACCELERATOR.move(
                visit_counts_cpu,
                raw_priors_cpu,
            )
            history = []
            with torch.no_grad():
                l0, v0 = net(train_states)
                initial = float(
                    -(target_pi * l0.log_softmax(1)).sum(1).mean()
                    + F.mse_loss(v0, target_z)
                )
            for _ in range(200):
                history.append(
                    portfolio_agent.update(train_states, target_pi, target_z)
                )
            """,
            "training-evaluation",
        ),
        paired_markdown(
            "왜 raw network policy p 대신 MCTS visit policy pi를 target으로 쓰나요?",
            (
                "search는 model rollouts와 value estimates를 여러 번 결합해 p보다 강한 정책을 "
                "만듭니다. pi를 다시 학습하면 network가 search의 개선을 amortize해 다음 search의 "
                "prior가 좋아집니다."
            ),
            "reflection",
        ),
    ),
)


_PORTFOLIO_GUIDES = {
    "reinforce": {
        "equation": (
            r"\nabla_\theta J(\theta)=\mathbb E[(G_t-b_t)"
            r"\nabla_\theta\log\pi_\theta(a_t|s_t)]"
        ),
        "symbols": (
            r"$G_t$는 return, $b_t$는 action과 무관한 baseline, "
            r"$\pi_\theta$는 softmax policy입니다."
        ),
        "shape": "bandit logits/probability [A], sampled action scalar, reward/advantage scalar",
        "task": "return-to-go, score-function loss, baseline update와 최적 arm 확률을 구현합니다.",
        "flow": "policy 확률 → action sample → reward-baseline → log-prob update → arm 평가",
    },
    "dqn": {
        "equation": (
            r"y_t=r_t+\gamma(1-d_t)\max_{a'}Q_{\theta^-}(s_{t+1},a'),\qquad "
            r"\mathcal L=\mathbb E[(y_t-Q_\theta(s_t,a_t))^2]"
        ),
        "symbols": (
            r"$Q_\theta$는 online network, $Q_{\theta^-}$는 frozen target, "
            r"$d_t$는 terminal mask입니다."
        ),
        "shape": "one-hot state [B, S] → Q [B, A]; action/reward/done [B]",
        "task": "replay batch, detached Bellman target, Huber loss와 target sync를 구현합니다.",
        "flow": "uniform replay → online gather → target max → TD update → 주기적 sync/평가",
    },
    "double_dqn": {
        "equation": (
            r"a^*=\arg\max_aQ_\theta(s',a),\qquad "
            r"y=r+\gamma(1-d)Q_{\theta^-}(s',a^*)"
        ),
        "symbols": (
            r"$Q_\theta$는 action을 선택하고 $Q_{\theta^-}$는 그 action을 평가해 "
            r"maximization bias를 분리합니다."
        ),
        "shape": "online_next/target_next [B, A] → chosen [B, 1] → target [B]",
        "task": "선택/평가 분리 target, noisy max bias 실험과 Double TD update를 구현합니다.",
        "flow": "online argmax → target gather → detached TD loss → sync → bias 비교",
    },
    "dueling_dqn": {
        "equation": (
            r"Q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal A|}"
            r"\sum_{a'}A(s,a')"
        ),
        "symbols": (
            r"$V(s)$는 상태가치 scalar, $A(s,a)$는 action advantage, "
            r"평균 제거는 두 stream의 식별 가능성을 만듭니다."
        ),
        "shape": "one-hot state [B, S] → V [B, 1], A/Q [B, A]",
        "task": "shared feature, value/advantage heads, centered aggregation과 Q 평가를 구현합니다.",
        "flow": "state encode → V/A streams → mean-center → Q loss → stream 해석",
    },
    "prioritized_replay": {
        "equation": (
            r"P(i)=\frac{p_i^\alpha}{\sum_kp_k^\alpha},\qquad "
            r"w_i=\left(NP(i)\right)^{-\beta}/\max_jw_j"
        ),
        "symbols": (
            r"$p_i=|\delta_i|+\epsilon$은 priority, $\alpha$는 prioritization, "
            r"$\beta$는 importance correction 강도입니다."
        ),
        "shape": "priority/probability/weight [N_replay], sampled indices [B], TD error [B]",
        "task": "비균등 sampling, IS-weighted TD loss와 priority 갱신을 구현합니다.",
        "flow": "priority → categorical sample → weighted TD update → |TD| priority → 빈도 평가",
    },
    "a3c": {
        "equation": (
            r"\mathcal L=-\log\pi(a_t|s_t)(R_t-V(s_t))"
            r"+c_v(R_t-V(s_t))^2-c_e\mathcal H(\pi(\cdot|s_t))"
        ),
        "symbols": (
            r"$R_t$는 n-step bootstrap return, $V$는 critic, $\mathcal H$는 "
            r"worker 탐색을 유지하는 entropy입니다."
        ),
        "shape": "state [T] → policy logits [T, A], value/return/advantage [T]",
        "task": "n-step return, shared actor-critic loss, worker rollout update를 구현합니다.",
        "flow": "worker rollout → bootstrap return → advantage → shared update → greedy 평가",
    },
    "ddpg": {
        "equation": (
            r"\mathcal L_Q=\mathbb E[(r+\gamma Q'(s',\mu'(s'))-Q(s,a))^2],\qquad "
            r"J_\mu=-\mathbb E[Q(s,\mu(s))]"
        ),
        "symbols": (
            r"$\mu$는 deterministic actor, $Q$는 critic, prime 표기는 soft target network입니다."
        ),
        "shape": "state/action [B, 1] → concat [B, 2] → Q [B, 1]",
        "task": "critic Bellman loss, actor Q maximization과 Polyak soft update를 구현합니다.",
        "flow": (
            "continuous replay → critic update → actor update → target "
            "interpolation → action 평가"
        ),
    },
    "ppo": {
        "equation": (
            r"L^{CLIP}=\mathbb E_t\left[\min(r_tA_t,"
            r"\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]"
        ),
        "symbols": (
            r"$r_t=\pi_\theta/\pi_{old}$는 probability ratio, $A_t$는 advantage, "
            r"$\epsilon$은 policy step 제한입니다."
        ),
        "shape": "state [B] → policy [B, A], value [B]; logp/advantage/return [B]",
        "task": "old log-prob 고정, clipped surrogate, value/entropy 결합 update를 구현합니다.",
        "flow": "old-policy rollout → advantage normalize → multi-epoch clip update → policy 평가",
    },
    "sac": {
        "equation": (
            r"J_\pi=\mathbb E_{s,\epsilon}[\alpha\log\pi_\theta(a|s)-Q_\phi(s,a)],"
            r"\qquad a=\tanh(\mu_\theta(s)+\sigma_\theta(s)\epsilon)"
        ),
        "symbols": (
            r"$\alpha$는 entropy temperature, $Q_\phi$는 soft critic, "
            r"tanh Jacobian은 bounded action의 log-prob를 보정합니다."
        ),
        "shape": "one-hot state [B, S] + action [B, 1] → Q [B, 1]; logp [B]",
        "task": "reparameterized action/log-prob, critic MSE와 alpha logpi-Q를 구현합니다.",
        "flow": "off-policy batch → critic update → rsample actor update → mean/diversity 평가",
    },
    "alphazero": {
        "equation": (
            r"\mathcal L=(z-v)^2-\pi^\top\log p+c\lVert\theta\rVert_2^2,\qquad "
            r"U(s,a)=c_{puct}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}"
        ),
        "symbols": (
            r"$p/v$는 network policy/value, $\pi$는 MCTS visit target, "
            r"$P,Q,N$은 tree prior/mean value/visit count입니다."
        ),
        "shape": "state [B] → policy logits [B, A], value [B]; root P/Q/N/pi [A]",
        "task": "PUCT selection, visit policy, policy-value-L2 loss와 search target을 구현합니다.",
        "flow": "network prior/value → MCTS simulations → visit pi → joint update → agreement 평가",
    },
}


_PORTFOLIO_CLASS_NAMES = {
    "reinforce": "ReinforceBanditAgent",
    "dqn": "DqnAgent",
    "double_dqn": "DoubleDqnAgent",
    "dueling_dqn": "DuelingValueAgent",
    "prioritized_replay": "PrioritizedReplayAgent",
    "a3c": "SharedActorCriticAgent",
    "ddpg": "DeterministicPolicyAgent",
    "ppo": "ProximalPolicyAgent",
    "sac": "SoftActorCriticAgent",
    "alphazero": "AlphaZeroSearchAgent",
}


def _portfolio_markdown(spec: FieldPaperSpec) -> object:
    guide = _PORTFOLIO_GUIDES[spec.slug]
    return markdown(
        rf"""
        ## 포트폴리오 구현 설계: 수식 → 에이전트 책임 → 검증

        논문의 핵심 update를 실행 가능한 에이전트의 `forward`, `loss`, `update`,
        `evaluate`로 나눕니다. 아래 식의 선택·평가·gradient 경계를 메서드 본문에서
        한 줄씩 추적하세요.

        $$
        {guide["equation"]}
        $$

        - **기호와 역할:** {guide["symbols"]}
        - **shape/state 계약:** `{guide["shape"]}`
        - **논문 위치:** `{spec.mappings[0][0]}`의 **{spec.mappings[0][1]}**
        - **코드 Task:** {guide["task"]}
        - **학습·평가 흐름:** {guide["flow"]}
        - **원문 대비 한계:** {spec.original_scale}
        - **구현 이유:** target, gradient, optimizer 책임을 Agent 메서드별로
          분리해 코드와 논문을 한 줄씩 대조할 수 있게 합니다.
        - **완료 증거:** 마지막 셀에서 `update`, parameter 변화, 평가 metric을
          모두 `assert`합니다.

        정답 클래스는 Bellman target, policy objective, search score 같은 핵심 계산을
        불투명한 trainer 뒤에 감추지 않습니다. 실습본에서는 공개 API를 유지하면서 TODO를
        채우고, 각 메서드의 입력 shape와 gradient가 끊기는 위치를 설명하세요.
        """,
        "portfolio-explanation",
        "equation",
    )


def _method_contract(spec: FieldPaperSpec, method_name: str) -> str:
    shape = _PORTFOLIO_GUIDES[spec.slug]["shape"]
    contracts = {
        "__init__": "network, optimizer, target 또는 table 상태를 명시적으로 저장한다.",
        "config": "재현 범위와 핵심 하이퍼파라미터를 dict로 반환한다.",
        "forward": f"policy/value/Q 순전파를 계산한다. shape 계약: {shape}.",
        "loss": "논문의 목적 함수를 미분 가능한 scalar tensor로 반환한다.",
        "update": "zero_grad, backward, step을 수행하고 유한한 float loss를 반환한다.",
        "evaluate": "no_grad 경로에서 정책 또는 value 품질 지표 dict를 반환한다.",
        "sync_target": "online network의 가중치를 target network로 복사한다.",
        "sampling_probabilities": "priority를 정규화한 replay sampling 확률을 반환한다.",
        "_soft_update": "Polyak 평균으로 target parameter를 제자리 갱신한다.",
        "select_action": (
            "PUCT simulation을 수행하고 temperature visit policy, count, prior와 "
            "행동을 반환한다."
        ),
    }
    return contracts.get(method_name, "논문 알고리즘의 공개 보조 연산을 수행한다.")


def _with_contract_docstrings(spec: FieldPaperSpec, source: str) -> ast.Module:
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    class_doc = fill(
        f"{spec.short_title}의 핵심 학습·평가 경로. "
        f"입출력 계약: {_PORTFOLIO_GUIDES[spec.slug]['shape']}.",
        width=76,
    )
    class_node.body.insert(0, ast.Expr(value=ast.Constant(class_doc)))
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        node.body.insert(
            0,
            ast.Expr(
                value=ast.Constant(fill(_method_contract(spec, node.name), width=76))
            ),
        )
    ast.fix_missing_locations(tree)
    return tree


def _exercise_architecture(spec: FieldPaperSpec, solution: str) -> str:
    tree = deepcopy(_with_contract_docstrings(spec, solution))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = node.body[0]
        todo = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                args=[ast.Constant(f"TODO: {node.name} 본문을 구현하세요.")],
                keywords=[],
            )
        )
        node.body = [docstring, todo]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _stub_function(
    spec: FieldPaperSpec,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    stub = deepcopy(node)
    contract = fill(
        f"정답과 동일한 공개 signature를 유지한다. "
        f"shape 계약: {_PORTFOLIO_GUIDES[spec.slug]['shape']}.",
        width=76,
    )
    stub.body = [
        ast.Expr(value=ast.Constant(contract)),
        ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                args=[ast.Constant(f"TODO: {node.name} 본문을 구현하세요.")],
                keywords=[],
            )
        ),
    ]
    return stub


def _exercise_with_missing_public_api(spec: FieldPaperSpec, cell: object) -> object:
    """Prepend exact-signature TODO stubs omitted by an older exercise cell."""

    exercise_tree = ast.parse(cell.exercise)
    solution_tree = ast.parse(cell.solution)
    exercise_functions = {
        node.name
        for node in exercise_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    exercise_classes = {
        node.name for node in exercise_tree.body if isinstance(node, ast.ClassDef)
    }
    stubs: list[ast.stmt] = []
    for node in solution_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or node.name in exercise_functions:
                continue
            stubs.append(_stub_function(spec, node))
        elif isinstance(node, ast.ClassDef) and node.name not in exercise_classes:
            class_stub = deepcopy(node)
            class_stub.body = [
                ast.Expr(
                    value=ast.Constant(
                        fill(
                            f"{node.name} 연습 skeleton. 공개 method의 signature와 "
                            "반환 계약은 정답과 같다.",
                            width=76,
                        )
                    )
                )
            ]
            class_stub.body.extend(
                _stub_function(spec, method)
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (not method.name.startswith("_") or method.name == "__init__")
            )
            stubs.append(class_stub)
    if not stubs:
        return cell
    module = ast.Module(body=stubs, type_ignores=[])
    ast.fix_missing_locations(module)
    stub_source = ast.unparse(module)
    return replace(cell, exercise=f"{stub_source}\n\n{cell.exercise}")


_PORTFOLIO_SOLUTIONS = {
    "reinforce": (
        r"""
        class ReinforceBanditAgent(nn.Module):
            def __init__(self, policy_logits, optimizer, baseline_decay=0.95):
                super().__init__()
                self.policy_logits = policy_logits
                self.optimizer = optimizer
                self.baseline_decay = baseline_decay
                self.baseline = 0.0

            def config(self):
                return {
                    "action_count": self.policy_logits.numel(),
                    "baseline_decay": self.baseline_decay,
                }

            def forward(self):
                return torch.softmax(self.policy_logits, dim=0)

            def loss(self, action, reward):
                probabilities = self.forward()
                advantage = reward - self.baseline
                log_probability = torch.log(probabilities[action].clamp_min(1e-8))
                return -log_probability * advantage

            def update(self, action, reward):
                objective = self.loss(action, reward)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                self.baseline = self.baseline_decay * self.baseline
                self.baseline += (1.0 - self.baseline_decay) * reward
                return float(objective.detach())

            def evaluate(self, reward_means):
                probabilities = self.forward().detach()
                best_action = int(reward_means.argmax())
                return {
                    "best_action": best_action,
                    "best_action_probability": float(probabilities[best_action]),
                }


        portfolio_agent = ReinforceBanditAgent(logits, opt)
        assert portfolio_agent.config()["action_count"] == len(means)
        """
    ),
    "dqn": (
        r"""
        class DqnAgent(nn.Module):
            def __init__(self, online_network, target_network, optimizer, discount):
                super().__init__()
                self.online_network = online_network
                self.target_network = target_network
                self.optimizer = optimizer
                self.discount = discount

            def config(self):
                return {
                    "state_count": n_states,
                    "action_count": n_actions,
                    "gamma": self.discount,
                }

            def forward(self, state):
                encoded_state = encode_device(state)
                return self.online_network(encoded_state)

            def loss(self, state, action, reward, next_state, done):
                predicted_q = self.forward(state).gather(1, action[:, None]).squeeze(1)
                with torch.no_grad():
                    encoded_next_state = encode_device(next_state)
                    next_q = self.target_network(encoded_next_state).max(dim=1).values
                    target_q = reward + self.discount * (1.0 - done) * next_q
                return F.smooth_l1_loss(predicted_q, target_q)

            def update(self, state, action, reward, next_state, done):
                objective = self.loss(state, action, reward, next_state, done)
                self.optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
                self.optimizer.step()
                return float(objective.detach())

            def sync_target(self):
                self.target_network.load_state_dict(self.online_network.state_dict())

            def evaluate(self, reference_q):
                with torch.no_grad():
                    learned_q = self.forward(torch.arange(n_states))
                    nonterminal = (
                        torch.arange(n_states, device=DEVICE) != terminal_state
                    )
                    agreement = (
                        (
                            learned_q.argmax(dim=1)[nonterminal]
                            == reference_q.to(DEVICE).argmax(dim=1)[nonterminal]
                        )
                        .float()
                        .mean()
                    )
                return {"greedy_agreement": float(agreement)}


        portfolio_agent = DqnAgent(online, target, opt, gamma)
        assert portfolio_agent.config()["action_count"] == n_actions
        """
    ),
    "double_dqn": (
        r"""
        class DoubleDqnAgent(nn.Module):
            def __init__(self, online_network, target_network, optimizer, discount):
                super().__init__()
                self.online_network = online_network
                self.target_network = target_network
                self.optimizer = optimizer
                self.discount = discount

            def config(self):
                return {
                    "selection": "online",
                    "evaluation": "target",
                    "gamma": self.discount,
                }

            def forward(self, state):
                return self.online_network(encode_device(state))

            def loss(self, state, action, reward, next_state, done):
                predicted_q = self.forward(state).gather(1, action[:, None]).squeeze(1)
                with torch.no_grad():
                    online_next_q = self.online_network(encode_device(next_state))
                    selected_action = online_next_q.argmax(dim=1, keepdim=True)
                    target_next_q = self.target_network(encode_device(next_state))
                    evaluated_q = target_next_q.gather(1, selected_action).squeeze(1)
                    target_q = reward + self.discount * (1.0 - done) * evaluated_q
                return F.mse_loss(predicted_q, target_q)

            def update(self, state, action, reward, next_state, done):
                objective = self.loss(state, action, reward, next_state, done)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def sync_target(self):
                self.target_network.load_state_dict(self.online_network.state_dict())

            def evaluate(self, state):
                with torch.no_grad():
                    online_q = self.forward(state)
                    selected_action = online_q.argmax(dim=1, keepdim=True)
                    target_q = self.target_network(encode_device(state))
                    evaluated_q = target_q.gather(1, selected_action).squeeze(1)
                return {
                    "selected_actions": selected_action.squeeze(1).tolist(),
                    "evaluated_mean": float(evaluated_q.mean()),
                }


        portfolio_agent = DoubleDqnAgent(online, target, opt, gamma)
        assert portfolio_agent.config()["selection"] == "online"
        """
    ),
    "dueling_dqn": (
        r"""
        class DuelingValueAgent(nn.Module):
            def __init__(self, network, optimizer):
                super().__init__()
                self.network = network
                self.optimizer = optimizer

            def config(self):
                return {
                    "state_count": n_states,
                    "action_count": n_actions,
                    "centering": "mean",
                }

            def forward(self, state):
                encoded_state = encode_device(state)
                value, advantage = self.network.streams(encoded_state)
                centered_advantage = advantage - advantage.mean(dim=1, keepdim=True)
                q_value = value + centered_advantage
                return q_value, value, centered_advantage

            def loss(self, state, target_q):
                predicted_q, _, _ = self.forward(state)
                return F.mse_loss(predicted_q, target_q)

            def update(self, state, target_q):
                objective = self.loss(state, target_q)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def evaluate(self, state, target_q):
                with torch.no_grad():
                    predicted_q, value, centered_advantage = self.forward(state)
                    mse = F.mse_loss(predicted_q, target_q)
                    centering_error = centered_advantage.mean(dim=1).abs().max()
                return {
                    "q_mse": float(mse),
                    "centering_error": float(centering_error),
                    "value_shape": tuple(value.shape),
                }


        portfolio_agent = DuelingValueAgent(model, opt)
        assert portfolio_agent.config()["centering"] == "mean"
        """
    ),
    "prioritized_replay": (
        r"""
        class PrioritizedReplayAgent(nn.Module):
            def __init__(self, q_table, optimizer, priorities, discount, alpha=0.7):
                super().__init__()
                self.q_table = q_table
                self.optimizer = optimizer
                self.priorities = priorities
                self.discount = discount
                self.alpha = alpha

            def config(self):
                return {"replay_size": len(self.priorities), "alpha": self.alpha}

            def forward(self, state, action):
                return self.q_table[state, action]

            def sampling_probabilities(self):
                scaled_priority = self.priorities.clamp_min(1e-6).pow(self.alpha)
                return scaled_priority / scaled_priority.sum()

            def loss(self, indices, beta):
                predicted_q = self.forward(states[indices], actions[indices])
                with torch.no_grad():
                    next_q = self.q_table[next_states[indices]].max(dim=1).values
                    target_q = rewards[indices]
                    target_q = (
                        target_q + self.discount * (1.0 - dones[indices]) * next_q
                    )
                td_error = target_q - predicted_q
                probability = self.sampling_probabilities()[indices]
                raw_weight = (len(self.priorities) * probability).pow(-beta)
                importance_weight = raw_weight / raw_weight.max()
                objective = (importance_weight * td_error.square()).mean()
                return objective, td_error

            def update(self, indices, beta):
                objective, td_error = self.loss(indices, beta)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                self.priorities[indices] = td_error.detach().abs() + 1e-3
                return float(objective.detach())

            def evaluate(self, reference_q):
                with torch.no_grad():
                    mse = F.mse_loss(self.q_table, reference_q)
                    probability = self.sampling_probabilities()
                return {
                    "q_mse": float(mse),
                    "probability_sum": float(probability.sum()),
                }


        portfolio_agent = PrioritizedReplayAgent(q, opt, priority, gamma)
        assert portfolio_agent.config()["replay_size"] == len(rows)
        """
    ),
    "a3c": (
        r"""
        class SharedActorCriticAgent(nn.Module):
            def __init__(
                self, network, optimizer, value_weight=0.5, entropy_weight=0.01
            ):
                super().__init__()
                self.network = network
                self.optimizer = optimizer
                self.value_weight = value_weight
                self.entropy_weight = entropy_weight

            def config(self):
                return {
                    "value_weight": self.value_weight,
                    "entropy_weight": self.entropy_weight,
                    "worker_mode": "deterministic-sequential",
                }

            def forward(self, state):
                policy_logits, value = self.network(state)
                probability = torch.softmax(policy_logits, dim=-1)
                return probability, value

            def loss(self, log_probabilities, values, returns, entropies):
                advantage = returns - values
                policy_loss = -(log_probabilities * advantage.detach()).mean()
                value_loss = advantage.square().mean()
                entropy_bonus = entropies.mean()
                objective = policy_loss + self.value_weight * value_loss
                objective = objective - self.entropy_weight * entropy_bonus
                return objective

            def update(self, log_probabilities, values, returns, entropies):
                objective = self.loss(log_probabilities, values, returns, entropies)
                self.optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                self.optimizer.step()
                return float(objective.detach())

            def evaluate(self, state, reference_q):
                with torch.no_grad():
                    probability, value = self.forward(state)
                    greedy_action = probability.argmax(dim=1)
                    optimal_action = reference_q.to(DEVICE).argmax(dim=1)
                    agreement = (greedy_action == optimal_action).float().mean()
                return {
                    "greedy_agreement": float(agreement),
                    "mean_value": float(value.mean()),
                }


        portfolio_agent = SharedActorCriticAgent(model, opt)
        assert portfolio_agent.config()["worker_mode"] == "deterministic-sequential"
        """
    ),
    "ddpg": (
        r"""
        class DeterministicPolicyAgent(nn.Module):
            def __init__(
                self,
                actor_network,
                critic_network,
                target_actor,
                target_critic,
                actor_optimizer,
                critic_optimizer,
                discount,
                tau=0.02,
            ):
                super().__init__()
                self.actor_network = actor_network
                self.critic_network = critic_network
                self.target_actor = target_actor
                self.target_critic = target_critic
                self.actor_optimizer = actor_optimizer
                self.critic_optimizer = critic_optimizer
                self.discount = discount
                self.tau = tau

            def config(self):
                return {"action_dim": 1, "gamma": self.discount, "tau": self.tau}

            def forward(self, state):
                action = self.actor_network(state)
                q_value = self.critic_network(state, action)
                return action, q_value

            def loss(self, state, action, reward, next_state, done):
                with torch.no_grad():
                    next_action = self.target_actor(next_state)
                    target_q = self.target_critic(next_state, next_action)
                    target_q = reward + self.discount * (1.0 - done) * target_q
                critic_loss = F.mse_loss(self.critic_network(state, action), target_q)
                policy_action = self.actor_network(state)
                actor_loss = -self.critic_network(state, policy_action).mean()
                return critic_loss, actor_loss

            def update(self, state, action, reward, next_state, done):
                with torch.no_grad():
                    next_action = self.target_actor(next_state)
                    target_q = self.target_critic(next_state, next_action)
                    target_q = reward + self.discount * (1.0 - done) * target_q
                critic_loss = F.mse_loss(self.critic_network(state, action), target_q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                policy_action = self.actor_network(state)
                actor_loss = -self.critic_network(state, policy_action).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self._soft_update(self.target_actor, self.actor_network)
                self._soft_update(self.target_critic, self.critic_network)
                return float(critic_loss.detach()), float(actor_loss.detach())

            def _soft_update(self, target_network, source_network):
                with torch.no_grad():
                    for target_parameter, source_parameter in zip(
                        target_network.parameters(),
                        source_network.parameters(),
                    ):
                        target_parameter.mul_(1.0 - self.tau)
                        target_parameter.add_(source_parameter, alpha=self.tau)

            def evaluate(self, state, target_action):
                with torch.no_grad():
                    action, q_value = self.forward(state)
                    action_mse = F.mse_loss(action, target_action)
                return {
                    "action_mse": float(action_mse),
                    "mean_q": float(q_value.mean()),
                }


        portfolio_agent = DeterministicPolicyAgent(
            actor,
            critic,
            actor_t,
            critic_t,
            oa,
            oc,
            gamma,
        )
        assert portfolio_agent.config()["tau"] == 0.02
        """
    ),
    "ppo": (
        r"""
        class ProximalPolicyAgent(nn.Module):
            def __init__(self, logits, value_table, optimizer, clip_epsilon=0.2):
                super().__init__()
                self.logits = logits
                self.value_table = value_table
                self.optimizer = optimizer
                self.clip_epsilon = clip_epsilon

            def config(self):
                return {"clip_epsilon": self.clip_epsilon, "epochs_per_batch": 4}

            def forward(self, state):
                probability = torch.softmax(self.logits[state], dim=1)
                value = self.value_table[state]
                return probability, value

            def loss(self, state, action, old_log_probability, advantage, returns):
                probability, value = self.forward(state)
                row = torch.arange(len(state), device=probability.device)
                new_log_probability = torch.log(
                    probability[row, action].clamp_min(1e-8)
                )
                ratio = torch.exp(new_log_probability - old_log_probability)
                unclipped = ratio * advantage
                clipped_ratio = ratio.clamp(
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )
                clipped = clipped_ratio * advantage
                policy_objective = torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, returns)
                entropy = (
                    -(probability * torch.log(probability.clamp_min(1e-8)))
                    .sum(dim=1)
                    .mean()
                )
                return -policy_objective + 0.5 * value_loss - 0.01 * entropy

            def update(self, state, action, old_log_probability, advantage, returns):
                objective = self.loss(
                    state,
                    action,
                    old_log_probability,
                    advantage,
                    returns,
                )
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def evaluate(self, reference_q):
                with torch.no_grad():
                    probability = torch.softmax(self.logits, dim=1)
                    nonterminal = torch.arange(n_states) != terminal_state
                    agreement = (
                        (
                            probability.argmax(dim=1)[nonterminal]
                            == reference_q.argmax(dim=1)[nonterminal]
                        )
                        .float()
                        .mean()
                    )
                    entropy = (
                        -(probability * torch.log(probability.clamp_min(1e-8)))
                        .sum(dim=1)
                        .mean()
                    )
                return {"greedy_agreement": float(agreement), "entropy": float(entropy)}


        portfolio_agent = ProximalPolicyAgent(policy_logits, values, opt)
        assert portfolio_agent.config()["clip_epsilon"] == 0.2
        """
    ),
    "sac": (
        r"""
        class SoftActorCriticAgent(nn.Module):
            def __init__(
                self,
                actor_network,
                critic_network,
                actor_optimizer,
                critic_optimizer,
                temperature=0.08,
            ):
                super().__init__()
                self.actor_network = actor_network
                self.critic_network = critic_network
                self.actor_optimizer = actor_optimizer
                self.critic_optimizer = critic_optimizer
                self.temperature = temperature

            def config(self):
                return {"action_dim": 1, "temperature": self.temperature}

            def forward(self, state):
                action, log_probability, mean_action = self.actor_network.sample(state)
                encoded_state = encode_device(state)
                q_value = self.critic_network(torch.cat([encoded_state, action], dim=1))
                return action, log_probability, mean_action, q_value

            def loss(self, state, replay_action, reward):
                encoded_state = encode_device(state)
                predicted_q = self.critic_network(
                    torch.cat([encoded_state, replay_action], dim=1)
                )
                critic_loss = F.mse_loss(predicted_q, reward)
                sampled_action, log_probability, _, q_value = self.forward(state)
                del sampled_action
                actor_loss = (
                    self.temperature * log_probability - q_value.squeeze(1)
                ).mean()
                return critic_loss, actor_loss

            def update(self, state, replay_action, reward):
                encoded_state = encode_device(state)
                predicted_q = self.critic_network(
                    torch.cat([encoded_state, replay_action], dim=1)
                )
                critic_loss = F.mse_loss(predicted_q, reward)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                action, log_probability, _, q_value = self.forward(state)
                del action
                actor_loss = (
                    self.temperature * log_probability - q_value.squeeze(1)
                ).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                return float(critic_loss.detach()), float(actor_loss.detach())

            def evaluate(self, state, target_action):
                with torch.no_grad():
                    _, _, mean_action, q_value = self.forward(state)
                    action_mse = F.mse_loss(mean_action, target_action)
                return {
                    "action_mse": float(action_mse),
                    "mean_q": float(q_value.mean()),
                }


        portfolio_agent = SoftActorCriticAgent(actor, critic, op, oq, alpha)
        assert portfolio_agent.config()["temperature"] == alpha
        """
    ),
    "alphazero": (
        r"""
        class AlphaZeroSearchAgent(nn.Module):
            def __init__(
                self,
                policy_value_network,
                optimizer,
                transition_fn,
                leaf_action_values,
                discount,
                action_count,
                exploration=1.5,
            ):
                super().__init__()
                self.policy_value_network = policy_value_network
                self.optimizer = optimizer
                self.transition_fn = transition_fn
                self.leaf_action_values = leaf_action_values.detach().cpu()
                self.discount = discount
                self.action_count = action_count
                self.exploration = exploration
                self.last_visit_count = torch.zeros(action_count)
                self.last_root_policy = torch.zeros(action_count)

            def config(self):
                return {
                    "action_count": self.action_count,
                    "c_puct": self.exploration,
                }

            def forward(self, state):
                policy_logits, value = self.policy_value_network(state)
                return torch.softmax(policy_logits, dim=1), value

            def select_action(
                self,
                state,
                simulations=80,
                temperature=1.0,
            ):
                if simulations < 1 or temperature < 0.0:
                    raise ValueError("simulations must be positive and temperature nonnegative")
                with torch.no_grad():
                    prior, _ = self.forward(torch.tensor([state]))
                prior = prior[0].detach().cpu()
                visit_count = torch.zeros(self.action_count)
                total_value = torch.zeros(self.action_count)
                for _ in range(simulations):
                    mean_q = torch.where(
                        visit_count > 0,
                        total_value / visit_count.clamp_min(1.0),
                        torch.zeros_like(total_value),
                    )
                    exploration_bonus = self.exploration * prior
                    exploration_bonus = exploration_bonus * torch.sqrt(
                        visit_count.sum() + 1.0
                    )
                    exploration_bonus = exploration_bonus / (1.0 + visit_count)
                    action = int((mean_q + exploration_bonus).argmax())
                    next_state, reward, done = self.transition_fn(state, action)
                    leaf = reward
                    if not done:
                        leaf += self.discount * float(
                            self.leaf_action_values[next_state].max()
                        )
                    visit_count[action] += 1.0
                    total_value[action] += leaf
                if temperature == 0.0:
                    root_policy = torch.zeros_like(visit_count)
                    root_policy[visit_count.argmax()] = 1.0
                else:
                    scaled_visit = visit_count.pow(1.0 / temperature)
                    root_policy = scaled_visit / scaled_visit.sum().clamp_min(1.0)
                self.last_visit_count = visit_count.clone()
                self.last_root_policy = root_policy.clone()
                action = int(root_policy.argmax())
                return action, root_policy, visit_count, prior

            def loss(self, state, target_policy, target_value):
                policy_logits, predicted_value = self.policy_value_network(state)
                policy_loss = (
                    -(target_policy * torch.log_softmax(policy_logits, dim=1))
                    .sum(dim=1)
                    .mean()
                )
                value_loss = F.mse_loss(predicted_value, target_value)
                regularization = sum(
                    parameter.square().sum()
                    for parameter in self.policy_value_network.parameters()
                )
                return policy_loss + value_loss + 1e-4 * regularization

            def update(self, state, target_policy, target_value):
                objective = self.loss(state, target_policy, target_value)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def evaluate(
                self,
                state,
                target_policy,
                target_value,
                visit_count,
                raw_prior,
            ):
                with torch.no_grad():
                    predicted_policy, predicted_value = self.forward(state)
                    policy_cross_entropy = (
                        -(target_policy * torch.log(predicted_policy.clamp_min(1e-8)))
                        .sum(dim=1)
                        .mean()
                    )
                    value_mse = F.mse_loss(predicted_value, target_value)
                    improved_target_l1 = (target_policy - raw_prior).abs().mean()
                    target_entropy = -(
                        target_policy
                        * torch.log(target_policy.clamp_min(1e-8))
                    ).sum(dim=1).mean()
                    mean_visit_total = visit_count.sum(dim=1).float().mean()
                return {
                    "policy_cross_entropy": float(policy_cross_entropy),
                    "value_mse": float(value_mse),
                    "improved_target_l1": float(improved_target_l1),
                    "target_policy_entropy": float(target_entropy),
                    "mean_visit_total": float(mean_visit_total),
                }


        portfolio_agent = None
        """
    ),
}


_PORTFOLIO_RUNS = {
    "reinforce": r"""
        before = portfolio_agent.policy_logits.detach().clone()
        evidence_action = int(means.argmax())
        evidence_reward = float(means[evidence_action])
        integrated_loss = portfolio_agent.update(evidence_action, evidence_reward)
        metrics = portfolio_agent.evaluate(means)
        parameter_delta = float(
            (portfolio_agent.policy_logits.detach() - before).abs().sum()
        )
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert 0.0 <= metrics["best_action_probability"] <= 1.0
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "dqn": r"""
        evidence_state, evidence_action, evidence_reward = ACCELERATOR.move(
            states, actions, rewards
        )
        evidence_next_state, evidence_done = ACCELERATOR.move(next_states, dones)
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        integrated_loss = portfolio_agent.update(
            evidence_state,
            evidence_action,
            evidence_reward,
            evidence_next_state,
            evidence_done,
        )
        portfolio_agent.sync_target()
        metrics = portfolio_agent.evaluate(q_star)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert 0.0 <= metrics["greedy_agreement"] <= 1.0
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "double_dqn": r"""
        evidence_state, evidence_action, evidence_reward = ACCELERATOR.move(
            states, actions, rewards
        )
        evidence_next_state, evidence_done = ACCELERATOR.move(next_states, dones)
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        integrated_loss = portfolio_agent.update(
            evidence_state,
            evidence_action,
            evidence_reward,
            evidence_next_state,
            evidence_done,
        )
        portfolio_agent.sync_target()
        metrics = portfolio_agent.evaluate(torch.arange(n_states))
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert len(metrics["selected_actions"]) == n_states
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "dueling_dqn": r"""
        evidence_state = torch.arange(n_states)
        evidence_target = q_star.to(DEVICE)
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        integrated_loss = portfolio_agent.update(evidence_state, evidence_target)
        metrics = portfolio_agent.evaluate(evidence_state, evidence_target)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert metrics["centering_error"] < 1e-5
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "prioritized_replay": r"""
        evidence_indices = torch.arange(min(32, len(rows)))
        before = portfolio_agent.q_table.detach().clone()
        integrated_loss = portfolio_agent.update(evidence_indices, beta=0.6)
        metrics = portfolio_agent.evaluate(q_star)
        parameter_delta = float(
            (portfolio_agent.q_table.detach() - before).abs().sum()
        )
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert abs(metrics["probability_sum"] - 1.0) < 1e-5
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "a3c": r"""
        evidence_states = torch.arange(n_states)
        probabilities, evidence_values = portfolio_agent.forward(evidence_states)
        evidence_actions = probabilities.argmax(dim=1)
        rows_device = torch.arange(n_states, device=DEVICE)
        log_probabilities = torch.log(
            probabilities[rows_device, evidence_actions].clamp_min(1e-8)
        )
        entropies = -(
            probabilities * torch.log(probabilities.clamp_min(1e-8))
        ).sum(dim=1)
        evidence_returns = q_star.max(dim=1).values.to(DEVICE)
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        integrated_loss = portfolio_agent.update(
            log_probabilities,
            evidence_values,
            evidence_returns,
            entropies,
        )
        metrics = portfolio_agent.evaluate(evidence_states, q_star)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert 0.0 <= metrics["greedy_agreement"] <= 1.0
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "ddpg": r"""
        evidence_slice = slice(0, 64)
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        critic_loss, actor_loss = portfolio_agent.update(
            replay_s[evidence_slice],
            replay_a[evidence_slice],
            replay_r[evidence_slice],
            replay_ns[evidence_slice],
            replay_done[evidence_slice],
        )
        metrics = portfolio_agent.evaluate(state_grid, target_action)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(actor_loss)
        assert math.isfinite(critic_loss) and math.isfinite(metrics["action_mse"])
        print({"actor_loss": actor_loss, "delta": parameter_delta, **metrics})
    """,
    "ppo": r"""
        evidence_states = torch.arange(n_states).repeat_interleave(4)
        with torch.no_grad():
            old_probability, old_value = portfolio_agent.forward(evidence_states)
            evidence_actions = q_star[evidence_states].argmax(dim=1)
            rows = torch.arange(len(evidence_states))
            old_log_probability = torch.log(
                old_probability[rows, evidence_actions].clamp_min(1e-8)
            )
            evidence_returns = q_star[evidence_states, evidence_actions]
            evidence_advantage = evidence_returns - old_value
        before = portfolio_agent.logits.detach().clone()
        integrated_loss = portfolio_agent.update(
            evidence_states,
            evidence_actions,
            old_log_probability,
            evidence_advantage,
            evidence_returns,
        )
        metrics = portfolio_agent.evaluate(q_star)
        parameter_delta = float(
            (portfolio_agent.logits.detach() - before).abs().sum()
        )
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert 0.0 <= metrics["greedy_agreement"] <= 1.0
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "sac": r"""
        evidence_slice = slice(0, 64)
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        critic_loss, actor_loss = portfolio_agent.update(
            replay_s[evidence_slice],
            replay_a[evidence_slice],
            replay_r[evidence_slice],
        )
        metrics = portfolio_agent.evaluate(torch.arange(n_states), target)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(actor_loss)
        assert math.isfinite(critic_loss) and math.isfinite(metrics["action_mse"])
        print({"actor_loss": actor_loss, "delta": parameter_delta, **metrics})
    """,
}


def _split_portfolio_solution(spec: FieldPaperSpec) -> tuple[str, str]:
    source = dedent(_PORTFOLIO_SOLUTIONS[spec.slug]).strip()
    lines = source.splitlines()
    verification_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("portfolio_agent = ")
    )
    class_source = "\n".join(lines[:verification_index])
    verification_source = "\n".join(lines[verification_index:])
    return class_source, verification_source


def _portfolio_cell(spec: FieldPaperSpec) -> object:
    class_source, _ = _split_portfolio_solution(spec)
    documented_tree = _with_contract_docstrings(spec, class_source)
    documented_source = ast.unparse(documented_tree)
    return code(
        _exercise_architecture(spec, class_source),
        documented_source,
        "portfolio-architecture",
    )


def _portfolio_verification_cell(spec: FieldPaperSpec) -> object:
    exercise = f"""
    # TODO 5: {_PORTFOLIO_CLASS_NAMES[spec.slug]} 기반 학습의 최종 증거를 검증하세요.
    # 계약: history, portfolio_metrics, parameter_delta를 모두 검사합니다.
    raise NotImplementedError
    """
    solution = """
    config_contract = portfolio_agent.config()
    numeric_metrics = [
        float(value)
        for value in portfolio_metrics.values()
        if isinstance(value, (int, float))
    ]
    assert config_contract and len(history) > 0
    assert np.isfinite(np.asarray(history, dtype=float)).all()
    assert parameter_delta > 0.0
    assert numeric_metrics and np.isfinite(numeric_metrics).all()
    print(
        {
            "class": type(portfolio_agent).__name__,
            "steps": len(history),
            "parameter_delta": parameter_delta,
            "metrics": portfolio_metrics,
        }
    )
    """
    return code(
        exercise,
        solution,
        "portfolio-verification",
    )


def _integration_markdown(spec: FieldPaperSpec) -> object:
    guide = _PORTFOLIO_GUIDES[spec.slug]
    paper_part, lab_part, evidence = spec.mappings[0]
    return markdown(
        rf"""
        ### 단계 5. 통합 Agent가 실제 update와 evaluate를 소유하는지 검증

        - **논문 위치:** `{paper_part}`의 **{lab_part}**
        - **핵심 식:**

          $$
          {guide["equation"]}
          $$

        - **입출력 shape:** `{guide["shape"]}`
        - **구현 이유:** 앞 셀은 target, objective, search를 작은 연산으로
          분해해 확인하고, 이 셀은 같은 구성 요소를
          `{_PORTFOLIO_CLASS_NAMES[spec.slug]}`에 주입해 최종 학습·평가 경로를
          하나로 묶습니다. {evidence}
        - **완료 증거:** `update(...)`가 유한한 loss를 반환하고,
          `parameter_delta > 0`이며, `evaluate(...)`의 정책·가치 지표가 유한해야 합니다.
          `config()`만 호출하는 것은 실행 재현으로 보지 않습니다.
        """,
        "integration-explanation",
        "portfolio-verification",
        "equation",
    )


_TRAINING_LOOP_MARKERS = {
    "reinforce": "for _ in range(400):",
    "dqn": "for step in range(260):",
    "double_dqn": "for step in range(220):",
    "dueling_dqn": "for _ in range(220):",
    "prioritized_replay": "for step in range(300):",
    "a3c": "for round_id in range(100):",
    "ddpg": "for _ in range(320):",
    "ppo": "for _ in range(45):",
    "sac": "for _ in range(360):",
    "alphazero": "for _ in range(200):",
}


_INTEGRATED_TRAINING_DRIVERS = {
    "reinforce": r"""
        portfolio_agent = ReinforceBanditAgent(logits, opt)
        initial_parameters = logits.detach().clone()
        history = []
        for _ in range(400):
            probabilities = portfolio_agent.forward()
            action = int(torch.multinomial(probabilities, 1, generator=rng))
            reward = float(means[action]) + float(noise_rng.normal(0, 0.15))
            history.append(portfolio_agent.update(action, reward))
            reward_history.append(reward)
            prob_history.append(probabilities.detach().cpu().numpy())
        parameter_delta = float(
            (logits.detach() - initial_parameters).abs().sum()
        )
        assert parameter_delta > 0.0 and np.isfinite(history).all()
        assert int(logits.argmax()) == int(means.argmax())
    """,
    "dqn": r"""
        portfolio_agent = DqnAgent(online, target, opt, gamma)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        for step in range(260):
            index = torch.randint(
                len(rows), (min(32, len(rows)),), generator=rng
            )
            state, action, reward, next_state, done = ACCELERATOR.move(
                states[index],
                actions[index],
                rewards[index],
                next_states[index],
                dones[index],
            )
            history.append(
                portfolio_agent.update(
                    state, action, reward, next_state, done
                )
            )
            if (step + 1) % 20 == 0:
                portfolio_agent.sync_target()
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
        assert np.mean(history[-20:]) < np.mean(history[:20])
    """,
    "double_dqn": r"""
        portfolio_agent = DoubleDqnAgent(online, target, opt, gamma)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        for step in range(220):
            index = torch.randint(
                len(rows), (min(32, len(rows)),), generator=gen
            )
            state, action, reward, next_state, done = ACCELERATOR.move(
                states[index],
                actions[index],
                rewards[index],
                next_states[index],
                dones[index],
            )
            history.append(
                portfolio_agent.update(
                    state, action, reward, next_state, done
                )
            )
            if (step + 1) % 20 == 0:
                portfolio_agent.sync_target()
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in online.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        portfolio_metrics = portfolio_agent.evaluate(torch.arange(n_states))
        assert parameter_delta > 0.0 and np.isfinite(history).all()
        assert len(portfolio_metrics["selected_actions"]) == n_states
    """,
    "dueling_dqn": r"""
        portfolio_agent = DuelingValueAgent(model, opt)
        state_ids = torch.arange(n_states)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        for _ in range(220):
            history.append(portfolio_agent.update(state_ids, q_ref))
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert history[-1] < 0.05 * initial and parameter_delta > 0.0
    """,
    "prioritized_replay": r"""
        portfolio_agent = PrioritizedReplayAgent(q, opt, priority, gamma)
        initial_parameters = q.detach().clone()
        for step in range(300):
            probability = portfolio_agent.sampling_probabilities().detach().numpy()
            index_numpy = rng.choice(
                len(rows),
                size=min(16, len(rows)),
                replace=True,
                p=probability,
            )
            counts += np.bincount(index_numpy, minlength=len(rows))
            index = torch.tensor(index_numpy)
            beta = 0.4 + 0.6 * step / 299
            history.append(portfolio_agent.update(index, beta))
        parameter_delta = float((q.detach() - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
        assert priority.min() > 0
    """,
    "a3c": r"""
        portfolio_agent = SharedActorCriticAgent(model, opt)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        for round_id in range(100):
            for worker in range(4):
                generator = torch.Generator().manual_seed(
                    452 + 1000 * round_id + worker
                )
                state = start_state
                log_probabilities = []
                rollout_values = []
                entropies = []
                rollout_rewards = []
                done = False
                for _ in range(2 * n_states + 2):
                    probabilities, value = portfolio_agent.forward(
                        torch.tensor([state])
                    )
                    probability = probabilities[0]
                    action = int(
                        torch.multinomial(
                            probability.detach().cpu(), 1, generator=generator
                        )
                    )
                    next_state, reward, done = env_step(state, action)
                    log_probabilities.append(
                        torch.log(probability[action].clamp_min(1e-8))
                    )
                    entropies.append(
                        -(probability * torch.log(probability.clamp_min(1e-8))).sum()
                    )
                    rollout_values.append(value[0])
                    rollout_rewards.append(reward)
                    state = next_state
                    if done:
                        break
                bootstrap = torch.tensor(0.0, device=DEVICE)
                if not done:
                    bootstrap = portfolio_agent.forward(
                        torch.tensor([state])
                    )[1][0].detach()
                returns = nstep_returns(rollout_rewards, bootstrap, gamma)
                history.append(
                    portfolio_agent.update(
                        torch.stack(log_probabilities),
                        torch.stack(rollout_values),
                        returns,
                        torch.stack(entropies),
                    )
                )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "ddpg": r"""
        portfolio_agent = DeterministicPolicyAgent(
            actor,
            critic,
            actor_t,
            critic_t,
            oa,
            oc,
            gamma,
        )
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        for _ in range(320):
            index = torch.randint(len(replay_s), (64,), generator=gen).to(DEVICE)
            history.append(
                portfolio_agent.update(
                    replay_s[index],
                    replay_a[index],
                    replay_r[index],
                    replay_ns[index],
                    replay_done[index],
                )
            )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "ppo": r"""
        portfolio_agent = ProximalPolicyAgent(policy_logits, values, opt)
        initial_parameters = policy_logits.detach().clone()
        for _ in range(45):
            with torch.no_grad():
                old_probability, old_value = portfolio_agent.forward(batch_states)
                batch_actions = torch.multinomial(
                    old_probability, 1, generator=gen
                ).squeeze(1)
                rows = torch.arange(len(batch_states))
                old_log_probability = torch.log(
                    old_probability[rows, batch_actions].clamp_min(1e-8)
                )
                returns = q_star[batch_states, batch_actions]
                advantage = returns - old_value
                advantage = (advantage - advantage.mean()) / (
                    advantage.std() + 1e-6
                )
            epoch_losses = []
            for _ in range(4):
                epoch_losses.append(
                    portfolio_agent.update(
                        batch_states,
                        batch_actions,
                        old_log_probability,
                        advantage,
                        returns,
                    )
                )
            with torch.no_grad():
                new_probability, _ = portfolio_agent.forward(batch_states)
                new_log_probability = torch.log(
                    new_probability[rows, batch_actions].clamp_min(1e-8)
                )
                ratio = torch.exp(new_log_probability - old_log_probability)
                clip_fraction = float(
                    ((ratio - 1.0).abs() > 0.2).float().mean()
                )
            history.append((epoch_losses[-1], clip_fraction))
        parameter_delta = float(
            (policy_logits.detach() - initial_parameters).abs().sum()
        )
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "sac": r"""
        portfolio_agent = SoftActorCriticAgent(actor, critic, op, oq, alpha)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        for _ in range(360):
            index = torch.randint(len(replay_s), (64,), generator=gen).to(DEVICE)
            history.append(
                portfolio_agent.update(
                    replay_s[index], replay_a[index], replay_r[index]
                )
            )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in actor.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "alphazero": r"""
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in net.parameters()]
        )
        for _ in range(200):
            history.append(
                portfolio_agent.update(train_states, target_pi, target_z)
            )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in net.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        portfolio_metrics = portfolio_agent.evaluate(
            train_states,
            target_pi,
            target_z,
            visit_counts,
            raw_priors,
        )
        with torch.no_grad():
            final_policy, _ = portfolio_agent.forward(train_states)
        assert history[-1] < initial and parameter_delta > 0.0
        assert torch.allclose(target_pi.sum(1), torch.ones_like(target_pi.sum(1)))
        assert torch.equal(
            visit_counts.sum(1),
            torch.full_like(visit_counts.sum(1), 80),
        )
        assert portfolio_metrics["improved_target_l1"] > 0.0
        assert portfolio_metrics["mean_visit_total"] == 80.0
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].bar(np.arange(n_actions), root_visits.detach().cpu())
        axes[0].set_title("root visit counts")
        axes[1].imshow(final_policy.detach().cpu().T, aspect="auto", vmin=0, vmax=1)
        axes[1].set_title("learned p")
        axes[2].plot(history)
        axes[2].set_title("Eq. (1) loss")
        fig.tight_layout()
        plt.show()
    """,
}


_EVALUATION_CALLS = {
    "reinforce": "portfolio_metrics = portfolio_agent.evaluate(means)",
    "dqn": "portfolio_metrics = portfolio_agent.evaluate(q_star)",
    "dueling_dqn": r"""
        portfolio_metrics = portfolio_agent.evaluate(torch.arange(n_states), q_ref)
    """,
    "prioritized_replay": r"""
        portfolio_metrics = portfolio_agent.evaluate(q_star)
    """,
    "a3c": r"""
        portfolio_metrics = portfolio_agent.evaluate(torch.arange(n_states), q_star)
    """,
    "ddpg": "portfolio_metrics = portfolio_agent.evaluate(state_grid, target_action)",
    "ppo": "portfolio_metrics = portfolio_agent.evaluate(q_star)",
    "sac": r"""
        portfolio_metrics = portfolio_agent.evaluate(torch.arange(n_states), target)
    """,
}


_TRAINING_CODE_INDEX = {
    "double_dqn": 2,
    "alphazero": 2,
}


def _integrated_training_cell(spec: FieldPaperSpec, cell: object) -> object:
    marker = _TRAINING_LOOP_MARKERS[spec.slug]
    prefix, separator, _ = cell.solution.partition(marker)
    if not separator:
        raise ValueError(f"training loop marker not found for {spec.slug}: {marker}")
    driver = dedent(_INTEGRATED_TRAINING_DRIVERS[spec.slug]).strip()
    exercise = _exercise_with_missing_public_api(spec, cell).exercise
    exercise = "\n".join(
        line
        for line in exercise.splitlines()
        if line.strip() != "raise NotImplementedError"
    )
    exercise = (
        f"{exercise}\n\n"
        f"# 통합 완료 조건: {_PORTFOLIO_CLASS_NAMES[spec.slug]}.update(...)를 반복하고,\n"
        "# history, parameter_delta, 유한 loss를 검증합니다.\n"
        "raise NotImplementedError"
    )
    return replace(cell, exercise=exercise, solution=f"{prefix.rstrip()}\n\n{driver}")


def _integrated_evaluation_cell(spec: FieldPaperSpec, cell: object) -> object:
    call = dedent(_EVALUATION_CALLS[spec.slug]).strip()
    exercise = (
        f"# TODO: {_PORTFOLIO_CLASS_NAMES[spec.slug]}.evaluate(...)를 호출해 "
        "portfolio_metrics를 만들고 시각화하세요.\nraise NotImplementedError"
    )
    solution = f"{call}\nassert portfolio_metrics\n\n{cell.solution}"
    return replace(cell, exercise=exercise, solution=solution)


_STAGE_MAPPING_ORDER = {
    "reinforce": (2, 0, 1),
    "dqn": (0, 2, 1),
    "double_dqn": (1, 2, 0),
    "dueling_dqn": (1, 0, 2),
    "prioritized_replay": (1, 2, 0),
    "a3c": (0, 1, 2),
    "ddpg": (2, 1, 0),
    "ppo": (1, 2, 0),
    "sac": (2, 1, 0),
    "alphazero": (1, 2, 0),
}


def _stage_markdown(spec: FieldPaperSpec, stage_index: int, code_tag: str) -> object:
    mapping_index = _STAGE_MAPPING_ORDER[spec.slug][stage_index]
    paper_part, lab_part, evidence = spec.mappings[mapping_index]
    stage_names = ("핵심 연산 검증", "학습·업데이트", "평가·시각화")
    return markdown(
        f"""
        ### 단계 {stage_index + 2}. {stage_names[stage_index]}

        - **논문의 어느 부분인가:** `{paper_part}` — **{lab_part}**
        - **구현 이유:** {evidence}
        - **읽을 코드:** 아래 `{code_tag}` 셀에서 state/action tensor의 shape, gradient가
          흐르는 경로와 target 또는 old-policy가 고정되는 경계를 확인합니다.

        실행 결과만 보지 말고, 이 단계의 입력과 출력이 앞선 수식의 어떤 기호인지 먼저
        주석으로 적은 뒤 실습하세요.
        - **완료 증거:** 아래 셀의 `assert`와 출력 metric이 shape·loss 계약을 검사합니다.
        """,
        "step-explanation",
        code_tag,
    )


def _upgrade_for_portfolio(spec: FieldPaperSpec) -> FieldPaperSpec:
    intro, setup, *algorithm_cells = spec.cells
    staged_cells: list[object] = []
    code_index = 0
    training_index = _TRAINING_CODE_INDEX.get(spec.slug, 1)
    for cell in algorithm_cells:
        if cell.cell_type == "code":
            if code_index == training_index:
                cell = _integrated_training_cell(spec, cell)
            elif code_index == 2 and spec.slug not in {"double_dqn", "alphazero"}:
                cell = _integrated_evaluation_cell(spec, cell)
            else:
                cell = _exercise_with_missing_public_api(spec, cell)
            code_tag = cell.tags[0] if cell.tags else f"task-{code_index + 1}"
            staged_cells.append(_stage_markdown(spec, code_index, code_tag))
            code_index += 1
        staged_cells.append(cell)
    return replace(
        spec,
        cells=(
            intro,
            setup,
            _portfolio_markdown(spec),
            _portfolio_cell(spec),
            *staged_cells,
            _integration_markdown(spec),
            _portfolio_verification_cell(spec),
        ),
    )


SPECS = tuple(
    _upgrade_for_portfolio(spec)
    for spec in (
        REINFORCE,
        DQN,
        DOUBLE_DQN,
        DUELING_DQN,
        PER,
        A3C,
        DDPG,
        PPO,
        SAC,
        ALPHAZERO,
    )
)

assert len(SPECS) == 10
assert [spec.number for spec in SPECS] == list(range(10))
