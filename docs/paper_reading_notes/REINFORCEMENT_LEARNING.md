# 강화학습·에이전트 논문 독해 노트

> **AI 보조 작성 안내:** 이 한국어 요약은 공식 번역이 아닙니다. 내용과 인용의 최종 기준은
> 연결된 원문입니다. [작성 원칙과 한계](AI_ASSISTED_NOTICE.md) · [70편 목차](README.md)

이 문서는 구현 노트북 10편을 policy gradient, value learning, actor–critic, search의 흐름으로 읽기 위한 안내서다. 로컬 `gridworld.json`과 bandit으로 핵심 update만 검증하므로, 논문의 Atari·MuJoCo·self-play 규모와 결과를 재현했다고 해석하면 안 된다.

권장 학습법은 **핵심 수식 Task → 포트폴리오 agent class 조립 → contract 검증**이다. TODO 1–3에서 target·objective·update와 평가를 먼저 완성하고, TODO 4에서 이를 명시적 `Agent` class의 policy/value·loss·update·evaluate 경계로 재구성한다. 학습 loop는 이 agent의 `update`를 실제로 반복 호출한다. TODO 5에서는 config뿐 아니라 history, parameter 변화와 `evaluate` metric을 함께 검증한다.

## 00. REINFORCE (1992)

원문: [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696)

1. **한 문장 요약:** 보상으로 가중한 선택 확률의 log-gradient를 따라가면 stochastic policy의 기대 보상을 직접 높일 수 있다.
2. **배경과 이전 한계:** action의 표본 추출은 미분할 수 없고 환경 dynamics의 미분값도 보통 모른다. supervised target 없이 delayed reward만으로 stochastic units를 학습할 일반 규칙이 필요했다.
3. **읽기 전 기초지식:** softmax/Bernoulli policy, log-derivative trick, Monte Carlo return, discounted return-to-go, baseline과 분산, episodic trajectory.
4. **핵심 아이디어:**
   - characteristic eligibility $\nabla_w\log\pi_w(a)$에 reward를 곱한다.
   - action과 무관한 baseline은 기대 gradient를 바꾸지 않고 분산을 줄인다.
   - episodic 문제에서는 각 시점 이후 return과 누적 eligibility를 결합한다.
5. **꼭 볼 원문 위치:** `§3, p. 234, core REINFORCE rule` ↔ reward-minus-baseline update; `§3, Eq. (7)–(10)` ↔ eligibility와 adaptive baseline; `§5, Eq. (11) and Theorem 2` ↔ episodic accumulation.
6. **수식·알고리즘 직관:** 우연히 높은 return을 만든 action은 그 상황에서 log-probability를 올리고, 낮은 return action은 내린다. $\nabla\log\pi$는 action sampling을 통과해 미분하는 대신 선택된 action의 확률을 미분하는 우회로다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/00_reinforce.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/00_reinforce.ipynb)
   - TODO 1: return-to-go를 역방향으로 계산한다.
   - TODO 2: `(reward-baseline)*log pi(a)`로 softmax bandit을 학습한다.
   - TODO 3: 최적 arm 확률과 moving reward를 평가한다.
   - TODO 4: `ReinforceBanditAgent`에 policy, return/baseline loss, update와 평가를 명시적으로 묶는다.
   - TODO 5: `config()["action_count"] == len(means)`와 agent 기반 history·parameter 변화·reward metric을 함께 검증한다.
8. **5분 마무리:** REINFORCE는 model-free·unbiased이지만 sample variance가 크다. bandit 축소판은 장기 credit assignment를 거의 드러내지 않는다. **자가점검:** (a) baseline 기대 gradient가 0인 이유는? (b) loss에 마이너스 부호가 필요한 이유는? (c) 전체 episode return과 return-to-go의 차이는?

## 01. DQN (2015)

원문: [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236)

1. **한 문장 요약:** convolutional Q-network에 experience replay와 지연 target network를 결합해 raw Atari 화면에서 안정적인 value learning을 보였다.
2. **배경과 이전 한계:** online transition은 시간적으로 상관되고, TD target도 현재 network에 의존해 움직인다. 비선형 function approximation, bootstrapping, off-policy learning의 결합은 쉽게 발산한다.
3. **읽기 전 기초지식:** MDP, Bellman optimality equation, Q-learning, epsilon-greedy, replay buffer, target network, terminal masking.
4. **핵심 아이디어:**
   - replay에서 무작위 mini-batch를 뽑아 상관을 낮추고 데이터를 재사용한다.
   - 일정 기간 고정한 $Q_{target}$이 Bellman 회귀 목표를 만든다.
   - $y=r+\gamma(1-d)\max_{a'}Q_{target}(s',a')$로 online Q를 갱신한다.
5. **꼭 볼 원문 위치:** `Methods, ‘Deep reinforcement learning’, Eq. (1)–(2)` ↔ squared TD loss/target; `Methods, experience replay paragraph` ↔ uniform replay; `Extended Data, Algorithm 1` ↔ 전체 DQN loop.
6. **수식·알고리즘 직관:** 신경망을 매 순간 자기 예측값으로 바로 쫓아가게 하지 않고, 과거 transition을 섞은 데이터와 잠시 고정된 교사 network로 supervised regression처럼 만든다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/01_dqn.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/01_dqn.ipynb)
   - TODO 1: terminal mask와 detach가 있는 Bellman target을 만든다.
   - TODO 2: replay mini-batch 학습과 20-step target sync를 구현한다.
   - TODO 3: Bellman residual과 optimal greedy agreement를 평가한다.
   - TODO 4: `DqnAgent`에 online/target Q, Bellman loss, replay update와 평가를 구현한다.
   - TODO 5: `config()["action_count"] == n_actions`와 agent 기반 history·parameter 변화·Q metric을 함께 검증한다.
8. **5분 마무리:** DQN의 안정화 장치는 replay와 frozen target이다. one-hot 7-state 실습은 pixel representation, exploration, Atari preprocessing을 재현하지 않는다. **자가점검:** (a) done일 때 bootstrap을 끄는 이유는? (b) target을 detach하지 않으면 어떤 objective가 되는가? (c) replay가 on-policy 순서를 어떻게 바꾸는가?

## 02. Double DQN (2016)

원문: [Deep Reinforcement Learning with Double Q-learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295)

1. **한 문장 요약:** next action의 **선택**과 그 action의 **평가**에 서로 다른 estimator를 사용해 max 연산의 과대추정을 줄인다.
2. **배경과 이전 한계:** 여러 noisy Q 추정값 중 최대를 택하면 실제 값이 같아도 양의 noise가 선택되기 쉽다. DQN의 target network는 있었지만 표준 target은 target estimator가 선택과 평가를 모두 담당했다.
3. **읽기 전 기초지식:** DQN target, Jensen/maximization bias, 독립 noise estimator, online/target network, gather/argmax.
4. **핵심 아이디어:**
   - DQN target의 `max`가 만드는 positive bias를 이론과 toy example로 분리한다.
   - $a^*=\arg\max_a Q_{online}(s',a)$로 선택한다.
   - $Q_{target}(s',a^*)$로 평가해 선택 noise와 평가 noise의 결합을 약화한다.
5. **꼭 볼 원문 위치:** `§2, Eq. (3)` ↔ DQN max target; `§2, Eq. (4); §3 Double DQN displayed target` ↔ decoupling; `§2, Theorem 1 and Figure 1` ↔ overestimation.
6. **수식·알고리즘 직관:** 시험을 가장 잘 본 학생을 같은 noisy 시험 점수로 평가하면 최고점이 부풀려진다. 한 시험으로 학생을 고르고 독립적인 다른 시험으로 평가하면 선택 bias가 줄어든다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/02_double_dqn.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/02_double_dqn.ipynb)
   - TODO 1: online argmax/target gather를 분리한다.
   - TODO 2: 8개 true-zero action의 두 noisy estimator로 bias를 계량한다.
   - TODO 3: Double DQN replay update와 bias histogram을 그린다.
   - TODO 4: `DoubleDqnAgent`에 선택·평가가 분리된 target, update와 bias 평가를 구현한다.
   - TODO 5: `config()["selection"] == "online"`과 agent 기반 history·parameter 변화·Double-Q metric을 함께 검증한다.
8. **5분 마무리:** 핵심은 두 network 자체보다 선택·평가 역할의 분리다. 두 estimator가 완전히 같은 noise를 가지면 이득이 사라진다. **자가점검:** (a) target network가 action을 선택하는가? (b) 독립 noise toy에서 Double 평균이 0에 가까운 이유는? (c) Double DQN도 과소추정할 수 있는가?

## 03. Dueling DQN (2016)

원문: [Dueling Network Architectures for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/wangf16.html)

1. **한 문장 요약:** Q를 상태가치 $V(s)$와 action별 advantage $A(s,a)$ stream으로 나눠 action 차이가 작아도 공통 상태가치를 효율적으로 학습한다.
2. **배경과 이전 한계:** 많은 상태에서는 어떤 action을 택해도 결과가 비슷하지만 표준 Q-network는 각 action 값을 별도로 출력한다. 단순 $Q=V+A$는 상수 이동 때문에 분해가 식별되지 않는다.
3. **읽기 전 기초지식:** DQN, $Q/V/A$ 정의, shared feature backbone, identifiability, mean-centering.
4. **핵심 아이디어:**
   - shared representation 뒤에 scalar V stream과 action-vector A stream을 둔다.
   - $A$의 max 또는 mean을 빼 분해를 식별 가능하게 만든다.
   - 논문 실험은 mean-centered aggregation을 사용한다.
5. **꼭 볼 원문 위치:** `§3, Eq. (7)` ↔ naive $V+A$; `§3, Eq. (8)–(9)` ↔ max/mean centering; `§4.1, corridor policy-evaluation experiment` ↔ action redundancy 해석.
6. **수식·알고리즘 직관:** $V$는 “이 상태가 전반적으로 얼마나 좋은가”, centered $A$는 “이 action이 상태 평균보다 얼마나 나은가”를 담당한다. 평균 $A=0$이면 $V$가 action 평균 Q로 정해진다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/03_dueling_dqn.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/03_dueling_dqn.ipynb)
   - TODO 1: shared body와 V/A heads, mean aggregation을 만든다.
   - TODO 2: `q_star` table을 작은 network로 fit한다.
   - TODO 3: centered A의 평균 0과 Q MSE를 검증한다.
   - TODO 4: `DuelingValueAgent`에 V/A heads, centered aggregation, loss·update·evaluate를 구현한다.
   - TODO 5: `config()["centering"] == "mean"`과 agent 기반 history·parameter 변화·Q metric을 함께 검증한다.
8. **5분 마무리:** dueling은 Bellman target이 아니라 Q-network architecture의 변경이다. supervised `q_star` fit은 실제 replay 학습의 성능 이득을 증명하지 않는다. **자가점검:** (a) naive 분해의 상수 자유도는? (b) mean과 max aggregation의 차이는? (c) action이 모두 중요한 환경에서도 반드시 이득인가?

## 04. Prioritized Experience Replay (2016)

원문: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

1. **한 문장 요약:** TD error가 큰 transition을 더 자주 replay하고 importance sampling으로 생긴 편향을 보정한다.
2. **배경과 이전 한계:** uniform replay는 이미 잘 학습된 쉬운 transition과 드물고 놀라운 transition에 같은 확률을 준다. 우선순위 sampling은 효율적이지만 학습 분포를 바꿔 bias를 만든다.
3. **읽기 전 기초지식:** replay buffer, TD error, categorical sampling, importance sampling, bias–variance trade-off, sum-tree 개념.
4. **핵심 아이디어:**
   - priority를 $p_i=|\delta_i|+\epsilon$으로 둔다.
   - $P(i)=p_i^\alpha/\sum_kp_k^\alpha$로 proportional replay한다.
   - $w_i=(N P(i))^{-\beta}$를 정규화해 weighted TD loss를 만든다.
   - $\alpha$는 prioritization 강도, $\beta$는 correction 강도를 조절한다.
5. **꼭 볼 원문 위치:** `§3.2` ↔ TD-error priority; `§3.3, Eq. (1)` ↔ proportional probabilities; `§3.4 and Algorithm 1` ↔ importance weights와 prioritized Double DQN.
6. **수식·알고리즘 직관:** 현재 모델을 가장 놀라게 하는 사례를 자주 복습하되, 자주 뽑혔다는 이유만으로 gradient 영향이 과해지지 않게 역확률 weight를 곱한다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/04_prioritized_replay.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/04_prioritized_replay.ipynb)
   - TODO 1: $P(i)$와 max-normalized IS weight를 구현한다.
   - TODO 2: priority 갱신과 weighted Q-table TD loss를 수행한다.
   - TODO 3: 기대 확률과 empirical sampling 빈도의 상관을 확인한다.
   - TODO 4: `PrioritizedReplayAgent`에 확률·IS weight, weighted TD update와 sampling 평가를 구현한다.
   - TODO 5: `config()["replay_size"] == len(rows)`와 agent 기반 history·priority 변화·TD metric을 함께 검증한다.
8. **5분 마무리:** PER은 replay의 “무엇을 얼마나 자주 볼지”를 바꾼다. 작은 buffer의 exact categorical sampling은 대규모 sum-tree 효율을 재현하지 않는다. **자가점검:** (a) $\alpha=0$은 무엇인가? (b) $\beta=1$의 의미는? (c) $\epsilon$이 없으면 어떤 transition이 영원히 사라질 수 있는가?

## 05. A3C (2016)

원문: [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html)

1. **한 문장 요약:** 여러 actor-learner가 서로 다른 환경을 비동기 탐색하며 공유 actor–critic을 n-step advantage와 entropy로 갱신한다.
2. **배경과 이전 한계:** DQN은 correlation을 줄이기 위해 큰 replay memory를 썼고, on-policy deep RL은 표본 상관과 느린 학습에 취약했다. 여러 환경의 decorrelated experience를 병렬로 얻는 대안이 필요했다.
3. **읽기 전 기초지식:** actor–critic, n-step return, bootstrapping, advantage, entropy regularization, asynchronous shared parameters.
4. **핵심 아이디어:**
   - 서로 다른 seed/environment instance의 worker가 trajectory를 수집한다.
   - $R_t-V(s_t)$가 actor의 advantage와 critic의 regression error가 된다.
   - entropy bonus가 premature deterministic policy를 막는다.
   - n-step return은 Monte Carlo와 one-step TD 사이를 잇는다.
5. **꼭 볼 원문 위치:** `§3, n-step Q-learning paragraph` ↔ bootstrapped return; `§4, ‘Asynchronous advantage actor-critic’` ↔ actor/value losses; `§4 entropy equation; Supplementary Algorithm S2` ↔ A3C loop.
6. **수식·알고리즘 직관:** 각 worker는 조금 다른 세계에서 gradient를 계산하므로 한 trajectory의 편향과 상관이 평균화된다. critic은 baseline을 제공하고 actor는 그보다 좋았던 action의 확률을 높인다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/05_a3c.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/05_a3c.ipynb)
   - TODO 1: 마지막 bootstrap을 포함한 n-step return을 계산한다.
   - TODO 2: 네 sequential seed worker가 shared actor–critic을 갱신한다.
   - TODO 3: greedy return과 optimal-action agreement를 평가한다.
   - TODO 4: `SharedActorCriticAgent`에 actor/critic, n-step loss, worker update와 평가를 구현한다.
   - TODO 5: worker mode와 agent 기반 history·parameter 변화·actor-critic metric을 함께 검증한다.
8. **5분 마무리:** A3C의 핵심은 shared actor–critic, decorrelated workers, n-step/entropy다. 결정성을 위한 sequential worker는 실제 stale gradient와 thread race를 재현하지 않는다. **자가점검:** (a) terminal이면 bootstrap이 왜 0인가? (b) entropy 부호는 왜 policy loss에서 음수인가? (c) A3C와 replay DQN의 decorrelation 방식 차이는?

## 06. DDPG (2016)

원문: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)

1. **한 문장 요약:** deterministic actor와 action-value critic을 replay/target network와 결합해 연속 action에서 off-policy actor–critic을 수행한다.
2. **배경과 이전 한계:** discrete Q-learning의 `argmax`는 고차원 연속 action에서 직접 계산하기 어렵다. stochastic policy gradient는 환경 표본 외에도 action sampling variance가 있다.
3. **읽기 전 기초지식:** deterministic policy gradient, actor–critic, continuous action, replay, exploration noise, soft target update.
4. **핵심 아이디어:**
   - actor $\mu(s)$가 critic $Q(s,a)$를 최대화하는 action을 직접 출력한다.
   - critic은 target actor/critic으로 만든 Bellman target을 회귀한다.
   - actor gradient는 $Q(s,\mu(s))$에서 action을 거쳐 actor로 흐른다.
   - target parameter를 $\theta'\leftarrow\tau\theta+(1-\tau)\theta'$로 천천히 옮긴다.
5. **꼭 볼 원문 위치:** `§2, Eq. (3)` ↔ deterministic Bellman equation; `§3, Eq. (4)–(6)` ↔ critic loss와 policy gradient; `§3, Algorithm 1` ↔ replay, noise, soft targets.
6. **수식·알고리즘 직관:** critic이 상태–action 지형을 배우고 actor는 매 상태에서 그 지형의 높은 쪽으로 action을 이동한다. behavior exploration에는 noise가 필요하지만 actor update 자체는 현재 deterministic action을 쓴다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/06_ddpg.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/06_ddpg.ipynb)
   - TODO 1: tanh actor, state-action critic, soft update를 구현한다.
   - TODO 2: `offline_action_noise` continuous replay에서 critic/actor를 교대 학습한다.
   - TODO 3: target action MSE와 online–target gap을 확인한다.
   - TODO 4: `DeterministicPolicyAgent`에 actor/critic, 두 loss, soft target update와 평가를 구현한다.
   - TODO 5: `config()["tau"] == 0.02`와 agent 기반 history·parameter 변화·target metric을 함께 검증한다.
8. **5분 마무리:** DDPG는 continuous argmax를 actor network로 amortize한다. 실습은 one-step proxy reward이며 논문의 multi-step dynamics와 OU exploration을 재현하지 않는다. **자가점검:** (a) actor loss에 replay action을 쓰지 않는 이유는? (b) soft update의 $\tau$가 작을수록 무엇이 변하는가? (c) critic 오차가 actor에 어떻게 전파되는가?

## 07. PPO (2017)

원문: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

1. **한 문장 요약:** old/new policy probability ratio를 clip한 surrogate로 한 rollout batch를 여러 epoch 안정적으로 재사용한다.
2. **배경과 이전 한계:** vanilla policy gradient의 큰 update는 성능을 급격히 망칠 수 있고, TRPO의 trust-region constraint와 second-order optimization은 구현이 복잡하다.
3. **읽기 전 기초지식:** policy gradient, advantage estimation, importance ratio, surrogate objective, entropy/value loss, on-policy rollout.
4. **핵심 아이디어:**
   - $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$를 계산한다.
   - $\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)$로 과도한 개선 추정을 잘라낸다.
   - old log-prob를 고정한 채 같은 batch를 소수 epoch 반복한다.
   - 실제 actor–critic에서는 policy/value/entropy 항을 함께 최적화한다.
5. **꼭 볼 원문 위치:** `§2, Eq. (1) and Eq. (6)` ↔ PG/CPI surrogate; `§3, Eq. (7)` ↔ clipped objective; `§5, Eq. (9), Algorithm 1` ↔ combined multi-epoch update.
6. **수식·알고리즘 직관:** ratio가 1에서 너무 멀어져 얻는 “추가 보상”을 제한해 old policy 주변에 머물도록 유도한다. hard constraint나 항상 단조 개선을 보장하는 장치는 아니다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/07_ppo.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/07_ppo.ipynb)
   - TODO 1: clipped surrogate와 clip fraction을 구현한다.
   - TODO 2: old batch마다 4 epoch update한다.
   - TODO 3: optimal agreement, entropy, clip fraction을 함께 본다.
   - TODO 4: `ProximalPolicyAgent`에 actor/value, clipped objective, multi-epoch update와 평가를 구현한다.
   - TODO 5: `config()["clip_epsilon"] == 0.2`와 agent 기반 history·parameter 변화·PPO metric을 함께 검증한다.
8. **5분 마무리:** PPO는 간단한 first-order 근사로 policy 변화의 이득을 제한한다. 실습의 exact `q_star` advantage는 GAE나 rollout noise를 대체한다. **자가점검:** (a) old log-prob를 update 중 다시 계산하면 왜 안 되는가? (b) 음수 advantage에서 min의 효과는? (c) clip이 KL bound를 보장하는가?

## 08. Soft Actor-Critic (2018)

원문: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html)

1. **한 문장 요약:** expected reward와 policy entropy를 함께 최대화하는 off-policy stochastic actor–critic으로 연속제어의 안정성과 탐색을 개선한다.
2. **배경과 이전 한계:** DDPG류 deterministic actor는 hyperparameter와 Q error에 민감하고 탐색 noise를 별도로 설계해야 했다. entropy regularization을 일관된 soft policy iteration으로 정리할 필요가 있었다.
3. **읽기 전 기초지식:** maximum-entropy objective, soft Bellman backup, KL projection, Gaussian reparameterization, tanh change-of-variables, off-policy replay.
4. **핵심 아이디어:**
   - reward 합에 $\alpha\mathcal H(\pi(\cdot|s))$를 더한다.
   - soft value는 Q뿐 아니라 미래 policy entropy를 포함한다.
   - actor는 reparameterized sample로 $\alpha\log\pi(a|s)-Q(s,a)$를 최소화한다.
   - tanh squash 뒤 log-probability의 Jacobian correction이 필요하다.
5. **꼭 볼 원문 위치:** `§3.2, Eq. (1)` ↔ maximum entropy; `§4.1, Eq. (2)–(3), Eq. (7)–(8)` ↔ soft backup/Q loss; `§4.2, Eq. (10)–(12), Algorithm 1` ↔ KL/reparameterized actor.
6. **수식·알고리즘 직관:** actor는 Q가 높은 action을 선호하지만 $\alpha\log\pi$ 항 때문에 너무 일찍 한 action으로 붕괴하지 않는다. $\alpha$는 reward 최적화와 다양성의 환율이다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/08_sac.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/08_sac.ipynb)
   - TODO 1: tanh Gaussian sample과 corrected log-prob를 구현한다.
   - TODO 2: uniform-action replay에서 critic과 soft actor를 교대 학습한다.
   - TODO 3: mean-action MSE와 stochastic standard deviation을 평가한다.
   - TODO 4: `SoftActorCriticAgent`에 stochastic actor/critic, entropy loss, update와 평가를 구현한다.
   - TODO 5: `config()["temperature"] == alpha`와 agent 기반 history·parameter 변화·entropy metric을 함께 검증한다.
8. **5분 마무리:** SAC는 soft Bellman iteration과 stochastic reparameterized actor의 결합이다. one-step contextual 실습은 multi-step soft target 및 twin-Q 구현을 축소한다. **자가점검:** (a) tanh Jacobian correction이 필요한 이유는? (b) $\alpha$가 0이면 어떤 계열에 가까워지는가? (c) 높은 entropy가 항상 높은 task return을 뜻하는가?

## 09. AlphaZero / MCTS (2017)

원문: [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

1. **한 문장 요약:** 하나의 policy/value network와 MCTS를 self-play로 반복 개선해 인간 기보 없이 여러 보드게임을 학습한다.
2. **배경과 이전 한계:** AlphaGo 계열은 인간 기보와 game-specific feature/rollout 요소에 의존했다. 큰 action tree에서 pure model-free policy만으로 정확한 장기 계획을 얻기도 어렵다.
3. **읽기 전 기초지식:** MCTS, visit count, UCB/PUCT, policy/value network, self-play, cross-entropy, terminal outcome.
4. **핵심 아이디어:**
   - network가 state에서 prior policy $p$와 value $v$를 동시에 낸다.
   - PUCT가 $Q$, visit $N$, prior $P$를 결합해 search action을 선택한다.
   - root visit count를 정규화한 $\pi$가 raw policy보다 강한 training target이 된다.
   - loss는 outcome value MSE, visit-policy cross-entropy, L2를 결합한다.
5. **꼭 볼 원문 위치:** `Main text, network paragraph and Eq. (1)` ↔ $(p,v)=f_\theta(s)$ 및 joint loss; `Main text, MCTS paragraph` ↔ PUCT/visit counts; `Main text, self-play paragraph` ↔ $a_t\sim\pi_t$, terminal $z$.
6. **수식·알고리즘 직관:** network는 search를 빠르게 시작할 prior와 leaf evaluation을 주고, search는 여러 가상 미래를 모아 더 강한 $\pi$를 만든다. 그 $\pi$를 다시 network에 distill하는 policy-iteration loop다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/reinforcement_learning/exercises/09_alphazero.ipynb) · [정답본](../../notebooks/field_reproductions/reinforcement_learning/solutions/09_alphazero.ipynb)
   - TODO 1: PUCT score를 구현한다.
   - TODO 2: exact local transition과 leaf oracle로 root visit policy를 만든다.
   - TODO 3: search $\pi$와 normalized value target으로 Eq. (1) loss를 학습한다.
   - TODO 4: `AlphaZeroSearchAgent`에 policy/value network, PUCT search loss, update와 평가를 구현한다.
   - TODO 5: action count와 agent 기반 history·parameter 변화·policy/value metric을 함께 검증한다.
8. **5분 마무리:** AlphaZero는 search와 network가 서로를 개선하는 expert-iteration 구조다. 실습은 `q_star` leaf oracle을 사용하며 진짜 self-play tree, backup, game symmetry, 대규모 residual network를 재현하지 않는다. **자가점검:** (a) $P$, $Q$, $N$의 역할은? (b) visit $\pi$가 raw $p$보다 강한 이유는? (c) 실습의 oracle 사용이 원 논문보다 쉬운 지점은?

---

## 2023–2026 후속 읽기 5편

### 선정 기준

- 2023년 이후 ICML·ICLR·NeurIPS 본 학회 공식 기록에서 채택과 venue/year를 확인할 수 있는 논문만 넣었다.
- 기존 10편 다음에 읽었을 때 **sample-efficient value learning, offline-to-online, world-model planning, scalable value objective, code-writing agent**로 시야가 넓어지도록 골랐다.
- benchmark 최고점만으로 고르지 않았고, 서로 다른 문제 설정과 재현 시 배울 개념의 다양성을 우선했다.
- 추천은 절대적 순위가 아니며 각 논문의 실험 조건 밖 성능을 보장하지 않는다.

1. **Bigger, Better, Faster: Human-level Atari with human-level efficiency — ICML 2023**  
   [PMLR 공식 프로시딩](https://proceedings.mlr.press/v202/schwarzer23a.html) · **선정 이유:** DQN 이후 value-based agent를 더 큰 network와 sample-efficient design으로 확장하며 Atari 100K에서 구성요소 ablation을 제공한다. **선행지식:** DQN/Double/Dueling/PER, distributional value learning, representation learning, evaluation IQM.

2. **Efficient Online Reinforcement Learning with Offline Data — ICML 2023**  
   [PMLR 공식 프로시딩](https://proceedings.mlr.press/v202/ball23a.html) · **선정 이유:** 복잡한 새 알고리즘보다 off-policy RL이 offline prior data와 online interaction을 함께 잘 쓰기 위한 실무적 조건을 체계적으로 분석한다. **선행지식:** SAC, replay mixture, offline RL의 distribution shift, ensemble critic, update-to-data ratio.

3. **TD-MPC2: Scalable, Robust World Models for Continuous Control — ICLR 2024 Spotlight**  
   [ICLR 공식 OpenReview](https://openreview.net/forum?id=Oxh5CstDJU) · **선정 이유:** latent world model, temporal-difference learning, model-predictive control을 결합하고 multi-task/scale 관점까지 확장해 SAC·AlphaZero 이후 model-based planning을 잇는다. **선행지식:** DDPG/SAC, latent dynamics, TD target, planning horizon, CEM/MPC.

4. **Stop Regressing: Training Value Functions via Classification for Scalable Deep RL — ICML 2024**  
   [PMLR 공식 프로시딩](https://proceedings.mlr.press/v235/farebrother24a.html) · **선정 이유:** bootstrapped scalar target을 MSE로 회귀하는 관행을 categorical cross-entropy로 바꿔 noisy/non-stationary value learning과 network scaling을 다시 검토한다. **선행지식:** DQN/actor–critic value loss, distributional RL, two-hot/categorical targets, cross-entropy, bootstrapping.

5. **WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment — NeurIPS 2024**  
   [NeurIPS 공식 프로시딩](https://proceedings.neurips.cc/paper_files/paper/2024/hash/820c61a0cd419163ccbd2c33b268816e-Abstract-Conference.html) · **선정 이유:** 학습된 neural dynamics 대신 interaction을 설명하는 Python world model과 planner를 결합해 “에이전트가 무엇을 학습해 계획하는가”를 상징적 관점에서 비교할 수 있다. **선행지식:** model-based RL, optimistic exploration, planning, Python program synthesis, LLM prompting과 verification.

### 검증 메모

공식 기록은 순서대로 `Proceedings of the 40th ICML (2023)`, 같은 ICML 2023 volume, `ICLR 2024 Spotlight`, `Proceedings of the 41st ICML (2024)`, `NeurIPS 2024 Main Conference Track`을 명시한다. arXiv 제목만 보고 venue를 추정하지 않았다.
