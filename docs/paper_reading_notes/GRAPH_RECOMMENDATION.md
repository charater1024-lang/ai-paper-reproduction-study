# 그래프 학습·추천 시스템 논문 독해 노트

> **AI 보조 작성 안내:** 이 한국어 요약은 공식 번역이 아닙니다. 내용과 인용의 최종 기준은
> 연결된 원문입니다. [작성 원칙과 한계](AI_ASSISTED_NOTICE.md) · [70편 목차](README.md)

이 문서는 원 논문을 읽으면서 `exercises` 노트북의 빈칸을 먼저 풀고, 막힐 때만 `solutions`를 확인하는 방식으로 설계했다. 모든 실습은 로컬의 `data/field_curriculum/graph_recommendation.npz`를 사용하므로 외부 다운로드가 필요 없다. 논문의 대규모 결과를 그대로 복제하는 것이 아니라, 결과를 낳는 핵심 연산을 CPU에서 짧게 검증한다.

권장 학습법은 **핵심 수식 Task → portfolio class 조립 → contract 검증**이다. 번호가 붙은 TODO에서 논문의 graph 연산과 지표를 먼저 구현하고, 포트폴리오 P1에서 `PortfolioConfig`, 전처리, `forward`·`compute_loss`·`training_step`을 가진 class로 조립한다. P2의 최적화 loop와 P3의 finite history·quality metric 검증까지 끝낸 뒤 자가점검으로 넘어간다.

---

## 00. DeepWalk — Online Learning of Social Representations (KDD 2014)

원문: [DeepWalk](https://arxiv.org/abs/1403.6652)

### ① 한 문장 요약

그래프 위의 짧은 랜덤 워크를 문장으로, 노드를 단어로 바꾸어 Skip-gram을 학습하면 이웃 구조를 보존하는 노드 임베딩을 얻을 수 있다.

### ② 배경과 이전 접근의 한계

전통적인 spectral embedding은 그래프 전체 행렬의 분해가 필요해 큰 그래프나 계속 변하는 그래프에 적용하기 어렵다. 당시 언어 모델의 Skip-gram은 국소 문맥 샘플만으로 온라인 학습할 수 있었고, DeepWalk는 랜덤 워크의 노드 출현 분포가 자연어의 단어 출현 분포와 닮았다는 관찰을 연결했다.

### ③ 선수지식

- 무방향 그래프, 인접 행렬, 이웃 집합
- Markov 성질과 균등 랜덤 워크
- Skip-gram과 softmax cross-entropy
- 코사인 유사도와 임베딩

### ④ 핵심 아이디어

- 각 노드에서 여러 번 고정 길이 랜덤 워크를 시작해 노드 시퀀스 corpus를 만든다.
- 시퀀스의 중심 노드 주변 window를 문맥 노드로 간주한다.
- 중심 임베딩으로 문맥 노드의 조건부 확률을 높이는 Skip-gram 목적을 최적화한다.
- 학습이 국소 샘플에만 의존하므로 그래프가 커져도 streaming·병렬화하기 쉽다.

### ⑤ 꼭 볼 원문 위치

- **§3.1**: 랜덤 워크가 국소 구조와 연결성을 포착하는 이유.
- **§4.2, Algorithm 1**: 각 정점에서 `γ`회, 길이 `t`의 워크를 만드는 절차.
- **§4.2.1, Algorithm 2**: walk의 중심–문맥 쌍과 window 처리.
- **§3.3, Eq. (2)**: 중심 노드로부터 문맥 노드 확률을 최대화하는 Skip-gram 목적.
- **Figure 3**: 랜덤 워크 생성부터 표현 학습까지의 전체 파이프라인.

### ⑥ 수식·알고리즘 직관

워크 `v1, …, vt`에서 `vi`의 앞뒤 `w`개 노드는 실제 edge만이 아니라 여러 hop의 공출현 정보를 준다. `P(vj | vi)`를 높이면 자주 같은 워크 문맥에 나타나는 노드 벡터가 가까워진다. 원문은 hierarchical softmax로 큰 노드 어휘를 처리하지만, 실습은 작은 그래프이므로 모든 노드를 한 번에 분류하는 full softmax를 써서 같은 목적을 투명하게 본다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/00_deepwalk.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/00_deepwalk.ipynb)
- **TODO 00-1 ↔ Algorithm 1**: seed가 고정된 균등 랜덤 워크를 구현하고 모든 연속 노드가 edge로 연결됐는지 검사한다.
- **TODO 00-2 ↔ Algorithm 2**: window 안의 `(center, context)` 쌍을 만들고 위치가 같은 self-pair만 제외한다.
- **TODO 00-3 ↔ Eq. (2)**: full-softmax Skip-gram을 학습해 초기 대비 cross-entropy 감소를 확인한다.
- **TODO 00-4 ↔ Figure 3의 표현 평가**: 같은 label 쌍과 다른 label 쌍의 평균 cosine을 비교한다.
- **포트폴리오 P1 → P2 → P3**: `DeepwalkPortfolioModel`의 graph 전처리·Skip-gram forward/loss/update를 완성하고, `fit_portfolio_model` 학습 뒤 class cosine gap·finite-history 계약을 평가한다.

### ⑧ 5분 정리

- **5분 요약:** “워크로 문장을 만들고 Skip-gram으로 공출현을 압축한다”가 전부다. 워크 정책은 어떤 구조를 유사하다고 볼지 결정하고, window와 목적함수는 그 통계를 벡터에 담는다.
- **한계:** 균등 워크의 구조 편향을 조절하기 어렵고, 속성 정보와 edge type을 직접 쓰지 않으며, 관측되지 않은 새 노드에는 별도 재학습이 필요하다.
- **자가점검:**
  1. 인접 노드가 아니어도 positive 문맥 쌍이 될 수 있는 이유는 무엇인가?
  2. window를 크게 하면 포착하는 구조와 노이즈가 어떻게 변하는가?
  3. full softmax와 hierarchical softmax의 계산량 차이는 무엇인가?

---

## 01. node2vec — Scalable Feature Learning for Networks (KDD 2016)

원문: [node2vec](https://arxiv.org/abs/1607.00653)

### ① 한 문장 요약

직전 노드와의 거리를 보는 2차 랜덤 워크에 `p`, `q` 편향을 주어 BFS형 동질성 탐색과 DFS형 구조적 역할 탐색을 연속적으로 조절한다.

### ② 배경과 이전 접근의 한계

DeepWalk의 균등 워크는 어떤 종류의 네트워크 유사성을 보존할지 선택할 수 없다. 가까운 community의 노드를 묶는 **homophily**와 서로 멀어도 비슷한 역할을 하는 허브 등을 묶는 **structural equivalence**는 서로 다른 탐색이 필요한데, 하나의 고정 워크는 두 목적을 제어하지 못했다.

### ③ 선수지식

- DeepWalk와 Skip-gram
- 그래프 최단 거리와 BFS/DFS
- 조건부 확률과 정규화
- 1차·2차 Markov walk

### ④ 핵심 아이디어

- 직전 노드 `t`, 현재 노드 `v`, 후보 `x`의 관계를 사용해 다음 전이를 정한다.
- return parameter `p`는 방금 지나온 edge로 되돌아가는 정도를 조절한다.
- in–out parameter `q`는 현재 지역에 머물지, 바깥으로 뻗을지 조절한다.
- 편향 walk로 corpus를 만든 뒤에는 DeepWalk와 같은 Skip-gram 최적화를 쓴다.

### ⑤ 꼭 볼 원문 위치

- **§3.2.2, Eq. (2)**: 거리 `d_tx ∈ {0,1,2}`에 따른 `α_pq(t,x)`의 세 경우.
- **Figure 2**: 직전 노드 `t`를 조건으로 후보 전이 확률이 달라지는 그림.
- **§3.2.3, Algorithm 1**: 2차 walk를 반복해 node representation을 학습하는 전체 절차.
- **§3.2.2의 BFS/DFS 설명**: `q`가 지역적·외향적 탐색에 미치는 영향.

### ⑥ 수식·알고리즘 직관

후보 `x`가 `t`이면 가중치 `1/p`, `t`와 인접하면 `1`, 두 hop 떨어졌으면 `1/q`를 준 뒤 후보들 사이에서 정규화한다. 따라서 `p`가 작으면 되돌아가며 국소를 재확인하고, `q>1`이면 먼 후보가 억제되어 BFS처럼, `q<1`이면 먼 후보가 강조되어 DFS처럼 움직인다. “2차”란 다음 상태가 현재 노드뿐 아니라 직전 노드에도 의존한다는 뜻이다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/01_node2vec.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/01_node2vec.ipynb)
- **TODO 01-1 ↔ Eq. (2)**: return/in/out 후보 가중치를 만들고 합이 1인 전이 확률로 정규화한다.
- **TODO 01-2 ↔ Algorithm 1**: 첫 step은 균등, 이후 step은 `α_pq`를 쓰는 재현 가능한 biased walk를 만든다.
- **TODO 01-3 ↔ Figure 2**: `q=0.5`와 `q=2`의 시작점 거리·고유 노드 수 분포를 비교해 DFS/BFS 효과를 그린다.
- **포트폴리오 P1 → P2 → P3**: `Node2VecPortfolioModel`의 biased-walk 전처리·forward/loss/update를 완성하고, fit loop 뒤 class cosine gap·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** node2vec은 새로운 임베딩 loss라기보다, 어떤 positive 문맥을 수집할지를 `p`, `q`로 학습자에게 맡긴 DeepWalk의 탐색 확장이다.
- **한계:** 좋은 `p`, `q`는 task·그래프마다 달라 탐색 비용이 들고, 정적·동질 edge를 가정하며, 여전히 새 노드에 대한 직접적 함수가 없다.
- **자가점검:**
  1. `p`와 `q`는 각각 어떤 전이를 바꾸는가?
  2. `q<1`이 구조적 역할을 찾는 데 유리할 수 있는 이유는 무엇인가?
  3. node2vec의 최종 loss가 DeepWalk와 본질적으로 같은 이유는 무엇인가?

---

## 02. GCN — Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)

원문: [GCN](https://openreview.net/forum?id=SJU4ayYgl)

### ① 한 문장 요약

자기 자신을 포함해 정규화한 이웃 특징을 선형 변환·집계하는 간단한 층을 쌓아, 소수 label만으로 그래프 전체 노드를 반지도 분류한다.

### ② 배경과 이전 접근의 한계

Spectral graph convolution은 Laplacian 고유분해와 그래프별 basis가 필요했고, 고차 다항 필터는 연산과 파라미터가 많았다. 논문은 Chebyshev 근사를 1차로 단순화하고 안정적인 재정규화 trick을 적용해 edge 수에 비례하는 message passing 형태로 만들었다.

### ③ 선수지식

- 인접 행렬, degree 행렬, graph Laplacian
- 행렬 곱과 선형 층
- 반지도 node classification과 mask
- softmax cross-entropy
- 선택 사항: spectral convolution·Chebyshev 다항식

### ④ 핵심 아이디어

- 인접 행렬에 identity를 더해 자기 특징도 메시지에 포함한다.
- `D̃^{-1/2} Ã D̃^{-1/2}` 대칭 정규화로 degree가 큰 노드의 영향 폭주를 막는다.
- 같은 weight를 모든 node·edge에 공유해 sparse 연산으로 확장한다.
- label이 있는 train node에서만 loss를 계산하되, 전파에는 전체 그래프를 사용한다.

### ⑤ 꼭 볼 원문 위치

- **§2, Eq. (2)**: 층별 전파 규칙의 최종 형태.
- **§2.2, Eq. (8)**: `I + D^{-1/2} A D^{-1/2}`를 재정규화한 안정화 과정.
- **§3.1, Eq. (9)**: 두 층 GCN의 node-classification 모델.
- **§3.1, Eq. (10)**: label이 있는 노드에만 적용하는 cross-entropy.

### ⑥ 수식·알고리즘 직관

`H^{l+1}=σ(S H^l W^l)`, `S=D̃^{-1/2}ÃD̃^{-1/2}`로 읽는다. `H W`가 각 노드의 메시지를 만들고 `S`가 이웃 메시지를 degree 보정 평균하며, `σ`가 비선형성을 더한다. 두 층이면 각 노드는 최대 두 hop의 정보를 보게 된다. train mask는 loss 위치만 가릴 뿐 메시지 전달 그래프를 자르지 않는다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/02_gcn.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/02_gcn.ipynb)
- **TODO 02-1 ↔ Eq. (8)**: self-loop을 추가하고 대칭 정규화 행렬을 계산한다.
- **TODO 02-2 ↔ Eq. (2), (9)**: `GCNLayer`와 2층 `MiniGCN`을 구성하고 출력 shape을 검증한다.
- **TODO 02-3 ↔ Eq. (10)**: train mask 위치에서만 loss를 학습하고 test accuracy·loss curve를 확인한다.
- **포트폴리오 P1 → P2 → P3**: `GcnPortfolioModel`의 정규화·GCN forward/masked loss/update를 완성하고, fit loop 뒤 test accuracy·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** GCN은 “degree로 보정된 이웃 평균 → 공유 선형 변환”을 반복하는 저주파 graph filter다.
- **한계:** 깊어지면 표현이 비슷해지는 over-smoothing, 먼 거리 정보의 over-squashing, 전체 그래프를 전제로 한 transductive 학습과 동질성 편향이 있다.
- **자가점검:**
  1. self-loop이 없으면 어떤 정보가 한 층 뒤 사라질 수 있는가?
  2. 대칭 정규화가 high-degree node의 영향력을 어떻게 바꾸는가?
  3. test node 특징이 training 전파에 쓰이는 것이 label leakage가 아닌 이유는 무엇인가?

---

## 03. GraphSAGE — Inductive Representation Learning on Large Graphs (NeurIPS 2017)

원문: [GraphSAGE](https://papers.nips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)

### ① 한 문장 요약

노드별 embedding table 대신 “이웃을 sample하고 aggregate하는 함수”를 학습해, 학습 때 보지 못한 노드에도 특징으로부터 임베딩을 생성한다.

### ② 배경과 이전 접근의 한계

DeepWalk·node2vec과 전통 transductive embedding은 각 node ID의 벡터를 직접 최적화하므로 새 노드가 생기면 임베딩이 없다. full-neighborhood GCN은 degree가 크거나 그래프가 거대할 때 한 mini-batch의 계산량이 폭증한다. GraphSAGE는 고정 개수의 이웃 sampling과 공유 aggregator로 두 문제를 함께 다룬다.

### ③ 선수지식

- GCN·message passing
- 집합 함수와 permutation invariance
- 이웃 sampling과 mini-batch
- inductive 대 transductive 학습
- 평균·pooling·LSTM aggregator

### ④ 핵심 아이디어

- 각 층에서 고정 수의 이웃을 sample해 계산 graph의 크기를 제한한다.
- 이웃 표현을 mean, pooling, LSTM 등의 aggregator로 하나의 벡터로 만든다.
- 자신의 이전 표현과 이웃 aggregate를 결합해 다음 표현을 계산한다.
- node ID가 아닌 feature-to-embedding 함수를 공유하므로 새 노드에도 적용한다.

### ⑤ 꼭 볼 원문 위치

- **§3.1, Algorithm 1 lines 1–7**: 층별 sampling·aggregation·concatenation 절차.
- **§3.1의 neighborhood 정의**: 고정 크기 이웃 표본이 계산 복잡도를 제한하는 방식.
- **§3.3, Eq. (2)**: mean aggregator와 GCN 변형의 관계.
- **§2 및 §3.1**: unseen node에 일반화하는 inductive setting.

### ⑥ 수식·알고리즘 직관

먼저 `h_N(v)^k = AGGREGATE_k({h_u^{k-1}:u∈N(v)})`, 이어 `h_v^k=σ(W^k·CONCAT(h_v^{k-1},h_N(v)^k))`로 읽는다. 평균은 이웃 순서에 무관하며, 표본 수 `S_k`를 고정하면 깊이 `K`의 노드당 계산은 대략 `∏S_k`로 그래프 전체 크기와 분리된다. 단, 표본 때문에 같은 노드의 출력에도 분산이 생긴다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/03_graphsage.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/03_graphsage.ipynb)
- **TODO 03-1 ↔ Algorithm 1**: seed와 fanout을 받는 고정 크기 이웃 sampler를 구현한다.
- **TODO 03-2 ↔ §3.3**: mean aggregate와 자기 표현 결합을 수행하는 `MeanSAGE` 층을 만든다.
- **TODO 03-3 ↔ inductive 평가**: 분류기를 학습하고 unseen/test node 성능 및 이웃 교란 민감도를 본다.
- **포트폴리오 P1 → P2 → P3**: `GraphsagePortfolioModel`의 sampling·mean aggregation·loss/update를 완성하고, fit loop 뒤 test accuracy·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** GraphSAGE의 핵심 산출물은 node 벡터 자체가 아니라 어떤 node feature와 이웃 집합에서도 벡터를 만드는 공유 함수다.
- **한계:** sampling 분산과 fanout의 지수적 팽창이 남고, 단순 mean은 서로 다른 이웃 multiset을 구분하지 못하며, 특징이 빈약한 새 노드에는 inductive 장점이 작다.
- **자가점검:**
  1. GraphSAGE가 새 노드를 처리할 수 있고 DeepWalk는 바로 처리할 수 없는 이유는?
  2. fanout을 줄일 때 bias·variance·속도는 어떻게 바뀌는가?
  3. mean aggregator가 permutation invariant인 이유는 무엇인가?

---

## 04. GAT — Graph Attention Networks (ICLR 2018)

원문: [GAT](https://openreview.net/forum?id=rJXMpikCZ)

### ① 한 문장 요약

모든 이웃을 동일한 규칙으로 평균하지 않고, 각 edge의 두 endpoint 특징으로 attention 계수를 학습해 중요한 이웃 메시지를 더 크게 집계한다.

### ② 배경과 이전 접근의 한계

GCN의 정규화 weight는 degree가 정하면 고정되므로 같은 이웃 집합에서도 task에 따라 중요한 neighbor를 선택할 수 없다. Spectral 방식은 graph-specific 연산과 고유분해 문제도 있다. GAT는 국소 edge에서 self-attention을 계산해 inductive하게 적용 가능하면서 이웃별 가중치를 데이터로 학습한다.

### ③ 선수지식

- GCN과 message passing
- attention·softmax
- LeakyReLU와 선형 변환
- masked 연산과 multi-head attention

### ④ 핵심 아이디어

- 모든 노드에 공유 선형 변환 `W`를 먼저 적용한다.
- edge `(i,j)`마다 변환된 두 특징으로 unnormalized score `e_ij`를 만든다.
- 실제 이웃에만 masked softmax를 적용해 `α_ij`를 계산한다.
- 여러 attention head를 concat하거나 평균해 안정성과 표현력을 높인다.

### ⑤ 꼭 볼 원문 위치

- **§2, Eq. (1)**: 공유 선형 변환 `Wh_i`.
- **§2, Eq. (2)**: endpoint 특징으로 attention score를 만드는 식.
- **§2, Eq. (3)**: 이웃 집합에 한정한 masked softmax.
- **§2, Eq. (4)–(6)**: weighted aggregation과 multi-head 결합.

### ⑥ 수식·알고리즘 직관

`e_ij = LeakyReLU(aᵀ[Wh_i || Wh_j])`, `α_ij=softmax_j(e_ij)`, `h'_i=σ(Σ_j α_ij Wh_j)`다. softmax의 분모는 전체 노드가 아니라 `i`의 이웃만 포함하므로 각 행의 edge weight 합은 1이다. attention 값은 설명의 단서가 될 수 있지만, 그것만으로 인과적 중요도라고 단정할 수는 없다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/04_gat.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/04_gat.ipynb)
- **TODO 04-1 ↔ Eq. (1)–(3)**: edge score와 masked row-softmax를 구현하고 비-edge의 weight가 0인지 검사한다.
- **TODO 04-2 ↔ Eq. (4)**: attention-weighted 이웃 합을 수행하는 `GATLayer`를 만든다.
- **TODO 04-3 ↔ §2 평가 직관**: 작은 분류기를 학습하고 accuracy와 attention heatmap을 함께 본다.
- **포트폴리오 P1 → P2 → P3**: `GatPortfolioModel`의 masked attention·forward/loss/update를 완성하고, fit loop 뒤 test accuracy·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** GAT은 GCN의 고정 degree weight를 feature-dependent edge weight로 바꾼 message-passing 층이다.
- **한계:** 모든 edge score를 계산해야 하고 고차수 그래프에서 메모리 부담이 크며, heterophily·장거리 병목을 자동 해결하지 않고 attention 해석도 신중해야 한다.
- **자가점검:**
  1. masked softmax가 없다면 어떤 노드가 메시지에 섞이는가?
  2. `α_ij`가 일반적으로 `α_ji`와 같지 않은 이유는?
  3. multi-head가 단일 head보다 안정적일 수 있는 이유는 무엇인가?

---

## 05. VGAE — Variational Graph Auto-Encoders (NIPS Workshop 2016)

원문: [VGAE](https://arxiv.org/abs/1611.07308)

### ① 한 문장 요약

GCN이 각 노드의 잠재 Gaussian 분포를 추론하고, 잠재 벡터 내적으로 edge를 복원하도록 변분 하한을 최적화하는 확률적 graph autoencoder다.

### ② 배경과 이전 접근의 한계

결정론적 graph embedding은 한 점 벡터만 내놓아 표현의 불확실성을 드러내지 않고, 단순 matrix factorization은 node feature와 국소 graph 구조를 함께 쓰기 어렵다. VGAE는 VAE의 확률적 latent와 GCN의 구조적 encoder를 결합해 label 없이 link prediction 표현을 학습한다.

### ③ 선수지식

- GCN과 인접 행렬 정규화
- VAE, latent Gaussian, KL divergence
- reparameterization trick
- Bernoulli likelihood와 link prediction
- inner-product decoder

### ④ 핵심 아이디어

- 공유 GCN 층 뒤에 평균 `μ`와 로그 표준편차 `log σ`를 내는 두 GCN head를 둔다.
- `z=μ+σ⊙ε`로 미분 가능한 표본을 만든다.
- `σ(z_iᵀz_j)`를 node pair의 edge 확률로 사용한다.
- reconstruction 항과 prior에 대한 KL 항을 합친 ELBO를 최대화한다.

### ⑤ 꼭 볼 원문 위치

- **Eq. (1)**: node별 factorized Gaussian posterior `q(Z|X,A)`와 GCN parameterization.
- **Eq. (2)**: inner-product Bernoulli decoder `p(A_ij=1|z_i,z_j)`.
- **Eq. (3)**: expected reconstruction log-likelihood와 KL로 된 ELBO.
- **§2 및 Figure 1**: 첫 GCN을 공유하고 `μ`, `log σ` head로 갈라지는 encoder.

### ⑥ 수식·알고리즘 직관

복원 항은 관측 edge의 node 벡터 내적을 크게, negative pair의 내적을 작게 만든다. KL은 posterior가 표준정규 prior에서 지나치게 벗어나 memorization하는 것을 막는다. `ε~N(0,I)`를 외부 잡음으로 분리한 reparameterization 덕분에 `μ`, `σ`까지 gradient가 흐른다. 작은 희소 graph에서는 non-edge가 훨씬 많으므로 negative sampling·class weighting이 중요하다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/05_vgae.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/05_vgae.ipynb)
- **TODO 05-1 ↔ Eq. (1)**: self-loop과 대칭 정규화를 만들어 GCN encoder 입력을 준비한다.
- **TODO 05-2 ↔ Eq. (1)**: `μ`, `log σ` head와 reparameterization을 갖는 `VGAEEncoder`를 구현한다.
- **TODO 05-3 ↔ Eq. (2)–(3)**: inner-product decoder, positive/negative reconstruction, KL을 결합해 학습하고 link AUC 성격의 순위 지표를 본다.
- **포트폴리오 P1 → P2 → P3**: `VgaePortfolioModel`의 encoder·reparameterization·negative ELBO/update를 완성하고, fit loop 뒤 edge-score gap·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** VGAE는 “GCN posterior + 내적 edge decoder + VAE ELBO”이며, node embedding을 확률분포로 만든다.
- **한계:** 내적 decoder는 대칭·저차 구조로 제한되고, dense `N×N` 복원은 확장성이 낮으며, KL collapse와 negative sampling 편향이 생길 수 있다.
- **자가점검:**
  1. `log σ`를 직접 예측하는 수치적 이유는 무엇인가?
  2. KL 항을 제거하면 모델이 어떤 식으로 과적합할 수 있는가?
  3. inner-product decoder가 방향성 edge를 표현하기 어려운 이유는?

---

## 06. R-GCN — Modeling Relational Data with Graph Convolutional Networks (ESWC 2018)

원문: [R-GCN](https://link.springer.com/chapter/10.1007/978-3-319-93417-4_38)

### ① 한 문장 요약

관계 종류마다 다른 메시지 변환을 적용하고 관계별 이웃을 정규화해, multi-relational knowledge graph의 entity classification과 link prediction을 수행한다.

### ② 배경과 이전 접근의 한계

기본 GCN은 모든 edge를 동일하게 취급하지만 knowledge graph의 `bornIn`, `worksFor`, `locatedIn`은 의미와 방향이 다르다. 단순히 relation마다 완전한 weight matrix를 두면 relation 수가 많을 때 파라미터가 폭증하고 희귀 relation은 충분히 학습되지 않는다.

### ③ 선수지식

- GCN·message passing
- knowledge graph의 entity, relation, directed triple
- 관계별 adjacency와 degree 정규화
- basis decomposition·block diagonal decomposition
- link prediction과 negative sampling

### ④ 핵심 아이디어

- 관계 `r`별 이웃 `N_i^r`에서 별도 matrix `W_r`로 메시지를 보낸다.
- relation별 normalization `c_{i,r}`로 이웃 수 차이를 보정한다.
- basis 또는 block decomposition으로 많은 `W_r`가 파라미터를 공유한다.
- 역방향 relation과 self-loop를 추가해 정보 흐름 방향을 보완한다.

### ⑤ 꼭 볼 원문 위치

- **§2, Eq. (2)**: 관계별 이웃 합과 self-loop을 포함한 R-GCN update.
- **Eq. (2)의 `c_{i,r}` 설명**: relation-specific normalization 선택.
- **§2, Eq. (3)**: `W_r = Σ_b a_rb V_b` basis decomposition.
- **Figure 2**: edge type마다 다른 변환이 entity 표현에 미치는 구조.
- **§3, Eq. (4)**: DistMult decoder를 붙인 link prediction 설정.

### ⑥ 수식·알고리즘 직관

`h_i^{l+1}=σ(Σ_r Σ_{j∈N_i^r}(1/c_{i,r})W_r^l h_j^l + W_0^l h_i^l)`다. 같은 neighbor라도 relation이 바뀌면 다른 좌표계로 투영된다. basis decomposition은 각 relation matrix를 소수 basis의 선형결합으로 만들어, 희귀 relation이 공통 구조를 빌리게 하는 저차 regularizer 역할을 한다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/06_rgcn.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/06_rgcn.ipynb)
- **TODO 06-1 ↔ Eq. (2)**: relation adjacency를 목적 노드 기준으로 정규화한다.
- **TODO 06-2 ↔ Eq. (3)**: basis와 relation coefficient로 `W_r`를 합성한다.
- **TODO 06-3 ↔ Eq. (2)**: relation-specific 합과 self-loop을 구현한 R-GCN 층을 만든다.
- **TODO 06-4 ↔ Figure 2**: 특정 edge의 relation만 바꾸고 출력 변화량을 측정해 type sensitivity를 확인한다.
- **포트폴리오 P1 → P2 → P3**: `RgcnPortfolioModel`의 relation 전처리·basis message/loss/update를 완성하고, fit loop 뒤 relation-edge gap·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** R-GCN은 GCN 메시지에 relation별 동사를 붙이고, basis 공유로 relation 수에 따른 비용을 억제한다.
- **한계:** relation이 매우 많거나 동적으로 생기면 여전히 비용이 크고, 여러 층에서 모든 relation을 합치며 over-smoothing할 수 있으며, basis 수 선택이 성능을 좌우한다.
- **자가점검:**
  1. self-loop matrix `W_0`가 별도로 필요한 이유는?
  2. basis 수가 1일 때 relation 표현력은 어떻게 제한되는가?
  3. 역관계 edge를 추가하면 어떤 정보 흐름이 가능해지는가?

---

## 07. GIN — How Powerful are Graph Neural Networks? (ICLR 2019)

원문: [GIN](https://openreview.net/forum?id=ryGs6iA5Km)

### ① 한 문장 요약

이웃 multiset에 대한 **sum**과 충분히 표현력 있는 MLP를 사용하면 message-passing GNN이 1-WL graph isomorphism test와 같은 구별 능력에 도달할 수 있음을 보인다.

### ② 배경과 이전 접근의 한계

많은 GNN이 경험적으로 제안됐지만 어떤 graph 구조를 구별할 수 있는지 이론적 기준이 불명확했다. mean과 max aggregator는 각각 분포 비율이나 대표 원소만 남겨 서로 다른 multiset을 같은 값으로 collapse할 수 있다. GIN은 injective multiset 함수라는 관점으로 aggregator의 표현력을 정리한다.

### ③ 선수지식

- message-passing GNN
- multiset과 permutation invariance
- injective function
- Weisfeiler–Lehman(1-WL) test
- MLP와 graph-level readout

### ④ 핵심 아이디어

- 유한 multiset에서 sum은 적절한 mapping과 결합하면 injective하게 만들 수 있다.
- 자기 표현에 `(1+ε)`를 곱해 이웃 합과 구분하고 MLP로 update한다.
- 각 layer는 서로 다른 neighborhood radius를 나타내므로 모든 layer readout을 concat한다.
- MPNN의 상한이 1-WL임을 밝히고, GIN이 그 상한에 도달하는 조건을 제시한다.

### ⑤ 꼭 볼 원문 위치

- **§4, Theorem 3 및 Corollary 6**: injective aggregation·update가 1-WL만큼 강하다는 조건.
- **§4.1, Lemma 5**: 유한 multiset에서 sum 표현이 injective할 수 있다는 핵심 결과.
- **§4, Eq. (4.1)**: `(1+ε)h_v + Σ_u h_u` 뒤 MLP를 적용하는 GIN update.
- **§4, Eq. (4.2)**: 모든 층의 graph readout을 결합하는 식.
- **Figure 2**: sum·mean·max가 서로 다른 multiset을 구별하거나 놓치는 예.

### ⑥ 수식·알고리즘 직관

mean은 `{a,b}`와 `{a,a,b,b}`를 같은 평균으로 보고, max는 중복 횟수를 모두 잊는다. sum은 원소의 종류와 개수를 함께 보존할 여지가 있다. `ε`는 중심 node와 neighbor 합의 상대 비중을 조절한다. MLP가 충분한 표현력을 가져야 합으로 인코딩한 서로 다른 multiset을 서로 다른 출력으로 보낼 수 있다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/07_gin.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/07_gin.ipynb)
- **TODO 07-1 ↔ Eq. (4.1)**: self term과 neighbor sum으로 GIN aggregate를 구현한다.
- **TODO 07-2 ↔ Figure 2**: 두 multiset을 mean과 sum으로 비교해 mean collision을 직접 만든다.
- **TODO 07-3 ↔ Eq. (4.1)–(4.2)**: 여러 GIN 층과 layer-wise readout을 구성해 graph representation을 검증한다.
- **포트폴리오 P1 → P2 → P3**: `GinPortfolioModel`의 sum aggregation·MLP/readout/loss/update를 완성하고, fit loop 뒤 test accuracy·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** GIN의 메시지는 “평균은 개수를 잊지만 합은 multiset을 구별할 수 있다”이며, MPNN 표현력을 1-WL과 연결했다.
- **한계:** 1-WL 자체가 구별하지 못하는 비동형 그래프가 있고, sum은 graph 크기에 민감하며, 높은 표현력이 항상 일반화·수치 안정성을 보장하지 않는다.
- **자가점검:**
  1. mean이 node degree 정보를 잃을 수 있는 구체적 예를 들 수 있는가?
  2. `ε=0`과 learnable `ε`의 차이는 무엇인가?
  3. GIN이 모든 비동형 그래프를 구별하지는 못하는 이유는?

---

## 08. PinSage — Graph Convolutional Neural Networks for Web-Scale Recommender Systems (KDD 2018)

원문: [PinSage](https://arxiv.org/abs/1806.01973)

### ① 한 문장 요약

Pinterest 규모의 item graph에서 random-walk visit count로 중요한 이웃만 골라 importance-weighted convolution을 수행하고, margin ranking으로 추천 임베딩을 학습한다.

### ② 배경과 이전 접근의 한계

full-batch GCN은 수십억 node·edge에서 메모리에 들어가지 않고, 균등 이웃 sampling은 high-degree graph에서 관련 없는 neighbor에 계산을 낭비한다. 또한 실제 추천은 node classification이 아니라 query item보다 positive item을 negative보다 가깝게 순위화해야 한다. PinSage는 sampling, convolution, loss, serving을 web-scale 목표에 맞춘다.

### ③ 선수지식

- GraphSAGE와 neighborhood sampling
- random walk with restart
- importance sampling·weighted mean
- metric learning과 hinge margin loss
- item–item bipartite projection

### ④ 핵심 아이디어

- target node에서 여러 random walk를 실행해 방문 횟수가 큰 이웃을 중요 이웃으로 고른다.
- 방문 확률을 정규화해 neighbor 메시지의 importance weight로 쓴다.
- producer–consumer mini-batch와 localized convolution으로 거대 graph에서 계산한다.
- positive item이 hard negative보다 margin만큼 더 유사하도록 ranking loss를 학습한다.

### ⑤ 꼭 볼 원문 위치

- **§3.2 “Importance Pooling”**: random-walk visit count로 neighborhood를 정의하는 이유.
- **§3.2, Algorithm 1 line 1**: importance-based neighbor selection.
- **Algorithm 1 lines 2–3**: weighted aggregate와 node feature 결합.
- **§3.3, Eq. (1)**: positive와 negative item의 margin ranking loss.
- **Figure 2**: PinSage의 graph convolution과 훈련 파이프라인.

### ⑥ 수식·알고리즘 직관

walk가 자주 방문한 node는 단순히 1-hop이라는 이유보다 target과 경로상 관련성이 높다고 본다. `γ_u`를 정규화한 visit score라 하면 neighbor 합은 `Σ_u γ_u ReLU(W h_u+b)` 형태다. ranking loss `max(0, z_q·z_n - z_q·z_p + Δ)`는 positive score가 negative보다 최소 `Δ`만큼 높으면 더 이상 벌점을 주지 않는다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/08_pinsage.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/08_pinsage.ipynb)
- **TODO 08-1 ↔ §3.2**: random-walk 방문 횟수로 top importance neighbors와 정규화 weight를 만든다.
- **TODO 08-2 ↔ Algorithm 1**: importance-weighted aggregate를 적용하는 `PinSageConv`를 구현한다.
- **TODO 08-3 ↔ Eq. (1)**: positive/negative triplet margin ranking을 학습하고 positive–negative score gap을 확인한다.
- **포트폴리오 P1 → P2 → P3**: `PinsagePortfolioModel`의 walk importance·pooling/ranking loss/update를 완성하고, fit loop 뒤 pairwise accuracy·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** PinSage는 GraphSAGE의 이웃 sampling을 “walk가 알려주는 관련성”으로 바꾸고, 추천 순위 loss와 production scaling을 결합했다.
- **한계:** random-walk sampling 자체가 비싸고 인기 node 편향을 강화할 수 있으며, 정적 graph와 관측 implicit feedback의 노출 편향을 그대로 물려받는다.
- **자가점검:**
  1. 균등 sampling보다 visit-count sampling이 유리한 상황은?
  2. margin을 지나치게 크게 잡으면 학습에 어떤 일이 생기는가?
  3. hard negative가 무작위 negative보다 유익하면서 위험한 이유는?

---

## 09. LightGCN — Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR 2020)

원문: [LightGCN](https://dl.acm.org/doi/10.1145/3397271.3401063)

### ① 한 문장 요약

협업 필터링에서는 feature transform과 비선형성이 오히려 불필요할 수 있음을 보이고, user–item embedding의 정규화 이웃 전파와 층별 합만 남긴 간결한 GCN 추천 모델이다.

### ② 배경과 이전 접근의 한계

NGCF 계열은 일반 GCN에서 가져온 matrix transform, activation, feature interaction을 user/item ID embedding에도 복잡하게 적용했다. 하지만 입력에 풍부한 node feature가 없는 collaborative filtering에서는 이 구성요소가 표현을 왜곡하고 최적화를 어렵게 할 수 있었다. LightGCN은 ablation으로 핵심이 neighborhood aggregation임을 분리한다.

### ③ 선수지식

- implicit-feedback collaborative filtering
- user–item bipartite graph
- GCN degree normalization
- matrix factorization과 dot-product score
- BPR pairwise ranking loss·Recall@K

### ④ 핵심 아이디어

- user와 item의 trainable ID embedding에서 시작한다.
- 선형 변환·bias·activation 없이 정규화된 이웃 embedding만 전파한다.
- 0층 초기 embedding부터 K층 embedding까지 weighted sum해 최종 표현을 만든다.
- 최종 user/item 내적으로 score를 내고 BPR loss로 학습한다.

### ⑤ 꼭 볼 원문 위치

- **§3.1.1, Eq. (3)–(4)**: user↔item 양방향 LightGCN propagation.
- **§3.1.2, Eq. (7)**: 여러 layer embedding의 weighted sum.
- **§3.1.2, Eq. (8)**: 최종 user–item inner-product score.
- **§3.1.3, Eq. (9)**: positive item과 negative item을 비교하는 BPR loss.
- **Figure 1 및 §2.2 ablation 논의**: NGCF에서 무엇을 제거했는지.

### ⑥ 수식·알고리즘 직관

한 user의 다음 층 표현은 `e_u^{k+1}=Σ_{i∈N_u} e_i^k / √(|N_u||N_i|)`다. feature 변환이 없으므로 graph structure가 유일한 전파 연산이며, 0층까지 합쳐 over-smoothing 전의 개인 ID 신호를 보존한다. BPR `-log σ(s_ui-s_uj)`는 관측 item `i`의 score가 미관측 `j`보다 높도록 만든다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/graph_recommendation/exercises/09_lightgcn.ipynb) · [정답 노트북](../../notebooks/field_reproductions/graph_recommendation/solutions/09_lightgcn.ipynb)
- **TODO 09-1 ↔ Eq. (3)–(4)**: user–item bipartite adjacency를 만들고 대칭 정규화한다.
- **TODO 09-2 ↔ Eq. (7)–(8)**: K회 선형 전파 뒤 0…K층을 평균해 score를 계산한다.
- **TODO 09-3 ↔ Eq. (9)**: positive/negative triplet으로 BPR을 학습한다.
- **TODO 09-4 ↔ 추천 평가**: 이미 본 item을 mask하고 test interaction의 Recall@3를 계산한다.
- **포트폴리오 P1 → P2 → P3**: `LightgcnPortfolioModel`의 bipartite propagation·BPR loss/update를 완성하고, fit loop 뒤 pairwise accuracy·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** LightGCN은 “추천에서 GCN의 본체는 graph smoothing뿐”이라는 강한 단순화이며, layer sum으로 여러 hop 협업 신호를 결합한다.
- **한계:** side feature·시간·순서를 직접 사용하지 않고, 미관측을 negative로 보는 BPR의 노출 편향과 인기 편향, 깊은 전파의 over-smoothing이 남는다.
- **자가점검:**
  1. 0층 embedding을 최종 합에 포함하는 이유는?
  2. 대칭 degree normalization이 인기 item의 메시지를 어떻게 완화하는가?
  3. Recall@K 계산 전에 train item을 mask해야 하는 이유는?

---

## 다음에 읽을 최신 논문 5편 (2023–2024, 공식 게재 확인)

아래 목록은 2026년 8월 기준으로 **공식 conference/journal proceedings에서 제목·venue·연도를 직접 확인**했다. arXiv의 주장만으로 venue를 추정하지 않았다.

### A. Exphormer: Sparse Transformers for Graphs

- **Venue / year:** ICML 2023, PMLR 202.
- **왜 읽나:** GAT의 attention을 모든 node pair에 적용할 때 생기는 제곱 복잡도를, expander edge와 virtual global node를 이용한 희소 attention으로 선형 규모에 가깝게 줄인다.
- **선수지식:** GAT, Transformer self-attention, expander graph의 직관, spectral gap, GraphGPS.
- **공식 URL:** [PMLR proceedings](https://proceedings.mlr.press/v202/shirzad23a.html)

### B. LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation

- **Venue / year:** ICLR 2023.
- **왜 읽나:** LightGCN 이후 추천 graph의 희소성·인기 편향을 다루며, 임의 edge drop 대신 SVD에서 얻은 global collaborative view를 contrastive target으로 쓴다.
- **선수지식:** LightGCN, SVD, InfoNCE, graph contrastive learning, BPR.
- **공식 URL:** [OpenReview 게재본](https://openreview.net/pdf?id=FKXVK9dyMM)

### C. Simple and Asymmetric Graph Contrastive Learning without Augmentations (GraphACL)

- **Venue / year:** NeurIPS 2023 Main Conference Track.
- **왜 읽나:** 수작업 graph augmentation과 homophily 가정 없이 asymmetric neighbor view로 homophilic·heterophilic graph를 함께 다루어, GCN류의 동질성 편향 다음 문제를 보여준다.
- **선수지식:** GCN, contrastive learning, stop-gradient 또는 asymmetric branch, homophily·heterophily, 1-hop·2-hop 구조.
- **공식 URL:** [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3430bcc30cdaabd0bf6c5d0c31bda67c-Abstract-Conference.html)

### D. An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations

- **Venue / year:** NeurIPS 2023 Main Conference Track.
- **왜 읽나:** graph contrastive pretraining과 downstream 추천 목적의 불일치를 개인화 prompt로 줄이는 CPTPP를 제안해, “사전학습 표현을 추천에 어떻게 이전하는가”를 직접 다룬다.
- **선수지식:** LightGCN, graph contrastive pretraining, prompt tuning, user profiling, representation uniformity.
- **공식 URL:** [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c6af791af7ef0f3e02bccef011211ca5-Abstract-Conference.html)

### E. Subgraphormer: Unifying Subgraph GNNs and Graph Transformers via Graph Products

- **Venue / year:** ICML 2024, PMLR 235.
- **왜 읽나:** GIN에서 본 1-WL 표현력 한계를 넘기 위한 subgraph GNN과 graph Transformer를 product graph 관점으로 통합하고, 그 구조에 맞는 attention·positional encoding을 설계한다.
- **선수지식:** GIN·1-WL, subgraph GNN, Cartesian/product graph, Transformer attention, positional encoding.
- **공식 URL:** [PMLR proceedings](https://proceedings.mlr.press/v235/bar-shalom24a.html)

### 최신 5편 선정 기준

1. **공식성:** 2023–2026 범위에서 ICML·ICLR·NeurIPS 또는 동급 공식 proceedings에 게재가 확인될 것.
2. **학습 경로의 연속성:** 위 10편의 축인 sampling, message passing, expressivity, graph recommendation을 한 단계 확장할 것.
3. **주제 다양성:** sparse graph Transformer, subgraph 표현력, heterophily contrastive learning, 추천용 spectral view, 추천 prompt transfer를 중복 없이 포함할 것.
4. **재현 가치:** 논문의 핵심을 작은 graph에서 attention mask, SVD view, asymmetric loss, prompt vector, product-graph adjacency 같은 단위 실험으로 축소할 수 있을 것.

---

## 읽기 완료 체크리스트

- 각 논문의 원문 위치를 먼저 읽고 수식의 입력·출력 shape을 적었다.
- exercise의 TODO를 solution을 열기 전에 한 번 이상 구현했다.
- assertion이 무엇을 반증하는지, metric이 원 논문의 어떤 주장을 축소 검증하는지 말할 수 있다.
- 원 논문 규모와 로컬 실습 규모의 차이를 “같은 점 / 생략한 점”으로 나누어 설명할 수 있다.
