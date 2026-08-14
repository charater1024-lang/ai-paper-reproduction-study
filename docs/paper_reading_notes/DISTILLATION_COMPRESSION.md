# 지식 증류·모델 압축 핵심 논문 10편 독해 노트

> **AI 보조 작성 안내:** 이 한국어 요약은 공식 번역이 아닙니다. 내용과 인용의 최종 기준은
> 연결된 원문입니다. [작성 원칙과 한계](AI_ASSISTED_NOTICE.md) · [70편 목차](README.md)

이 문서는 큰 모델의 지식을 작은 모델로 옮기는 방법부터 희소화, 양자화, 모바일 구조, 한 번의 학습으로 여러 배포 제약을 만족하는 방법까지 이어서 읽도록 설계했다. 각 절의 원문 위치는 [압축 분야 실습 사양](../../tools/field_curriculum/compression_specs.py)과 맞췄으며, 실습은 저장소에 포함된 작은 로컬 데이터만 사용한다. 따라서 논문의 대규모 결과를 그대로 재현한다기보다, 논문을 성립시키는 핵심 계산과 비교 논리를 CPU에서 빠르게 검증하는 것이 목표다.

권장 분야 순서는 `00→03`에서 지식 이전의 손실을 익히고, `04→06`에서 가중치와 수치 표현을 줄이는 법을 배운 다음, `07→09`에서 하드웨어 친화적 구조와 배포별 특화를 연결하는 것이다.

각 노트북 안에서는 **핵심 수식 Task → portfolio class 조립 → quality·cost contract 검증**으로 진행한다. TODO 1–3에서 논문의 핵심 계산을 구현하고, 포트폴리오 P1에서 `PortfolioConfig`, batch 준비, `forward`·`compute_loss`·`training_step`을 가진 class로 조립한다. P2의 update와 P3가 제공하는 정확도·오차·parameter·sparsity·bit·latency 중 해당 논문의 품질·비용 지표 검증을 함께 끝낸다.

## 00. Knowledge Distillation — Distilling the Knowledge in a Neural Network (2015)

**① 한 문장 요약.** 큰 교사 모델이 만든 부드러운 클래스 확률을 높은 온도에서 작은 학생에게 모사시키면, 정답 레이블만으로는 보이지 않는 클래스 사이의 유사성까지 전달할 수 있다는 방법이다.

**② 왜 나왔나 / 이전 한계.** 앙상블과 큰 신경망은 정확하지만 추론 비용과 메모리가 커서 실제 서비스에 배치하기 어렵다. 작은 모델을 정답 레이블만으로 다시 학습하면 교사가 학습한 “이 입력은 2이지만 3과도 조금 비슷하다” 같은 상대적 정보를 버린다. 이 논문은 교사의 전체 출력 분포를 압축 대상 지식으로 보고, 배포 시에는 학생 하나만 남기는 방법을 제안했다.

**③ 읽기 전 기초지식.** softmax와 logit, temperature, 교차엔트로피, KL divergence, 앙상블, teacher–student 학습.

**④ 핵심 아이디어.**

- 온도 `T>1`로 교사 확률을 평탄하게 하여 정답 이외 클래스의 관계를 드러낸다.
- 학생도 같은 온도에서 만든 분포로 교사를 모사하며, 이 항의 기울기 크기를 보정하려고 `T²`를 곱한다.
- 부드러운 교사 표적과 실제 정답의 hard-label 손실을 함께 사용한다.
- 학습 뒤에는 온도 1의 작은 학생만 추론에 사용하므로 교사는 배포 비용에 포함되지 않는다.

**⑤ 꼭 볼 원문 위치.** [원문 페이지](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/)의 §2 Eq. (1)에서 temperature softmax를 보고, §2 Eq. (2)–(4)에서 큰 `T`일 때 증류 기울기가 어떻게 바뀌는지 읽는다. 이어 §2의 soft target과 correct label 결합이 실습의 최종 손실과 연결되고, §3.1의 specialist/teacher transfer는 교사 지식을 단순한 정답보다 넓게 보는 근거다.

**⑥ 수식·알고리즘 직관.** 클래스 `i`의 확률은 `q_i=exp(z_i/T)/Σ_j exp(z_j/T)`이다. `T`가 크면 logit 차이가 완만해져 작은 확률도 0에 눌리지 않는다. 학생 손실을 `α T² KL(q_teacher^T || q_student^T) + (1-α) CE(y,q_student)`로 생각하면, 첫 항은 교사의 상대적 판단을, 둘째 항은 실제 정답을 지킨다. `T²`는 온도를 높일 때 대략 `1/T²`로 작아지는 logit 기울기를 보상한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/00_knowledge_distillation.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/00_knowledge_distillation.ipynb). TODO 1은 temperature soft target을 구현하고 `T`에 따른 entropy 증가를 확인한다. TODO 2는 `T²`로 보정한 KL과 hard-label CE를 결합한다. TODO 3은 작은 학생을 학습해 교사와의 KL, 분류 정확도, 손실 곡선을 함께 본다. 이 실습은 §2의 손실 메커니즘을 재현하며 specialist 앙상블 자체는 축소한다.

- **포트폴리오 P1 → P2 → P3:** `KnowledgeDistillationPortfolioModel`의 teacher/student logits·KD loss/update를 완성하고, fit loop 뒤 test accuracy·teacher KL 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 5분 안에는 “높은 온도로 숨은 클래스 관계를 펼치고, `T² KL + CE`로 학생에게 옮긴다”를 설명할 수 있으면 된다. 한계는 좋은 교사가 필요하고, 온도와 혼합 계수에 민감하며, 교사의 오류와 편향도 함께 전달할 수 있다는 점이다.

1. `T`를 높일 때 교사 분포의 entropy가 커지는 이유는 무엇인가?
2. 증류 KL에 `T²`를 곱하지 않으면 두 손실의 상대적 크기는 어떻게 달라질까?
3. hard-label CE를 완전히 제거하면 어떤 상황에서 학생이 교사의 오류를 그대로 따를까?

## 01. FitNets — Hints for Thin Deep Nets (2015)

**① 한 문장 요약.** 교사의 중간 표현을 “힌트”로 삼아 더 얇지만 더 깊은 학생의 중간층을 먼저 안내한 뒤 출력 증류를 수행하는 2단계 학습법이다.

**② 왜 나왔나 / 이전 한계.** 출력 확률만 모사하는 증류는 학생이 매우 얇거나 구조가 교사와 다를 때 최적화 신호가 부족할 수 있다. 특히 깊고 가는 학생은 처음부터 최종 출력만 맞추려 하면 유용한 표현에 도달하기 어렵다. FitNets는 교사의 내부 표현을 중간 목표로 제공해 학생이 좋은 표현 공간에 먼저 진입하도록 했다.

**③ 읽기 전 기초지식.** feature map, hidden representation, 회귀(regression), mean squared error, knowledge distillation, 2단계 최적화.

**④ 핵심 아이디어.**

- 교사의 한 hidden layer를 hint layer, 학생의 대응 layer를 guided layer로 고른다.
- 두 표현의 차원이 다르면 학습 가능한 regressor로 학생 표현을 교사 차원에 맞춘다.
- 1단계에서 hint MSE로 학생 앞부분을 사전학습하고, 2단계에서 출력 KD와 정답 CE로 전체 학생을 미세조정한다.
- 학생이 교사보다 깊더라도 폭을 줄여 연산량을 낮출 수 있다는 구조적 선택을 보여준다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1412.6550)의 §3.1 Eq. (2)는 hint regression 목적함수이고, Figure 1은 hint 학습 뒤 distillation을 하는 두 단계를 그린다. §3.2 Eq. (3)은 두 번째 단계의 KD 목적이며, §4는 얇고 깊은 학생을 비교하는 실험 설계다.

**⑥ 수식·알고리즘 직관.** 교사 힌트를 `u_h(x)`, 학생 guided 표현을 `v_g(x)`라 하면 먼저 `||u_h(x)-r(v_g(x))||²`를 줄인다. `r`은 단순히 차원을 맞추는 보조 regressor다. 이는 학생이 최종 정답이라는 먼 목표만 쫓기 전에 중간 “경유지”를 통과하게 만든다. 이후 출력 분포 KD와 CE를 적용해 중간 표현의 유사성이 실제 예측으로 이어지게 한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/01_fitnets.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/01_fitnets.ipynb). TODO 1은 12차원 guided layer와 24차원 교사 hint를 잇는 regressor를 만든다. TODO 2는 §3.1 Eq. (2)의 hint MSE로 guided layer와 regressor를 사전학습한다. TODO 3은 §3.2 Eq. (3)에 대응하는 출력 KD+CE로 학생을 미세조정하고 정확도와 손실을 비교한다.

- **포트폴리오 P1 → P2 → P3:** `FitnetsPortfolioModel`의 guided/regressor·hint/output loss·2단계 update를 완성하고, fit loop 뒤 hint MSE·test accuracy 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “중간층 힌트로 길을 잡고, 출력 증류로 목적지에 도착하는 2단계 학습”이다. 어떤 층끼리 연결할지, 표현 차이를 어떤 regressor가 흡수할지에 따라 결과가 달라지고, 교사의 표현을 지나치게 강제하면 학생 고유의 더 나은 표현을 막을 수 있다.

1. 출력 KD만 할 때보다 중간 hint가 얇고 깊은 학생의 최적화를 돕는 이유는 무엇인가?
2. 교사와 학생 hidden 차원이 다를 때 regressor가 필요한 이유는 무엇인가?
3. hint MSE가 낮아도 최종 정확도가 보장되지 않는 이유는 무엇인가?

## 02. Attention Transfer — Paying More Attention to Attention (2017)

**① 한 문장 요약.** 교사와 학생의 채널 수가 달라도 공간적으로 어디를 중요하게 보는지 나타내는 attention map을 정규화해 맞추는 표현 증류법이다.

**② 왜 나왔나 / 이전 한계.** FitNets처럼 전체 feature tensor를 직접 맞추면 채널 수와 의미가 다른 네트워크 사이에서 대응을 만들기 어렵다. 또한 모든 activation 값을 같게 만드는 것은 학생에게 지나치게 강한 제약이다. Attention Transfer는 채널을 하나의 공간 중요도 지도로 요약해, 구조가 달라도 “어디를 보았는가”만 전달한다.

**③ 읽기 전 기초지식.** convolutional feature map, channel reduction, activation energy, L2 normalization, attention map, 보조 손실(auxiliary loss).

**④ 핵심 아이디어.**

- 각 공간 위치에서 채널 activation의 절댓값 또는 제곱을 합쳐 attention map을 만든다.
- map을 벡터화하고 L2 정규화하여 절대 크기보다 공간 패턴을 비교한다.
- 여러 층에서 교사와 학생 attention의 거리를 분류 손실에 더한다.
- feature channel을 일대일로 맞추지 않아 서로 다른 폭의 CNN에도 적용할 수 있다.

**⑤ 꼭 볼 원문 위치.** [OpenReview 원문](https://openreview.net/forum?id=Sks9_ajex)의 §2.1 Eq. (1)–(2)에서 activation 기반 attention 연산을, §2.1 Eq. (3)에서 벡터 정규화를 확인한다. §2.2 Eq. (4)는 전체 attention-transfer 손실이며, §3 실험은 이 보조 신호가 분류 학습과 어떻게 결합되는지 보여준다.

**⑥ 수식·알고리즘 직관.** feature `A∈R^{C×H×W}`에서 `F(A)_{h,w}=Σ_c |A_{c,h,w}|^p`로 채널을 접으면 밝은 위치가 모델의 관심 영역이 된다. `q=vec(F)/||vec(F)||₂`로 정규화한 뒤 `||q_S-q_T||₂`를 줄이면, 학생의 activation 크기가 작거나 채널 수가 달라도 공간적 초점은 같아진다. 총손실은 `CE + βΣ_l attention_distance_l`이다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/02_attention_transfer.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/02_attention_transfer.ipynb). TODO 1은 §2.1 Eq. (1)–(3)의 channel-power attention과 L2 정규화를 구현한다. TODO 2는 §2.2 Eq. (4)의 교사–학생 attention 거리를 계산한다. TODO 3은 분류 CE와 AT 손실을 함께 최적화하고 attention map과 학습 곡선을 비교한다.

- **포트폴리오 P1 → P2 → P3:** `AttentionTransferPortfolioModel`의 attention map·CE+AT loss/update를 완성하고, fit loop 뒤 attention norm error·test accuracy 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** “feature 값 전체 대신 정규화된 공간적 관심을 전달한다”가 5분 요약이다. attention 정의가 실제 인과적 중요도를 보장하지 않고, 어느 층을 연결할지와 `β`에 민감하며, 공간 해상도가 크게 다르면 별도의 정렬이 필요하다.

1. 채널을 합친 attention map이 서로 다른 폭의 네트워크를 연결할 수 있는 이유는 무엇인가?
2. L2 정규화를 빼면 activation scale 차이가 손실에 어떤 영향을 주는가?
3. attention map이 비슷하지만 분류 결과는 다를 수 있는 예를 들어볼 수 있는가?

## 03. TinyBERT — Distilling BERT for Natural Language Understanding (2020)

**① 한 문장 요약.** BERT의 embedding, hidden state, attention, prediction을 층별로 모사하고 일반 단계와 과제별 단계를 나눠 작은 Transformer에 지식을 옮긴다.

**② 왜 나왔나 / 이전 한계.** BERT는 다양한 NLP 과제에서 강력하지만 층과 hidden 차원이 커서 지연시간과 메모리 부담이 크다. logits만 맞추는 증류는 Transformer 내부의 attention 관계와 문맥 표현을 충분히 전달하지 못한다. TinyBERT는 여러 수준의 손실과 2단계 학습으로 범용 언어 지식과 과제별 판단을 함께 압축한다.

**③ 읽기 전 기초지식.** Transformer encoder, token embedding, hidden state, self-attention matrix, layer alignment, logits와 soft target.

**④ 핵심 아이디어.**

- 학생 embedding과 hidden state를 projection으로 교사 차원에 맞춰 MSE로 모사한다.
- attention score matrix를 직접 맞춰 토큰 간 관계를 전달한다.
- prediction-layer 손실로 최종 행동도 교사에 가깝게 만든다.
- Figure 1의 general distillation과 task-specific distillation을 분리해 사전학습 지식과 과제 지식을 단계적으로 옮긴다.

**⑤ 꼭 볼 원문 위치.** [ACL Anthology 원문](https://aclanthology.org/2020.findings-emnlp.372/)의 Figure 1에서 general/task-specific distillation 흐름을 먼저 본다. §3.1 Eq. (4)–(5)는 embedding과 hidden-state 손실, Eq. (6)은 attention 손실, Eq. (7)–(8)은 prediction 손실과 전체 목적함수다.

**⑥ 수식·알고리즘 직관.** 학생 hidden `H_S`를 projection `W_h`로 교사 차원에 올려 `MSE(H_SW_h,H_T)`를 줄이고, 각 head의 attention matrix도 `MSE(A_S,A_T)`로 맞춘다. 출력에서는 교사 logits 또는 확률을 모사한다. 즉 단어의 좌표, 문맥 표현, 토큰 관계, 최종 판단을 서로 다른 관측 창으로 보고 동시에 정렬하는 셈이다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/03_tinybert.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/03_tinybert.ipynb). TODO 1은 16차원 학생 hidden을 24차원 교사로 보내는 regressor와 Eq. (4)–(5)의 hidden MSE를 구현한다. TODO 2는 Eq. (6)의 attention MSE를 계산한다. TODO 3은 hidden+attention+prediction 손실을 합쳐 학생을 학습한다. 로컬의 작은 문장 표현을 사용하므로 대규모 일반 증류 전체가 아니라 손실 구조를 검증한다.

- **포트폴리오 P1 → P2 → P3:** `TinybertPortfolioModel`의 hidden regressor·attention/prediction loss·update를 완성하고, fit loop 뒤 hidden/attention MSE·test accuracy 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 5분 요약은 “BERT의 최종 출력뿐 아니라 hidden과 attention까지, 일반→과제별 두 단계로 증류한다”이다. 층 대응 규칙과 손실 가중치가 필요하고, 여러 중간 값을 저장해 학습 비용이 늘며, 작은 실습의 성능을 원 논문의 대규모 NLU 결과와 직접 비교할 수 없다.

1. prediction loss만으로는 attention과 hidden 지식이 충분히 전달되지 않을 수 있는 이유는 무엇인가?
2. 학생과 교사의 층 수가 다르면 어떤 기준으로 layer alignment를 정할 수 있는가?
3. hidden MSE와 attention MSE가 서로 보완하거나 충돌하는 상황은 무엇인가?

## 04. Deep Compression — Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding (2016)

**① 한 문장 요약.** 중요하지 않은 연결을 가지치기하고, 남은 가중치를 공유 centroid로 양자화한 뒤, 인덱스를 Huffman 부호화하는 3단계 압축 파이프라인이다.

**② 왜 나왔나 / 이전 한계.** 대형 신경망은 중복된 연결과 높은 정밀도의 가중치 때문에 저장 공간과 메모리 대역폭을 많이 소비한다. 단순 pruning은 sparse index 비용을, 단순 양자화는 정확도 손상을, 일반 압축은 신경망 구조를 활용하지 못하는 한계가 있다. 논문은 서로 다른 중복성을 차례로 제거하는 세 기술을 조합했다.

**③ 읽기 전 기초지식.** magnitude pruning, sparsity, k-means와 centroid, weight sharing, entropy와 Huffman coding, sparse index.

**④ 핵심 아이디어.**

- 작은 magnitude의 가중치를 제거하고 남은 연결을 재학습한다.
- 남은 값들을 몇 개 centroid에 할당해 각 weight 대신 짧은 codebook index를 저장한다.
- 빈도가 높은 centroid와 index 차이에 짧은 Huffman code를 배정한다.
- pruning→trained quantization→coding 순서로 구조적·수치적·통계적 중복을 단계별로 줄인다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1510.00149)의 Figure 1이 전체 3단계 파이프라인이다. §3은 network pruning, §4와 Figure 2는 trained quantization과 weight sharing, §5는 Huffman coding을 다룬다. 실습 TODO 1–3이 이 순서를 그대로 따른다.

**⑥ 수식·알고리즘 직관.** pruning은 `|w|<τ`인 연결에 mask 0을 주는 선택이고, quantization은 남은 `w_i`를 가장 가까운 centroid `c_{k(i)}`로 치환하는 1차원 군집화다. 저장량은 값마다 32비트를 쓰는 대신 sparse 위치 정보, 작은 index, codebook으로 바뀐다. Huffman coding은 index 분포의 entropy가 낮을수록 평균 비트 수를 더 줄인다. 다만 이론적 비트 절감이 실제 하드웨어 속도 향상과 같지는 않다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/04_deep_compression.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/04_deep_compression.ipynb). TODO 1은 §3처럼 magnitude 기준으로 70%를 pruning한다. TODO 2는 §4/Figure 2처럼 남은 weight에 8-centroid 1D k-means를 적용한다. TODO 3은 §5와 연결해 index entropy, Huffman 하한의 비트 추정, 압축률을 계산한다. 수치는 교육용 배열의 결과이며 원 논문의 모델 압축률을 재현한 값이 아니다.

- **포트폴리오 P1 → P2 → P3:** `DeepCompressionPortfolioModel`의 prune→quantize pipeline과 update를 완성하고, fit stage 뒤 test accuracy·weight sparsity 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** “연결 수를 줄이고, 값의 종류를 줄이고, 자주 나오는 기호를 짧게 저장한다”가 5분 설명이다. 비정형 sparsity는 전용 커널 없이는 빠르지 않을 수 있고, index·codebook 오버헤드가 있으며, 각 단계 뒤 재학습과 하이퍼파라미터 선택이 필요하다.

1. pruning과 quantization은 각각 어떤 종류의 중복을 제거하는가?
2. 희소도가 높아도 실제 추론이 빨라지지 않을 수 있는 이유는 무엇인가?
3. centroid 사용 빈도가 균등할 때와 치우칠 때 Huffman coding의 이득은 어떻게 달라지는가?

## 05. Lottery Ticket Hypothesis — The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (2019)

**① 한 문장 요약.** 무작위 초기화된 큰 네트워크 안에는 원래 초기값으로 되감았을 때 독립적으로 잘 학습되는 희소한 부분망, 즉 “winning ticket”이 존재한다는 가설과 탐색 절차다.

**② 왜 나왔나 / 이전 한계.** pruning은 보통 큰 모델을 끝까지 학습한 뒤 불필요한 연결을 제거하고 다시 미세조정했다. 이는 희소 구조가 처음부터 학습 가능한지, 큰 과매개변수 모델이 왜 최적화를 돕는지 설명하지 못했다. 이 논문은 최종 가중치 자체보다 학습으로 발견한 mask와 그에 대응하는 초기값의 조합에 주목했다.

**③ 읽기 전 기초지식.** random initialization, binary mask, unstructured pruning, weight rewinding, iterative magnitude pruning, sparsity와 capacity.

**④ 핵심 아이디어.**

- winning ticket을 “초기화된 dense network의 일부로서 비슷한 시간 안에 비교 가능한 성능에 도달하는 부분망”으로 정의한다.
- dense 모델을 학습하고 작은 magnitude의 연결을 제거한 뒤, 생존 weight를 원래 초기값으로 되감는다.
- 이 과정을 반복하는 iterative magnitude pruning으로 mask를 점진적으로 찾는다.
- 동일한 희소 구조라도 초기값을 임의로 다시 뽑으면 학습 거동이 달라질 수 있어 구조와 초기화의 결합이 중요하다.

**⑤ 꼭 볼 원문 위치.** [OpenReview 원문](https://openreview.net/forum?id=rJl-b3RcF7)의 §2에서 winning ticket 정의를 먼저 읽는다. §2 Figure 1은 학습→pruning→초기값 복원의 반복 흐름이고, §3은 mask를 유지하는 학습 방법, §4는 sparsity와 정확도를 비교하는 결과다.

**⑥ 수식·알고리즘 직관.** 초기 weight를 `w₀`, 학습된 weight를 `w_t`, mask를 `m∈{0,1}^d`라 하자. `|w_t|`가 큰 연결을 남겨 `m`을 얻은 뒤 새 출발점은 `m⊙w₀`이다. 학습 중에도 gradient와 weight에 mask를 적용해 제거된 연결이 되살아나지 않게 한다. dense 학습은 좋은 희소 경로를 탐색하는 과정이고, rewinding은 그 경로가 학습 전부터 유효했는지를 시험한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/05_lottery_ticket.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/05_lottery_ticket.ipynb). TODO 1은 dense MLP를 학습하고 magnitude 상위 30% 연결의 mask를 만든다. TODO 2는 Figure 1처럼 생존 연결을 초기 weight로 되감고 mask를 지키며 재학습한다. TODO 3은 ticket 정확도와 active-parameter ratio를 그려 §4의 비교 논리를 축소 재현한다.

- **포트폴리오 P1 → P2 → P3:** `LotteryTicketPortfolioModel`의 mask·rewind·masked update를 완성하고, fit loop 뒤 test accuracy·weight sparsity와 mask 불변 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 5분 안에는 “학습된 값이 아니라 학습으로 찾은 mask와 원래 초기값이 함께 ticket을 이룬다”고 설명하면 된다. ticket 탐색 자체가 dense 학습을 요구하고, 큰 모델·다른 optimizer에서는 단순 초기값 복원만으로 잘 작동하지 않을 수 있으며, 비정형 mask의 이론적 연산 절감이 실제 속도로 바로 이어지지 않는다.

1. 일반적인 사후 pruning과 winning-ticket 실험의 가장 중요한 차이는 무엇인가?
2. 제거한 weight에 gradient mask를 적용하지 않으면 어떤 일이 생기는가?
3. 같은 mask에 새로운 random initialization을 넣는 대조실험은 무엇을 검증하는가?

## 06. Integer-only Quantization — Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (2018)

**① 한 문장 요약.** 실수 tensor를 scale과 zero-point를 가진 정수로 표현하고, 학습 중 fake quantization을 거쳐 추론의 핵심 연산을 정수 산술만으로 실행하는 방법이다.

**② 왜 나왔나 / 이전 한계.** 모바일 CPU와 전용 가속기에서는 부동소수점 연산과 메모리 이동이 비싸다. 가중치만 낮은 정밀도로 저장해도 activation과 matmul이 실수이면 정수 하드웨어의 이점을 충분히 얻지 못한다. 반대로 학습 뒤 단순 반올림하면 범위 포화와 양자화 오차로 정확도가 크게 흔들릴 수 있다.

**③ 읽기 전 기초지식.** affine quantization, scale과 zero-point, uint8/int32 accumulator, clipping과 rounding, fake quantization, straight-through estimator(STE).

**④ 핵심 아이디어.**

- 실수 `r`을 `r=S(q-Z)` 형태로 나타내 0도 정확히 표현하는 affine grid를 만든다.
- activation과 weight의 zero-point를 뺀 정수끼리 곱하고 넓은 int32 누산기에 합산한다.
- 입력·가중치 scale의 곱으로 정수 합을 다시 실수 의미에 맞춘다.
- 학습 forward에서는 quantize–dequantize를 흉내 내고 backward에서는 STE로 gradient를 통과시킨다.

**⑤ 꼭 볼 원문 위치.** [CVPR 공식 원문](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)의 §2.1 Eq. (1)은 `r=S(q-Z)`, §2.2 Eq. (2)–(4)는 quantized matrix multiplication이다. §2.4에서 zero-point와 범위를 확인하고, §3에서 simulated quantization training을 읽는다.

**⑥ 수식·알고리즘 직관.** 범위 `[r_min,r_max]`를 정수 `[0,255]`에 놓으면 대략 `S=(r_max-r_min)/255`, `q=clip(round(r/S)+Z)`다. matmul에서는 `Σ_j(q^a_j-Z_a)(q^w_j-Z_w)`를 int32로 누산하고 `S_aS_w`를 곱해 실수 결과의 크기를 복원한다. fake quantization은 forward에 계단 함수를 보여 주되 backward에서 그 미분 불가능성을 STE로 우회해 모델이 오차에 적응하게 한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/06_integer_quantization.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/06_integer_quantization.ipynb). TODO 1은 §2.1/§2.4의 uint8 affine quantize–dequantize를 구현한다. TODO 2는 §2.2 Eq. (2)–(4)처럼 centered integer activation/weight matmul 뒤 scale을 복원한다. TODO 3은 §3의 STE 기반 fake-quantized classifier를 학습한다.

- **포트폴리오 P1 → P2 → P3:** `IntegerQuantizationPortfolioModel`의 scale/zero-point·integer matmul·fake-quant update를 완성하고, fit loop 뒤 round-trip MSE·test accuracy 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “scale과 zero-point로 실수 의미를 보존하고, 정수 곱·넓은 누산·재스케일을 한다”이다. 범위 calibration이 outlier에 민감하고, 매우 낮은 비트에서는 오차가 커지며, 실제 속도는 지원되는 정수 커널과 장치에 달려 있다. STE는 진짜 미분이 아니라 근사다.

1. asymmetric affine quantization에서 zero-point가 필요한 이유는 무엇인가?
2. uint8 곱의 합을 같은 uint8에 누산하면 왜 위험한가?
3. fake quantization forward와 STE backward는 각각 어떤 역할을 하는가?

## 07. MobileNetV2 — Inverted Residuals and Linear Bottlenecks (2018)

**① 한 문장 요약.** 좁은 입력을 넓게 확장한 뒤 depthwise convolution을 적용하고 다시 선형 저차원으로 투영하는 inverted residual block으로 모바일 CNN의 연산과 표현 손실을 줄인다.

**② 왜 나왔나 / 이전 한계.** depthwise separable convolution은 표준 convolution보다 싸지만, 좁은 feature에 비선형성을 반복하면 정보가 손실될 수 있다. 일반 ResNet block은 넓은 입력을 좁혔다가 다시 넓히며 넓은 tensor 사이에 shortcut을 둔다. MobileNetV2는 모바일 환경의 메모리·연산 제약을 고려해 이 관계를 뒤집었다.

**③ 읽기 전 기초지식.** standard convolution, depthwise와 pointwise convolution, bottleneck, residual connection, expansion ratio, ReLU6와 선형 projection.

**④ 핵심 아이디어.**

- `3×3` spatial filtering을 channel별 depthwise convolution으로 분리하고 `1×1` pointwise로 channel을 섞는다.
- 낮은 차원 입력을 `1×1`로 확장해 비선형 변환이 일어날 충분한 공간을 만든다.
- 마지막 좁은 bottleneck에는 activation을 두지 않는 linear projection으로 정보 붕괴를 줄인다.
- stride 1이고 입출력 shape가 같을 때 좁은 bottleneck 사이에 shortcut을 연결한다.

**⑤ 꼭 볼 원문 위치.** [CVPR 공식 원문](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)의 §3.1은 depthwise separable convolution, §3.2는 linear bottleneck을 설명한다. §3.3 Figure 3에서 inverted residual block을 보고, §3.4 Table 2에서 전체 architecture가 block을 어떻게 쌓는지 확인한다.

**⑥ 수식·알고리즘 직관.** 표준 `k×k` convolution 비용이 대략 `HW·C_in·C_out·k²`라면, depthwise+pointwise는 `HW·C_in·k² + HW·C_in·C_out`이다. block은 `C → tC → tC → C'`의 expand–depthwise–project 흐름이다. 좁은 `C'`에서 ReLU를 쓰면 저차원 부분공간의 음수 성분이 잘려 복구하기 어려우므로 마지막은 선형으로 둔다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/07_mobilenet_v2.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/07_mobilenet_v2.ipynb). TODO 1은 §3.1의 표준 `3×3`과 depthwise+pointwise parameter 수를 비교한다. TODO 2는 §3.2–§3.3의 expand→depthwise→linear-project와 조건부 shortcut을 구현한다. TODO 3은 mini MobileNetV2를 학습해 parameter count와 정확도를 함께 본다.

- **포트폴리오 P1 → P2 → P3:** `MobilenetV2PortfolioModel`의 inverted residual·linear bottleneck/loss/update를 완성하고, fit loop 뒤 test accuracy·parameter 수·depthwise groups 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** “넓은 중간 공간에서 가볍게 처리하고, 좁은 선형 bottleneck을 shortcut으로 잇는다”가 요약이다. FLOPs 감소가 메모리 접근과 커널 구현 때문에 모든 장치에서 같은 지연시간 감소를 보장하지 않으며, 지나치게 작은 width에서는 표현력이 급격히 떨어질 수 있다.

1. inverted residual은 전통적인 residual bottleneck과 무엇이 반대인가?
2. 마지막 좁은 projection에서 ReLU를 제거한 이유는 무엇인가?
3. depthwise convolution만으로 channel 간 정보를 섞을 수 없는 이유는 무엇인가?

## 08. ShuffleNetV2 — Practical Guidelines for Efficient CNN Architecture Design (2018)

**① 한 문장 요약.** FLOPs만이 아니라 메모리 접근과 병렬성까지 고려한 네 가지 실용 지침을 세우고, channel split·concat·shuffle로 효율적인 block을 설계한다.

**② 왜 나왔나 / 이전 한계.** 같은 FLOPs의 모델도 실제 장치 지연시간은 크게 다를 수 있다. group convolution을 과도하게 쓰면 fragmentation과 메모리 접근 비용이 늘고, 많은 작은 연산은 병렬화에 불리하다. 논문은 이론적 연산량만 최소화하는 관행을 비판하고 실제 속도 측정에서 출발했다.

**③ 읽기 전 기초지식.** FLOPs와 latency, memory access cost, group convolution, channel split/concat, channel shuffle, operator fragmentation.

**④ 핵심 아이디어.**

- G1: 입출력 channel 폭을 균형 있게 하면 같은 연산량에서 메모리 접근 비용이 낮아진다.
- G2–G4: 과도한 group 수, network fragmentation, 많은 element-wise 연산을 피한다.
- stride 1 block은 channel 절반을 identity로 보내고 나머지 절반만 pointwise–depthwise–pointwise branch로 처리한다.
- 두 branch를 concat한 뒤 channel shuffle로 다음 block에서 정보가 섞이게 한다.

**⑤ 꼭 볼 원문 위치.** [ECCV 공식 원문](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html)의 §2에서 G1–G4를 읽고, §3 Figure 3(c)에서 channel split branch를 확인한다. 같은 Figure 3의 concat+shuffle이 실습 구현과 직접 대응하며, §4는 FLOPs와 measured speed를 구분해 평가한다.

**⑥ 수식·알고리즘 직관.** tensor를 channel 축으로 두 절반 `x₁,x₂`로 나누고 `y=concat(x₁,F(x₂))`로 계산하면 절반은 비용 없이 보존된다. shuffle은 channel을 `groups×channels_per_group`으로 reshape한 뒤 두 축을 transpose하고 다시 펼치는 순열이다. 값 자체를 계산하지 않고 배치만 바꿔 다음 block의 두 경로가 고정된 channel 집단에 갇히는 것을 막는다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/08_shufflenet_v2.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/08_shufflenet_v2.ipynb). TODO 1은 reshape–transpose 기반 channel shuffle을 구현한다. TODO 2는 Figure 3(c)의 절반 identity, 절반 pointwise–depthwise–pointwise branch를 만든다. TODO 3은 작은 분류기를 학습하고 FLOPs와 별도로 forward latency를 기록해 §4의 메시지를 확인한다.

- **포트폴리오 P1 → P2 → P3:** `ShufflenetV2PortfolioModel`의 split/branch/shuffle·loss/update를 완성하고, fit loop 뒤 test accuracy·`parameter_count`·depthwise groups 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** “싸 보이는 연산량보다 장치에서 빠른 데이터 흐름을 설계하라”가 5분 요약이다. 지침은 측정한 하드웨어와 라이브러리에 의존하고, Python 수준의 짧은 latency 측정은 잡음이 크며, shuffle·concat도 장치에 따라 무시할 수 없는 비용이 된다.

1. 같은 FLOPs인데도 두 모델의 실제 latency가 다른 원인은 무엇인가?
2. channel shuffle이 없다면 반복된 split block에서 어떤 정보 고립이 생기는가?
3. latency를 공정하게 비교하려면 warm-up, 반복 횟수, batch size를 어떻게 통제해야 하는가?

## 09. Once-for-All — Once-for-All: Train One Network and Specialize it for Efficient Deployment (2020)

**① 한 문장 요약.** 하나의 큰 supernet을 progressive shrinking으로 학습해 kernel·depth·width·resolution이 다른 많은 subnet을 품고, 각 장치 제약에 맞는 subnet을 추가 학습 없이 선택한다.

**② 왜 나왔나 / 이전 한계.** 장치마다 latency, 메모리, 에너지 예산이 다른데, 각 제약에 대해 neural architecture search를 다시 수행하고 모델을 처음부터 학습하면 비용이 매우 크다. 단순 weight sharing supernet은 수많은 작은 subnet이 충분히 학습되지 않거나 서로의 gradient가 충돌할 수 있다. OFA는 큰 모델의 지식을 보존하며 검색 공간을 점진적으로 확장한다.

**③ 읽기 전 기초지식.** neural architecture search, supernet과 subnet, weight sharing/slicing, elastic width·depth·kernel·resolution, progressive shrinking, hardware constraint.

**④ 핵심 아이디어.**

- 하나의 최대 network가 여러 kernel size, depth, width, input resolution 선택을 weight sharing으로 포함한다.
- 가장 큰 network를 먼저 학습한 뒤 kernel→depth→width 순으로 작은 선택지를 점진적으로 활성화한다.
- 큰 subnet의 출력을 작은 subnet의 학습 표적으로 활용해 shared weight의 간섭을 완화한다.
- 배포 시 latency 예산을 만족하는 후보만 평가해 정확도가 가장 높은 subnet을 specialize한다.

**⑤ 꼭 볼 원문 위치.** [OpenReview 원문](https://openreview.net/forum?id=HylxE1HKwS)의 Figure 1에서 “train once, specialize for deployment” 흐름을 본다. §3.1은 elastic kernel/depth/width/resolution, §3.2는 progressive shrinking, §3.3은 장치별 specialization을 설명한다.

**⑥ 수식·알고리즘 직관.** 최대 폭 weight `W[:24,:]`의 앞부분을 잘라 폭 `8/12/16/24` subnet이 공유한다고 생각하면 된다. 처음부터 모든 크기를 무작위로 학습하면 같은 weight가 상충하는 요구를 받는다. 먼저 큰 모델을 안정화하고 작은 모델을 하나씩 추가하면 큰 모델이 움직이는 교사 역할을 한다. 마지막 선택은 `max accuracy(a)` subject to `latency(a)≤budget`인 제약 최적화다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/distillation_compression/exercises/09_once_for_all.ipynb) / [정답본](../../notebooks/field_reproductions/distillation_compression/solutions/09_once_for_all.ipynb). TODO 1은 최대 폭 24의 weight를 slicing해 8/12/16/24 subnet을 만든다. TODO 2는 §3.2처럼 large-to-small progressive shrinking으로 공유 weight를 학습한다. TODO 3은 latency budget을 만족하는 subnet 중 정확도가 가장 높은 것을 골라 §3.3의 specialization을 축소 재현한다.

- **포트폴리오 P1 → P2 → P3:** `OnceForAllPortfolioModel`의 elastic-width slice·shared loss/update를 완성하고, fit loop 뒤 width별 accuracy·latency budget 이하 feasible-subnet 수 계약을 검증한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 5분 요약은 “한 supernet을 큰 것부터 작은 것까지 안정적으로 학습하고, 배포 때 제약에 맞는 일부만 꺼낸다”이다. supernet 학습과 search space 설계가 복잡하고, weight-sharing 순위가 독립 학습 순위와 다를 수 있으며, latency 예측기가 새 하드웨어에 일반화하지 않을 수 있다.

1. 모든 subnet을 처음부터 동시에 학습하는 것보다 progressive shrinking이 안정적인 이유는 무엇인가?
2. width slicing이 가능한 weight-sharing 구조에는 어떤 제약이 필요한가?
3. parameter 수가 가장 작은 subnet이 latency budget에서 항상 최선이 아닌 이유는 무엇인가?

---

## 최신 top-tier 후속 읽기 7편 (2023–2025)

### 선정 기준과 읽는 법

아래 목록은 2026년 8월 기준으로 ① ICML·MLSys·ICLR·NeurIPS의 공식 proceedings에 채택 사실과 연도가 확인되고, ② 대규모 Transformer 또는 diffusion model의 pruning·quantization을 핵심 문제로 다루며, ③ 앞의 고전 10편에서 익힌 magnitude, activation, affine quantization, low-rank decomposition, hardware-aware inference를 한 단계 확장하는 논문을 골랐다. arXiv 표기만으로 venue를 추정하지 않았으며 각 링크는 해당 학회의 공식 논문 페이지다. 서로 비슷한 정확도 숫자를 나열하기보다 “무엇을 보존하면서 어떤 계산을 줄이는가”를 비교해 읽는 것이 좋다.

#### A. SparseGPT — Massive Language Models Can Be Accurately Pruned in One-Shot

- **venue / year:** ICML 2023. [PMLR 공식 논문 페이지](https://proceedings.mlr.press/v202/frantar23a.html)에 ICML 40회 proceedings와 2023년 출판 정보가 명시되어 있다.
- **왜 주목할 만한가:** 거대한 언어 모델을 전체 재학습하지 않고 layer 단위의 one-shot pruning 문제로 다뤘다. 단순 magnitude가 아니라 작은 calibration 입력에서 얻은 2차 정보의 근사를 사용해 여러 weight를 제거할 때 남은 weight를 보정한다. 고전 Deep Compression과 Lottery Ticket이 “희소 구조를 어떻게 얻는가”를 다뤘다면, 이 논문은 dense 재학습이 부담스러운 LLM 규모에서 그 질문을 현실화한다.
- **읽기 전 선수지식:** unstructured/semi-structured sparsity, Hessian 또는 2차 근사, layer-wise reconstruction, calibration set, weight quantization.
- **이 목록의 선정 이유:** 2023년 이후 LLM one-shot pruning 흐름의 대표 출발점이며, pruning과 quantization을 함께 생각할 수 있는 연결 고리라서 골랐다.

#### B. SmoothQuant — Accurate and Efficient Post-Training Quantization for Large Language Models

- **venue / year:** ICML 2023. [PMLR 공식 논문 페이지](https://proceedings.mlr.press/v202/xiao23c.html)에서 venue와 연도를 확인할 수 있다.
- **왜 주목할 만한가:** LLM의 activation outlier 때문에 activation까지 낮은 비트로 양자화하기 어려운 문제를, 함수 출력을 바꾸지 않는 channel별 rescaling으로 weight 쪽에 이전한다. 학습 없이 적용하는 post-training 방식이라는 점과 weight뿐 아니라 activation을 함께 정수화하려는 점이 Integer-only Quantization의 현대적 확장이다.
- **읽기 전 선수지식:** post-training quantization(PTQ), activation outlier, per-channel scale, W8A8 표기, affine quantization, Transformer linear layer.
- **이 목록의 선정 이유:** “outlier를 없애지 않고 더 양자화하기 쉬운 곳으로 이동한다”는 직관이 명확하고 이후 LLM 양자화 논문들의 기준점이 되었기 때문에 선정했다.

#### C. AWQ — Activation-aware Weight Quantization for LLM Compression and Acceleration

- **venue / year:** MLSys 2024. [MLSys 공식 proceedings](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)에 2024 conference 논문으로 등재되어 있다.
- **왜 주목할 만한가:** 모든 weight가 똑같이 중요하지 않으며, activation 통계가 salient weight channel을 알려 준다는 관점에서 weight-only 저비트 양자화를 설계한다. 중요한 channel의 scale을 조절해 양자화 오차를 줄이고, 학습이나 큰 reconstruction 절차 없이 실제 inference kernel까지 연결한다.
- **읽기 전 선수지식:** weight-only quantization, activation statistics, salient channel, group-wise scale, memory bandwidth, GPU kernel.
- **이 목록의 선정 이유:** 압축 알고리즘의 오차 감소와 시스템 수준 가속을 함께 제시한 MLSys 논문으로, Deep Compression의 weight sharing과 하드웨어 실현 사이를 현대 LLM에서 잇기 때문에 골랐다.

#### D. Wanda — A Simple and Effective Pruning Approach for Large Language Models

- **venue / year:** ICLR 2024. [ICLR 공식 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html)에서 2024 채택 논문임을 확인할 수 있다.
- **왜 주목할 만한가:** 각 weight의 절댓값에 그 weight가 받는 입력 activation의 크기를 결합한 단순한 importance score로 LLM을 pruning한다. 복잡한 weight update나 재학습 없이도 data-aware 신호를 사용하는 것이 핵심이며, SparseGPT와 함께 정확도–복잡도 trade-off를 비교하기 좋다.
- **읽기 전 선수지식:** magnitude pruning, activation norm, per-output importance ranking, calibration samples, unstructured와 structured sparsity.
- **이 목록의 선정 이유:** 구현 가능한 간단한 기준선이면서 “weight 크기만 보면 충분한가?”라는 질문에 activation을 결합한 답을 주므로 교육용 후속 읽기로 선정했다.

#### E. QuaRot — Outlier-Free 4-Bit Inference in Rotated LLMs

- **venue / year:** NeurIPS 2024. [NeurIPS 공식 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html)에 NeurIPS 2024 main conference 논문으로 수록되어 있다.
- **왜 주목할 만한가:** LLM의 함수 출력을 보존하는 직교 회전을 network에 삽입·흡수해 hidden state의 outlier를 여러 차원으로 퍼뜨린다. 이를 통해 weight뿐 아니라 activation과 KV cache까지 4-bit로 다루기 쉬운 분포로 바꾼다. SmoothQuant의 rescaling과 비교하면 outlier 문제에 대한 “이동”과 “회전”이라는 두 좌표변환을 볼 수 있다.
- **읽기 전 선수지식:** orthogonal/Hadamard rotation, RMSNorm, activation outlier, KV cache, weight–activation equivalence, 4-bit quantization.
- **이 목록의 선정 이유:** 수학적으로 출력을 보존하는 재매개변수화와 실제 저비트 inference를 연결하며, 후속 learned-rotation 연구를 이해하는 필수 전 단계여서 골랐다.

#### F. SpinQuant — LLM Quantization with Learned Rotations

- **venue / year:** ICLR 2025. [ICLR 공식 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html)에 2025 채택 논문으로 등재되어 있다.
- **왜 주목할 만한가:** random 또는 고정 회전도 선택에 따라 양자화 결과가 달라진다는 관찰에서 출발해, full-precision 출력을 보존하는 회전 매개변수 공간에서 양자화에 유리한 회전을 학습한다. QuaRot을 “가능한 회전 중 어느 것을 선택할 것인가?”라는 최적화 문제로 확장한다.
- **읽기 전 선수지식:** QuaRot, Cayley transform 또는 직교행렬 최적화, quantization error, rotation invariance, STE/학습 기반 PTQ.
- **이 목록의 선정 이유:** 2025년의 learned rotation 흐름을 대표하고, 고정 변환과 데이터에 맞춘 변환의 trade-off를 명확히 비교할 수 있어 선정했다.

#### G. SVDQuant — Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models

- **venue / year:** ICLR 2025. [ICLR 공식 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html)에서 venue와 2025년을 확인할 수 있다.
- **왜 주목할 만한가:** diffusion model의 weight와 activation을 4-bit로 낮출 때 outlier를 먼저 이동시키고, 그 어려운 성분을 고정밀 low-rank branch에 흡수하며 나머지 residual만 저비트로 계산한다. 알고리즘뿐 아니라 두 branch를 효율적으로 실행하는 inference engine까지 함께 다룬다.
- **읽기 전 선수지식:** diffusion/DiT 구조, singular value decomposition(SVD), low-rank residual, activation smoothing, W4A4, kernel fusion.
- **이 목록의 선정 이유:** LLM 중심 목록을 생성형 vision으로 확장하고, quantization과 low-rank decomposition, 시스템 최적화를 한 설계 안에서 비교할 수 있어 포함했다.

### 최신 7편을 묶어 보는 비교 질문

1. SparseGPT와 Wanda는 둘 다 재학습 없는 pruning을 지향하지만, 중요도를 추정하고 남은 weight를 다루는 복잡성이 어떻게 다른가?
2. SmoothQuant, QuaRot, SpinQuant는 모두 outlier 문제를 좌표변환으로 완화한다. scale 이동, 고정 회전, 학습 회전의 비용과 유연성을 비교해 보라.
3. AWQ와 SVDQuant는 일부 중요한 성분을 보호한다는 공통점이 있다. channel scaling과 high-precision low-rank branch가 각각 어떤 오차를 격리하는가?
