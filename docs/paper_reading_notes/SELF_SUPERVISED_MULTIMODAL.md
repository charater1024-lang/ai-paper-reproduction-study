# 자기지도 학습·멀티모달 논문 독해 노트

> **AI 보조 작성 안내:** 이 한국어 요약은 공식 번역이 아닙니다. 내용과 인용의 최종 기준은
> 연결된 원문입니다. [작성 원칙과 한계](AI_ASSISTED_NOTICE.md) · [70편 목차](README.md)

이 문서는 contrastive/self-distillation/masked prediction에서 vision–language 연결까지 한 흐름으로 읽도록 구성했다. 모든 노트북은 로컬 `data/field_curriculum/multimodal_pairs.npz`만 사용한다. 원 논문의 인터넷 규모 데이터·대형 backbone을 복제하는 대신, 핵심 loss와 정보 경로가 실제로 동작하는지를 CPU에서 짧게 재현한다.

권장 학습법은 **핵심 수식 Task → portfolio class 조립 → contract 검증**이다. 번호가 붙은 TODO에서 view·modality 처리와 논문 objective를 먼저 구현하고, 포트폴리오 P1에서 `PortfolioConfig`, modality 준비, `forward`·`compute_loss`·`training_step`을 명시적 class로 조립한다. P2의 최적화 loop와 P3의 finite history·representation/retrieval metric 검증까지 마쳐야 한 편의 실습이 끝난다.

---

## 00. CPC — Representation Learning with Contrastive Predictive Coding (2018)

원문: [CPC](https://arxiv.org/abs/1807.03748)

### ① 한 문장 요약

과거로부터 만든 context가 여러 후보 중 실제 미래 latent를 식별하도록 InfoNCE를 학습해, 관측의 세부 잡음보다 예측 가능한 고수준 정보를 표현에 남긴다.

### ② 배경과 이전 접근의 한계

픽셀·파형을 그대로 복원하는 생성 목적은 표현에 불필요한 저수준 세부까지 모델링해야 한다. mutual information 최대화는 좋은 원칙이지만 직접 계산하기 어렵다. CPC는 future prediction을 density-ratio classification으로 바꾸고, 같은 batch의 다른 sample을 negative로 써 tractable한 lower bound를 만든다.

### ③ 선수지식

- sequence encoder와 autoregressive model/GRU
- conditional probability와 mutual information
- contrastive classification·cross-entropy
- dot product 또는 bilinear score
- in-batch negative sampling

### ④ 핵심 아이디어

- encoder `g_enc`가 각 관측 `x_t`를 latent `z_t`로 압축한다.
- autoregressive model `g_ar`가 과거 latent만 보고 context `c_t`를 만든다.
- context와 실제 미래 latent의 score를 높이고 다른 sequence의 미래를 낮춘다.
- `N`개 후보의 InfoNCE loss가 작아질수록 context–future mutual information lower bound가 커진다.

### ⑤ 꼭 볼 원문 위치

- **§2.1, Eq. (1)**: 관측 `x_t`에서 latent `z_t`를 만드는 encoder.
- **§2.1, Eq. (2)**: 과거 `z_≤t`만 요약하는 autoregressive context `c_t`.
- **§2.2, Eq. (3)**: bilinear density-ratio score `exp(zᵀW_k c)`.
- **§2.3, Eq. (4)**: 하나의 positive와 `N-1` negatives를 구분하는 InfoNCE.
- **Figure 1**: encoder–context–future prediction의 전체 정보 흐름.

### ⑥ 수식·알고리즘 직관

각 query context를 행, batch의 future latent를 열로 놓으면 score가 `N×N` 행렬이 된다. 대각선은 같은 sequence의 올바른 미래이고 나머지는 negative다. 각 행에서 대각선 class를 맞히는 cross-entropy가 InfoNCE다. 미래를 raw input으로 복원하지 않고 latent 중 정답을 찾으므로 공통·예측 가능 신호에 집중한다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/00_cpc.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/00_cpc.ipynb)
- **TODO 00-1 ↔ Eq. (1)–(2)**: point encoder와 GRU context를 만들고 미래를 보지 않는 prefix shape을 검증한다.
- **TODO 00-2 ↔ Eq. (3)–(4)**: normalized bilinear score와 in-batch InfoNCE를 구현한다.
- **TODO 00-3 ↔ Figure 1**: loss를 학습하고 positive-future top-1과 logit heatmap을 chance와 비교한다.
- **포트폴리오 P1 → P2 → P3**: `CpcPortfolioModel`의 sequence 준비·encoder/context·InfoNCE/update를 완성하고, fit loop 뒤 future top-1·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** CPC는 “과거 context로 진짜 미래 latent를 batch 후보 중 찾기”이며, 어려운 mutual information 문제를 분류 문제로 바꾼다.
- **한계:** negative 수·품질에 민감하고 batch 내 false negative가 생기며, 어떤 horizon을 예측할지에 따라 표현이 달라지고 sequence의 비예측 정보는 버릴 수 있다.
- **자가점검:**
  1. logit 행렬의 대각선이 positive인 이유는 무엇인가?
  2. horizon이 너무 짧거나 너무 길면 각각 어떤 shortcut·난점이 생기는가?
  3. raw reconstruction보다 latent discrimination이 고수준 표현을 유도할 수 있는 이유는?

---

## 01. MoCo — Momentum Contrast for Unsupervised Visual Representation Learning (CVPR 2020)

원문: [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)

### ① 한 문장 요약

천천히 갱신되는 key encoder와 FIFO queue를 사용해 batch 크기와 분리된 크고 일관된 negative dictionary로 contrastive representation을 학습한다.

### ② 배경과 이전 접근의 한계

instance discrimination은 많은 negatives가 필요하지만 batch를 크게 하면 메모리와 통신 비용이 커진다. 이전 memory bank는 많은 key를 저장하지만 encoder가 급히 변하면 오래된 key와 새 query의 표현 공간이 불일치한다. MoCo는 queue로 규모를, momentum encoder로 dictionary의 일관성을 확보한다.

### ③ 선수지식

- contrastive/InfoNCE loss
- 두 augmentation view
- exponential moving average(EMA)
- queue의 enqueue/dequeue
- stop-gradient

### ④ 핵심 아이디어

- query encoder는 backprop으로, key encoder는 query parameter의 EMA로 갱신한다.
- 현재 positive key와 queue의 오래된 negative keys를 함께 분류한다.
- 새 mini-batch key를 enqueue하고 가장 오래된 key를 dequeue한다.
- key branch로 gradient를 보내지 않아 안정적인 target dictionary를 유지한다.

### ⑤ 꼭 볼 원문 위치

- **§3.1, Eq. (1)**: query, positive key, negative keys에 대한 contrastive loss.
- **§3.2, Eq. (2)**: `θ_k ← mθ_k +(1-m)θ_q` momentum update.
- **§3.2 “Dictionary as a queue”**: batch와 dictionary 크기를 분리하는 FIFO 설계.
- **Algorithm 1**: shuffle BN, stop-gradient, enqueue/dequeue를 포함한 학습 순서.
- **Figure 1**: end-to-end, memory bank, MoCo dictionary 비교.

### ⑥ 수식·알고리즘 직관

logits는 `[q·k_+, q·k_1, …, q·k_K]/τ`이고 target index는 항상 첫 열이다. 큰 `m`은 key encoder를 천천히 움직여 여러 step에서 저장된 queue key가 비슷한 좌표계에 있게 한다. queue는 gradient graph가 아니라 feature cache이므로 큰 negative set을 작은 batch로 유지한다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/01_moco.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/01_moco.ipynb)
- **TODO 01-1 ↔ Eq. (2)**: query parameter를 이용한 key encoder EMA를 구현하고 수치적으로 확인한다.
- **TODO 01-2 ↔ Eq. (1)**: positive 1개와 queue negatives의 logits·target을 만든다.
- **TODO 01-3 ↔ Algorithm 1**: momentum update, stop-gradient, FIFO queue를 연결해 학습하고 positive similarity가 queue 평균보다 큰지 본다.
- **포트폴리오 P1 → P2 → P3**: `MocoPortfolioModel`의 두 view·query/key/queue loss·EMA update를 완성하고, fit loop 뒤 view agreement·feature spread·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** MoCo는 loss보다 dictionary 엔지니어링이 핵심이다. queue가 “크기”, EMA가 시간에 걸친 “일관성”을 담당한다.
- **한계:** queue의 stale negative와 false negative가 남고, momentum·queue 길이 tuning이 필요하며, 강한 augmentation 가정과 대규모 pretraining 비용이 있다.
- **자가점검:**
  1. key encoder도 backprop으로 즉시 갱신하면 queue 일관성이 왜 나빠지는가?
  2. momentum `m→1`과 `m→0`은 각각 어떤 trade-off를 만드는가?
  3. queue를 매 step detach해야 하는 이유는 무엇인가?

---

## 02. SimCLR — A Simple Framework for Contrastive Learning of Visual Representations (ICML 2020)

원문: [SimCLR](https://proceedings.mlr.press/v119/chen20j.html)

### ① 한 문장 요약

복잡한 memory mechanism 없이 강한 augmentation, projection head, 큰 batch의 symmetric NT-Xent만으로 강력한 self-supervised visual representation을 학습한다.

### ② 배경과 이전 접근의 한계

초기 contrastive 방법들은 memory bank, specialized architecture, multi-stage procedure가 얽혀 어떤 요소가 성능을 내는지 불명확했다. SimCLR은 파이프라인을 단순화하고 체계적 ablation을 통해 augmentation 조합, nonlinear projection head, normalized temperature-scaled loss가 핵심임을 보였다.

### ③ 선수지식

- image augmentation과 invariance
- encoder·projection MLP
- cosine similarity와 temperature
- in-batch negatives
- cross-entropy와 linear evaluation

### ④ 핵심 아이디어

- 한 이미지에서 독립적인 두 augmentation view를 만들어 positive pair로 둔다.
- encoder representation `h` 뒤에 학습 중에만 쓰는 nonlinear projection `g(h)=z`를 둔다.
- batch의 `2N` views에서 각 view의 짝만 positive, 나머지를 negative로 둔다.
- 두 방향 loss를 평균하는 symmetric NT-Xent를 사용한다.

### ⑤ 꼭 볼 원문 위치

- **Figure 2**: augmentation `t,t'`, shared encoder `f`, projection head `g`, contrastive loss 흐름.
- **§2.1, Eq. (1)**: L2-normalized representation의 cosine similarity.
- **§2.1, Eq. (2)**: positive index를 맞히는 NT-Xent.
- **Algorithm 1**: `2N` view를 한 번 encode하고 symmetric loss를 계산하는 절차.
- **§3.1–§3.2, Tables 3–5**: augmentation과 projection head ablation.

### ⑥ 수식·알고리즘 직관

각 anchor `i`에 대해 짝 `j`의 `exp(sim(z_i,z_j)/τ)`가 분자이고, 자기 자신을 제외한 모든 view가 분모다. 작은 `τ`는 높은 유사도 차이를 날카롭게 확대한다. loss는 projection 공간 `z`에서 계산하지만 downstream에는 projection 전 `h`를 써서 contrastive 목적이 버린 정보 일부를 보존한다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/02_simclr.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/02_simclr.ipynb)
- **TODO 02-1 ↔ Figure 2**: 한 batch에서 서로 다른 두 stochastic view를 만든다.
- **TODO 02-2 ↔ Eq. (1)–(2)**: diagonal self-similarity를 mask하고 symmetric NT-Xent를 구현한다.
- **TODO 02-3 ↔ Algorithm 1**: encoder·projector를 학습하고 같은 sample의 view cosine이 다른 sample보다 커지는지 확인한다.
- **포트폴리오 P1 → P2 → P3**: `SimclrPortfolioModel`의 두 view·encoder/projector·NT-Xent/update를 완성하고, fit loop 뒤 view-retrieval top-1·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** SimCLR의 세 축은 “좋은 view 정의, projection 공간에서의 symmetric contrast, 충분한 negatives”다.
- **한계:** 큰 batch와 긴 학습이 필요하고, augmentation이 보존해야 할 task 정보까지 없앨 수 있으며, 같은 class를 false negative로 밀어낼 수 있다.
- **자가점검:**
  1. anchor 자기 자신을 denominator에서 제외해야 하는 이유는?
  2. projection head를 downstream에서 버리는 이유를 어떻게 설명할 수 있는가?
  3. color jitter가 어떤 dataset에서는 해로운 invariance가 될 수 있는가?

---

## 03. BYOL — Bootstrap Your Own Latent (NeurIPS 2020)

원문: [BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)

### ① 한 문장 요약

negative pair 없이 online network가 EMA target network의 다른 view 표현을 예측하게 하고, predictor와 stop-gradient의 비대칭으로 collapse를 피한다.

### ② 배경과 이전 접근의 한계

SimCLR·MoCo는 많은 negatives와 false-negative 문제를 감수한다. 단순히 같은 이미지의 두 view를 같게만 만들면 모든 입력을 상수 벡터로 보내는 collapse가 global optimum이다. BYOL은 target을 느리게 움직이는 bootstrap 신호로 만들고 online 쪽에만 predictor를 두는 비대칭으로 유용한 학습을 가능하게 했다.

### ③ 선수지식

- Siamese/two-view learning
- EMA와 stop-gradient
- encoder·projector·predictor
- cosine distance와 L2 normalization
- representation collapse

### ④ 핵심 아이디어

- online branch는 encoder, projector, predictor를 모두 학습한다.
- target branch는 encoder와 projector만 갖고 online parameter의 EMA로 갱신한다.
- online prediction이 반대 view의 target projection을 맞히는 normalized MSE를 쓴다.
- 두 view의 방향을 바꿔 같은 loss를 한 번 더 계산한다.

### ⑤ 꼭 볼 원문 위치

- **§3.1, Figure 2**: online/target branch, predictor, stop-gradient의 비대칭 구조.
- **§3.1, Eq. (1)**: target parameter의 exponential moving average.
- **§3.1, Eq. (2)**: L2-normalized prediction–target regression loss.
- **§3.1, Eq. (3)**: 두 augmentation 방향의 symmetric loss.
- **§5 ablation**: target network·predictor가 제거될 때의 collapse와 성능 변화.

### ⑥ 수식·알고리즘 직관

정규화 벡터의 squared distance는 `2-2·cosine`이므로, BYOL loss는 online prediction과 target의 cosine을 높인다. target에는 gradient가 흐르지 않고 EMA로만 움직여 online network가 쫓아갈 비교적 안정적인 목표가 된다. predictor는 두 branch의 역할을 다르게 만들어 상수 해로 바로 수렴하는 동역학을 바꾼다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/03_byol.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/03_byol.ipynb)
- **TODO 03-1 ↔ Eq. (1)**: target network를 online network의 EMA로 갱신한다.
- **TODO 03-2 ↔ Eq. (2)–(3)**: normalized MSE와 양방향 view loss를 구현한다.
- **TODO 03-3 ↔ Figure 2**: online만 optimizer로 갱신하며 학습하고 loss 감소와 embedding dimension별 표준편차로 collapse 여부를 본다.
- **포트폴리오 P1 → P2 → P3**: `ByolPortfolioModel`의 online/target/predictor·normalized loss·EMA update를 완성하고, fit loop 뒤 view agreement·feature spread·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** BYOL은 negatives 대신 “느린 자기 자신”을 target으로 삼는 asymmetric self-distillation이다.
- **한계:** collapse 회피의 원인이 optimizer·normalization·predictor와 얽혀 있고, augmentation·EMA schedule에 민감하며, 대규모 compute 요구와 shortcut 가능성이 남는다.
- **자가점검:**
  1. target parameter가 optimizer parameter 목록에 들어가면 안 되는 이유는?
  2. embedding 표준편차가 거의 0이면 무엇을 뜻하는가?
  3. predictor를 target branch에도 대칭으로 두면 어떤 핵심 비대칭이 사라지는가?

---

## 04. DINO — Emerging Properties in Self-Supervised Vision Transformers (ICCV 2021)

원문: [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)

### ① 한 문장 요약

EMA teacher의 sharpened·centered class-probability 분포를 student가 다른 crop에서 맞히게 해 label 없이 ViT의 semantic feature와 object-like attention을 학습한다.

### ② 배경과 이전 접근의 한계

CNN 중심의 self-supervised 표현은 강했지만 ViT의 잠재적 semantic attention이 label 없이 어떻게 나타나는지 불분명했다. BYOL류 회귀만으로는 teacher 출력 분포의 collapse를 제어하는 장치가 제한적이다. DINO는 prototype-like output distribution, multi-crop, teacher temperature, centering을 결합한 self-distillation을 제안했다.

### ③ 선수지식

- Vision Transformer와 `[CLS]` token
- knowledge distillation·cross-entropy
- EMA teacher와 stop-gradient
- softmax temperature
- multi-crop augmentation과 collapse

### ④ 핵심 아이디어

- student와 teacher가 같은 architecture를 쓰되 teacher는 EMA로만 갱신한다.
- teacher는 global crop만, student는 global·local crop을 모두 본다.
- 서로 같은 crop인 경우를 제외하고 teacher distribution을 student가 예측한다.
- teacher logits의 batch center를 빼고 낮은 temperature로 sharpen해 상수·단일 prototype collapse를 억제한다.

### ⑤ 꼭 볼 원문 위치

- **§3, Figure 2**: multi-crop student–teacher 학습 파이프라인.
- **§3, Eq. (1)–(2)**: temperature softmax와 cross-view cross-entropy.
- **§3, Eq. (3)**: teacher parameter EMA update.
- **§3.2 “Avoiding collapse”**: centering과 sharpening의 상보적 역할.
- **Figure 3**: self-attention이 객체 영역을 드러내는 qualitative 결과.

### ⑥ 수식·알고리즘 직관

teacher 확률 `P_t(x)=softmax((g_t(x)-c)/τ_t)`를 target으로, student `P_s(x')`와 cross-entropy를 계산한다. 작은 teacher temperature는 목표를 확신 있게 만들고, moving center `c`는 특정 output dimension만 항상 우세해지는 것을 막는다. 둘의 균형이 너무 날카로운 delta collapse와 완전 균일 collapse 사이를 제어한다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/04_dino.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/04_dino.ipynb)
- **TODO 04-1 ↔ Eq. (1)**: center·서로 다른 temperature를 적용한 teacher/student 분포를 만든다.
- **TODO 04-2 ↔ Eq. (2)–(3)**: cross-view CE와 teacher EMA update를 구현한다.
- **TODO 04-3 ↔ §3.2**: center를 batch teacher logits의 EMA로 갱신하며 학습하고 loss·teacher entropy로 collapse를 감시한다.
- **포트폴리오 P1 → P2 → P3**: `DinoPortfolioModel`의 multi-view student/teacher·centered CE·EMA update를 완성하고, fit loop 뒤 view agreement·feature spread·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** DINO는 BYOL식 EMA teacher를 확률분포 distillation으로 바꾸고, centering+sharpening으로 label 없는 prototype 사용을 균형 잡는다.
- **한계:** temperature·center·teacher momentum schedule이 민감하고 긴 학습이 필요하며, attention map이 semantic해 보여도 segmentation supervision이나 인과 설명은 아니다.
- **자가점검:**
  1. centering과 sharpening은 각각 어떤 collapse 방향을 억제하는가?
  2. teacher가 local crop을 보지 않는 설계의 직관은 무엇인가?
  3. teacher entropy가 0 또는 `log K`에 가까우면 각각 무엇을 의심해야 하는가?

---

## 05. MAE — Masked Autoencoders Are Scalable Vision Learners (CVPR 2022)

원문: [MAE](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)

### ① 한 문장 요약

이미지 patch의 대부분을 가리고 보이는 patch만 무거운 ViT encoder에 넣은 뒤 가벼운 decoder가 가려진 픽셀만 복원하게 해, 효율적이고 확장 가능한 visual pretraining을 만든다.

### ② 배경과 이전 접근의 한계

BERT의 masked token prediction은 vision에서도 매력적이지만, 이미지 pixel은 인접 중복이 커서 낮은 mask 비율이면 주변 복사라는 쉬운 shortcut이 생긴다. 기존 masked image modeling은 mask token까지 encoder에 넣어 고해상도에서 계산을 낭비했다. MAE는 높은 masking ratio와 asymmetric encoder–decoder로 난이도와 효율을 함께 해결한다.

### ③ 선수지식

- Vision Transformer와 patch embedding
- autoencoder와 reconstruction loss
- masking·gather/scatter indexing
- MSE와 per-patch normalization
- pretrain–finetune 패러다임

### ④ 핵심 아이디어

- 무작위로 약 75% patch를 제거해 spatial redundancy를 줄인다.
- encoder는 visible patch만 처리하므로 self-attention 비용을 크게 절감한다.
- 작은 decoder 입력에서 mask token을 원래 위치에 다시 끼워 전체 sequence를 복원한다.
- loss는 masked patch의 pixel prediction에만 계산한다.

### ⑤ 꼭 볼 원문 위치

- **§3.1 및 Figure 1**: 높은 random masking ratio와 비대칭 전체 파이프라인.
- **§3.2 “MAE encoder”**: mask token 없이 visible patches만 encoder에 넣는 설계.
- **§3.3 “MAE decoder”**: encoded visibles와 mask tokens를 원위치에 합치는 절차.
- **§3.4 “Reconstruction target”**: masked patch에만 적용하는 pixel MSE와 normalized target.
- **Figure 5**: masking ratio 변화에 따른 정확도·속도 trade-off.

### ⑥ 수식·알고리즘 직관

64개 patch 중 16개만 보이면 encoder의 attention pair는 `64²`가 아니라 `16²` 규모다. decoder는 위치 embedding을 이용해 mask token이 어느 patch를 복원해야 하는지 안다. 보이는 patch까지 loss에 넣으면 identity 복사가 objective를 지배할 수 있으므로, 실제 학습 신호는 숨긴 위치에만 둔다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/05_mae.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/05_mae.ipynb)
- **TODO 05-1 ↔ §3.1**: `8×8` 이미지를 patch sequence로 바꾸고 `unpatchify`로 정확히 되돌린다.
- **TODO 05-2 ↔ §3.1–§3.2**: seed 고정 random mask와 visible/restoration index를 만든다.
- **TODO 05-3 ↔ Figure 1**: visible-only encoder와 mask-token decoder를 갖는 `TinyMAE`를 구성한다.
- **TODO 05-4 ↔ §3.4**: masked patch MSE만 학습하고 reconstruction·loss curve를 시각화한다.
- **포트폴리오 P1 → P2 → P3**: `MaePortfolioModel`의 patch/mask 준비·visible encoder/decoder·masked loss/update를 완성하고, fit loop 뒤 masked-patch MSE·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** MAE의 핵심은 “많이 가려 어렵게 만들고, 가린 token은 비싼 encoder에서 빼 효율을 얻는다”이다.
- **한계:** pixel MSE가 semantic 중요도를 반영하지 않고 pretraining–finetuning architecture가 비대칭이며, 데이터·모델 규모가 작으면 높은 mask 비율이 지나치게 어려울 수 있다.
- **자가점검:**
  1. 75%처럼 높은 mask ratio가 이미지에서 가능한 이유는 무엇인가?
  2. mask token을 encoder에 넣지 않을 때 계산량이 어떻게 줄어드는가?
  3. visible patch에 reconstruction loss를 주지 않는 이유는?

---

## 06. CLIP — Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)

원문: [CLIP](https://proceedings.mlr.press/v139/radford21a.html)

### ① 한 문장 요약

대규모 image–text pair의 두 encoder를 symmetric contrastive loss로 정렬하고, class 이름을 자연어 prompt로 바꿔 별도 label 학습 없이 zero-shot classifier를 만든다.

### ② 배경과 이전 접근의 한계

전통 visual model은 고정 class label의 curated dataset에 의존해 새 task마다 labeled data와 classifier fine-tuning이 필요했다. caption generation처럼 모든 단어를 순서대로 예측하는 목적은 representation transfer 관점에서 비효율적일 수 있다. CLIP은 web text를 유연한 supervision으로 사용하고 matching 문제로 단순화했다.

### ③ 선수지식

- image encoder·text Transformer
- contrastive learning과 in-batch negatives
- cosine similarity와 learnable temperature
- zero-shot classification·prompt template
- retrieval Recall@K

### ④ 핵심 아이디어

- image와 text를 별도 encoder로 embedding한 뒤 unit norm으로 정규화한다.
- batch의 모든 image–text cosine으로 `N×N` similarity matrix를 만든다.
- image→text와 text→image cross-entropy를 평균한다.
- 각 class 이름의 prompt embedding을 classifier weight처럼 사용해 zero-shot 예측한다.

### ⑤ 꼭 볼 원문 위치

- **Figure 1, step (1)**: natural-language supervision으로 image–text representation을 함께 학습.
- **Figure 3 pseudocode**: projection, normalization, pairwise logits, symmetric loss의 정확한 순서.
- **Figure 1, step (2)**: class label을 prompt로 만들어 text encoder에 넣는 zero-shot classifier synthesis.
- **Figure 1, step (3) 및 §3.1**: image와 prompt embedding의 similarity로 class를 고르는 과정.
- **§2.3**: contrastive objective 선택과 scale의 역할.

### ⑥ 수식·알고리즘 직관

logit `L_ij = exp(t)·⟨I_i,T_j⟩`에서 대각선은 matching pair다. 각 image가 올바른 text를 찾는 row CE와 각 text가 올바른 image를 찾는 column CE를 평균하므로 양방향 retrieval 공간이 된다. zero-shot에서는 “a photo of a {class}” 같은 text embedding이 학습되지 않은 classifier의 class vector가 된다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/06_clip.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/06_clip.ipynb)
- **TODO 06-1 ↔ Figure 3**: image/text encoder와 projection을 가진 작은 dual encoder를 만든다.
- **TODO 06-2 ↔ Figure 3**: normalized similarity의 row·column CE를 평균한 CLIP loss를 구현한다.
- **TODO 06-3 ↔ Figure 1 step (1)**: 로컬 matching pairs로 pretraining하고 대각 retrieval 성능을 본다.
- **TODO 06-4 ↔ Figure 1 steps (2)–(3)**: class text prototype으로 zero-shot classification과 retrieval을 평가한다.
- **포트폴리오 P1 → P2 → P3**: `ClipPortfolioModel`의 image/text 준비·dual encoder·symmetric loss/update를 완성하고, fit loop 뒤 image-to-text Recall@1·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** CLIP은 web caption을 열린 class vocabulary로 바꾸는 dual-encoder contrastive pretraining이며, prompt embedding이 zero-shot classifier가 된다.
- **한계:** web 데이터의 사회적 편향·저작권·privacy 문제가 전달되고, fine-grained relation·counting·compositionality가 약하며, prompt 표현과 dataset shift에 성능이 민감하다.
- **자가점검:**
  1. row loss와 column loss를 모두 쓰는 이유는 무엇인가?
  2. embedding을 unit norm으로 만들면 logit이 무엇을 비교하게 되는가?
  3. prompt ensemble이 단일 class name보다 안정적일 수 있는 이유는?

---

## 07. ALIGN — Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ICML 2021)

원문: [ALIGN](https://proceedings.mlr.press/v139/jia21b.html)

### ① 한 문장 요약

복잡한 수동 정제 대신 10억 개가 넘는 noisy image alt-text pair의 규모를 활용해, 단순 dual encoder contrastive 학습으로 강한 zero-shot·retrieval 표현을 만든다.

### ② 배경과 이전 접근의 한계

기존 vision–language dataset은 사람의 annotation이나 여러 단계의 filtering이 필요해 크기가 제한됐다. CLIP이 natural-language supervision의 가능성을 보였지만 ALIGN은 “데이터가 noisy해도 충분한 scale이 이를 상쇄할 수 있는가”를 더 직접적으로 탐구했다. 모델 복잡성보다 데이터 수집 recipe와 scale을 중심에 둔다.

### ③ 선수지식

- CLIP식 dual encoder
- noisy label·weak supervision
- symmetric in-batch contrastive loss
- EfficientNet·BERT의 개념
- image–text retrieval

### ④ 핵심 아이디어

- image와 연결된 raw alt-text를 최소한의 frequency-based filtering만 거쳐 사용한다.
- image encoder와 text encoder 사이 cross-attention 없이 독립 embedding을 만든다.
- batch 전체를 negative pool로 쓰는 양방향 normalized softmax loss를 적용한다.
- scale이 noise를 상쇄하는지 zero-shot classification과 cross-modal retrieval로 검증한다.

### ⑤ 꼭 볼 원문 위치

- **§2.1 “Training Data”**: noisy alt-text pair 수집과 제한적 filtering 원칙.
- **§2.2, Eq. (1)**: image-to-text normalized softmax loss.
- **§2.2, Eq. (2)**: text-to-image 방향 loss와 최종 평균.
- **Figure 1**: 독립 image/text tower와 pairwise score 구조.
- **§3.2**: zero-shot transfer 및 image–text retrieval 평가.

### ⑥ 수식·알고리즘 직관

loss의 형태는 CLIP과 유사하지만 논문의 주된 독립변수는 data curation–scale trade-off다. 어떤 pair가 틀려도 거대한 corpus 전체에서 반복되는 올바른 시각–언어 상관이 평균적으로 signal을 제공한다는 가설이다. dual encoder는 모든 pair를 미리 embedding해 approximate nearest-neighbor search할 수 있어 serving에도 적합하다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/07_align.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/07_align.ipynb)
- **TODO 07-1 ↔ §2.1**: 일부 caption token을 다른 sample과 섞어 통제된 noisy pairs를 만든다.
- **TODO 07-2 ↔ §2.2, Eq. (1)–(2)**: 작은 ALIGN dual encoder와 symmetric loss를 구현한다.
- **TODO 07-3 ↔ §3.2**: noisy data로 학습한 뒤 clean test pair의 양방향 retrieval을 chance와 비교한다.
- **포트폴리오 P1 → P2 → P3**: `AlignPortfolioModel`의 noisy-pair 준비·dual encoder/loss/update를 완성하고, fit loop 뒤 clean image-to-text Recall@1·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** ALIGN의 질문은 “더 깨끗한 작은 데이터인가, 더 noisy한 거대 데이터인가?”이고, 단순 모델에서도 후자의 scale이 강력함을 보였다.
- **한계:** scale이 systematic bias나 잘못된 상관을 없애지 않으며 막대한 compute·환경 비용이 들고, 비공개 데이터 recipe는 완전한 재현과 auditing을 어렵게 한다.
- **자가점검:**
  1. random noise와 systematic bias는 scale 증가에 대해 어떻게 다르게 행동하는가?
  2. dual encoder가 cross-encoder보다 retrieval serving에 유리한 이유는?
  3. clean test 성능만으로 training pair의 안전성을 판단할 수 없는 이유는?

---

## 08. BLIP — Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (ICML 2022)

원문: [BLIP](https://proceedings.mlr.press/v162/li22n.html)

### ① 한 문장 요약

하나의 parameter-sharing encoder–decoder를 contrastive·matching·language-modeling 세 목적으로 학습하고, captioner와 filter가 noisy web caption을 생성·정제하는 CapFilt로 이해와 생성을 함께 강화한다.

### ② 배경과 이전 접근의 한계

CLIP·ALIGN 같은 dual encoder는 retrieval·zero-shot understanding에는 효율적이지만 자유로운 text generation에는 바로 맞지 않는다. 반대로 encoder–decoder 생성 모델은 retrieval 효율이 낮다. 또 web caption의 noise는 규모만 늘린다고 모두 해결되지 않는다. BLIP은 모델 구조와 데이터 품질을 동시에 bootstrap한다.

### ③ 선수지식

- CLIP/ALIGN contrastive learning
- Transformer encoder–decoder와 causal mask
- image–text matching binary classification
- language modeling cross-entropy
- pseudo-labeling·data filtering

### ④ 핵심 아이디어

- Multimodal mixture of Encoder-Decoder(MED)가 text encoder, image-grounded encoder, image-grounded decoder 역할을 parameter 공유한다.
- ITC는 unimodal 표현을 정렬하고, ITM은 세밀한 pair 일치 여부를, LM은 caption 생성을 학습한다.
- captioner가 web image에 synthetic caption을 추가한다.
- filter가 원래·synthetic caption 중 image와 맞지 않는 pair를 제거해 새 pretraining corpus를 만든다.

### ⑤ 꼭 볼 원문 위치

- **§3.1 및 Figure 2**: MED의 세 기능과 attention mask 차이.
- **§3.1 “Image-Text Contrastive Learning”**: ITC와 momentum encoder/queue 사용.
- **§3.1 “Image-Text Matching” 및 “Language Modeling”**: ITM hard negatives와 LM objective.
- **§3.2 및 Figure 3**: captioner·filter로 구성된 CapFilt 데이터 bootstrapping.
- **Table 5**: synthetic caption과 filtering의 ablation.

### ⑥ 수식·알고리즘 직관

ITC는 전역 embedding 공간을 정렬하고, ITM은 image token과 text token의 cross-attention 뒤 pair가 진짜인지 판단하며, LM은 이전 token과 image를 조건으로 다음 token을 생성한다. 세 loss는 같은 pair에서 “거칠게 찾기 → 세밀히 확인하기 → 설명 생성하기”를 담당한다. CapFilt는 모델을 데이터 생성·검수 도구로 다시 사용한다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/08_blip.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/08_blip.ipynb)
- **TODO 08-1 ↔ Figure 3**: clean pair에서 noisy web caption과 synthetic caption 후보를 만든다.
- **TODO 08-2 ↔ §3.2 CapFilt**: image–text compatibility score로 낮은 품질 pair를 걸러낸다.
- **TODO 08-3 ↔ Figure 2**: 작은 shared multimodal model의 ITC·ITM·LM loss를 구현한다.
- **TODO 08-4 ↔ §3.1**: 세 objective를 함께 학습하고 retrieval·matching·generation 지표를 비교한다.
- **포트폴리오 P1 → P2 → P3**: `BlipPortfolioModel`의 CapFilt 입력·ITC/ITM/LM loss·update를 완성하고, fit loop 뒤 retrieval Recall@1·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** BLIP은 모델의 이해·생성 능력을 세 loss로 합치고, 그 모델이 다시 caption을 만들고 거르는 선순환을 설계한다.
- **한계:** captioner·filter의 편향과 오류가 self-reinforcing될 수 있고, 다중 objective의 weight 균형이 필요하며, 생성 사실성·환각을 보장하지 않는다.
- **자가점검:**
  1. ITC와 ITM이 서로 다른 granularity를 보는 이유는?
  2. synthetic caption을 추가만 하지 않고 filter도 해야 하는 이유는?
  3. causal LM mask와 bidirectional text-encoder mask의 목적 차이는?

---

## 09. Flamingo — A Visual Language Model for Few-Shot Learning (NeurIPS 2022)

원문: [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)

### ① 한 문장 요약

동결된 vision encoder와 language model 사이에 Perceiver Resampler와 gated cross-attention만 학습해, 임의로 섞인 image/video–text sequence를 prompt로 받아 multimodal in-context few-shot 학습한다.

### ② 배경과 이전 접근의 한계

기존 VLM은 task마다 fine-tuning하거나 고정된 한 장 image–한 문장 형식에 묶이는 경우가 많았다. 거대한 vision·language backbone을 처음부터 함께 학습하면 비용이 막대하고 이미 획득한 능력을 잃을 수 있다. Flamingo는 pretrained unimodal model을 얼리고 작은 bridge로 연결해 few-shot transfer를 목표로 한다.

### ③ 선수지식

- causal language modeling과 in-context learning
- cross-attention
- Perceiver의 latent queries
- residual connection과 `tanh` gate
- frozen backbone·parameter-efficient adaptation

### ④ 핵심 아이디어

- Perceiver Resampler가 크기가 다른 image/video feature grid를 고정 수 visual tokens로 압축한다.
- LM block 사이에 새 gated cross-attention layer를 삽입해 text가 앞선 visual tokens를 참조한다.
- gate를 0으로 초기화해 학습 시작 시 원래 LM과 같은 함수가 되게 한다.
- interleaved web sequences의 next-token likelihood를 학습해 prompt 속 예시를 따르는 능력을 얻는다.

### ⑤ 꼭 볼 원문 위치

- **§2.1 및 Figure 3**: Perceiver Resampler의 learned latent query와 visual token 압축.
- **§2.2 및 Figure 4**: frozen LM 사이에 삽입한 GATED XATTN-DENSE block.
- **§2, Eq. (1)**: interleaved image/video와 text를 조건으로 한 next-token likelihood.
- **§2.3**: vision encoder·LM을 freezing하고 bridge parameter만 학습하는 전략.
- **Figure 1**: multimodal few-shot prompt와 task transfer 예시.

### ⑥ 수식·알고리즘 직관

Resampler의 소수 learned latents가 수백 visual features를 cross-attend해 고정 길이 summary를 만든다. LM residual에 `tanh(α)·CrossAttn(text, visual)`을 더하며 `α=0`에서 시작하므로 처음에는 pretrained language behavior를 훼손하지 않는다. 학습 후 gate가 열리면 필요할 때만 visual 조건이 next-token distribution을 바꾼다.

### ⑦ 실습·정답과 TODO 연결

- [실습 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/exercises/09_flamingo.ipynb) · [정답 노트북](../../notebooks/field_reproductions/self_supervised_multimodal/solutions/09_flamingo.ipynb)
- **TODO 09-1 ↔ Figure 3**: learned latents가 visual features를 압축하는 `PerceiverResampler`를 구현한다.
- **TODO 09-2 ↔ Figure 4**: `tanh` zero-init gate를 가진 cross-attention residual block을 만든다.
- **TODO 09-3 ↔ §2.3**: frozen vision/text backbone은 고정하고 bridge parameter만 학습한다.
- **TODO 09-4 ↔ Eq. (1)**: 올바른 visual condition과 shuffled visual condition의 token prediction loss를 비교한다.
- **포트폴리오 P1 → P2 → P3**: `FlamingoPortfolioModel`의 visual/token 준비·resampler/gated bridge·loss/update를 완성하고, fit loop 뒤 language loss·conditioning gap·finite-history 계약을 검증한다.

### ⑧ 5분 정리

- **5분 요약:** Flamingo는 거대 unimodal backbone을 보존한 채 “visual token 압축기 + 안전하게 열리는 cross-attention”으로 multimodal few-shot LM을 만든다.
- **한계:** 폐쇄적 대규모 데이터·compute 때문에 완전 재현이 어렵고, hallucination·bias·prompt sensitivity가 있으며, frozen backbone의 결함과 context 길이 비용을 물려받는다.
- **자가점검:**
  1. Resampler가 visual token 수를 고정해야 하는 이유는?
  2. gate를 0으로 초기화하면 학습 초기에 어떤 함수가 보존되는가?
  3. shuffled image에서 loss가 같다면 bridge에 대해 무엇을 의심해야 하는가?

---

## 다음에 읽을 최신 논문 5편 (2023–2024, 공식 게재 확인)

아래 논문은 2026년 8월 기준으로 **CVF, PMLR, NeurIPS, TMLR/OpenReview의 공식 게재 페이지에서 venue·연도를 직접 확인**했다. arXiv만으로 채택 venue를 추정하지 않았다.

### A. I-JEPA — Self-Supervised Learning From Images With a Joint-Embedding Predictive Architecture

- **Venue / year:** CVPR 2023.
- **왜 읽나:** MAE처럼 block을 가리지만 pixel을 생성하지 않고 context embedding으로 target block embedding을 예측한다. CPC의 latent prediction과 MAE의 masking을 잇고, 수작업 view augmentation 의존을 낮춘다.
- **선수지식:** CPC, MAE, ViT, EMA target encoder, block masking, representation collapse.
- **공식 URL:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)

### B. SigLIP — Sigmoid Loss for Language Image Pre-Training

- **Venue / year:** ICCV 2023.
- **왜 읽나:** CLIP의 batch-global softmax 대신 각 image–text pair의 독립 sigmoid loss를 써 distributed all-gather 의존을 줄이고 batch size와 objective를 분리한다.
- **선수지식:** CLIP, binary logistic loss, positive/negative sampling ratio, distributed contrastive training, temperature/bias calibration.
- **공식 URL:** [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html)

### C. BLIP-2 — Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

- **Venue / year:** ICML 2023, PMLR 202.
- **왜 읽나:** BLIP·Flamingo의 다음 단계로, frozen image encoder와 frozen LLM 사이의 lightweight Q-Former만 두 단계로 학습해 modality gap을 적은 trainable parameter로 연결한다.
- **선수지식:** BLIP, Flamingo, cross-attention, frozen LLM, query tokens, causal language modeling.
- **공식 URL:** [PMLR proceedings](https://proceedings.mlr.press/v202/li23q.html)

### D. LLaVA — Visual Instruction Tuning

- **Venue / year:** NeurIPS 2023 Main Conference Track.
- **왜 읽나:** CLIP vision encoder와 LLM을 projection으로 연결한 뒤 machine-generated visual instruction data로 tuning하여, retrieval 중심 VLM에서 대화형 general-purpose assistant로 넘어가는 전환을 보여준다.
- **선수지식:** CLIP, autoregressive LLM, instruction tuning, synthetic data generation, multimodal projector, supervised fine-tuning.
- **공식 URL:** [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)

### E. DINOv2 — Learning Robust Visual Features without Supervision

- **Venue / year:** Transactions on Machine Learning Research, 2024년 1월 게재.
- **왜 읽나:** DINO의 student–teacher recipe를 curated unlabeled data pipeline, regularization, larger ViT로 확장해 여러 task와 image distribution에서 재사용 가능한 visual foundation feature를 목표로 한다.
- **선수지식:** DINO, iBOT식 masked token objective, ViT registers/patch tokens, data curation·deduplication, linear·dense probing.
- **공식 URL:** [TMLR/OpenReview 게재본](https://openreview.net/pdf?id=a68SUt6zFt)

### 최신 5편 선정 기준

1. **공식성:** 2023–2026 범위의 peer-reviewed CVPR·ICCV·ICML·NeurIPS·TMLR 공식 페이지에서 게재 정보가 확인될 것.
2. **학습 경로의 연속성:** 위 실습의 predictive SSL, masked modeling, vision–language loss, frozen-bridge, instruction tuning을 직접 확장할 것.
3. **기술 다양성:** latent prediction(I-JEPA), pairwise loss(SigLIP), efficient modality bridge(BLIP-2), instruction data(LLaVA), visual foundation scaling(DINOv2)을 하나씩 포함할 것.
4. **실습 가능성:** 작은 데이터에서도 block-latent loss, pairwise sigmoid, query bottleneck, instruction formatting, teacher–student objective로 핵심 가설을 축소 검증할 수 있을 것.

---

## 읽기 완료 체크리스트

- 각 objective의 positive, negative 또는 teacher target이 무엇인지 표시했다.
- gradient가 흐르는 branch와 stop-gradient·EMA로만 갱신되는 branch를 구분했다.
- exercise를 먼저 풀고 solution과 함수 단위로 비교했다.
- loss 감소만 보지 않고 collapse statistic, positive-vs-negative similarity, retrieval 또는 shuffled-condition 대조를 해석했다.
- 원 논문의 데이터 규모·backbone·평가와 로컬 축소 실습에서 생략한 요소를 설명할 수 있다.
