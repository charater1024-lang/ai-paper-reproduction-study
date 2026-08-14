# 유명 AI 논문 20편: 하드웨어 인식 미니 재현 트랙

이 트랙은 유명 논문의 핵심 수식과 구조를 **직접 구현하는 실습본**과, 같은 위치에 해설 코드를
제공하는 **정답본**으로 학습합니다. 모든 노트북은 외부 데이터 다운로드 없이 합성 데이터와 고정
seed를 사용하며, CUDA → Apple MPS → CPU를 자동 선택하고 어느 장치에서도 짧게 실행되도록
축소되어 있습니다.

> [!IMPORTANT]
> 이 자료는 논문의 전체 데이터셋, 모델 크기, 학습 시간, 최종 성능을 다시 만드는 완전 재현이
> 아닙니다. 논문의 핵심 메커니즘을 shape, 수치, loss, gradient, 작은 metric으로 검증하는
> **교육용 미니 재현**입니다. 여기서 나온 정확도나 loss를 원 논문의 결과와 직접 비교하면 안 됩니다.

상세한 학습 절차와 기록 방법은 [논문 재현 학습 가이드](../../docs/PAPER_REPRODUCTION_GUIDE.md)를
참고하세요. class·함수·수식·shape·구현 이유를 작성하는 기준은
[포트폴리오용 논문 재현 표준](../../docs/PORTFOLIO_NOTEBOOK_STANDARD.md)에 정리했습니다.

## 시작하기

프로젝트 루트에서 먼저 환경을 준비한 뒤 다음 파일을 실행합니다.

```bat
Install_or_Repair.cmd
Start_Paper_Reproductions.cmd
```

번호를 바로 지정할 수도 있습니다.

```bat
Start_Paper_Reproductions.cmd 09
```

JupyterLab 서버 하나가 실행되고 왼쪽에는 실습본, 오른쪽에는 같은 논문의 정답본이 열립니다.
정답본을 먼저 복사하지 말고 실습본의 TODO와 예상 shape를 작성한 다음 필요한 부분만 비교하세요.

## 전체 커리큘럼

시간은 정답 셀을 읽고 한 번 구현하는 기준의 대략적인 값이며, 추가 실험 시간은 포함하지 않습니다.

| 번호 | 논문 / 연도 | 핵심 개념 | 실습 | 정답 | 예상 |
|---:|---|---|---|---|---:|
| 00 | LeNet-5 (1998) | local receptive field, shared convolution, subsampling | [열기](exercises/00_lenet5.ipynb) | [열기](solutions/00_lenet5.ipynb) | 70분 |
| 01 | AlexNet (2012) | ReLU, LRN, overlapping max pooling | [열기](exercises/01_alexnet.ipynb) | [열기](solutions/01_alexnet.ipynb) | 85분 |
| 02 | U-Net (2015) | encoder-decoder, skip concatenation, border weighting | [열기](exercises/02_unet.ipynb) | [열기](solutions/02_unet.ipynb) | 95분 |
| 03 | ResNet (2016) | residual mapping, identity/projection shortcut | [열기](exercises/03_resnet.ipynb) | [열기](solutions/03_resnet.ipynb) | 85분 |
| 04 | Dropout (2014) | Bernoulli masking, train/test scaling | [열기](exercises/04_dropout.ipynb) | [열기](solutions/04_dropout.ipynb) | 75분 |
| 05 | Batch Normalization (2015) | batch statistics, γ/β, running statistics | [열기](exercises/05_batch_normalization.ipynb) | [열기](solutions/05_batch_normalization.ipynb) | 90분 |
| 06 | word2vec (2013) | skip-gram, negative sampling | [열기](exercises/06_word2vec_negative_sampling.ipynb) | [열기](solutions/06_word2vec_negative_sampling.ipynb) | 65분 |
| 07 | seq2seq (2014) | LSTM encoder-decoder, sequence state | [열기](exercises/07_sequence_to_sequence_lstm.ipynb) | [열기](solutions/07_sequence_to_sequence_lstm.ipynb) | 75분 |
| 08 | Bahdanau Attention (2015) | additive alignment, context vector | [열기](exercises/08_bahdanau_additive_attention.ipynb) | [열기](solutions/08_bahdanau_additive_attention.ipynb) | 65분 |
| 09 | Attention Is All You Need (2017) | scaled dot-product, multi-head attention, mask | [열기](exercises/09_attention_is_all_you_need.ipynb) | [열기](solutions/09_attention_is_all_you_need.ipynb) | 80분 |
| 10 | GPT-1 (2018) | causal pre-training, task adaptation | [열기](exercises/10_gpt1_generative_pretraining.ipynb) | [열기](solutions/10_gpt1_generative_pretraining.ipynb) | 85분 |
| 11 | BERT (2018) | bidirectional encoder, MLM, NSP | [열기](exercises/11_bert_pretraining.ipynb) | [열기](solutions/11_bert_pretraining.ipynb) | 90분 |
| 12 | VAE (2014) | ELBO, KL divergence, reparameterization | [열기](exercises/12_vae.ipynb) | [열기](solutions/12_vae.ipynb) | 60분 |
| 13 | GAN (2014) | adversarial objective, alternating optimization | [열기](exercises/13_gan.ipynb) | [열기](solutions/13_gan.ipynb) | 55분 |
| 14 | DQN (2013) | Bellman target, replay buffer, target network | [열기](exercises/14_dqn.ipynb) | [열기](solutions/14_dqn.ipynb) | 65분 |
| 15 | GCN (2017) | normalized adjacency, graph message passing | [열기](exercises/15_gcn.ipynb) | [열기](solutions/15_gcn.ipynb) | 55분 |
| 16 | SimCLR (2020) | augmented views, NT-Xent contrastive loss | [열기](exercises/16_simclr_contrastive_learning.ipynb) | [열기](solutions/16_simclr_contrastive_learning.ipynb) | 75분 |
| 17 | Vision Transformer (2021) | image patches, class token, Transformer encoder | [열기](exercises/17_vision_transformer.ipynb) | [열기](solutions/17_vision_transformer.ipynb) | 75분 |
| 18 | DDPM (2020) | forward diffusion, noise prediction, reverse sampling | [열기](exercises/18_ddpm_diffusion.ipynb) | [열기](solutions/18_ddpm_diffusion.ipynb) | 90분 |
| 19 | CLIP (2021) | dual encoder, symmetric contrastive loss, zero-shot logits | [열기](exercises/19_clip_multimodal_contrastive.ipynb) | [열기](solutions/19_clip_multimodal_contrastive.ipynb) | 80분 |

전체 기본 실습은 약 25시간 분량입니다. 한 번에 끝내기보다 논문 하나마다 핵심 변수를 하나 바꿔
추가 실험까지 남기는 것을 권장합니다.

## 추천 학습 경로

처음이라면 모든 논문을 발표 순서대로 진행할 필요는 없습니다.

- **공통 신경망 기초:** `00 → 01 → 04 → 05 → 03`  
  convolution, activation, regularization, normalization, shortcut을 차례로 익힙니다.
- **NLP에서 LLM까지:** `06 → 07 → 08 → 09 → 10 → 11`  
  분산 표현에서 encoder-decoder와 attention을 거쳐 사전학습 언어모델로 이어집니다.
- **생성 모델:** `12 → 13 → 18`  
  likelihood lower bound, adversarial learning, diffusion이라는 서로 다른 생성 원리를 비교합니다.
- **현대 비전·멀티모달:** `01 → 03 → 16 → 17 → 19`  
  CNN 표현에서 대조학습, patch token, 이미지-텍스트 공동 공간으로 확장합니다.
- **구조화된 의사결정·데이터:** `14 → 15`  
  temporal-difference target과 graph message passing을 독립 주제로 학습합니다.
- **분할이 우선이라면:** `00 → 01 → 02 → 03` 순서가 가장 짧습니다.

## 완료 기준

노트북 하나를 완료했다는 것은 정답과 줄 단위로 같다는 뜻이 아닙니다. 다음을 모두 만족하면 됩니다.

1. 실습본의 TODO를 직접 구현했다.
2. 포함된 모든 `assert`가 통과한다.
3. mapping 표의 원문 수식·절과 구현 위치를 자신의 말로 연결할 수 있다.
4. 마지막 metric 또는 시각화를 해석했다.
5. seed는 유지한 채 변수 하나만 바꾼 추가 실험을 기록했다.

검증과 전체 정답 실행 명령은 [상세 가이드의 검증 절](../../docs/PAPER_REPRODUCTION_GUIDE.md#검증-명령)을
참고하세요.
