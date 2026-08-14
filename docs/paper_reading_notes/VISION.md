# 컴퓨터 비전 핵심 논문 10편 독해 노트

> **AI 보조 작성 안내:** 이 한국어 요약은 공식 번역이 아닙니다. 내용과 인용의 최종 기준은
> 연결된 원문입니다. [작성 원칙과 한계](AI_ASSISTED_NOTICE.md) · [70편 목차](README.md)

이 문서는 논문 전문을 처음부터 끝까지 읽기 전에 필요한 맥락을 잡고, 바로 실습 노트북으로 핵심 계산을 확인하도록 만든 안내서다. 논문의 보고 수치를 그대로 재현하는 과정이 아니라 작은 합성 데이터에서 아이디어의 작동 원리를 검증한다. 원문 위치는 [비전 사양 파일](../../tools/field_curriculum/vision_specs.py)의 매핑과 일치한다.

## 00. LeNet-5 — Gradient-Based Learning Applied to Document Recognition (1998)

**① 한 문장 요약.** 사람이 설계한 문자 특징 대신, 작은 필터를 공유하는 합성곱과 해상도를 줄이는 subsampling을 연결해 픽셀부터 분류까지 한 번에 학습한 초기 CNN의 표준형이다.

**② 왜 나왔나 / 이전 한계.** 당시 문서 인식은 글자 분할, 특징 추출, 분류기를 별도 단계로 설계하는 경우가 많았다. 완전연결망은 이미지 위치가 조금만 바뀌어도 별개의 패턴으로 취급하고 파라미터가 급증했다. LeNet은 국소 수용영역과 가중치 공유로 이 문제를 줄이고 전체 파이프라인을 역전파로 함께 최적화했다.

**③ 읽기 전 기초지식.** 2차원 convolution, kernel과 feature map, valid padding, receptive field, 평균 pooling/subsampling, cross-entropy와 역전파.

**④ 핵심 아이디어.**

- 한 필터를 모든 공간 위치에 적용해 같은 모양을 위치와 무관하게 찾는다.
- C 계층은 특징을 만들고 S 계층은 공간 해상도와 작은 위치 변화의 민감도를 낮춘다.
- 여러 C-S 단계를 쌓아 선·모서리에서 글자 전체로 수용영역을 넓힌다.
- 특징 추출기와 분류기를 하나의 미분 가능한 모델로 학습한다.

**⑤ 꼭 볼 원문 위치.** [원문 PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)의 §II.A에서 local receptive field와 shared weights를, §II.B와 Fig. 2에서 C1-S2-C3-S4-C5-F6 흐름을 본다. §II.B의 average subsampling이 오늘날의 max pooling과 다름을 확인하고, Table I은 전체 시스템 비교 맥락만 읽는다.

**⑥ 수식·알고리즘 직관.** 합성곱은 작은 가중치 행렬을 이미지 위로 이동시키며 내적한다. 같은 가중치를 재사용하므로 “이 특징이 어디에 있는가”보다 “존재하는가”를 효율적으로 배운다. 2×2 평균 subsampling은 인접 응답을 요약해 해상도를 절반으로 줄인다. 깊어질수록 한 출력이 참조하는 원본 영역, 즉 수용영역이 커진다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/00_lenet5.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/00_lenet5.ipynb). TODO 1은 valid 5×5 convolution과 평균 subsampling 뒤 shape를 추적한다. TODO 2는 Fig. 2 순서의 축소 모델을 구현한다. TODO 3은 로컬 `vision_shapes.npz`로 학습해 loss 감소와 test accuracy를 그린다. 원 논문의 부분 연결 C3, 학습 가능한 subsampling 계수, RBF 출력은 교육용 모델에서 생략했다.

**⑧ 5분 요약 / 한계 / 자가점검.** 5분 안에는 “가중치 공유→평균 축소→계층적 특징”만 설명할 수 있으면 된다. 한계는 얕은 구조, tanh 포화, 현대 데이터 규모에 부족한 표현력이며 작은 실습 정확도는 원 논문 MNIST 결과와 비교할 수 없다.

1. 완전연결층보다 convolution이 이미지에서 파라미터를 적게 쓰는 이유는?
2. valid 5×5 convolution 뒤 가로·세로 크기는 왜 4씩 줄어드는가?
3. 평균 subsampling과 max pooling은 어떤 정보를 각각 보존하는가?

## 01. AlexNet — ImageNet Classification with Deep Convolutional Neural Networks (2012)

**① 한 문장 요약.** 깊은 CNN, ReLU, GPU 학습과 강한 정규화를 결합해 대규모 자연 이미지 분류에서 학습 기반 특징의 잠재력을 분명히 보인 모델이다.

**② 왜 나왔나 / 이전 한계.** LeNet류 CNN은 작은 문자 데이터에서는 성공했지만 ImageNet처럼 클래스와 외형 변화가 큰 데이터에서는 계산량과 최적화가 장벽이었다. tanh/sigmoid는 깊은 모델에서 학습이 느렸고, 큰 모델은 과적합하기 쉬웠다. AlexNet은 GPU 계산, ReLU, augmentation, dropout을 함께 사용해 규모를 키웠다.

**③ 읽기 전 기초지식.** ReLU와 gradient, channel, max pooling, local response normalization(LRN), dropout, data augmentation.

**④ 핵심 아이디어.**

- ReLU로 포화 비선형성보다 빠른 최적화를 유도한다.
- 인접 channel의 큰 응답끼리 경쟁시키는 LRN을 사용한다.
- 3×3 window를 stride 2로 이동하는 overlapping max pooling을 쓴다.
- 여러 convolution 뒤 큰 fully connected 계층을 두고 augmentation/dropout으로 과적합을 줄인다.

**⑤ 꼭 볼 원문 위치.** [NeurIPS 원문 PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)의 §3.1에서 `f(x)=max(0,x)`, §3.3 Eq. (1)에서 LRN, §3.4에서 `z=3, s=2` pooling, §3.5와 Fig. 2에서 5개 convolution의 전체 구조를 본다.

**⑥ 수식·알고리즘 직관.** ReLU는 음수를 0으로 만들고 양수에서는 기울기 1을 유지한다. LRN은 한 channel 응답을 주변 channel 제곱합의 거듭제곱으로 나눠 큰 응답을 상대적으로 억제한다. overlapping pooling은 window가 겹치므로 non-overlapping 설정보다 더 촘촘한 출력과 완만한 정보 손실을 만든다. LRN은 역사적으로 중요하지만 현대 모델에서는 batch/layer normalization이 더 일반적이다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/01_alexnet.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/01_alexnet.ipynb). TODO 1은 Eq. (1)의 channel-local LRN을 직접 구현한다. TODO 2는 stride 2/3 pooling shape를 비교하고 5-conv 축소 모델을 만든다. TODO 3은 학습 loss와 accuracy를 시각화한다. GPU 분할, 대규모 augmentation, 원래 fully connected 크기는 재현하지 않는다.

**⑧ 5분 요약 / 한계 / 자가점검.** AlexNet의 핵심은 단일 새 연산이 아니라 데이터·계산·최적화·정규화의 묶음이다. 한계는 큰 메모리/파라미터, LRN의 제한된 효용, 현대 기준으로 비효율적인 head다.

1. ReLU가 tanh보다 깊은 모델 최적화에 유리할 수 있는 이유는?
2. `kernel=3, stride=2` pooling을 overlapping이라 부르는 이유는?
3. LRN과 오늘날 normalization 계층의 목적은 완전히 같은가?

## 02. VGG — Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)

**① 한 문장 요약.** 큰 필터 대신 동일한 3×3 convolution을 규칙적으로 깊게 쌓아 구조를 단순화하면서 표현력을 높인 CNN 설계 연구다.

**② 왜 나왔나 / 이전 한계.** 초기 CNN은 필터 크기와 stride가 층마다 달라 설계 비교가 어려웠다. “깊이가 실제로 어떤 효과를 내는가”도 다른 설계 요소와 뒤섞였다. VGG는 거의 모든 공간 convolution을 3×3/stride 1로 통일하고 깊이만 체계적으로 늘려 이 질문을 분석했다.

**③ 읽기 전 기초지식.** receptive field, parameter count, stacked convolution, max pooling, network depth, 비선형성의 합성.

**④ 핵심 아이디어.**

- 3×3 convolution을 반복해 단순하고 재사용하기 쉬운 block을 만든다.
- 세 개의 3×3은 한 개의 7×7과 같은 7×7 수용영역을 갖는다.
- 작은 필터 stack은 큰 필터 하나보다 중간 ReLU를 더 넣고, 동일 channel 가정에서 파라미터도 줄인다.
- configuration A–E를 비교해 깊이의 효과를 분리한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1409.1556)의 §2.1에서 3×3/stride 1 규칙, §2.2와 Table 1에서 configuration별 깊이, §2.3에서 작은 필터 stack의 논의를 읽는다. §3은 classification 학습 절차가 구조 실험과 어떻게 연결되는지 본다.

**⑥ 수식·알고리즘 직관.** stride 1일 때 필터 하나를 통과할 때마다 수용영역은 `k-1`만큼 늘어난다. 따라서 `1+3×(3-1)=7`이다. 64 channel을 유지한다면 3×3 세 층의 weight 수는 `3·3·64²·3`, 7×7 한 층은 `7·7·64²`다. 단, channel 변화와 activation memory까지 포함하면 실제 비용 비교는 더 복잡하다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/02_vgg.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/02_vgg.ipynb). TODO 1은 수용영역과 파라미터 수를 계산한다. TODO 2는 2/2/3개의 3×3 convolution을 가진 block을 구현한다. TODO 3은 7-conv 축소 모델의 loss와 accuracy를 확인한다.

**⑧ 5분 요약 / 한계 / 자가점검.** VGG는 “작고 균일한 연산을 깊게 쌓는다”는 강한 baseline을 남겼다. 한계는 큰 activation/parameter 비용과 shortcut 부재로, 더 깊어질수록 최적화가 어려워진다는 점이다.

1. 3×3 세 층이 7×7 한 층보다 비선형성을 더 많이 갖는 이유는?
2. 수용영역이 같다고 두 구조의 계산과 표현이 완전히 같지는 않은 이유는?
3. VGG가 오늘날에도 backbone 비교 기준으로 유용한 이유는?

## 03. GoogLeNet / Inception — Going Deeper with Convolutions (2014)

**① 한 문장 요약.** 여러 크기의 convolution과 pooling을 병렬로 실행하고 1×1 projection으로 channel을 줄여, 계산 예산 안에서 폭과 깊이를 함께 늘린 구조다.

**② 왜 나왔나 / 이전 한계.** 깊고 넓은 모델은 표현력이 커지지만 파라미터와 계산량도 급증하고 과적합 위험이 커졌다. 한 위치에서 어떤 필터 크기가 적절한지 미리 하나로 정하기도 어렵다. Inception은 다양한 공간 척도를 병렬로 처리하되 비싼 3×3/5×5 앞에 1×1 reduction을 둔다.

**③ 읽기 전 기초지식.** 1×1 convolution, channel projection, 병렬 branch, tensor concatenation, multiply-accumulate 비용, global average pooling.

**④ 핵심 아이디어.**

- 1×1, 3×3, 5×5, pooling branch로 같은 feature map을 서로 다른 척도에서 본다.
- branch 출력을 channel 방향으로 이어 붙인다.
- 1×1 convolution이 비싼 큰 필터 앞의 입력 channel을 줄인다.
- global average pooling과 auxiliary classifier를 포함해 깊은 네트워크 학습을 돕는다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1409.4842)의 §4와 Fig. 2(a)에서 naïve module, Fig. 2(b)에서 dimension-reduction module을 비교한다. §5와 Table 1은 GoogLeNet 전체 배치, §6.3은 classification 설정을 확인한다.

**⑥ 수식·알고리즘 직관.** 32 입력 channel에서 16 출력 channel의 5×5를 바로 쓰면 `32·16·25`개 weight가 필요하다. 먼저 1×1로 4 channel로 줄이면 `32·4 + 4·16·25`가 된다. projection은 정보 병목이므로 너무 작으면 손실되지만, 적절하면 계산을 크게 줄이고 비선형성도 하나 추가한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/03_googlenet_inception.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/03_googlenet_inception.ipynb). TODO 1은 직접 5×5와 1×1-reduced 5×5의 weight 수를 비교한다. TODO 2는 네 병렬 branch와 channel concat을 구현한다. TODO 3은 두 module을 쌓은 모델을 학습한다. auxiliary head는 생략했다.

**⑧ 5분 요약 / 한계 / 자가점검.** 5분 요약은 “multi-scale 병렬 처리 + 1×1 계산 병목”이다. 한계는 module 설계가 복잡하고, 실제 효율이 하드웨어/branch 병렬화에 의존한다는 점이다.

1. 1×1 convolution이 공간 크기를 바꾸지 않고 계산량을 줄이는 방법은?
2. 병렬 branch 출력을 더하지 않고 concatenate하면 다음 층 channel은 어떻게 변하는가?
3. reduction channel을 지나치게 줄이면 어떤 문제가 생기는가?

## 04. U-Net — Convolutional Networks for Biomedical Image Segmentation (2015)

**① 한 문장 요약.** 문맥을 얻는 contracting path와 위치 정밀도를 되찾는 expanding path를 대칭으로 두고, 같은 해상도의 encoder 특징을 decoder에 직접 연결한 segmentation 모델이다.

**② 왜 나왔나 / 이전 한계.** 생의학 영상은 pixel annotation이 비싸 데이터가 적고, sliding-window 분류는 중복 계산이 크며 문맥과 위치 정밀도를 동시에 잡기 어렵다. 단순 encoder-decoder는 downsampling 중 경계 정보를 잃는다. U-Net은 강한 augmentation과 skip concatenation으로 적은 데이터에서도 dense prediction을 가능하게 했다.

**③ 읽기 전 기초지식.** semantic segmentation, encoder-decoder, transposed convolution/upsampling, skip connection, pixel-wise loss, Dice overlap.

**④ 핵심 아이디어.**

- contracting path는 pooling으로 수용영역과 의미 정보를 키운다.
- expanding path는 up-convolution으로 공간 해상도를 복구한다.
- encoder의 고해상도 특징을 decoder에 concatenate해 경계를 보완한다.
- 원 논문은 valid convolution 때문에 skip feature를 crop하고, elastic deformation으로 데이터를 늘린다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1505.04597)의 §2와 Fig. 1에서 U자형 구조와 crop-and-copy를 본다. §2.1 Eq. (1)은 pixel-wise softmax와 cross-entropy, §3과 Fig. 3은 segmentation 결과/겹침의 맥락을 제공한다.

**⑥ 수식·알고리즘 직관.** 분류는 이미지당 label 하나를 예측하지만 segmentation은 각 pixel마다 class logit을 낸다. skip concatenation은 decoder가 “무엇인가”를 나타내는 저해상도 문맥과 “정확히 어디인가”를 나타내는 고해상도 특징을 함께 보게 한다. Dice는 `2|P∩T|/(|P|+|T|)`로 두 mask의 겹침을 잰다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/04_unet.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/04_unet.ipynb). TODO 1은 1-level encoder, transposed convolution, skip concat을 구현한다. TODO 2는 저장된 `masks`에 pixel-wise BCE를 적용한다. TODO 3은 test Dice와 target/prediction을 그린다. same padding을 써서 원 논문의 crop은 생략한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “압축 경로의 의미 + skip 경로의 위치”다. 한계는 큰 고해상도 feature를 보관하는 메모리 비용, 도메인 변화, 경계/클래스 불균형에 대한 민감도다.

1. encoder와 decoder만 있고 skip connection이 없다면 어떤 정보가 가장 쉽게 손실되는가?
2. concatenation과 residual addition은 channel 수와 정보 결합 방식이 어떻게 다른가?
3. pixel accuracy가 높아도 작은 물체 Dice가 낮을 수 있는 이유는?

## 05. ResNet — Deep Residual Learning for Image Recognition (2015)

**① 한 문장 요약.** 층이 원하는 함수를 직접 배우게 하는 대신 입력에 더할 잔차 `F(x)`를 배우게 해 매우 깊은 네트워크의 최적화를 쉽게 만든 구조다.

**② 왜 나왔나 / 이전 한계.** 깊이를 늘리면 이론적으로 표현력이 줄지 않아야 하지만 실제 plain network에서는 training error까지 나빠지는 degradation 문제가 나타났다. 이는 단순 과적합만으로 설명되지 않는다. identity mapping조차 여러 nonlinear layer가 새로 근사해야 하기 때문이다.

**③ 읽기 전 기초지식.** identity mapping, residual function, shortcut connection, gradient flow, batch normalization, projection shortcut.

**④ 핵심 아이디어.**

- block의 목표를 `H(x)`에서 `F(x)=H(x)-x`로 바꾼다.
- 출력은 `H(x)=F(x)+x`이며 identity shortcut에는 보통 추가 파라미터가 없다.
- channel/해상도가 달라지면 projection shortcut으로 차원을 맞춘다.
- basic block 또는 bottleneck block을 반복해 깊이를 크게 늘린다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1512.03385)의 §3.1 Eq. (1) `H(x)=F(x)+x`, Eq. (2) 실제 두 weight layer 표현을 읽는다. §3.2와 Fig. 2에서 plain/ResNet 구조를 비교하고 §4.2 Fig. 4와 Fig. 6에서 optimization behavior를 본다.

**⑥ 수식·알고리즘 직관.** 최적해가 identity에 가깝다면 residual branch의 weight를 0 근처로 만드는 것으로 충분하다. 역전파 때도 덧셈의 identity 경로를 통해 gradient가 직접 이전 block으로 전달될 수 있다. 이것이 gradient가 절대 소실되지 않음을 보장하지는 않지만 깊은 네트워크의 최적화 경로를 크게 개선한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/05_resnet.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/05_resnet.ipynb). TODO 1은 `ReLU(F(x)+x)` block을 만들고 residual weights가 0일 때 양수 입력이 그대로 나오는지 확인한다. TODO 2는 block 네 개와 global-average head를 조립한다. TODO 3은 loss/accuracy를 그린다. BatchNorm과 projection shortcut은 작은 same-shape 실습에서 생략했다.

**⑧ 5분 요약 / 한계 / 자가점검.** 5분 요약은 “identity를 우회도로 두고 변화량만 학습”이다. 한계는 shortcut만으로 모든 학습 문제가 해결되지 않으며, normalization·초기화·학습률 설계가 여전히 중요하다는 점이다.

1. residual branch가 정확히 0이면 block 출력은 무엇인가?
2. 입력과 출력 channel이 다르면 단순 덧셈을 할 수 없는 이유는?
3. degradation problem과 test-set overfitting은 어떻게 구별하는가?

## 06. Faster R-CNN — Region Proposal Networks (2015)

**① 한 문장 요약.** 별도의 느린 region proposal 알고리즘을 shared convolution feature 위의 작은 RPN으로 바꾸어 proposal과 detection을 함께 학습한 two-stage detector다.

**② 왜 나왔나 / 이전 한계.** Fast R-CNN은 proposal마다 CNN을 다시 돌리는 비용은 줄였지만 selective search가 여전히 CPU 병목이었다. proposal 생성과 분류가 서로 다른 목적/구현으로 분리되어 end-to-end 최적화도 어려웠다. Faster R-CNN은 같은 feature map에서 objectness와 box offset을 예측한다.

**③ 읽기 전 기초지식.** bounding box `xyxy`, intersection over union(IoU), anchor, objectness, box regression, Smooth-L1, two-stage detection.

**④ 핵심 아이디어.**

- feature map의 각 위치에서 여러 scale/aspect-ratio anchor를 기준으로 proposal을 예측한다.
- anchor마다 물체 여부와 네 개의 box delta를 출력한다.
- RPN과 Fast R-CNN detector가 backbone feature를 공유한다.
- positive/negative anchor sampling과 multi-task loss로 분류·위치 회귀를 함께 학습한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1506.01497)의 §3.1과 Fig. 3에서 sliding RPN, §3.1.1에서 anchors를 읽는다. §3.1.2 Eq. (1)은 classification+regression loss, Eq. (2)는 중심/크기 기반 box parameterization이다.

**⑥ 수식·알고리즘 직관.** anchor는 정답 그 자체가 아니라 “수정하기 쉬운 시작 box”다. IoU가 큰 anchor를 positive로 정하고, 중심 이동은 anchor 폭/높이로 정규화하며 크기 변화는 로그 비율로 표현한다. Eq. (1)은 모든 선택 anchor에 objectness loss를 주되 box loss는 positive에만 적용한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/06_faster_rcnn.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/06_faster_rcnn.ipynb). TODO 1은 4×4 anchor grid, IoU와 delta를 구현한다. TODO 2는 RPN head와 BCE+Smooth-L1 loss를 만든다. TODO 3은 loss와 첫 이미지의 anchor IoU 분포를 그린다. 실습은 anchor 1개/위치이며 second-stage ROI head는 생략한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “shared feature 위에서 proposal도 신경망이 만든다”이다. 한계는 anchor hyperparameter, 두 단계 추론 비용, NMS와 label assignment의 복잡성이다.

1. negative anchor에 box regression loss를 주지 않는 이유는?
2. 크기 비율에 로그를 쓰면 어떤 대칭성이 생기는가?
3. RPN과 detector가 feature를 공유할 때 얻는 이점은?

## 07. YOLOv1 — You Only Look Once (2015)

**① 한 문장 요약.** 물체 검출을 전체 이미지에서 grid별 box·confidence·class를 한 번에 회귀하는 단일 네트워크 문제로 바꾼 one-stage detector다.

**② 왜 나왔나 / 이전 한계.** R-CNN 계열은 proposal, feature extraction, classification, box refinement, NMS라는 복잡한 pipeline을 거쳐 느렸다. 각 단계가 따로 최적화되어 전체 검출 목표와 어긋날 수 있었다. YOLO는 전역 문맥을 보며 하나의 forward pass에서 모든 예측을 낸다.

**③ 읽기 전 기초지식.** grid cell, normalized box center/size, conditional class probability, IoU, one-stage detection, sum-squared error.

**④ 핵심 아이디어.**

- 물체 중심이 들어간 cell이 해당 물체를 책임진다.
- 각 cell이 `B`개 box와 confidence, `C`개 conditional class probability를 예측한다.
- class 확률과 box confidence를 곱해 class-specific confidence를 만든다.
- coordinate, object confidence, no-object confidence, class loss의 가중치를 다르게 둔다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1506.02640)의 §2와 Fig. 2에서 `S×S×(B·5+C)` 출력을, Eq. (1)에서 class-specific confidence를 본다. §2.2 Eq. (3)은 `λcoord=5`, `λnoobj=.5`를 포함한 손실이고 §2.3은 single-evaluation inference를 설명한다.

**⑥ 수식·알고리즘 직관.** confidence는 `P(object)·IoU`로 물체 존재와 위치 품질을 함께 표현한다. 폭·높이에는 제곱근을 적용해 큰 box의 절대 오차가 손실을 지배하는 현상을 줄인다. 빈 cell이 훨씬 많으므로 no-object loss를 낮추지 않으면 모든 confidence를 0으로 만드는 방향이 우세해질 수 있다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/07_yolov1.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/07_yolov1.ipynb). TODO 1은 `S=4,B=1,C=4` target과 Eq. (1)을 만든다. TODO 2는 Eq. (3)을 네 항으로 구현하고 perfect prediction loss가 0인지 확인한다. TODO 3은 prediction tensor 자체를 최적화해 loss를 시각화한다. CNN detector와 NMS는 손실 이해에 집중하기 위해 생략했다.

**⑧ 5분 요약 / 한계 / 자가점검.** 요약은 “proposal 없이 전체 이미지를 한 번 보고 grid tensor를 회귀”한다는 것이다. v1의 한계는 한 cell당 제한된 물체, 작은 밀집 물체, localization error와 MSE 목적의 불일치다.

1. 물체가 cell 경계에 걸쳐 있어도 어느 cell이 책임지는가?
2. `λnoobj`를 작게 두는 이유는?
3. class probability와 confidence를 곱한 값이 함께 나타내는 두 정보는?

## 08. Vision Transformer — An Image is Worth 16×16 Words (2020)

**① 한 문장 요약.** 이미지를 고정 크기 patch sequence로 바꾸고 class token을 포함한 표준 Transformer encoder로 분류해, 대규모 사전학습에서 convolution 없이도 강한 비전 모델을 만들 수 있음을 보였다.

**② 왜 나왔나 / 이전 한계.** Transformer는 NLP에서 확장성이 검증됐지만 비전에서는 CNN의 locality와 translation equivariance가 필수라는 인식이 강했다. 기존 attention 비전 모델도 CNN과 섞거나 특수 구조를 사용했다. ViT는 최소한의 변경으로 순수 Transformer를 적용하고 데이터 규모가 inductive bias 부족을 보완하는지 시험했다.

**③ 읽기 전 기초지식.** image patch, token embedding, class token, positional embedding, multi-head self-attention, pre-norm residual block.

**④ 핵심 아이디어.**

- `P×P` patch를 펼쳐 vector로 만들고 linear projection한다.
- sequence 앞에 학습 가능한 class token을 붙인다.
- 위치 embedding으로 patch 순서를 제공한다.
- Transformer encoder의 class-token 출력으로 분류한다.
- 큰 데이터에서 사전학습하고 작은 downstream dataset에 fine-tune한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/2010.11929)의 §3.1과 Fig. 1에서 전체 흐름을 본다. Eq. (1)은 patch/class/position embedding, Eq. (2–4)는 pre-norm MSA/MLP residual과 head다. §3.2는 fine-tuning 및 해상도 변화 시 position embedding을 설명한다.

**⑥ 수식·알고리즘 직관.** `H×W` 이미지는 `N=HW/P²`개의 token이 된다. patch가 작을수록 공간 정보는 세밀하지만 attention 비용 `O(N²)`이 커진다. class token은 모든 patch와 attention하며 분류용 요약 표현을 학습한다. CNN보다 locality bias가 약해 작은 데이터만으로는 불리할 수 있다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/08_vision_transformer.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/08_vision_transformer.ipynb). TODO 1은 `unfold`로 16개의 4×4 patch와 17-token sequence를 만든다. TODO 2는 projection, CLS, position, 4-head encoder를 구현한다. TODO 3은 로컬 도형 분류 loss와 accuracy를 그린다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “image→patch tokens→표준 encoder”다. 한계는 token 수의 제곱 attention 비용, 약한 locality bias, 대규모 사전학습 의존성이다.

1. patch 크기를 절반으로 줄이면 token 수와 attention matrix 크기는 어떻게 변하는가?
2. position embedding이 없다면 어떤 정보가 사라지는가?
3. class token과 global average pooling은 어떤 방식으로 다르게 요약하는가?

## 09. DETR — End-to-End Object Detection with Transformers (2020)

**① 한 문장 요약.** 검출을 고정 개수 object query가 만드는 순서 없는 prediction set으로 보고, Hungarian bipartite matching으로 정답과 일대일 대응시켜 anchor와 NMS를 제거했다.

**② 왜 나왔나 / 이전 한계.** 전통 검출기는 anchor 설계, positive assignment, proposal, NMS 등 사람이 정한 구성요소가 많았다. 중복 box를 후처리로 제거해야 했고 pipeline이 복잡했다. DETR은 “정답 물체 집합”과 “예측 집합”의 직접 비교를 학습 목표로 삼는다.

**③ 읽기 전 기초지식.** Transformer encoder-decoder, object query, permutation invariance, bipartite/Hungarian matching, no-object class, L1와 generalized IoU.

**④ 핵심 아이디어.**

- CNN feature를 Transformer memory sequence로 바꾼다.
- 학습 가능한 object query 각각이 하나의 물체 또는 no-object를 예측한다.
- 모든 정답과 예측 사이 비용을 계산해 최소 비용 일대일 matching을 찾는다.
- matched pair에 class와 box loss를 적용해 중복 없는 set prediction을 유도한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/2005.12872)의 §3.1 Eq. (1)에서 optimal bipartite matching, Eq. (2)에서 Hungarian loss를 본다. §3.2와 Fig. 2는 CNN encoder, Transformer, object query와 prediction FFN을 보여 준다.

**⑥ 수식·알고리즘 직관.** 예측에는 원래 순서가 없으므로 query 0이 반드시 첫 물체일 필요가 없다. 가능한 대응 중 class/box 비용 합이 최소인 permutation을 선택한 뒤 그 대응으로 loss를 계산한다. 남는 query는 no-object를 맞힌다. 이렇게 query가 서로 다른 물체를 맡도록 유도하므로 NMS 의존성을 없앨 수 있다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/vision/exercises/09_detr.ipynb) / [정답본](../../notebooks/field_reproductions/vision/solutions/09_detr.ipynb). TODO 1은 작은 cost matrix의 모든 permutation을 조사해 matching과 set loss를 계산한다. TODO 2는 CNN memory와 query 3개를 가진 decoder를 구현한다. TODO 3은 query 0을 물체, 나머지를 no-object로 단순화해 학습하고 box를 그린다. 다중 물체 Hungarian 학습과 GIoU는 축소했다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “검출을 순서 없는 집합으로 직접 예측”하는 것이다. 원 DETR의 한계는 긴 학습 수렴, 작은 물체 성능, 고해상도 attention 비용이며 이후 deformable 계열이 이를 보완했다.

1. 정답과 예측을 index 순서대로 비교하면 왜 잘못될 수 있는가?
2. 사용되지 않은 query에 no-object class가 필요한 이유는?
3. bipartite matching이 중복 예측을 줄이는 방식은?

## 2023–2026 후속 학습 후보 5편

선정 기준은 다음 네 가지다. (1) 2023년 이후 정식 top-tier conference/proceedings 또는 저널 기록이 확인될 것, (2) 위 10편에서 자연스럽게 이어질 것, (3) segmentation·open-vocabulary·depth·3D rendering·multi-view geometry로 주제가 겹치지 않을 것, (4) 공개 원문에서 주장과 venue/year를 직접 확인할 수 있을 것이다. 2026년 논문은 현재 출판·검증이 끝난 후보가 제한적이므로, 제출 상태를 정식 채택으로 오인하지 않고 2023–2025의 확정 논문만 포함했다.

### A. Segment Anything — ICCV 2023

- **왜 주목할까:** segmentation을 point/box/mask prompt에 반응하는 범용 interface로 재정의하고, model-data-engine을 함께 설계했다. U-Net의 task-specific dense prediction에서 promptable foundation model로 넘어가는 연결점이다.
- **선수지식:** U-Net, ViT image encoder, prompt encoding, mask decoder, zero-shot transfer, 데이터 엔진.
- **공식 근거/원문:** CVF 공식 ICCV 2023 프로시딩은 논문명, 연도, 페이지와 promptable task/model/dataset을 명시한다. [ICCV 2023 공식 페이지](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html).

### B. 3D Gaussian Splatting for Real-Time Radiance Field Rendering — ACM TOG 42(4) / SIGGRAPH 2023

- **왜 주목할까:** 장면을 신경망의 암시적 field만으로 표현하지 않고 최적화 가능한 anisotropic 3D Gaussian 집합으로 표현하며, visibility-aware splatting으로 고품질 novel-view rendering과 실시간성을 함께 겨냥한다.
- **선수지식:** camera projection, point cloud, covariance/anisotropic Gaussian, alpha compositing, differentiable rendering, NeRF의 volume rendering.
- **공식 근거/원문:** ACM Digital Library는 제목, TOG 42(4), article 139와 2023년 출판 기록을 제공하고 SIGGRAPH 2023 공식 프로그램은 해당 TOG 호가 journal track임을 설명한다. [ACM 공식 논문](https://doi.org/10.1145/3592433), [SIGGRAPH 2023 공식 프로그램](https://s2023.siggraph.org/program/technical-papers/index.html).

### C. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data — CVPR 2024

- **왜 주목할까:** 새로운 복잡한 module보다 대규모 unlabeled image와 pseudo-label/data engine을 통해 monocular depth의 다양한 장면 일반화를 개선하는 방향을 보여 준다. ViT/self-supervised representation과 dense prediction의 후속 과정으로 적합하다.
- **선수지식:** monocular depth, affine-invariant depth loss, teacher-student/pseudo-label, strong augmentation, semantic encoder prior.
- **공식 근거/원문:** CVF 공식 페이지가 CVPR 2024, pp. 10371–10381과 62M unlabeled image 활용을 명시한다. [CVPR 2024 공식 페이지](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.html).

### D. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection — ECCV 2024

- **왜 주목할까:** DETR/DINO 계열 detector에 언어 조건을 긴밀하게 결합해 category name이나 referring expression으로 open-set object를 찾는다. Faster R-CNN·YOLO의 고정 class 검출과 DETR query를 학습한 뒤 읽기 좋다.
- **선수지식:** DETR object query, open-vocabulary detection, text encoder, cross-modal attention, grounding/referring expression.
- **공식 근거/원문:** ECVA의 ECCV 2024 공식 페이지가 논문 제목과 open-set detector의 feature enhancer, language-guided query selection, cross-modality decoder를 설명한다. [ECCV 2024 공식 페이지](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6319_ECCV_2024_paper.php).

### E. VGGT: Visual Geometry Grounded Transformer — CVPR 2025

- **왜 주목할까:** 하나부터 수백 장의 view에서 camera parameter, depth, point map과 point track을 한 feed-forward Transformer가 함께 추론해, 개별 3D task와 반복 최적화 중심 pipeline을 통합하려는 흐름을 보여 준다.
- **선수지식:** ViT/Transformer, camera intrinsic·extrinsic, multi-view geometry, depth/point map, 3D point tracking.
- **공식 근거/원문:** CVF 공식 프로시딩은 CVPR 2025, pp. 5294–5306과 단일 forward network의 여러 3D attribute 예측을 명시한다. [CVPR 2025 공식 페이지](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html).
