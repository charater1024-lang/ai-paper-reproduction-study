# 생성 모델 논문 독해 노트

> **AI 보조 작성 안내:** 이 한국어 요약은 공식 번역이 아닙니다. 내용과 인용의 최종 기준은
> 연결된 원문입니다. [작성 원칙과 한계](AI_ASSISTED_NOTICE.md) · [70편 목차](README.md)

이 문서는 구현 노트북 10편을 읽기 위한 길잡이다. 각 실습은 논문의 전체 규모나 최종 성능을 재현하지 않고, 핵심 목적식과 알고리즘을 작은 로컬 데이터에서 확인한다.

권장 학습법은 **핵심 수식 Task → 포트폴리오 class 조립 → contract 검증**이다. TODO 1–3에서 논문의 계산·학습·평가를 재현하고, TODO 4에서 같은 계산을 명시적 `Lab` class의 `forward`·loss·update·evaluate 경계로 나눈다. 학습 loop는 이 class의 `update`를 실제로 반복 호출한다. TODO 5에서는 config뿐 아니라 history, parameter 변화와 `evaluate` metric을 함께 검증한 뒤 정답본과 비교하고 재현 한계를 기록한다.

## 00. Stacked Denoising Autoencoders (2010)

원문: [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://www.jmlr.org/papers/v11/vincent10a.html)

1. **한 문장 요약:** 입력을 일부러 훼손한 뒤 깨끗한 입력을 복원하게 하면 단순 복사보다 훼손에 견고한 표현을 배울 수 있다.
2. **배경과 이전 한계:** 보통 autoencoder는 입력과 출력을 같게 두므로 용량이 충분하면 항등함수를 배우기 쉽다. 깊은 비지도 표현학습은 층별 사전학습이 필요했지만, 어떤 국소 학습 기준이 유용한 특징을 만드는지는 분명하지 않았다.
3. **읽기 전 기초지식:** encoder/decoder, 평균제곱오차와 cross-entropy, Bernoulli masking noise, 비선형 활성화, 층별 사전학습.
4. **핵심 아이디어:**
   - $x$를 직접 복원하는 대신 $\tilde x\sim q_D(\tilde x\mid x)$를 입력하고 원본 $x$를 목표로 둔다.
   - corruption은 학습할 필요가 없는 확률 과정이며 encoder가 안정적인 구조를 찾도록 압박한다.
   - 한 DAE의 hidden representation을 다음 DAE의 입력으로 삼아 깊게 쌓는다.
5. **꼭 볼 원문 위치:** `§2.2, Eq. (4)` ↔ reconstruction loss; `§3.1, Denoising Autoencoder Algorithm` ↔ corrupt-then-denoise; `§3.5 and Figure 3` ↔ layer-wise stacking.
6. **수식·알고리즘 직관:** 같은 원본에서 나온 여러 훼손본을 한 점으로 되돌리는 벡터장을 학습한다고 생각하면 된다. 목표가 $\tilde x$ 자체라면 훼손도 복사하지만, 목표가 $x$이면 데이터 manifold 쪽으로 되돌아가는 규칙이 필요하다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/00_stacked_denoising_autoencoder.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/00_stacked_denoising_autoencoder.ipynb)
   - TODO 1: masking corruption과 `64→24→64` DAE가 §3.1의 두 단계다.
   - TODO 2: 매 step 새 훼손본을 만들고 clean target으로 학습한다.
   - TODO 3: noisy-input MSE와 denoised MSE를 비교해 국소 기준의 효과를 확인한다.
   - TODO 4: `DenoisingAutoencoderLab`에 corruption, encode/decode, reconstruction loss, update와 평가를 명시적으로 묶는다.
   - TODO 5: `config()["latent_dim"] == 24`와 class 기반 history·parameter 변화·복원 metric을 함께 검증한다.
8. **5분 마무리:** 요점은 “훼손 입력, 깨끗한 목표”다. 이 노트북은 한 층·8×8 데이터이므로 논문의 깊은 stacking 및 downstream 분류 성능을 재현하지 않는다. **자가점검:** (a) clean target이 필요한 이유는? (b) corruption 확률이 0 또는 1이면 무엇이 잘못되는가? (c) 첫 층 표현을 다음 층에 넣는 절차를 말할 수 있는가?

## 01. Auto-Encoding Variational Bayes (2014)

원문: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

1. **한 문장 요약:** 다루기 어려운 잠재변수 likelihood를 ELBO로 낮춰 쓰고, 재매개변수화로 stochastic encoder를 역전파 가능하게 만든다.
2. **배경과 이전 한계:** 잠재변수 모델의 posterior $p(z\mid x)$는 대개 계산할 수 없고, Monte Carlo 표본을 직접 미분하면 encoder parameter에 gradient를 전달하기 어렵다.
3. **읽기 전 기초지식:** 조건부 Gaussian, KL divergence, Jensen 부등식, Monte Carlo 추정, 자동미분, 로그 likelihood.
4. **핵심 아이디어:**
   - $q_\phi(z\mid x)$를 도입해 log evidence의 하한인 ELBO를 최적화한다.
   - $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$으로 표본의 무작위성을 parameter 밖으로 옮긴다.
   - Gaussian posterior와 standard-normal prior 사이 KL은 닫힌형으로 계산한다.
5. **꼭 볼 원문 위치:** `§2.2, Eq. (1)–(3)` ↔ ELBO; `§2.3, Eq. (4)–(6), Algorithm 1` ↔ SGVB; `§2.4 and §3, Eq. (9)–(10)` ↔ Gaussian 재매개변수화와 KL.
6. **수식·알고리즘 직관:** ELBO는 “데이터를 잘 복원하라”와 “각 입력의 posterior를 공통 prior 가까이 정리하라”의 균형이다. 재매개변수화는 확률 노드를 없애는 것이 아니라 $\epsilon$만 무작위로 두고 나머지를 미분 가능한 함수로 바꾼다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/01_vae.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/01_vae.ipynb)
   - TODO 1: `reparameterize`와 diagonal-Gaussian KL을 구현한다.
   - TODO 2: 작은 encoder/decoder와 negative ELBO를 구성해 SGVB로 학습한다.
   - TODO 3: posterior mean reconstruction, KL, latent geometry를 함께 본다.
   - TODO 4: `VariationalAutoencoderLab`에 encode, 재매개변수화, decode, ELBO와 update/evaluate 경계를 구현한다.
   - TODO 5: `config()["beta"] == 0.08`과 class 기반 history·parameter 변화·ELBO metric을 함께 검증한다.
8. **5분 마무리:** VAE는 autoencoder 모양을 가진 확률적 잠재변수 모델이다. 축소 실습의 MSE likelihood와 작은 $\beta$는 논문 실험의 likelihood 설정과 규모를 대신하지 않는다. **자가점검:** (a) ELBO가 log evidence 이하인 이유는? (b) `logvar`에서 표준편차를 구하는 식은? (c) KL 항을 0으로 두면 sampling에 어떤 문제가 생기는가?

## 02. Generative Adversarial Nets (2014)

원문: [Generative Adversarial Nets](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)

1. **한 문장 요약:** generator와 discriminator의 두 플레이어 게임으로 명시적 likelihood 없이 데이터 분포를 모사한다.
2. **배경과 이전 한계:** likelihood 기반 생성 모델은 확률밀도나 partition function 계산이 어렵고, 당시 생성 샘플 품질과 학습 확장성에도 제약이 있었다.
3. **읽기 전 기초지식:** binary cross-entropy, minimax optimization, 확률변수 변환, Jensen–Shannon divergence, alternating gradient update.
4. **핵심 아이디어:**
   - $D$는 real과 fake의 log-likelihood를 최대화하고 $G$는 이를 속인다.
   - 초기 gradient가 약한 minimax G loss 대신 non-saturating $-\log D(G(z))$를 쓸 수 있다.
   - 이상적 용량과 최적화에서 $p_g=p_{data}$일 때 $D=1/2$가 된다.
5. **꼭 볼 원문 위치:** `§3, Eq. (1)` ↔ minimax objective; `§3, Algorithm 1` ↔ 교대 update; `§3 after Algorithm 1; §4.1 Proposition 1 and Eq. (6)` ↔ non-saturating loss와 균형.
6. **수식·알고리즘 직관:** $D$는 두 분포 사이의 현재 차이를 드러내는 학습된 loss이고, $G$는 그 loss가 가리키는 방향으로 이동한다. 둘의 loss 값이 함께 작아져야 한다는 단순한 해석은 틀릴 수 있다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/02_gan.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/02_gan.ipynb)
   - TODO 1: 2D G/D와 두 BCE loss를 구현한다.
   - TODO 2: fake를 D update에서만 detach하고 D/G를 번갈아 갱신한다.
   - TODO 3: 평균 거리, $D(fake)$, 분포 plot을 함께 해석한다.
   - TODO 4: `AdversarialGameLab`에 generator/discriminator forward, 두 loss와 분리된 D/G update를 구현한다.
   - TODO 5: `config()["objective"] == "non-saturating"`과 class 기반 history·parameter 변화·분포 metric을 함께 검증한다.
8. **5분 마무리:** 핵심은 고정 loss가 아니라 함께 변하는 판별기를 학습 신호로 쓴다는 점이다. 2D 평균 거리는 mode coverage나 실제 이미지 품질을 보장하지 않는다. **자가점검:** (a) G update에서 detach하면 왜 실패하는가? (b) non-saturating loss의 장점은? (c) $D=1/2$만 보고 수렴을 단정할 수 없는 이유는?

## 03. DCGAN (2016)

원문: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

1. **한 문장 요약:** GAN에 안정적인 convolutional 설계 규칙을 적용해 이미지 생성과 잠재표현 분석을 가능하게 했다.
2. **배경과 이전 한계:** 초기 GAN은 MLP 중심이었고 deep convolution을 그대로 붙이면 학습이 불안정했다. pooling, normalization, activation 배치에 대한 재현 가능한 설계 지침이 부족했다.
3. **읽기 전 기초지식:** convolution/transposed convolution, stride와 padding, batch normalization, ReLU/LeakyReLU/tanh, GAN 교대학습.
4. **핵심 아이디어:**
   - pooling 대신 learnable strided convolution을 쓴다.
   - G에는 ReLU와 마지막 tanh, D에는 LeakyReLU를 쓴다.
   - G output과 D input 등 일부 위치를 제외하고 batch normalization을 활용한다.
5. **꼭 볼 원문 위치:** `§3, bullet list` ↔ architecture constraints; `§3 and Figure 1` ↔ generator topology; `§4–§5` ↔ 학습 안정성과 representation/latent 분석.
6. **수식·알고리즘 직관:** 새로운 adversarial objective보다 “gradient가 지나가는 영상 변환 경로”를 규격화한 논문이다. transposed convolution은 latent spatial tensor를 학습 가능한 방식으로 확대한다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/03_dcgan.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/03_dcgan.ipynb)
   - TODO 1: `ConvTranspose2d` G와 strided `Conv2d` D를 만든다.
   - TODO 2: 영상을 `[-1,1]`로 맞춰 tanh 출력과 교대 학습한다.
   - TODO 3: 출력 범위 assertion과 sample/loss plot으로 구조 계약을 확인한다.
   - TODO 4: `ConvolutionalGanLab`에 convolutional G/D, adversarial loss, 교대 update와 평가를 구현한다.
   - TODO 5: `config()["image_shape"] == (1, 8, 8)`과 class 기반 history·parameter 변화·영상 metric을 함께 검증한다.
8. **5분 마무리:** DCGAN의 기여는 architecture recipe와 그 표현의 경험적 분석이다. 8×8·50 update 결과는 LSUN 품질이나 representation transfer를 증명하지 않는다. **자가점검:** (a) tanh와 데이터 범위를 왜 맞추는가? (b) D 입력에 BN을 생략한 지침은 어디에 있는가? (c) stride가 pooling을 대체하는 의미는?

## 04. Wasserstein GAN (2017)

원문: [Wasserstein GAN](https://proceedings.mlr.press/v70/arjovsky17a.html)

1. **한 문장 요약:** 분포 지지집합이 겹치지 않아도 유용한 신호를 주는 Wasserstein-1 거리를 critic의 차이로 근사한다.
2. **배경과 이전 한계:** GAN에서 JSD/KL 계열 신호는 저차원 manifold가 분리되면 포화될 수 있고, discriminator loss와 sample quality가 잘 연동되지 않았다.
3. **읽기 전 기초지식:** Earth Mover distance, Lipschitz 함수, Kantorovich–Rubinstein duality, critic과 classifier 차이, RMSProp, parameter clipping.
4. **핵심 아이디어:**
   - 확률 판별기 대신 제한된 Lipschitz scalar critic을 학습한다.
   - $E[f(x)]-E[f(G(z))]$가 Wasserstein 거리의 근사가 된다.
   - 원 논문의 단순한 Lipschitz 근사는 weight clipping이며 critic을 G보다 여러 번 갱신한다.
5. **꼭 볼 원문 위치:** `§2, Eq. (1)` ↔ Earth Mover distance; `§3, Eq. (2)–(3)` ↔ KR dual critic; `§3, Algorithm 1` ↔ `n_critic`, RMSProp, clipping.
6. **수식·알고리즘 직관:** critic은 real에 높은 값, fake에 낮은 값을 주되 기울기가 무제한 커지지 않는 지형을 만든다. G는 그 지형을 따라 fake를 real 쪽으로 옮긴다; sigmoid가 없는 이유다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/04_wgan.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/04_wgan.ipynb)
   - TODO 1: scalar critic loss와 clipping을 구현한다.
   - TODO 2: critic 3회/G 1회 순서로 Algorithm 1을 축소 실행한다.
   - TODO 3: critic gap과 real/fake 분포를 함께 본다.
   - TODO 4: `WassersteinGanLab`에 scalar critic, Wasserstein loss, clipping과 critic/G update를 구현한다.
   - TODO 5: `config()["critic_steps"] == 3`과 class 기반 history·parameter 변화·critic metric을 함께 검증한다.
8. **5분 마무리:** WGAN은 거리와 학습 신호의 연속성에 초점을 맞춘다. weight clipping은 거친 근사이며 이후 gradient penalty류가 제안됐다. **자가점검:** (a) 마지막 sigmoid가 왜 없는가? (b) clipping은 무엇을 근사하는가? (c) critic gap을 실제 Wasserstein 거리와 동일시할 수 없는 조건은?

## 05. pix2pix (2017)

원문: [Image-to-Image Translation with Conditional Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)

1. **한 문장 요약:** paired 입력–출력 데이터에서 conditional GAN의 현실성 신호와 L1 대응 오차를 결합해 범용 image translation을 학습한다.
2. **배경과 이전 한계:** task별 손실을 손으로 설계하면 perceptual realism을 포착하기 어렵고, L1/L2만으로는 평균화된 흐릿한 출력이 생길 수 있었다.
3. **읽기 전 기초지식:** conditional GAN, paired supervision, L1 loss, U-Net skip connection, receptive field와 PatchGAN.
4. **핵심 아이디어:**
   - D가 $y$만 보지 않고 condition $x$와 쌍 $[x,y]$를 판별한다.
   - cGAN loss에 $\lambda L_1$을 더해 현실성과 입력 대응을 동시에 요구한다.
   - U-Net은 저수준 구조를 우회 전달하고 PatchGAN은 local texture를 판별한다.
5. **꼭 볼 원문 위치:** `§3.1, Eq. (1)` ↔ cGAN; `§3.1, Eq. (3)–(4)` ↔ cGAN+L1; `§3.2` ↔ U-Net과 PatchGAN.
6. **수식·알고리즘 직관:** L1은 “어디에 무엇이 있어야 하는지”를 맞추고 D는 “그 출력이 target domain처럼 보이는지”를 학습한다. 두 항은 서로 대체재가 아니다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/05_pix2pix.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/05_pix2pix.ipynb)
   - TODO 1: source→target G와 `[x,y]` conditional D를 만든다.
   - TODO 2: BCE와 `lambda=10` L1을 결합해 교대 학습한다.
   - TODO 3: paired MAE 개선과 세 분포를 확인한다.
   - TODO 4: `ConditionalTranslationLab`에 conditional G/D, adversarial+L1 objective와 update/evaluate를 구현한다.
   - TODO 5: `config()["l1_weight"] == 10.0`과 class 기반 history·parameter 변화·translation metric을 함께 검증한다.
8. **5분 마무리:** pix2pix는 paired 조건부 생성의 강한 baseline이다. 2D MLP 실습은 U-Net의 spatial skip과 PatchGAN의 실제 patch grid를 직접 재현하지 않는다. **자가점검:** (a) D에 x를 넣어야 하는 이유는? (b) L1만 쓸 때의 위험은? (c) paired 데이터가 없으면 Eq. (3)을 그대로 쓸 수 있는가?

## 06. CycleGAN (2017)

원문: [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)

1. **한 문장 요약:** 짝이 없는 두 domain 사이에서 양방향 adversarial mapping과 cycle consistency를 결합한다.
2. **배경과 이전 한계:** pix2pix류는 정렬된 paired 데이터가 필요하지만 현실에서는 같은 장면의 두 표현을 수집하기 어렵다. 분포만 맞추는 GAN은 입력 내용을 보존하지 않는 임의 mapping도 허용한다.
3. **읽기 전 기초지식:** GAN, unpaired sampling, 양방향 함수, L1 cycle loss, domain translation, identifiability.
4. **핵심 아이디어:**
   - $G:A\to B$, $F:B\to A$와 각 domain discriminator를 둔다.
   - $F(G(a))\approx a$와 $G(F(b))\approx b$를 강제한다.
   - adversarial loss는 target 분포를, cycle loss는 입력 정보의 복구 가능성을 제약한다.
5. **꼭 볼 원문 위치:** `§3.1, Eq. (1)` ↔ adversarial mapping; `§3.2, Eq. (2)` ↔ cycle consistency; `§3.3, Eq. (3)–(4)` ↔ 전체 min–max objective.
6. **수식·알고리즘 직관:** cycle은 두 함수가 서로 근사 역함수가 되게 하지만 의미적으로 올바른 대응을 유일하게 정하지는 않는다. 그래서 domain realism과 cycle만으로 해결되지 않는 permutation/semantic 문제가 남는다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/06_cyclegan.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/06_cyclegan.ipynb)
   - TODO 1: 두 G, 두 D, 양방향 cycle loss를 구성한다.
   - TODO 2: A/B index를 따로 뽑아 실제로 unpaired 학습한다.
   - TODO 3: full-domain cycle error와 $G_{AB}(A)$를 평가한다.
   - TODO 4: `CycleConsistentTranslationLab`에 양방향 G/D, adversarial·cycle loss와 update 경계를 구현한다.
   - TODO 5: `config()["cycle_weight"] == 8.0`과 class 기반 history·parameter 변화·cycle metric을 함께 검증한다.
8. **5분 마무리:** “분포 일치 + 왕복 복원”이 핵심이다. cycle이 낮아도 원하는 semantic mapping이라는 보장은 없고, 실습도 2D domain일 뿐 이미지 artifact를 다루지 않는다. **자가점검:** (a) cycle loss만으로 충분하지 않은 반례는? (b) B batch를 별도로 섞는 이유는? (c) 두 adversarial loss가 각각 어느 domain을 제약하는가?

## 07. DDPM (2020)

원문: [Denoising Diffusion Probabilistic Models](https://papers.nips.cc/paper_files/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)

1. **한 문장 요약:** 고정 Gaussian forward diffusion의 noise를 신경망이 예측하게 학습하고, 이를 역순으로 제거해 표본을 생성한다.
2. **배경과 이전 한계:** likelihood 모델과 GAN 사이에는 학습 안정성, sample quality, tractability의 절충이 있었다. 초기 diffusion 모델은 계산량과 parameterization 면에서 경쟁력이 낮았다.
3. **읽기 전 기초지식:** Gaussian Markov chain, variance schedule, 누적곱 $\bar\alpha_t$, variational bound, U-Net, reparameterized noise.
4. **핵심 아이디어:**
   - forward process $q(x_t\mid x_{t-1})$는 고정하고 점차 Gaussian noise를 더한다.
   - Eq. (4)로 임의 $t$의 $x_t$를 $x_0$에서 한 번에 만든다.
   - reverse mean을 $\epsilon_\theta(x_t,t)$로 parameterize하고 simple noise MSE로 학습한다.
5. **꼭 볼 원문 위치:** `§2, Eq. (2) and Eq. (4)` ↔ forward/noising; `§3.2, Eq. (11)` ↔ epsilon parameterization; `§3.4, Eq. (14), Algorithm 1 and Algorithm 2` ↔ 학습과 sampling.
6. **수식·알고리즘 직관:** $t$가 주어지면 모델은 noisy sample에 섞인 표준 Gaussian 성분을 추정한다. 모든 $t$에서 이 국소 denoising 방향을 배우면 큰 noise에서 데이터까지 작은 역방향 step을 연결할 수 있다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/07_ddpm.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/07_ddpm.ipynb)
   - TODO 1: schedule과 closed-form `q_sample`을 구현한다.
   - TODO 2: 시간 조건 MLP로 random-$t$ noise MSE를 학습한다.
   - TODO 3: Algorithm 2의 reverse mean을 따라 8개 샘플을 만든다.
   - TODO 4: `DiffusionProcessLab`에 schedule buffer, forward noising, noise loss, update와 reverse sampling을 구현한다.
   - TODO 5: `config()["timesteps"] == T`와 class 기반 history·parameter 변화·noise metric을 함께 검증한다.
8. **5분 마무리:** DDPM 학습은 random time의 noise regression, 생성은 전체 time을 역순으로 걷는 과정이다. T=20 MLP 실습은 논문의 U-Net, likelihood weighting, 이미지 품질을 재현하지 않는다. **자가점검:** (a) Eq. (4)가 계산을 어떻게 줄이는가? (b) terminal step에서 noise를 더하지 않는 이유는? (c) $\epsilon$ 예측에서 reverse mean으로 가는 식을 설명할 수 있는가?

## 08. Score-Based Generative Modeling through SDEs (2021)

원문: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

1. **한 문장 요약:** discrete noise levels를 연속시간 SDE로 통합하고, time-dependent score로 reverse-time SDE를 구성한다.
2. **배경과 이전 한계:** score matching과 diffusion은 유사한 denoising 구조를 가졌지만 서로 다른 discrete formulation과 sampler로 다뤄졌다. sampling 정확도와 likelihood 계산을 하나의 틀에서 설명하기 어려웠다.
3. **읽기 전 기초지식:** score $\nabla_x\log p(x)$, Ito SDE, reverse-time diffusion, denoising score matching, Euler–Maruyama, Langevin dynamics.
4. **핵심 아이디어:**
   - forward SDE가 data를 tractable prior로 연속 변환한다.
   - reverse SDE의 drift는 현재 marginal score를 필요로 한다.
   - continuous denoising score matching으로 모든 시간의 score를 하나의 network가 학습한다.
   - predictor-corrector와 probability-flow ODE 등 여러 sampler가 같은 모델에서 나온다.
5. **꼭 볼 원문 위치:** `§3.1 Eq. (5); §3.2 Eq. (6)` ↔ forward/reverse SDE; `§3.3, Eq. (7)` ↔ score objective; `§3.4 Eq. (11); §4.2 and Appendix G Algorithm 1` ↔ VP-SDE와 PC sampler.
6. **수식·알고리즘 직관:** forward noise가 확률질량을 퍼뜨리면 score는 noisy density가 증가하는 방향을 가리킨다. reverse drift에서 score를 빼고 시간을 뒤로 적분하면 퍼짐을 되돌릴 수 있다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/08_score_sde.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/08_score_sde.ipynb)
   - TODO 1: VP marginal과 conditional target score를 만든다.
   - TODO 2: 시간 조건 score MLP를 Eq. (7) weighted MSE로 학습한다.
   - TODO 3: Eq. (6)의 reverse drift와 diffusion으로 Euler sampling한다.
   - TODO 4: `ScoreSdeLab`에 VP perturbation, score loss, update와 reverse-SDE 평가를 구현한다.
   - TODO 5: `config()["state_dim"] == 2`와 class 기반 history·parameter 변화·score metric을 함께 검증한다.
8. **5분 마무리:** score network 하나가 reverse SDE와 ODE 계열 sampler의 공통 재료다. 2D Euler 축소판은 solver 오차, corrector, likelihood ODE, 고차원 성능을 평가하지 않는다. **자가점검:** (a) conditional score target은 왜 $-z/\sigma(t)$인가? (b) reverse drift에 score가 어디에 들어가는가? (c) predictor와 corrector의 역할 차이는?

## 09. Latent Diffusion Models (2022)

원문: [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)

1. **한 문장 요약:** perceptual autoencoder가 만든 저차원 latent에서 diffusion을 수행해 이미지 공간 diffusion의 계산 부담을 줄인다.
2. **배경과 이전 한계:** pixel-space diffusion은 고해상도에서 대부분의 계산을 지각적으로 중요하지 않은 세부에 쓰며, 조건부 생성 구조를 task마다 별도로 설계하기 쉽다.
3. **읽기 전 기초지식:** autoencoder와 reconstruction, DDPM epsilon loss, latent downsampling factor, U-Net, attention의 Q/K/V, conditioning.
4. **핵심 아이디어:**
   - 먼저 $E,D$가 의미를 보존하는 압축 latent space를 학습한다.
   - diffusion loss를 $x$가 아니라 $z=E(x)$에서 계산한다.
   - cross-attention으로 text·layout 등 다양한 condition을 U-Net에 주입한다.
   - 압축률이 너무 낮으면 계산 이득이 없고 너무 높으면 세부가 비가역적으로 사라진다.
5. **꼭 볼 원문 위치:** `§3.1` ↔ perceptual compression; `§3.2, Eq. (1)–(2)` ↔ pixel/latent diffusion objectives; `§3.3, Eq. (3)` ↔ cross-attention conditioning.
6. **수식·알고리즘 직관:** 생성 모델이 어려운 두 일을 분리한다. autoencoder는 perceptual compression을, diffusion은 압축 표현의 분포 학습을 맡는다. decoder가 잃은 정보는 diffusion이 되살릴 수 없다.
7. **실습 연결:** [실습본](../../notebooks/field_reproductions/generative/exercises/09_latent_diffusion.ipynb) · [정답본](../../notebooks/field_reproductions/generative/solutions/09_latent_diffusion.ipynb)
   - TODO 1: `64→8` autoencoder를 먼저 학습한다.
   - TODO 2: detached/standardized latent에서 diffusion noise predictor를 학습한다.
   - TODO 3: latent reverse sample을 decoder로 복원하고 8배 차원 축소를 확인한다.
   - TODO 4: `LatentDiffusionLab`에 autoencoder, latent diffusion loss, 단계별 update와 decode 평가를 구현한다.
   - TODO 5: `config()["latent_dim"] == 8`과 class 기반 history·parameter 변화·latent metric을 함께 검증한다.
8. **5분 마무리:** LDM의 핵심은 “좋은 압축 공간에서 diffusion”이다. 실습의 MSE autoencoder는 논문의 perceptual/adversarial compression과 cross-attention을 구현하지 않는다. **자가점검:** (a) 압축률 trade-off는 무엇인가? (b) encoder를 먼저 고정하는 이유는? (c) Eq. (2)가 Eq. (1)과 다른 변수는?

---

## 2023–2026 후속 읽기 5편

### 선정 기준

- 2023년 이후 CVPR/ICCV/ICML/ICLR/NeurIPS 본 학회에 채택된 논문만 포함했다.
- venue와 연도는 arXiv 표기가 아니라 CVF, PMLR, OpenReview의 ICLR conference record, NeurIPS proceedings에서 확인했다.
- 위 10편 다음에 읽었을 때 **backbone scaling, 연속 흐름, 빠른 sampling, 학습 안정성, autoregressive 대안**으로 지식이 확장되도록 골랐다.
- “최고”라는 단일 순위를 뜻하지 않으며, 대규모 benchmark 수치를 로컬 실습에서 재현할 수 있다는 뜻도 아니다.

1. **Flow Matching for Generative Modeling — ICLR 2023**  
   [ICLR 공식 OpenReview](https://openreview.net/forum?id=PqvMRDCJT9t) · **선정 이유:** simulation 없이 conditional probability path의 vector field를 회귀하는 formulation으로 Score-SDE 다음의 연속시간 흐름을 정리하기 좋다. **선행지식:** continuous normalizing flow, ODE, change of variables, optimal transport 기초, Score-SDE.

2. **Consistency Models — ICML 2023**  
   [PMLR 공식 프로시딩](https://proceedings.mlr.press/v202/song23a.html) · **선정 이유:** 반복 diffusion sampling을 one/few-step mapping으로 줄이는 핵심 계열이며 distillation과 standalone training을 비교해 볼 수 있다. **선행지식:** DDPM/Score-SDE, probability-flow ODE, teacher distillation, boundary condition.

3. **Scalable Diffusion Models with Transformers — ICCV 2023**  
   [CVF 공식 프로시딩](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) · **선정 이유:** latent diffusion의 U-Net을 patch Transformer(DiT)로 바꾸고 model compute와 생성 품질의 scaling을 분석한다. **선행지식:** LDM, Vision Transformer, adaptive layer normalization, classifier-free guidance, FID.

4. **Analyzing and Improving the Training Dynamics of Diffusion Models — CVPR 2024**  
   [CVF 공식 프로시딩](https://openaccess.thecvf.com/content/CVPR2024/html/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.html) · **선정 이유:** diffusion network의 weight·activation·update magnitude 불균형과 post-hoc EMA를 체계적으로 분석해 “목적식 이후의 학습 공학”을 배울 수 있다. **선행지식:** DDPM/EDM parameterization, normalization, initialization, EMA, FID.

5. **Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction — NeurIPS 2024**  
   [NeurIPS 공식 프로시딩](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a24e284b187f662681440ba15c416fb-Abstract-Conference.html) · **선정 이유:** raster-order next-token 대신 coarse-to-fine next-scale prediction을 사용해 diffusion과 다른 현대 이미지 생성 축을 제공한다. **선행지식:** autoregressive likelihood, VQ tokenizer, multi-scale representation, Transformer, classifier-free guidance.

### 검증 메모

위 링크의 공식 기록은 각각 `ICLR 2023 conference paper`, `Proceedings of the 40th ICML (2023)`, `ICCV 2023`, `CVPR 2024`, `NeurIPS 2024 Main Conference Track`을 명시한다. 추천은 발표 venue 확인과 교육적 연결성을 기준으로 했으며, 후속 연구의 우열을 단정하지 않는다.
