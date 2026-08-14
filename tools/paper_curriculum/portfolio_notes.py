"""Paper-specific equations and implementation notes for the 20-paper track."""

from __future__ import annotations

from paper_curriculum.common import clean

PORTFOLIO_NOTES: dict[int, str] = {
    0: r"""
    ## 핵심 수식과 구현 설계

    $$
    h^{(l)}_{k,i,j}=\sigma\!\left(
    b_k^{(l)}+\sum_c\sum_{u,v}W^{(l)}_{k,c,u,v}
    h^{(l-1)}_{c,i+u,j+v}\right)
    $$

    - $h^{(l)}$은 $l$번째 feature map, $W$는 공유되는 convolution kernel입니다.
    - 핵심 역할은 국소 receptive field와 weight sharing으로 공간 패턴을 추출하는 것입니다.
    - 코드의 convolution/pooling 모듈이 식의 합과 샘플링을 담당합니다.
    - shape은 $[B,C,H,W]\rightarrow[B,C',H',W']$로 변하고 classifier에서 $[B,K]$가 됩니다.
    """,
    1: r"""
    ## 핵심 수식과 구현 설계

    $$
    \operatorname{ReLU}(x)=\max(0,x),\qquad
    p(y=k\mid x)=\operatorname{softmax}(z)_k
    $$

    - ReLU는 음수 activation을 0으로 만들어 깊은 CNN의 gradient 전달을 돕습니다.
    - $z$는 마지막 linear layer의 logit이고 softmax는 $K$개 class 확률로 변환합니다.
    - 코드에서 feature extractor, dropout, classifier를 따로 두어 AlexNet의 블록 경계를 보이게 합니다.
    - 입력 $[B,1,H,W]$는 feature map을 거쳐 logit $[B,K]$로 변합니다.
    """,
    2: r"""
    ## 핵심 수식과 구현 설계

    $$
    d_l=\operatorname{Conv}\!\left(
    \operatorname{Concat}(\operatorname{Up}(d_{l+1}),e_l)\right)
    $$

    - $e_l$은 encoder의 고해상도 feature, $d_l$은 decoder feature입니다.
    - skip connection은 downsampling에서 잃을 수 있는 위치 정보를 decoder에 직접 전달합니다.
    - encoder, bottleneck, up block을 class로 분리하고 concat 전후 channel을 assertion으로 검사합니다.
    - 입력과 mask는 $[B,1,H,W]$이며 출력 logit도 같은 공간 shape을 가져야 합니다.
    """,
    3: r"""
    ## 핵심 수식과 구현 설계

    $$
    y=\mathcal{F}(x;W)+x
    $$

    - $\mathcal{F}$는 residual branch가 학습하는 변환이고 $x$는 identity shortcut입니다.
    - 목표는 전체 mapping을 새로 학습하기보다 입력에 더할 잔차를 학습하는 것입니다.
    - residual block class가 main branch와 shortcut projection을 명시적으로 구분합니다.
    - 더셈을 위해 두 branch의 shape이 같아야 하며, stride/channel 변화 시 projection을 사용합니다.
    """,
    4: r"""
    ## 핵심 수식과 구현 설계

    $$
    m_i\sim\operatorname{Bernoulli}(q),\qquad
    h_i^{\mathrm{train}}=m_i h_i,\qquad
    h_i^{\mathrm{test}}=q h_i
    $$

    - $q=1-p$는 unit 보존 확률이고 $m_i$는 학습 시 샘플되는 mask입니다.
    - 위 식은 원 논문의 관례입니다. train에서는 mask만 곱하고 test에서는 outgoing weight를
      $q$배하며, 실습은 동치인 activation $q h_i$를 계산합니다.
    - PyTorch의 inverted dropout은 train에서 $m_i h_i/q$, eval에서 $h_i$를 사용합니다.
      두 관례의 기대값은 같지만 각 mode의 중간 activation scale은 다릅니다.
    - 실습은 원 논문 관례의 mask, mode 전환, PyTorch 관례와의 차이를 함수로 확인합니다.
    - activation shape은 유지되지만 train mode에서만 원소별 0이 생깁니다.
    """,
    5: r"""
    ## 핵심 수식과 구현 설계

    $$
    \hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},
    \qquad y_i=\gamma\hat x_i+\beta
    $$

    - $\mu_B,\sigma_B^2$는 mini-batch 통계, $\gamma,\beta$는 학습 가능한 affine parameter입니다.
    - 학습 중에는 batch 통계, 평가 중에는 running 통계를 사용하는 차이가 핵심입니다.
    - 직접 구한 normalization과 framework layer의 출력·gradient를 비교합니다.
    - dense 입력은 feature 축, CNN 입력은 channel 축별로 통계를 계산합니다.
    """,
    6: r"""
    ## 핵심 수식과 구현 설계

    $$
    \mathcal{L}_{\mathrm{SGNS}}=-\log\sigma(v'_w{}^\top v_c)
    -\sum_{j=1}^{K}\log\sigma(-v'_{n_j}{}^\top v_c)
    $$

    - $v_c$는 center, $v'_w$는 positive context, $v'_{n_j}$는 negative word embedding입니다.
    - 전체 vocabulary softmax 대신 positive pair와 $K$개 negative pair를 binary 분류합니다.
    - window pair 생성, unigram negative sampling, embedding 점수, loss를 따로 구현합니다.
    - index $[B]$는 embedding $[B,D]$로 변하고 dot product는 logit $[B]$를 만듭니다.
    """,
    7: r"""
    ## 핵심 수식과 구현 설계

    $$
    p(y_{1:T}\mid x_{1:S})=\prod_{t=1}^{T}
    p(y_t\mid y_{<t},c),\qquad c=\operatorname{Encoder}(x_{1:S})
    $$

    - $c$는 source sequence를 요약한 encoder state, $y_{<t}$는 이전 target token입니다.
    - decoder는 teacher forcing으로 학습하고 autoregressive loop로 추론합니다.
    - vocabulary, encoder, decoder, sequence loss, greedy decode를 명시적으로 분리합니다.
    - token $[B,S]$는 recurrent state $[L,B,H]$를 거쳐 target logit $[B,T,V]$가 됩니다.
    """,
    8: r"""
    ## 핵심 수식과 구현 설계

    $$
    e_{t,i}=v_a^\top\tanh(W_s s_{t-1}+W_h h_i),\quad
    \alpha_{t,i}=\operatorname{softmax}_i(e_{t,i}),\quad
    c_t=\sum_i\alpha_{t,i}h_i
    $$

    - $h_i$는 encoder state, $s_{t-1}$는 decoder state, $\alpha_{t,i}$는 alignment weight입니다.
    - 하나의 고정 context 대신 decoder step마다 source의 다른 위치를 가중 합산합니다.
    - energy, masked softmax, context aggregation을 분리해 alignment을 직접 검증합니다.
    - score와 weight는 $[B,T,S]$, context는 $[B,T,H]$이며 padding 위치 가중치는 0이어야 합니다.
    """,
    9: r"""
    ## 핵심 수식과 구현 설계

    $$
    \operatorname{Attention}(Q,K,V)=
    \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
    $$

    - $Q,K,V$는 query·key·value, $d_k$는 head dimension, $M$은 causal/padding mask입니다.
    - $1/\sqrt{d_k}$ scaling은 dot-product variance를 줄여 softmax saturation을 방지합니다.
    - projection, head split/merge, mask, attention weight를 순서대로 직접 구현합니다.
    - $[B,T,D]\rightarrow[B,H,T,d_k]\rightarrow[B,T,D]$의 shape 변화를 assertion으로 확인합니다.
    """,
    10: r"""
    ## 핵심 수식과 구현 설계

    $$
    \mathcal{L}_{\mathrm{LM}}=-\sum_{t=1}^{T}
    \log p(x_t\mid x_{<t}),\qquad
    \mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda\mathcal{L}_{\mathrm{LM}}
    $$

    - $\mathcal{L}_{\mathrm{LM}}$은 비라벨 문장의 다음 token 예측, $\mathcal{L}_{\mathrm{task}}$는 감독 목표입니다.
    - 동일 Transformer decoder를 pretraining 후 작은 태스크에 fine-tuning하는 연결을 재현합니다.
    - causal model, LM batch, task head, 두 loss를 따로 작성해 objective 경계를 보입니다.
    - token logit은 $[B,T,V]$, task logit은 sequence representation으로부터 $[B,C]$가 됩니다.
    """,
    11: r"""
    ## 핵심 수식과 구현 설계

    $$
    \mathcal{L}_{\mathrm{BERT}}=
    \mathcal{L}_{\mathrm{MLM}}+\mathcal{L}_{\mathrm{NSP}},\qquad
    \mathcal{L}_{\mathrm{MLM}}=-\sum_{i\in\mathcal{M}}
    \log p(x_i\mid x_{\setminus\mathcal{M}})
    $$

    - $\mathcal{M}$은 masking된 token 집합이고 NSP는 두 segment의 연속 여부를 판별합니다.
    - bidirectional attention이 mask 좌·우 context를 모두 사용하는지 단방향 기준과 비교합니다.
    - token/position/segment embedding, encoder, MLM/NSP head를 별도 모듈로 나눕니다.
    - encoder 출력 $[B,T,D]$에서 MLM $[B,T,V]$와 NSP $[B,2]$를 계산합니다.
    """,
    12: r"""
    ## 핵심 수식과 구현 설계

    $$
    \log p_\theta(x)\ge
    \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
    -D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))
    $$

    - encoder $q_\phi$는 $\mu,\log\sigma^2$를, decoder $p_\theta$는 reconstruction을 출력합니다.
    - reconstruction과 latent prior regularization의 균형이 VAE objective의 핵심입니다.
    - reparameterization $z=\mu+\sigma\odot\epsilon$을 함수로 분리해 gradient 경로를 확인합니다.
    - $x:[B,D_x]$에서 $\mu,\log\sigma^2,z:[B,D_z]$를 거쳐 $\hat x:[B,D_x]$로 복원합니다.
    """,
    13: r"""
    ## 핵심 수식과 구현 설계

    $$
    \min_G\max_D\;
    \mathbb{E}_{x\sim p_{data}}[\log D(x)]
    +\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
    $$

    - generator $G$는 noise를 sample로, discriminator $D$는 sample을 real 확률로 변환합니다.
    - 두 모델의 목표가 반대인 minimax game이므로 optimizer와 gradient 경로를 분리합니다.
    - discriminator step과 generator step을 독립 함수로 구현하고 해당 step의 parameter만 변하는지 보장합니다.
    - $z:[B,D_z]\rightarrow G(z):[B,D_x]\rightarrow D(\cdot):[B,1]$의 흐름입니다.
    """,
    14: r"""
    ## 핵심 수식과 구현 설계

    $$
    y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a'),\qquad
    \mathcal{L}(\theta)=\mathbb{E}[(y-Q_\theta(s,a))^2]
    $$

    - $\theta$는 online network, $\theta^-$는 고정된 target network, $d$는 terminal flag입니다.
    - replay buffer는 상관된 transition을 섞고 target network는 bootstrapping target을 느리게 변화시킵니다.
    - transition 저장, batch sampling, TD target, optimization, target sync를 명시적으로 나눕니다.
    - state batch $[B,D_s]$에서 action value $[B,A]$를 만들고 gather로 $Q(s,a):[B]$를 선택합니다.
    """,
    15: r"""
    ## 핵심 수식과 구현 설계

    $$
    H^{(l+1)}=\sigma\!\left(
    \hat D^{-1/2}\hat A\hat D^{-1/2}H^{(l)}W^{(l)}\right),
    \qquad \hat A=A+I
    $$

    - $\hat A$는 self-loop를 추가한 adjacency, $\hat D$는 degree matrix입니다.
    - symmetric normalization은 이웃 수가 다른 node의 message 크기를 보정합니다.
    - adjacency 준비, normalization, graph convolution layer, node loss를 직접 분리합니다.
    - $H^{(l)}:[N,D_l]$는 normalized adjacency $[N,N]$와 가중치 $[D_l,D_{l+1}]$를 거칩니다.
    """,
    16: r"""
    ## 핵심 수식과 구현 설계

    $$
    \ell_{i,j}=-\log
    \frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
    {\sum_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}
    $$

    - $z_i,z_j$는 같은 image의 두 augmentation view, $\tau$는 temperature입니다.
    - positive pair는 가깝게, batch의 다른 view는 멀게 만드는 NT-Xent objective입니다.
    - augmentation, encoder, projection head, similarity matrix, masked loss를 단계별로 구현합니다.
    - $2B$개 view의 embedding $[2B,D]$에서 similarity matrix $[2B,2B]$가 만들어집니다.
    """,
    17: r"""
    ## 핵심 수식과 구현 설계

    $$
    z_0=[x_{class};x_p^1E;\dots;x_p^NE]+E_{pos},\qquad
    N=\frac{HW}{P^2}
    $$

    - $x_p^i$는 $P\times P$ image patch, $E$는 patch projection, $E_{pos}$는 position embedding입니다.
    - 2D image를 token sequence로 바꾸어 Transformer encoder에 입력하는 것이 핵심입니다.
    - patchify, linear projection, class token, position embedding, encoder, head를 별도 모듈로 보입니다.
    - image $[B,C,H,W]$는 patch $[B,N,CP^2]$를 거쳐 token $[B,N+1,D]$가 됩니다.
    """,
    18: r"""
    ## 핵심 수식과 구현 설계

    $$
    q(x_t\mid x_0)=\mathcal{N}\!\left(
    x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right),\quad
    \mathcal{L}=\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2
    $$

    - $\bar\alpha_t$는 누적 noise schedule, $\epsilon_\theta$는 timestep 조건부 noise predictor입니다.
    - forward process는 닫힌 형태로 $x_t$를 만들고 reverse model은 주입된 noise를 예측합니다.
    - schedule, timestep sampling, noising, predictor, denoising step을 각각 함수로 구현합니다.
    - 이 축소 실험의 $x_0,x_t,\epsilon$ shape은 모두 $[B,2]$이며 timestep $[B]$가 조건으로 들어갑니다.
    """,
    19: r"""
    ## 핵심 수식과 구현 설계

    $$
    s_{ij}=\frac{f_I(I_i)^\top f_T(T_j)}{\tau},\qquad
    \mathcal{L}_{\mathrm{CLIP}}=
    \tfrac12\bigl(\operatorname{CE}(s,y)+\operatorname{CE}(s^\top,y)\bigr)
    $$

    - $f_I,f_T$는 image/text encoder, $s_{ij}$는 정규화된 cross-modal 유사도입니다.
    - 같은 batch index를 positive로 두고 image-to-text와 text-to-image 분류를 대칭 학습합니다.
    - 두 encoder, L2 normalization, logit scale, 대칭 loss, zero-shot 비교를 분리해 구현합니다.
    - embedding $[B,D]$ 두 개의 행렬곱으로 similarity matrix $[B,B]$를 만듭니다.
    """,
}


PORTFOLIO_SHAPES: dict[int, str] = {
    0: "image [B,1,32,32] → feature [B,C,H,W] → logits [B,10]",
    1: "image [B,1,H,W] → five convolution stages → logits [B,K]",
    2: "image/mask [B,1,H,W] → skip features [B,C,H',W'] → logits [B,1,H,W]",
    3: "input/shortcut [B,C,H,W] → residual output [B,C',H',W'] → logits [B,K]",
    4: "activation [B,H] → 같은 shape의 masked/scaled activation [B,H]",
    5: "batch [B,D] 또는 [B,C,H,W] → 같은 shape의 normalized activation",
    6: "center/context ids [B] → embeddings [B,D] → positive/negative logits [B]",
    7: "source [B,S] → recurrent state [L,B,H] → target logits [B,T,V]",
    8: "encoder/decoder [B,S/T,H] → attention [B,T,S] → context [B,T,H]",
    9: "hidden [B,T,D] ↔ heads [B,H,T,dₖ] → block output [B,T,D]",
    10: "tokens [B,T] → LM logits [B,T,V]와 task logits [B,C]",
    11: "tokens [B,T] → hidden [B,T,D] → MLM [B,T,V], NSP [B,2]",
    12: "x [B,Dₓ] → μ/logσ²/z [B,D_z] → reconstruction [B,Dₓ]",
    13: "noise [B,D_z] → samples [B,Dₓ] → discriminator logits [B,1]",
    14: "states [B,Dₛ] → action values [B,A] → selected Q/TD target [B]",
    15: "A [N,N], X [N,D] → graph hidden [N,H] → node logits [N,K]",
    16: "two views [B,C,H,W] → projections [B,D] → similarity [2B,2B]",
    17: "image [B,C,H,W] → patches [B,N,CP²] → tokens [B,N+1,D]",
    18: "clean/noisy/noise [B,2] + timestep [B] → predicted noise [B,2]",
    19: "image/text features [B,*] → embeddings [B,D] → similarity [B,B]",
}


def portfolio_note(number: int) -> str:
    """Return the equation-to-code explanation for one paper number."""

    try:
        return clean(PORTFOLIO_NOTES[number])
    except KeyError as error:
        raise ValueError(f"missing portfolio note for paper {number:02d}") from error


def portfolio_shape(number: int) -> str:
    """Return the primary tensor-shape contract for one paper lab."""

    try:
        return PORTFOLIO_SHAPES[number]
    except KeyError as error:
        raise ValueError(f"missing shape contract for paper {number:02d}") from error
