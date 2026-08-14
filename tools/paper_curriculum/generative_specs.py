"""Curriculum specifications for generative, reinforcement, and graph models.

The exercises in this module intentionally use small synthetic data.  They keep
the mathematical mechanism of each paper while remaining deterministic and
fast enough to run on a CPU without network access.
"""

from .common import PaperSpec, code, markdown, paired_markdown, shared_code

VAE = PaperSpec(
    number=12,
    slug="vae",
    short_title="VAE",
    paper_title="Auto-Encoding Variational Bayes",
    authors="Diederik P. Kingma and Max Welling",
    year=2014,
    primary_url="https://arxiv.org/abs/1312.6114",
    venue="ICLR",
    difficulty="중급",
    expected_minutes=60,
    prerequisites="PyTorch 기본, 확률분포, KL divergence, 자동미분",
    reproduction_goal=(
        "2차원 혼합분포를 대상으로 재매개변수화, 닫힌형태 KL, ELBO를 직접 구현하고 "
        "작은 VAE가 입력을 복원하면서 잠재분포를 표준정규 사전에 가깝게 만드는지 확인한다."
    ),
    original_scale=(
        "논문은 MNIST와 Frey Face에 확률적 인코더·디코더를 적용한다. 여기서는 다운로드 없이 "
        "512개의 2차원 점과 2차원 잠재변수로 같은 SGVB 메커니즘만 재현한다."
    ),
    mappings=(
        (
            "§2.2, Eq. (1)–(3)",
            "`vae_objective`: reconstruction loss와 `kl_standard_normal`의 합",
            "ELBO의 두 항, scalar loss, finite gradient를 각각 검사",
        ),
        (
            "§2.3, Eq. (4)–(8), Algorithm 1",
            "`reparameterize`와 `TinyVAE.forward`의 `z = mu + std * epsilon`",
            "latent shape와 encoder까지 이어지는 gradient를 검사",
        ),
        (
            "§2.4 및 §3, Eq. (9)–(10)",
            "`TinyVAE.encode`의 `mu`, `logvar` head와 닫힌형태 KL",
            "`mu`·`logvar` shape와 KL의 비음수성을 검사",
        ),
    ),
    cells=(
        markdown(
            r"""
            ## 1. 논문의 핵심을 작은 문제로 옮기기

            VAE는 관측 $x$의 로그우도 대신 다음 evidence lower bound(ELBO)를 최대화합니다.

            $$\mathcal L(x)=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
            -D_{KL}(q_\phi(z\mid x)\Vert p(z)).$$

            이 식은 논문 §2.2 Eq. (3)에 대응합니다. §2.3–2.4의 핵심은
            $z=\mu+\sigma\odot\epsilon$, $\epsilon\sim\mathcal N(0,I)$로 표본 경로를
            미분 가능하게 만드는 것입니다. 원 논문의 이미지 likelihood 대신 여기서는
            2차원 점의 Gaussian reconstruction error를 사용합니다.

            **성공 기준:** 학습 뒤 결정론적 복원 MSE가 줄고, 전체 loss가 초기값보다 낮아야 합니다.
            """,
            "paper-map",
        ),
        shared_code(
            """
            import math
            import numpy as np
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            import torch.nn.functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(120)
            np.random.seed(120)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device
            # This tiny teaching batch may be faster with AI_LAB_DEVICE=cpu;
            # auto acceleration preserves the same code path for larger variants.

            data_rng = torch.Generator().manual_seed(120)
            centers = torch.tensor([
                [-1.6, -1.6], [-1.6, 1.6], [1.6, -1.6], [1.6, 1.6]
            ], dtype=torch.float32)
            component = torch.randint(0, len(centers), (512,), generator=data_rng)
            points = centers[component] + 0.22 * torch.randn(512, 2, generator=data_rng)
            order = torch.randperm(len(points), generator=data_rng)
            points, component = points[order], component[order]
            points, component = ACCELERATOR.move(points, component)
            x_train, x_test = points[:384], points[384:]
            y_train, y_test = component[:384], component[384:]

            print(ACCELERATOR.summary())
            print(f"train={tuple(x_train.shape)}, test={tuple(x_test.shape)}")
            assert x_train.shape == (384, 2)
            assert torch.isfinite(x_train).all()
            """,
            "setup",
        ),
        code(
            """
            # TODO 1: 논문 §2.4의 z = mu + sigma * epsilon을 완성하세요.
            def reparameterize(mu, logvar, eps=None):
                raise NotImplementedError("mu + exp(0.5*logvar) * eps를 구현하세요")

            # TODO 2: q=N(mu, diag(sigma^2)), p=N(0,I)의 샘플별 KL을 완성하세요.
            def kl_standard_normal(mu, logvar):
                raise NotImplementedError("Eq. (10)의 닫힌형태 KL을 구현하세요")

            probe_mu = torch.zeros(4, 2, requires_grad=True)
            probe_logvar = torch.zeros(4, 2)
            probe_z = reparameterize(probe_mu, probe_logvar, eps=torch.zeros_like(probe_mu))
            assert torch.allclose(probe_z, probe_mu)
            assert torch.allclose(kl_standard_normal(probe_mu, probe_logvar), torch.zeros(4))
            probe_z.sum().backward()
            assert probe_mu.grad is not None
            """,
            """
            def reparameterize(mu, logvar, eps=None):
                # §2.4: Gaussian sample을 미분 가능한 위치-척도 변환으로 표현한다.
                std = torch.exp(0.5 * logvar)
                if eps is None:
                    eps = torch.randn_like(std)
                return mu + std * eps

            def kl_standard_normal(mu, logvar):
                # §3 Eq. (10)의 -KL 항에 부호를 바꾼, 양수 KL을 샘플별로 반환한다.
                return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

            probe_mu = torch.zeros(4, 2, requires_grad=True)
            probe_logvar = torch.zeros(4, 2)
            probe_z = reparameterize(probe_mu, probe_logvar, eps=torch.zeros_like(probe_mu))
            assert torch.allclose(probe_z, probe_mu)
            assert torch.allclose(kl_standard_normal(probe_mu, probe_logvar), torch.zeros(4))
            probe_z.sum().backward()
            assert probe_mu.grad is not None
            print("재매개변수화 경로와 Gaussian KL 검증 통과")
            """,
            "implementation",
        ),
        code(
            """
            class TinyVAE(nn.Module):
                def __init__(self, latent_dim=2):
                    super().__init__()
                    self.encoder = nn.Sequential(nn.Linear(2, 24), nn.Tanh())
                    # TODO: 평균/로그분산 head와 2층 decoder를 정의하세요.

                def encode(self, x):
                    raise NotImplementedError

                def decode(self, z):
                    raise NotImplementedError

                def forward(self, x):
                    raise NotImplementedError

            vae = TinyVAE().to(DEVICE)
            reconstruction, mu, logvar = vae(x_train[:8])
            assert reconstruction.shape == (8, 2)
            assert mu.shape == logvar.shape == (8, 2)
            """,
            """
            class TinyVAE(nn.Module):
                def __init__(self, latent_dim=2):
                    super().__init__()
                    self.encoder = nn.Sequential(nn.Linear(2, 24), nn.Tanh())
                    self.mu_head = nn.Linear(24, latent_dim)
                    self.logvar_head = nn.Linear(24, latent_dim)
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 24), nn.Tanh(), nn.Linear(24, 2)
                    )

                def encode(self, x):
                    h = self.encoder(x)
                    return self.mu_head(h), self.logvar_head(h).clamp(-8.0, 8.0)

                def decode(self, z):
                    return self.decoder(z)

                def forward(self, x):
                    mu, logvar = self.encode(x)
                    z = reparameterize(mu, logvar)
                    return self.decode(z), mu, logvar

            vae = TinyVAE().to(DEVICE)
            reconstruction, mu, logvar = vae(x_train[:8])
            assert reconstruction.shape == (8, 2)
            assert mu.shape == logvar.shape == (8, 2)
            assert all(parameter.requires_grad for parameter in vae.parameters())
            print(f"TinyVAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
            """,
            "implementation",
        ),
        code(
            """
            # TODO 1: reconstruction + beta * KL인 음의 ELBO를 구현하세요.
            def vae_objective(model, batch, beta=0.10, deterministic=False):
                raise NotImplementedError

            # TODO 2: Adam으로 300번의 작은 full-batch SGVB update를 수행하세요.
            optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
            history = []
            with torch.no_grad():
                initial_loss = float(vae_objective(vae, x_train, deterministic=True)[0])

            for step in range(300):
                pass

            assert len(history) == 300
            """,
            """
            def vae_objective(model, batch, beta=0.10, deterministic=False):
                mu, logvar = model.encode(batch)
                eps = torch.zeros_like(mu) if deterministic else None
                reconstruction = model.decode(reparameterize(mu, logvar, eps))
                recon_term = F.mse_loss(reconstruction, batch, reduction="none").sum(dim=1)
                kl_term = kl_standard_normal(mu, logvar)
                loss = (recon_term + beta * kl_term).mean()
                return loss, recon_term.mean(), kl_term.mean()

            optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
            history = []
            with torch.no_grad():
                initial_loss = float(vae_objective(vae, x_train, deterministic=True)[0])

            for step in range(300):
                optimizer.zero_grad()
                loss, recon_term, kl_term = vae_objective(vae, x_train)
                loss.backward()
                optimizer.step()
                history.append(
                    (
                        float(loss.detach()),
                        float(recon_term.detach()),
                        float(kl_term.detach()),
                    )
                )

            with torch.no_grad():
                final_loss = float(vae_objective(vae, x_train, deterministic=True)[0])
            print(f"deterministic negative-ELBO: {initial_loss:.3f} -> {final_loss:.3f}")
            assert len(history) == 300
            assert math.isfinite(final_loss) and final_loss < 0.65 * initial_loss
            """,
            "training",
        ),
        code(
            """
            # TODO: test 복원 MSE와 평균 KL을 계산하고, loss/latent/reconstruction을 그리세요.
            raise NotImplementedError("metric과 세 개의 subplot을 완성하세요")
            """,
            """
            vae.eval()
            with torch.no_grad():
                test_mu, test_logvar = vae.encode(x_test)
                test_reconstruction = vae.decode(test_mu)
                test_mse = float(F.mse_loss(test_reconstruction, x_test))
                test_kl = float(kl_standard_normal(test_mu, test_logvar).mean())

            print(f"test reconstruction MSE={test_mse:.4f}, mean KL={test_kl:.4f}")
            assert math.isfinite(test_mse) and test_mse < 0.20
            assert math.isfinite(test_kl) and test_kl > 0.0

            history_array = np.asarray(history)
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            axes[0].plot(history_array[:, 0], label="negative ELBO")
            axes[0].plot(history_array[:, 1], alpha=0.7, label="reconstruction")
            axes[0].set(title="SGVB optimization", xlabel="step")
            axes[0].legend()
            test_mu_cpu = test_mu.detach().cpu()
            test_reconstruction_cpu = test_reconstruction.detach().cpu()
            x_test_cpu, y_test_cpu = x_test.detach().cpu(), y_test.detach().cpu()
            axes[1].scatter(test_mu_cpu[:, 0], test_mu_cpu[:, 1], c=y_test_cpu, cmap="tab10", s=16)
            axes[1].set(title="encoder means q(z|x)", xlabel="z1", ylabel="z2")
            axes[2].scatter(x_test_cpu[:, 0], x_test_cpu[:, 1], c="lightgray", s=18, label="input")
            axes[2].scatter(
                test_reconstruction_cpu[:, 0], test_reconstruction_cpu[:, 1],
                c=y_test_cpu, cmap="tab10", s=13, label="reconstruction"
            )
            axes[2].set(title="deterministic reconstruction", xlabel="x1", ylabel="x2")
            axes[2].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            """
            ## 7. 해석 TODO

            1. `beta`를 0으로 만들면 reconstruction과 latent geometry가 각각 어떻게 바뀔까요?
            2. `eps=0`인 평가가 무작위 표본 평가보다 안정적인 이유를 한 문장으로 쓰세요.
            3. 이 실험이 논문 Eq. (8)의 미니배치 추정과 다른 점을 적으세요.
            """,
            """
            ## 7. 해석 예시 답안

            1. `beta=0`이면 모델은 사전분포 정규화를 무시해 복원은 더 쉬워질 수 있습니다.
               하지만 잠재점은 표준정규와 무관하게 흩어져 prior sampling의 의미가 약해집니다.
            2. `eps=0`은 encoder 분포의 평균을 쓰므로 Monte Carlo 잡음 없이 모델 자체의 복원 품질을 비교합니다.
            3. 논문은 데이터 미니배치를 매번 표본화하지만, 이 축소판은 384개 전체를 한
               미니배치로 사용합니다. 각 step의 latent Monte Carlo 표본과 재매개변수화
               경로는 같습니다.
            """,
            "reflection",
        ),
    ),
)


GAN = PaperSpec(
    number=13,
    slug="gan",
    short_title="GAN",
    paper_title="Generative Adversarial Nets",
    authors="Ian J. Goodfellow et al.",
    year=2014,
    primary_url=(
        "https://papers.nips.cc/paper_files/paper/2014/hash/"
        "f033ed80deb0234979a61f95710dbe25-Abstract.html"
    ),
    venue="NeurIPS",
    difficulty="중급",
    expected_minutes=55,
    prerequisites="PyTorch 기본, binary cross entropy, 확률분포, 역전파",
    reproduction_goal=(
        "2차원 Gaussian 데이터에 대해 MLP 생성자와 판별자를 번갈아 갱신하고, 논문이 제안한 "
        "non-saturating 생성자 목적함수로 고정 noise 표본의 분포 오차가 줄어드는지 확인한다."
    ),
    original_scale=(
        "논문은 MNIST, TFD, CIFAR-10에서 MLP GAN을 학습한다. 여기서는 2차원 점 1,024개와 "
        "작은 MLP를 사용해 Algorithm 1의 경쟁적 업데이트만 수 초 내에 재현한다."
    ),
    mappings=(
        (
            "§3, Eq. (1)",
            "`discriminator_loss`와 `generator_loss`의 두 플레이어 목적함수",
            "실제·가짜 logit별 BCE와 양쪽 parameter gradient를 검사",
        ),
        (
            "§3, Algorithm 1 및 Eq. (1) 직후 문단",
            "`gan_step`의 판별자 1회 갱신 뒤 생성자 1회 갱신",
            "두 optimizer의 gradient 경로가 분리되고 loss가 유한한지 검사",
        ),
        (
            "§4.1, Proposition 1 Eq. (2), Theorem 1 및 Eq. (6)",
            "`rbf_mmd`와 판별 확률로 생성 분포 수렴을 진단",
            "학습 전후 MMD와 판별자 평균 출력을 비교",
        ),
    ),
    cells=(
        markdown(
            r"""
            ## 1. 논문의 게임을 2차원으로 축소하기

            논문 §3 Eq. (1)은

            $$\min_G\max_D\;\mathbb E_{x\sim p_{data}}\log D(x)
            +\mathbb E_{z\sim p_z}\log(1-D(G(z)))$$

            를 정의합니다. 여기서는 타원형 2차원 Gaussian을 실제 분포로 두고 두 MLP를
            번갈아 갱신합니다. 생성자에는 같은 절에서 권한 `maximize log D(G(z))`, 즉
            non-saturating loss를 사용합니다. 이 변형은 고정점은 같고 초기 gradient가 더 강합니다.

            **성공 기준:** 고정 noise에서 생성 분포와 실제 분포 사이 RBF-MMD가 초기보다 작아야 합니다.
            """,
            "paper-map",
        ),
        shared_code(
            """
            import math
            import numpy as np
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            import torch.nn.functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(130)
            np.random.seed(130)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device
            # This tiny teaching batch may be faster with AI_LAB_DEVICE=cpu;
            # auto acceleration preserves the same code path for larger variants.

            real_mean = torch.tensor([1.4, -0.8])
            real_transform = torch.tensor([[0.75, 0.0], [0.28, 0.42]])

            def sample_real(n, generator):
                base = torch.randn(n, 2, generator=generator)
                return (real_mean + base @ real_transform.T).to(DEVICE)

            def sample_noise(n, generator):
                return torch.randn(n, 2, generator=generator).to(DEVICE)

            data_rng = torch.Generator().manual_seed(131)
            real_data = sample_real(1024, data_rng)
            print(ACCELERATOR.summary())
            print(f"real mean={real_data.mean(0).tolist()}, shape={tuple(real_data.shape)}")
            assert real_data.shape == (1024, 2)
            """,
            "setup",
        ),
        code(
            """
            class Generator(nn.Module):
                def __init__(self):
                    super().__init__()
                    # TODO: 2 -> 32 -> 32 -> 2 MLP를 정의하세요.

                def forward(self, z):
                    raise NotImplementedError

            class Discriminator(nn.Module):
                def __init__(self):
                    super().__init__()
                    # TODO: 2 -> 32 -> 32 -> 1 logit MLP를 정의하세요.

                def forward(self, x):
                    raise NotImplementedError

            def discriminator_loss(real_logits, fake_logits):
                raise NotImplementedError("Eq. (1)의 두 BCE 항을 구현하세요")

            def generator_loss(fake_logits):
                raise NotImplementedError("-log D(G(z))를 구현하세요")

            G, D = ACCELERATOR.move(Generator(), Discriminator())
            assert G(torch.zeros(5, 2, device=DEVICE)).shape == (5, 2)
            assert D(torch.zeros(5, 2, device=DEVICE)).shape == (5,)
            """,
            """
            class Generator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(2, 32), nn.LeakyReLU(0.2),
                        nn.Linear(32, 32), nn.LeakyReLU(0.2),
                        nn.Linear(32, 2),
                    )

                def forward(self, z):
                    return self.net(z)

            class Discriminator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(2, 32), nn.LeakyReLU(0.2),
                        nn.Linear(32, 32), nn.LeakyReLU(0.2),
                        nn.Linear(32, 1),
                    )

                def forward(self, x):
                    return self.net(x).squeeze(1)

            def discriminator_loss(real_logits, fake_logits):
                real_term = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                )
                fake_term = F.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
                return real_term + fake_term

            def generator_loss(fake_logits):
                # §3: minimax와 고정점이 같지만 초기 포화를 줄이는 대안.
                return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

            G, D = ACCELERATOR.move(Generator(), Discriminator())
            assert G(torch.zeros(5, 2, device=DEVICE)).shape == (5, 2)
            assert D(torch.zeros(5, 2, device=DEVICE)).shape == (5,)
            zero_logits = torch.zeros(2, device=DEVICE)
            assert float(discriminator_loss(zero_logits, zero_logits)) > 0.0
            """,
            "implementation",
        ),
        code(
            """
            def gan_step(G, D, g_optimizer, d_optimizer, batch_size, generator):
                # TODO 1: fake를 detach하고 D의 실제/가짜 BCE update를 수행하세요.
                # TODO 2: 새 noise를 뽑아 non-saturating G update를 수행하세요.
                raise NotImplementedError
            """,
            """
            def gan_step(G, D, g_optimizer, d_optimizer, batch_size, generator):
                # Algorithm 1: k=1 discriminator update.
                real_batch = sample_real(batch_size, generator)
                noise = sample_noise(batch_size, generator)
                fake_batch = G(noise)
                d_optimizer.zero_grad()
                d_loss = discriminator_loss(D(real_batch), D(fake_batch.detach()))
                d_loss.backward()
                d_optimizer.step()

                # §3의 non-saturating 대안으로 generator update.
                noise = sample_noise(batch_size, generator)
                g_optimizer.zero_grad()
                for parameter in D.parameters():
                    parameter.requires_grad_(False)
                g_loss = generator_loss(D(G(noise)))
                g_loss.backward()
                g_optimizer.step()
                for parameter in D.parameters():
                    parameter.requires_grad_(True)
                return float(d_loss.detach()), float(g_loss.detach())

            # detach가 없으면 이 검사 시 D update가 G에 불필요한 gradient를 남길 수 있다.
            """,
            "implementation",
        ),
        code(
            """
            # TODO: 고정 probe noise의 초기 출력을 저장하고 600번 교대 update를 수행하세요.
            train_rng = torch.Generator().manual_seed(132)
            probe_rng = torch.Generator().manual_seed(133)
            probe_noise = sample_noise(384, probe_rng)
            initial_fake = None
            history = []
            raise NotImplementedError
            """,
            """
            train_rng = torch.Generator().manual_seed(132)
            probe_rng = torch.Generator().manual_seed(133)
            probe_noise = sample_noise(384, probe_rng)
            with torch.no_grad():
                initial_fake = G(probe_noise).clone()

            g_optimizer = torch.optim.Adam(G.parameters(), lr=1.5e-3, betas=(0.5, 0.9))
            d_optimizer = torch.optim.Adam(D.parameters(), lr=1.0e-3, betas=(0.5, 0.9))
            history = []
            for step in range(600):
                history.append(gan_step(G, D, g_optimizer, d_optimizer, 128, train_rng))

            assert len(history) == 600
            assert np.isfinite(np.asarray(history)).all()
            print(f"final D loss={history[-1][0]:.3f}, G loss={history[-1][1]:.3f}")
            """,
            "training",
        ),
        code(
            """
            # TODO: RBF-MMD를 구현해 initial/final 분포 오차를 비교하고 산점도와 loss를 그리세요.
            def rbf_mmd(x, y, sigma=1.0):
                raise NotImplementedError

            raise NotImplementedError("평가 metric과 plot을 완성하세요")
            """,
            """
            def rbf_mmd(x, y, sigma=1.0):
                def kernel(a, b):
                    squared_distance = torch.cdist(a, b).pow(2)
                    return torch.exp(-squared_distance / (2.0 * sigma ** 2))
                return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()

            eval_rng = torch.Generator().manual_seed(134)
            real_eval = sample_real(len(probe_noise), eval_rng)
            with torch.no_grad():
                final_fake = G(probe_noise)
                initial_mmd = float(rbf_mmd(initial_fake, real_eval))
                final_mmd = float(rbf_mmd(final_fake, real_eval))
                mean_error = float((final_fake.mean(0) - real_eval.mean(0)).norm())
                d_real = float(torch.sigmoid(D(real_eval)).mean())
                d_fake = float(torch.sigmoid(D(final_fake)).mean())

            print(f"RBF-MMD: {initial_mmd:.4f} -> {final_mmd:.4f}")
            print(f"mean error={mean_error:.3f}, D(real)={d_real:.3f}, D(fake)={d_fake:.3f}")
            assert math.isfinite(final_mmd) and final_mmd < initial_mmd
            assert mean_error < 0.45

            losses = np.asarray(history)
            real_eval_cpu = real_eval.detach().cpu()
            initial_fake_cpu = initial_fake.detach().cpu()
            final_fake_cpu = final_fake.detach().cpu()
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            axes[0].scatter(
                real_eval_cpu[:, 0],
                real_eval_cpu[:, 1],
                s=10,
                alpha=0.55,
                label="real",
            )
            axes[0].scatter(
                initial_fake_cpu[:, 0],
                initial_fake_cpu[:, 1],
                s=10,
                alpha=0.55,
                label="initial G",
            )
            axes[0].set_title("before training")
            axes[0].legend()
            axes[1].scatter(
                real_eval_cpu[:, 0],
                real_eval_cpu[:, 1],
                s=10,
                alpha=0.55,
                label="real",
            )
            axes[1].scatter(
                final_fake_cpu[:, 0],
                final_fake_cpu[:, 1],
                s=10,
                alpha=0.55,
                label="final G",
            )
            axes[1].set_title("after adversarial training")
            axes[1].legend()
            axes[2].plot(losses[:, 0], alpha=0.75, label="D loss")
            axes[2].plot(losses[:, 1], alpha=0.75, label="G loss")
            axes[2].set(title="Algorithm 1 updates", xlabel="step")
            axes[2].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            """
            ## 7. 해석 TODO

            1. GAN loss가 단조감소하지 않아도 학습이 실패했다고 단정할 수 없는 이유를 쓰세요.
            2. `fake_batch.detach()`가 막는 gradient 경로를 설명하세요.
            3. Proposition 1의 이상적 균형에서 `D(real)`과 `D(fake)`는 어디로 가야 할까요?
            """,
            """
            ## 7. 해석 예시 답안

            1. 두 목적함수가 함께 변하는 게임이므로 한 모델이 좋아지면 상대 모델의 loss가
               다시 커질 수 있습니다. 따라서 표본과 분포 metric도 함께 봐야 합니다.
            2. 판별자 단계에서 D의 loss가 G까지 역전파되어 G를 갱신하는 경로를 끊습니다. G는 별도의 생성자 단계에서만 갱신됩니다.
            3. $p_g=p_{data}$이면 최적 판별자 $D^*(x)=1/2$이므로 두 평균 모두 0.5에
               가까워집니다. 유한 MLP·유한 step에서는 정확히 0.5일 필요는 없습니다.
            """,
            "reflection",
        ),
    ),
)


DQN = PaperSpec(
    number=14,
    slug="dqn",
    short_title="DQN",
    paper_title="Playing Atari with Deep Reinforcement Learning",
    authors="Volodymyr Mnih et al.",
    year=2013,
    primary_url="https://arxiv.org/abs/1312.5602",
    venue="arXiv preprint",
    difficulty="중급",
    expected_minutes=65,
    prerequisites="PyTorch 기본, 강화학습의 state/action/reward, Bellman equation",
    reproduction_goal=(
        "외부 Gym 없이 5-state LineWorld에서 replay memory, epsilon-greedy 행동, 고정된 "
        "Bellman target과 Q-network MSE update를 구현해 greedy 정책이 오른쪽 보상에 도달하게 한다."
    ),
    original_scale=(
        "논문은 Atari 화면 4장을 convolutional Q-network에 넣어 수백만 frame을 학습한다. "
        "여기서는 one-hot 상태 5개와 2층 MLP로 Algorithm 1의 학습 역학만 재현한다."
    ),
    mappings=(
        (
            "§2 Background, Eq. (1)",
            "`bellman_targets`의 terminal-aware TD target",
            "done 상태에는 bootstrap 값이 더해지지 않는지 검사",
        ),
        (
            "§2, Eq. (2)–(3)",
            "`QNetwork`가 선택한 action의 Q와 고정 target 사이 오차를 최소화",
            "Q-value shape, finite loss와 gradient를 검사",
        ),
        (
            "§4 Deep Reinforcement Learning, Algorithm 1",
            "`ReplayBuffer`와 `epsilon_greedy`의 수집·표본화·탐험 경로",
            "buffer batch 계약과 학습 뒤 greedy return을 검사",
        ),
    ),
    cells=(
        markdown(
            r"""
            ## 1. Atari DQN을 LineWorld로 축소하기

            논문 §2 Eq. (1)의 Bellman 관계에서 한 step target은

            $$y=r+\gamma(1-\text{done})\max_{a'}Q(s',a')$$

            입니다. §2 Eq. (2)는 $(y-Q(s,a;\theta))^2$을 최소화하고, §4 Algorithm 1은
            상관된 연속 경험 대신 replay memory에서 무작위 전이를 뽑습니다. 여기서는 상태 0과
            4가 terminal인 다섯 칸 환경을 사용합니다. 4에 도착하면 +1, 0이면 -1입니다.

            **성공 기준:** 학습 뒤 greedy agent의 평균 return이 0.8보다 크고, 시작 상태에서
            `Q(right) > Q(left)`여야 합니다.
            """,
            "paper-map",
        ),
        shared_code(
            """
            from collections import deque
            import math
            import numpy as np
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            import torch.nn.functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(140)
            np.random.seed(140)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device
            # Python environment steps stay on CPU; AI_LAB_DEVICE=cpu can avoid
            # accelerator overhead for this very small neural update.

            class LineWorld:
                n_states = 5
                n_actions = 2  # 0=left, 1=right

                def reset(self):
                    self.state = 2
                    self.steps = 0
                    return self.state

                def step(self, action):
                    self.steps += 1
                    self.state = int(np.clip(self.state + (-1 if action == 0 else 1), 0, 4))
                    done = self.state in (0, 4) or self.steps >= 8
                    reward = 1.0 if self.state == 4 else (-1.0 if self.state == 0 else -0.02)
                    return self.state, reward, done

            def encode_states(states):
                states = torch.as_tensor(states, dtype=torch.long)
                return F.one_hot(states, num_classes=LineWorld.n_states).float().to(DEVICE)

            env = LineWorld()
            print(ACCELERATOR.summary())
            print("LineWorld and replay storage stay on CPU; neural batches use DEVICE.")
            assert env.reset() == 2
            assert encode_states([0, 2, 4]).shape == (3, 5)
            """,
            "setup",
        ),
        code(
            """
            class ReplayBuffer:
                def __init__(self, capacity):
                    self.capacity = capacity
                    self.storage = []
                    self.position = 0

                def __len__(self):
                    return len(self.storage)

                def push(self, state, action, reward, next_state, done):
                    # TODO: 고정 용량 원형 buffer를 구현하세요.
                    raise NotImplementedError

                def sample(self, batch_size, rng):
                    # TODO: replacement 없이 index를 뽑아 5개의 tensor를 반환하세요.
                    raise NotImplementedError

            probe_buffer = ReplayBuffer(3)
            for i in range(5):
                probe_buffer.push(i % 4, i % 2, float(i), (i + 1) % 5, False)
            assert len(probe_buffer) == 3
            """,
            """
            class ReplayBuffer:
                def __init__(self, capacity):
                    self.capacity = capacity
                    self.storage = []
                    self.position = 0

                def __len__(self):
                    return len(self.storage)

                def push(self, state, action, reward, next_state, done):
                    transition = (state, action, reward, next_state, done)
                    if len(self.storage) < self.capacity:
                        self.storage.append(transition)
                    else:
                        self.storage[self.position] = transition
                    self.position = (self.position + 1) % self.capacity

                def sample(self, batch_size, rng):
                    indices = rng.choice(len(self.storage), size=batch_size, replace=False)
                    batch = [self.storage[int(index)] for index in indices]
                    states, actions, rewards, next_states, dones = zip(*batch)
                    return (
                        torch.tensor(states, dtype=torch.long),
                        torch.tensor(actions, dtype=torch.long),
                        torch.tensor(rewards, dtype=torch.float32),
                        torch.tensor(next_states, dtype=torch.long),
                        torch.tensor(dones, dtype=torch.float32),
                    )

            probe_buffer = ReplayBuffer(3)
            for i in range(5):
                probe_buffer.push(i % 4, i % 2, float(i), (i + 1) % 5, False)
            assert len(probe_buffer) == 3
            probe_batch = probe_buffer.sample(2, np.random.default_rng(141))
            assert all(tensor.shape == (2,) for tensor in probe_batch)
            """,
            "implementation",
        ),
        code(
            """
            class QNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    # TODO: 5 -> 24 -> 2 action-value MLP를 정의하세요.

                def forward(self, encoded_state):
                    raise NotImplementedError

            def bellman_targets(rewards, dones, next_q_values, gamma=0.95):
                # TODO: terminal mask와 max를 사용하고 target graph를 detach하세요.
                raise NotImplementedError

            def hard_sync_target(online_network, target_network):
                # TODO: online parameter를 target으로 복사하고 freeze하세요.
                raise NotImplementedError

            def parameter_distance(first_network, second_network):
                # TODO: 두 network 사이 parameter L2 distance를 계산하세요.
                raise NotImplementedError

            q_network = QNetwork().to(DEVICE)
            target_network = QNetwork().to(DEVICE)
            hard_sync_target(q_network, target_network)
            probe_next_q = torch.tensor([[2.0, 3.0], [8.0, 9.0]], requires_grad=True)
            probe_target = bellman_targets(
                torch.tensor([1.0, -1.0]),
                torch.tensor([0.0, 1.0]),
                probe_next_q,
            )
            assert torch.allclose(probe_target, torch.tensor([3.85, -1.0]))
            assert not probe_target.requires_grad
            """,
            """
            class QNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(LineWorld.n_states, 24), nn.ReLU(),
                        nn.Linear(24, LineWorld.n_actions),
                    )

                def forward(self, encoded_state):
                    return self.net(encoded_state)

            def bellman_targets(rewards, dones, next_q_values, gamma=0.95):
                # Eq. (2)의 theta_(i-1) 역할: target을 gradient graph에서 분리한다.
                best_next = next_q_values.max(dim=1).values.detach()
                return rewards + gamma * (1.0 - dones) * best_next

            def hard_sync_target(online_network, target_network):
                target_network.load_state_dict(online_network.state_dict())
                target_network.requires_grad_(False)
                target_network.eval()


            def parameter_distance(first_network, second_network):
                squared_distance = 0.0
                parameter_pairs = zip(
                    first_network.parameters(),
                    second_network.parameters(),
                    strict=True,
                )
                for first_parameter, second_parameter in parameter_pairs:
                    difference = first_parameter.detach() - second_parameter.detach()
                    squared_distance = squared_distance + difference.square().sum()
                return float(torch.sqrt(squared_distance).cpu())


            q_network = QNetwork().to(DEVICE)
            target_network = QNetwork().to(DEVICE)
            hard_sync_target(q_network, target_network)
            probe_next_q = torch.tensor([[2.0, 3.0], [8.0, 9.0]], requires_grad=True)
            probe_target = bellman_targets(
                torch.tensor([1.0, -1.0]),
                torch.tensor([0.0, 1.0]),
                probe_next_q,
            )
            assert torch.allclose(probe_target, torch.tensor([3.85, -1.0]))
            assert not probe_target.requires_grad
            assert q_network(encode_states([0, 1])).shape == (2, 2)
            assert q_network is not target_network
            assert all(
                not parameter.requires_grad
                for parameter in target_network.parameters()
            )
            assert parameter_distance(q_network, target_network) == 0.0
            """,
            "implementation",
        ),
        code(
            """
            def epsilon_greedy(network, state, epsilon, rng):
                # TODO: epsilon 확률은 무작위 action, 아니면 argmax Q를 반환하세요.
                raise NotImplementedError

            # TODO: Algorithm 1을 180 episode 실행하세요. replay가 32개 이상이면 MSE update합니다.
            replay = ReplayBuffer(capacity=400)
            optimizer = torch.optim.Adam(q_network.parameters(), lr=8e-3)
            rng = np.random.default_rng(142)
            returns, losses = [], []
            raise NotImplementedError
            """,
            """
            def epsilon_greedy(network, state, epsilon, rng):
                if rng.random() < epsilon:
                    return int(rng.integers(LineWorld.n_actions))
                with torch.no_grad():
                    return int(network(encode_states([state])).argmax(dim=1).item())

            replay = ReplayBuffer(capacity=400)
            optimizer = torch.optim.Adam(q_network.parameters(), lr=8e-3)
            rng = np.random.default_rng(142)
            returns, losses = [], []
            sync_interval = 40
            update_count = 0
            pre_sync_distances = []
            post_sync_distances = []

            for episode in range(180):
                state = env.reset()
                episode_return = 0.0
                epsilon = max(0.05, 0.85 * (1.0 - episode / 150.0))
                done = False
                while not done:
                    action = epsilon_greedy(q_network, state, epsilon, rng)
                    next_state, reward, done = env.step(action)
                    replay.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if len(replay) >= 32:
                        states, actions, rewards, next_states, dones = replay.sample(32, rng)
                        states, actions, rewards, next_states, dones = ACCELERATOR.move(
                            states, actions, rewards, next_states, dones
                        )
                        predicted_q = q_network(encode_states(states))
                        predicted_q = predicted_q.gather(
                            1,
                            actions[:, None],
                        ).squeeze(1)
                        with torch.no_grad():
                            next_q = target_network(encode_states(next_states))
                            targets = bellman_targets(rewards, dones, next_q)
                        loss = F.mse_loss(predicted_q, targets)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        update_count += 1

                        if update_count % sync_interval == 0:
                            distance_before = parameter_distance(
                                q_network,
                                target_network,
                            )
                            pre_sync_distances.append(distance_before)
                            assert distance_before > 0.0
                            hard_sync_target(q_network, target_network)
                            distance_after = parameter_distance(
                                q_network,
                                target_network,
                            )
                            post_sync_distances.append(distance_after)
                            assert distance_after == 0.0

                returns.append(episode_return)

            print(f"episodes={len(returns)}, replay={len(replay)}, updates={len(losses)}")
            assert len(returns) == 180 and len(losses) > 100
            assert np.isfinite(losses).all()
            assert pre_sync_distances and min(pre_sync_distances) > 0.0
            assert post_sync_distances and max(post_sync_distances) == 0.0
            assert all(
                parameter.grad is None
                for parameter in target_network.parameters()
            )
            """,
            "training",
        ),
        code(
            """
            # TODO: epsilon=0 정책을 40회 평가하고 state별 Q와 학습곡선을 시각화하세요.
            raise NotImplementedError
            """,
            """
            evaluation_returns = []
            eval_rng = np.random.default_rng(143)
            for _ in range(40):
                state = env.reset()
                total = 0.0
                done = False
                while not done:
                    action = epsilon_greedy(q_network, state, 0.0, eval_rng)
                    state, reward, done = env.step(action)
                    total += reward
                evaluation_returns.append(total)

            with torch.no_grad():
                all_states = encode_states(torch.arange(LineWorld.n_states))
                all_q = q_network(all_states).detach().cpu().numpy()
            mean_return = float(np.mean(evaluation_returns))
            print(f"greedy mean return={mean_return:.3f}")
            print("Q(start=2) [left, right] =", np.round(all_q[2], 3).tolist())
            assert mean_return > 0.80
            assert all_q[2, 1] > all_q[2, 0]

            moving_return = np.convolve(returns, np.ones(15) / 15, mode="valid")
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            axes[0].plot(returns, alpha=0.3, label="episode return")
            axes[0].plot(np.arange(14, len(returns)), moving_return, label="15-episode mean")
            axes[0].set(title="epsilon-greedy learning", xlabel="episode")
            axes[0].legend()
            axes[1].plot(losses)
            axes[1].set(title="replay minibatch MSE", xlabel="update")
            states_axis = np.arange(LineWorld.n_states)
            axes[2].plot(states_axis, all_q[:, 0], "o-", label="left")
            axes[2].plot(states_axis, all_q[:, 1], "o-", label="right")
            axes[2].set(title="learned action values", xlabel="state", ylabel="Q")
            axes[2].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            """
            ## 7. 해석 TODO

            1. replay에서 연속 전이가 아니라 무작위 전이를 뽑는 논문의 이유 두 가지를 쓰세요.
            2. TD target을 `detach`하지 않으면 Eq. (2)와 어떤 점이 달라지나요?
            3. Atari 원 재현과 이 축소판 사이 가장 큰 representation 차이를 적으세요.
            """,
            """
            ## 7. 해석 예시 답안

            1. 시간적으로 강하게 상관된 표본을 섞어 update 분산을 낮추고, 한 경험을 여러 번 재사용해 데이터 효율을 높입니다.
            2. Eq. (2)는 이전/고정 Q가 만든 target을 상수로 보고 현재 Q만 회귀합니다.
               detach하지 않으면 target 쪽 Q도 동시에 움직여 다른 목적함수가 됩니다.
            3. 원 논문은 최근 4개 Atari frame을 convolution으로 표현하지만, 여기서는
               Markov 상태를 이미 알고 one-hot 벡터로 제공합니다. replay와 Bellman
               update는 같지만 시각표현 학습은 생략했습니다.
            """,
            "reflection",
        ),
    ),
)


GCN = PaperSpec(
    number=15,
    slug="gcn",
    short_title="GCN",
    paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
    authors="Thomas N. Kipf and Max Welling",
    year=2017,
    primary_url="https://arxiv.org/abs/1609.02907",
    venue="ICLR",
    difficulty="중급",
    expected_minutes=55,
    prerequisites="PyTorch 기본, 행렬곱, 그래프 adjacency/degree, cross entropy",
    reproduction_goal=(
        "두 community의 작은 합성 그래프에서 self-loop와 대칭 degree normalization을 직접 구현하고, "
        "노드 네 개의 label만으로 2-layer GCN이 나머지 노드를 분류하는지 확인한다."
    ),
    original_scale=(
        "논문은 Cora, Citeseer, Pubmed 등의 citation graph에서 sparse 연산을 사용한다. 여기서는 "
        "24-node dense adjacency를 사용하지만 Eq. (2), Eq. (8)–(10)의 전파와 masked loss는 같다."
    ),
    mappings=(
        (
            "§2, Eq. (2)",
            "`GraphConvolution.forward`의 `A_hat @ X @ W` 전파",
            "node 수를 보존하는 layer 출력 shape와 finite gradient를 검사",
        ),
        (
            "§2.2, Eq. (7)–(8)",
            "`normalize_adjacency`의 self-loop와 대칭 degree normalization",
            "정규화 행렬의 shape·대칭성·유한값을 검사",
        ),
        (
            "§3.1, Eq. (9)–(10)",
            "`TinyGCN`의 2-layer propagation과 train mask cross entropy",
            "미공개 test node 정확도와 학습 loss 감소를 검사",
        ),
    ),
    cells=(
        markdown(
            r"""
            ## 1. Citation graph를 두 community로 축소하기

            논문 §2 Eq. (2)의 층은

            $$H^{(l+1)}=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)})$$

            입니다. §2.2 Eq. (8)의 `renormalization trick`은 $\tilde A=A+I$를 사용합니다.
            여기서는 두 community를 가진 24-node graph와 약한 noisy feature를 만들고, 각 class에서
            단 두 노드만 label로 공개합니다. 그래프 이웃 집계가 약한 feature 신호를 매끄럽게 만듭니다.

            **성공 기준:** 공개하지 않은 20개 노드의 정확도가 85%보다 커야 합니다.
            """,
            "paper-map",
        ),
        shared_code(
            """
            import math
            import numpy as np
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            import torch.nn.functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(150)
            np.random.seed(150)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device
            # This 24-node dense example may be faster with AI_LAB_DEVICE=cpu;
            # auto acceleration preserves the same code path for larger graphs.

            n_per_class = 12
            n_nodes = 2 * n_per_class
            labels = torch.tensor([0] * n_per_class + [1] * n_per_class, dtype=torch.long)
            adjacency = torch.zeros(n_nodes, n_nodes)

            # 각 community 안에서 ring의 1-hop/2-hop 이웃을 연결한다.
            for offset in (0, n_per_class):
                for i in range(n_per_class):
                    for delta in (1, 2):
                        j = offset + (i + delta) % n_per_class
                        k = offset + i
                        adjacency[k, j] = adjacency[j, k] = 1.0
            # 소수의 community 간 edge는 실제 그래프의 noise 역할을 한다.
            for left, right in ((2, 14), (8, 20)):
                adjacency[left, right] = adjacency[right, left] = 1.0

            feature_rng = torch.Generator().manual_seed(151)
            weak_signal = (2 * labels.float() - 1.0)[:, None] * 0.45
            features = torch.cat([
                weak_signal + 0.65 * torch.randn(n_nodes, 1, generator=feature_rng),
                0.75 * torch.randn(n_nodes, 2, generator=feature_rng),
                torch.ones(n_nodes, 1),
            ], dim=1)
            train_idx = torch.tensor([0, 5, 12, 17])
            test_mask = torch.ones(n_nodes, dtype=torch.bool)
            test_mask[train_idx] = False
            labels, adjacency, features, train_idx, test_mask = ACCELERATOR.move(
                labels, adjacency, features, train_idx, test_mask
            )

            assert torch.allclose(adjacency, adjacency.T)
            assert features.shape == (24, 4)
            print(ACCELERATOR.summary())
            edge_count = int(adjacency.sum().item() / 2)
            print(
                f"nodes={n_nodes}, undirected edges={edge_count}, labels used=4"
            )
            """,
            "setup",
        ),
        code(
            """
            def normalize_adjacency(adjacency):
                # TODO: A_tilde=A+I와 D_tilde^(-1/2) A_tilde D_tilde^(-1/2)를 구현하세요.
                raise NotImplementedError

            a_hat = normalize_adjacency(adjacency)
            assert a_hat.shape == adjacency.shape
            assert torch.allclose(a_hat, a_hat.T, atol=1e-6)
            assert torch.isfinite(a_hat).all()
            """,
            """
            def normalize_adjacency(adjacency):
                # §2.2 Eq. (8): self-loop를 먼저 넣은 뒤 degree로 대칭 정규화한다.
                a_tilde = adjacency + torch.eye(
                    adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
                )
                degree = a_tilde.sum(dim=1)
                inv_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)
                return inv_sqrt_degree[:, None] * a_tilde * inv_sqrt_degree[None, :]

            a_hat = normalize_adjacency(adjacency)
            assert a_hat.shape == adjacency.shape
            assert torch.allclose(a_hat, a_hat.T, atol=1e-6)
            assert torch.isfinite(a_hat).all()
            assert torch.all(a_hat.diag() > 0), "self-loop가 모든 node에 있어야 한다"
            print("renormalized adjacency 검증 통과")
            """,
            "implementation",
        ),
        code(
            """
            class GraphConvolution(nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    # TODO: 논문 식의 W만 정의하세요(bias 없음).

                def forward(self, x, normalized_adjacency):
                    # TODO: A_hat @ X @ W를 반환하세요.
                    raise NotImplementedError

            class TinyGCN(nn.Module):
                def __init__(self):
                    super().__init__()
                    # TODO: 4 -> 12 -> 2 두 층을 정의하세요.

                def forward(self, x, normalized_adjacency):
                    raise NotImplementedError

            gcn = TinyGCN().to(DEVICE)
            logits = gcn(features, a_hat)
            assert logits.shape == (n_nodes, 2)
            """,
            """
            class GraphConvolution(nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.weight = nn.Parameter(torch.empty(in_features, out_features))
                    nn.init.xavier_uniform_(self.weight)

                def forward(self, x, normalized_adjacency):
                    return normalized_adjacency @ x @ self.weight

            class TinyGCN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gcn1 = GraphConvolution(4, 12)
                    self.gcn2 = GraphConvolution(12, 2)

                def forward(self, x, normalized_adjacency):
                    hidden = F.relu(self.gcn1(x, normalized_adjacency))
                    return self.gcn2(hidden, normalized_adjacency)

            gcn = TinyGCN().to(DEVICE)
            logits = gcn(features, a_hat)
            assert logits.shape == (n_nodes, 2)
            assert sum(parameter.numel() for parameter in gcn.parameters()) == 72
            """,
            "implementation",
        ),
        code(
            """
            # TODO: Eq. (10)처럼 train_idx의 네 label에 대해서만 cross entropy를 계산해 250회 학습하세요.
            optimizer = torch.optim.Adam(gcn.parameters(), lr=0.03, weight_decay=5e-4)
            loss_history = []
            raise NotImplementedError
            """,
            """
            optimizer = torch.optim.Adam(gcn.parameters(), lr=0.03, weight_decay=5e-4)
            loss_history = []
            with torch.no_grad():
                initial_logits = gcn(features, a_hat)[train_idx]
                initial_loss = float(
                    F.cross_entropy(initial_logits, labels[train_idx])
                )

            for epoch in range(250):
                logits = gcn(features, a_hat)
                # §3.1 Eq. (10): Y_L, 즉 label이 공개된 node만 loss에 들어간다.
                loss = F.cross_entropy(logits[train_idx], labels[train_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(float(loss.detach()))

            print(f"masked CE: {initial_loss:.4f} -> {loss_history[-1]:.4f}")
            assert len(loss_history) == 250
            assert math.isfinite(loss_history[-1]) and loss_history[-1] < initial_loss
            """,
            "training",
        ),
        code(
            """
            # TODO: train/test accuracy를 계산하고 true/predicted graph 및 loss를 그리세요.
            raise NotImplementedError
            """,
            """
            gcn.eval()
            with torch.no_grad():
                final_logits = gcn(features, a_hat)
                predictions = final_logits.argmax(dim=1)
            train_accuracy = float((predictions[train_idx] == labels[train_idx]).float().mean())
            test_accuracy = float((predictions[test_mask] == labels[test_mask]).float().mean())
            print(
                f"labeled accuracy={train_accuracy:.3f}, "
                f"held-out node accuracy={test_accuracy:.3f}"
            )
            assert train_accuracy == 1.0
            assert test_accuracy > 0.85

            angles = np.linspace(0, 2 * np.pi, n_per_class, endpoint=False)
            positions = np.vstack([
                np.c_[np.cos(angles) - 1.35, np.sin(angles)],
                np.c_[np.cos(angles) + 1.35, np.sin(angles)],
            ])
            train_idx_cpu = train_idx.detach().cpu().numpy()

            def draw_graph(axis, colors, title):
                edge_i, edge_j = torch.where(torch.triu(adjacency, diagonal=1) > 0)
                for i, j in zip(edge_i.tolist(), edge_j.tolist()):
                    axis.plot(
                        positions[[i, j], 0],
                        positions[[i, j], 1],
                        color="0.82",
                        lw=0.8,
                        zorder=0,
                    )
                axis.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    c=colors,
                    cmap="coolwarm",
                    s=55,
                    zorder=2,
                )
                axis.scatter(
                    positions[train_idx_cpu, 0],
                    positions[train_idx_cpu, 1],
                    facecolors="none",
                    edgecolors="black",
                    marker="s",
                    s=120,
                    linewidths=1.4,
                    label="labeled",
                )
                axis.set_title(title)
                axis.set_aspect("equal")
                axis.axis("off")
                axis.legend(loc="upper center")

            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            draw_graph(axes[0], labels.detach().cpu().numpy(), "ground truth (4 labels exposed)")
            draw_graph(axes[1], predictions.detach().cpu().numpy(), "GCN predictions")
            axes[2].plot(loss_history)
            axes[2].set(title="masked cross entropy", xlabel="epoch", ylabel="loss")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            """
            ## 7. 해석 TODO

            1. self-loop를 제거하면 한 층에서 자기 feature는 어떻게 되나요?
            2. loss에 `test_mask`를 포함하면 왜 semi-supervised 실험이 아니게 되나요?
            3. 이 dense 구현과 논문의 citation graph 구현 사이 계산 복잡도 차이를 쓰세요.
            """,
            """
            ## 7. 해석 예시 답안

            1. 해당 층의 출력은 이웃 feature만 집계하고 자기 feature의 직접 경로를 잃습니다. $A+I$는 이를 보존합니다.
            2. 평가할 노드의 정답까지 gradient에 사용하므로 label이 일부만 주어진 전이적
               node classification 조건과 test 분리가 깨집니다.
            3. 여기서는 $24\times24$ dense 행렬곱이라 일반적으로 $O(N^2)$ 저장/계산을
               합니다. 논문은 sparse adjacency를 사용해 한 층의 전파를 edge 수에
               선형인 $O(|E|)$로 수행합니다.
            """,
            "reflection",
        ),
    ),
)


SPECS = (VAE, GAN, DQN, GCN)
