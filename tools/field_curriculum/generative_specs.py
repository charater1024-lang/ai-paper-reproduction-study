"""Ten portfolio-grade reproductions of landmark generative-model papers.

Every notebook produced from these specs reads the prepared local NPZ file.  The
small models are deliberately not benchmark replicas; they isolate the paper's
central equation or algorithm and include deterministic executable checks.
"""

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

FIELD_ID = "generative"
FIELD_TITLE = "생성 모델"
DATASET = "data/field_curriculum/generative_samples.npz"


def _setup(seed: int) -> object:
    return shared_code(
        f"""
        from pathlib import Path
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
        # These teaching batches are intentionally tiny; GPU launch/transfer
        # overhead can dominate, so use AI_LAB_DEVICE=cpu for minimum wall time.

        DATA_PATH = Path({DATASET!r})
        assert DATA_PATH.exists(), f"준비된 로컬 데이터가 없습니다: {{DATA_PATH}}"
        raw = np.load(DATA_PATH)
        expected = {{"points", "point_labels", "images", "image_labels", "domain_a", "domain_b"}}
        assert expected.issubset(raw.files), f"NPZ 키 불일치: {{sorted(raw.files)}}"
        points = torch.tensor(raw["points"], dtype=torch.float32)
        point_labels = torch.tensor(raw["point_labels"], dtype=torch.long)
        images = torch.tensor(raw["images"], dtype=torch.float32)
        image_labels = torch.tensor(raw["image_labels"], dtype=torch.long)
        domain_a = torch.tensor(raw["domain_a"], dtype=torch.float32)
        domain_b = torch.tensor(raw["domain_b"], dtype=torch.float32)
        points, point_labels, images, image_labels, domain_a, domain_b = ACCELERATOR.move(
            points, point_labels, images, image_labels, domain_a, domain_b
        )

        # Seeded random draws stay on CPU so CUDA, MPS, and CPU follow the same
        # sampling stream; complete tensors are then transferred to the lab device.
        def randn_device(size, *, generator):
            return torch.randn(size, generator=generator).to(DEVICE)

        def rand_device(size, *, generator):
            return torch.rand(size, generator=generator).to(DEVICE)

        def randint_device(high, size, *, generator):
            return torch.randint(high, size, generator=generator).to(DEVICE)
        if float(images.min()) < 0.0:
            images_01 = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        else:
            images_01 = images.clamp(0.0, 1.0)
        assert points.shape == (600, 2) and images.shape == (300, 1, 8, 8)
        assert domain_a.shape == domain_b.shape == (600, 2)
        print(ACCELERATOR.summary())
        print("local data:", tuple(points.shape), tuple(images.shape), tuple(domain_a.shape))
        """,
        "setup",
    )


def _intro(body: str) -> object:
    return markdown(body, "paper-map")


SDAE = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=0,
    slug="stacked_denoising_autoencoder",
    short_title="Denoising Autoencoder",
    paper_title=(
        "Stacked Denoising Autoencoders: Learning Useful "
        "Representations in a Deep Network with a Local Denoising "
        "Criterion"
    ),
    authors=(
        "Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua "
        "Bengio, and Pierre-Antoine Manzagol"
    ),
    year=2010,
    primary_url="https://www.jmlr.org/papers/v11/vincent10a.html",
    venue="JMLR",
    difficulty="초급",
    expected_minutes=50,
    dataset_file=DATASET,
    prerequisites="PyTorch 선형층, 평균제곱오차, 확률적 마스킹",
    reproduction_goal="8×8 로컬 영상을 훼손한 뒤 깨끗한 입력을 복원하는 denoising autoencoder를 학습한다.",
    original_scale=(
        "논문은 MNIST와 자연 이미지 패치 등에 여러 층을 쌓아 평가한다. 여기서는 한 층 DAE로 국소 "
        "denoising criterion을 보존한다."
    ),
    mappings=(
        (
            "§2.2, Eq. (4)",
            "autoencoder reconstruction",
            "입력을 잠재 표현으로 인코딩하고 reconstruction loss를 최소화한다.",
        ),
        (
            "§3.1, Denoising Autoencoder Algorithm",
            "corrupt-then-denoise training",
            "x를 q_D로 훼손하되 목표는 원본 x로 둔다.",
        ),
        (
            "§3.5 and Figure 3",
            "원 논문의 layer-wise stacking — 이번 축소 실습에서는 생략",
            "1-layer DAE 이후 확장할 항목이며 완료 증거로 주장하지 않는다.",
        ),
    ),
    cells=(
        _intro(r"""
        ## 핵심 재현

        §3.1의 알고리즘은 깨끗한 $x$에서 $\tilde x\sim q_D(\tilde x\mid x)$를 만든 뒤
        $g(f(\tilde x))$가 **원본** $x$를 복원하도록 학습합니다. 작은 8×8 영상과 masking
        noise로 이 차이를 직접 확인합니다.
        """),
        _setup(301),
        code(
            """
            # TODO 1: 각 픽셀을 확률 p로 0으로 만드는 masking corruption을 구현하세요.
            def corrupt(x, p=0.35, generator=None):
                raise NotImplementedError


            # TODO 2: 64 -> 24 -> 64 sigmoid DAE를 구성하세요.
            class DAE(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError

                def forward(self, x):
                    raise NotImplementedError


            flat = images_01.flatten(1)
            probe = corrupt(flat[:8], generator=torch.Generator().manual_seed(1))
            model = DAE().to(DEVICE)
            assert probe.shape == (8, 64) and model(probe).shape == (8, 64)
            """,
            """
            def corrupt(x, p=0.35, generator=None):
                keep = rand_device(x.shape, generator=generator) >= p
                return x * keep.to(x.dtype)


            class DAE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(nn.Linear(64, 24), nn.ReLU())
                    self.decoder = nn.Sequential(nn.Linear(24, 64), nn.Sigmoid())

                def forward(self, x):
                    return self.decoder(self.encoder(x))


            flat = images_01.flatten(1)
            probe = corrupt(flat[:8], generator=torch.Generator().manual_seed(1))
            model = DAE().to(DEVICE)
            assert probe.shape == (8, 64) and model(probe).shape == (8, 64)
            assert int((probe == 0).sum()) > int((flat[:8] == 0).sum())
            """,
            "implementation",
        ),
        code(
            """
            # TODO: 매 step 새 corruption을 만들고 원본 flat을 목표로 100회 학습하세요.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            history = []
            raise NotImplementedError
            """,
            """
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            history = []
            train_rng = torch.Generator().manual_seed(302)
            with torch.no_grad():
                initial = float(F.mse_loss(model(corrupt(flat, generator=train_rng)), flat))
            for _ in range(100):
                noisy = corrupt(flat, generator=train_rng)
                loss = F.mse_loss(model(noisy), flat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach()))
            assert history[-1] < initial and np.isfinite(history).all()
            print(f"denoising MSE: {initial:.4f} -> {history[-1]:.4f}")
            """,
            "training",
        ),
        code(
            """
            # TODO: 고정 corruption에서 noisy input과 DAE output의 원본 MSE를 비교하고 3개 영상을 그리세요.
            raise NotImplementedError
            """,
            """
            eval_noisy = corrupt(flat, generator=torch.Generator().manual_seed(303))
            with torch.no_grad():
                restored = model(eval_noisy)
            noisy_mse = float(F.mse_loss(eval_noisy, flat))
            restored_mse = float(F.mse_loss(restored, flat))
            print(f"noisy={noisy_mse:.4f}, restored={restored_mse:.4f}")
            assert restored_mse < noisy_mse
            fig, axes = plt.subplots(1, 3, figsize=(7, 2.4))
            for ax, value, title in zip(
                axes,
                (flat[0], eval_noisy[0], restored[0]),
                ("clean", "corrupted", "denoised"),
            ):
                ax.imshow(value.detach().cpu().reshape(8, 8), cmap="gray", vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "왜 noisy input 자체가 아니라 clean x를 target으로 써야 유용한 표현이 생기는지 설명하세요.",
            (
                "동일 입력 복사는 훼손에 불변인 구조를 요구하지 않습니다. clean target은 여러 훼손본을 같은 "
                "원본으로 보내도록 해 데이터 manifold의 안정적인 특징을 학습시킵니다."
            ),
            "reflection",
        ),
    ),
)


VAE = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=1,
    slug="vae",
    short_title="VAE",
    paper_title="Auto-Encoding Variational Bayes",
    authors="Diederik P. Kingma and Max Welling",
    year=2014,
    primary_url="https://arxiv.org/abs/1312.6114",
    venue="ICLR",
    difficulty="중급",
    expected_minutes=55,
    dataset_file=DATASET,
    prerequisites="Gaussian 분포, KL divergence, PyTorch autograd",
    reproduction_goal="2차원 로컬 점에서 reparameterization, 닫힌형 KL, SGVB ELBO를 구현한다.",
    original_scale=(
        "논문은 MNIST/Frey Face에 stochastic variational inference를 적용한다. "
        "이 축소판은 동일한 ELBO를 2차원 mixture에 적용한다."
    ),
    mappings=(
        (
            "§2.2, Eq. (1)–(3)",
            "evidence lower bound",
            "reconstruction 기대값과 posterior-prior KL로 log evidence의 하한을 구성한다.",
        ),
        (
            "§2.3, Eq. (4)–(6), Algorithm 1",
            "SGVB estimator",
            "재매개변수화한 표본으로 encoder까지 역전파한다.",
        ),
        (
            "§2.4 and §3, Eq. (9)–(10)",
            "Gaussian posterior",
            "z=mu+sigma*epsilon 및 diagonal Gaussian KL을 구현한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (3)의 ELBO와 Eq. (9)–(10)의 Gaussian 특수화를 사용합니다. 고정된 로컬 mixture에서
        $z=\mu+\exp(\frac12\log\sigma^2)\epsilon$ 경로가 미분 가능한지 검증합니다."""),
        _setup(311),
        code(
            """
            # TODO 1: Gaussian reparameterization을 구현하세요.
            def reparameterize(mu, logvar, eps=None):
                raise NotImplementedError


            # TODO 2: sample별 KL(q||N(0,I))를 구현하세요.
            def gaussian_kl(mu, logvar):
                raise NotImplementedError


            mu0 = torch.zeros(4, 2, requires_grad=True)
            lv0 = torch.zeros_like(mu0)
            z0 = reparameterize(mu0, lv0, torch.zeros_like(mu0))
            assert torch.allclose(z0, mu0)
            """,
            """
            def reparameterize(mu, logvar, eps=None):
                std = torch.exp(0.5 * logvar)
                return mu + std * (torch.randn_like(std) if eps is None else eps)


            def gaussian_kl(mu, logvar):
                return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)


            mu0 = torch.zeros(4, 2, requires_grad=True)
            lv0 = torch.zeros_like(mu0)
            z0 = reparameterize(mu0, lv0, torch.zeros_like(mu0))
            assert torch.allclose(z0, mu0) and torch.allclose(
                gaussian_kl(mu0, lv0), torch.zeros(4)
            )
            z0.sum().backward()
            assert mu0.grad is not None
            """,
            "implementation",
        ),
        code(
            """
            # TODO: 2-16-(mu,logvar), z-16-2 VAE와 negative ELBO를 구현해 150회 학습하세요.
            raise NotImplementedError
            """,
            """
            class TinyVAE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.enc = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
                    self.mu, self.lv = nn.Linear(16, 2), nn.Linear(16, 2)
                    self.dec = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))

                def encode(self, x):
                    h = self.enc(x)
                    return self.mu(h), self.lv(h).clamp(-6, 6)

                def forward(self, x, deterministic=False):
                    mu, lv = self.encode(x)
                    eps = torch.zeros_like(mu) if deterministic else None
                    return self.dec(reparameterize(mu, lv, eps)), mu, lv


            def vae_loss(model, x, deterministic=False):
                recon, mu, lv = model(x, deterministic)
                rec = (recon - x).square().sum(1)
                return (rec + 0.08 * gaussian_kl(mu, lv)).mean(), rec.mean()


            model = TinyVAE().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
            history = []
            x_train = points[:480]
            with torch.no_grad():
                initial = float(vae_loss(model, x_train, True)[0])
            for _ in range(150):
                loss, _ = vae_loss(model, x_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach()))
            assert history[-1] < initial and np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: posterior mean reconstruction MSE와 평균 KL을 보고 latent/reconstruction을
            # 시각화하세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                recon, mu, lv = model(points[480:], True)
                mse = float(F.mse_loss(recon, points[480:]))
                mean_kl = float(gaussian_kl(mu, lv).mean())
            print(f"reconstruction MSE={mse:.4f}, KL={mean_kl:.4f}")
            assert math.isfinite(mse) and math.isfinite(mean_kl) and mean_kl >= 0
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].plot(history)
            axes[0].set_title("negative ELBO")
            mu_cpu, labels_cpu = mu.detach().cpu(), point_labels[480:].detach().cpu()
            points_cpu, recon_cpu = points[480:].detach().cpu(), recon.detach().cpu()
            axes[1].scatter(mu_cpu[:, 0], mu_cpu[:, 1], c=labels_cpu, s=10)
            axes[1].set_title("q mean")
            axes[2].scatter(points_cpu[:, 0], points_cpu[:, 1], alpha=0.35, label="data")
            axes[2].scatter(recon_cpu[:, 0], recon_cpu[:, 1], s=10, label="recon")
            axes[2].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "왜 eps=0 평가가 stochastic 평가보다 안정적인가요?",
            "posterior mean을 사용하면 Monte Carlo 표본 잡음을 제거해 모델 자체의 평균 reconstruction을 비교할 수 있습니다.",
            "reflection",
        ),
    ),
)


GAN = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=2,
    slug="gan",
    short_title="GAN",
    paper_title="Generative Adversarial Nets",
    authors="Ian J. Goodfellow et al.",
    year=2014,
    primary_url=(
        "https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80d"
        "eb0234979a61f95710dbe25-Abstract.html"
    ),
    venue="NeurIPS",
    difficulty="중급",
    expected_minutes=55,
    dataset_file=DATASET,
    prerequisites="binary cross entropy, alternating optimization, PyTorch",
    reproduction_goal="2차원 점 분포에서 discriminator와 non-saturating generator를 번갈아 학습한다.",
    original_scale="논문은 MNIST, TFD, CIFAR-10을 사용한다. 여기서는 Algorithm 1의 게임만 2차원으로 축소한다.",
    mappings=(
        (
            "§3, Eq. (1)",
            "minimax game",
            "real/fake log-likelihood의 두 플레이어 목적을 구현한다.",
        ),
        ("§3, Algorithm 1", "alternating updates", "D update 뒤 G update를 수행한다."),
        (
            "§3 after Algorithm 1; §4.1 Proposition 1 and Eq. (6)",
            "non-saturating generator and equilibrium",
            "-log D(G(z))와 p_g=p_data에서 D=1/2인 성질을 확인한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (1)의 minimax game을 2차원에 옮기고, §3이 초기 gradient를 위해 권한
        non-saturating $-\log D(G(z))$를 사용합니다."""),
        _setup(321),
        code(
            """
            # TODO: 2->32->2 generator, 2->32->1 discriminator와 BCE losses를 구현하세요.
            raise NotImplementedError
            """,
            """
            G = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 2)).to(
                DEVICE
            )
            D = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1)).to(
                DEVICE
            )


            def d_loss(real_logits, fake_logits):
                return F.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                ) + F.binary_cross_entropy_with_logits(
                    fake_logits, torch.zeros_like(fake_logits)
                )


            def g_loss(fake_logits):
                return F.binary_cross_entropy_with_logits(
                    fake_logits, torch.ones_like(fake_logits)
                )


            assert G(torch.zeros(5, 2, device=DEVICE)).shape == (5, 2) and D(
                torch.zeros(5, 2, device=DEVICE)
            ).shape == (5, 1)
            """,
            "implementation",
        ),
        code(
            """
            # TODO: 고정 seed로 Algorithm 1의 D/G 교대 update를 220회 수행하세요.
            raise NotImplementedError
            """,
            """
            opt_g = torch.optim.Adam(G.parameters(), lr=0.004)
            opt_d = torch.optim.Adam(D.parameters(), lr=0.004)
            rng = torch.Generator().manual_seed(322)
            history = []
            fixed_z = randn_device((256, 2), generator=torch.Generator().manual_seed(323))
            with torch.no_grad():
                initial_fake = G(fixed_z).clone()
            for _ in range(220):
                idx = randint_device(len(points), (96,), generator=rng)
                real = points[idx]
                z = randn_device((96, 2), generator=rng)
                loss_d = d_loss(D(real), D(G(z).detach()))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                z = randn_device((96, 2), generator=rng)
                loss_g = g_loss(D(G(z)))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                history.append((float(loss_d.detach()), float(loss_g.detach())))
            assert np.isfinite(history).all() and not torch.allclose(
                initial_fake, G(fixed_z).detach()
            )
            """,
            "training",
        ),
        code(
            """
            # TODO: real/fake mean distance와 평균 D 확률을 출력하고 분포 및 두 loss를 그리세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                fake = G(fixed_z)
                mean_gap = float((fake.mean(0) - points.mean(0)).norm())
                d_prob = float(torch.sigmoid(D(fake)).mean())
            print(f"mean gap={mean_gap:.3f}, D(fake)={d_prob:.3f}")
            assert math.isfinite(mean_gap) and 0 <= d_prob <= 1
            h = np.asarray(history)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            points_cpu, fake_cpu = points[:300].detach().cpu(), fake.detach().cpu()
            axes[0].scatter(
                points_cpu[:, 0], points_cpu[:, 1], s=8, alpha=0.4, label="real"
            )
            axes[0].scatter(fake_cpu[:, 0], fake_cpu[:, 1], s=8, alpha=0.4, label="G(z)")
            axes[0].legend()
            axes[1].plot(h[:, 0], label="D")
            axes[1].plot(h[:, 1], label="G")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "왜 G update에서 fake를 detach하면 안 되나요?",
            "G loss의 gradient가 D의 계산을 통과해 G 파라미터까지 도달해야 하기 때문입니다. detach는 그 경로를 끊습니다.",
            "reflection",
        ),
    ),
)


DCGAN = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=3,
    slug="dcgan",
    short_title="DCGAN",
    paper_title=(
        "Unsupervised Representation Learning with Deep Convolutional "
        "Generative Adversarial Networks"
    ),
    authors="Alec Radford, Luke Metz, and Soumith Chintala",
    year=2016,
    primary_url="https://arxiv.org/abs/1511.06434",
    venue="ICLR",
    difficulty="중급",
    expected_minutes=60,
    dataset_file=DATASET,
    prerequisites="convolution, batch normalization, GAN loss",
    reproduction_goal="8×8 영상용 작은 DCGAN에서 strided convolution, batch normalization, tanh 설계를 검증한다.",
    original_scale=(
        "논문은 LSUN 등 64×64 이미지와 깊은 convolution을 사용한다. 축소판은 §3의 "
        "architecture constraints를 8×8에 적용한다."
    ),
    mappings=(
        (
            "§3, bullet list",
            "architecture constraints",
            "pooling 대신 strided convolution, G의 ReLU/tanh와 D의 LeakyReLU를 적용한다.",
        ),
        (
            "§3 and Figure 1",
            "generator topology",
            "latent spatial tensor를 fractional-strided convolution으로 확대한다.",
        ),
        (
            "§4–§5",
            "training/representation checks",
            "adversarial loss와 생성 샘플을 함께 관찰한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        §3의 설계 규칙을 8×8에 맞춥니다. Generator는 transposed convolution으로 해상도를
        키우고 마지막에 `tanh`, discriminator는 strided convolution과 LeakyReLU를 씁니다."""),
        _setup(331),
        code(
            """
            # TODO: z [N,8,1,1] -> image [N,1,8,8]인 G와 반대 방향 D를 작성하세요.
            raise NotImplementedError
            """,
            """
            class ConvG(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.ConvTranspose2d(8, 16, 4, 1, 0),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.ConvTranspose2d(16, 1, 4, 2, 1),
                        nn.Tanh(),
                    )

                def forward(self, z):
                    return self.net(z)


            class ConvD(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(1, 16, 4, 2, 1),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(16, 1, 4, 1, 0),
                    )

                def forward(self, x):
                    return self.net(x).flatten()


            G, D = ConvG().to(DEVICE), ConvD().to(DEVICE)
            z = torch.zeros(5, 8, 1, 1, device=DEVICE)
            assert G(z).shape == (5, 1, 8, 8) and D(G(z)).shape == (5,)
            assert isinstance(G.net[-1], nn.Tanh) and any(
                isinstance(m, nn.BatchNorm2d) for m in G.modules()
            )
            """,
            "implementation",
        ),
        code(
            """
            # TODO: 로컬 영상을 [-1,1]로 바꾸고 50번의 교대 adversarial update를 수행하세요.
            raise NotImplementedError
            """,
            """
            real_images = images_01 * 2 - 1
            og = torch.optim.Adam(G.parameters(), lr=0.003, betas=(0.5, 0.999))
            od = torch.optim.Adam(D.parameters(), lr=0.003, betas=(0.5, 0.999))
            history = []
            rng = torch.Generator().manual_seed(332)
            for _ in range(50):
                idx = randint_device(len(real_images), (64,), generator=rng)
                real = real_images[idx]
                z = randn_device((64, 8, 1, 1), generator=rng)
                ld = F.binary_cross_entropy_with_logits(
                    D(real), torch.ones(64, device=DEVICE)
                ) + F.binary_cross_entropy_with_logits(
                    D(G(z).detach()), torch.zeros(64, device=DEVICE)
                )
                od.zero_grad()
                ld.backward()
                od.step()
                z = randn_device((64, 8, 1, 1), generator=rng)
                lg = F.binary_cross_entropy_with_logits(
                    D(G(z)), torch.ones(64, device=DEVICE)
                )
                og.zero_grad()
                lg.backward()
                og.step()
                history.append((float(ld.detach()), float(lg.detach())))
            assert len(history) == 50 and np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: 고정 latent의 샘플 8개와 loss를 그리고 출력 범위를 검증하세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                samples = G(
                    randn_device((8, 8, 1, 1), generator=torch.Generator().manual_seed(333))
                )
            assert samples.shape == (8, 1, 8, 8) and float(samples.abs().max()) <= 1.0001
            fig, axes = plt.subplots(2, 5, figsize=(9, 4))
            h = np.asarray(history)
            axes[0, 0].plot(h[:, 0], label="D")
            axes[0, 0].plot(h[:, 1], label="G")
            axes[0, 0].legend()
            axes[1, 0].axis("off")
            for ax, img in zip(axes[:, 1:].ravel(), samples.detach().cpu()):
                ax.imshow(img[0], cmap="gray", vmin=-1, vmax=1)
                ax.axis("off")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "DCGAN이 pooling을 제거한 이유를 설명하세요.",
            "공간 축소·확대를 학습 가능한 strided convolution에 맡겨 네트워크가 자체적인 sampling 변환을 학습하도록 하기 위해서입니다.",
            "reflection",
        ),
    ),
)


WGAN = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=4,
    slug="wgan",
    short_title="WGAN",
    paper_title="Wasserstein GAN",
    authors="Martin Arjovsky, Soumith Chintala, and Léon Bottou",
    year=2017,
    primary_url="https://proceedings.mlr.press/v70/arjovsky17a.html",
    venue="ICML",
    difficulty="중급",
    expected_minutes=55,
    dataset_file=DATASET,
    prerequisites="GAN, Lipschitz 함수, optimizer",
    reproduction_goal="2차원 분포에서 sigmoid 없는 critic, n_critic 업데이트, weight clipping을 구현한다.",
    original_scale="논문은 이미지 모델과 8-Gaussian toy를 사용한다. 여기서는 toy 분포로 Algorithm 1을 그대로 축소한다.",
    mappings=(
        (
            "§2, Eq. (1)",
            "Earth Mover distance",
            "겹치지 않는 분포에서도 연속적인 거리 신호가 생기는 이유를 연결한다.",
        ),
        (
            "§3, Eq. (2)–(3)",
            "Kantorovich–Rubinstein critic",
            "sigmoid 없는 scalar critic의 real-fake 차이를 최대화한다.",
        ),
        (
            "§3, Algorithm 1",
            "clipped critic algorithm",
            "n_critic, RMSProp, parameter clipping 순서를 구현한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (3)의 critic은 확률이 아니라 실수 점수를 냅니다. Algorithm 1처럼 critic을 여러 번
        갱신하고 가중치를 compact set으로 clip해 1-Lipschitz 제약을 근사합니다."""),
        _setup(341),
        code(
            """
            # TODO: 2D generator/critic, critic_loss=E[fake]-E[real], clipping 함수를 구현하세요.
            raise NotImplementedError
            """,
            """
            G = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2)).to(DEVICE)
            C = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEVICE)


            def critic_loss(real_score, fake_score):
                return fake_score.mean() - real_score.mean()


            def clip_critic(model, c=0.05):
                with torch.no_grad():
                    for p in model.parameters():
                        p.clamp_(-c, c)


            probe = critic_loss(
                torch.tensor([2.0, 3.0], device=DEVICE),
                torch.tensor([-1.0, 0.0], device=DEVICE),
            )
            assert torch.allclose(probe, torch.tensor(-3.0, device=DEVICE))
            """,
            "implementation",
        ),
        code(
            """
            # TODO: Algorithm 1처럼 critic 3회당 generator 1회 update를 140회 반복하세요.
            raise NotImplementedError
            """,
            """
            oc = torch.optim.RMSprop(C.parameters(), lr=0.003)
            og = torch.optim.RMSprop(G.parameters(), lr=0.003)
            rng = torch.Generator().manual_seed(342)
            history = []
            fixed_z = randn_device((256, 2), generator=torch.Generator().manual_seed(343))
            for _ in range(140):
                for _ in range(3):
                    real = points[randint_device(len(points), (96,), generator=rng)]
                    fake = G(randn_device((96, 2), generator=rng)).detach()
                    lc = critic_loss(C(real), C(fake))
                    oc.zero_grad()
                    lc.backward()
                    oc.step()
                    clip_critic(C)
                lg = -C(G(randn_device((96, 2), generator=rng))).mean()
                og.zero_grad()
                lg.backward()
                og.step()
                history.append((-float(lc.detach()), float(lg.detach())))
            assert (
                max(float(p.detach().abs().max()) for p in C.parameters()) <= 0.05001
                and np.isfinite(history).all()
            )
            """,
            "training",
        ),
        code(
            """
            # TODO: critic gap을 metric으로 출력하고 real/fake 및 history를 그리세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                fake = G(fixed_z)
                gap = float(C(points).mean() - C(fake).mean())
            print(f"estimated Wasserstein critic gap={gap:.4f}")
            assert math.isfinite(gap)
            h = np.asarray(history)
            points_cpu, fake_cpu = points.detach().cpu(), fake.detach().cpu()
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].scatter(
                points_cpu[:, 0], points_cpu[:, 1], s=6, alpha=0.3, label="real"
            )
            axes[0].scatter(fake_cpu[:, 0], fake_cpu[:, 1], s=7, alpha=0.4, label="fake")
            axes[0].legend()
            axes[1].plot(h[:, 0], label="critic gap")
            axes[1].plot(h[:, 1], label="G objective")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "왜 WGAN critic 마지막에 sigmoid가 없나요?",
            (
                "KR dual의 critic은 확률 분류기가 아니라 Lipschitz 실함수입니다. 값의 차이가 "
                "Wasserstein 거리 추정량이므로 [0,1] 제약이 없습니다."
            ),
            "reflection",
        ),
    ),
)


PIX2PIX = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=5,
    slug="pix2pix",
    short_title="pix2pix",
    paper_title="Image-to-Image Translation with Conditional Adversarial Networks",
    authors="Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros",
    year=2017,
    primary_url=(
        "https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_I"
        "mage-To-Image_Translation_With_CVPR_2017_paper.html"
    ),
    venue="CVPR",
    difficulty="중급",
    expected_minutes=55,
    dataset_file=DATASET,
    prerequisites="conditional GAN, L1 loss, concatenation",
    reproduction_goal=(
        "paired 8×8 images에서 U-Net skip generator, spatial PatchGAN, "
        "adversarial BCE+L1 objective를 재현한다."
    ),
    original_scale=(
        "논문은 paired image datasets와 큰 U-Net/PatchGAN을 사용한다. 여기서는 "
        "로컬 8×8 paired images와 2×2 patch logits로 같은 공간적 계산 경로를 축소한다."
    ),
    mappings=(
        (
            "§3.1, Eq. (1)",
            "conditional GAN",
            "D가 source x와 target/generated y를 함께 보도록 한다.",
        ),
        (
            "§3.1, Eq. (3)–(4)",
            "cGAN plus L1",
            "adversarial realism과 paired reconstruction을 결합한다.",
        ),
        (
            "§3.2",
            "U-Net and PatchGAN",
            "skip-connected generator와 local patch discriminator를 실제 공간 tensor에 적용한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (4)의 $G^*=\arg\min_G\max_D\mathcal L_{cGAN}+\lambda\mathcal L_{L1}$을
        paired 8×8 image translation으로 재현합니다. `D([x,y])`의 channel concatenation,
        U-Net skip, 2×2 PatchGAN logits를 모두 실제 학습 경로에 포함합니다."""),
        _setup(351),
        code(
            """
            # TODO: spatial generator의 skip shape와 PatchGAN grid를 검증하세요.
            raise NotImplementedError
            """,
            """
            spatial_generator = TinyUNetGenerator().to(DEVICE)
            spatial_discriminator = SpatialPatchDiscriminator().to(DEVICE)
            spatial_generator_optimizer = torch.optim.Adam(
                spatial_generator.parameters(),
                lr=0.01,
            )
            spatial_discriminator_optimizer = torch.optim.Adam(
                spatial_discriminator.parameters(),
                lr=0.006,
            )
            portfolio_model = Pix2pixLab(
                spatial_generator,
                spatial_discriminator,
                spatial_generator_optimizer,
                spatial_discriminator_optimizer,
            )
            with torch.no_grad():
                preview, preview_logits = portfolio_model(spatial_train_source[:3])
                real_logits = portfolio_model.discriminate(
                    spatial_train_source[:3],
                    spatial_train_target[:3],
                )
            assert preview.shape == (3, 1, 8, 8)
            assert preview_logits.shape == real_logits.shape == (3, 1, 2, 2)
            assert portfolio_model.generator.skip_shapes == (
                (3, 8, 4, 4),
                (3, 4, 8, 8),
            )
            """,
            "implementation",
        ),
        code(
            """
            # TODO: train split만 사용해 spatial BCE와 L1을 교대로 학습하세요.
            raise NotImplementedError
            """,
            """
            rng = torch.Generator().manual_seed(352)
            history = []
            with torch.no_grad():
                initial_mae = float(
                    F.l1_loss(
                        portfolio_model.generator(spatial_eval_source),
                        spatial_eval_target,
                    )
                )
            for _ in range(180):
                idx = randint_device(len(domain_a), (96,), generator=rng)
                x, y = domain_a[idx], domain_b[idx]
                fake = G(x)
                ld = F.binary_cross_entropy_with_logits(
                    pair_score(x, y), torch.ones(96, 1, device=DEVICE)
                ) + F.binary_cross_entropy_with_logits(
                    pair_score(x, fake.detach()), torch.zeros(96, 1, device=DEVICE)
                )
                od.zero_grad()
                ld.backward()
                od.step()
                fake = G(x)
                adv = F.binary_cross_entropy_with_logits(
                    pair_score(x, fake), torch.ones(96, 1, device=DEVICE)
                )
                lg = adv + 10 * F.l1_loss(fake, y)
                og.zero_grad()
                lg.backward()
                og.step()
                history.append((float(ld.detach()), float(lg.detach())))
            assert np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: held-out paired image MAE와 spatial prediction을 시각화하세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                pred = portfolio_model.generator(spatial_eval_source)
                final_mae = float(F.l1_loss(pred, spatial_eval_target))
            print(f"held-out paired MAE: {initial_mae:.4f} -> {final_mae:.4f}")
            assert final_mae < initial_mae
            a_cpu = spatial_eval_source[0, 0].detach().cpu()
            b_cpu = spatial_eval_target[0, 0].detach().cpu()
            pred_cpu = pred[0, 0].detach().cpu()
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(a_cpu, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("source x")
            axes[1].imshow(b_cpu, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("paired y")
            axes[2].imshow(pred_cpu, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("G(x)")
            for axis in axes:
                axis.axis("off")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "L1 term만 쓰는 것과 cGAN을 함께 쓰는 것의 차이는 무엇인가요?",
            (
                "L1은 대응 위치의 평균 오차를 줄이지만 여러 가능한 출력에서 평균화될 수 있습니다. "
                "discriminator는 출력이 target-domain의 실제 표본처럼 보이는지 학습 신호를 더합니다."
            ),
            "reflection",
        ),
    ),
)


CYCLEGAN = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=6,
    slug="cyclegan",
    short_title="CycleGAN",
    paper_title="Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks",
    authors="Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros",
    year=2017,
    primary_url=(
        "https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unp"
        "aired_Image-To-Image_Translation_ICCV_2017_paper.html"
    ),
    venue="ICCV",
    difficulty="중급",
    expected_minutes=65,
    dataset_file=DATASET,
    prerequisites="GAN, cycle consistency, two-domain translation",
    reproduction_goal=(
        "unpaired 2D domains에서 두 generator와 두 discriminator를 "
        "adversarial+cycle loss로 학습한다."
    ),
    original_scale=(
        "논문은 unpaired image collections와 ResNet generators를 사용한다. "
        "축소판은 샘플 pairing을 섞고 양방향 mapping/cycle을 보존한다."
    ),
    mappings=(
        (
            "§3.1, Eq. (1)",
            "adversarial domain mapping",
            "G:A→B와 D_B의 adversarial objective를 구성한다.",
        ),
        (
            "§3.2, Eq. (2)",
            "cycle consistency",
            "F(G(a))≈a 및 G(F(b))≈b의 L1 loss를 구현한다.",
        ),
        (
            "§3.3, Eq. (3)–(4)",
            "full objective",
            "두 adversarial loss와 lambda-weighted cycle loss를 공동 최적화한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (2)는 직접 짝이 없는 상황에서 $F(G(x))\approx x$, $G(F(y))\approx y$를 요구합니다.
        B mini-batch를 별도로 섞어 index pairing을 제거하고 두 방향을 동시에 학습합니다."""),
        _setup(361),
        code(
            """
            # TODO: G_AB, G_BA와 각 domain scalar discriminator를 정의하고 cycle_loss를 구현하세요.
            raise NotImplementedError
            """,
            """
            def mapper():
                return nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2))


            def disc():
                return nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))


            G_AB, G_BA, D_A, D_B = ACCELERATOR.move(mapper(), mapper(), disc(), disc())


            def cycle_loss(a, b):
                return F.l1_loss(G_BA(G_AB(a)), a) + F.l1_loss(G_AB(G_BA(b)), b)


            assert cycle_loss(domain_a[:4], domain_b[:4]).ndim == 0
            """,
            "implementation",
        ),
        code(
            """
            # TODO: unpaired mini-batch로 두 D와 G를 220회 학습하고 cycle history를 저장하세요.
            raise NotImplementedError
            """,
            """
            opt_d = torch.optim.Adam(
                list(D_A.parameters()) + list(D_B.parameters()), lr=0.003
            )
            opt_g = torch.optim.Adam(
                list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.004
            )
            rng = torch.Generator().manual_seed(362)
            history = []
            with torch.no_grad():
                initial_cycle = float(cycle_loss(domain_a, domain_b))
            for _ in range(220):
                ia = randint_device(len(domain_a), (96,), generator=rng)
                ib = randint_device(len(domain_b), (96,), generator=rng)
                a, b = domain_a[ia], domain_b[ib]
                fb = G_AB(a)
                fa = G_BA(b)
                ld = (
                    F.binary_cross_entropy_with_logits(
                        D_A(a), torch.ones(96, 1, device=DEVICE)
                    )
                    + F.binary_cross_entropy_with_logits(
                        D_A(fa.detach()), torch.zeros(96, 1, device=DEVICE)
                    )
                    + F.binary_cross_entropy_with_logits(
                        D_B(b), torch.ones(96, 1, device=DEVICE)
                    )
                    + F.binary_cross_entropy_with_logits(
                        D_B(fb.detach()), torch.zeros(96, 1, device=DEVICE)
                    )
                )
                opt_d.zero_grad()
                ld.backward()
                opt_d.step()
                fb, fa = G_AB(a), G_BA(b)
                adv = F.binary_cross_entropy_with_logits(
                    D_B(fb), torch.ones(96, 1, device=DEVICE)
                ) + F.binary_cross_entropy_with_logits(
                    D_A(fa), torch.ones(96, 1, device=DEVICE)
                )
                cyc = cycle_loss(a, b)
                lg = adv + 8 * cyc
                opt_g.zero_grad()
                lg.backward()
                opt_g.step()
                history.append((float(ld.detach()), float(cyc.detach())))
            assert np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: full-domain cycle error 감소를 검증하고 A, G(A), B를 시각화하세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                mapped = G_AB(domain_a)
                final_cycle = float(cycle_loss(domain_a, domain_b))
            print(f"cycle L1: {initial_cycle:.4f} -> {final_cycle:.4f}")
            assert final_cycle < initial_cycle
            a_cpu, mapped_cpu, b_cpu = (
                domain_a.detach().cpu(),
                mapped.detach().cpu(),
                domain_b.detach().cpu(),
            )
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].scatter(a_cpu[:, 0], a_cpu[:, 1], s=7)
            axes[0].set_title("A (unpaired)")
            axes[1].scatter(mapped_cpu[:, 0], mapped_cpu[:, 1], s=7)
            axes[1].set_title("G_AB(A)")
            axes[2].scatter(b_cpu[:, 0], b_cpu[:, 1], s=7)
            axes[2].set_title("B")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "cycle loss만 최소화하면 왜 충분하지 않나요?",
            (
                "서로 역인 임의 변환도 cycle loss는 0이 될 수 있습니다. adversarial loss가 각 "
                "방향의 출력 분포를 실제 target domain에 맞춥니다."
            ),
            "reflection",
        ),
    ),
)


DDPM = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=7,
    slug="ddpm",
    short_title="DDPM",
    paper_title="Denoising Diffusion Probabilistic Models",
    authors="Jonathan Ho, Ajay Jain, and Pieter Abbeel",
    year=2020,
    primary_url=(
        "https://papers.nips.cc/paper_files/paper/2020/hash/4c5bcfec8"
        "584af0d967f1ab10179ca4b-Abstract.html"
    ),
    venue="NeurIPS",
    difficulty="고급",
    expected_minutes=70,
    dataset_file=DATASET,
    prerequisites="Gaussian Markov chain, variance schedule, neural noise prediction",
    reproduction_goal=(
        "8×8 로컬 영상에서 closed-form forward noising, epsilon objective, "
        "reverse sampling을 구현한다."
    ),
    original_scale=(
        "논문은 긴 diffusion chain과 U-Net으로 CIFAR-10 등을 생성한다. 여기서는 T=20과 "
        "MLP denoiser로 Algorithm 1–2를 보존한다."
    ),
    mappings=(
        (
            "§2, Eq. (2) and Eq. (4)",
            "forward process",
            "q(x_t|x_0)를 누적 alpha_bar로 한 번에 표본화한다.",
        ),
        (
            "§3.2, Eq. (11)",
            "epsilon parameterization",
            "reverse mean을 predicted noise로 표현한다.",
        ),
        (
            "§3.4, Eq. (14), Algorithm 1 and Algorithm 2",
            "simple objective and algorithms",
            "무작위 t의 noise MSE로 학습하고 역순 sampling한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        Eq. (4) $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$로 임의 시점을
        한 번에 만들고 Eq. (14)의 $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$를 최소화합니다."""),
        _setup(371),
        code(
            """
            # TODO: T=20 linear beta schedule과 q_sample(x0,t,noise)를 구현하세요.
            raise NotImplementedError
            """,
            """
            T = 20
            beta = torch.linspace(1e-4, 0.08, T, device=DEVICE)
            alpha = 1 - beta
            alpha_bar = torch.cumprod(alpha, 0)


            def extract(v, t, x):
                return v[t].reshape(-1, *([1] * (x.ndim - 1)))


            def q_sample(x0, t, noise):
                return (
                    extract(alpha_bar.sqrt(), t, x0) * x0
                    + extract((1 - alpha_bar).sqrt(), t, x0) * noise
                )


            probe = images_01[:4] * 2 - 1
            zero_t = torch.zeros(4, dtype=torch.long, device=DEVICE)
            eps = torch.zeros_like(probe)
            assert torch.allclose(q_sample(probe, zero_t, eps), alpha_bar[0].sqrt() * probe)
            """,
            "implementation",
        ),
        code(
            """
            # TODO: flattened x_t와 t/T를 받아 noise를 예측하는 MLP를 180회 학습하세요.
            raise NotImplementedError
            """,
            """
            denoiser = nn.Sequential(nn.Linear(65, 96), nn.SiLU(), nn.Linear(96, 64)).to(
                DEVICE
            )
            opt = torch.optim.Adam(denoiser.parameters(), lr=0.004)
            x0 = images_01.flatten(1) * 2 - 1
            rng = torch.Generator().manual_seed(372)
            history = []
            with torch.no_grad():
                t0 = randint_device(T, (len(x0),), generator=rng)
                n0 = randn_device(x0.shape, generator=rng)
                initial = float(
                    F.mse_loss(
                        denoiser(
                            torch.cat([q_sample(x0, t0, n0), t0[:, None] / (T - 1)], 1)
                        ),
                        n0,
                    )
                )
            for _ in range(180):
                t = randint_device(T, (len(x0),), generator=rng)
                noise = randn_device(x0.shape, generator=rng)
                xt = q_sample(x0, t, noise)
                pred = denoiser(torch.cat([xt, t[:, None] / (T - 1)], 1))
                loss = F.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                history.append(float(loss.detach()))
            assert history[-1] < initial and np.isfinite(history).all()
            """,
            "training",
        ),
        code(
            """
            # TODO: Algorithm 2의 reverse mean으로 8 samples를 만들고 loss/sample을 그리세요.
            raise NotImplementedError
            """,
            """
            sample_rng = torch.Generator().manual_seed(373)
            x = randn_device((8, 64), generator=sample_rng)
            with torch.no_grad():
                for step in reversed(range(T)):
                    t = torch.full((len(x),), step, dtype=torch.long, device=DEVICE)
                    pred = denoiser(torch.cat([x, t[:, None] / (T - 1)], 1))
                    mean = (
                        x - beta[step] / torch.sqrt(1 - alpha_bar[step]) * pred
                    ) / torch.sqrt(alpha[step])
                    x = (
                        mean
                        if step == 0
                        else mean
                        + beta[step].sqrt() * randn_device(x.shape, generator=sample_rng)
                    )
            assert x.shape == (8, 64) and torch.isfinite(x).all()
            fig, axes = plt.subplots(2, 5, figsize=(9, 4))
            axes[0, 0].plot(history)
            axes[0, 0].set_title("noise MSE")
            axes[1, 0].axis("off")
            for ax, img in zip(axes[:, 1:].ravel(), x.detach().cpu()):
                ax.imshow(img.reshape(8, 8).clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
                ax.axis("off")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "왜 임의 t를 뽑는 Eq. (14)가 전체 chain 학습에 충분한가요?",
            (
                "t에 대한 균등 Monte Carlo 평균은 모든 timestep loss 합의 불편추정량입니다. Eq. "
                "(4) 덕분에 앞선 모든 step을 순차 실행하지 않아도 x_t를 만들 수 있습니다."
            ),
            "reflection",
        ),
    ),
)


SCORE_SDE = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=8,
    slug="score_sde",
    short_title="Score SDE",
    paper_title="Score-Based Generative Modeling through Stochastic Differential Equations",
    authors=(
        "Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, "
        "Abhishek Kumar, Stefano Ermon, and Ben Poole"
    ),
    year=2021,
    primary_url="https://arxiv.org/abs/2011.13456",
    venue="ICLR",
    difficulty="고급",
    expected_minutes=70,
    dataset_file=DATASET,
    prerequisites="score matching, SDE, Euler discretization",
    reproduction_goal="2D 점에서 VP-SDE marginal score를 학습하고 reverse-time Euler sampling을 수행한다.",
    original_scale=(
        "논문은 continuous-time score networks와 predictor-corrector "
        "samplers로 고해상도 이미지를 생성한다. 축소판은 VP-SDE와 reverse drift를 2D로 "
        "구현한다."
    ),
    mappings=(
        (
            "§3.1 Eq. (5); §3.2 Eq. (6)",
            "forward and reverse SDE",
            "forward drift/diffusion과 score가 들어간 reverse drift를 연결한다.",
        ),
        (
            "§3.3, Eq. (7)",
            "continuous denoising score matching",
            "perturbation kernel의 conditional score를 회귀한다.",
        ),
        (
            "§3.4, Eq. (11)",
            "VP-SDE marginal과 conditional score target",
            "m(t), σ(t), -z/σ(t)의 shape와 수치 범위를 검증한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        VP-SDE의 marginal $x_t=m(t)x_0+\sigma(t)z$를 이용해 Eq. (7)의 conditional score
        $-z/\sigma(t)$를 회귀합니다. 이후 Eq. (6)의 reverse-time drift를 Euler로 적분합니다."""),
        _setup(381),
        code(
            """
            # TODO: beta(t)=.1+9.9t VP marginal의 mean coefficient/std와 perturbed
            # sample/target score를 구현하세요.
            raise NotImplementedError
            """,
            """
            def vp_stats(t):
                integral = 0.1 * t + 4.95 * t.square()
                mean = torch.exp(-0.5 * integral)
                std = torch.sqrt((1 - mean.square()).clamp_min(1e-5))
                return mean, std


            def perturb(x0, t, z):
                m, s = vp_stats(t)
                return m[:, None] * x0 + s[:, None] * z, -z / s[:, None]


            t = torch.full((4,), 0.5, device=DEVICE)
            z = torch.ones(4, 2, device=DEVICE)
            xt, target = perturb(points[:4], t, z)
            assert xt.shape == target.shape == (4, 2) and torch.isfinite(target).all()
            """,
            "implementation",
        ),
        code(
            """
            # TODO: [x_t,t] -> score MLP를 Eq. (7) weighted MSE로 220회 학습하세요.
            raise NotImplementedError
            """,
            """
            score = nn.Sequential(
                nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2)
            ).to(DEVICE)
            opt = torch.optim.Adam(score.parameters(), lr=0.004)
            rng = torch.Generator().manual_seed(382)
            history = []
            for _ in range(220):
                idx = randint_device(len(points), (192,), generator=rng)
                x0 = points[idx]
                t = 0.02 + 0.96 * rand_device((192,), generator=rng)
                z = randn_device((192, 2), generator=rng)
                xt, target = perturb(x0, t, z)
                _, s = vp_stats(t)
                pred = score(torch.cat([xt, t[:, None]], 1))
                loss = ((pred - target).square().sum(1) * s.square()).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                history.append(float(loss.detach()))
            assert np.isfinite(history).all() and np.mean(history[-20:]) < np.mean(
                history[:20]
            )
            """,
            "training",
        ),
        code(
            """
            # TODO: Eq. (6)의 reverse drift로 t=1에서 .02까지 Euler sampling하고 분포를 그리세요.
            raise NotImplementedError
            """,
            """
            rng2 = torch.Generator().manual_seed(383)
            x = randn_device((400, 2), generator=rng2)
            grid = torch.linspace(1, 0.02, 50, device=DEVICE)
            with torch.no_grad():
                for i in range(len(grid) - 1):
                    t = grid[i]
                    dt = grid[i + 1] - t
                    beta_t = 0.1 + 9.9 * t
                    s = score(
                        torch.cat([x, torch.full((len(x), 1), float(t), device=DEVICE)], 1)
                    )
                    reverse_drift = -0.5 * beta_t * x - beta_t * s
                    diffusion_noise = randn_device(x.shape, generator=rng2)
                    x = (
                        x
                        + reverse_drift * dt
                        + torch.sqrt(beta_t * (-dt)) * diffusion_noise
                    )
            assert x.shape == (400, 2) and torch.isfinite(x).all()
            points_cpu, x_cpu = points.detach().cpu(), x.detach().cpu()
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].plot(history)
            axes[0].set_title("weighted score loss")
            axes[1].scatter(points_cpu[:, 0], points_cpu[:, 1], s=7, alpha=0.4)
            axes[1].set_title("data")
            axes[2].scatter(x_cpu[:, 0], x_cpu[:, 1], s=7, alpha=0.4)
            axes[2].set_title("reverse-SDE samples")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "reverse SDE에서 score가 필요한 이유는 무엇인가요?",
            (
                "forward diffusion이 확률질량을 퍼뜨린 방향을 뒤집으려면 현재 noisy density의 "
                "log-gradient가 필요합니다. Eq. (6)은 이 score를 drift correction으로 "
                "사용합니다."
            ),
            "reflection",
        ),
    ),
)


LATENT_DIFFUSION = FieldPaperSpec(
    field_id=FIELD_ID,
    field_title=FIELD_TITLE,
    number=9,
    slug="latent_diffusion",
    short_title="Latent Diffusion",
    paper_title="High-Resolution Image Synthesis with Latent Diffusion Models",
    authors="Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer",
    year=2022,
    primary_url=(
        "https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_"
        "High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models"
        "_CVPR_2022_paper.html"
    ),
    venue="CVPR",
    difficulty="고급",
    expected_minutes=75,
    dataset_file=DATASET,
    prerequisites="autoencoder, DDPM, latent representation",
    reproduction_goal=(
        "8×8 영상을 8D latent로 압축한 뒤 latent-space noise predictor와 "
        "decoder sampling을 구현한다."
    ),
    original_scale=(
        "논문은 perceptual autoencoder, convolutional U-Net, "
        "cross-attention으로 고해상도 이미지를 생성한다. 여기서는 Eq. (2)의 latent "
        "diffusion과 압축 이득에 집중한다."
    ),
    mappings=(
        (
            "§3.1",
            "perceptual compression",
            "encoder E와 decoder D로 의미 보존 latent space를 만든다.",
        ),
        (
            "§3.2, Eq. (1)–(2)",
            "latent diffusion objective",
            "pixel x 대신 z=E(x)에서 epsilon prediction을 학습한다.",
        ),
        (
            "§3.2, Eq. (1)–(2)와 DDPM reverse process",
            "latent reverse sampling과 decoder 복원",
            "reverse trajectory와 decoded sample의 shape·finite 값을 검증한다.",
        ),
    ),
    cells=(
        _intro(r"""## 핵심 재현

        §3.1처럼 먼저 $z=E(x)$와 $\tilde x=D(z)$를 학습합니다. 이어 Eq. (2)의 diffusion
        objective를 64D pixel이 아니라 8D latent에서 실행해 계산 차원을 직접 비교합니다."""),
        _setup(391),
        code(
            """
            # TODO: 64->32->8 encoder와 8->32->64 sigmoid decoder를 만들고 140회 reconstruction
            # 학습하세요.
            raise NotImplementedError
            """,
            """
            E = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8)).to(DEVICE)
            Dec = nn.Sequential(
                nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid()
            ).to(DEVICE)
            x_data = images_01.flatten(1)
            ae_opt = torch.optim.Adam(
                list(E.parameters()) + list(Dec.parameters()), lr=0.01
            )
            ae_history = []
            with torch.no_grad():
                initial_ae = float(F.mse_loss(Dec(E(x_data)), x_data))
            for _ in range(140):
                rec = Dec(E(x_data))
                loss = F.mse_loss(rec, x_data)
                ae_opt.zero_grad()
                loss.backward()
                ae_opt.step()
                ae_history.append(float(loss.detach()))
            assert ae_history[-1] < initial_ae and E(x_data).shape == (300, 8)
            """,
            "compression",
        ),
        code(
            """
            # TODO: detached latent에 T=16 forward noise를 적용하고 [z_t,t] noise MLP를 180회 학습하세요.
            raise NotImplementedError
            """,
            """
            with torch.no_grad():
                latent = E(x_data)
                latent_mean = latent.mean(0)
                latent_std = latent.std(0).clamp_min(0.1)
                latent = (latent - latent_mean) / latent_std
            Tz = 16
            bz = torch.linspace(1e-4, 0.10, Tz, device=DEVICE)
            az = 1 - bz
            abz = torch.cumprod(az, 0)
            eps_model = nn.Sequential(nn.Linear(9, 48), nn.SiLU(), nn.Linear(48, 8)).to(
                DEVICE
            )
            opt = torch.optim.Adam(eps_model.parameters(), lr=0.005)
            rng = torch.Generator().manual_seed(392)
            diff_history = []
            for _ in range(180):
                t = randint_device(Tz, (len(latent),), generator=rng)
                noise = randn_device(latent.shape, generator=rng)
                zt = abz[t, None].sqrt() * latent + (1 - abz[t, None]).sqrt() * noise
                pred = eps_model(torch.cat([zt, t[:, None] / (Tz - 1)], 1))
                loss = F.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                diff_history.append(float(loss.detach()))
            assert np.isfinite(diff_history).all() and np.mean(
                diff_history[-20:]
            ) < np.mean(diff_history[:20])
            """,
            "latent-diffusion",
        ),
        code(
            """
            # TODO: latent reverse sampling 후 decoder로 복원하고 compression ratio와 plots를 출력하세요.
            raise NotImplementedError
            """,
            """
            rng2 = torch.Generator().manual_seed(393)
            z = randn_device((8, 8), generator=rng2)
            with torch.no_grad():
                for step in reversed(range(Tz)):
                    t = torch.full((len(z),), step, dtype=torch.long, device=DEVICE)
                    pred = eps_model(torch.cat([z, t[:, None] / (Tz - 1)], 1))
                    mean = (z - bz[step] / torch.sqrt(1 - abz[step]) * pred) / torch.sqrt(
                        az[step]
                    )
                    z = (
                        mean
                        if step == 0
                        else mean + bz[step].sqrt() * randn_device(z.shape, generator=rng2)
                    )
                decoded = Dec(z * latent_std + latent_mean)
            ratio = 64 / 8
            print(
                f"diffusion dimension: 64 -> 8 ({ratio:.0f}x smaller), AE MSE={ae_history[-1]:.4f}"
            )
            assert ratio == 8 and decoded.shape == (8, 64) and torch.isfinite(decoded).all()
            fig, axes = plt.subplots(2, 5, figsize=(9, 4))
            axes[0, 0].plot(ae_history)
            axes[0, 0].set_title("AE loss")
            axes[1, 0].plot(diff_history)
            axes[1, 0].set_title("latent DM loss")
            for ax, img in zip(axes[:, 1:].ravel(), decoded.detach().cpu()):
                ax.imshow(img.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            fig.tight_layout()
            plt.show()
            """,
            "evaluation",
        ),
        paired_markdown(
            "latent compression을 너무 강하게 하면 어떤 문제가 생기나요?",
            (
                "encoder가 고주파와 세부 구조를 제거하면 diffusion이 latent 분포를 완벽히 학습해도 "
                "decoder가 그 정보를 복원할 수 없습니다. §4.1의 downsampling-factor "
                "trade-off입니다."
            ),
            "reflection",
        ),
    ),
)


_PORTFOLIO_GUIDES = {
    "stacked_denoising_autoencoder": {
        "equation": (
            r"\mathcal L_{DAE}=\mathbb E_{x,\tilde x\sim q_D(\tilde x|x)}"
            r"\left[\lVert x-g_\theta(f_\theta(\tilde x))\rVert_2^2\right]"
        ),
        "symbols": (
            r"$x$는 깨끗한 64차원 영상, $\tilde x$는 masking으로 훼손한 영상, "
            r"$f_\theta/g_\theta$는 encoder/decoder입니다."
        ),
        "shape": "clean/noisy [N, 64] → latent [N, 24] → restored [N, 64]",
        "task": "mask를 직접 만들고 clean target MSE, optimizer update, 복원 성능을 구현합니다.",
        "flow": "고정 seed 훼손 → encode/decode → clean MSE 역전파 → noisy/복원 MSE 비교",
    },
    "vae": {
        "equation": (
            r"-\mathcal L_{ELBO}=\mathbb E_{q_\phi(z|x)}[-\log p_\theta(x|z)]"
            r"+\beta D_{KL}(q_\phi(z|x)\Vert p(z))"
        ),
        "symbols": (
            r"$q_\phi$는 encoder posterior, $p_\theta$는 decoder likelihood, "
            r"$z=\mu+\sigma\epsilon$은 재매개화 표본입니다."
        ),
        "shape": "point x [N, 2] → (mu, logvar) [N, 2] → z [N, 2] → recon [N, 2]",
        "task": "재매개화, diagonal Gaussian KL, negative ELBO와 deterministic 평가를 작성합니다.",
        "flow": "encode → rsample → decode → reconstruction+KL update → posterior-mean 평가",
    },
    "gan": {
        "equation": (
            r"\min_G\max_D\;\mathbb E_x[\log D(x)]"
            r"+\mathbb E_z[\log(1-D(G(z)))]"
        ),
        "symbols": (
            r"$z$는 2차원 noise, $G(z)$는 fake point, $D(\cdot)$는 real logit이며 "
            r"실습의 $G$는 non-saturating $-\log D(G(z))$를 씁니다."
        ),
        "shape": "z [N, 2] → fake [N, 2]; real/fake [N, 2] → logit [N, 1]",
        "task": "BCE 두 목적과 detach 경계, discriminator/generator 교대 update를 구현합니다.",
        "flow": "real/noise batch → D update → 새 noise → G update → 분포와 D(fake) 평가",
    },
    "dcgan": {
        "equation": (
            r"\mathcal L_G=-\mathbb E_z[\log \sigma(D(G(z)))]"
            r",\qquad \mathcal L_D=\operatorname{BCE}_{real}+\operatorname{BCE}_{fake}"
        ),
        "symbols": (
            r"$G$는 transposed convolution, $D$는 strided convolution, "
            r"$\sigma$는 logit을 확률로 바꾸는 sigmoid입니다."
        ),
        "shape": "z [N, 8, 1, 1] → image [N, 1, 8, 8] → D logit [N]",
        "task": "DCGAN 설계 규칙과 adversarial update, tanh 출력 범위 검사를 구현합니다.",
        "flow": "[-1,1] image batch → convolutional D/G update → 고정 latent sample 평가",
    },
    "wgan": {
        "equation": (
            r"W(P_r,P_g)\approx\max_{\lVert f\rVert_L\le1}"
            r"\mathbb E_{x\sim P_r}[f(x)]-\mathbb E_{z}[f(G(z))]"
        ),
        "symbols": (
            r"$f$는 sigmoid가 없는 critic, $P_r/P_g$는 real/generated 분포, "
            r"$\lVert f\rVert_L\le1$은 Lipschitz 제약입니다."
        ),
        "shape": "real/fake [N, 2] → critic score [N, 1]",
        "task": "critic gap, n_critic 반복, RMSProp, parameter clipping 순서를 구현합니다.",
        "flow": "critic 3회 update+clip → generator 1회 update → critic gap 평가",
    },
    "pix2pix": {
        "equation": (
            r"G^*=\arg\min_G\max_D\mathcal L_{cGAN}(G,D)"
            r"+\lambda\mathbb E_{x,y}\lVert y-G(x)\rVert_1"
        ),
        "symbols": (
            r"$x/y$는 paired source/target, $D(x,y)$는 conditional critic, "
            r"$\lambda$는 구조 보존 L1 가중치입니다."
        ),
        "shape": (
            "source [N,1,8,8] → generated [N,1,8,8]; "
            "concat(x,y) [N,2,8,8] → patch logits [N,1,2,2]"
        ),
        "task": "tiny U-Net skip, spatial PatchGAN, adversarial BCE+L1을 구현합니다.",
        "flow": "paired train images → patch D update → U-Net BCE+L1 update → held-out MAE",
    },
    "cyclegan": {
        "equation": (
            r"\mathcal L_{cyc}=\mathbb E_x\lVert F(G(x))-x\rVert_1"
            r"+\mathbb E_y\lVert G(F(y))-y\rVert_1"
        ),
        "symbols": (
            r"$G:A\to B$와 $F:B\to A$는 두 generator, $D_A/D_B$는 각 domain critic입니다."
        ),
        "shape": "unpaired A/B [N, 2] ↔ mapped/cycled [N, 2]",
        "task": "양방향 변환, cycle consistency, 두 adversarial loss의 교대 update를 구현합니다.",
        "flow": "독립 A/B batch → 두 D update → 두 G+cycle update → cycle L1 평가",
    },
    "ddpm": {
        "equation": (
            r"x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad "
            r"\mathcal L=\mathbb E\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2"
        ),
        "symbols": (
            r"$t$는 timestep, $\bar\alpha_t$는 누적 signal 비율, "
            r"$\epsilon_\theta$는 주입한 Gaussian noise predictor입니다."
        ),
        "shape": "x0/xt/noise [N, 64], t [N] → concat [N, 65] → predicted noise [N, 64]",
        "task": "forward noising, epsilon loss, reverse mean과 Algorithm 2 sampling을 구현합니다.",
        "flow": "무작위 t/noise → q_sample → noise MSE update → t=T-1..0 reverse sample",
    },
    "score_sde": {
        "equation": (
            r"\mathcal L=\mathbb E_{t,x_0,x_t}\lambda(t)"
            r"\lVert s_\theta(x_t,t)-\nabla_{x_t}\log p_{0t}(x_t|x_0)\rVert_2^2"
        ),
        "symbols": (
            r"$s_\theta$는 score network, $p_{0t}$는 VP-SDE perturbation kernel, "
            r"$\lambda(t)=\sigma(t)^2$는 시간별 가중치입니다."
        ),
        "shape": "point xt [N, 2] + time [N, 1] → score [N, 2]",
        "task": "VP marginal, conditional score target, weighted loss와 reverse Euler step을 구현합니다.",
        "flow": "continuous t perturb → score update → t=1..0.02 reverse-SDE sampling",
    },
    "latent_diffusion": {
        "equation": (
            r"\mathcal L_{LDM}=\mathbb E_{z=E(x),t,\epsilon}"
            r"\lVert\epsilon-\epsilon_\theta(z_t,t)\rVert_2^2"
        ),
        "symbols": (
            r"$E/D$는 perceptual encoder/decoder, $z_t$는 noisy latent, "
            r"$\epsilon_\theta$는 latent noise predictor입니다."
        ),
        "shape": "image [N, 64] → z [N, 8]; concat(zt,t) [N, 9] → noise [N, 8]",
        "task": "autoencoder와 latent diffusion을 분리 학습하고 8배 차원 축소를 평가합니다.",
        "flow": "AE reconstruction → latent normalize → epsilon update → reverse latent → decode",
    },
}


_PORTFOLIO_CLASS_NAMES = {
    "stacked_denoising_autoencoder": "DenoisingAutoencoderLab",
    "vae": "VariationalAutoencoderLab",
    "gan": "AdversarialGameLab",
    "dcgan": "ConvolutionalGanLab",
    "wgan": "WassersteinGanLab",
    "pix2pix": "Pix2pixLab",
    "cyclegan": "CycleConsistentTranslationLab",
    "ddpm": "DiffusionProcessLab",
    "score_sde": "ScoreSdeLab",
    "latent_diffusion": "LatentDiffusionLab",
}


def _portfolio_markdown(spec: FieldPaperSpec) -> object:
    guide = _PORTFOLIO_GUIDES[spec.slug]
    return markdown(
        rf"""
        ## 포트폴리오 구현 설계: 수식 → 책임 → 검증

        논문의 핵심 목적을 실행 가능한 객체의 `forward`, `loss`, `update`, `evaluate`로
        나눕니다. 아래 식이 클래스 내부에서 생략되지 않고 어떤 tensor 연산이 되는지
        메서드별로 추적해 보세요.

        $$
        {guide["equation"]}
        $$

        - **기호와 역할:** {guide["symbols"]}
        - **shape 계약:** `{guide["shape"]}`
        - **논문 위치:** `{spec.mappings[0][0]}`의 **{spec.mappings[0][1]}**
        - **코드 Task:** {guide["task"]}
        - **학습·평가 흐름:** {guide["flow"]}
        - **원문 대비 한계:** {spec.original_scale}
        - **구현 이유:** 수식의 tensor 경계와 optimizer 책임을 클래스 메서드별로
          분리해 코드와 논문을 한 줄씩 대조할 수 있게 합니다.
        - **완료 증거:** 마지막 셀에서 `update`, parameter 변화, 평가 metric을
          모두 `assert`합니다.

        정답 클래스는 교육용 편의 함수가 핵심 계산을 대신하지 않도록 loss와 update 순서를
        메서드 본문에 드러냅니다. 실습본에서는 같은 공개 API를 유지한 채 TODO를 채우세요.
        """,
        "portfolio-explanation",
        "equation",
    )


def _method_contract(spec: FieldPaperSpec, method_name: str) -> str:
    shape = _PORTFOLIO_GUIDES[spec.slug]["shape"]
    contracts = {
        "__init__": "구성 요소와 optimizer를 저장한다. 학습은 update에서만 수행한다.",
        "config": "재현 범위와 핵심 하이퍼파라미터를 dict로 반환한다.",
        "forward": f"논문 순전파를 계산한다. shape 계약: {shape}.",
        "loss": "논문의 목적 함수를 미분 가능한 scalar tensor로 반환한다.",
        "update": "zero_grad, backward, step을 수행하고 유한한 float loss를 반환한다.",
        "evaluate": "no_grad 경로에서 논문과 연결된 정량 지표 dict를 반환한다.",
        "marginal": "VP-SDE 주변분포의 noisy sample, target score, std를 반환한다.",
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
    "stacked_denoising_autoencoder": (
        r"""
        class DenoisingAutoencoderLab(nn.Module):
            def __init__(self, network, optimizer, corruption_probability=0.35):
                super().__init__()
                self.network = network
                self.optimizer = optimizer
                self.corruption_probability = corruption_probability

            def config(self):
                return {
                    "input_dim": 64,
                    "latent_dim": 24,
                    "corruption_probability": self.corruption_probability,
                }

            def forward(self, clean, generator):
                keep = rand_device(clean.shape, generator=generator)
                keep = keep >= self.corruption_probability
                noisy = clean * keep.to(clean.dtype)
                restored = self.network(noisy)
                return noisy, restored

            def loss(self, clean, generator):
                _, restored = self.forward(clean, generator)
                return (restored - clean).square().mean()

            def update(self, clean, generator):
                objective = self.loss(clean, generator)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def evaluate(self, clean, generator):
                with torch.no_grad():
                    noisy, restored = self.forward(clean, generator)
                    noisy_mse = (noisy - clean).square().mean()
                    restored_mse = (restored - clean).square().mean()
                return {
                    "noisy_mse": float(noisy_mse),
                    "restored_mse": float(restored_mse),
                }


        portfolio_model = DenoisingAutoencoderLab(model, optimizer)
        assert portfolio_model.config()["latent_dim"] == 24
        """
    ),
    "vae": (
        r"""
        class VariationalAutoencoderLab(nn.Module):
            def __init__(self, network, optimizer, beta=0.08):
                super().__init__()
                self.network = network
                self.optimizer = optimizer
                self.beta = beta

            def config(self):
                return {"data_dim": 2, "latent_dim": 2, "beta": self.beta}

            def forward(self, batch, deterministic=False):
                mu, logvar = self.network.encode(batch)
                standard_deviation = torch.exp(0.5 * logvar)
                epsilon = (
                    torch.zeros_like(mu) if deterministic else torch.randn_like(mu)
                )
                latent = mu + standard_deviation * epsilon
                reconstruction = self.network.dec(latent)
                return reconstruction, mu, logvar

            def loss(self, batch):
                reconstruction, mu, logvar = self.forward(batch)
                reconstruction_term = (reconstruction - batch).square().sum(dim=1)
                kl_term = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
                return (reconstruction_term + self.beta * kl_term).mean()

            def update(self, batch):
                objective = self.loss(batch)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def evaluate(self, batch):
                with torch.no_grad():
                    reconstruction, mu, logvar = self.forward(batch, deterministic=True)
                    mse = F.mse_loss(reconstruction, batch)
                    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
                return {"mse": float(mse), "mean_kl": float(kl.mean())}


        portfolio_model = VariationalAutoencoderLab(model, optimizer)
        assert portfolio_model.config()["beta"] == 0.08
        """
    ),
    "gan": (
        r"""
        class AdversarialGameLab(nn.Module):
            def __init__(
                self,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
            ):
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator
                self.generator_optimizer = generator_optimizer
                self.discriminator_optimizer = discriminator_optimizer

            def config(self):
                return {"noise_dim": 2, "data_dim": 2, "objective": "non-saturating"}

            def forward(self, noise):
                fake = self.generator(noise)
                return fake, self.discriminator(fake)

            def loss(self, real, noise):
                fake, fake_logits = self.forward(noise)
                real_logits = self.discriminator(real)
                real_term = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                )
                fake_term = F.binary_cross_entropy_with_logits(
                    fake_logits.detach(),
                    torch.zeros_like(fake_logits),
                )
                generator_term = F.binary_cross_entropy_with_logits(
                    self.discriminator(fake),
                    torch.ones_like(fake_logits),
                )
                return real_term + fake_term, generator_term

            def update(self, real, discriminator_noise, generator_noise):
                fake = self.generator(discriminator_noise).detach()
                real_logits = self.discriminator(real)
                fake_logits = self.discriminator(fake)
                discriminator_loss = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                ) + F.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                generated = self.generator(generator_noise)
                generator_logits = self.discriminator(generated)
                generator_loss = F.binary_cross_entropy_with_logits(
                    generator_logits,
                    torch.ones_like(generator_logits),
                )
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()
                return float(discriminator_loss.detach()), float(
                    generator_loss.detach()
                )

            def evaluate(self, real, noise):
                with torch.no_grad():
                    fake, logits = self.forward(noise)
                    mean_gap = (fake.mean(dim=0) - real.mean(dim=0)).norm()
                    fake_probability = torch.sigmoid(logits).mean()
                return {
                    "mean_gap": float(mean_gap),
                    "discriminator_fake_probability": float(fake_probability),
                }


        portfolio_model = AdversarialGameLab(G, D, opt_g, opt_d)
        assert portfolio_model.config()["objective"] == "non-saturating"
        """
    ),
    "dcgan": (
        r"""
        class ConvolutionalGanLab(nn.Module):
            def __init__(
                self,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
            ):
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator
                self.generator_optimizer = generator_optimizer
                self.discriminator_optimizer = discriminator_optimizer

            def config(self):
                return {"noise_shape": (8, 1, 1), "image_shape": (1, 8, 8)}

            def forward(self, noise):
                image = self.generator(noise)
                return image, self.discriminator(image)

            def loss(self, real, noise):
                fake, fake_logits = self.forward(noise)
                real_logits = self.discriminator(real)
                discriminator_loss = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                ) + F.binary_cross_entropy_with_logits(
                    fake_logits.detach(),
                    torch.zeros_like(fake_logits),
                )
                generator_loss = F.binary_cross_entropy_with_logits(
                    self.discriminator(fake),
                    torch.ones_like(fake_logits),
                )
                return discriminator_loss, generator_loss

            def update(self, real, discriminator_noise, generator_noise):
                fake = self.generator(discriminator_noise).detach()
                real_logits = self.discriminator(real)
                fake_logits = self.discriminator(fake)
                discriminator_loss = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                ) + F.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                generated = self.generator(generator_noise)
                generated_logits = self.discriminator(generated)
                generator_loss = F.binary_cross_entropy_with_logits(
                    generated_logits,
                    torch.ones_like(generated_logits),
                )
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()
                return float(discriminator_loss.detach()), float(
                    generator_loss.detach()
                )

            def evaluate(self, noise):
                with torch.no_grad():
                    images, logits = self.forward(noise)
                return {
                    "sample_shape": tuple(images.shape),
                    "absolute_max": float(images.abs().max()),
                    "mean_logit": float(logits.mean()),
                }


        portfolio_model = ConvolutionalGanLab(G, D, og, od)
        assert portfolio_model.config()["image_shape"] == (1, 8, 8)
        """
    ),
    "wgan": (
        r"""
        class WassersteinGanLab(nn.Module):
            def __init__(
                self, generator, critic, generator_optimizer, critic_optimizer
            ):
                super().__init__()
                self.generator = generator
                self.critic = critic
                self.generator_optimizer = generator_optimizer
                self.critic_optimizer = critic_optimizer
                self.clip_value = 0.05

            def config(self):
                return {"noise_dim": 2, "critic_steps": 3, "clip": self.clip_value}

            def forward(self, noise):
                fake = self.generator(noise)
                return fake, self.critic(fake)

            def loss(self, real, noise):
                fake, fake_score = self.forward(noise)
                real_score = self.critic(real)
                critic_objective = fake_score.mean() - real_score.mean()
                generator_objective = -self.critic(fake).mean()
                return critic_objective, generator_objective

            def update(self, real, critic_noises, generator_noise):
                critic_value = None
                for noise in critic_noises:
                    fake = self.generator(noise).detach()
                    critic_value = self.critic(fake).mean() - self.critic(real).mean()
                    self.critic_optimizer.zero_grad()
                    critic_value.backward()
                    self.critic_optimizer.step()
                    with torch.no_grad():
                        for parameter in self.critic.parameters():
                            parameter.clamp_(-self.clip_value, self.clip_value)
                generated = self.generator(generator_noise)
                generator_value = -self.critic(generated).mean()
                self.generator_optimizer.zero_grad()
                generator_value.backward()
                self.generator_optimizer.step()
                return float(critic_value.detach()), float(generator_value.detach())

            def evaluate(self, real, noise):
                with torch.no_grad():
                    fake, fake_score = self.forward(noise)
                    gap = self.critic(real).mean() - fake_score.mean()
                return {
                    "critic_gap": float(gap),
                    "fake_mean": fake.mean(dim=0).tolist(),
                }


        portfolio_model = WassersteinGanLab(G, C, og, oc)
        assert portfolio_model.config()["critic_steps"] == 3
        """
    ),
    "pix2pix": (
        r"""
        class TinyUNetGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_full = nn.Conv2d(1, 4, 3, padding=1)
                self.encoder_half = nn.Conv2d(4, 8, 4, stride=2, padding=1)
                self.bottleneck = nn.Conv2d(8, 12, 4, stride=2, padding=1)
                self.up_half = nn.ConvTranspose2d(12, 8, 4, stride=2, padding=1)
                self.fuse_half = nn.Conv2d(16, 8, 3, padding=1)
                self.up_full = nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1)
                self.output = nn.Conv2d(8, 1, 3, padding=1)
                self.skip_shapes = ()

            def forward(self, source):
                skip_full = F.relu(self.encoder_full(source))
                skip_half = F.relu(self.encoder_half(skip_full))
                latent = F.relu(self.bottleneck(skip_half))
                decoded_half = F.relu(self.up_half(latent))
                decoded_half = torch.cat([decoded_half, skip_half], dim=1)
                decoded_half = F.relu(self.fuse_half(decoded_half))
                decoded_full = F.relu(self.up_full(decoded_half))
                decoded_full = torch.cat([decoded_full, skip_full], dim=1)
                self.skip_shapes = (tuple(skip_half.shape), tuple(skip_full.shape))
                return torch.sigmoid(self.output(decoded_full))


        class SpatialPatchDiscriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(2, 8, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(8, 16, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16, 1, 1),
                )

            def forward(self, paired_images):
                return self.layers(paired_images)


        class Pix2pixLab(nn.Module):
            def __init__(
                self,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
            ):
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator
                self.generator_optimizer = generator_optimizer
                self.discriminator_optimizer = discriminator_optimizer
                self.l1_weight = 20.0

            def config(self):
                return {
                    "image_shape": (1, 8, 8),
                    "patch_grid": (2, 2),
                    "l1_weight": self.l1_weight,
                }

            def discriminate(self, source, target):
                paired_images = torch.cat([source, target], dim=1)
                return self.discriminator(paired_images)

            def forward(self, source):
                generated = self.generator(source)
                return generated, self.discriminate(source, generated)

            def loss(self, source, target):
                generated, generated_logits = self.forward(source)
                real_logits = self.discriminate(source, target)
                discriminator_loss = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                ) + F.binary_cross_entropy_with_logits(
                    generated_logits.detach(),
                    torch.zeros_like(generated_logits),
                )
                adversarial = F.binary_cross_entropy_with_logits(
                    generated_logits,
                    torch.ones_like(generated_logits),
                )
                generator_loss = adversarial + self.l1_weight * F.l1_loss(
                    generated, target
                )
                return discriminator_loss, generator_loss

            def update(self, source, target):
                self.train()
                self.discriminator.requires_grad_(True)
                generated = self.generator(source).detach()
                real_logits = self.discriminate(source, target)
                fake_logits = self.discriminate(source, generated)
                discriminator_loss = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                ) + F.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                self.discriminator.requires_grad_(False)
                generated, generated_logits = self.forward(source)
                adversarial = F.binary_cross_entropy_with_logits(
                    generated_logits,
                    torch.ones_like(generated_logits),
                )
                generator_loss = adversarial + self.l1_weight * F.l1_loss(
                    generated, target
                )
                self.generator_optimizer.zero_grad(set_to_none=True)
                generator_loss.backward()
                self.generator_optimizer.step()
                self.discriminator.requires_grad_(True)
                return float(discriminator_loss.detach()), float(
                    generator_loss.detach()
                )

            def evaluate(self, source, target):
                self.eval()
                with torch.no_grad():
                    generated, logits = self.forward(source)
                return {
                    "mae": float(F.l1_loss(generated, target)),
                    "mean_realism": float(logits.sigmoid().mean()),
                    "generated_std": float(generated.std()),
                    "patch_grid": tuple(logits.shape[-2:]),
                }


        spatial_pair_ids = torch.arange(80, device=DEVICE)
        spatial_source = images_01[spatial_pair_ids]
        local_average = F.avg_pool2d(spatial_source, 3, stride=1, padding=1)
        spatial_target = (0.25 * spatial_source + 0.75 * local_average).clamp(0.0, 1.0)
        spatial_train_ids = spatial_pair_ids[:64]
        spatial_eval_ids = spatial_pair_ids[64:]
        assert not torch.isin(spatial_train_ids, spatial_eval_ids).any()
        spatial_train_source = spatial_source[:64]
        spatial_train_target = spatial_target[:64]
        spatial_eval_source = spatial_source[64:]
        spatial_eval_target = spatial_target[64:]

        portfolio_model = Pix2pixLab(None, None, None, None)
        """
    ),
    "cyclegan": (
        r"""
        class CycleConsistentTranslationLab(nn.Module):
            def __init__(
                self,
                generator_ab,
                generator_ba,
                discriminator_a,
                discriminator_b,
                generator_optimizer,
                discriminator_optimizer,
            ):
                super().__init__()
                self.generator_ab = generator_ab
                self.generator_ba = generator_ba
                self.discriminator_a = discriminator_a
                self.discriminator_b = discriminator_b
                self.generator_optimizer = generator_optimizer
                self.discriminator_optimizer = discriminator_optimizer
                self.cycle_weight = 8.0

            def config(self):
                return {"domain_dim": 2, "cycle_weight": self.cycle_weight}

            def forward(self, domain_a_batch, domain_b_batch):
                fake_b = self.generator_ab(domain_a_batch)
                fake_a = self.generator_ba(domain_b_batch)
                cycle_a = self.generator_ba(fake_b)
                cycle_b = self.generator_ab(fake_a)
                return fake_a, fake_b, cycle_a, cycle_b

            def loss(self, domain_a_batch, domain_b_batch):
                fake_a, fake_b, cycle_a, cycle_b = self.forward(
                    domain_a_batch,
                    domain_b_batch,
                )
                cycle = F.l1_loss(cycle_a, domain_a_batch)
                cycle = cycle + F.l1_loss(cycle_b, domain_b_batch)
                adversarial = F.binary_cross_entropy_with_logits(
                    self.discriminator_b(fake_b),
                    torch.ones_like(self.discriminator_b(fake_b)),
                )
                adversarial = adversarial + F.binary_cross_entropy_with_logits(
                    self.discriminator_a(fake_a),
                    torch.ones_like(self.discriminator_a(fake_a)),
                )
                return adversarial + self.cycle_weight * cycle, cycle

            def update(self, domain_a_batch, domain_b_batch):
                with torch.no_grad():
                    detached_fake_a = self.generator_ba(domain_b_batch)
                    detached_fake_b = self.generator_ab(domain_a_batch)
                discriminator_loss = F.binary_cross_entropy_with_logits(
                    self.discriminator_a(domain_a_batch),
                    torch.ones_like(self.discriminator_a(domain_a_batch)),
                )
                discriminator_loss = discriminator_loss + F.binary_cross_entropy_with_logits(
                    self.discriminator_a(detached_fake_a),
                    torch.zeros_like(self.discriminator_a(detached_fake_a)),
                )
                discriminator_loss = discriminator_loss + F.binary_cross_entropy_with_logits(
                    self.discriminator_b(domain_b_batch),
                    torch.ones_like(self.discriminator_b(domain_b_batch)),
                )
                discriminator_loss = discriminator_loss + F.binary_cross_entropy_with_logits(
                    self.discriminator_b(detached_fake_b),
                    torch.zeros_like(self.discriminator_b(detached_fake_b)),
                )
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                generator_loss, cycle = self.loss(domain_a_batch, domain_b_batch)
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()
                return (
                    float(discriminator_loss.detach()),
                    float(generator_loss.detach()),
                    float(cycle.detach()),
                )

            def evaluate(self, domain_a_batch, domain_b_batch):
                with torch.no_grad():
                    fake_a, fake_b, cycle_a, cycle_b = self.forward(
                        domain_a_batch,
                        domain_b_batch,
                    )
                    cycle = F.l1_loss(cycle_a, domain_a_batch)
                    cycle = cycle + F.l1_loss(cycle_b, domain_b_batch)
                return {
                    "cycle_l1": float(cycle),
                    "fake_a_shape": tuple(fake_a.shape),
                    "fake_b_shape": tuple(fake_b.shape),
                }


        portfolio_model = CycleConsistentTranslationLab(
            G_AB,
            G_BA,
            D_A,
            D_B,
            opt_g,
            opt_d,
        )
        assert portfolio_model.config()["cycle_weight"] == 8.0
        """
    ),
    "ddpm": (
        r"""
        class DiffusionProcessLab(nn.Module):
            def __init__(self, noise_predictor, optimizer, beta_schedule):
                super().__init__()
                self.noise_predictor = noise_predictor
                self.optimizer = optimizer
                self.register_buffer("beta_schedule", beta_schedule)
                alpha_schedule = 1.0 - beta_schedule
                self.register_buffer("alpha_schedule", alpha_schedule)
                self.register_buffer("alpha_bar", torch.cumprod(alpha_schedule, dim=0))

            def config(self):
                return {"timesteps": len(self.beta_schedule), "data_dim": 64}

            def forward(self, clean, timestep, noise):
                signal = self.alpha_bar[timestep].sqrt().reshape(-1, 1)
                noise_scale = (1.0 - self.alpha_bar[timestep]).sqrt().reshape(-1, 1)
                noisy = signal * clean + noise_scale * noise
                predicted_noise = self.predict_noise(noisy, timestep)
                return noisy, predicted_noise

            def predict_noise(self, noisy, timestep):
                time_feature = timestep[:, None] / (len(self.beta_schedule) - 1)
                return self.noise_predictor(
                    torch.cat([noisy, time_feature], dim=1)
                )

            def loss(self, clean, timestep, noise):
                _, predicted_noise = self.forward(clean, timestep, noise)
                return (predicted_noise - noise).square().mean()

            def update(self, clean, timestep, noise):
                objective = self.loss(clean, timestep, noise)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def reverse_step(self, noisy, step, generator):
                timestep = torch.full(
                    (len(noisy),),
                    step,
                    dtype=torch.long,
                    device=noisy.device,
                )
                predicted_noise = self.predict_noise(noisy, timestep)
                mean = noisy - self.beta_schedule[step] * predicted_noise / torch.sqrt(
                    1.0 - self.alpha_bar[step]
                )
                mean = mean / torch.sqrt(self.alpha_schedule[step])
                if step == 0:
                    return mean
                innovation = randn_device(noisy.shape, generator=generator)
                return mean + self.beta_schedule[step].sqrt() * innovation

            def sample(self, batch_size, generator):
                state = randn_device(
                    (batch_size, self.config()["data_dim"]),
                    generator=generator,
                )
                trajectory = [state.detach().clone()]
                with torch.no_grad():
                    for step in reversed(range(len(self.beta_schedule))):
                        state = self.reverse_step(state, step, generator)
                        trajectory.append(state.detach().clone())
                return state, torch.stack(trajectory)

            def evaluate(self, clean, timestep, noise):
                with torch.no_grad():
                    noisy, predicted_noise = self.forward(clean, timestep, noise)
                    mse = (predicted_noise - noise).square().mean()
                    sample, trajectory = self.sample(
                        min(8, len(clean)),
                        torch.Generator().manual_seed(4071),
                    )
                return {
                    "noise_mse": float(mse),
                    "noisy_shape": tuple(noisy.shape),
                    "sample_shape": tuple(sample.shape),
                    "trajectory_steps": len(trajectory),
                    "sample_abs_mean": float(sample.abs().mean()),
                }


        portfolio_model = DiffusionProcessLab(denoiser, opt, beta)
        assert portfolio_model.config()["timesteps"] == T
        """
    ),
    "score_sde": (
        r"""
        class ScoreSdeLab(nn.Module):
            def __init__(self, score_network, optimizer):
                super().__init__()
                self.score_network = score_network
                self.optimizer = optimizer

            def config(self):
                return {"state_dim": 2, "beta_min": 0.1, "beta_max": 10.0}

            def marginal(self, clean, time, noise):
                integrated_beta = 0.1 * time + 4.95 * time.square()
                mean_coefficient = torch.exp(-0.5 * integrated_beta)
                standard_deviation = torch.sqrt(
                    (1.0 - mean_coefficient.square()).clamp_min(1e-5)
                )
                noisy = mean_coefficient[:, None] * clean
                noisy = noisy + standard_deviation[:, None] * noise
                target_score = -noise / standard_deviation[:, None]
                return noisy, target_score, standard_deviation

            def forward(self, noisy, time):
                return self.score_network(torch.cat([noisy, time[:, None]], dim=1))

            def loss(self, clean, time, noise):
                noisy, target_score, standard_deviation = self.marginal(
                    clean, time, noise
                )
                predicted_score = self.forward(noisy, time)
                squared_error = (predicted_score - target_score).square().sum(dim=1)
                return (squared_error * standard_deviation.square()).mean()

            def update(self, clean, time, noise):
                objective = self.loss(clean, time, noise)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def reverse_euler_maruyama(self, batch_size, steps, generator):
                state = randn_device(
                    (batch_size, self.config()["state_dim"]),
                    generator=generator,
                )
                time_grid = torch.linspace(1.0, 0.02, steps + 1, device=state.device)
                trajectory = [state.detach().clone()]
                with torch.no_grad():
                    for index in range(steps):
                        time = time_grid[index]
                        delta_time = time_grid[index + 1] - time
                        beta = self.config()["beta_min"]
                        beta = beta + (
                            self.config()["beta_max"] - self.config()["beta_min"]
                        ) * time
                        time_batch = time.expand(batch_size)
                        score = self.forward(state, time_batch)
                        reverse_drift = -0.5 * beta * state - beta * score
                        innovation = randn_device(state.shape, generator=generator)
                        state = state + reverse_drift * delta_time
                        state = state + torch.sqrt(beta * -delta_time) * innovation
                        trajectory.append(state.detach().clone())
                return state, torch.stack(trajectory)

            def evaluate(self, clean, time, noise):
                with torch.no_grad():
                    noisy, target_score, _ = self.marginal(clean, time, noise)
                    predicted_score = self.forward(noisy, time)
                    mse = F.mse_loss(predicted_score, target_score)
                    sample, trajectory = self.reverse_euler_maruyama(
                        128,
                        32,
                        torch.Generator().manual_seed(4081),
                    )
                return {
                    "score_mse": float(mse),
                    "score_shape": tuple(predicted_score.shape),
                    "sample_shape": tuple(sample.shape),
                    "trajectory_steps": len(trajectory),
                    "sample_radius_mean": float(sample.norm(dim=1).mean()),
                }


        portfolio_model = ScoreSdeLab(score, opt)
        assert portfolio_model.config()["state_dim"] == 2
        """
    ),
    "latent_diffusion": (
        r"""
        class LatentDiffusionLab(nn.Module):
            def __init__(self, encoder, decoder, noise_predictor, optimizer):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.noise_predictor = noise_predictor
                self.optimizer = optimizer
                self.register_buffer("beta_schedule", bz)
                self.register_buffer("alpha_schedule", 1.0 - bz)
                self.register_buffer(
                    "alpha_bar",
                    torch.cumprod(1.0 - bz, dim=0),
                )
                self.register_buffer("latent_location", latent_mean)
                self.register_buffer("latent_scale", latent_std)

            def config(self):
                return {
                    "pixel_dim": 64,
                    "latent_dim": 8,
                    "timesteps": len(self.beta_schedule),
                }

            def forward(self, clean, timestep, noise):
                with torch.no_grad():
                    encoded = self.encoder(clean)
                    normalized = (encoded - self.latent_location) / self.latent_scale
                signal = self.alpha_bar[timestep, None].sqrt()
                noise_scale = (1.0 - self.alpha_bar[timestep, None]).sqrt()
                noisy_latent = signal * normalized + noise_scale * noise
                predicted_noise = self.predict_noise(noisy_latent, timestep)
                return noisy_latent, predicted_noise

            def predict_noise(self, noisy_latent, timestep):
                time_feature = timestep[:, None] / (len(self.beta_schedule) - 1)
                return self.noise_predictor(
                    torch.cat([noisy_latent, time_feature], dim=1)
                )

            def loss(self, clean, timestep, noise):
                _, predicted_noise = self.forward(clean, timestep, noise)
                return F.mse_loss(predicted_noise, noise)

            def update(self, clean, timestep, noise):
                objective = self.loss(clean, timestep, noise)
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                return float(objective.detach())

            def reverse_latent_step(self, noisy_latent, step, generator):
                timestep = torch.full(
                    (len(noisy_latent),),
                    step,
                    dtype=torch.long,
                    device=noisy_latent.device,
                )
                predicted_noise = self.predict_noise(noisy_latent, timestep)
                mean = noisy_latent - self.beta_schedule[step] * predicted_noise / torch.sqrt(
                    1.0 - self.alpha_bar[step]
                )
                mean = mean / torch.sqrt(self.alpha_schedule[step])
                if step == 0:
                    return mean
                innovation = randn_device(noisy_latent.shape, generator=generator)
                return mean + self.beta_schedule[step].sqrt() * innovation

            def sample(self, batch_size, generator):
                latent = randn_device(
                    (batch_size, self.config()["latent_dim"]),
                    generator=generator,
                )
                trajectory = [latent.detach().clone()]
                with torch.no_grad():
                    for step in reversed(range(len(self.beta_schedule))):
                        latent = self.reverse_latent_step(latent, step, generator)
                        trajectory.append(latent.detach().clone())
                    decoded = self.decoder(
                        latent * self.latent_scale + self.latent_location
                    )
                return decoded, latent, torch.stack(trajectory)

            def evaluate(self, clean):
                with torch.no_grad():
                    reconstruction = self.decoder(self.encoder(clean))
                    reconstruction_mse = F.mse_loss(reconstruction, clean)
                    sample, sampled_latent, trajectory = self.sample(
                        min(8, len(clean)),
                        torch.Generator().manual_seed(4091),
                    )
                return {
                    "reconstruction_mse": float(reconstruction_mse),
                    "compression_ratio": self.config()["pixel_dim"]
                    / self.config()["latent_dim"],
                    "sample_shape": tuple(sample.shape),
                    "sampled_latent_shape": tuple(sampled_latent.shape),
                    "trajectory_steps": len(trajectory),
                    "sample_pixel_mean": float(sample.mean()),
                }


        portfolio_model = LatentDiffusionLab(E, Dec, eps_model, opt)
        assert portfolio_model.config()["latent_dim"] == 8
        """
    ),
}


_PORTFOLIO_RUNS = {
    "stacked_denoising_autoencoder": r"""
        evidence_rng = torch.Generator().manual_seed(3001)
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        integrated_loss = portfolio_model.update(flat, evidence_rng)
        metrics = portfolio_model.evaluate(
            flat,
            torch.Generator().manual_seed(3002),
        )
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert math.isfinite(integrated_loss) and parameter_delta > 0.0
        assert math.isfinite(metrics["restored_mse"])
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "vae": r"""
        evidence_batch = points[:128]
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        integrated_loss = portfolio_model.update(evidence_batch)
        metrics = portfolio_model.evaluate(evidence_batch)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert math.isfinite(integrated_loss) and parameter_delta > 0.0
        assert math.isfinite(metrics["mse"]) and metrics["mean_kl"] >= 0.0
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "gan": r"""
        discriminator_noise = randn_device(
            (128, 2), generator=torch.Generator().manual_seed(3201)
        )
        generator_noise = randn_device(
            (128, 2), generator=torch.Generator().manual_seed(3202)
        )
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        discriminator_loss, generator_loss = portfolio_model.update(
            points[:128], discriminator_noise, generator_noise
        )
        metrics = portfolio_model.evaluate(points, fixed_z)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(generator_loss)
        assert math.isfinite(discriminator_loss) and math.isfinite(metrics["mean_gap"])
        print({"g_loss": generator_loss, "delta": parameter_delta, **metrics})
    """,
    "dcgan": r"""
        discriminator_noise = randn_device(
            (64, 8, 1, 1), generator=torch.Generator().manual_seed(3301)
        )
        generator_noise = randn_device(
            (64, 8, 1, 1), generator=torch.Generator().manual_seed(3302)
        )
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        discriminator_loss, generator_loss = portfolio_model.update(
            real_images[:64], discriminator_noise, generator_noise
        )
        metrics = portfolio_model.evaluate(generator_noise[:8])
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and metrics["sample_shape"] == (8, 1, 8, 8)
        assert math.isfinite(discriminator_loss) and math.isfinite(generator_loss)
        print({"g_loss": generator_loss, "delta": parameter_delta, **metrics})
    """,
    "wgan": r"""
        critic_noises = [
            randn_device((128, 2), generator=torch.Generator().manual_seed(seed))
            for seed in range(3401, 3404)
        ]
        generator_noise = randn_device(
            (128, 2), generator=torch.Generator().manual_seed(3404)
        )
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        critic_loss, generator_loss = portfolio_model.update(
            points[:128], critic_noises, generator_noise
        )
        metrics = portfolio_model.evaluate(points, fixed_z)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(metrics["critic_gap"])
        assert math.isfinite(critic_loss) and math.isfinite(generator_loss)
        print({"critic_loss": critic_loss, "delta": parameter_delta, **metrics})
    """,
    "pix2pix": r"""
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        discriminator_loss, generator_loss = portfolio_model.update(
            spatial_train_source[:16], spatial_train_target[:16]
        )
        metrics = portfolio_model.evaluate(spatial_eval_source, spatial_eval_target)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(metrics["mae"])
        assert metrics["patch_grid"] == (2, 2)
        assert math.isfinite(discriminator_loss) and math.isfinite(generator_loss)
        print({"g_loss": generator_loss, "delta": parameter_delta, **metrics})
    """,
    "cyclegan": r"""
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        generator_loss, cycle_loss_value = portfolio_model.update(
            domain_a[:192], domain_b[:192]
        )
        metrics = portfolio_model.evaluate(domain_a, domain_b)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(metrics["cycle_l1"])
        assert math.isfinite(generator_loss) and math.isfinite(cycle_loss_value)
        print({"g_loss": generator_loss, "delta": parameter_delta, **metrics})
    """,
    "ddpm": r"""
        evidence_batch = x0[:96]
        evidence_timestep = randint_device(
            T, (len(evidence_batch),), generator=torch.Generator().manual_seed(3701)
        )
        evidence_noise = randn_device(
            evidence_batch.shape, generator=torch.Generator().manual_seed(3702)
        )
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        integrated_loss = portfolio_model.update(
            evidence_batch, evidence_timestep, evidence_noise
        )
        metrics = portfolio_model.evaluate(
            evidence_batch, evidence_timestep, evidence_noise
        )
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert math.isfinite(metrics["noise_mse"])
        assert metrics["sample_shape"] == (8, 64)
        assert metrics["trajectory_steps"] == T + 1
        assert math.isfinite(metrics["sample_abs_mean"])
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "score_sde": r"""
        evidence_batch = points[:160]
        evidence_time = rand_device(
            (len(evidence_batch),), generator=torch.Generator().manual_seed(3801)
        )
        evidence_time = evidence_time * 0.98 + 0.01
        evidence_noise = randn_device(
            evidence_batch.shape, generator=torch.Generator().manual_seed(3802)
        )
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        integrated_loss = portfolio_model.update(
            evidence_batch, evidence_time, evidence_noise
        )
        metrics = portfolio_model.evaluate(
            evidence_batch, evidence_time, evidence_noise
        )
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert math.isfinite(metrics["score_mse"])
        assert metrics["sample_shape"] == (128, 2)
        assert metrics["trajectory_steps"] == 33
        assert math.isfinite(metrics["sample_radius_mean"])
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
    "latent_diffusion": r"""
        evidence_batch = x_data[:96]
        evidence_timestep = randint_device(
            Tz, (len(evidence_batch),), generator=torch.Generator().manual_seed(3901)
        )
        evidence_noise = randn_device(
            (len(evidence_batch), 8), generator=torch.Generator().manual_seed(3902)
        )
        before = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        integrated_loss = portfolio_model.update(
            evidence_batch, evidence_timestep, evidence_noise
        )
        metrics = portfolio_model.evaluate(evidence_batch)
        after = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in portfolio_model.parameters()]
        )
        parameter_delta = float((after - before).abs().sum())
        assert parameter_delta > 0.0 and math.isfinite(integrated_loss)
        assert math.isfinite(metrics["reconstruction_mse"])
        assert metrics["sample_shape"] == (8, 64)
        assert metrics["sampled_latent_shape"] == (8, 8)
        assert metrics["trajectory_steps"] == Tz + 1
        assert math.isfinite(metrics["sample_pixel_mean"])
        print({"loss": integrated_loss, "delta": parameter_delta, **metrics})
    """,
}


def _portfolio_cell(spec: FieldPaperSpec) -> object:
    class_source, _ = _split_portfolio_solution(spec)
    documented_tree = _with_contract_docstrings(spec, class_source)
    documented_source = ast.unparse(documented_tree)
    return code(
        _exercise_architecture(spec, class_source),
        documented_source,
        "portfolio-architecture",
    )


def _split_portfolio_solution(spec: FieldPaperSpec) -> tuple[str, str]:
    source = dedent(_PORTFOLIO_SOLUTIONS[spec.slug]).strip()
    lines = source.splitlines()
    verification_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("portfolio_model = ")
    )
    class_source = "\n".join(lines[:verification_index])
    verification_source = "\n".join(lines[verification_index:])
    return class_source, verification_source


def _portfolio_verification_cell(spec: FieldPaperSpec) -> object:
    history_name = "diff_history" if spec.slug == "latent_diffusion" else "history"
    exercise = f"""
    # TODO 5: {_PORTFOLIO_CLASS_NAMES[spec.slug]} 기반 학습의 최종 증거를 검증하세요.
    # 계약: {history_name}, portfolio_metrics, parameter_delta를 모두 검사합니다.
    raise NotImplementedError
    """
    solution = f"""
    config_contract = portfolio_model.config()
    numeric_metrics = [
        float(value)
        for value in portfolio_metrics.values()
        if isinstance(value, (int, float))
    ]
    assert config_contract and len({history_name}) > 0
    assert np.isfinite(np.asarray({history_name}, dtype=float)).all()
    assert parameter_delta > 0.0
    assert numeric_metrics and np.isfinite(numeric_metrics).all()
    print(
        {{
            "class": type(portfolio_model).__name__,
            "steps": len({history_name}),
            "parameter_delta": parameter_delta,
            "metrics": portfolio_metrics,
        }}
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
        ### 단계 5. 통합 클래스가 실제 update와 evaluate를 소유하는지 검증

        - **논문 위치:** `{paper_part}`의 **{lab_part}**
        - **핵심 식:**

          $$
          {guide["equation"]}
          $$

        - **입출력 shape:** `{guide["shape"]}`
        - **구현 이유:** 앞 셀은 수식을 작은 연산으로 분해해 확인하고,
          이 셀은 같은 구성 요소를 `{_PORTFOLIO_CLASS_NAMES[spec.slug]}`에 주입해
          최종 학습·평가 경로를 하나로 묶습니다. {evidence}
        - **완료 증거:** `update(...)`가 유한한 loss를 반환하고,
          `parameter_delta > 0`이며, `evaluate(...)`의 논문 대응 지표가 유한해야 합니다.
          `config()`만 호출하는 것은 실행 재현으로 보지 않습니다.
        """,
        "integration-explanation",
        "portfolio-verification",
        "equation",
    )


_TRAINING_LOOP_MARKERS = {
    "stacked_denoising_autoencoder": "for _ in range(100):",
    "vae": "for _ in range(150):",
    "gan": "for _ in range(220):",
    "dcgan": "for _ in range(50):",
    "wgan": "for _ in range(140):",
    "pix2pix": "for _ in range(180):",
    "cyclegan": "for _ in range(220):",
    "ddpm": "for _ in range(180):",
    "score_sde": "for _ in range(220):",
    "latent_diffusion": "for _ in range(180):",
}


_INTEGRATED_TRAINING_DRIVERS = {
    "stacked_denoising_autoencoder": r"""
        portfolio_model = DenoisingAutoencoderLab(model, optimizer)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        for _ in range(100):
            history.append(portfolio_model.update(flat, train_rng))
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert history[-1] < initial and parameter_delta > 0.0
        assert np.isfinite(history).all()
    """,
    "vae": r"""
        portfolio_model = VariationalAutoencoderLab(model, optimizer)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        for _ in range(150):
            history.append(portfolio_model.update(x_train))
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in model.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert history[-1] < initial and parameter_delta > 0.0
        assert np.isfinite(history).all()
    """,
    "gan": r"""
        portfolio_model = AdversarialGameLab(G, D, opt_g, opt_d)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G.parameters()]
        )
        for _ in range(220):
            index = randint_device(len(points), (96,), generator=rng)
            discriminator_noise = randn_device((96, 2), generator=rng)
            generator_noise = randn_device((96, 2), generator=rng)
            history.append(
                portfolio_model.update(
                    points[index], discriminator_noise, generator_noise
                )
            )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "dcgan": r"""
        portfolio_model = ConvolutionalGanLab(G, D, og, od)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G.parameters()]
        )
        for _ in range(50):
            index = randint_device(len(real_images), (64,), generator=rng)
            discriminator_noise = randn_device((64, 8, 1, 1), generator=rng)
            generator_noise = randn_device((64, 8, 1, 1), generator=rng)
            history.append(
                portfolio_model.update(
                    real_images[index], discriminator_noise, generator_noise
                )
            )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "wgan": r"""
        portfolio_model = WassersteinGanLab(G, C, og, oc)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G.parameters()]
        )
        for _ in range(140):
            real = points[randint_device(len(points), (96,), generator=rng)]
            critic_noises = [
                randn_device((96, 2), generator=rng) for _ in range(3)
            ]
            generator_noise = randn_device((96, 2), generator=rng)
            history.append(
                portfolio_model.update(real, critic_noises, generator_noise)
            )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "pix2pix": r"""
        initial_parameters = torch.cat(
            [
                parameter.detach().flatten().cpu()
                for parameter in portfolio_model.generator.parameters()
            ]
        )
        for _ in range(48):
            index = randint_device(
                len(spatial_train_source),
                (16,),
                generator=rng,
            )
            history.append(
                portfolio_model.update(
                    spatial_train_source[index],
                    spatial_train_target[index],
                )
            )
        final_parameters = torch.cat(
            [
                parameter.detach().flatten().cpu()
                for parameter in portfolio_model.generator.parameters()
            ]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "cyclegan": r"""
        portfolio_model = CycleConsistentTranslationLab(
            G_AB,
            G_BA,
            D_A,
            D_B,
            opt_g,
            opt_d,
        )
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G_AB.parameters()]
        )
        for _ in range(220):
            index_a = randint_device(len(domain_a), (96,), generator=rng)
            index_b = randint_device(len(domain_b), (96,), generator=rng)
            history.append(
                portfolio_model.update(domain_a[index_a], domain_b[index_b])
            )
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in G_AB.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
    """,
    "ddpm": r"""
        portfolio_model = DiffusionProcessLab(denoiser, opt, beta)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in denoiser.parameters()]
        )
        for _ in range(180):
            timestep = randint_device(T, (len(x0),), generator=rng)
            noise = randn_device(x0.shape, generator=rng)
            history.append(portfolio_model.update(x0, timestep, noise))
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in denoiser.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert history[-1] < initial and parameter_delta > 0.0
        assert np.isfinite(history).all()
    """,
    "score_sde": r"""
        portfolio_model = ScoreSdeLab(score, opt)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in score.parameters()]
        )
        for _ in range(220):
            index = randint_device(len(points), (192,), generator=rng)
            time = 0.02 + 0.96 * rand_device((192,), generator=rng)
            noise = randn_device((192, 2), generator=rng)
            history.append(portfolio_model.update(points[index], time, noise))
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in score.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(history).all()
        assert np.mean(history[-20:]) < np.mean(history[:20])
    """,
    "latent_diffusion": r"""
        portfolio_model = LatentDiffusionLab(E, Dec, eps_model, opt)
        initial_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in eps_model.parameters()]
        )
        for _ in range(180):
            timestep = randint_device(Tz, (len(latent),), generator=rng)
            noise = randn_device(latent.shape, generator=rng)
            diff_history.append(portfolio_model.update(x_data, timestep, noise))
        final_parameters = torch.cat(
            [parameter.detach().flatten().cpu() for parameter in eps_model.parameters()]
        )
        parameter_delta = float((final_parameters - initial_parameters).abs().sum())
        assert parameter_delta > 0.0 and np.isfinite(diff_history).all()
        assert np.mean(diff_history[-20:]) < np.mean(diff_history[:20])
    """,
}


_EVALUATION_CALLS = {
    "stacked_denoising_autoencoder": r"""
        portfolio_metrics = portfolio_model.evaluate(
            flat,
            torch.Generator().manual_seed(303),
        )
    """,
    "vae": "portfolio_metrics = portfolio_model.evaluate(points[480:])",
    "gan": "portfolio_metrics = portfolio_model.evaluate(points, fixed_z)",
    "dcgan": r"""
        evaluation_noise = randn_device(
            (8, 8, 1, 1), generator=torch.Generator().manual_seed(333)
        )
        portfolio_metrics = portfolio_model.evaluate(evaluation_noise)
    """,
    "wgan": "portfolio_metrics = portfolio_model.evaluate(points, fixed_z)",
    "pix2pix": r"""
        portfolio_metrics = portfolio_model.evaluate(
            spatial_eval_source,
            spatial_eval_target,
        )
        assert portfolio_metrics["patch_grid"] == (2, 2)
    """,
    "cyclegan": r"""
        portfolio_metrics = portfolio_model.evaluate(domain_a, domain_b)
    """,
    "ddpm": r"""
        evaluation_timestep = torch.zeros(len(x0), dtype=torch.long, device=DEVICE)
        evaluation_noise = randn_device(
            x0.shape, generator=torch.Generator().manual_seed(373)
        )
        portfolio_metrics = portfolio_model.evaluate(
            x0, evaluation_timestep, evaluation_noise
        )
    """,
    "score_sde": r"""
        evaluation_time = torch.full((len(points),), 0.5, device=DEVICE)
        evaluation_noise = randn_device(
            points.shape, generator=torch.Generator().manual_seed(383)
        )
        portfolio_metrics = portfolio_model.evaluate(
            points, evaluation_time, evaluation_noise
        )
    """,
    "latent_diffusion": "portfolio_metrics = portfolio_model.evaluate(x_data)",
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
    "stacked_denoising_autoencoder": (0, 1, 1),
    "vae": (2, 1, 0),
    "gan": (0, 1, 2),
    "dcgan": (0, 2, 2),
    "wgan": (1, 2, 0),
    "pix2pix": (2, 1, 1),
    "cyclegan": (1, 2, 1),
    "ddpm": (0, 2, 2),
    "score_sde": (2, 1, 0),
    "latent_diffusion": (0, 1, 2),
}


def _stage_markdown(spec: FieldPaperSpec, stage_index: int, code_tag: str) -> object:
    mapping_index = _STAGE_MAPPING_ORDER[spec.slug][stage_index]
    paper_part, lab_part, evidence = spec.mappings[mapping_index]
    stage_names = ("핵심 연산 검증", "학습·업데이트", "평가·시각화")
    evaluation_scope = ""
    if stage_index == 2:
        evaluation_scope = (
            "\n- **평가 범위:** 이 값은 작은 학습 batch의 메커니즘 진단이며, "
            "held-out 생성 품질이나 원 논문의 benchmark 성능으로 해석하지 않습니다."
        )
    return markdown(
        f"""
        ### 단계 {stage_index + 2}. {stage_names[stage_index]}

        - **논문의 어느 부분인가:** `{paper_part}` — **{lab_part}**
        - **구현 이유:** {evidence}
        - **읽을 코드:** 아래 `{code_tag}` 셀에서 중간 tensor의 shape, gradient가 흐르는
          경로와 `detach` 또는 `no_grad` 경계를 확인합니다.

        실행 결과만 보지 말고, 이 단계의 입력과 출력이 앞선 수식의 어떤 기호인지 먼저
        주석으로 적은 뒤 실습하세요.
        - **완료 증거:** 아래 셀의 `assert`와 출력 metric이 shape·loss 계약을 검사합니다.
        {evaluation_scope}
        """,
        "step-explanation",
        code_tag,
    )


def _upgrade_for_portfolio(spec: FieldPaperSpec) -> FieldPaperSpec:
    intro, setup, *algorithm_cells = spec.cells
    staged_cells: list[object] = []
    code_index = 0
    for cell in algorithm_cells:
        if cell.cell_type == "code":
            if code_index == 0:
                cell = _exercise_with_missing_public_api(spec, cell)
            elif code_index == 1:
                cell = _integrated_training_cell(spec, cell)
            elif code_index == 2:
                cell = _integrated_evaluation_cell(spec, cell)
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
        SDAE,
        VAE,
        GAN,
        DCGAN,
        WGAN,
        PIX2PIX,
        CYCLEGAN,
        DDPM,
        SCORE_SDE,
        LATENT_DIFFUSION,
    )
)

assert len(SPECS) == 10
assert [spec.number for spec in SPECS] == list(range(10))
