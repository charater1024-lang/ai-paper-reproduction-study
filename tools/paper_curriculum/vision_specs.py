"""Paired mini-reproductions for six landmark computer-vision papers."""

from __future__ import annotations

from .common import PaperSpec, code, markdown, shared_code

LENET5 = PaperSpec(
    number=0,
    slug="lenet5",
    short_title="LeNet-5",
    paper_title="Gradient-Based Learning Applied to Document Recognition",
    authors="Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner",
    year=1998,
    primary_url="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
    venue="Proceedings of the IEEE 86(11)",
    difficulty="입문",
    expected_minutes=70,
    prerequisites="PyTorch tensor, Conv2d, pooling, cross-entropy의 기초",
    reproduction_goal=(
        "32×32 입력에서 C1-S2-C3-S4-C5-F6으로 이어지는 LeNet-5의 공간 크기 변화를 "
        "재현하고, 공유 합성곱 필터와 평균 subsampling을 사용한 작은 분류기를 합성 패턴에 "
        "학습한다. 각 층의 shape, 유한한 loss, 한 번 이상의 loss 감소를 검증한다."
    ),
    original_scale=(
        "원 논문은 MNIST 및 문서 인식 파이프라인, 부분 연결 C3, 학습 가능한 subsampling 계수, "
        "84차원 F6과 Euclidean RBF 출력까지 사용한다. 여기서는 다운로드 없는 20개 합성 영상, "
        "완전 연결 C3, AvgPool2d, 10-way linear logits로 핵심 계층 구조만 재현한다."
    ),
    mappings=(
        (
            "§II.B ‘LeNet-5’와 Fig. 2: 32×32 → C1(6@28²) → S2(6@14²)",
            "`LeNet5Mini.c1`, `s2`와 shape 추적 셀",
            "assert가 `(6,28,28)`과 `(6,14,14)`를 직접 확인",
        ),
        (
            "§II.B, Fig. 2: C3(16@10²) → S4(16@5²) → C5(120@1²)",
            "`c3`, `s4`, `c5` 및 중간 활성 시각화",
            "5×5 valid convolution과 2×2 평균 subsampling의 출력 크기 검증",
        ),
        (
            "§II.B ‘LeNet-5’: tanh 비선형성과 F6의 84 units",
            "`torch.tanh`와 `f6 = nn.Linear(120, 84)`",
            "파라미터/출력 shape와 10-class logits 검증",
        ),
        (
            "§II.A ‘Convolutional Networks’: local receptive fields와 shared weights",
            "C1 feature-map 시각화와 합성 위치 패턴 학습",
            "동일 Conv2d kernel이 전체 영상 위치에 적용되는 결과를 관찰",
        ),
    ),
    cells=(
        markdown(
            """
            ## 1. 논문 구조를 읽는 법

            Fig. 2에서 `Cx`는 학습되는 합성곱 층, `Sx`는 해상도를 절반으로 줄이는
            subsampling 층입니다. 원 논문의 S2/S4는 2×2 평균에 학습 가능한 계수와 bias를
            붙였지만, 이 미니 재현에서는 평균 연산 자체에 초점을 맞춰 `AvgPool2d`를 씁니다.
            또한 원 C3의 **부분 연결표**는 완전 연결 합성곱으로 단순화합니다. 그러므로 여기서
            확인할 것은 최종 MNIST 정확도가 아니라 Fig. 2의 receptive-field/shape 흐름입니다.
            """,
            "paper-reading",
        ),
        shared_code(
            """
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            from torch.nn import functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(0)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device

            def make_position_patterns(repeats=2, size=32):
                # 10개 위치를 서로 다른 class로 쓰는 다운로드 없는 toy digits.
                generator = torch.Generator().manual_seed(101)
                images, labels = [], []
                for repeat in range(repeats):
                    for label in range(10):
                        image = torch.zeros(1, size, size)
                        row = 4 + (label // 5) * 14
                        col = 2 + (label % 5) * 6
                        image[:, row:row + 7, col:col + 5] = 1.0
                        # class마다 작은 가로 획을 더해 단순 위치 복사 이상의 패턴을 만든다.
                        image[:, row + label % 4:row + label % 4 + 2, col:col + 8] = 0.55
                        noise = 0.04 * torch.randn(image.shape, generator=generator)
                        images.append((image + noise).clamp(0, 1))
                        labels.append(label)
                return torch.stack(images), torch.tensor(labels)

            lenet_x, lenet_y = make_position_patterns()
            lenet_x, lenet_y = ACCELERATOR.move(lenet_x, lenet_y)
            assert lenet_x.shape == (20, 1, 32, 32)
            assert lenet_y.tolist() == list(range(10)) * 2
            print(ACCELERATOR.summary())
            print("synthetic batch:", tuple(lenet_x.shape))
            """,
            "setup",
            "synthetic-data",
        ),
        code(
            """
            # TODO: Fig. 2를 따라 C1-S2-C3-S4-C5-F6을 정의하세요.
            # 힌트: valid 5×5 Conv, 2×2 AvgPool, tanh를 사용합니다.
            class LeNet5Mini(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError("TODO: LeNet-5 층을 정의하세요")

                def forward(self, x):
                    raise NotImplementedError("TODO: Fig. 2 순서로 forward를 구현하세요")

            lenet = LeNet5Mini().to(DEVICE)
            """,
            """
            class LeNet5Mini(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.c1 = nn.Conv2d(1, 6, kernel_size=5)
                    self.c3 = nn.Conv2d(6, 16, kernel_size=5)
                    self.c5 = nn.Conv2d(16, 120, kernel_size=5)
                    self.f6 = nn.Linear(120, 84)
                    self.classifier = nn.Linear(84, 10)

                def forward(self, x):
                    x = torch.tanh(self.c1(x))
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                    x = torch.tanh(self.c3(x))
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)
                    x = torch.tanh(self.c5(x))
                    x = torch.flatten(x, 1)
                    x = torch.tanh(self.f6(x))
                    return self.classifier(x)

            torch.manual_seed(0)
            lenet = LeNet5Mini().to(DEVICE)
            """,
            "implementation",
            "todo",
        ),
        code(
            """
            # TODO: 중간 텐서를 계산하고 Fig. 2의 다섯 공간 크기를 assert로 검증하세요.
            # 마지막에는 C1의 여섯 feature map을 imshow로 그리세요.
            def trace_lenet_features(model, batch):
                raise NotImplementedError("TODO: shape 추적")

            raise NotImplementedError("TODO: feature-map 시각화")
            """,
            """
            def trace_lenet_features(model, batch):
                c1 = torch.tanh(model.c1(batch))
                s2 = F.avg_pool2d(c1, 2, 2)
                c3 = torch.tanh(model.c3(s2))
                s4 = F.avg_pool2d(c3, 2, 2)
                c5 = torch.tanh(model.c5(s4))
                return {
                    "c1": c1,
                    "s2": s2,
                    "c3": c3,
                    "s4": s4,
                    "c5": c5,
                }

            with torch.no_grad():
                feature_trace = trace_lenet_features(lenet, lenet_x[:1])
                logits = lenet(lenet_x[:1])

            c1 = feature_trace["c1"]
            s2 = feature_trace["s2"]
            c3 = feature_trace["c3"]
            s4 = feature_trace["s4"]
            c5 = feature_trace["c5"]

            assert c1.shape == (1, 6, 28, 28)
            assert s2.shape == (1, 6, 14, 14)
            assert c3.shape == (1, 16, 10, 10)
            assert s4.shape == (1, 16, 5, 5)
            assert c5.shape == (1, 120, 1, 1)
            assert logits.shape == (1, 10)

            fig, axes = plt.subplots(1, 6, figsize=(10, 2))
            for channel, axis in enumerate(axes):
                axis.imshow(c1[0, channel].detach().cpu().numpy(), cmap="coolwarm")
                axis.set_title(f"C1-{channel}")
                axis.axis("off")
            fig.suptitle("LeNet-5 C1 shared-filter responses")
            fig.tight_layout()
            plt.show()
            print("shape trace:", [tuple(t.shape) for t in (c1, s2, c3, s4, c5)])
            """,
            "verification",
            "visualization",
            "todo",
        ),
        markdown(
            """
            ## 4. 원 논문과 이 구현의 경계

            C3의 모든 출력 map을 모든 S2 map에 연결한 것은 원 논문의 연결표와 다릅니다.
            출력도 원 논문의 RBF distance가 아니라 오늘날 익숙한 linear logits입니다. 반면
            32×32 입력, 5×5 valid convolution, 2×2 평균 subsampling, `6→16→120→84`라는
            핵심 shape 경로는 그대로이므로 합성곱 계층의 설계 아이디어를 검증할 수 있습니다.
            """,
            "scope-note",
        ),
        code(
            """
            # TODO: Adam과 cross_entropy로 15 step 학습하세요.
            # 매 step loss를 기록하고, 유한성/감소 여부를 assert한 뒤 loss curve와 accuracy를 출력하세요.
            def train_lenet_step(model, batch, targets, optimizer):
                raise NotImplementedError("TODO: 한 step 학습")

            raise NotImplementedError("TODO: 반복 학습과 평가")
            """,
            """
            optimizer = torch.optim.Adam(lenet.parameters(), lr=0.01)
            lenet_losses = []

            def train_lenet_step(model, batch, targets, optimizer):
                model.train()
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            for _ in range(15):
                lenet_losses.append(
                    train_lenet_step(
                        lenet,
                        lenet_x,
                        lenet_y,
                        optimizer,
                    )
                )

            lenet.eval()
            with torch.no_grad():
                lenet_accuracy = (lenet(lenet_x).argmax(1) == lenet_y).float().mean().item()
            assert torch.isfinite(torch.tensor(lenet_losses)).all()
            assert min(lenet_losses[1:]) < lenet_losses[0]

            plt.figure(figsize=(5, 3))
            plt.plot(lenet_losses, marker="o", markersize=3)
            plt.xlabel("optimization step")
            plt.ylabel("cross-entropy")
            plt.title("LeNet-5 mini reproduction")
            plt.grid(alpha=0.3)
            plt.show()
            print(
                f"loss {lenet_losses[0]:.3f} → {lenet_losses[-1]:.3f}; "
                f"train accuracy={lenet_accuracy:.1%}"
            )
            """,
            "experiment",
            "metric",
            "visualization",
            "todo",
        ),
    ),
)


ALEXNET = PaperSpec(
    number=1,
    slug="alexnet",
    short_title="AlexNet",
    paper_title="ImageNet Classification with Deep Convolutional Neural Networks",
    authors="Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton",
    year=2012,
    primary_url=(
        "https://proceedings.neurips.cc/paper_files/paper/2012/file/"
        "c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"
    ),
    venue="NeurIPS 2012",
    difficulty="초중급",
    expected_minutes=85,
    prerequisites="Conv2d, ReLU, max pooling, broadcasting",
    reproduction_goal=(
        "AlexNet의 ReLU, channel-local response normalization(LRN), 겹치는 max pooling, "
        "5-convolution/3-linear-stage 구성을 축소 모델에서 구현한다. LRN 수식의 channel window와 "
        "pooling stride를 수치·shape로 검증하고 합성 색상 패턴에서 짧게 최적화한다."
    ),
    original_scale=(
        "원 논문은 ImageNet 1.2M 영상, 1000 classes, 약 60M parameters, 두 GPU로 분할된 "
        "5 convolution + 3 fully-connected network를 수일간 학습한다. 여기서는 64×64 합성 영상 "
        "24개, 최대 32 channels와 4 classes를 사용하며 data augmentation과 multi-GPU는 생략한다."
    ),
    mappings=(
        (
            "§3.1 ReLU Nonlinearity, 식 f(x)=max(0,x)",
            "각 convolution 직후 `F.relu`",
            "모델 코드에서 다섯 conv 모두 ReLU 뒤에 배치",
        ),
        (
            "§3.3 Local Response Normalization, Eq. (1)",
            "`paper_lrn`의 인접 channel 제곱합",
            "단일 channel spike의 중심/이웃 정규화 값을 assert",
        ),
        (
            "§3.4 Overlapping Pooling: z=3, s=2",
            "`max_pool2d(..., kernel_size=3, stride=2)` 비교 셀",
            "stride 2와 stride 3의 출력 크기 및 heatmap 비교",
        ),
        (
            "§3.5 Overall Architecture와 Fig. 2: 5 conv + 3 FC, FC dropout",
            "`AlexNetMini`의 conv1…conv5와 두 hidden/output linear layers",
            "모듈 개수, logits shape, loss 감소를 검증",
        ),
    ),
    cells=(
        markdown(
            """
            ## 1. 무엇을 재현하는가

            AlexNet의 역사적 성능은 대규모 데이터·GPU 학습과 함께 나온 결과입니다. 이 실습은
            그 정확도를 재현하지 않고 §3의 세 가지 구체적 선택—ReLU, Eq. (1)의 LRN,
            `3×3/stride 2` overlapping pooling—을 분리해서 확인한 뒤 작은 5-conv 모델에 합칩니다.
            PyTorch의 `LocalResponseNorm`은 `alpha/n` 관례를 쓰므로 논문 수식을 눈으로 그대로
            추적할 수 있도록 직접 구현합니다.
            """,
            "paper-reading",
        ),
        shared_code(
            """
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            from torch.nn import functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(1)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device

            def make_color_patches(repeats=6, size=64):
                generator = torch.Generator().manual_seed(202)
                images, labels = [], []
                for index in range(4 * repeats):
                    label = index % 4
                    image = torch.zeros(3, size, size)
                    row = 7 + (label // 2) * 30
                    col = 7 + (label % 2) * 30
                    image[label % 3, row:row + 18, col:col + 18] = 1.0
                    image[(label + 1) % 3, row + 4:row + 14, col + 4:col + 14] = 0.45
                    noise = 0.035 * torch.randn(image.shape, generator=generator)
                    images.append((image + noise).clamp(0, 1))
                    labels.append(label)
                return torch.stack(images), torch.tensor(labels)

            alex_x, alex_y = make_color_patches()
            alex_x, alex_y = ACCELERATOR.move(alex_x, alex_y)
            assert alex_x.shape == (24, 3, 64, 64)
            print(ACCELERATOR.summary())
            print("synthetic ImageNet stand-in:", tuple(alex_x.shape))
            """,
            "setup",
            "synthetic-data",
        ),
        code(
            """
            # TODO: 논문 Eq. (1)을 구현하세요.
            # 각 channel i를 중심으로 n개 channel의 제곱합을 구해
            # x_i / (k + alpha * sum(x_j**2))**beta 를 반환합니다.
            def paper_lrn(x, n=5, k=2.0, alpha=1e-4, beta=0.75):
                raise NotImplementedError("TODO: channel-local response normalization")

            spike = torch.zeros(1, 7, 1, 1)
            spike[:, 3] = 2.0
            normalized_spike = paper_lrn(spike)
            """,
            """
            def paper_lrn(x, n=5, k=2.0, alpha=1e-4, beta=0.75):
                if n < 1 or n % 2 == 0:
                    raise ValueError("n must be a positive odd channel-window size")
                half = n // 2
                padded_square = F.pad(x.square(), (0, 0, 0, 0, half, half))
                channel_sum = padded_square.unfold(1, n, 1).sum(dim=-1)
                return x / (k + alpha * channel_sum).pow(beta)

            spike = torch.zeros(1, 7, 1, 1)
            spike[:, 3] = 2.0
            normalized_spike = paper_lrn(spike)
            expected_center = 2.0 / (2.0 + 1e-4 * 4.0) ** 0.75
            assert torch.allclose(normalized_spike[0, 3, 0, 0], torch.tensor(expected_center))
            assert normalized_spike[0, 3, 0, 0] < spike[0, 3, 0, 0]
            print("LRN center: %.6f → %.6f" % (spike[0, 3], normalized_spike[0, 3]))
            """,
            "equation",
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: 7×7 grid에 3×3 max pooling을 적용하세요.
            # stride=2(overlap)와 stride=3(non-overlap)의 출력 shape를 assert하고 나란히 그리세요.
            raise NotImplementedError("TODO: overlapping pooling 비교")
            """,
            """
            pool_input = torch.arange(49.0).reshape(1, 1, 7, 7)
            overlapping = F.max_pool2d(pool_input, kernel_size=3, stride=2)
            non_overlapping = F.max_pool2d(pool_input, kernel_size=3, stride=3)
            assert overlapping.shape == (1, 1, 3, 3)
            assert non_overlapping.shape == (1, 1, 2, 2)

            fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
            for axis, value, title in zip(
                axes,
                (pool_input, overlapping, non_overlapping),
                ("input 7×7", "z=3, s=2", "z=3, s=3"),
            ):
                axis.imshow(value[0, 0], cmap="viridis")
                axis.set_title(title)
                axis.axis("off")
            fig.tight_layout()
            plt.show()
            print("overlap/non-overlap elements:", overlapping.numel(), non_overlapping.numel())
            """,
            "pooling",
            "visualization",
            "todo",
        ),
        code(
            """
            # TODO: 5개의 convolution과 논문식 LRN/overlapping pool을 갖는 축소 AlexNet을 만드세요.
            # classifier에는 두 hidden/output linear stage와 dropout을 둡니다.
            class AlexNetMini(nn.Module):
                def __init__(self, classes=4):
                    super().__init__()
                    raise NotImplementedError("TODO: AlexNetMini modules")

                def forward(self, x):
                    raise NotImplementedError("TODO: AlexNetMini forward")

            alexnet = AlexNetMini().to(DEVICE)
            """,
            """
            class AlexNetMini(nn.Module):
                def __init__(self, classes=4):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 12, kernel_size=11, stride=4, padding=2)
                    self.conv2 = nn.Conv2d(12, 24, kernel_size=5, padding=2)
                    self.conv3 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
                    self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
                    self.conv5 = nn.Conv2d(32, 24, kernel_size=3, padding=1)
                    self.fc6 = nn.Linear(24, 32)
                    self.fc7 = nn.Linear(32, 16)
                    self.fc8 = nn.Linear(16, classes)
                    self.dropout = nn.Dropout(p=0.5)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.max_pool2d(paper_lrn(x), 3, 2)
                    x = F.relu(self.conv2(x))
                    x = F.max_pool2d(paper_lrn(x), 3, 2)
                    x = F.relu(self.conv3(x))
                    x = F.relu(self.conv4(x))
                    x = F.relu(self.conv5(x))
                    x = F.max_pool2d(x, 3, 2)
                    x = torch.flatten(x, 1)
                    x = self.dropout(F.relu(self.fc6(x)))
                    x = self.dropout(F.relu(self.fc7(x)))
                    return self.fc8(x)

            torch.manual_seed(1)
            alexnet = AlexNetMini().to(DEVICE)
            """,
            "implementation",
            "todo",
        ),
        code(
            """
            # TODO: logits shape, convolution 개수, parameter 수 범위를 assert하세요.
            raise NotImplementedError("TODO: AlexNet 구조 검증")
            """,
            """
            alexnet.eval()
            with torch.no_grad():
                alex_logits = alexnet(alex_x[:3])
            alex_conv_count = sum(isinstance(module, nn.Conv2d) for module in alexnet.modules())
            alex_parameter_count = sum(parameter.numel() for parameter in alexnet.parameters())
            assert alex_logits.shape == (3, 4)
            assert alex_conv_count == 5
            assert 20_000 < alex_parameter_count < 100_000
            print(
                f"5 conv layers; {alex_parameter_count:,} parameters; "
                f"logits={tuple(alex_logits.shape)}"
            )
            """,
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: 모델을 train mode로 두고 12 step 학습하세요.
            # loss 감소를 검증하고 curve 및 최종 합성 정확도를 표시하세요.
            raise NotImplementedError("TODO: AlexNet 미니 최적화")
            """,
            """
            alexnet.train()
            alex_optimizer = torch.optim.Adam(alexnet.parameters(), lr=0.01)
            alex_losses = []
            for _ in range(12):
                alex_optimizer.zero_grad()
                loss = F.cross_entropy(alexnet(alex_x), alex_y)
                loss.backward()
                alex_optimizer.step()
                alex_losses.append(float(loss.detach()))

            alexnet.eval()
            with torch.no_grad():
                alex_accuracy = (alexnet(alex_x).argmax(1) == alex_y).float().mean().item()
            assert torch.isfinite(torch.tensor(alex_losses)).all()
            assert min(alex_losses[1:]) < alex_losses[0]

            plt.figure(figsize=(5, 3))
            plt.plot(alex_losses, color="tab:orange", marker="o", markersize=3)
            plt.xlabel("optimization step")
            plt.ylabel("cross-entropy")
            plt.title("AlexNetMini on synthetic color patches")
            plt.grid(alpha=0.3)
            plt.show()
            print(
                f"loss {alex_losses[0]:.3f} → {alex_losses[-1]:.3f}; "
                f"accuracy={alex_accuracy:.1%}"
            )
            """,
            "experiment",
            "metric",
            "visualization",
            "todo",
        ),
    ),
)


UNET = PaperSpec(
    number=2,
    slug="unet",
    short_title="U-Net",
    paper_title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
    authors="Olaf Ronneberger, Philipp Fischer, and Thomas Brox",
    year=2015,
    primary_url="https://arxiv.org/pdf/1505.04597",
    venue="MICCAI 2015",
    difficulty="중급",
    expected_minutes=95,
    prerequisites="encoder-decoder CNN, ConvTranspose2d, pixel-wise binary loss",
    reproduction_goal=(
        "U자형 contracting/expanding path와 같은 해상도 encoder feature의 concatenation을 2-level "
        "모델로 구현한다. Fig. 1의 channel/shape 흐름, Eq. (3)의 경계 가중치 형태, pixel logits의 "
        "shape를 검증하고 합성 인접 사각형 분할에서 loss와 IoU를 측정한다."
    ),
    original_scale=(
        "원 논문은 572×572 입력, valid 3×3 convolution 때문에 발생하는 crop-and-copy, 64→1024 "
        "channels, 탄성 변형 augmentation, 세포별 거리변환 weight map을 사용한다. 여기서는 "
        "32×32 입력, padding=1, 4→16 channels, 합성 binary masks를 사용한다."
    ),
    mappings=(
        (
            "§2 및 Fig. 1 왼쪽: repeated 3×3 conv + ReLU와 2×2 max pool",
            "`DoubleConv`, `enc1`, `enc2`, `pool`",
            "encoder feature shape가 32²→16²→8²인지 assert",
        ),
        (
            "§2 및 Fig. 1 오른쪽: 2×2 up-convolution과 cropped feature copy/concat",
            "`up2/up1`와 `torch.cat` skip connections",
            "decoder concat channel 수와 최종 32×32 logits 검증",
        ),
        (
            "§2, Eq. (1): pixel-wise softmax와 cross-entropy",
            "binary인 미니 과제의 `binary_cross_entropy_with_logits`",
            "logits/target 동일 shape와 유한 loss를 확인",
        ),
        (
            "§2, Eq. (3): w(x)=wc(x)+w0 exp(-(d1+d2)^2/(2σ^2))",
            "`paper_border_weight`에서 두 object boundary 거리 사용",
            "객체 사이 gap의 weight가 먼 corner보다 큰지 assert하고 heatmap 표시",
        ),
    ),
    cells=(
        markdown(
            """
            ## 1. U의 두 팔과 skip connection

            Fig. 1의 왼쪽은 위치 정보를 압축하며 문맥을 얻고, 오른쪽은 해상도를 복원합니다.
            핵심은 같은 scale의 왼쪽 feature를 오른쪽에 **channel 방향으로 복사·연결**하는 것입니다.
            원 논문은 valid convolution 때문에 왼쪽 feature를 crop하지만, padding을 쓰는 이 미니
            모델에서는 두 공간 크기가 이미 같습니다. 이는 의도적인 구현 차이입니다.
            """,
            "paper-reading",
        ),
        shared_code(
            """
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            from torch.nn import functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(2)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device

            def make_square_segments(samples=8, size=32):
                generator = torch.Generator().manual_seed(303)
                images = torch.zeros(samples, 1, size, size)
                masks = torch.zeros_like(images)
                for index in range(samples):
                    top = 5 + index % 4
                    left = 3 + index % 3
                    masks[index, :, top:top + 11, left:left + 9] = 1
                    masks[index, :, top + 1:top + 12, left + 11:left + 20] = 1
                blurred = F.avg_pool2d(masks, kernel_size=5, stride=1, padding=2)
                noise = 0.04 * torch.randn(images.shape, generator=generator)
                images = (0.75 * masks + 0.35 * blurred + noise).clamp(0, 1)
                return images, masks

            unet_x, unet_y = make_square_segments()
            unet_x, unet_y = ACCELERATOR.move(unet_x, unet_y)
            assert unet_x.shape == unet_y.shape == (8, 1, 32, 32)
            assert set(unet_y.unique().tolist()) == {0.0, 1.0}
            print(ACCELERATOR.summary())
            print("segmentation batch:", tuple(unet_x.shape))
            """,
            "setup",
            "synthetic-data",
        ),
        code(
            """
            # TODO: (3×3 Conv → ReLU)를 두 번 수행하는 DoubleConv와
            # 2-level encoder/decoder U-Net을 구현하세요. skip은 dim=1로 concatenate합니다.
            class DoubleConv(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    raise NotImplementedError("TODO: repeated 3×3 convolution")

                def forward(self, x):
                    raise NotImplementedError("TODO: DoubleConv forward")

            class UNetMini(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError("TODO: contracting/expanding paths")

                def forward(self, x, return_shapes=False):
                    raise NotImplementedError("TODO: upsample, copy-and-concat, pixel logits")

            unet = UNetMini().to(DEVICE)
            """,
            """
            class DoubleConv(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.ReLU(),
                    )

                def forward(self, x):
                    return self.layers(x)

            class UNetMini(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.enc1 = DoubleConv(1, 4)
                    self.enc2 = DoubleConv(4, 8)
                    self.bridge = DoubleConv(8, 16)
                    self.pool = nn.MaxPool2d(2)
                    self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
                    self.dec2 = DoubleConv(16, 8)
                    self.up1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
                    self.dec1 = DoubleConv(8, 4)
                    self.out = nn.Conv2d(4, 1, kernel_size=1)

                def forward(self, x, return_shapes=False):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    bridge = self.bridge(self.pool(e2))
                    up2 = self.up2(bridge)
                    d2 = self.dec2(torch.cat((up2, e2), dim=1))
                    up1 = self.up1(d2)
                    d1 = self.dec1(torch.cat((up1, e1), dim=1))
                    logits = self.out(d1)
                    if return_shapes:
                        shapes = [
                            e1.shape,
                            e2.shape,
                            bridge.shape,
                            up2.shape,
                            d2.shape,
                            up1.shape,
                        ]
                        return logits, shapes
                    return logits

            torch.manual_seed(2)
            unet = UNetMini().to(DEVICE)
            """,
            "implementation",
            "todo",
        ),
        code(
            """
            # TODO: return_shapes=True로 Fig. 1의 축소/확대 shape와 최종 mask shape를 검증하세요.
            raise NotImplementedError("TODO: U-Net shape/skip 검증")
            """,
            """
            unet_logits, unet_shapes = unet(unet_x[:2], return_shapes=True)
            expected_shapes = [
                (2, 4, 32, 32),
                (2, 8, 16, 16),
                (2, 16, 8, 8),
                (2, 8, 16, 16),
                (2, 8, 16, 16),
                (2, 4, 32, 32),
            ]
            assert [tuple(shape) for shape in unet_shapes] == expected_shapes
            assert unet_logits.shape == unet_y[:2].shape
            assert torch.isfinite(unet_logits).all()
            print("contract → bridge → expand:", [tuple(shape) for shape in unet_shapes])
            """,
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: 두 instance mask의 boundary까지 거리 d1,d2를 구하고 Eq. (3)을 반환하세요.
            # w_c는 이 toy 예제에서 1, 논문 값 w0=10, sigma=5를 사용합니다.
            def paper_border_weight(instances, w0=10.0, sigma=5.0):
                raise NotImplementedError("TODO: U-Net Eq. (3) border weight")

            raise NotImplementedError("TODO: weight-map 검증 및 시각화")
            """,
            """
            def _binary_boundary(mask):
                mask4d = mask[None, None].float()
                eroded = 1.0 - F.max_pool2d(1.0 - mask4d, 3, 1, 1)
                return (mask4d - eroded).clamp_min(0)[0, 0]

            def paper_border_weight(instances, w0=10.0, sigma=5.0):
                if instances.ndim != 3 or instances.shape[0] < 2:
                    raise ValueError(
                        "instances must have shape [objects, height, width] "
                        "with at least two objects"
                    )
                height, width = instances.shape[-2:]
                yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
                pixels = torch.stack((yy, xx), dim=-1).reshape(-1, 2).float()
                distances = []
                for instance in instances:
                    boundary_points = _binary_boundary(instance).nonzero().float()
                    distances.append(torch.cdist(pixels, boundary_points).amin(dim=1))
                two_nearest = torch.stack(distances).topk(k=2, dim=0, largest=False).values
                d1, d2 = two_nearest[0], two_nearest[1]
                weight = 1.0 + w0 * torch.exp(-((d1 + d2).square()) / (2.0 * sigma**2))
                return weight.reshape(height, width)

            instances = torch.zeros(2, 32, 32)
            instances[0, 7:20, 4:14] = 1
            instances[1, 7:20, 16:26] = 1
            border_weight = paper_border_weight(instances)
            assert border_weight.shape == (32, 32)
            assert border_weight[12, 15] > border_weight[0, 0]
            assert 1.0 <= border_weight.min() <= border_weight.max() <= 11.0

            fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
            axes[0].imshow(unet_x[0, 0].detach().cpu(), cmap="gray")
            axes[0].set_title("synthetic input")
            axes[1].imshow(unet_y[0, 0].detach().cpu(), cmap="gray")
            axes[1].set_title("target mask")
            image = axes[2].imshow(border_weight, cmap="magma")
            axes[2].set_title("Eq. (3) weight")
            for axis in axes:
                axis.axis("off")
            fig.colorbar(image, ax=axes[2], fraction=0.046)
            fig.tight_layout()
            plt.show()
            print("gap/corner weight: %.2f / %.2f" % (border_weight[12, 15], border_weight[0, 0]))
            """,
            "equation",
            "visualization",
            "todo",
        ),
        markdown(
            """
            ## 6. Loss의 대응 관계

            원 논문 Eq. (1)은 두 class의 pixel-wise softmax입니다. binary foreground/background에서는
            한 개 logit에 대한 `binary_cross_entropy_with_logits`가 같은 log-likelihood를 표현합니다.
            아래 학습은 architecture를 빠르게 검증하면서 foreground 희소성을 보정하도록 binary
            `pos_weight`를 사용합니다. 이는 Eq. (2)의 class-balancing `w_c`에 대응합니다. 바로 앞
            셀의 Eq. (3) map을 loss의 `weight=` 인자로 넣으면 경계 강조 실험을 확장할 수 있습니다.
            """,
            "scope-note",
        ),
        code(
            """
            # TODO: class-balancing pos_weight를 둔 BCEWithLogits로 18 step 학습한 뒤 pixel IoU를 계산하세요.
            # target과 sigmoid prediction을 나란히 시각화하고 loss 감소를 assert하세요.
            raise NotImplementedError("TODO: U-Net 미니 분할 학습")
            """,
            """
            positive_weight = ((unet_y.numel() - unet_y.sum()) / unet_y.sum()).detach()
            unet_optimizer = torch.optim.Adam(unet.parameters(), lr=0.02)
            unet_losses = []
            for _ in range(18):
                unet_optimizer.zero_grad()
                logits = unet(unet_x)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    unet_y,
                    pos_weight=positive_weight,
                )
                loss.backward()
                unet_optimizer.step()
                unet_losses.append(float(loss.detach()))

            unet.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(unet(unet_x))
                predictions = probabilities >= 0.5
                targets = unet_y.bool()
                intersection = (predictions & targets).sum().float()
                union = (predictions | targets).sum().float().clamp_min(1)
                unet_iou = (intersection / union).item()
            assert torch.isfinite(torch.tensor(unet_losses)).all()
            assert min(unet_losses[1:]) < unet_losses[0]

            fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
            axes[0].imshow(unet_x[0, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("input")
            axes[1].imshow(unet_y[0, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("target")
            axes[2].imshow(probabilities[0, 0].detach().cpu(), cmap="viridis", vmin=0, vmax=1)
            axes[2].set_title("prediction")
            for axis in axes:
                axis.axis("off")
            fig.tight_layout()
            plt.show()
            print(f"loss {unet_losses[0]:.3f} → {unet_losses[-1]:.3f}; pixel IoU={unet_iou:.3f}")
            """,
            "experiment",
            "metric",
            "visualization",
            "todo",
        ),
    ),
)


RESNET = PaperSpec(
    number=3,
    slug="resnet",
    short_title="ResNet",
    paper_title="Deep Residual Learning for Image Recognition",
    authors="Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun",
    year=2016,
    primary_url="https://arxiv.org/pdf/1512.03385",
    venue="CVPR 2016",
    difficulty="중급",
    expected_minutes=85,
    prerequisites="Conv2d, BatchNorm, autograd, matrix multiplication",
    reproduction_goal=(
        "Eq. (1)의 residual mapping H(x)=F(x)+x와 dimension-changing projection shortcut을 "
        "BasicBlock으로 구현한다. residual branch가 0일 때의 identity, stage shape, 깊이에 따른 "
        "직접 gradient path를 검증하고 작은 residual classifier를 합성 영상에 학습한다."
    ),
    original_scale=(
        "원 논문은 ImageNet/CIFAR-10에서 18~152 layer networks와 224×224 images, 대규모 "
        "augmentation/training schedule을 비교한다. 여기서는 16×16 합성 영상, 8/16 channels, "
        "3 basic blocks만 사용하며 benchmark 정확도나 degradation curve를 재현하지 않는다."
    ),
    mappings=(
        (
            "§3.1, Eq. (1): H(x)=F(x)+x",
            "`BasicBlock.forward`의 `out + self.shortcut(x)`",
            "residual branch를 0으로 만들었을 때 양수 입력 identity를 assert",
        ),
        (
            "§3.2, Eq. (2): y=F(x,{Wi})+Ws x projection shortcut",
            "stride/channel 변화 때의 1×1 Conv+BN shortcut",
            "8→16 channels, stride 2 block의 `(N,16,8,8)` shape 검증",
        ),
        (
            "§3.3 및 Fig. 3: 3×3 두 개로 이루어진 residual building block",
            "`conv1-bn-relu-conv2-bn-add-relu`",
            "block 모듈과 `MiniResNet` stage 구성을 코드로 확인",
        ),
        (
            "§4.1 ‘Plain Networks’/‘Residual Networks’: shortcut의 optimization path",
            "동일한 작은 linear transform을 깊게 합성한 gradient 실험",
            "plain 0.1^L 경로와 residual (I+0.1I)^L 경로의 input-gradient plot",
        ),
    ),
    cells=(
        markdown(
            """
            ## 1. Residual learning의 최소 단위

            네트워크가 목표 함수 `H(x)` 전체를 직접 맞추는 대신 `F(x)=H(x)-x`를 학습하고
            출력에서 `x`를 더합니다. shortcut에는 parameter가 없고 shape가 달라질 때만 Eq. (2)의
            projection `W_s`를 씁니다. 원 논문의 post-activation basic block을 구현하므로 덧셈 뒤
            ReLU가 있습니다. 따라서 ‘정확한 identity’ 검증에는 비음수가 아닌 입력을 사용합니다.
            """,
            "paper-reading",
        ),
        shared_code(
            """
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            from torch.nn import functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(3)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device

            def make_quadrant_images(repeats=8, size=16):
                generator = torch.Generator().manual_seed(404)
                images, labels = [], []
                locations = ((2, 2), (2, 10), (10, 2), (10, 10))
                for index in range(4 * repeats):
                    label = index % 4
                    image = torch.zeros(3, size, size)
                    row, col = locations[label]
                    image[label % 3, row:row + 4, col:col + 4] = 1.0
                    noise = 0.04 * torch.randn(image.shape, generator=generator)
                    images.append((image + noise).clamp(0, 1))
                    labels.append(label)
                return torch.stack(images), torch.tensor(labels)

            res_x, res_y = make_quadrant_images()
            res_x, res_y = ACCELERATOR.move(res_x, res_y)
            assert res_x.shape == (32, 3, 16, 16)
            print(ACCELERATOR.summary())
            print("residual toy batch:", tuple(res_x.shape))
            """,
            "setup",
            "synthetic-data",
        ),
        code(
            """
            # TODO: 두 3×3 Conv-BN과 shortcut을 갖는 post-activation BasicBlock을 구현하세요.
            # stride/channel이 바뀌면 shortcut에 1×1 Conv-BN projection을 사용합니다.
            class BasicBlock(nn.Module):
                def __init__(self, in_channels, out_channels, stride=1):
                    super().__init__()
                    raise NotImplementedError("TODO: residual and shortcut branches")

                def forward(self, x):
                    raise NotImplementedError("TODO: F(x) + shortcut(x)")
            """,
            """
            class BasicBlock(nn.Module):
                def __init__(self, in_channels, out_channels, stride=1):
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                    self.bn1 = nn.BatchNorm2d(out_channels)
                    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
                    self.bn2 = nn.BatchNorm2d(out_channels)
                    if stride != 1 or in_channels != out_channels:
                        self.shortcut = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                            nn.BatchNorm2d(out_channels),
                        )
                    else:
                        self.shortcut = nn.Identity()

                def forward(self, x):
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    return F.relu(out + self.shortcut(x))
            """,
            "implementation",
            "todo",
        ),
        code(
            """
            # TODO: identity block의 residual branch를 0으로 만들고 양수 입력이 보존되는지 확인하세요.
            # 이어서 8→16, stride=2 projection block의 shape를 assert하세요.
            raise NotImplementedError("TODO: identity/projection shortcut 검증")
            """,
            """
            identity_block = BasicBlock(4, 4).eval()
            with torch.no_grad():
                identity_block.conv1.weight.zero_()
                identity_block.conv2.weight.zero_()
            positive_input = torch.rand(2, 4, 8, 8)
            with torch.no_grad():
                identity_output = identity_block(positive_input)
            assert torch.allclose(identity_output, positive_input, atol=1e-6)

            projection_block = BasicBlock(8, 16, stride=2).eval()
            with torch.no_grad():
                projection_output = projection_block(torch.randn(2, 8, 16, 16))
            assert projection_output.shape == (2, 16, 8, 8)
            assert isinstance(projection_block.shortcut[0], nn.Conv2d)
            print("identity max error:", (identity_output - positive_input).abs().max().item())
            print("projection output:", tuple(projection_output.shape))
            """,
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: stem(3→8), identity blocks, 8→16 projection, global average pool, FC로
            # 작은 residual classifier를 구현하고 logits shape를 검증하세요.
            class MiniResNet(nn.Module):
                def __init__(self, classes=4):
                    super().__init__()
                    raise NotImplementedError("TODO: residual stages")

                def forward(self, x):
                    raise NotImplementedError("TODO: stem → blocks → GAP → FC")

            resnet = MiniResNet().to(DEVICE)
            """,
            """
            class MiniResNet(nn.Module):
                def __init__(self, classes=4):
                    super().__init__()
                    self.stem = nn.Sequential(
                        nn.Conv2d(3, 8, 3, padding=1, bias=False),
                        nn.BatchNorm2d(8),
                        nn.ReLU(),
                    )
                    self.blocks = nn.Sequential(
                        BasicBlock(8, 8),
                        BasicBlock(8, 8),
                        BasicBlock(8, 16, stride=2),
                    )
                    self.classifier = nn.Linear(16, classes)

                def forward(self, x):
                    x = self.blocks(self.stem(x))
                    x = F.adaptive_avg_pool2d(x, 1).flatten(1)
                    return self.classifier(x)

            torch.manual_seed(3)
            resnet = MiniResNet().to(DEVICE)
            res_logits = resnet(res_x[:5])
            assert res_logits.shape == (5, 4)
            assert sum(isinstance(module, BasicBlock) for module in resnet.modules()) == 3
            print("MiniResNet logits:", tuple(res_logits.shape))
            """,
            "implementation",
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: depth 0..12에서 h←0.1Ih(plain)와 h←h+0.1Ih(residual)의
            # input-gradient norm을 구해 semilog plot으로 비교하세요.
            raise NotImplementedError("TODO: shortcut gradient-path 실험")
            """,
            """
            def input_gradient_norm(depth, residual):
                x = torch.ones(1, 8, requires_grad=True)
                transform = 0.1 * torch.eye(8)
                hidden = x
                for _ in range(depth):
                    branch = hidden @ transform
                    hidden = hidden + branch if residual else branch
                hidden.sum().backward()
                return x.grad.norm().item()

            depths = list(range(13))
            plain_gradients = [input_gradient_norm(depth, False) for depth in depths]
            residual_gradients = [input_gradient_norm(depth, True) for depth in depths]
            assert residual_gradients[-1] > plain_gradients[-1] * 1_000

            plt.figure(figsize=(5, 3))
            plt.semilogy(depths, plain_gradients, marker="o", label="plain 0.1I")
            plt.semilogy(depths, residual_gradients, marker="o", label="residual I+0.1I")
            plt.xlabel("depth")
            plt.ylabel("input-gradient norm")
            plt.title("Direct gradient path through identity shortcuts")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
            print("depth 12 gradient, plain/residual:", plain_gradients[-1], residual_gradients[-1])
            """,
            "experiment",
            "visualization",
            "metric",
            "todo",
        ),
        code(
            """
            # TODO: 합성 quadrant batch에 30 step 학습하고 loss/accuracy를 보고하세요.
            raise NotImplementedError("TODO: MiniResNet 최적화")
            """,
            """
            res_optimizer = torch.optim.Adam(resnet.parameters(), lr=0.02)
            res_losses = []
            resnet.train()
            for _ in range(30):
                res_optimizer.zero_grad()
                loss = F.cross_entropy(resnet(res_x), res_y)
                loss.backward()
                res_optimizer.step()
                res_losses.append(float(loss.detach()))

            resnet.eval()
            with torch.no_grad():
                res_accuracy = (resnet(res_x).argmax(1) == res_y).float().mean().item()
            assert torch.isfinite(torch.tensor(res_losses)).all()
            assert min(res_losses[1:]) < res_losses[0]
            print(f"loss {res_losses[0]:.3f} → {res_losses[-1]:.3f}; accuracy={res_accuracy:.1%}")
            """,
            "experiment",
            "metric",
            "todo",
        ),
    ),
)


DROPOUT = PaperSpec(
    number=4,
    slug="dropout",
    short_title="Dropout",
    paper_title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    authors=(
        "Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, "
        "Ilya Sutskever, and Ruslan Salakhutdinov"
    ),
    year=2014,
    primary_url="https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf",
    venue="Journal of Machine Learning Research 15",
    difficulty="초중급",
    expected_minutes=75,
    prerequisites="MLP, Bernoulli sampling, train/eval mode, expectation",
    reproduction_goal=(
        "논문 §2의 retain probability p를 사용하는 Bernoulli mask와 test-time weight scaling을 "
        "직접 구현한다. train/test 출력의 기대값 일치, masking 비율, 모듈 모드 전환을 검증하고 "
        "작은 XOR-like 분류기에서 loss 및 decision boundary를 관찰한다."
    ),
    original_scale=(
        "원 논문은 MNIST, TIMIT, CIFAR-10/100, SVHN, ImageNet에서 큰 networks와 여러 "
        "hyperparameter를 비교한다. 여기서는 80개의 2D Gaussian points와 2-layer MLP만 사용해 "
        "stochastic thinning과 test-time scaling 메커니즘을 검증한다."
    ),
    mappings=(
        (
            "§2, Fig. 1–2: 각 hidden unit을 probability p로 retain",
            "`paper_dropout`의 Bernoulli mask",
            "10,000개 unit의 실제 retain fraction이 p 근처인지 assert",
        ),
        (
            "§2, Eq. (1): r_j^(l) ~ Bernoulli(p), y~=r*y",
            "training branch의 `mask * x`",
            "고정 generator로 mask와 출력의 재현성 확인",
        ),
        (
            "§2 test time: outgoing weights에 p를 곱한 scaled network",
            "activation 관점에서 같은 `p * x`를 반환하는 eval branch",
            "train Monte-Carlo mean과 deterministic eval mean 비교",
        ),
        (
            "§7.2 ‘Effect of Dropout on Sparsity’: sparse activations 관찰",
            "`DropoutMLP` hidden activation mask와 합성 분류 실험",
            "loss curve와 2D decision probability map을 표시",
        ),
    ),
    cells=(
        markdown(
            """
            ## 1. 논문식 dropout과 PyTorch식 dropout

            논문은 training 때 `mask*x`를 쓰고 test 때 weights를 `p`배 합니다. 이 실습은 activation
            관점에서 test 출력을 `p*x`로 계산합니다. PyTorch의 `nn.Dropout(dropout_probability)`은
            **inverted dropout**이라 training 때 살아남은 값을 `1/p`배하고 eval에서는 그대로 둡니다.
            두 관례의 기대값은 같지만 중간 tensor의 scale이 다릅니다. 여기서 `p`는 drop 확률이
            아니라 **retain 확률**임에 주의하세요.
            """,
            "paper-reading",
        ),
        shared_code(
            """
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            from torch.nn import functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(4)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device

            def make_xor_blobs(points_per_blob=20):
                generator = torch.Generator().manual_seed(505)
                centers = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
                labels = torch.tensor([0, 1, 1, 0])
                points, targets = [], []
                for center, label in zip(centers, labels):
                    noise = 0.22 * torch.randn(points_per_blob, 2, generator=generator)
                    points.append(center + noise)
                    targets.append(label.repeat(points_per_blob))
                return torch.cat(points), torch.cat(targets)

            drop_x, drop_y = make_xor_blobs()
            drop_x, drop_y = ACCELERATOR.move(drop_x, drop_y)
            assert drop_x.shape == (80, 2)
            assert drop_y.bincount().tolist() == [40, 40]
            print(ACCELERATOR.summary())
            print("XOR-like batch:", tuple(drop_x.shape))
            """,
            "setup",
            "synthetic-data",
        ),
        code(
            """
            # TODO: 논문 관례의 dropout을 구현하세요.
            # training: Bernoulli(p) mask*x, evaluation: p*x
            def paper_dropout(x, retain_probability=0.5, training=True, generator=None):
                raise NotImplementedError("TODO: paper-convention dropout")

            raise NotImplementedError("TODO: retain fraction과 train/test expectation 검증")
            """,
            """
            def paper_dropout(x, retain_probability=0.5, training=True, generator=None):
                if not 0.0 < retain_probability <= 1.0:
                    raise ValueError("retain_probability must be in (0, 1]")
                if not training:
                    return retain_probability * x
                probabilities = torch.full_like(x, retain_probability)
                mask = torch.bernoulli(probabilities, generator=generator)
                return mask * x

            ones = torch.ones(10_000)
            dropout_generator = torch.Generator().manual_seed(606)
            sampled = paper_dropout(ones, 0.6, True, dropout_generator)
            evaluated = paper_dropout(ones, 0.6, False)
            retain_fraction = sampled.mean().item()
            assert abs(retain_fraction - 0.6) < 0.02
            assert torch.allclose(evaluated, torch.full_like(ones, 0.6))
            assert abs(sampled.mean().item() - evaluated.mean().item()) < 0.02
            print(
                f"sampled retain={retain_fraction:.3f}; "
                f"deterministic test mean={evaluated.mean():.3f}"
            )
            """,
            "equation",
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: self.training을 paper_dropout에 전달하는 PaperDropout module과
            # 2→24→24→2 MLP를 구현하세요. 두 hidden ReLU 뒤에 retain p=0.75를 적용합니다.
            class PaperDropout(nn.Module):
                def __init__(self, retain_probability=0.5):
                    super().__init__()
                    raise NotImplementedError("TODO: retain probability 저장")

                def forward(self, x):
                    raise NotImplementedError("TODO: self.training에 따른 dropout")

            class DropoutMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError("TODO: MLP layers")

                def forward(self, x):
                    raise NotImplementedError("TODO: linear-ReLU-dropout 흐름")

            dropout_model = DropoutMLP().to(DEVICE)
            """,
            """
            class PaperDropout(nn.Module):
                def __init__(self, retain_probability=0.5):
                    super().__init__()
                    if not 0.0 < retain_probability <= 1.0:
                        raise ValueError("retain_probability must be in (0, 1]")
                    self.retain_probability = retain_probability

                def forward(self, x):
                    return paper_dropout(x, self.retain_probability, self.training)

            class DropoutMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(2, 24)
                    self.fc2 = nn.Linear(24, 24)
                    self.out = nn.Linear(24, 2)
                    self.dropout = PaperDropout(retain_probability=0.75)

                def forward(self, x):
                    x = self.dropout(F.relu(self.fc1(x)))
                    x = self.dropout(F.relu(self.fc2(x)))
                    return self.out(x)

            torch.manual_seed(4)
            dropout_model = DropoutMLP().to(DEVICE)
            dropout_model.eval()
            first_eval = dropout_model(drop_x[:4])
            second_eval = dropout_model(drop_x[:4])
            assert torch.equal(first_eval, second_eval)
            assert first_eval.shape == (4, 2)
            """,
            "implementation",
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: train mode에서 60 step 최적화하고 eval mode 정확도를 구하세요.
            # loss가 한 번 이상 초기값 아래로 내려갔는지 검증하세요.
            raise NotImplementedError("TODO: dropout MLP 학습")
            """,
            """
            dropout_optimizer = torch.optim.Adam(dropout_model.parameters(), lr=0.03)
            dropout_losses = []
            dropout_model.train()
            for _ in range(60):
                dropout_optimizer.zero_grad()
                loss = F.cross_entropy(dropout_model(drop_x), drop_y)
                loss.backward()
                dropout_optimizer.step()
                dropout_losses.append(float(loss.detach()))

            dropout_model.eval()
            with torch.no_grad():
                dropout_accuracy = (dropout_model(drop_x).argmax(1) == drop_y).float().mean().item()
            assert torch.isfinite(torch.tensor(dropout_losses)).all()
            assert min(dropout_losses[1:]) < dropout_losses[0]
            print(
                f"loss {dropout_losses[0]:.3f} → {dropout_losses[-1]:.3f}; "
                f"accuracy={dropout_accuracy:.1%}"
            )
            """,
            "experiment",
            "metric",
            "todo",
        ),
        code(
            """
            # TODO: [-1.7,1.7]^2 grid에서 class-1 probability를 계산해 decision map과
            # loss curve를 나란히 그리세요.
            raise NotImplementedError("TODO: dropout metric 시각화")
            """,
            """
            grid_axis = torch.linspace(-1.7, 1.7, 80, device=DEVICE)
            grid_y, grid_x = torch.meshgrid(grid_axis, grid_axis, indexing="ij")
            grid_points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
            dropout_model.eval()
            with torch.no_grad():
                class_one_probability = dropout_model(grid_points).softmax(1)[:, 1].reshape(80, 80)

            fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
            image = axes[0].contourf(
                grid_x.detach().cpu(),
                grid_y.detach().cpu(),
                class_one_probability.detach().cpu(),
                levels=20,
                cmap="coolwarm",
            )
            axes[0].scatter(
                drop_x[:, 0].detach().cpu(),
                drop_x[:, 1].detach().cpu(),
                c=drop_y.detach().cpu(),
                cmap="coolwarm",
                edgecolor="black",
                s=18,
            )
            axes[0].set_title("test-time scaled decision map")
            fig.colorbar(image, ax=axes[0], fraction=0.046)
            axes[1].plot(dropout_losses)
            axes[1].set_title("stochastic training loss")
            axes[1].set_xlabel("optimization step")
            axes[1].set_ylabel("cross-entropy")
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            plt.show()
            print(
                "decision probability range:",
                float(class_one_probability.min()),
                float(class_one_probability.max()),
            )
            """,
            "visualization",
            "metric",
            "todo",
        ),
    ),
)


BATCHNORM = PaperSpec(
    number=5,
    slug="batch_normalization",
    short_title="Batch Normalization",
    paper_title=(
        "Batch Normalization: Accelerating Deep Network Training by "
        "Reducing Internal Covariate Shift"
    ),
    authors="Sergey Ioffe and Christian Szegedy",
    year=2015,
    primary_url="https://proceedings.mlr.press/v37/ioffe15.pdf",
    venue="ICML 2015",
    difficulty="중급",
    expected_minutes=90,
    prerequisites="평균·분산, nn.Module parameters/buffers, train/eval mode",
    reproduction_goal=(
        "Algorithm 1의 mini-batch mean/variance, normalization, 학습 가능한 γ/β를 직접 구현하고 "
        "running statistics를 갖는 module로 확장한다. 표준화 수치와 train/eval 동작을 assert하며 "
        "서로 다른 feature scale의 합성 분류에서 loss curve를 비교한다."
    ),
    original_scale=(
        "원 논문은 ImageNet의 Inception network와 MNIST에서 큰 batch training, population-statistic "
        "추론, 여러 optimization schedule을 평가한다. 여기서는 96×4 tabular mini-batches와 "
        "16-unit MLP를 사용하며 논문의 claimed speedup 자체를 benchmark하지 않는다."
    ),
    mappings=(
        (
            "§3, Algorithm 1 lines 1–2: μ_B와 σ_B²",
            "`manual_batch_norm`의 mean과 biased batch variance",
            "정규화 결과의 feature-wise mean≈0, variance≈1 assert",
        ),
        (
            "§3, Eqs. (1)–(2)와 Algorithm 1 lines 3–4: x-hat, y=γx-hat+β",
            "epsilon 정규화와 learnable `weight`/`bias`",
            "임의 γ/β를 넣은 출력 mean이 β인지 검증",
        ),
        (
            "§3.1: inference에서 population statistics 사용",
            "`RunningBatchNorm1d.running_mean/running_var` buffers",
            "eval 두 호출이 같고 running buffers가 변하지 않는지 assert",
        ),
        (
            "§3.2: BN transform 뒤 nonlinearity 적용",
            "`ScaledMLP`: Linear → BN → ReLU",
            "서로 다른 입력 scale에서 BN/no-BN loss와 정확도 plot",
        ),
    ),
    cells=(
        markdown(
            """
            ## 1. Algorithm 1을 네 줄로 분해하기

            feature마다 mini-batch 평균 `μ_B`와 분산 `σ_B²`를 구하고, epsilon을 더해 정규화한
            뒤 학습되는 `γ, β`로 표현력을 복원합니다. 이 실습의 batch variance는 Algorithm 1처럼
            분모가 `m`인 biased variance입니다. 추론 때는 batch 자체가 아니라 학습 중 누적한
            running statistics를 씁니다. 이는 단일 batch 함수와 stateful module을 따로 구현하는 이유입니다.
            """,
            "paper-reading",
        ),
        shared_code(
            """
            import matplotlib.pyplot as plt
            import torch
            from torch import nn
            from torch.nn import functional as F
            from llm_engineering_lab.acceleration import get_accelerator

            torch.manual_seed(5)
            ACCELERATOR = get_accelerator()
            DEVICE = ACCELERATOR.device

            generator = torch.Generator().manual_seed(707)
            latent_features = torch.randn(96, 4, generator=generator)
            feature_scale = torch.tensor([1.0, 20.0, 0.05, 5.0])
            feature_shift = torch.tensor([0.0, 3.0, -2.0, 10.0])
            bn_x = latent_features * feature_scale + feature_shift
            bn_y = (latent_features[:, 0] + latent_features[:, 1]
                    - 0.8 * latent_features[:, 2] + 0.5 * latent_features[:, 3] > 0).long()
            bn_x, bn_y = ACCELERATOR.move(bn_x, bn_y)
            assert bn_x.shape == (96, 4)
            assert bn_y.unique().tolist() == [0, 1]
            print(ACCELERATOR.summary())
            print("raw means:", bn_x.mean(0).round(decimals=2).tolist())
            print("raw stds: ", bn_x.std(0, unbiased=False).round(decimals=2).tolist())
            """,
            "setup",
            "synthetic-data",
        ),
        code(
            """
            # TODO: Algorithm 1 lines 1–4를 구현해 (output, mean, variance)를 반환하세요.
            # variance는 unbiased=False, epsilon은 sqrt 안에 둡니다.
            def manual_batch_norm(x, gamma, beta, eps=1e-5):
                raise NotImplementedError("TODO: batch mean/variance, normalize, scale-and-shift")

            raise NotImplementedError("TODO: zero mean/unit variance와 gamma-beta 검증")
            """,
            """
            def manual_batch_norm(x, gamma, beta, eps=1e-5):
                mean = x.mean(dim=0)
                variance = x.var(dim=0, unbiased=False)
                normalized = (x - mean) / torch.sqrt(variance + eps)
                return gamma * normalized + beta, mean, variance

            gamma = torch.tensor([1.0, 0.5, 2.0, 1.5], device=DEVICE)
            beta = torch.tensor([0.0, 1.0, -1.0, 0.25], device=DEVICE)
            transformed, batch_mean, batch_variance = manual_batch_norm(bn_x, gamma, beta)
            recovered_normalized = (transformed - beta) / gamma
            assert torch.allclose(
                recovered_normalized.mean(0),
                torch.zeros(4, device=DEVICE),
                atol=1e-5,
            )
            assert torch.allclose(
                recovered_normalized.var(0, unbiased=False),
                torch.ones(4, device=DEVICE),
                atol=5e-3,
            )
            assert torch.allclose(transformed.mean(0), beta, atol=1e-5)
            print("normalized mean:", recovered_normalized.mean(0).round(decimals=5).tolist())
            print(
                "normalized var: ",
                recovered_normalized.var(0, unbiased=False)
                .round(decimals=5)
                .tolist(),
            )
            """,
            "algorithm",
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: learnable weight/bias와 running_mean/running_var buffers를 갖는 BN module을 만드세요.
            # training이면 batch stats로 정규화하고 EMA를 갱신, eval이면 running stats를 사용합니다.
            class RunningBatchNorm1d(nn.Module):
                def __init__(self, features, eps=1e-5, momentum=0.1):
                    super().__init__()
                    raise NotImplementedError("TODO: parameters and buffers")

                def forward(self, x):
                    raise NotImplementedError("TODO: train/eval statistics")

            running_bn = RunningBatchNorm1d(4).to(DEVICE)
            """,
            """
            class RunningBatchNorm1d(nn.Module):
                def __init__(self, features, eps=1e-5, momentum=0.1):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(features))
                    self.bias = nn.Parameter(torch.zeros(features))
                    self.register_buffer("running_mean", torch.zeros(features))
                    self.register_buffer("running_var", torch.ones(features))
                    self.eps = eps
                    self.momentum = momentum

                def forward(self, x):
                    if self.training:
                        mean = x.mean(dim=0)
                        variance = x.var(dim=0, unbiased=False)
                        population_variance_estimate = x.var(dim=0, unbiased=True)
                        with torch.no_grad():
                            self.running_mean.lerp_(mean.detach(), self.momentum)
                            self.running_var.lerp_(
                                population_variance_estimate.detach(),
                                self.momentum,
                            )
                    else:
                        mean = self.running_mean
                        variance = self.running_var
                    normalized = (x - mean) / torch.sqrt(variance + self.eps)
                    return self.weight * normalized + self.bias

            running_bn = RunningBatchNorm1d(4).to(DEVICE)
            running_bn.train()
            train_output = running_bn(bn_x[:24])
            assert torch.allclose(train_output.mean(0), torch.zeros(4, device=DEVICE), atol=1e-5)
            saved_mean = running_bn.running_mean.clone()
            saved_var = running_bn.running_var.clone()

            running_bn.eval()
            eval_one = running_bn(bn_x[:8])
            eval_two = running_bn(bn_x[:8])
            assert torch.equal(eval_one, eval_two)
            assert torch.equal(saved_mean, running_bn.running_mean)
            assert torch.equal(saved_var, running_bn.running_var)
            print("running mean after one batch:", saved_mean.round(decimals=3).tolist())
            """,
            "implementation",
            "verification",
            "todo",
        ),
        code(
            """
            # TODO: Linear(4,16) → optional BN → ReLU → Linear(16,2) 모델을 만들고,
            # 같은 초기 seed의 BN/no-BN 모델을 48 mini-batch step 학습해 histories를 반환하세요.
            raise NotImplementedError("TODO: scaled-feature optimization comparison")
            """,
            """
            class ScaledMLP(nn.Module):
                def __init__(self, use_batch_norm):
                    super().__init__()
                    self.hidden = nn.Linear(4, 16)
                    self.normalizer = RunningBatchNorm1d(16) if use_batch_norm else nn.Identity()
                    self.output = nn.Linear(16, 2)

                def forward(self, x):
                    return self.output(F.relu(self.normalizer(self.hidden(x))))

            def train_scaled_model(use_batch_norm):
                torch.manual_seed(808)
                model = ScaledMLP(use_batch_norm).to(DEVICE)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
                history = []
                model.train()
                for step in range(48):
                    start = (step * 24) % len(bn_x)
                    batch_indices = torch.arange(start, start + 24, device=bn_x.device) % len(bn_x)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(bn_x[batch_indices]), bn_y[batch_indices])
                    loss.backward()
                    optimizer.step()
                    history.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    accuracy = (model(bn_x).argmax(1) == bn_y).float().mean().item()
                return model, history, accuracy

            plain_model, plain_losses, plain_accuracy = train_scaled_model(False)
            bn_model, bn_losses, bn_accuracy = train_scaled_model(True)
            assert torch.isfinite(torch.tensor(plain_losses + bn_losses)).all()
            assert min(bn_losses[1:]) < bn_losses[0]
            """,
            "experiment",
            "metric",
            "todo",
        ),
        code(
            """
            # TODO: BN/no-BN loss curve를 나란히 그리고 최종 loss와 accuracy를 출력하세요.
            # 이 한 seed의 toy 결과를 일반적인 speedup 증명으로 해석하지 마세요.
            raise NotImplementedError("TODO: BatchNorm 비교 시각화")
            """,
            """
            plt.figure(figsize=(6, 3.4))
            plt.plot(plain_losses, label="without BN", alpha=0.85)
            plt.plot(bn_losses, label="with Algorithm-1 BN", alpha=0.85)
            plt.xlabel("mini-batch step")
            plt.ylabel("cross-entropy")
            plt.title("Optimization on differently scaled features")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
            print(f"without BN: loss={plain_losses[-1]:.3f}, accuracy={plain_accuracy:.1%}")
            print(f"with BN:    loss={bn_losses[-1]:.3f}, accuracy={bn_accuracy:.1%}")
            """,
            "visualization",
            "metric",
            "todo",
        ),
    ),
)


SPECS: tuple[PaperSpec, ...] = (LENET5, ALEXNET, UNET, RESNET, DROPOUT, BATCHNORM)


__all__ = ["SPECS"]
