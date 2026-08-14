"""Portfolio-grade paired reproductions for ten landmark vision papers."""

from __future__ import annotations

from .common import FieldPaperSpec, code, markdown, shared_code

FIELD_ID = "vision"
FIELD_TITLE = "컴퓨터 비전"
DATASET = "data/field_curriculum/vision_shapes.npz"


def _setup():
    return shared_code(
        f"""
        from dataclasses import dataclass
        from pathlib import Path
        import itertools
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        from llm_engineering_lab.acceleration import get_accelerator

        torch.manual_seed(20260814)
        ACCELERATOR = get_accelerator()
        DEVICE = ACCELERATOR.device

        def locate(relative_path):
            for base in (Path.cwd(), *Path.cwd().parents):
                candidate = base / relative_path
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(relative_path)

        dataset_path = locate("{DATASET}")
        raw = np.load(dataset_path)
        images = torch.from_numpy(raw["images"]).float()
        labels = torch.from_numpy(raw["labels"]).long()
        masks = torch.from_numpy(raw["masks"]).float()
        boxes = torch.from_numpy(raw["boxes"]).float()
        train_idx = torch.from_numpy(raw["train_idx"][:128]).long()
        test_idx = torch.from_numpy(raw["test_idx"][:64]).long()
        images, labels, masks, boxes = ACCELERATOR.move(
            images,
            labels,
            masks,
            boxes,
        )
        train_idx, test_idx = ACCELERATOR.move(train_idx, test_idx)
        class_names = raw["class_names"].tolist()
        x_train = images[train_idx]
        y_train = labels[train_idx]
        assert images.shape == (480, 1, 16, 16)
        assert masks.shape == (480, 16, 16)
        assert boxes.shape == (480, 4)
        print(ACCELERATOR.summary())
        print(dataset_path.name, images.shape, class_names)
        """,
        "setup",
        "local-data",
    )


def _concept(equation: str, symbols: str, roles: str, shapes: str, limits: str):
    return markdown(
        f"""
        ## 핵심 아이디어와 수식

        $${equation}$$

        **기호 해설:** {symbols}

        **코드 대응:** {roles}

        **입력과 출력 shape:** {shapes}

        **포트폴리오 구현 범위:** {limits}

        아래 코드는 결과만 내는 라이브러리 호출로 핵심을 감추지 않는다. config, 논문 고유
        연산, 모델의 `forward`, 목적함수, `train_step`, 평가 함수를 순서대로 직접 작성한다.
        """,
        "paper-reading",
        "equation",
        "implementation-map",
    )


def _implementation_note(task_number: int, mapping):
    paper_part, lab_part, evidence = mapping
    return markdown(
        f"""
        ### 구현 단계 {task_number} — 원문 근거를 코드로 옮기기

        - **논문의 어느 부분인가:** {paper_part}
        - **노트북에서 구현할 부분:** {lab_part}
        - **구현 이유:** 논문의 계산 경로를 작은 tensor로 보존하면서 각 중간값을
          assertion으로 관찰하기 위해서다.
        - **완료 증거:** {evidence}

        먼저 입력 shape와 기대 출력 shape를 종이에 적은 뒤 아래 TODO를 구현한다.
        """,
        "paper-section",
        f"task-{task_number}",
    )


def _classification_training(title: str) -> tuple[str, str]:
    exercise = """
    # TODO 3: 분류 손실, 한 학습 step, 평가 함수를 각각 구현하세요.
    def classification_loss(logits, targets):
        raise NotImplementedError

    def train_step(model, optimizer, inputs, targets):
        raise NotImplementedError

    def evaluate_classifier(model, inputs, targets):
        raise NotImplementedError
    """
    solution = f"""
    def classification_loss(logits, targets):
        return F.cross_entropy(logits, targets)

    def train_step(model, optimizer, inputs, targets):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = classification_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def evaluate_classifier(model, inputs, targets):
        model.eval()
        with torch.no_grad():
            predictions = model(inputs).argmax(dim=1)
        return float((predictions == targets).float().mean())

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses = []
    for step in range(config.steps):
        step_loss = train_step(model, optimizer, x_train, y_train)
        losses.append(step_loss)

    accuracy = evaluate_classifier(
        model,
        images[test_idx],
        labels[test_idx],
    )
    assert min(losses[3:]) < losses[0]
    assert 0.0 <= accuracy <= 1.0
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("cross entropy")
    plt.title("{title}: accuracy={{:.1%}}".format(accuracy))
    plt.tight_layout()
    plt.show()
    print(f"loss={{losses[0]:.3f}} -> {{losses[-1]:.3f}}")
    """
    return exercise, solution


def _classification_cells(
    concept,
    task1_exercise: str,
    task1_solution: str,
    task2_exercise: str,
    task2_solution: str,
    title: str,
):
    train_exercise, train_solution = _classification_training(title)
    return (
        concept,
        _setup(),
        code(task1_exercise, task1_solution, "todo", "equation"),
        code(task2_exercise, task2_solution, "todo", "model"),
        code(train_exercise, train_solution, "todo", "training", "evaluation"),
    )


def _paper(**values):
    mappings = values.pop("mappings")
    cells = values.pop("cells")
    mapping_indices = (0, min(1, len(mappings) - 1), len(mappings) - 1)
    annotated_cells = []
    task_number = 0
    for cell in cells:
        is_task = cell.cell_type == "code" and cell.exercise != cell.solution
        if is_task:
            task_number += 1
            mapping_index = mapping_indices[min(task_number - 1, 2)]
            annotated_cells.append(
                _implementation_note(task_number, mappings[mapping_index])
            )
        annotated_cells.append(cell)
    return FieldPaperSpec(
        field_id=FIELD_ID,
        field_title=FIELD_TITLE,
        dataset_file=DATASET,
        difficulty="중급",
        expected_minutes=120,
        prerequisites="Python, PyTorch tensor, 합성곱, 역전파의 기초",
        mappings=mappings,
        cells=tuple(annotated_cells),
        **values,
    )


LENET5 = _paper(
    number=0,
    slug="lenet5",
    short_title="LeNet-5",
    paper_title="Gradient-Based Learning Applied to Document Recognition",
    authors="Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner",
    year=1998,
    primary_url="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
    venue="Proceedings of the IEEE 86(11)",
    reproduction_goal="공유 커널과 평균 subsampling으로 C1-S2-C3-S4 경로를 재현한다.",
    original_scale=(
        "원 논문의 32×32 문자, 부분 연결 C3, 학습 가능한 subsampling, RBF 출력 대신 "
        "16×16 합성 도형과 완전 연결 convolution, 4-class linear head를 사용한다."
    ),
    mappings=(
        (
            "§II.A: local receptive fields와 shared weights",
            "Task 1 Conv2d",
            "커널/shape assert",
        ),
        ("§II.B, Fig. 2: C1-S2-C3-S4-C5-F6", "Task 2 LeNetMini", "중간 shape assert"),
        ("§II.B: average subsampling", "Task 2 AvgPool2d", "max pool과 구별"),
        ("Table I: gradient-based training", "Task 3", "loss/accuracy plot"),
    ),
    cells=_classification_cells(
        _concept(
            r"h_{c,i,j}=\tanh\!\left(b_c+\sum_{k,u,v}W_{c,k,u,v}x_{k,i+u,j+v}\right)",
            "x는 입력, W는 공유 커널, b는 편향, h는 특징 맵이다.",
            "Task 1은 공간 크기를 계산하고 Task 2의 LeNetMini가 C1-S2-C3-S4를 구현한다.",
            "입력 [B,1,16,16] → C1/S2 [B,6,6,6] → C3/S4 [B,16,1,1] → [B,4].",
            "원 논문의 부분 연결과 RBF head는 생략하지만 평균 pooling과 공유 가중치는 보존한다.",
        ),
        """
        # TODO 1: convolution/pooling 출력 크기 함수와 설정 dataclass를 구현하세요.
        @dataclass
        class LeNetConfig:
            pass

        def spatial_size(size, kernel, stride=1, padding=0):
            raise NotImplementedError
        """,
        """
        @dataclass
        class LeNetConfig:
            channels_c1: int = 6
            channels_c3: int = 16
            num_classes: int = 4
            learning_rate: float = 0.015
            steps: int = 14

        def spatial_size(size, kernel, stride=1, padding=0):
            numerator = size + 2 * padding - kernel
            return numerator // stride + 1

        config = LeNetConfig()
        c1_size = spatial_size(16, 5)
        s2_size = spatial_size(c1_size, 2, stride=2)
        c3_size = spatial_size(s2_size, 5)
        s4_size = spatial_size(c3_size, 2, stride=2)
        assert (c1_size, s2_size, c3_size, s4_size) == (12, 6, 2, 1)
        """,
        """
        # TODO 2: C1-S2-C3-S4와 classifier를 갖는 LeNetMini를 구현하세요.
        class LeNetMini(nn.Module):
            def __init__(self, config):
                super().__init__()

            def forward_features(self, inputs):
                '''C1-S2-C3-S4 특징 맵 [B, C, 1, 1]을 반환한다.'''
                raise NotImplementedError

            def forward(self, inputs):
                raise NotImplementedError
        """,
        """
        class LeNetMini(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.c1 = nn.Conv2d(1, config.channels_c1, kernel_size=5)
                self.c3 = nn.Conv2d(
                    config.channels_c1,
                    config.channels_c3,
                    kernel_size=5,
                )
                self.classifier = nn.Linear(config.channels_c3, config.num_classes)

            def forward_features(self, inputs):
                features = torch.tanh(self.c1(inputs))
                features = F.avg_pool2d(features, kernel_size=2)
                features = torch.tanh(self.c3(features))
                return F.avg_pool2d(features, kernel_size=2)

            def forward(self, inputs):
                features = self.forward_features(inputs)
                return self.classifier(features.flatten(start_dim=1))

        model = LeNetMini(config).to(DEVICE)
        assert model.forward_features(images[:2]).shape == (2, 16, 1, 1)
        assert model(images[:2]).shape == (2, 4)
        """,
        "LeNet-5 mini",
    ),
)


ALEXNET = _paper(
    number=1,
    slug="alexnet",
    short_title="AlexNet",
    paper_title="ImageNet Classification with Deep Convolutional Neural Networks",
    authors="Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton",
    year=2012,
    primary_url=(
        "https://proceedings.neurips.cc/paper_files/paper/2012/file/"
        "c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"
    ),
    venue="NeurIPS 2012",
    reproduction_goal="ReLU, channel-local LRN, overlapping pooling, 5-conv 경로를 재현한다.",
    original_scale=(
        "ImageNet 1.2M장과 약 60M parameter, 2-GPU 분할 대신 16×16 합성 도형과 작은 "
        "채널을 사용한다. 논문의 다섯 convolution 구조와 LRN 수식은 유지한다."
    ),
    mappings=(
        ("§3.1: f(x)=max(0,x)", "Task 1", "ReLU gradient assert"),
        ("§3.3 Eq. (1): LRN", "Task 1 paper_lrn", "수치 assert"),
        ("§3.4: overlapping pooling z=3,s=2", "Task 2", "stride shape"),
        ("§3.5 Fig. 2: five conv layers", "Task 2–3", "layer/loss 검증"),
    ),
    cells=_classification_cells(
        _concept(
            r"b^i_{x,y}=a^i_{x,y}\left(k+\alpha\sum_j(a^j_{x,y})^2\right)^{-\beta}",
            "a는 ReLU 활성값, j는 이웃 channel, k·α·β는 LRN 상수, b는 정규화 출력이다.",
            "Task 1의 local_response_norm이 수식을 직접 계산하고 Task 2가 5-conv를 조립한다.",
            "입력 [B,1,16,16] → 두 번의 overlap pooling → [B,12,4,4] → logits [B,4].",
            "224×224 입력, data augmentation, multi-GPU는 제외하고 구조적 선택만 검증한다.",
        ),
        """
        # TODO 1: AlexNet 설정과 논문 Eq. (1)의 channel-local LRN을 구현하세요.
        @dataclass
        class AlexNetConfig:
            pass

        def local_response_norm(inputs, size, alpha, beta, k):
            raise NotImplementedError
        """,
        """
        @dataclass
        class AlexNetConfig:
            channels: tuple = (8, 12, 16, 16, 12)
            num_classes: int = 4
            lrn_size: int = 5
            learning_rate: float = 0.01
            steps: int = 12

        def local_response_norm(inputs, size, alpha, beta, k):
            half = size // 2
            squared = inputs.square().movedim(1, -1)
            squared = F.pad(squared, (half, half))
            local_energy = squared.unfold(-1, size, 1).sum(dim=-1)
            scale = (k + alpha * local_energy).pow(beta)
            return (inputs.movedim(1, -1) / scale).movedim(-1, 1)

        config = AlexNetConfig()
        spike = torch.zeros(1, 7, 1, 1, device=DEVICE)
        spike[:, 3] = 2.0
        normalized = local_response_norm(
            spike,
            config.lrn_size,
            1e-4,
            0.75,
            2.0,
        )
        expected = 2.0 / (2.0 + 4e-4) ** 0.75
        assert torch.allclose(normalized[0, 3, 0, 0], spike.new_tensor(expected))
        """,
        """
        # TODO 2: 다섯 convolution과 overlapping pooling을 명시적으로 구현하세요.
        class AlexNetMini(nn.Module):
            def __init__(self, config):
                super().__init__()

            def activate(self, inputs):
                '''AlexNet §3.1의 ReLU를 적용한다.'''
                raise NotImplementedError

            def forward_features(self, inputs):
                '''다섯 합성곱 층의 특징 추출 경로를 반환한다.'''
                raise NotImplementedError

            def forward(self, inputs):
                raise NotImplementedError
        """,
        """
        class AlexNetMini(nn.Module):
            def __init__(self, config):
                super().__init__()
                c1, c2, c3, c4, c5 = config.channels
                self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
                self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
                self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
                self.conv4 = nn.Conv2d(c3, c4, 3, padding=1)
                self.conv5 = nn.Conv2d(c4, c5, 3, padding=1)
                self.classifier = nn.Linear(c5, config.num_classes)
                self.config = config

            def activate(self, inputs):
                return F.relu(inputs)

            def forward_features(self, inputs):
                features = self.activate(self.conv1(inputs))
                features = local_response_norm(
                    features,
                    self.config.lrn_size,
                    1e-4,
                    0.75,
                    2.0,
                )
                features = F.max_pool2d(features, 3, stride=2, padding=1)
                features = self.activate(self.conv2(features))
                features = F.max_pool2d(features, 3, stride=2, padding=1)
                features = self.activate(self.conv3(features))
                features = self.activate(self.conv4(features))
                return self.activate(self.conv5(features))

            def forward(self, inputs):
                features = self.forward_features(inputs)
                return self.classifier(features.mean(dim=(2, 3)))

        model = AlexNetMini(config).to(DEVICE)
        assert model.config.lrn_size == config.lrn_size == 5
        assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 5
        assert model(images[:2]).shape == (2, 4)
        """,
        "AlexNet mini",
    ),
)


VGG = _paper(
    number=2,
    slug="vgg",
    short_title="VGG",
    paper_title="Very Deep Convolutional Networks for Large-Scale Image Recognition",
    authors="Karen Simonyan, Andrew Zisserman",
    year=2014,
    primary_url="https://arxiv.org/abs/1409.1556",
    venue="ICLR 2015",
    reproduction_goal="연속 3×3 convolution의 receptive field와 2/2/3 block을 재현한다.",
    original_scale="16/19 weight layers와 ImageNet 대신 7-conv, 16×16 합성 도형을 사용한다.",
    mappings=(
        ("§2.1: stride-1 3×3 convolution", "Task 1", "receptive field assert"),
        ("§2.2 Table 1: configurations", "Task 2", "2/2/3 conv blocks"),
        ("§2.3: small filters", "Task 1", "3×3 stack/7×7 parameters"),
        ("§3: classification framework", "Task 3", "loss/accuracy"),
    ),
    cells=_classification_cells(
        _concept(
            r"R_L=1+\sum_{\ell=1}^{L}(k_\ell-1)",
            "R_L은 누적 receptive field, L은 층 수, k_l은 각 convolution kernel 크기다.",
            "Task 1이 receptive field와 parameter 수를 비교하고 Task 2가 2/2/3 block을 만든다.",
            "입력 [B,1,16,16] → block별 8,4,2 해상도 → GAP [B,16] → [B,4].",
            "VGG-16의 13 conv 대신 7 conv를 쓰지만 작은 kernel을 깊게 쌓는 원칙은 동일하다.",
        ),
        """
        # TODO 1: VGG 설정, receptive field, convolution parameter 함수를 구현하세요.
        @dataclass
        class VGGConfig:
            pass

        def receptive_field(kernels):
            raise NotImplementedError

        def convolution_parameters(in_channels, out_channels, kernel):
            raise NotImplementedError
        """,
        """
        @dataclass
        class VGGConfig:
            block_depths: tuple = (2, 2, 3)
            block_channels: tuple = (8, 12, 16)
            num_classes: int = 4
            learning_rate: float = 0.01
            steps: int = 12

        def receptive_field(kernels):
            return 1 + sum(kernel - 1 for kernel in kernels)

        def convolution_parameters(in_channels, out_channels, kernel):
            return in_channels * out_channels * kernel * kernel

        config = VGGConfig()
        stacked = 3 * convolution_parameters(64, 64, 3)
        single = convolution_parameters(64, 64, 7)
        assert receptive_field([3, 3, 3]) == receptive_field([7]) == 7
        assert stacked < single
        """,
        """
        # TODO 2: VGG block builder와 VGGMini를 구현하세요.
        def make_vgg_block(in_channels, out_channels, depth):
            raise NotImplementedError

        class VGGMini(nn.Module):
            def __init__(self, config):
                super().__init__()

            def forward_features(self, inputs):
                '''VGG 합성곱 블록의 특징 맵을 반환한다.'''
                raise NotImplementedError

            def forward(self, inputs):
                raise NotImplementedError
        """,
        """
        def make_vgg_block(in_channels, out_channels, depth):
            layers = []
            for index in range(depth):
                current_in = in_channels if index == 0 else out_channels
                layers.append(nn.Conv2d(current_in, out_channels, 3, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            return nn.Sequential(*layers)

        class VGGMini(nn.Module):
            def __init__(self, config):
                super().__init__()
                blocks = []
                in_channels = 1
                for depth, out_channels in zip(
                    config.block_depths,
                    config.block_channels,
                ):
                    blocks.append(make_vgg_block(in_channels, out_channels, depth))
                    in_channels = out_channels
                self.features = nn.Sequential(*blocks)
                self.classifier = nn.Linear(in_channels, config.num_classes)

            def forward_features(self, inputs):
                return self.features(inputs)

            def forward(self, inputs):
                features = self.forward_features(inputs)
                return self.classifier(features.mean(dim=(2, 3)))

        model = VGGMini(config).to(DEVICE)
        conv_count = sum(isinstance(layer, nn.Conv2d) for layer in model.modules())
        assert conv_count == 7
        assert model(images[:2]).shape == (2, 4)
        """,
        "VGG mini",
    ),
)


INCEPTION = _paper(
    number=3,
    slug="googlenet_inception",
    short_title="GoogLeNet / Inception",
    paper_title="Going Deeper with Convolutions",
    authors="Christian Szegedy et al.",
    year=2014,
    primary_url="https://arxiv.org/abs/1409.4842",
    venue="CVPR 2015",
    reproduction_goal="multi-scale 병렬 분기와 1×1 dimension reduction을 재현한다.",
    original_scale="22-layer GoogLeNet과 auxiliary head 대신 Inception module 두 개를 사용한다.",
    mappings=(
        ("§4 Fig. 2(a): naïve Inception", "Task 2 branches", "concat shape"),
        ("§4 Fig. 2(b): dimension reduction", "Task 1", "parameter 절감"),
        ("§5 Table 1: GoogLeNet", "Task 2", "두 modules"),
        ("§6.3: classification", "Task 3", "loss/accuracy"),
    ),
    cells=_classification_cells(
        _concept(
            (
                r"y=\operatorname{Concat}\!\left(f_{1\times1}(x),"
                r"f_{3\times3}(x),f_{5\times5}(x),f_p(x)\right)"
            ),
            "각 f는 서로 다른 공간 척도의 branch이고 Concat은 channel 축 결합이다.",
            "Task 1은 1×1 reduction 비용을 계산하고 Task 2가 네 branch를 각각 구현한다.",
            "입력 [B,C,H,W] → 네 branch [B,4/6/3/3,H,W] → [B,16,H,W].",
            "원 논문의 전체 stage와 auxiliary classifier는 제외하고 module 내부 계산은 보존한다.",
        ),
        """
        # TODO 1: Inception 설정과 reduction 전후 parameter 계산 함수를 구현하세요.
        @dataclass
        class InceptionConfig:
            pass

        def branch_parameter_count(in_channels, reduced_channels, out_channels, kernel):
            raise NotImplementedError
        """,
        """
        @dataclass
        class InceptionConfig:
            stem_channels: int = 8
            branch_channels: tuple = (4, 6, 3, 3)
            num_classes: int = 4
            learning_rate: float = 0.012
            steps: int = 12

        def branch_parameter_count(in_channels, reduced_channels, out_channels, kernel):
            reduction = in_channels * reduced_channels
            spatial = reduced_channels * out_channels * kernel * kernel
            return reduction + spatial

        config = InceptionConfig()
        direct = 32 * 16 * 5 * 5
        reduced = branch_parameter_count(32, 4, 16, 5)
        assert reduced < direct
        assert reduced / direct < 0.2
        """,
        """
        # TODO 2: 네 branch의 InceptionBlock과 분류 모델을 구현하세요.
        class InceptionBlock(nn.Module):
            def __init__(self, in_channels, config):
                super().__init__()

            def forward(self, inputs):
                raise NotImplementedError

        class InceptionMini(nn.Module):
            def __init__(self, config):
                super().__init__()

            def forward(self, inputs):
                raise NotImplementedError
        """,
        """
        class InceptionBlock(nn.Module):
            def __init__(self, in_channels, config):
                super().__init__()
                c1, c3, c5, cp = config.branch_channels
                self.branch1 = nn.Conv2d(in_channels, c1, 1)
                self.reduce3 = nn.Conv2d(in_channels, 4, 1)
                self.branch3 = nn.Conv2d(4, c3, 3, padding=1)
                self.reduce5 = nn.Conv2d(in_channels, 2, 1)
                self.branch5 = nn.Conv2d(2, c5, 5, padding=2)
                self.pool_projection = nn.Conv2d(in_channels, cp, 1)

            def forward(self, inputs):
                branch1 = F.relu(self.branch1(inputs))
                branch3 = F.relu(self.reduce3(inputs))
                branch3 = F.relu(self.branch3(branch3))
                branch5 = F.relu(self.reduce5(inputs))
                branch5 = F.relu(self.branch5(branch5))
                pooled = F.max_pool2d(inputs, 3, stride=1, padding=1)
                pooled = F.relu(self.pool_projection(pooled))
                return torch.cat((branch1, branch3, branch5, pooled), dim=1)

        class InceptionMini(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.stem = nn.Conv2d(1, config.stem_channels, 3, padding=1)
                self.block1 = InceptionBlock(config.stem_channels, config)
                self.block2 = InceptionBlock(sum(config.branch_channels), config)
                self.classifier = nn.Linear(16, config.num_classes)

            def forward(self, inputs):
                features = F.relu(self.stem(inputs))
                features = self.block1(features)
                features = self.block2(features)
                return self.classifier(features.mean(dim=(2, 3)))

        model = InceptionMini(config).to(DEVICE)
        assert model.block1(images.new_zeros(2, 8, 16, 16)).shape == (2, 16, 16, 16)
        assert model(images[:2]).shape == (2, 4)
        """,
        "Inception mini",
    ),
)


UNET = _paper(
    number=4,
    slug="unet",
    short_title="U-Net",
    paper_title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
    authors="Olaf Ronneberger, Philipp Fischer, Thomas Brox",
    year=2015,
    primary_url="https://arxiv.org/abs/1505.04597",
    venue="MICCAI 2015",
    reproduction_goal="contracting/expanding path와 skip concatenation으로 mask를 분할한다.",
    original_scale=(
        "572×572 biomedical images, valid convolution, cropping, elastic deformation 대신 "
        "16×16 합성 mask와 same-padding 1-level U-Net을 사용한다."
    ),
    mappings=(
        ("§2 Fig. 1: contracting/expansive path", "Task 1 MiniUNet", "입출력 해상도"),
        (
            "§2: cropped feature-map concatenation",
            "Task 1 skip concat",
            "decoder channels",
        ),
        ("§2.1 Eq. (1): pixel-wise softmax/CE", "Task 2 BCE 이진 대응", "loss 감소"),
        ("§3 Fig. 3: overlap quality", "Task 3 Dice", "mask plot"),
    ),
    cells=(
        _concept(
            (
                r"\operatorname{Dice}(P,Y)="
                r"\frac{2\sum_i P_iY_i+\epsilon}{\sum_iP_i+\sum_iY_i+\epsilon}"
            ),
            "P는 예측 mask, Y는 정답 mask, i는 pixel, ε은 0 나눗셈 방지 상수다.",
            "Task 1은 encoder/decoder와 skip을 만들고 Task 2가 BCE, Task 3이 Dice를 구현한다.",
            "입력 [B,1,16,16] → bottleneck [B,12,8,8] → logits [B,1,16,16].",
            "논문의 4-level 깊이와 crop은 줄였지만 localization skip과 pixel loss는 보존한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정, double-conv block, skip U-Net class를 구현하세요.
            @dataclass
            class UNetConfig:
                pass

            def make_double_conv(in_channels, out_channels):
                raise NotImplementedError

            class MiniUNet(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def encode(self, inputs):
                    '''Skip과 저해상도 특징 맵을 튜플로 반환한다.'''
                    raise NotImplementedError

                def forward(self, inputs):
                    raise NotImplementedError
            """,
            """
            @dataclass
            class UNetConfig:
                encoder_channels: int = 6
                bottleneck_channels: int = 12
                learning_rate: float = 0.02
                steps: int = 16

            def make_double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(),
                )

            class MiniUNet(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    c1 = config.encoder_channels
                    c2 = config.bottleneck_channels
                    self.encoder = make_double_conv(1, c1)
                    self.bottleneck = make_double_conv(c1, c2)
                    self.upsample = nn.ConvTranspose2d(c2, c1, 2, stride=2)
                    self.decoder = make_double_conv(c1 * 2, c1)
                    self.mask_head = nn.Conv2d(c1, 1, kernel_size=1)

                def encode(self, inputs):
                    skip = self.encoder(inputs)
                    low = self.bottleneck(F.max_pool2d(skip, 2))
                    return skip, low

                def forward(self, inputs):
                    skip, low = self.encode(inputs)
                    upsampled = self.upsample(low)
                    decoded = self.decoder(torch.cat((upsampled, skip), dim=1))
                    return self.mask_head(decoded)

            config = UNetConfig()
            model = MiniUNet(config).to(DEVICE)
            assert model(images[:2]).shape == (2, 1, 16, 16)
            assert model.decoder[0].in_channels == 12
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 2: pixel BCE와 한 학습 step을 별도 함수로 구현하세요.
            def segmentation_loss(logits, targets):
                raise NotImplementedError

            def train_segmentation_step(model, optimizer, inputs, targets):
                raise NotImplementedError
            """,
            """
            def segmentation_loss(logits, targets):
                return F.binary_cross_entropy_with_logits(logits, targets)

            def train_segmentation_step(model, optimizer, inputs, targets):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = segmentation_loss(logits, targets)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            target_masks = masks[train_idx, None]
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                loss = train_segmentation_step(
                    model,
                    optimizer,
                    x_train,
                    target_masks,
                )
                losses.append(loss)
            assert min(losses[3:]) < losses[0]
            """,
            "todo",
            "training",
        ),
        code(
            """
            # TODO 3: Dice 함수와 test 평가/시각화를 구현하세요.
            def dice_score(predictions, targets, epsilon=1.0):
                raise NotImplementedError

            def evaluate_segmentation(model, inputs, targets):
                raise NotImplementedError
            """,
            """
            def dice_score(predictions, targets, epsilon=1.0):
                dimensions = (1, 2, 3)
                intersection = (predictions * targets).sum(dim=dimensions)
                total = predictions.sum(dim=dimensions) + targets.sum(dim=dimensions)
                return ((2 * intersection + epsilon) / (total + epsilon)).mean()

            def evaluate_segmentation(model, inputs, targets):
                model.eval()
                with torch.no_grad():
                    probabilities = model(inputs).sigmoid()
                    predictions = (probabilities > 0.5).float()
                return float(dice_score(predictions, targets)), predictions

            truth = masks[test_idx, None]
            dice, predictions = evaluate_segmentation(model, images[test_idx], truth)
            assert 0.0 <= dice <= 1.0
            figure, axes = plt.subplots(1, 3, figsize=(8, 2.7))
            axes[0].plot(losses)
            axes[0].set_title("pixel BCE")
            axes[1].imshow(truth[0, 0].detach().cpu())
            axes[1].set_title("target")
            axes[2].imshow(predictions[0, 0].detach().cpu())
            axes[2].set_title(f"prediction Dice={dice:.2f}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "evaluation",
            "visualization",
        ),
    ),
)


RESNET = _paper(
    number=5,
    slug="resnet",
    short_title="ResNet",
    paper_title="Deep Residual Learning for Image Recognition",
    authors="Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
    year=2015,
    primary_url="https://arxiv.org/abs/1512.03385",
    venue="CVPR 2016",
    reproduction_goal="residual mapping과 identity/projection shortcut을 직접 구현한다.",
    original_scale="ImageNet 152-layer/CIFAR 1,000-layer 대신 4개 basic block을 사용한다.",
    mappings=(
        ("§3.1 Eq. (1): H(x)=F(x)+x", "Task 1", "zero residual identity"),
        ("§3.1 Eq. (2)", "Task 1 block", "shape assert"),
        ("§3.2 Fig. 2: parameter-free shortcut", "Task 1", "shortcut parameter 0"),
        ("§4.2 Fig. 4/6: optimization", "Task 3", "loss curve"),
    ),
    cells=_classification_cells(
        _concept(
            r"\mathbf{y}=\mathcal{F}(\mathbf{x},\{W_i\})+W_s\mathbf{x}",
            "F는 학습할 residual branch, x는 shortcut 입력, W_s는 shape 조정 projection이다.",
            "Task 1이 config와 shortcut 조건을 정의하고 Task 2가 ResidualBlock을 구현한다.",
            "입력 [B,1,16,16] → stem [B,8,16,16] → 4 blocks → GAP → [B,4].",
            "BatchNorm과 downsampling stage는 생략하지만 덧셈 경로와 zero-residual 성질은 보존한다.",
        ),
        """
        # TODO 1: ResNet 설정과 projection 필요 여부 함수를 구현하세요.
        @dataclass
        class ResNetConfig:
            pass

        def needs_projection(in_channels, out_channels, stride):
            raise NotImplementedError
        """,
        """
        @dataclass
        class ResNetConfig:
            channels: int = 8
            blocks: int = 4
            num_classes: int = 4
            learning_rate: float = 0.012
            steps: int = 14

        def needs_projection(in_channels, out_channels, stride):
            return in_channels != out_channels or stride != 1

        config = ResNetConfig()
        assert not needs_projection(8, 8, 1)
        assert needs_projection(8, 16, 2)
        """,
        """
        # TODO 2: residual branch와 shortcut을 분리한 block/model을 구현하세요.
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()

            def residual(self, inputs):
                '''두 합성곱 층으로 residual branch F(x)를 계산한다.'''
                raise NotImplementedError

            def forward(self, inputs):
                raise NotImplementedError

        class ResNetMini(nn.Module):
            def __init__(self, config):
                super().__init__()

            def forward(self, inputs):
                raise NotImplementedError
        """,
        """
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
                self.conv2 = nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    bias=False,
                )
                if needs_projection(in_channels, out_channels, stride):
                    self.shortcut = nn.Conv2d(
                        in_channels,
                        out_channels,
                        1,
                        stride=stride,
                        bias=False,
                    )
                else:
                    self.shortcut = nn.Identity()

            def residual(self, inputs):
                hidden = F.relu(self.conv1(inputs))
                return self.conv2(hidden)

            def forward(self, inputs):
                return F.relu(self.residual(inputs) + self.shortcut(inputs))

        class ResNetMini(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.stem = nn.Conv2d(1, config.channels, 3, padding=1)
                self.blocks = nn.Sequential(
                    *[
                        ResidualBlock(config.channels, config.channels)
                        for _ in range(config.blocks)
                    ]
                )
                self.classifier = nn.Linear(config.channels, config.num_classes)

            def forward(self, inputs):
                features = F.relu(self.stem(inputs))
                features = self.blocks(features)
                return self.classifier(features.mean(dim=(2, 3)))

        config = ResNetConfig()
        model = ResNetMini(config).to(DEVICE)
        identity_block = ResidualBlock(4, 4).to(DEVICE)
        nn.init.zeros_(identity_block.conv1.weight)
        nn.init.zeros_(identity_block.conv2.weight)
        probe = torch.rand(2, 4, 8, 8, device=DEVICE)
        assert torch.allclose(identity_block(probe), probe)
        assert model(images[:2]).shape == (2, 4)
        """,
        "ResNet mini",
    ),
)


FASTER_RCNN = _paper(
    number=6,
    slug="faster_rcnn",
    short_title="Faster R-CNN",
    paper_title="Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
    authors="Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun",
    year=2015,
    primary_url="https://arxiv.org/abs/1506.01497",
    venue="NeurIPS 2015",
    reproduction_goal="RPN anchor, IoU assignment, objectness+box multi-task loss를 재현한다.",
    original_scale="VGG-16, 9 anchors/location, VOC/COCO 대신 4×4 단일-scale anchor grid를 쓴다.",
    mappings=(
        ("§3.1 Fig. 3: sliding RPN", "Task 1 anchor grid", "16 anchors"),
        ("§3.1.1: anchors", "Task 1 IoU", "best anchor"),
        ("§3.1.2 Eq. (1): Lcls+λLreg", "Task 2", "finite loss"),
        ("§3.1.2 Eq. (2): box parameterization", "Task 1 delta", "identity delta"),
    ),
    cells=(
        _concept(
            (
                r"L=\frac{1}{N_{cls}}\sum_iL_{cls}(p_i,p_i^*)"
                r"+\lambda\frac{1}{N_{reg}}\sum_i p_i^*L_{reg}(t_i,t_i^*)"
            ),
            "p는 objectness, p*는 anchor label, t는 box delta, λ는 regression 가중치다.",
            "Task 1이 anchor/IoU/box encoding, Task 2가 RPNHead, Task 3이 multi-task loss를 맡는다.",
            "입력 [B,1,16,16] → feature [B,12,4,4] → object [B,16], delta [B,16,4].",
            "두 번째 Fast R-CNN stage와 NMS는 제외하고 논문의 RPN 학습 경로를 깊게 구현한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: config, anchor grid, IoU, box delta encoding을 구현하세요.
            @dataclass
            class RPNConfig:
                pass

            def generate_anchors(grid_size, box_size, device):
                raise NotImplementedError

            def pairwise_iou(first, second):
                raise NotImplementedError

            def encode_box(anchor, target):
                raise NotImplementedError
            """,
            """
            @dataclass
            class RPNConfig:
                grid_size: int = 4
                box_size: float = 0.35
                feature_channels: int = 12
                learning_rate: float = 0.02
                steps: int = 18
                regression_weight: float = 1.0

            def generate_anchors(grid_size, box_size, device):
                centers = torch.linspace(
                    0.5 / grid_size,
                    1.0 - 0.5 / grid_size,
                    grid_size,
                    device=device,
                )
                grid_y, grid_x = torch.meshgrid(centers, centers, indexing="ij")
                half = box_size / 2
                anchors = torch.stack(
                    (grid_x - half, grid_y - half, grid_x + half, grid_y + half),
                    dim=-1,
                )
                return anchors.reshape(-1, 4).clamp(0.0, 1.0)

            def pairwise_iou(first, second):
                left_top = torch.maximum(first[:, None, :2], second[None, :, :2])
                right_bottom = torch.minimum(first[:, None, 2:], second[None, :, 2:])
                intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
                first_area = (first[:, 2:] - first[:, :2]).prod(dim=-1)[:, None]
                second_area = (second[:, 2:] - second[:, :2]).prod(dim=-1)[None]
                union = first_area + second_area - intersection
                return intersection / union.clamp_min(1e-8)

            def encode_box(anchor, target):
                anchor_center = (anchor[:2] + anchor[2:]) / 2
                anchor_size = anchor[2:] - anchor[:2]
                target_center = (target[:2] + target[2:]) / 2
                target_size = target[2:] - target[:2]
                center_delta = (target_center - anchor_center) / anchor_size
                size_delta = torch.log(target_size / anchor_size)
                return torch.cat((center_delta, size_delta))

            config = RPNConfig()
            anchors = generate_anchors(config.grid_size, config.box_size, DEVICE)
            assert anchors.shape == (16, 4)
            assert torch.allclose(encode_box(anchors[0], anchors[0]), anchors.new_zeros(4))
            """,
            "todo",
            "anchors",
            "equation",
        ),
        code(
            """
            # TODO 2: shared feature와 두 sibling head를 갖는 RegionProposalNetwork를 구현하세요.
            class RegionProposalNetwork(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def forward(self, inputs):
                    raise NotImplementedError
            """,
            """
            class RegionProposalNetwork(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    channels = config.feature_channels
                    self.backbone = nn.Sequential(
                        nn.Conv2d(1, channels, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(channels, channels, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    self.shared = nn.Conv2d(channels, channels, 3, padding=1)
                    self.objectness = nn.Conv2d(channels, 1, kernel_size=1)
                    self.box_regression = nn.Conv2d(channels, 4, kernel_size=1)

                def forward(self, inputs):
                    features = F.relu(self.shared(self.backbone(inputs)))
                    object_logits = self.objectness(features).flatten(start_dim=1)
                    box_deltas = self.box_regression(features)
                    box_deltas = box_deltas.permute(0, 2, 3, 1).reshape(len(inputs), -1, 4)
                    return object_logits, box_deltas

            model = RegionProposalNetwork(config).to(DEVICE)
            probe_object, probe_delta = model(images[:2])
            assert probe_object.shape == (2, 16)
            assert probe_delta.shape == (2, 16, 4)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: anchor target, RPN loss, train/evaluate 흐름을 구현하세요.
            def build_rpn_targets(anchors, target_boxes):
                raise NotImplementedError

            def rpn_loss(prediction, targets, config):
                raise NotImplementedError

            def train_rpn_step(model, optimizer, inputs, targets, config):
                raise NotImplementedError

            def evaluate_rpn(model, inputs, object_targets):
                '''가장 높은 objectness를 갖는 anchor의 recall을 반환한다.'''
                raise NotImplementedError
            """,
            """
            def build_rpn_targets(anchors, target_boxes):
                object_targets = []
                delta_targets = []
                for target_box in target_boxes:
                    overlaps = pairwise_iou(anchors, target_box[None]).squeeze(1)
                    positive_index = int(overlaps.argmax())
                    object_target = torch.zeros(len(anchors), device=anchors.device)
                    object_target[positive_index] = 1.0
                    delta_target = torch.zeros(len(anchors), 4, device=anchors.device)
                    delta_target[positive_index] = encode_box(
                        anchors[positive_index],
                        target_box,
                    )
                    object_targets.append(object_target)
                    delta_targets.append(delta_target)
                return torch.stack(object_targets), torch.stack(delta_targets)

            def rpn_loss(prediction, targets, config):
                object_logits, box_deltas = prediction
                object_targets, delta_targets = targets
                classification = F.binary_cross_entropy_with_logits(
                    object_logits,
                    object_targets,
                )
                positive = object_targets.bool()
                regression = F.smooth_l1_loss(
                    box_deltas[positive],
                    delta_targets[positive],
                )
                return classification + config.regression_weight * regression

            def train_rpn_step(model, optimizer, inputs, targets, config):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = rpn_loss(model(inputs), targets, config)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_rpn(model, inputs, object_targets):
                model.eval()
                with torch.no_grad():
                    object_logits, _ = model(inputs)
                predicted = object_logits.argmax(dim=1)
                expected = object_targets.argmax(dim=1)
                return float((predicted == expected).float().mean())

            targets = build_rpn_targets(anchors, boxes[train_idx])
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                loss = train_rpn_step(model, optimizer, x_train, targets, config)
                losses.append(loss)
            test_targets = build_rpn_targets(anchors, boxes[test_idx])
            recall = evaluate_rpn(model, images[test_idx], test_targets[0])
            assert min(losses[3:]) < losses[0]
            assert 0.0 <= recall <= 1.0
            plt.figure(figsize=(5, 3))
            plt.plot(losses)
            plt.title(f"RPN best-anchor recall={recall:.1%}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


YOLO = _paper(
    number=7,
    slug="yolov1",
    short_title="YOLOv1",
    paper_title="You Only Look Once: Unified, Real-Time Object Detection",
    authors="Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi",
    year=2015,
    primary_url="https://arxiv.org/abs/1506.02640",
    venue="CVPR 2016",
    reproduction_goal="grid target encoding과 coordinate/object/class 결합 손실을 재현한다.",
    original_scale="S=7, B=2, C=20과 24-conv backbone 대신 S=4, B=1, C=4를 사용한다.",
    mappings=(
        ("§2 Fig. 2: S×S×(B·5+C)", "Task 1 encoder", "4×4×9"),
        ("§2 Eq. (1): class confidence", "Task 1", "product assert"),
        ("§2.2 Eq. (3): λcoord=5, λnoobj=.5", "Task 2", "perfect loss 0"),
        ("§2.3: single evaluation", "Task 3", "loss optimization"),
    ),
    cells=(
        _concept(
            r"L=\lambda_{coord}L_{xywh}+L_{obj}+\lambda_{noobj}L_{noobj}+L_{class}",
            "xywh는 box, obj/noobj는 confidence 항, class는 조건부 class 확률 항이다.",
            "Task 1이 grid target, Task 2가 YOLONetwork, Task 3이 네 손실 항을 직접 계산한다.",
            "입력 [B,1,16,16] → prediction [B,4,4,9], 마지막 9=4 box/conf+4 classes.",
            "B=1과 작은 backbone이라 다중 물체/NMS는 제외하지만 unified loss는 보존한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: YOLO 설정과 normalized box를 grid tensor로 encoding하는 함수를 구현하세요.
            @dataclass
            class YOLOConfig:
                pass

            def encode_yolo_targets(boxes, labels, config):
                raise NotImplementedError
            """,
            """
            @dataclass
            class YOLOConfig:
                grid_size: int = 4
                num_classes: int = 4
                coordinate_weight: float = 5.0
                no_object_weight: float = 0.5
                learning_rate: float = 0.015
                steps: int = 18

            def encode_yolo_targets(boxes, labels, config):
                batch = len(boxes)
                channels = 5 + config.num_classes
                target = torch.zeros(
                    batch,
                    config.grid_size,
                    config.grid_size,
                    channels,
                    device=boxes.device,
                )
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                sizes = boxes[:, 2:] - boxes[:, :2]
                grid_xy = (centers * config.grid_size).long()
                grid_xy = grid_xy.clamp(0, config.grid_size - 1)
                rows = grid_xy[:, 1]
                columns = grid_xy[:, 0]
                offsets = centers * config.grid_size - grid_xy.float()
                indices = torch.arange(batch, device=boxes.device)
                target[indices, rows, columns, :2] = offsets
                target[indices, rows, columns, 2:4] = sizes.sqrt()
                target[indices, rows, columns, 4] = 1.0
                target[indices, rows, columns, 5 + labels] = 1.0
                return target

            config = YOLOConfig()
            encoded = encode_yolo_targets(boxes[:2], labels[:2], config)
            assert encoded.shape == (2, 4, 4, 9)
            assert torch.allclose(encoded[..., 4].sum(dim=(1, 2)), encoded.new_ones(2))
            """,
            "todo",
            "targets",
        ),
        code(
            """
            # TODO 2: 한 번의 forward로 전체 grid를 예측하는 YOLONetwork를 구현하세요.
            class YOLONetwork(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def forward(self, inputs):
                    raise NotImplementedError
            """,
            """
            class YOLONetwork(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    output_channels = 5 + config.num_classes
                    self.backbone = nn.Sequential(
                        nn.Conv2d(1, 12, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(12, 20, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(20, 20, 3, padding=1),
                        nn.ReLU(),
                    )
                    self.prediction = nn.Conv2d(20, output_channels, kernel_size=1)

                def forward(self, inputs):
                    features = self.backbone(inputs)
                    prediction = self.prediction(features)
                    return prediction.permute(0, 2, 3, 1)

            model = YOLONetwork(config).to(DEVICE)
            assert model(images[:2]).shape == (2, 4, 4, 9)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: YOLO 손실의 네 항, train step, 평가를 구현하세요.
            def yolo_loss(prediction, target, config):
                raise NotImplementedError

            def train_yolo_step(model, optimizer, inputs, target, config):
                raise NotImplementedError

            def evaluate_yolo(model, inputs, target):
                raise NotImplementedError
            """,
            """
            def yolo_loss(prediction, target, config):
                object_mask = target[..., 4] > 0
                no_object_mask = ~object_mask
                coordinate = F.mse_loss(
                    prediction[..., :4][object_mask],
                    target[..., :4][object_mask],
                )
                object_loss = F.mse_loss(
                    prediction[..., 4][object_mask],
                    target[..., 4][object_mask],
                )
                no_object = F.mse_loss(
                    prediction[..., 4][no_object_mask],
                    target[..., 4][no_object_mask],
                )
                class_loss = F.mse_loss(
                    prediction[..., 5:][object_mask],
                    target[..., 5:][object_mask],
                )
                return (
                    config.coordinate_weight * coordinate
                    + object_loss
                    + config.no_object_weight * no_object
                    + class_loss
                )

            def train_yolo_step(model, optimizer, inputs, target, config):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = yolo_loss(model(inputs), target, config)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_yolo(model, inputs, target):
                model.eval()
                with torch.no_grad():
                    prediction = model(inputs)
                    predicted_cells = prediction[..., 4].flatten(1).argmax(dim=1)
                expected_cells = target[..., 4].flatten(1).argmax(dim=1)
                return float((predicted_cells == expected_cells).float().mean())

            target = encode_yolo_targets(boxes[train_idx], y_train, config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(train_yolo_step(model, optimizer, x_train, target, config))
            test_target = encode_yolo_targets(boxes[test_idx], labels[test_idx], config)
            cell_accuracy = evaluate_yolo(model, images[test_idx], test_target)
            assert min(losses[3:]) < losses[0]
            assert 0.0 <= cell_accuracy <= 1.0
            plt.figure(figsize=(5, 3))
            plt.plot(losses)
            plt.title(f"YOLO cell accuracy={cell_accuracy:.1%}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


VIT = _paper(
    number=8,
    slug="vision_transformer",
    short_title="Vision Transformer (ViT)",
    paper_title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    authors="Alexey Dosovitskiy et al.",
    year=2020,
    primary_url="https://arxiv.org/abs/2010.11929",
    venue="ICLR 2021",
    reproduction_goal="patch embedding, class token, position embedding, pre-LN encoder를 재현한다.",
    original_scale="224×224/patch16, 대규모 사전학습 대신 16×16/patch4와 1 encoder layer를 쓴다.",
    mappings=(
        ("§3.1 Fig. 1: patches as tokens", "Task 1 patchify", "16 patches"),
        (
            "§3.1 Eq. (1–4): class/position embedding and LN/MSA/MLP residual",
            "Task 2 explicit Q/K/V pre-LN encoder",
            "17 tokens and [B,H,17,17] attention weights",
        ),
        ("§3.2: fine-tuning", "Task 3", "loss/accuracy"),
    ),
    cells=_classification_cells(
        _concept(
            r"\mathbf{z}_0=[\mathbf{x}_{class};\mathbf{x}_p^1E;\cdots;\mathbf{x}_p^NE]+E_{pos}",
            "x_p는 펼친 patch, E는 projection, x_class는 학습 token, E_pos는 위치 embedding이다.",
            (
                "Task 1이 patchify/config, Task 2가 class token과 Q/K/V projection, "
                "head 분할, pre-LN residual encoder를 구현한다."
            ),
            "입력 [B,1,16,16] → patches [B,16,16] → tokens [B,17,24] → logits [B,4].",
            "대규모 사전학습과 hybrid CNN은 제외하지만 tokenization과 encoder 식은 유지한다.",
        ),
        """
        # TODO 1: ViT 설정과 unfold를 이용한 patchify를 구현하세요.
        @dataclass
        class ViTConfig:
            pass

            @property
            def patch_count(self):
                '''이미지를 분할할 겹치지 않는 patch 개수를 반환한다.'''
                raise NotImplementedError

        def patchify(images, patch_size):
            raise NotImplementedError
        """,
        """
        @dataclass
        class ViTConfig:
            image_size: int = 16
            patch_size: int = 4
            embedding_dim: int = 24
            heads: int = 4
            mlp_dim: int = 48
            num_classes: int = 4
            learning_rate: float = 0.012
            steps: int = 15

            @property
            def patch_count(self):
                return (self.image_size // self.patch_size) ** 2

        def patchify(images, patch_size):
            patches = images.unfold(2, patch_size, patch_size)
            patches = patches.unfold(3, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5)
            return patches.reshape(len(images), -1, patch_size * patch_size)

        config = ViTConfig()
        patches = patchify(images[:2], config.patch_size)
        assert patches.shape == (2, 16, 16)
        """,
        """
        # TODO 2: class token, position embedding, pre-LN residual encoder를 구현하세요.
        class ExplicitMultiHeadAttention(nn.Module):
            '''ViT Eq. (2)에 사용하는 multi-head self-attention이다.'''

            def __init__(self, model_dim, heads):
                super().__init__()

            def split_heads(self, tensor):
                '''[B, T, D]를 [B, H, T, d_h]로 분할한다.'''
                raise NotImplementedError

            def merge_heads(self, tensor):
                '''[B, H, T, d_h]를 [B, T, D]로 병합한다.'''
                raise NotImplementedError

            def forward(self, query, key, value, attention_mask=None):
                '''Attention output과 weights [B, H, T_q, T_k]를 반환한다.'''
                raise NotImplementedError

        class TransformerEncoderBlock(nn.Module):
            def __init__(self, config):
                super().__init__()

            def forward(self, tokens):
                raise NotImplementedError

        class VisionTransformerMini(nn.Module):
            def __init__(self, config):
                super().__init__()

            def embed(self, inputs):
                '''[B, 1, H, W]를 [B, N+1, D] token sequence로 변환한다.'''
                raise NotImplementedError

            def forward(self, inputs):
                raise NotImplementedError
        """,
        """
        class ExplicitMultiHeadAttention(nn.Module):
            '''Q/K/V projection부터 scaled dot-product attention까지 계산한다.'''

            def __init__(self, model_dim, heads):
                super().__init__()
                if model_dim % heads != 0:
                    raise ValueError("model_dim must be divisible by heads")
                self.heads = heads
                self.head_dim = model_dim // heads
                self.query_projection = nn.Linear(model_dim, model_dim)
                self.key_projection = nn.Linear(model_dim, model_dim)
                self.value_projection = nn.Linear(model_dim, model_dim)
                self.output_projection = nn.Linear(model_dim, model_dim)

            def split_heads(self, tensor):
                batch, length, _ = tensor.shape
                tensor = tensor.reshape(batch, length, self.heads, self.head_dim)
                return tensor.transpose(1, 2)

            def merge_heads(self, tensor):
                batch, _, length, _ = tensor.shape
                tensor = tensor.transpose(1, 2).contiguous()
                return tensor.reshape(batch, length, self.heads * self.head_dim)

            def forward(self, query, key, value, attention_mask=None):
                queries = self.split_heads(self.query_projection(query))
                keys = self.split_heads(self.key_projection(key))
                values = self.split_heads(self.value_projection(value))
                scale = self.head_dim**-0.5
                scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
                if attention_mask is not None:
                    scores = scores.masked_fill(attention_mask, float("-inf"))
                weights = scores.softmax(dim=-1)
                context = torch.matmul(weights, values)
                output = self.output_projection(self.merge_heads(context))
                return output, weights

        class TransformerEncoderBlock(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.norm1 = nn.LayerNorm(config.embedding_dim)
                self.attention = ExplicitMultiHeadAttention(
                    config.embedding_dim,
                    config.heads,
                )
                self.norm2 = nn.LayerNorm(config.embedding_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(config.embedding_dim, config.mlp_dim),
                    nn.GELU(),
                    nn.Linear(config.mlp_dim, config.embedding_dim),
                )

            def forward(self, tokens):
                normalized = self.norm1(tokens)
                attended, _ = self.attention(
                    normalized,
                    normalized,
                    normalized,
                )
                tokens = tokens + attended
                return tokens + self.mlp(self.norm2(tokens))

        class VisionTransformerMini(nn.Module):
            def __init__(self, config):
                super().__init__()
                patch_dim = config.patch_size * config.patch_size
                self.config = config
                self.patch_projection = nn.Linear(patch_dim, config.embedding_dim)
                self.class_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
                self.position = nn.Parameter(
                    torch.zeros(1, config.patch_count + 1, config.embedding_dim)
                )
                self.encoder = TransformerEncoderBlock(config)
                self.norm = nn.LayerNorm(config.embedding_dim)
                self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

            def embed(self, inputs):
                patches = patchify(inputs, self.config.patch_size)
                tokens = self.patch_projection(patches)
                class_tokens = self.class_token.expand(len(inputs), -1, -1)
                return torch.cat((class_tokens, tokens), dim=1) + self.position

            def forward(self, inputs):
                encoded = self.encoder(self.embed(inputs))
                return self.classifier(self.norm(encoded[:, 0]))

        model = VisionTransformerMini(config).to(DEVICE)
        embedded = model.embed(images[:2])
        normalized = model.encoder.norm1(embedded)
        _, attention_weights = model.encoder.attention(
            normalized,
            normalized,
            normalized,
        )
        assert embedded.shape == (2, 17, 24)
        assert attention_weights.shape == (2, 4, 17, 17)
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones_like(attention_weights[..., 0]),
        )
        assert model(images[:2]).shape == (2, 4)
        """,
        "Vision Transformer mini",
    ),
)


DETR = _paper(
    number=9,
    slug="detr",
    short_title="DETR",
    paper_title="End-to-End Object Detection with Transformers",
    authors="Nicolas Carion et al.",
    year=2020,
    primary_url="https://arxiv.org/abs/2005.12872",
    venue="ECCV 2020",
    reproduction_goal="object queries, bipartite matching, no-object class, set loss를 재현한다.",
    original_scale="ResNet-50, 100 queries, COCO 대신 작은 CNN, 3 queries, 단일 물체를 사용한다.",
    mappings=(
        ("§3.1 Eq. (1): Hungarian matching", "Task 1 tiny_match", "최소 permutation"),
        (
            "§3.2 Fig. 2: decoder object queries and image cross-attention",
            "Task 2 explicit query self/cross-attention",
            "[B,H,3,16] cross-attention weights",
        ),
        (
            "§3.1 Eq. (2) and §3.2: Hungarian set loss/no-object class",
            "Task 3",
            "set loss; GT-independent objectness query로 held-out class/box 평가",
        ),
    ),
    cells=(
        _concept(
            r"\hat{\sigma}=\arg\min_{\sigma}\sum_i\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})",
            "σ는 prediction과 target의 일대일 배정, L_match는 class와 box matching cost다.",
            (
                "Task 1이 matching cost, Task 2가 query self-attention과 image "
                "cross-attention, Task 3이 set loss를 구현한다."
            ),
            "입력 [B,1,16,16] → memory [B,16,24] → 3 query class [B,3,5], box [B,3,4].",
            (
                "단일 물체라 완전 Hungarian 알고리즘은 최소 query 선택으로 축소된다. "
                "학습 matching은 GT를 쓰지만 평가는 no-object 확률만으로 query를 먼저 고른다."
            ),
        ),
        _setup(),
        code(
            """
            # TODO 1: DETR 설정, box L1 cost, 단일-target matching을 구현하세요.
            @dataclass
            class DETRConfig:
                pass

            def box_l1_cost(predicted_boxes, target_boxes):
                raise NotImplementedError

            def match_single_target(class_logits, predicted_boxes, labels, target_boxes):
                raise NotImplementedError
            """,
            """
            @dataclass
            class DETRConfig:
                hidden_dim: int = 24
                heads: int = 4
                queries: int = 3
                num_classes: int = 4
                no_object_weight: float = 0.2
                box_weight: float = 5.0
                learning_rate: float = 0.015
                steps: int = 18

            def box_l1_cost(predicted_boxes, target_boxes):
                return (predicted_boxes - target_boxes[:, None]).abs().sum(dim=-1)

            def match_single_target(class_logits, predicted_boxes, labels, target_boxes):
                probabilities = class_logits.softmax(dim=-1)
                query_count = class_logits.shape[1]
                class_indices = labels[:, None, None].expand(-1, query_count, 1)
                class_cost = -probabilities.gather(2, class_indices).squeeze(-1)
                total_cost = class_cost + box_l1_cost(predicted_boxes, target_boxes)
                return total_cost.argmin(dim=1)

            config = DETRConfig()
            probe_logits = torch.zeros(2, 3, 5, device=DEVICE)
            probe_boxes = torch.rand(2, 3, 4, device=DEVICE)
            matched = match_single_target(probe_logits, probe_boxes, labels[:2], boxes[:2])
            assert matched.shape == (2,)
            """,
            "todo",
            "matching",
            "equation",
        ),
        code(
            """
            # TODO 2: CNN memory와 learned object query decoder를 갖는 DETRMini를 구현하세요.
            class ExplicitMultiHeadAttention(nn.Module):
                '''DETR decoder에서 공유하는 self/cross attention이다.'''

                def __init__(self, model_dim, heads):
                    super().__init__()

                def split_heads(self, tensor):
                    '''[B, T, D]를 [B, H, T, d_h]로 분할한다.'''
                    raise NotImplementedError

                def merge_heads(self, tensor):
                    '''[B, H, T, d_h]를 [B, T, D]로 병합한다.'''
                    raise NotImplementedError

                def forward(self, query, key, value, attention_mask=None):
                    '''Self/cross attention output과 head별 weights를 반환한다.'''
                    raise NotImplementedError

            class DETRDecoderBlock(nn.Module):
                '''Object query self-attention과 image cross-attention을 연결한다.'''

                def __init__(self, config):
                    super().__init__()

                def forward(self, queries, memory):
                    '''Decoded queries와 image cross-attention weights를 반환한다.'''
                    raise NotImplementedError

            class DETRMini(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def encode(self, inputs):
                    '''CNN feature map을 memory [B, 16, D]로 펼친다.'''
                    raise NotImplementedError

                def forward(self, inputs):
                    raise NotImplementedError
            """,
            """
            class ExplicitMultiHeadAttention(nn.Module):
                '''Q/K/V projection과 head 분할이 드러나는 attention이다.'''

                def __init__(self, model_dim, heads):
                    super().__init__()
                    if model_dim % heads != 0:
                        raise ValueError("model_dim must be divisible by heads")
                    self.heads = heads
                    self.head_dim = model_dim // heads
                    self.query_projection = nn.Linear(model_dim, model_dim)
                    self.key_projection = nn.Linear(model_dim, model_dim)
                    self.value_projection = nn.Linear(model_dim, model_dim)
                    self.output_projection = nn.Linear(model_dim, model_dim)

                def split_heads(self, tensor):
                    batch, length, _ = tensor.shape
                    tensor = tensor.reshape(
                        batch,
                        length,
                        self.heads,
                        self.head_dim,
                    )
                    return tensor.transpose(1, 2)

                def merge_heads(self, tensor):
                    batch, _, length, _ = tensor.shape
                    tensor = tensor.transpose(1, 2).contiguous()
                    return tensor.reshape(
                        batch,
                        length,
                        self.heads * self.head_dim,
                    )

                def forward(self, query, key, value, attention_mask=None):
                    queries = self.split_heads(self.query_projection(query))
                    keys = self.split_heads(self.key_projection(key))
                    values = self.split_heads(self.value_projection(value))
                    scale = self.head_dim**-0.5
                    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
                    if attention_mask is not None:
                        scores = scores.masked_fill(
                            attention_mask,
                            float("-inf"),
                        )
                    weights = scores.softmax(dim=-1)
                    context = torch.matmul(weights, values)
                    output = self.output_projection(self.merge_heads(context))
                    return output, weights

            class DETRDecoderBlock(nn.Module):
                '''Learned query를 self-attend한 뒤 image memory를 조회한다.'''

                def __init__(self, config):
                    super().__init__()
                    hidden = config.hidden_dim
                    self.self_attention = ExplicitMultiHeadAttention(
                        hidden,
                        config.heads,
                    )
                    self.cross_attention = ExplicitMultiHeadAttention(
                        hidden,
                        config.heads,
                    )
                    self.norm1 = nn.LayerNorm(hidden)
                    self.norm2 = nn.LayerNorm(hidden)
                    self.norm3 = nn.LayerNorm(hidden)
                    self.feedforward = nn.Sequential(
                        nn.Linear(hidden, hidden * 2),
                        nn.ReLU(),
                        nn.Linear(hidden * 2, hidden),
                    )

                def forward(self, queries, memory):
                    self_attended, _ = self.self_attention(
                        queries,
                        queries,
                        queries,
                    )
                    queries = self.norm1(queries + self_attended)
                    cross_attended, cross_weights = self.cross_attention(
                        queries,
                        memory,
                        memory,
                    )
                    queries = self.norm2(queries + cross_attended)
                    output = self.norm3(queries + self.feedforward(queries))
                    return output, cross_weights

            class DETRMini(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    hidden = config.hidden_dim
                    self.backbone = nn.Sequential(
                        nn.Conv2d(1, hidden, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(hidden, hidden, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    self.memory_position = nn.Parameter(
                        torch.randn(1, 16, hidden) * 0.02
                    )
                    self.decoder = DETRDecoderBlock(config)
                    self.object_queries = nn.Parameter(
                        torch.randn(config.queries, hidden) * 0.02
                    )
                    self.class_head = nn.Linear(hidden, config.num_classes + 1)
                    self.box_head = nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, 4),
                    )

                def encode(self, inputs):
                    features = self.backbone(inputs)
                    memory = features.flatten(2).transpose(1, 2)
                    return memory + self.memory_position[:, : memory.shape[1]]

                def forward(self, inputs):
                    memory = self.encode(inputs)
                    queries = self.object_queries[None].expand(len(inputs), -1, -1)
                    decoded, _ = self.decoder(queries, memory)
                    return self.class_head(decoded), self.box_head(decoded).sigmoid()

            model = DETRMini(config).to(DEVICE)
            class_logits, predicted_boxes = model(images[:2])
            memory = model.encode(images[:2])
            queries = model.object_queries[None].expand(2, -1, -1)
            _, cross_weights = model.decoder(queries, memory)
            assert class_logits.shape == (2, 3, 5)
            assert predicted_boxes.shape == (2, 3, 4)
            assert cross_weights.shape == (2, 4, 3, 16)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: set loss와 GT-independent objectness-selected 평가를 구현하세요.
            def detr_set_loss(prediction, labels, target_boxes, config):
                raise NotImplementedError

            def train_detr_step(model, optimizer, inputs, labels, boxes, config):
                raise NotImplementedError

            def evaluate_detr(model, inputs, labels, boxes):
                raise NotImplementedError
            """,
            """
            def detr_set_loss(prediction, labels, target_boxes, config):
                class_logits, predicted_boxes = prediction
                matched = match_single_target(
                    class_logits,
                    predicted_boxes,
                    labels,
                    target_boxes,
                )
                no_object = config.num_classes
                class_targets = torch.full(
                    class_logits.shape[:2],
                    no_object,
                    dtype=torch.long,
                    device=labels.device,
                )
                batch_indices = torch.arange(len(labels), device=labels.device)
                class_targets[batch_indices, matched] = labels
                weights = class_logits.new_ones(config.num_classes + 1)
                weights[no_object] = config.no_object_weight
                class_loss = F.cross_entropy(
                    class_logits.flatten(0, 1),
                    class_targets.flatten(),
                    weight=weights,
                )
                selected_boxes = predicted_boxes[batch_indices, matched]
                box_loss = F.l1_loss(selected_boxes, target_boxes)
                return class_loss + config.box_weight * box_loss

            def train_detr_step(model, optimizer, inputs, labels, boxes, config):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = detr_set_loss(model(inputs), labels, boxes, config)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_detr(model, inputs, labels, boxes):
                model.eval()
                with torch.no_grad():
                    class_logits, predicted_boxes = model(inputs)
                    probabilities = class_logits.softmax(dim=-1)
                    no_object_probability = probabilities[..., -1]
                    selected = (1.0 - no_object_probability).argmax(dim=1)
                    batch = torch.arange(len(labels), device=labels.device)
                    selected_logits = class_logits[batch, selected, :-1]
                    selected_boxes = predicted_boxes[batch, selected]
                    predicted_labels = selected_logits.argmax(dim=-1)
                    class_accuracy = (predicted_labels == labels).float().mean()
                    box_l1 = F.l1_loss(selected_boxes, boxes)
                    mean_objectness = (
                        1.0 - no_object_probability[batch, selected]
                    ).mean()
                return {
                    "objectness_selected_class_accuracy": float(class_accuracy),
                    "objectness_selected_box_l1": float(box_l1),
                    "selected_query_mean_objectness": float(mean_objectness),
                }

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                loss = train_detr_step(
                    model,
                    optimizer,
                    x_train,
                    y_train,
                    boxes[train_idx],
                    config,
                )
                losses.append(loss)
            metrics = evaluate_detr(
                model,
                images[test_idx],
                labels[test_idx],
                boxes[test_idx],
            )
            assert min(losses[3:]) < losses[0]
            accuracy = metrics["objectness_selected_class_accuracy"]
            assert 0.0 <= accuracy <= 1.0
            assert metrics["objectness_selected_box_l1"] >= 0.0
            plt.figure(figsize=(5, 3))
            plt.plot(losses)
            plt.title(f"DETR objectness-selected class accuracy={accuracy:.1%}")
            plt.tight_layout()
            plt.show()
            print(metrics)
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


SPECS = (LENET5, ALEXNET, VGG, INCEPTION, UNET, RESNET, FASTER_RCNN, YOLO, VIT, DETR)
assert len(SPECS) == 10
assert tuple(spec.number for spec in SPECS) == tuple(range(10))
assert len({spec.slug for spec in SPECS}) == 10
