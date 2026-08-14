"""Paper specifications for self-supervised, vision-Transformer, diffusion, and CLIP labs."""

from __future__ import annotations

from .common import PaperSpec, code, markdown, shared_code

COMMON_SETUP = shared_code(
    """
    import math
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from llm_engineering_lab.acceleration import get_accelerator

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    ACCELERATOR = get_accelerator()
    DEVICE = ACCELERATOR.device
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(0)
    print(ACCELERATOR.summary())
    print("synthetic-data preprocessing stays on CPU; neural training uses DEVICE")
    """,
    "setup",
)


SIMCLR = PaperSpec(
    number=16,
    slug="simclr_contrastive_learning",
    short_title="SimCLR — 대조학습",
    paper_title="A Simple Framework for Contrastive Learning of Visual Representations",
    authors="Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton",
    year=2020,
    primary_url="https://arxiv.org/abs/2002.05709",
    venue="ICML",
    difficulty="중급",
    expected_minutes=75,
    prerequisites="cosine similarity, cross entropy, MLP, 데이터 증강",
    reproduction_goal=(
        "그림 2의 `x → 두 증강 view → encoder f → projection g → NT-Xent` 흐름을 작은 "
        "8×8 합성 패턴에 구현합니다. 양성 view의 cosine 유사도가 음성 쌍보다 커지는지, "
        "projection 이전 표현 `h`를 고정한 linear probe가 클래스를 분리하는지 확인합니다."
    ),
    original_scale=(
        "ImageNet, ResNet-50, 큰 batch와 긴 학습은 생략하고 8×8 합성 영상·작은 MLP·짧은 "
        "full-batch 학습을 사용합니다. 따라서 ImageNet 정확도와 표 1~7의 수치를 재현하지 않습니다."
    ),
    mappings=(
        (
            "§2.1 및 그림 2",
            "두 개의 확률적 view와 f/g 경로",
            "두 view shape와 독립 잡음",
        ),
        ("§2.1 식 (1)", "cosine similarity", "정규화한 표현의 내적"),
        (
            "§2.1 식 (2), Algorithm 1",
            "2N 표본의 NT-Xent",
            "대각선 제외·양성 index·유한 gradient",
        ),
        (
            "§2.2 및 그림 2",
            "projection 이전 h의 linear evaluation",
            "고정 encoder의 probe 정확도",
        ),
        ("§3, 그림 4~5", "증강과 temperature 미니 실험", "양성/음성 유사도 차이"),
    ),
    cells=(
        COMMON_SETUP,
        markdown(
            """
            ## 1) 한 샘플에서 두 개의 view 만들기

            SimCLR의 label은 사람이 붙이지 않습니다. 같은 원본에서 독립적으로 뽑은 두 view가
            양성 쌍이고, batch의 나머지가 음성입니다. 여기서는 crop·color distortion 대신
            grayscale 합성 데이터에 맞춘 밝기 jitter와 Gaussian noise를 씁니다.

            <details><summary>힌트</summary>
            view마다 별도 random draw를 사용하고, 밝기는 sample별 `[N,1,1,1]`로 broadcast하세요.
            </details>
            """
        ),
        code(
            """
            # TODO 16-1: 독립적인 두 증강 view를 반환하세요.
            def make_views(x: Tensor, noise: float = 0.08) -> tuple[Tensor, Tensor]:
                raise NotImplementedError("TODO 16-1")

            prototypes = torch.zeros(4, 1, 8, 8)
            prototypes[0, :, :, 1:3] = 1
            prototypes[1, :, :, 5:7] = 1
            prototypes[2, :, 1:3, :] = 1
            prototypes[3, :, 5:7, :] = 1
            labels = torch.arange(4).repeat_interleave(24)
            images = (prototypes[labels] + 0.04 * torch.randn(96, 1, 8, 8)).clamp(0, 1)
            view_a, view_b = make_views(images)
            """,
            """
            def make_views(x: Tensor, noise: float = 0.08) -> tuple[Tensor, Tensor]:
                def augment() -> Tensor:
                    brightness = 0.8 + 0.4 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
                    return (x * brightness + noise * torch.randn_like(x)).clamp(0, 1)

                return augment(), augment()

            prototypes = torch.zeros(4, 1, 8, 8)
            prototypes[0, :, :, 1:3] = 1
            prototypes[1, :, :, 5:7] = 1
            prototypes[2, :, 1:3, :] = 1
            prototypes[3, :, 5:7, :] = 1
            labels = torch.arange(4).repeat_interleave(24)
            images = (prototypes[labels] + 0.04 * torch.randn(96, 1, 8, 8)).clamp(0, 1)
            view_a, view_b = make_views(images)
            assert view_a.shape == view_b.shape == images.shape
            assert not torch.equal(view_a, view_b)
            print("두 view 평균 절대 차이:", float((view_a - view_b).abs().mean()))
            """,
            "task",
        ),
        markdown(
            """
            ## 2) 식 (2): NT-Xent

            `z=[z_i, z_j]`는 크기 `2N`입니다. 자기 자신은 denominator에서 제외하고, 앞쪽 view의
            양성 index는 `i+N`, 뒤쪽은 `i-N`입니다. temperature `τ`는 logit의 선명도를 조절합니다.
            """
        ),
        code(
            """
            # TODO 16-2: 식 (1)의 cosine similarity와 식 (2)의 NT-Xent를 벡터화하세요.
            def nt_xent(z1: Tensor, z2: Tensor, temperature: float = 0.2) -> Tensor:
                raise NotImplementedError("TODO 16-2")
            """,
            """
            def nt_xent(z1: Tensor, z2: Tensor, temperature: float = 0.2) -> Tensor:
                if z1.shape != z2.shape:
                    raise ValueError("the two views must have the same shape")
                n = z1.size(0)
                z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
                logits = z @ z.T / temperature
                logits.fill_diagonal_(float("-inf"))
                targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
                return F.cross_entropy(logits, targets)

            toy_z = torch.randn(6, 8, requires_grad=True)
            toy_loss = nt_xent(toy_z, toy_z + 0.01 * torch.randn_like(toy_z))
            toy_loss.backward()
            assert toy_loss.ndim == 0 and torch.isfinite(toy_loss)
            assert toy_z.grad is not None and torch.isfinite(toy_z.grad).all()
            print("toy NT-Xent:", round(float(toy_loss), 4))
            """,
            "task",
        ),
        code(
            """
            # TODO 16-3: encoder f와 비선형 projection head g를 별도 class로 구현하세요.
            class TinyEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError("TODO 16-3")

                def forward(self, x: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 16-3")

            class ProjectionHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError("TODO 16-3")

                def forward(self, hidden: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 16-3")

            class TinySimCLR(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError("TODO 16-3")

                def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                    raise NotImplementedError("TODO 16-3")
            """,
            """
            class TinyEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_projection = nn.Linear(64, 32)
                    self.output_projection = nn.Linear(32, 16)

                def forward(self, x: Tensor) -> Tensor:
                    flattened = x.flatten(start_dim=1)
                    hidden = F.relu(self.input_projection(flattened))
                    return self.output_projection(hidden)

            class ProjectionHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.hidden_projection = nn.Linear(16, 16)
                    self.output_projection = nn.Linear(16, 8)

                def forward(self, hidden: Tensor) -> Tensor:
                    projected = F.relu(self.hidden_projection(hidden))
                    return self.output_projection(projected)

            class TinySimCLR(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = TinyEncoder()
                    self.projector = ProjectionHead()

                def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                    h = self.encoder(x)
                    return h, self.projector(h)

            model = TinySimCLR()
            h, z = model(images[:5])
            assert h.shape == (5, 16) and z.shape == (5, 8)
            """,
            "task",
        ),
        code(
            """
            # TODO 16-4: 한 번의 contrastive update와 반복 학습을 구현하세요.
            model = ACCELERATOR.move(TinySimCLR())
            images_device, labels_device = ACCELERATOR.move(images, labels)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            losses = []

            def contrastive_update(model, batch, optimizer, temperature=0.2):
                raise NotImplementedError("TODO 16-4")

            raise NotImplementedError("TODO 16-4")
            """,
            """
            model = ACCELERATOR.move(TinySimCLR())
            images_device, labels_device = ACCELERATOR.move(images, labels)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            losses = []

            def contrastive_update(model, batch, optimizer, temperature=0.2):
                model.train()
                first_view, second_view = make_views(batch)
                _, first_projection = model(first_view)
                _, second_projection = model(second_view)
                loss = nt_xent(
                    first_projection,
                    second_projection,
                    temperature=temperature,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                return float(loss.detach().cpu())

            for _ in range(100):
                losses.append(
                    contrastive_update(
                        model,
                        images_device,
                        optimizer,
                    )
                )

            with torch.no_grad():
                a, b = make_views(images_device)
                ha, za = model(a)
                hb, zb = model(b)
                positive = float(F.cosine_similarity(za, zb).mean().cpu())
                negative = float(F.cosine_similarity(za, zb.roll(1, 0)).mean().cpu())
            assert np.mean(losses[-10:]) < np.mean(losses[:10])
            assert positive > negative
            print(
                f"loss {losses[0]:.3f} → {losses[-1]:.3f}, "
                f"positive={positive:.3f}, negative={negative:.3f}"
            )
            plt.figure(figsize=(5, 2.5))
            plt.plot(losses)
            plt.title("SimCLR mini training")
            plt.xlabel("step")
            plt.ylabel("NT-Xent")
            plt.grid(alpha=0.2)
            plt.show()
            """,
            "task",
        ),
        code(
            """
            # TODO 16-5: encoder를 고정하고 h 위에 linear probe만 학습하세요.
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)
            model.zero_grad(set_to_none=True)
            raise NotImplementedError("TODO 16-5")
            """,
            """
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                frozen_h = model.encoder(images_device)
            probe = ACCELERATOR.move(nn.Linear(frozen_h.size(1), 4))
            probe_opt = torch.optim.Adam(probe.parameters(), lr=0.05)
            for _ in range(80):
                probe_loss = F.cross_entropy(probe(frozen_h), labels_device)
                probe_opt.zero_grad()
                probe_loss.backward()
                probe_opt.step()
            accuracy = (probe(frozen_h).argmax(1) == labels_device).float().mean().item()
            assert all(parameter.grad is None for parameter in model.encoder.parameters())
            assert accuracy > 0.90
            print(f"frozen-h linear probe accuracy={accuracy:.1%}")
            """,
            "task",
        ),
    ),
)


VIT = PaperSpec(
    number=17,
    slug="vision_transformer",
    short_title="ViT — 이미지를 patch token으로",
    paper_title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    authors="Alexey Dosovitskiy et al.",
    year=2021,
    primary_url="https://arxiv.org/abs/2010.11929",
    venue="ICLR",
    difficulty="중급",
    expected_minutes=75,
    prerequisites="convolution shape, self-attention, class token, positional embedding",
    reproduction_goal=(
        "식 (1)의 patchify·linear projection·class token·position embedding과 식 (2)~(4)의 "
        "pre-LN Transformer encoder를 8×8 영상에 구현합니다. 위치 정보가 없으면 patch 순열을 "
        "구별하지 못하는 성질과 작은 막대 분류 학습을 확인합니다."
    ),
    original_scale=(
        "JFT-300M/ImageNet 사전학습과 16×16 patch의 ViT-Base/Large/Huge는 생략합니다. 8×8 "
        "합성 영상, 2×2 patch, encoder 1층을 사용하므로 표 1~6의 전이 성능은 검증하지 않습니다."
    ),
    mappings=(
        (
            "§3.1 식 (1), 그림 1",
            "patchify·E projection·[class]·E_pos",
            "[B,N+1,D] shape",
        ),
        ("§3.1 식 (2)", "pre-LN multi-head attention residual", "token shape 보존"),
        ("§3.1 식 (3)", "pre-LN MLP residual", "유한 gradient"),
        ("§3.1 식 (4)", "class token 분류 head", "두 막대 class 정확도"),
        (
            "§3.1 Inductive bias",
            "position embedding 제거 ablation",
            "patch 순열 불변성",
        ),
    ),
    cells=(
        COMMON_SETUP,
        markdown(
            """
            ## 1) 식 (1)의 patch sequence

            8×8 이미지를 겹치지 않는 2×2 patch로 나누면 `N=(8/2)^2=16`, patch vector 차원은
            `P²C=4`입니다. `patchify`와 `unpatchify`가 정확한 역함수인지 먼저 확인합니다.
            """
        ),
        code(
            """
            # TODO 17-1: [B,C,H,W] ↔ [B,N,P²C] 변환을 구현하세요.
            def patchify(x: Tensor, patch_size: int) -> Tensor:
                raise NotImplementedError("TODO 17-1")

            def unpatchify(tokens: Tensor, patch_size: int, height: int, width: int) -> Tensor:
                raise NotImplementedError("TODO 17-1")
            """,
            """
            def patchify(x: Tensor, patch_size: int) -> Tensor:
                b, c, h, w = x.shape
                if h % patch_size or w % patch_size:
                    raise ValueError("height and width must be divisible by patch_size")
                x = x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                return x.permute(0, 2, 4, 3, 5, 1).reshape(b, -1, patch_size * patch_size * c)

            def unpatchify(tokens: Tensor, patch_size: int, height: int, width: int) -> Tensor:
                b, _, flat = tokens.shape
                c = flat // (patch_size * patch_size)
                x = tokens.reshape(
                    b,
                    height // patch_size,
                    width // patch_size,
                    patch_size,
                    patch_size,
                    c,
                )
                return x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, height, width)

            checker = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
            patches = patchify(checker, 2)
            assert patches.shape == (2, 16, 4)
            assert torch.equal(unpatchify(patches, 2, 8, 8), checker)
            print("patch tokens:", tuple(patches.shape))
            """,
            "task",
        ),
        markdown(
            """
            ## 2) 식 (1)~(4)의 TinyViT

            원 논문은 각 encoder block을 `x' = MSA(LN(x)) + x`,
            `x = MLP(LN(x')) + x'`로 씁니다. 여기서는 이 pre-LN 순서를 고수준
            `TransformerEncoder` 호출에 숨기지 않습니다. `ViTSelfAttention`은 Q/K/V와 head
            결합을, `ViTMLP`는 GELU를 포함한 식 (3)을, `ViTEncoderBlock`은 두 residual
            경로를 각각 드러냅니다.
            """
        ),
        code(
            """
            # TODO 17-2: MSA, MLP, pre-LN residual block을 직접 구현하세요.
            class ViTSelfAttention(nn.Module):
                def __init__(self, dim: int, heads: int):
                    super().__init__()
                    raise NotImplementedError("TODO 17-2")

                def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
                    raise NotImplementedError("TODO 17-2")

            class ViTMLP(nn.Module):
                def __init__(self, dim: int, hidden_dim: int):
                    super().__init__()
                    raise NotImplementedError("TODO 17-2")

                def forward(self, hidden: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 17-2")

            class ViTEncoderBlock(nn.Module):
                def __init__(self, dim: int, heads: int, mlp_dim: int):
                    super().__init__()
                    raise NotImplementedError("TODO 17-2")

                def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
                    raise NotImplementedError("TODO 17-2")

            # TODO 17-3: patch projection, class/position token, encoder를 조립하세요.
            class TinyViT(nn.Module):
                def __init__(self, image_size=8, patch_size=2, dim=24, heads=3, classes=2):
                    super().__init__()
                    raise NotImplementedError("TODO 17-3")

                def embed_tokens(self, x: Tensor, use_position: bool = True) -> Tensor:
                    raise NotImplementedError("TODO 17-3")

                def encode_tokens(self, tokens: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 17-3")

                def forward(self, x: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 17-3")
            """,
            """
            class ViTSelfAttention(nn.Module):
                def __init__(self, dim: int, heads: int):
                    super().__init__()
                    if dim % heads:
                        raise ValueError("dim must be divisible by heads")
                    self.heads = heads
                    self.head_dim = dim // heads
                    self.query_key_value = nn.Linear(dim, 3 * dim, bias=False)
                    self.output = nn.Linear(dim, dim, bias=False)

                def _split_heads(self, tensor: Tensor) -> Tensor:
                    batch, tokens, width = tensor.shape
                    split = tensor.view(
                        batch,
                        tokens,
                        self.heads,
                        width // self.heads,
                    )
                    return split.transpose(1, 2)

                def _merge_heads(self, tensor: Tensor) -> Tensor:
                    batch, _, tokens, _ = tensor.shape
                    merged = tensor.transpose(1, 2).contiguous()
                    return merged.view(
                        batch,
                        tokens,
                        self.heads * self.head_dim,
                    )

                def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
                    query, key, value = self.query_key_value(hidden).chunk(3, dim=-1)
                    query = self._split_heads(query)
                    key = self._split_heads(key)
                    value = self._split_heads(value)
                    scores = query @ key.transpose(-2, -1)
                    scores = scores / math.sqrt(self.head_dim)
                    weights = torch.softmax(scores, dim=-1)
                    context = self._merge_heads(weights @ value)
                    return self.output(context), weights

            class ViTMLP(nn.Module):
                def __init__(self, dim: int, hidden_dim: int):
                    super().__init__()
                    self.input = nn.Linear(dim, hidden_dim)
                    self.output = nn.Linear(hidden_dim, dim)

                def forward(self, hidden: Tensor) -> Tensor:
                    return self.output(F.gelu(self.input(hidden)))

            class ViTEncoderBlock(nn.Module):
                def __init__(self, dim: int, heads: int, mlp_dim: int):
                    super().__init__()
                    self.attention_norm = nn.LayerNorm(dim)
                    self.attention = ViTSelfAttention(dim, heads)
                    self.mlp_norm = nn.LayerNorm(dim)
                    self.mlp = ViTMLP(dim, mlp_dim)

                def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
                    attended, weights = self.attention(
                        self.attention_norm(hidden)
                    )
                    hidden = hidden + attended
                    hidden = hidden + self.mlp(self.mlp_norm(hidden))
                    return hidden, weights

            class TinyViT(nn.Module):
                def __init__(self, image_size=8, patch_size=2, dim=24, heads=3, classes=2):
                    super().__init__()
                    self.patch_size = patch_size
                    count = (image_size // patch_size) ** 2
                    self.projection = nn.Linear(patch_size * patch_size, dim)
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
                    self.position = nn.Parameter(torch.randn(1, count + 1, dim) * 0.02)
                    self.encoder = ViTEncoderBlock(dim, heads, mlp_dim=2 * dim)
                    self.norm = nn.LayerNorm(dim)
                    self.head = nn.Linear(dim, classes)

                def embed_tokens(self, x: Tensor, use_position: bool = True) -> Tensor:
                    patch_tokens = self.projection(patchify(x, self.patch_size))
                    cls = self.cls_token.expand(x.size(0), -1, -1)
                    tokens = torch.cat([cls, patch_tokens], dim=1)
                    return tokens + self.position if use_position else tokens

                def encode_tokens(self, tokens: Tensor) -> Tensor:
                    encoded, _ = self.encoder(tokens)
                    return self.norm(encoded[:, 0])

                def forward(self, x: Tensor) -> Tensor:
                    return self.head(self.encode_tokens(self.embed_tokens(x)))

            vit = TinyViT()
            dummy = torch.randn(3, 1, 8, 8)
            assert vit.embed_tokens(dummy).shape == (3, 17, 24)
            assert vit(dummy).shape == (3, 2)
            embedded = vit.embed_tokens(dummy)
            encoded, attention_weights = vit.encoder(embedded)
            assert encoded.shape == embedded.shape
            assert attention_weights.shape == (3, 3, 17, 17)
            assert torch.allclose(
                attention_weights.sum(dim=-1),
                attention_weights.new_ones(3, 3, 17),
                atol=1e-6,
            )
            """,
            "task",
        ),
        code(
            """
            # TODO 17-4: patch 순서를 섞어 position 유무의 class representation을 비교하세요.
            tokens_no_pos = vit.embed_tokens(dummy, use_position=False)
            permutation = torch.randperm(16)
            raise NotImplementedError("TODO 17-3")
            """,
            """
            tokens_no_pos = vit.embed_tokens(dummy, use_position=False)
            permutation = torch.randperm(16)
            permuted_no_pos = torch.cat(
                [tokens_no_pos[:, :1], tokens_no_pos[:, 1:][:, permutation]],
                dim=1,
            )
            no_pos_a = vit.encode_tokens(tokens_no_pos)
            no_pos_b = vit.encode_tokens(permuted_no_pos)

            tokens_pos = tokens_no_pos + vit.position
            permuted_content_same_positions = torch.cat(
                [tokens_no_pos[:, :1], tokens_no_pos[:, 1:][:, permutation]],
                dim=1,
            )
            permuted_content_same_positions = (
                permuted_content_same_positions + vit.position
            )
            pos_a = vit.encode_tokens(tokens_pos)
            pos_b = vit.encode_tokens(permuted_content_same_positions)
            assert torch.allclose(no_pos_a, no_pos_b, atol=1e-5)
            assert not torch.allclose(pos_a, pos_b, atol=1e-5)
            print("no-pos permutation delta:", float((no_pos_a - no_pos_b).abs().max()))
            print("with-pos permutation delta:", float((pos_a - pos_b).abs().max()))
            """,
            "task",
        ),
        code(
            """
            # TODO 17-5: 세로/가로 막대 합성 데이터를 만들고 class token head를 학습하세요.
            raise NotImplementedError("TODO 17-4")
            """,
            """
            generator = torch.Generator().manual_seed(17)
            train_x = 0.08 * torch.randn(160, 1, 8, 8, generator=generator)
            train_y = torch.arange(160) % 2
            for index, label in enumerate(train_y.tolist()):
                if label == 0:
                    train_x[index, :, :, 3:5] += 1.0
                else:
                    train_x[index, :, 3:5, :] += 1.0
            train_x = train_x.clamp(0, 1)

            vit = ACCELERATOR.move(TinyViT())
            train_x_device, train_y_device = ACCELERATOR.move(train_x, train_y)
            opt = torch.optim.Adam(vit.parameters(), lr=0.01)
            history = []
            for _ in range(70):
                loss = F.cross_entropy(vit(train_x_device), train_y_device)
                opt.zero_grad()
                loss.backward()
                opt.step()
                history.append(float(loss.detach().cpu()))
            accuracy = (vit(train_x_device).argmax(1) == train_y_device).float().mean().item()
            assert history[-1] < history[0] and accuracy > 0.95
            print(f"training accuracy={accuracy:.1%}, loss={history[-1]:.4f}")
            plt.figure(figsize=(5, 2.5))
            plt.plot(history)
            plt.title("TinyViT mini training")
            plt.xlabel("step")
            plt.ylabel("cross entropy")
            plt.grid(alpha=0.2)
            plt.show()
            """,
            "task",
        ),
    ),
)


DDPM = PaperSpec(
    number=18,
    slug="ddpm_diffusion",
    short_title="DDPM — 확산과 역확산",
    paper_title="Denoising Diffusion Probabilistic Models",
    authors="Jonathan Ho, Ajay Jain, Pieter Abbeel",
    year=2020,
    primary_url="https://arxiv.org/abs/2006.11239",
    venue="NeurIPS",
    difficulty="중상급",
    expected_minutes=90,
    prerequisites="정규분포, 재매개화, MSE, 조건부 생성",
    reproduction_goal=(
        "식 (4)의 임의 시점 forward noising, 식 (11)의 noise parameterization, 식 (14)와 "
        "Algorithm 1의 noise-prediction 학습, Algorithm 2의 reverse sampling을 2차원 두 군집에 "
        "구현하고 시간에 따른 분포 변화를 그립니다."
    ),
    original_scale=(
        "CIFAR-10/LSUN 영상, U-Net, 1,000 diffusion step과 FID/Inception Score는 생략하고, "
        "2차원 데이터·작은 MLP·30 step을 씁니다. 그림 품질이나 논문의 benchmark 수치는 재현 대상이 아닙니다."
    ),
    mappings=(
        (
            "§2 식 (2), §3.2",
            "linear β schedule과 forward Markov 과정",
            "alpha 및 누적 alpha_bar",
        ),
        ("§2 식 (4)", "닫힌형식 q(x_t|x_0)", "t=0 경계·표본 shape"),
        ("§3.2 식 (11)", "ε_θ로 reverse mean 계산", "한 step 역확산"),
        ("§3.4 식 (14), Algorithm 1", "무작위 t noise MSE", "초기/후반 loss"),
        ("Algorithm 2", "T−1부터 0까지 sampling", "trajectory 산점도"),
    ),
    cells=(
        COMMON_SETUP,
        markdown(
            """
            ## 1) 식 (4): 한 번에 원하는 시점으로

            `α_t=1-β_t`, `ᾱ_t=∏_{s≤t} α_s`이면 반복해서 잡음을 넣지 않고도
            `x_t = sqrt(ᾱ_t)x_0 + sqrt(1-ᾱ_t)ε`로 직접 표본화할 수 있습니다.
            """
        ),
        code(
            """
            # TODO 18-1: schedule과 식 (4)의 q_sample을 구현하세요.
            T = 30
            betas = ...
            alphas = ...
            alpha_bar = ...

            def extract(values: Tensor, t: Tensor, x: Tensor) -> Tensor:
                raise NotImplementedError("TODO 18-1")

            def q_sample(
                x0: Tensor,
                t: Tensor,
                noise: Tensor | None = None,
            ) -> tuple[Tensor, Tensor]:
                raise NotImplementedError("TODO 18-1")
            """,
            """
            T = 30
            betas = torch.linspace(1e-4, 0.08, T)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)

            def extract(values: Tensor, t: Tensor, x: Tensor) -> Tensor:
                return values.gather(0, t).reshape(t.size(0), *([1] * (x.ndim - 1)))

            def q_sample(
                x0: Tensor,
                t: Tensor,
                noise: Tensor | None = None,
            ) -> tuple[Tensor, Tensor]:
                noise = torch.randn_like(x0) if noise is None else noise
                abar = extract(alpha_bar, t, x0)
                return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise, noise

            boundary_x = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
            fixed_noise = torch.zeros_like(boundary_x)
            noisy, _ = q_sample(boundary_x, torch.tensor([0, T - 1]), fixed_noise)
            assert noisy.shape == boundary_x.shape and torch.isfinite(noisy).all()
            assert 0 < alpha_bar[-1] < alpha_bar[0] < 1
            print("alpha_bar first/last:", float(alpha_bar[0]), float(alpha_bar[-1]))
            """,
            "task",
        ),
        shared_code(
            """
            # 준비: 외부 다운로드 없는 2차원 두 군집
            g = torch.Generator().manual_seed(18)
            half = 256
            data = torch.cat([
                torch.randn(half, 2, generator=g) * torch.tensor([0.35, 0.18])
                + torch.tensor([-1.5, 0.0]),
                torch.randn(half, 2, generator=g) * torch.tensor([0.35, 0.18])
                + torch.tensor([1.5, 0.0]),
            ])
            times = torch.tensor([0, T // 3, 2 * T // 3, T - 1])
            fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
            for axis, time in zip(axes, times):
                xt, _ = q_sample(data, torch.full((len(data),), int(time)))
                axis.scatter(xt[:, 0], xt[:, 1], s=4, alpha=0.35)
                axis.set_title(f"t={int(time)}")
                axis.set_xlim(-4, 4)
                axis.set_ylim(-3, 3)
            plt.tight_layout()
            plt.show()
            """,
            "experiment",
        ),
        code(
            """
            # TODO 18-2: timestep embedding과 ε_θ(x_t,t) MLP를 구현하세요.
            def time_embedding(t: Tensor, dim: int = 16) -> Tensor:
                raise NotImplementedError("TODO 18-2")

            class NoisePredictor(nn.Module):
                def __init__(self):
                    super().__init__()
                    raise NotImplementedError("TODO 18-2")

                def forward(self, x: Tensor, t: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 18-2")
            """,
            """
            def time_embedding(t: Tensor, dim: int = 16) -> Tensor:
                half_dim = dim // 2
                frequencies = torch.exp(
                    -math.log(10_000)
                    * torch.arange(half_dim, device=t.device)
                    / max(half_dim - 1, 1)
                )
                angles = t.float()[:, None] * frequencies[None]
                return torch.cat([angles.sin(), angles.cos()], dim=1)

            class NoisePredictor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(18, 64),
                        nn.SiLU(),
                        nn.Linear(64, 64),
                        nn.SiLU(),
                        nn.Linear(64, 2),
                    )

                def forward(self, x: Tensor, t: Tensor) -> Tensor:
                    return self.net(torch.cat([x, time_embedding(t)], dim=1))

            predictor = NoisePredictor()
            assert predictor(data[:7], torch.arange(7) % T).shape == (7, 2)
            """,
            "task",
        ),
        code(
            """
            # TODO 18-3: Algorithm 1 / 식 (14)의 random-t noise MSE로 학습하세요.
            predictor = ACCELERATOR.move(NoisePredictor())
            data, betas, alphas, alpha_bar = ACCELERATOR.move(data, betas, alphas, alpha_bar)
            optimizer = torch.optim.Adam(predictor.parameters(), lr=2e-3)
            losses = []
            raise NotImplementedError("TODO 18-3")
            """,
            """
            predictor = ACCELERATOR.move(NoisePredictor())
            data, betas, alphas, alpha_bar = ACCELERATOR.move(data, betas, alphas, alpha_bar)
            optimizer = torch.optim.Adam(predictor.parameters(), lr=2e-3)
            losses = []
            for step in range(260):
                index = torch.randint(len(data), (128,), device=DEVICE)
                x0 = data[index]
                t = torch.randint(T, (len(index),), device=DEVICE)
                xt, epsilon = q_sample(x0, t)
                loss = F.mse_loss(predictor(xt, t), epsilon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            assert np.mean(losses[-30:]) < np.mean(losses[:30])
            print(f"noise MSE {np.mean(losses[:20]):.3f} → {np.mean(losses[-20:]):.3f}")
            """,
            "task",
        ),
        code(
            """
            # TODO 18-4: 식 (11)과 Algorithm 2의 reverse step/sampling을 구현하세요.
            @torch.no_grad()
            def reverse_step(model: nn.Module, x: Tensor, step: int) -> Tensor:
                raise NotImplementedError("TODO 18-4")

            @torch.no_grad()
            def sample(model: nn.Module, count: int = 512) -> tuple[Tensor, list[Tensor]]:
                raise NotImplementedError("TODO 18-4")
            """,
            """
            @torch.no_grad()
            def reverse_step(model: nn.Module, x: Tensor, step: int) -> Tensor:
                t = torch.full((x.size(0),), step, dtype=torch.long, device=x.device)
                predicted_noise = model(x, t)
                correction = (
                    betas[step]
                    / torch.sqrt(1 - alpha_bar[step])
                    * predicted_noise
                )
                mean = (x - correction) / torch.sqrt(alphas[step])
                if step == 0:
                    return mean
                return mean + betas[step].sqrt() * torch.randn_like(x)

            @torch.no_grad()
            def sample(model: nn.Module, count: int = 512) -> tuple[Tensor, list[Tensor]]:
                model_device = next(model.parameters()).device
                x = torch.randn(count, 2, device=model_device)
                trajectory = [x.detach().cpu().clone()]
                for step in reversed(range(T)):
                    x = reverse_step(model, x, step)
                    if step in {20, 10, 0}:
                        trajectory.append(x.detach().cpu().clone())
                return x, trajectory

            generated, trajectory = sample(predictor)
            assert generated.shape == data.shape and torch.isfinite(generated).all()
            fig, axes = plt.subplots(1, len(trajectory), figsize=(10, 2.4))
            for axis, points, title in zip(axes, trajectory, ["start", "t=20", "t=10", "t=0"]):
                axis.scatter(points[:, 0], points[:, 1], s=4, alpha=0.35)
                axis.set_title(title)
                axis.set_xlim(-4, 4)
                axis.set_ylim(-3, 3)
            plt.tight_layout()
            plt.show()
            print(
                "generated mean:", generated.mean(0).detach().cpu().tolist(),
                "std:", generated.std(0).detach().cpu().tolist(),
            )
            """,
            "task",
        ),
    ),
)


CLIP = PaperSpec(
    number=19,
    slug="clip_multimodal_contrastive",
    short_title="CLIP — 이미지·텍스트 공동공간",
    paper_title="Learning Transferable Visual Models From Natural Language Supervision",
    authors="Alec Radford et al.",
    year=2021,
    primary_url="https://arxiv.org/abs/2103.00020",
    venue="ICML",
    difficulty="중상급",
    expected_minutes=80,
    prerequisites="대조학습, cosine similarity, dual encoder, zero-shot 분류",
    reproduction_goal=(
        "그림 1과 §2.3의 image/text dual encoder, L2 정규화, 학습 가능한 temperature, 대칭 "
        "cross entropy를 합성 paired feature에 구현합니다. image→text retrieval과 class-name "
        "prompt를 이용한 zero-shot 분류 경로를 확인합니다."
    ),
    original_scale=(
        "4억 개 웹 image-text pair, ResNet/ViT와 Transformer text encoder, prompt ensemble 및 "
        "30개 이상 benchmark는 생략합니다. 연속 벡터로 만든 작은 paired dataset이므로 실제 자연어·영상 "
        "일반화와 표 3~12의 수치를 재현하지 않습니다."
    ),
    mappings=(
        (
            "그림 1 (1)~(2)",
            "image/text encoder와 normalized embedding",
            "[N,D] 단위벡터",
        ),
        (
            "그림 1 (3), §2.3",
            "pairwise cosine logits와 temperature",
            "[N,N] similarity",
        ),
        (
            "그림 1 pseudocode",
            "image/text 양방향 symmetric loss",
            "두 cross entropy 평균",
        ),
        ("§3.1", "image↔text retrieval", "Recall@1"),
        (
            "§2.1 및 §3.1",
            "class-name text prototype zero-shot 분류",
            "held-out accuracy",
        ),
    ),
    cells=(
        COMMON_SETUP,
        shared_code(
            """
            # 외부 이미지/텍스트 대신 같은 latent 의미에서 서로 다른 관측 feature를 만듭니다.
            generator = torch.Generator().manual_seed(19)
            class_centers = torch.tensor([
                [2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0],
            ])
            pair_labels = torch.arange(4).repeat_interleave(24)
            latent = class_centers[pair_labels] + 0.35 * torch.randn(96, 4, generator=generator)
            image_map = torch.randn(4, 12, generator=generator)
            text_map = torch.randn(4, 10, generator=generator)
            image_features = latent @ image_map + 0.03 * torch.randn(96, 12, generator=generator)
            text_features = latent @ text_map + 0.03 * torch.randn(96, 10, generator=generator)
            print(image_features.shape, text_features.shape)
            """,
            "data",
        ),
        markdown(
            """
            ## 1) 그림 1의 dual encoder와 similarity matrix

            각 encoder의 출력은 L2 정규화합니다. 따라서 matrix product는 cosine similarity이고,
            `exp(t)`를 곱한 `N×N` logit의 대각선이 올바른 image-text pair입니다.
            """
        ),
        code(
            """
            # TODO 19-1: 두 encoder와 학습 가능한 logit scale을 구현하세요.
            class TinyCLIP(nn.Module):
                def __init__(self, embed_dim: int = 8):
                    super().__init__()
                    raise NotImplementedError("TODO 19-1")

                def encode_image(self, x: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 19-1")

                def encode_text(self, x: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 19-1")

                def logits(self, images: Tensor, texts: Tensor) -> Tensor:
                    raise NotImplementedError("TODO 19-1")
            """,
            """
            class TinyCLIP(nn.Module):
                def __init__(self, embed_dim: int = 8):
                    super().__init__()
                    # Linear encoders make the held-out class-center test identifiable in this
                    # synthetic setting; the original paper uses much larger nonlinear towers.
                    self.image_encoder = nn.Linear(12, embed_dim, bias=False)
                    self.text_encoder = nn.Linear(10, embed_dim, bias=False)
                    self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

                def encode_image(self, x: Tensor) -> Tensor:
                    return F.normalize(self.image_encoder(x), dim=-1)

                def encode_text(self, x: Tensor) -> Tensor:
                    return F.normalize(self.text_encoder(x), dim=-1)

                def logits(self, images: Tensor, texts: Tensor) -> Tensor:
                    scale = self.logit_scale.exp().clamp(max=100)
                    return scale * self.encode_image(images) @ self.encode_text(texts).T

            clip_model = TinyCLIP()
            image_z = clip_model.encode_image(image_features[:6])
            text_z = clip_model.encode_text(text_features[:6])
            assert image_z.shape == text_z.shape == (6, 8)
            assert torch.allclose(image_z.norm(dim=1), torch.ones(6), atol=1e-6)
            assert clip_model.logits(image_features[:6], text_features[:6]).shape == (6, 6)
            """,
            "task",
        ),
        code(
            """
            # TODO 19-2: 그림 1 pseudocode처럼 image/text 방향 CE의 평균을 반환하세요.
            def clip_loss(logits: Tensor) -> Tensor:
                raise NotImplementedError("TODO 19-2")
            """,
            """
            def clip_loss(logits: Tensor) -> Tensor:
                targets = torch.arange(logits.size(0), device=logits.device)
                return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2

            identity_logits = torch.eye(5) * 10
            assert clip_loss(identity_logits) < clip_loss(torch.zeros_like(identity_logits))
            print(
                "aligned vs flat loss:",
                float(clip_loss(identity_logits)),
                float(clip_loss(torch.zeros_like(identity_logits))),
            )
            """,
            "task",
        ),
        code(
            """
            # TODO 19-3: paired batch의 대각선을 정답으로 대칭 대조학습하세요.
            clip_model = ACCELERATOR.move(TinyCLIP())
            image_features_device, text_features_device = ACCELERATOR.move(
                image_features,
                text_features,
            )
            optimizer = torch.optim.Adam(clip_model.parameters(), lr=8e-3)
            history = []

            def retrieval_metrics(logits: Tensor) -> dict[str, float]:
                raise NotImplementedError("TODO 19-3")

            raise NotImplementedError("TODO 19-3")
            """,
            """
            clip_model = ACCELERATOR.move(TinyCLIP())
            image_features_device, text_features_device = ACCELERATOR.move(
                image_features,
                text_features,
            )
            optimizer = torch.optim.Adam(clip_model.parameters(), lr=3e-3)
            history = []

            def retrieval_metrics(logits: Tensor) -> dict[str, float]:
                if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
                    raise ValueError("paired retrieval logits must be square")
                targets = torch.arange(logits.shape[0], device=logits.device)
                image_to_text = (logits.argmax(dim=1) == targets).float().mean()
                text_to_image = (logits.argmax(dim=0) == targets).float().mean()
                diagonal = logits.diag().mean()
                off_diagonal_mask = ~torch.eye(
                    len(logits),
                    dtype=torch.bool,
                    device=logits.device,
                )
                off_diagonal = logits[off_diagonal_mask].mean()
                return {
                    "image_to_text_r1": float(image_to_text),
                    "text_to_image_r1": float(text_to_image),
                    "diagonal_logit": float(diagonal),
                    "off_diagonal_logit": float(off_diagonal),
                }

            for _ in range(300):
                logits = clip_model.logits(image_features_device, text_features_device)
                loss = clip_loss(logits)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach().cpu()))
            with torch.no_grad():
                logits = clip_model.logits(image_features_device, text_features_device)
                retrieval = retrieval_metrics(logits)
            recall_at_1 = retrieval["image_to_text_r1"]
            assert history[-1] < history[0]
            assert retrieval["diagonal_logit"] > retrieval["off_diagonal_logit"]
            assert recall_at_1 > 0.55
            print(f"loss {history[0]:.3f} → {history[-1]:.3f}, image→text R@1={recall_at_1:.1%}")
            plt.figure(figsize=(5, 2.5))
            plt.plot(history)
            plt.title("TinyCLIP symmetric contrastive loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.grid(alpha=0.2)
            plt.show()
            """,
            "task",
        ),
        code(
            """
            # TODO 19-4: class 이름 prompt를 text prototype으로 보고 held-out image를 zero-shot 분류하세요.
            test_labels = torch.arange(4).repeat_interleave(20)
            test_latent = class_centers[test_labels] + 0.35 * torch.randn(
                80,
                4,
                generator=generator,
            )
            test_images = test_latent @ image_map
            class_prompts = class_centers @ text_map
            test_images_device, class_prompts_device, test_labels_device = ACCELERATOR.move(
                test_images, class_prompts, test_labels
            )

            def zero_shot_predict(model, images, class_text_features):
                raise NotImplementedError("TODO 19-4")

            raise NotImplementedError("TODO 19-4")
            """,
            """
            test_labels = torch.arange(4).repeat_interleave(20)
            test_latent = class_centers[test_labels] + 0.35 * torch.randn(
                80,
                4,
                generator=generator,
            )
            test_images = test_latent @ image_map
            class_prompts = class_centers @ text_map
            test_images_device, class_prompts_device, test_labels_device = ACCELERATOR.move(
                test_images, class_prompts, test_labels
            )

            def zero_shot_predict(model, images, class_text_features):
                model.eval()
                logits = model.logits(images, class_text_features)
                return logits, logits.argmax(dim=1)

            with torch.no_grad():
                zero_shot_logits, zero_shot_predictions = zero_shot_predict(
                    clip_model,
                    test_images_device,
                    class_prompts_device,
                )
                zero_shot_accuracy = (
                    (zero_shot_predictions == test_labels_device)
                    .float()
                    .mean()
                    .item()
                )
            assert zero_shot_logits.shape == (len(test_images), len(class_prompts))
            assert zero_shot_accuracy > 0.85
            print(f"synthetic zero-shot accuracy={zero_shot_accuracy:.1%}")
            print("주의: class center vector는 실제 자연어 prompt를 단순화한 교육용 대응물입니다.")
            """,
            "task",
        ),
    ),
)


SPECS = [SIMCLR, VIT, DDPM, CLIP]
