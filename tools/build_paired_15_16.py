"""Build paired Transformer exercise/solution notebooks 15 and 16.

The paired notebooks deliberately share cell ids, markdown, ordering, and tests.
Only code-cell bodies differ: the exercise side exposes TODOs while the solution
side contains an executable reference implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat
from notebook_api_explanations import annotate_pair

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "notebooks" / "exercises"
SOLUTIONS = ROOT / "notebooks" / "solutions"


def clean(text: str) -> str:
    return dedent(text).strip()


def markdown(text: str) -> tuple[str, str]:
    return ("markdown", clean(text))


def code(exercise: str, solution: str) -> tuple[str, str, str]:
    return ("code", clean(exercise), clean(solution))


def build_pair(filename: str, title: str, specs: list[tuple[str, ...]]) -> None:
    EXERCISES.mkdir(parents=True, exist_ok=True)
    SOLUTIONS.mkdir(parents=True, exist_ok=True)
    notebooks: dict[str, nbformat.NotebookNode] = {}
    for role in ("exercise", "solution"):
        cells: list[nbformat.NotebookNode] = []
        for index, spec in enumerate(specs, start=1):
            cell_id = f"cell-{index:02d}"
            if spec[0] == "markdown":
                cell = nbformat.v4.new_markdown_cell(spec[1], id=cell_id)
            else:
                source = spec[1] if role == "exercise" else spec[2]
                cell = nbformat.v4.new_code_cell(source, id=cell_id)
                cell.metadata["tags"] = [role]
            cells.append(cell)
        notebook = nbformat.v4.new_notebook(cells=cells)
        notebook.metadata.update(
            {
                "kernelspec": {
                    "display_name": "Python (AI Engineering Lab)",
                    "language": "python",
                    "name": "ai-engineering-lab",
                },
                "language_info": {"name": "python", "version": "3.13"},
                "paired_lab": {
                    "title": title,
                    "role": role,
                    "counterpart": (
                        "../"
                        f"{'solutions' if role == 'exercise' else 'exercises'}"
                        f"/{filename}"
                    ),
                },
            }
        )
        notebooks[role] = notebook

    annotate_pair(notebooks["exercise"], notebooks["solution"])

    for role, notebook in notebooks.items():
        destination = (EXERCISES if role == "exercise" else SOLUTIONS) / filename
        nbformat.write(notebook, destination)


LAB15: list[tuple[str, ...]] = [
    markdown(
        """
        # 15. Transformer 구조 해부 실습 — 직접 만드는 causal Transformer

        이 파일은 **실습용/정답용이 같은 셀 순서를 갖는 페어 노트북**입니다. JupyterLab에서
        왼쪽에는 `exercises/15_...`, 오른쪽에는 `solutions/15_...`를 열고 셀 단위로 따라 치세요.

        이번 랩의 목표는 완제품 API를 호출하는 것이 아니라 `[B, T, D]` 텐서가 Q/K/V,
        head 분리, mask, residual을 통과하는 과정을 눈으로 확인하는 것입니다.
        """
    ),
    markdown(
        """
        ## 완료 기준과 권장 순서

        1. Token/position embedding을 더한다.
        2. Q/K/V projection과 scaled dot-product attention을 구현한다.
        3. `[B,T,D] ↔ [B,H,T,Dh]` 변환과 causal/padding mask를 검증한다.
        4. Pre-norm residual, FFN, decoder block, 작은 causal LM을 조립한다.
        5. shape, gradient, 미래 토큰 누설 방지 assert를 모두 통과시킨다.

        **학습법:** 정답 전체를 복사하지 말고 TODO 하나를 먼저 작성한 뒤 해당 검증 셀만 실행하세요.
        """
    ),
    code(
        """
        # 준비 셀: 이 셀은 그대로 실행해도 됩니다.
        import math
        import random
        from dataclasses import dataclass
        
        import torch
        from torch import Tensor, nn
        from torch.nn import functional as F
        from llm_engineering_lab.acceleration import get_accelerator
        
        SEED = 15
        random.seed(SEED)
        torch.manual_seed(SEED)
        ACCELERATOR = get_accelerator()
        device = ACCELERATOR.device
        print(f"{ACCELERATOR.summary()}, torch={torch.__version__}")
        """,
        """
        import math
        import random
        from dataclasses import dataclass
        
        import torch
        from torch import Tensor, nn
        from torch.nn import functional as F
        from llm_engineering_lab.acceleration import get_accelerator
        
        SEED = 15
        random.seed(SEED)
        torch.manual_seed(SEED)
        ACCELERATOR = get_accelerator()
        device = ACCELERATOR.device
        print(f"{ACCELERATOR.summary()}, torch={torch.__version__}")
        """,
    ),
    markdown(
        """
        ## 1) 미니 배치와 vocabulary

        `input_ids`의 축은 `[batch, time]`입니다. `<pad>`는 문장 길이를 맞추기 위한 토큰이고,
        `valid_tokens`가 False인 위치는 key로도 query로도 attention에 참여시키지 않습니다.
        실제 서비스에서는 tokenizer가 이 과정을 담당하지만 구조 학습을 위해 작은 단어 사전을 직접 만듭니다.
        """
    ),
    code(
        """
        # TODO 15-1: 아래 토큰으로 stoi/itos를 만들고 두 문장을 정수 텐서로 바꾸세요.
        token_rows = [
            ["<bos>", "나는", "모델을", "학습한다", "<eos>", "<pad>"],
            ["<bos>", "attention은", "문맥을", "연결한다", "<eos>", "<pad>"],
        ]
        vocab_tokens = ["<pad>", "<bos>", "<eos>"] + sorted(
            {token for row in token_rows for token in row if not token.startswith("<")}
        )
        # TODO: stoi, itos, input_ids, pad_id, valid_tokens를 완성하세요.
        stoi = ...
        itos = ...
        input_ids = ...
        pad_id = ...
        valid_tokens = ...
        """,
        """
        token_rows = [
            ["<bos>", "나는", "모델을", "학습한다", "<eos>", "<pad>"],
            ["<bos>", "attention은", "문맥을", "연결한다", "<eos>", "<pad>"],
        ]
        vocab_tokens = ["<pad>", "<bos>", "<eos>"] + sorted(
            {token for row in token_rows for token in row if not token.startswith("<")}
        )
        stoi = {token: index for index, token in enumerate(vocab_tokens)}
        itos = {index: token for token, index in stoi.items()}
        input_ids = torch.tensor(
            [[stoi[token] for token in row] for row in token_rows],
            dtype=torch.long,
            device=device,
        )
        pad_id = stoi["<pad>"]
        valid_tokens = input_ids.ne(pad_id)
        
        print("vocab:", stoi)
        print("input_ids shape:", tuple(input_ids.shape))
        print("valid tokens:\\n", valid_tokens)
        assert input_ids.shape == valid_tokens.shape == (2, 6)
        """,
    ),
    markdown(
        """
        ## 2) Token embedding + learned position embedding

        같은 단어라도 순서가 다르면 의미가 달라집니다. Token embedding은 **무엇인지**, position
        embedding은 **어디에 있는지**를 표현하고 둘은 같은 `model_dim`이므로 원소별로 더할 수 있습니다.
        출력 shape은 항상 `[B, T, D]`입니다.
        """
    ),
    code(
        """
        class TokenPositionEmbedding(nn.Module):
            def __init__(self, vocab_size: int, model_dim: int, max_length: int, pad_id: int):
                super().__init__()
                # TODO 15-2: token/position nn.Embedding 두 개를 정의하세요.
                self.token = ...
                self.position = ...
                self.max_length = max_length
        
            def forward(self, ids: Tensor) -> Tensor:
                # TODO: positions=[0..T-1]을 batch에 맞춰 더하세요.
                raise NotImplementedError("TODO 15-2")
        """,
        """
        class TokenPositionEmbedding(nn.Module):
            def __init__(self, vocab_size: int, model_dim: int, max_length: int, pad_id: int):
                super().__init__()
                self.token = nn.Embedding(vocab_size, model_dim, padding_idx=pad_id)
                self.position = nn.Embedding(max_length, model_dim)
                self.max_length = max_length
        
            def forward(self, ids: Tensor) -> Tensor:
                batch_size, length = ids.shape
                if length > self.max_length:
                    raise ValueError(f"length={length} exceeds max_length={self.max_length}")
                positions = torch.arange(length, device=ids.device).expand(batch_size, length)
                return self.token(ids) + self.position(positions)
        """,
    ),
    code(
        """
        # 검증: 구현 후 shape와 위치 변화가 표현에 반영되는지 확인하세요.
        model_dim = 32
        embedding = TokenPositionEmbedding(len(stoi), model_dim, 16, pad_id).to(device)
        hidden = embedding(input_ids)
        assert hidden.shape == (2, 6, model_dim)
        assert not torch.allclose(hidden[:, 0], hidden[:, 1])
        print("embedding:", tuple(hidden.shape), "dtype:", hidden.dtype)
        """,
        """
        model_dim = 32
        embedding = TokenPositionEmbedding(len(stoi), model_dim, 16, pad_id).to(device)
        hidden = embedding(input_ids)
        assert hidden.shape == (2, 6, model_dim)
        assert not torch.allclose(hidden[:, 0], hidden[:, 1])
        print("embedding:", tuple(hidden.shape), "dtype:", hidden.dtype)
        """,
    ),
    markdown(
        """
        ## 3) Q, K, V projection

        한 hidden vector를 세 관점으로 선형 변환합니다. Query는 "무엇을 찾는가", Key는
        "나는 무엇을 나타내는가", Value는 "선택되면 무엇을 전달하는가"로 이해할 수 있습니다.
        현업 구현은 메모리 접근을 줄이기 위해 보통 하나의 `Linear(D, 3D)` 결과를 세 덩어리로 나눕니다.
        """
    ),
    code(
        """
        # TODO 15-3: 하나의 projection 결과를 마지막 축에서 q, k, v로 나누세요.
        qkv_projection = nn.Linear(model_dim, 3 * model_dim).to(device)
        qkv = qkv_projection(hidden)
        query, key, value = ...
        """,
        """
        qkv_projection = nn.Linear(model_dim, 3 * model_dim).to(device)
        qkv = qkv_projection(hidden)
        query, key, value = qkv.chunk(3, dim=-1)
        assert query.shape == key.shape == value.shape == hidden.shape
        print("qkv packed:", tuple(qkv.shape), "q:", tuple(query.shape))
        """,
    ),
    markdown(
        """
        ## 4) Multi-head reshape

        `D = H × Dh`로 나누어 여러 관계를 병렬로 봅니다. 연산을 위해 `[B,T,D]`를
        `[B,H,T,Dh]`로 바꾸며, `transpose` 뒤에는 메모리가 비연속일 수 있으므로 합칠 때
        `contiguous()`를 사용하는 습관이 중요합니다.
        """
    ),
    code(
        """
        def split_heads(x: Tensor, num_heads: int) -> Tensor:
            # TODO 15-4: [B,T,D] -> [B,H,T,Dh]
            raise NotImplementedError("TODO 15-4")
        
        
        def merge_heads(x: Tensor) -> Tensor:
            # TODO 15-4: [B,H,T,Dh] -> [B,T,D]
            raise NotImplementedError("TODO 15-4")
        """,
        """
        def split_heads(x: Tensor, num_heads: int) -> Tensor:
            batch_size, length, width = x.shape
            if width % num_heads != 0:
                raise ValueError("model_dim must be divisible by num_heads")
            head_dim = width // num_heads
            return x.view(batch_size, length, num_heads, head_dim).transpose(1, 2)
        
        
        def merge_heads(x: Tensor) -> Tensor:
            batch_size, num_heads, length, head_dim = x.shape
            return x.transpose(1, 2).contiguous().view(batch_size, length, num_heads * head_dim)
        """,
    ),
    code(
        """
        num_heads = 4
        qh, kh, vh = (split_heads(t, num_heads) for t in (query, key, value))
        assert qh.shape == (2, 4, 6, 8)
        assert torch.allclose(merge_heads(qh), query)
        print("per head:", tuple(qh.shape), "merged:", tuple(merge_heads(qh).shape))
        """,
        """
        num_heads = 4
        qh, kh, vh = (split_heads(t, num_heads) for t in (query, key, value))
        assert qh.shape == (2, 4, 6, 8)
        assert torch.allclose(merge_heads(qh), query)
        print("per head:", tuple(qh.shape), "merged:", tuple(merge_heads(qh).shape))
        """,
    ),
    markdown(
        """
        ## 5) Scaled dot-product attention

        `scores = QKᵀ / sqrt(Dh)`입니다. 차원이 커지면 dot product 분산이 커져 softmax가
        지나치게 뾰족해지므로 `sqrt(Dh)`로 나눕니다. Mask는 **softmax 전에** 허용되지 않은
        score를 매우 작은 값으로 바꿉니다. 전부 mask된 padding query는 명시적으로 0으로 처리합니다.
        """
    ),
    code(
        """
        def scaled_dot_product_attention(
            q: Tensor, k: Tensor, v: Tensor, allowed: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            # TODO 15-5: score 계산 -> mask -> softmax -> value 가중합 순서로 구현하세요.
            raise NotImplementedError("TODO 15-5")
        """,
        """
        def scaled_dot_product_attention(
            q: Tensor, k: Tensor, v: Tensor, allowed: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            head_dim = q.shape[-1]
            scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
            if allowed is not None:
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            if allowed is not None:
                weights = torch.where(allowed, weights, torch.zeros_like(weights))
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            context = weights @ v
            return context, weights
        """,
    ),
    markdown(
        """
        ## 6) Causal mask + padding mask

        - Causal mask: query 위치 `t`가 `t+1` 이후 key를 보지 못하게 합니다.
        - Padding key mask: `<pad>`를 정보원으로 선택하지 못하게 합니다.
        - Padding query mask: 의미 없는 query 행의 attention을 0으로 만듭니다.

        최종 `allowed`는 broadcast 가능한 `[B,1,T,T]` boolean 텐서입니다.
        """
    ),
    code(
        """
        def build_allowed_mask(valid: Tensor) -> Tensor:
            # TODO 15-6: tril causal mask와 key/query validity를 AND 하세요.
            raise NotImplementedError("TODO 15-6")
        """,
        """
        def build_allowed_mask(valid: Tensor) -> Tensor:
            batch_size, length = valid.shape
            causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
            causal = causal.view(1, 1, length, length)
            key_is_valid = valid.view(batch_size, 1, 1, length)
            query_is_valid = valid.view(batch_size, 1, length, 1)
            return causal & key_is_valid & query_is_valid
        """,
    ),
    code(
        """
        allowed = build_allowed_mask(valid_tokens)
        context, attention_weights = scaled_dot_product_attention(qh, kh, vh, allowed)
        future = torch.ones(6, 6, dtype=torch.bool, device=device).triu(diagonal=1)
        assert allowed.shape == (2, 1, 6, 6)
        assert torch.all(attention_weights.masked_select(future.view(1, 1, 6, 6)) == 0)
        assert torch.all(attention_weights[..., -1, :] == 0)  # padding query
        assert torch.all(attention_weights[..., :, -1] == 0)  # padding key
        print("context:", tuple(context.shape), "mask passed")
        """,
        """
        allowed = build_allowed_mask(valid_tokens)
        context, attention_weights = scaled_dot_product_attention(qh, kh, vh, allowed)
        future = torch.ones(6, 6, dtype=torch.bool, device=device).triu(diagonal=1)
        assert allowed.shape == (2, 1, 6, 6)
        assert torch.all(attention_weights.masked_select(future.view(1, 1, 6, 6)) == 0)
        assert torch.all(attention_weights[..., -1, :] == 0)
        assert torch.all(attention_weights[..., :, -1] == 0)
        print("context:", tuple(context.shape), "mask passed")
        """,
    ),
    markdown(
        """
        ## 7) Attention을 사람이 읽을 수 있는 데이터로 보기

        아래 `attention_view`는 시각화 라이브러리에 바로 전달할 수 있는 토큰 label과 2차원 weight
        행렬입니다. 행은 query, 열은 key입니다. causal mask 때문에 대각선 오른쪽 값이 모두 0인지
        확인하세요. 정답 노트북은 추가 설치 없이도 보이도록 텍스트 heatmap 형태로 출력합니다.
        """
    ),
    code(
        """
        # TODO 15-7: sample=0, head=0의 [T,T] weight와 token label을 묶으세요.
        tokens_for_view = token_rows[0]
        matrix_for_view = ...
        attention_view = {"tokens": tokens_for_view, "weights": ...}
        """,
        """
        tokens_for_view = token_rows[0]
        matrix_for_view = attention_weights[0, 0].detach().cpu()
        attention_view = {
            "tokens": tokens_for_view,
            "weights": matrix_for_view.round(decimals=3).tolist(),
        }
        print("query/key".ljust(13), " ".join(f"{t[:7]:>7}" for t in tokens_for_view))
        for token, row in zip(tokens_for_view, attention_view["weights"], strict=True):
            print(f"{token[:11]:<13}", " ".join(f"{value:7.3f}" for value in row))
        """,
    ),
    markdown(
        """
        ## 8) Multi-head self-attention 모듈

        이제 흩어진 연산을 하나의 `nn.Module`로 묶습니다. `qkv` projection → head 분리 → attention
        → head 병합 → output projection 순서입니다. 학습/디버깅을 위해 weight도 함께 반환합니다.
        """
    ),
    code(
        """
        class MultiHeadSelfAttention(nn.Module):
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                # TODO 15-8: divisibility 검증, qkv/out projection을 정의하세요.
                self.num_heads = num_heads
                self.qkv = ...
                self.out = ...
        
            def forward(self, x: Tensor, allowed: Tensor) -> tuple[Tensor, Tensor]:
                # TODO: qkv -> split -> attention -> merge -> out
                raise NotImplementedError("TODO 15-8")
        """,
        """
        class MultiHeadSelfAttention(nn.Module):
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                if model_dim % num_heads != 0:
                    raise ValueError("model_dim must be divisible by num_heads")
                self.num_heads = num_heads
                self.qkv = nn.Linear(model_dim, 3 * model_dim)
                self.out = nn.Linear(model_dim, model_dim)
        
            def forward(self, x: Tensor, allowed: Tensor) -> tuple[Tensor, Tensor]:
                query, key, value = self.qkv(x).chunk(3, dim=-1)
                query, key, value = (
                    split_heads(tensor, self.num_heads) for tensor in (query, key, value)
                )
                context, weights = scaled_dot_product_attention(query, key, value, allowed)
                return self.out(merge_heads(context)), weights
        """,
    ),
    code(
        """
        mha = MultiHeadSelfAttention(model_dim, num_heads).to(device)
        mha_output, mha_weights = mha(hidden, allowed)
        assert mha_output.shape == hidden.shape
        assert mha_weights.shape == (2, num_heads, 6, 6)
        print("MHA output:", tuple(mha_output.shape))
        """,
        """
        mha = MultiHeadSelfAttention(model_dim, num_heads).to(device)
        mha_output, mha_weights = mha(hidden, allowed)
        assert mha_output.shape == hidden.shape
        assert mha_weights.shape == (2, num_heads, 6, 6)
        print("MHA output:", tuple(mha_output.shape))
        """,
    ),
    markdown(
        """
        ## 9) Pre-norm residual + Feed Forward Network

        Pre-norm block은 `x + Attention(LN(x))`, 이어서 `x + FFN(LN(x))`를 수행합니다.
        Residual 경로는 gradient가 깊은 네트워크를 통과하도록 돕고, FFN은 각 토큰 위치에 동일한
        두 층 MLP를 적용합니다. 중간 차원은 보통 model dimension보다 큽니다.
        """
    ),
    code(
        """
        class FeedForward(nn.Module):
            def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
                super().__init__()
                # TODO 15-9: Linear -> GELU -> Linear -> Dropout을 구성하세요.
                self.net = ...
        
            def forward(self, x: Tensor) -> Tensor:
                return self.net(x)
        
        
        class DecoderBlock(nn.Module):
            def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float):
                super().__init__()
                # TODO: 두 LayerNorm, attention, FFN, Dropout을 정의하세요.
                raise NotImplementedError("TODO 15-9")
        
            def forward(self, x: Tensor, allowed: Tensor) -> tuple[Tensor, Tensor]:
                # TODO: pre-norm residual 두 단계를 구현하세요.
                raise NotImplementedError("TODO 15-9")
        """,
        """
        class FeedForward(nn.Module):
            def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(model_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, model_dim),
                    nn.Dropout(dropout),
                )
        
            def forward(self, x: Tensor) -> Tensor:
                return self.net(x)
        
        
        class DecoderBlock(nn.Module):
            def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float):
                super().__init__()
                self.norm1 = nn.LayerNorm(model_dim)
                self.attention = MultiHeadSelfAttention(model_dim, num_heads)
                self.norm2 = nn.LayerNorm(model_dim)
                self.feed_forward = FeedForward(model_dim, ff_dim, dropout)
                self.dropout = nn.Dropout(dropout)
        
            def forward(self, x: Tensor, allowed: Tensor) -> tuple[Tensor, Tensor]:
                attention_output, weights = self.attention(self.norm1(x), allowed)
                x = x + self.dropout(attention_output)
                x = x + self.feed_forward(self.norm2(x))
                return x, weights
        """,
    ),
    code(
        """
        block = DecoderBlock(model_dim, num_heads, ff_dim=96, dropout=0.0).to(device)
        block_output, block_weights = block(hidden, allowed)
        probe_loss = block_output.square().mean()
        probe_loss.backward()
        gradient_norm = block.attention.qkv.weight.grad.norm().item()
        assert block_output.shape == hidden.shape
        assert gradient_norm > 0
        print(f"block={tuple(block_output.shape)}, qkv grad norm={gradient_norm:.6f}")
        """,
        """
        block = DecoderBlock(model_dim, num_heads, ff_dim=96, dropout=0.0).to(device)
        block_output, block_weights = block(hidden, allowed)
        probe_loss = block_output.square().mean()
        probe_loss.backward()
        gradient_norm = block.attention.qkv.weight.grad.norm().item()
        assert block_output.shape == hidden.shape
        assert gradient_norm > 0
        print(f"block={tuple(block_output.shape)}, qkv grad norm={gradient_norm:.6f}")
        """,
    ),
    markdown(
        """
        ## 10) Decoder-only causal language model 조립

        여러 decoder block 뒤에 final LayerNorm과 vocabulary projection(`lm_head`)을 붙입니다.
        언어 모델의 logit shape은 `[B,T,V]`입니다. 위치 `t`의 logit이 다음 token `t+1`을 맞히도록
        target을 한 칸 shift하며, padding target은 `ignore_index`로 loss에서 제외합니다.
        """
    ),
    code(
        """
        @dataclass(frozen=True)
        class ArchitectureConfig:
            vocab_size: int
            pad_id: int
            max_length: int = 16
            model_dim: int = 32
            num_heads: int = 4
            num_layers: int = 2
            ff_dim: int = 96
            dropout: float = 0.0
        
        
        class TinyCausalLM(nn.Module):
            def __init__(self, config: ArchitectureConfig):
                super().__init__()
                # TODO 15-10: embedding, block 목록, final norm, lm_head를 정의하세요.
                self.config = config
                raise NotImplementedError("TODO 15-10")
        
            def forward(
                self, ids: Tensor, labels: Tensor | None = None
            ) -> tuple[Tensor, Tensor | None, list[Tensor]]:
                # TODO: mask 생성 -> blocks -> logits -> shifted CE loss
                raise NotImplementedError("TODO 15-10")
        """,
        """
        @dataclass(frozen=True)
        class ArchitectureConfig:
            vocab_size: int
            pad_id: int
            max_length: int = 16
            model_dim: int = 32
            num_heads: int = 4
            num_layers: int = 2
            ff_dim: int = 96
            dropout: float = 0.0
        
        
        class TinyCausalLM(nn.Module):
            def __init__(self, config: ArchitectureConfig):
                super().__init__()
                self.config = config
                self.embedding = TokenPositionEmbedding(
                    config.vocab_size, config.model_dim, config.max_length, config.pad_id
                )
                self.blocks = nn.ModuleList(
                    [
                        DecoderBlock(
                            config.model_dim,
                            config.num_heads,
                            config.ff_dim,
                            config.dropout,
                        )
                        for _ in range(config.num_layers)
                    ]
                )
                self.final_norm = nn.LayerNorm(config.model_dim)
                self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
                self.lm_head.weight = self.embedding.token.weight
        
            def forward(
                self, ids: Tensor, labels: Tensor | None = None
            ) -> tuple[Tensor, Tensor | None, list[Tensor]]:
                allowed = build_allowed_mask(ids.ne(self.config.pad_id))
                x = self.embedding(ids)
                all_weights: list[Tensor] = []
                for block in self.blocks:
                    x, weights = block(x, allowed)
                    all_weights.append(weights)
                logits = self.lm_head(self.final_norm(x))
                loss = None
                if labels is not None:
                    targets = labels[:, 1:].masked_fill(
                        labels[:, 1:].eq(self.config.pad_id), -100
                    )
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, self.config.vocab_size),
                        targets.reshape(-1),
                        ignore_index=-100,
                    )
                return logits, loss, all_weights
        """,
    ),
    code(
        """
        architecture_config = ArchitectureConfig(vocab_size=len(stoi), pad_id=pad_id)
        tiny_lm = TinyCausalLM(architecture_config).to(device)
        logits, lm_loss, layer_weights = tiny_lm(input_ids, labels=input_ids)
        assert logits.shape == (2, 6, len(stoi))
        assert lm_loss is not None and torch.isfinite(lm_loss)
        assert len(layer_weights) == architecture_config.num_layers
        print(f"logits={tuple(logits.shape)}, shifted loss={lm_loss.item():.4f}")
        """,
        """
        architecture_config = ArchitectureConfig(vocab_size=len(stoi), pad_id=pad_id)
        tiny_lm = TinyCausalLM(architecture_config).to(device)
        logits, lm_loss, layer_weights = tiny_lm(input_ids, labels=input_ids)
        assert logits.shape == (2, 6, len(stoi))
        assert lm_loss is not None and torch.isfinite(lm_loss)
        assert len(layer_weights) == architecture_config.num_layers
        print(f"logits={tuple(logits.shape)}, shifted loss={lm_loss.item():.4f}")
        """,
    ),
    markdown(
        """
        ## 11) 누설(leakage) 테스트 — mask가 정말 작동하는가?

        미래 위치의 token 하나만 바꿨을 때 그보다 앞선 위치의 logit은 같아야 합니다. 이 테스트는
        mask 방향을 뒤집거나 softmax 뒤에 mask하는 흔한 버그를 잡습니다. Dropout은 비교를 흔들 수
        있으므로 `eval()`에서 검사합니다.
        """
    ),
    code(
        """
        tiny_lm.eval()
        changed = input_ids.clone()
        changed[0, 3] = stoi["연결한다"]
        with torch.inference_mode():
            original_logits, _, _ = tiny_lm(input_ids)
            changed_logits, _, _ = tiny_lm(changed)
        
        # 위치 1은 미래 위치 3을 볼 수 없어야 합니다.
        assert torch.allclose(original_logits[0, 1], changed_logits[0, 1], atol=1e-6)
        # 바뀐 위치 이후에는 차이가 생기는 것이 정상입니다.
        difference_after_change = (original_logits[0, 3] - changed_logits[0, 3]).abs().max()
        print(
            "causal leakage test passed; later max difference:", difference_after_change.item()
        )
        """,
        """
        tiny_lm.eval()
        changed = input_ids.clone()
        changed[0, 3] = stoi["연결한다"]
        with torch.inference_mode():
            original_logits, _, _ = tiny_lm(input_ids)
            changed_logits, _, _ = tiny_lm(changed)
        
        assert torch.allclose(original_logits[0, 1], changed_logits[0, 1], atol=1e-6)
        difference_after_change = (original_logits[0, 3] - changed_logits[0, 3]).abs().max()
        print(
            "causal leakage test passed; later max difference:", difference_after_change.item()
        )
        """,
    ),
    markdown(
        """
        ## 마무리 체크

        이제 다음 질문에 코드로 답할 수 있어야 합니다.

        - 왜 score를 `sqrt(head_dim)`으로 나누는가?
        - causal mask와 padding mask가 각각 어느 축을 막는가?
        - head를 분리/병합할 때 `transpose`와 `contiguous`가 왜 필요한가?
        - Pre-norm residual의 두 경로는 무엇인가?
        - 언어 모델 loss에서 logit과 target을 어떻게 한 칸 이동하는가?

        다음 랩에서는 이 구조와 프로젝트의 `TinyCausalTransformer`를 이용해 optimizer, AMP,
        gradient accumulation, checkpoint, generation까지 하나의 학습 파이프라인으로 완성합니다.
        """
    ),
]


LAB16: list[tuple[str, ...]] = [
    markdown(
        """
        # 16. Tiny Transformer 학습 실습 — corpus에서 생성까지

        이 랩은 15번에서 확인한 decoder-only Transformer를 실제로 학습시키는 작은 현업형 루프입니다.
        왼쪽에는 exercise, 오른쪽에는 solution을 띄우고 tokenizer → dataset → teacher forcing →
        optimizer/scheduler → AMP → accumulation/clipping → validation/perplexity → checkpoint → 생성
        순으로 따라 치세요.
        """
    ),
    markdown(
        """
        ## 실행 규모와 완료 기준

        기본 설정은 `D=64, heads=4, layers=2, context=64`로 CPU에서도 수분 이내이며 CUDA가 있으면
        자동으로 AMP를 사용합니다. RTX 3080에서는 구조를 이해한 뒤 `model_dim=128`, `layers=4`,
        `batch_size=64` 정도로 늘려 비교해 보세요. 큰 설정은 VRAM보다 먼저 데이터가 병목이 됩니다.

        완료 기준은 validation perplexity 계산, best checkpoint 재로딩, greedy/top-k 생성까지입니다.
        """
    ),
    code(
        """
        # 준비 셀: 그대로 실행합니다.
        import math
        import random
        import tempfile
        from contextlib import nullcontext
        from dataclasses import asdict
        from pathlib import Path
        
        import torch
        from torch import Tensor, nn
        from torch.utils.data import DataLoader, Dataset
        
        from llm_engineering_lab.transformer import (
            TinyCausalTransformer,
            TinyTransformerConfig,
        )
        from llm_engineering_lab.acceleration import get_accelerator
        
        SEED = 16
        random.seed(SEED)
        torch.manual_seed(SEED)
        ACCELERATOR = get_accelerator()
        DEVICE = ACCELERATOR.device
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        print(ACCELERATOR.summary())
        """,
        """
        import math
        import random
        import tempfile
        from contextlib import nullcontext
        from dataclasses import asdict
        from pathlib import Path
        
        import torch
        from torch import Tensor, nn
        from torch.utils.data import DataLoader, Dataset
        
        from llm_engineering_lab.transformer import (
            TinyCausalTransformer,
            TinyTransformerConfig,
        )
        from llm_engineering_lab.acceleration import get_accelerator
        
        SEED = 16
        random.seed(SEED)
        torch.manual_seed(SEED)
        ACCELERATOR = get_accelerator()
        DEVICE = ACCELERATOR.device
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        print(ACCELERATOR.summary())
        """,
    ),
    markdown(
        """
        ## 1) 작은 corpus 준비

        인터넷 다운로드 없이 반복 실행할 수 있는 짧은 한국어 corpus입니다. 작은 데이터는 좋은 문장을
        만들기보다 **파이프라인 검증**에 적합합니다. train/validation 문장을 나누되 tokenizer 문자는
        전체 목록으로 구성해 `<unk>` 처리도 실습할 수 있게 합니다.
        """
    ),
    code(
        """
        corpus = [
            "모델은 데이터에서 패턴을 배운다.",
            "학습 루프는 손실을 줄인다.",
            "어텐션은 토큰 사이의 관계를 찾는다.",
            "트랜스포머는 문맥을 병렬로 처리한다.",
            "옵티마이저는 가중치를 갱신한다.",
            "검증 데이터는 일반화를 확인한다.",
            "그라디언트는 역전파로 계산한다.",
            "체크포인트는 좋은 모델을 저장한다.",
            "언어 모델은 다음 토큰을 예측한다.",
            "마스크는 미래 정보를 가린다.",
            "임베딩은 토큰을 벡터로 바꾼다.",
            "생성은 한 토큰씩 반복한다.",
            "퍼플렉서티는 예측의 불확실성을 나타낸다.",
            "탑케이 샘플링은 후보를 제한한다.",
            "작은 실험으로 코드를 먼저 검증한다.",
            "재현성을 위해 시드를 기록한다.",
        ]
        train_documents = corpus[:12]
        validation_documents = corpus[12:]
        train_text = "\\n".join(train_documents * 6)
        validation_text = "\\n".join(validation_documents * 3)
        print(len(train_text), len(validation_text), train_text[:45])
        """,
        """
        corpus = [
            "모델은 데이터에서 패턴을 배운다.",
            "학습 루프는 손실을 줄인다.",
            "어텐션은 토큰 사이의 관계를 찾는다.",
            "트랜스포머는 문맥을 병렬로 처리한다.",
            "옵티마이저는 가중치를 갱신한다.",
            "검증 데이터는 일반화를 확인한다.",
            "그라디언트는 역전파로 계산한다.",
            "체크포인트는 좋은 모델을 저장한다.",
            "언어 모델은 다음 토큰을 예측한다.",
            "마스크는 미래 정보를 가린다.",
            "임베딩은 토큰을 벡터로 바꾼다.",
            "생성은 한 토큰씩 반복한다.",
            "퍼플렉서티는 예측의 불확실성을 나타낸다.",
            "탑케이 샘플링은 후보를 제한한다.",
            "작은 실험으로 코드를 먼저 검증한다.",
            "재현성을 위해 시드를 기록한다.",
        ]
        train_documents = corpus[:12]
        validation_documents = corpus[12:]
        train_text = "\\n".join(train_documents * 6)
        validation_text = "\\n".join(validation_documents * 3)
        print(len(train_text), len(validation_text), train_text[:45])
        """,
    ),
    markdown(
        """
        ## 2) Character tokenizer

        BPE 대신 문자 tokenizer를 쓰면 외부 라이브러리 없이 encode/decode와 special token 역할을
        완전히 볼 수 있습니다. `<pad>`는 batch 정렬, `<bos>/<eos>`는 문장 경계, `<unk>`는 사전에
        없는 문자를 담당합니다. 실무에서는 tokenizer도 model artifact와 함께 버전 관리합니다.
        """
    ),
    code(
        """
        class CharTokenizer:
            def __init__(self, texts: list[str]):
                # TODO 16-1: special token + 정렬된 문자로 stoi/itos를 만드세요.
                raise NotImplementedError("TODO 16-1")
        
            def encode(self, text: str, *, boundaries: bool = False) -> list[int]:
                # TODO: unknown fallback과 선택적 BOS/EOS를 구현하세요.
                raise NotImplementedError("TODO 16-1")
        
            def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
                # TODO: id를 문자로 복원하고 필요하면 special token을 제거하세요.
                raise NotImplementedError("TODO 16-1")
        """,
        """
        class CharTokenizer:
            def __init__(self, texts: list[str]):
                self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
                characters = sorted(set("".join(texts)))
                self.itos = self.special_tokens + characters
                self.stoi = {token: index for index, token in enumerate(self.itos)}
                self.pad_id = self.stoi["<pad>"]
                self.bos_id = self.stoi["<bos>"]
                self.eos_id = self.stoi["<eos>"]
                self.unk_id = self.stoi["<unk>"]
        
            def encode(self, text: str, *, boundaries: bool = False) -> list[int]:
                ids = [self.stoi.get(character, self.unk_id) for character in text]
                return [self.bos_id, *ids, self.eos_id] if boundaries else ids
        
            def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
                tokens = [self.itos[index] for index in ids]
                if skip_special:
                    tokens = [token for token in tokens if token not in self.special_tokens]
                return "".join(tokens)
        
            def __len__(self) -> int:
                return len(self.itos)
        """,
    ),
    code(
        """
        tokenizer = CharTokenizer(corpus)
        sample_ids = tokenizer.encode("모델은 학습한다.", boundaries=True)
        assert sample_ids[0] == tokenizer.bos_id and sample_ids[-1] == tokenizer.eos_id
        assert tokenizer.decode(sample_ids) == "모델은 학습한다."
        print("vocab size:", len(tokenizer), "round trip:", tokenizer.decode(sample_ids))
        """,
        """
        tokenizer = CharTokenizer(corpus)
        sample_ids = tokenizer.encode("모델은 학습한다.", boundaries=True)
        assert sample_ids[0] == tokenizer.bos_id and sample_ids[-1] == tokenizer.eos_id
        assert tokenizer.decode(sample_ids) == "모델은 학습한다."
        print("vocab size:", len(tokenizer), "round trip:", tokenizer.decode(sample_ids))
        """,
    ),
    markdown(
        """
        ## 3) Sliding-window language-model dataset

        긴 token stream을 `block_size` 길이의 창으로 자릅니다. 마지막 창은 pad하고 labels는 input을
        복사합니다. 모델 내부에서 `logits[:, :-1]`과 `labels[:, 1:]`를 맞추므로 한 샘플 안에
        최소 두 token이 필요합니다. `stride < block_size`이면 겹치는 문맥을 더 많이 학습합니다.
        """
    ),
    code(
        """
        class NextTokenDataset(Dataset):
            def __init__(self, token_ids: list[int], block_size: int, stride: int, pad_id: int):
                # TODO 16-2: 입력 검사와 window 시작 위치를 저장하세요.
                raise NotImplementedError("TODO 16-2")
        
            def __len__(self) -> int:
                raise NotImplementedError("TODO 16-2")
        
            def __getitem__(self, index: int) -> dict[str, Tensor]:
                # TODO: block_size만큼 자르고 pad한 input/mask/labels를 반환하세요.
                raise NotImplementedError("TODO 16-2")
        """,
        """
        class NextTokenDataset(Dataset):
            def __init__(self, token_ids: list[int], block_size: int, stride: int, pad_id: int):
                if len(token_ids) < 2 or block_size < 2 or stride < 1:
                    raise ValueError("token stream/block_size/stride를 확인하세요")
                self.token_ids = token_ids
                self.block_size = block_size
                self.pad_id = pad_id
                self.starts = list(range(0, len(token_ids) - 1, stride))
        
            def __len__(self) -> int:
                return len(self.starts)
        
            def __getitem__(self, index: int) -> dict[str, Tensor]:
                start = self.starts[index]
                ids = self.token_ids[start : start + self.block_size]
                ids = ids + [self.pad_id] * (self.block_size - len(ids))
                input_ids = torch.tensor(ids, dtype=torch.long)
                return {
                    "input_ids": input_ids,
                    "attention_mask": input_ids.ne(self.pad_id),
                    "labels": input_ids.clone(),
                }
        """,
    ),
    code(
        """
        block_size = 64
        train_ids = tokenizer.encode(train_text, boundaries=True)
        validation_ids = tokenizer.encode(validation_text, boundaries=True)
        train_dataset = NextTokenDataset(
            train_ids, block_size, stride=32, pad_id=tokenizer.pad_id
        )
        validation_dataset = NextTokenDataset(
            validation_ids, block_size, stride=64, pad_id=tokenizer.pad_id
        )
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, pin_memory=ACCELERATOR.pin_memory
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=16, shuffle=False, pin_memory=ACCELERATOR.pin_memory
        )
        first_batch = next(iter(train_loader))
        assert first_batch["input_ids"].shape[1] == block_size
        print(
            "windows:",
            len(train_dataset),
            len(validation_dataset),
            "batch:",
            first_batch["input_ids"].shape,
        )
        """,
        """
        block_size = 64
        train_ids = tokenizer.encode(train_text, boundaries=True)
        validation_ids = tokenizer.encode(validation_text, boundaries=True)
        train_dataset = NextTokenDataset(
            train_ids, block_size, stride=32, pad_id=tokenizer.pad_id
        )
        validation_dataset = NextTokenDataset(
            validation_ids, block_size, stride=64, pad_id=tokenizer.pad_id
        )
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, pin_memory=ACCELERATOR.pin_memory
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=16, shuffle=False, pin_memory=ACCELERATOR.pin_memory
        )
        first_batch = next(iter(train_loader))
        assert first_batch["input_ids"].shape[1] == block_size
        print(
            "windows:",
            len(train_dataset),
            len(validation_dataset),
            "batch:",
            first_batch["input_ids"].shape,
        )
        """,
    ),
    markdown(
        """
        ## 4) Teacher forcing의 shift 확인

        학습 시 정답 prefix를 입력으로 주고 각 위치가 바로 다음 token을 예측하게 합니다. 예를 들어
        `모 → 델`, `델 → 은`처럼 입력과 target을 한 칸 어긋나게 비교합니다. 생성 단계에서는 정답
        prefix가 없으므로 모델이 방금 생성한 token을 다시 입력하는 autoregressive loop가 필요합니다.
        """
    ),
    code(
        """
        preview = first_batch["input_ids"][0, :12]
        teacher_inputs = preview[:-1]
        next_token_targets = preview[1:]
        print("input :", tokenizer.decode(teacher_inputs.tolist(), skip_special=False))
        print("target:", tokenizer.decode(next_token_targets.tolist(), skip_special=False))
        assert torch.equal(teacher_inputs[1:], next_token_targets[:-1])
        """,
        """
        preview = first_batch["input_ids"][0, :12]
        teacher_inputs = preview[:-1]
        next_token_targets = preview[1:]
        print("input :", tokenizer.decode(teacher_inputs.tolist(), skip_special=False))
        print("target:", tokenizer.decode(next_token_targets.tolist(), skip_special=False))
        assert torch.equal(teacher_inputs[1:], next_token_targets[:-1])
        """,
    ),
    markdown(
        """
        ## 5) 작은 causal Transformer 구성

        15번 구조와 동일한 프로젝트 구현을 사용해 이번에는 학습 루프에 집중합니다. parameter 수를
        기록하면 실험 규모와 checkpoint 크기를 추적하기 쉽습니다. `max_seq_length`는 dataset의
        `block_size` 이상이어야 합니다.
        """
    ),
    code(
        """
        # TODO 16-3: 아래 크기로 config와 model을 만들고 DEVICE로 이동하세요.
        lm_config = TinyTransformerConfig(
            vocab_size=len(tokenizer),
            pad_id=tokenizer.pad_id,
            max_seq_length=block_size,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            feed_forward_dim=128,
            dropout=0.1,
        )
        model = ...
        parameter_count = ...
        """,
        """
        lm_config = TinyTransformerConfig(
            vocab_size=len(tokenizer),
            pad_id=tokenizer.pad_id,
            max_seq_length=block_size,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            feed_forward_dim=128,
            dropout=0.1,
        )
        model = TinyCausalTransformer(lm_config).to(DEVICE)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert parameter_count < 1_000_000
        print(f"parameters={parameter_count:,}")
        """,
    ),
    markdown(
        """
        ## 6) AdamW + warmup/cosine scheduler

        Warmup은 초기 불안정한 update를 줄이고, 이후 cosine decay로 learning rate를 낮춥니다.
        Scheduler는 **optimizer update 직후 한 번** 호출합니다. Gradient accumulation 중 micro-batch마다
        호출하면 계획보다 너무 빨리 감소하므로 update step 수를 기준으로 계산해야 합니다.
        """
    ),
    code(
        """
        epochs = 3
        accumulation_steps = 2
        updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        total_update_steps = epochs * updates_per_epoch
        warmup_steps = max(1, total_update_steps // 10)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        
        
        def lr_multiplier(step: int) -> float:
            # TODO 16-4: warmup 후 cosine decay가 되도록 0~1 배율을 반환하세요.
            raise NotImplementedError("TODO 16-4")
        
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        """,
        """
        epochs = 3
        accumulation_steps = 2
        updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        total_update_steps = epochs * updates_per_epoch
        warmup_steps = max(1, total_update_steps // 10)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        
        
        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return max(1e-3, (step + 1) / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        print("updates:", total_update_steps, "warmup:", warmup_steps)
        """,
    ),
    markdown(
        """
        ## 7) AMP — CUDA에서만 mixed precision

        RTX GPU에서는 autocast가 일부 연산을 FP16으로 수행해 속도와 메모리를 개선할 수 있습니다.
        `GradScaler`는 작은 gradient가 underflow되는 것을 완화합니다. CPU에서는 `nullcontext`와
        disabled scaler를 사용해 동일한 학습 함수가 그대로 동작하게 만듭니다.
        """
    ),
    code(
        """
        amp_enabled = ACCELERATOR.amp_enabled
        scaler = ACCELERATOR.grad_scaler()
        
        
        def amp_context():
            # TODO 16-5: CUDA면 float16 autocast, 아니면 nullcontext를 반환하세요.
            raise NotImplementedError("TODO 16-5")
        """,
        """
        amp_enabled = ACCELERATOR.amp_enabled
        scaler = ACCELERATOR.grad_scaler()
        
        
        def amp_context():
            return ACCELERATOR.autocast()
        
        
        print("AMP enabled:", amp_enabled)
        """,
    ),
    markdown(
        """
        ## 8) Gradient accumulation + clipping 학습 함수

        Accumulation은 여러 micro-batch의 gradient를 더해 큰 effective batch를 흉내 냅니다. Loss를
        `accumulation_steps`로 나눠 scale을 유지하고, 마지막 남은 micro-batch도 update해야 합니다.
        AMP 사용 시 `unscale_` 후 clipping하여 **실제 gradient norm**을 기준으로 제한합니다.
        """
    ),
    code(
        """
        def train_one_epoch(
            model: nn.Module,
            loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler,
            scaler,
            accumulation_steps: int,
            clip_norm: float,
        ) -> dict[str, float]:
            # TODO 16-6: AMP, accumulation, unscale, clipping, optimizer/scheduler step을 구현하세요.
            raise NotImplementedError("TODO 16-6")
        """,
        """
        def train_one_epoch(
            model: nn.Module,
            loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler,
            scaler,
            accumulation_steps: int,
            clip_norm: float,
        ) -> dict[str, float]:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            token_count = 0
            last_grad_norm = 0.0
        
            for micro_step, batch in enumerate(loader):
                input_ids, attention_mask, labels = ACCELERATOR.move(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                with amp_context():
                    output = model(input_ids, attention_mask=attention_mask, labels=labels)
                    raw_loss = output.loss
                    assert raw_loss is not None
                    scaled_loss = raw_loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
        
                targets = labels[:, 1:].ne(tokenizer.pad_id).sum().item()
                weighted_loss += raw_loss.detach().item() * targets
                token_count += targets
        
                update_now = (micro_step + 1) % accumulation_steps == 0
                update_now |= micro_step + 1 == len(loader)
                if update_now:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    last_grad_norm = float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
        
            mean_loss = weighted_loss / token_count
            return {
                "loss": mean_loss,
                "perplexity": math.exp(min(mean_loss, 20.0)),
                "tokens": float(token_count),
                "grad_norm": last_grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
            }
        """,
    ),
    markdown(
        """
        ## 9) Validation loss와 perplexity

        Validation에서는 `eval()`과 `inference_mode()`를 사용하고 optimizer를 건드리지 않습니다.
        Batch별 loss 평균이 아니라 **유효 target token 수로 가중 평균**해야 마지막 padding batch 때문에
        지표가 왜곡되지 않습니다. Perplexity는 `exp(cross_entropy)`이며 낮을수록 좋습니다.
        """
    ),
    code(
        """
        @torch.inference_mode()
        def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
            # TODO 16-7: token-weighted validation loss와 perplexity를 계산하세요.
            raise NotImplementedError("TODO 16-7")
        """,
        """
        @torch.inference_mode()
        def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
            model.eval()
            weighted_loss = 0.0
            token_count = 0
            for batch in loader:
                input_ids, attention_mask, labels = ACCELERATOR.move(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                assert output.loss is not None
                targets = labels[:, 1:].ne(tokenizer.pad_id).sum().item()
                weighted_loss += output.loss.item() * targets
                token_count += targets
            mean_loss = weighted_loss / token_count
            return {
                "loss": mean_loss,
                "perplexity": math.exp(min(mean_loss, 20.0)),
                "tokens": float(token_count),
            }
        """,
    ),
    markdown(
        """
        ## 10) Best checkpoint를 저장하는 학습 loop

        실제 프로젝트에서는 model뿐 아니라 config, optimizer, scheduler, epoch, metric, tokenizer 정보를
        함께 저장해야 재현할 수 있습니다. 이 랩은 작업 폴더를 어지럽히지 않도록 임시 디렉터리에
        checkpoint를 만들고, 뒤에서 실제로 재로딩한 후 삭제합니다.
        """
    ),
    code(
        """
        checkpoint_path = Path(tempfile.gettempdir()) / "ai_lab16_best_tiny_transformer.pt"
        history: list[dict[str, float]] = []
        best_validation_loss = float("inf")
        
        # TODO 16-8: epochs 동안 train/evaluate하고 best validation checkpoint를 저장하세요.
        raise NotImplementedError("TODO 16-8")
        """,
        """
        checkpoint_path = Path(tempfile.gettempdir()) / "ai_lab16_best_tiny_transformer.pt"
        history: list[dict[str, float]] = []
        best_validation_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                accumulation_steps=accumulation_steps,
                clip_norm=1.0,
            )
            validation_metrics = evaluate(model, validation_loader)
            record = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_ppl": train_metrics["perplexity"],
                "validation_loss": validation_metrics["loss"],
                "validation_ppl": validation_metrics["perplexity"],
                "lr": train_metrics["lr"],
                "grad_norm": train_metrics["grad_norm"],
            }
            history.append(record)
            print(
                f"epoch={epoch} train={record['train_loss']:.3f} "
                f"val={record['validation_loss']:.3f} "
                f"val_ppl={record['validation_ppl']:.2f} lr={record['lr']:.2e}"
            )
            if record["validation_loss"] < best_validation_loss:
                best_validation_loss = record["validation_loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "model_config": asdict(lm_config),
                        "epoch": epoch,
                        "metrics": record,
                        "tokenizer_itos": tokenizer.itos,
                    },
                    checkpoint_path,
                )
        """,
    ),
    code(
        """
        # 지표/파일 검증: loss가 반드시 단조 감소할 필요는 없고 best가 저장되면 됩니다.
        assert len(history) == epochs
        assert all(math.isfinite(row["validation_ppl"]) for row in history)
        assert checkpoint_path.exists() and checkpoint_path.stat().st_size > 0
        print("best validation loss:", round(best_validation_loss, 4))
        print("checkpoint:", checkpoint_path, checkpoint_path.stat().st_size, "bytes")
        """,
        """
        assert len(history) == epochs
        assert all(math.isfinite(row["validation_ppl"]) for row in history)
        assert checkpoint_path.exists() and checkpoint_path.stat().st_size > 0
        print("best validation loss:", round(best_validation_loss, 4))
        print("checkpoint:", checkpoint_path, checkpoint_path.stat().st_size, "bytes")
        """,
    ),
    markdown(
        """
        ## 11) 새 model 객체에 checkpoint 재로딩

        저장 직후 같은 객체를 계속 쓰는 것만으로는 checkpoint가 온전한지 알 수 없습니다. Config에서
        새 model을 만들고 state dict를 로드한 다음 validation을 다시 계산합니다. CPU/GPU 이동을 위해
        `map_location=DEVICE`를 명시합니다.
        """
    ),
    code(
        """
        # TODO 16-9: checkpoint를 불러와 새 TinyCausalTransformer에 state를 적용하세요.
        payload = ...
        reloaded_model = ...
        reloaded_metrics = ...
        """,
        """
        payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        reloaded_config = TinyTransformerConfig(**payload["model_config"])
        reloaded_model = TinyCausalTransformer(reloaded_config).to(DEVICE)
        reloaded_model.load_state_dict(payload["model_state"])
        reloaded_metrics = evaluate(reloaded_model, validation_loader)
        assert abs(reloaded_metrics["loss"] - payload["metrics"]["validation_loss"]) < 1e-5
        print(
            "reloaded epoch:",
            payload["epoch"],
            "ppl:",
            round(reloaded_metrics["perplexity"], 2),
        )
        """,
    ),
    markdown(
        """
        ## 12) Greedy와 top-k 생성

        Greedy는 매번 가장 큰 logit을 선택해 결정적이지만 반복이 생기기 쉽습니다. Top-k는 확률이 큰
        k개 token만 남긴 뒤 sampling하여 다양성을 줍니다. Temperature가 낮을수록 보수적입니다.
        이 tiny corpus 결과의 문장 품질보다 **train/eval 전환, context limit, 재현 가능한 sampling**을 보세요.
        """
    ),
    code(
        """
        def generate_text(
            model: TinyCausalTransformer,
            prompt: str,
            *,
            max_new_tokens: int = 28,
            strategy: str = "greedy",
        ) -> str:
            # TODO 16-10: prompt encode -> model.generate -> decode를 구현하세요.
            # greedy: temperature=0, top_k=None / top-k: temperature=0.8, top_k=8
            raise NotImplementedError("TODO 16-10")
        """,
        """
        def generate_text(
            model: TinyCausalTransformer,
            prompt: str,
            *,
            max_new_tokens: int = 28,
            strategy: str = "greedy",
        ) -> str:
            prompt_ids = tokenizer.encode(prompt, boundaries=False)
            input_ids = torch.tensor(
                [[tokenizer.bos_id, *prompt_ids]], dtype=torch.long, device=DEVICE
            )
            available = model.config.max_seq_length - input_ids.shape[1]
            max_new_tokens = max(0, min(max_new_tokens, available))
            if strategy == "greedy":
                temperature, top_k = 0.0, None
            elif strategy == "top-k":
                temperature, top_k = 0.8, 8
            else:
                raise ValueError("strategy must be 'greedy' or 'top-k'")
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=1.0,
                eos_id=tokenizer.eos_id,
                use_cache=True,
            )
            return tokenizer.decode(generated[0].tolist())
        """,
    ),
    code(
        """
        reloaded_model.eval()
        torch.manual_seed(SEED)
        prompt = "모델은"
        greedy_text = generate_text(reloaded_model, prompt, strategy="greedy")
        torch.manual_seed(SEED)
        topk_text = generate_text(reloaded_model, prompt, strategy="top-k")
        assert greedy_text.startswith(prompt) and topk_text.startswith(prompt)
        print("greedy:", greedy_text)
        print("top-k :", topk_text)
        """,
        """
        reloaded_model.eval()
        torch.manual_seed(SEED)
        prompt = "모델은"
        greedy_text = generate_text(reloaded_model, prompt, strategy="greedy")
        torch.manual_seed(SEED)
        topk_text = generate_text(reloaded_model, prompt, strategy="top-k")
        assert greedy_text.startswith(prompt) and topk_text.startswith(prompt)
        print("greedy:", greedy_text)
        print("top-k :", topk_text)
        """,
    ),
    markdown(
        """
        ## 13) GPU/모델 규모 확장 실험

        RTX 3080에서 아래를 한 번에 모두 키우지 말고 하나씩 바꾸며 `tokens/sec`, peak VRAM,
        validation perplexity를 기록하세요.

        - `model_dim`: 64 → 128 (head 수로 나누어져야 함)
        - `num_layers`: 2 → 4
        - `block_size`: 64 → 128 (attention 메모리는 길이의 제곱에 비례)
        - `batch_size`: 16 → 32/64, 부족하면 accumulation 증가
        - AMP on/off 비교: CUDA에서만 비교

        작은 corpus를 그대로 둔 채 모델만 키우면 과적합하기 쉬우므로 train/validation gap도 보세요.
        """
    ),
    code(
        """
        # 최종 자동 점검
        assert scheduler.last_epoch == total_update_steps
        assert payload["model_config"]["num_heads"] == 4
        assert payload["tokenizer_itos"] == tokenizer.itos
        assert math.isfinite(reloaded_metrics["perplexity"])
        assert len(greedy_text) >= len(prompt)
        print("all training/generation checks passed")
        """,
        """
        assert scheduler.last_epoch == total_update_steps
        assert payload["model_config"]["num_heads"] == 4
        assert payload["tokenizer_itos"] == tokenizer.itos
        assert math.isfinite(reloaded_metrics["perplexity"])
        assert len(greedy_text) >= len(prompt)
        print("all training/generation checks passed")
        """,
    ),
    code(
        """
        # 임시 checkpoint 정리. 영구 보관하려면 실행 전에 artifacts 경로로 바꾸세요.
        checkpoint_path.unlink(missing_ok=True)
        print("temporary checkpoint removed:", not checkpoint_path.exists())
        """,
        """
        checkpoint_path.unlink(missing_ok=True)
        print("temporary checkpoint removed:", not checkpoint_path.exists())
        """,
    ),
    markdown(
        """
        ## 마무리

        이 랩에서 만든 것은 작은 모델이지만 현업 학습 파이프라인의 핵심 계약을 모두 포함합니다.
        데이터/label shift, update 기준 scheduler, AMP, accumulation, clipping,
        token-weighted validation,
        재시작 가능한 checkpoint, 추론 전략을 각각 독립적으로 설명할 수 있는지 확인하세요.

        다음 확장 과제는 (1) tokenizer를 BPE로 교체, (2) KV cache 사용/미사용 생성 시간 비교,
        (3) validation 기반 early stopping, (4) 여러 seed의 평균/표준편차 기록입니다.
        """
    ),
]


LAB21: list[tuple[str, ...]] = [
    markdown(
        """
        # 21. GPU 학습·최적화 실습 — RTX 3080에서도 안전하게

        이 페어 랩은 GPU를 단순히 "사용"하는 수준을 넘어, 데이터 이동·AMP·gradient accumulation,
        clipping, 메모리 관찰, checkpoint/resume, 재현성까지 하나의 작은 학습 시스템으로 묶습니다.
        JupyterLab 왼쪽에는 exercise, 오른쪽에는 solution을 열고 TODO를 한 셀씩 완성하세요.
        """
    ),
    markdown(
        """
        ## 안전 원칙과 완료 기준

        기본 실험은 CPU에서도 빠르게 끝나는 synthetic classification입니다. CUDA가 있으면 pinned
        memory와 AMP를 자동 사용합니다. Batch-size 탐색은 **분석적 메모리 추정만 수행**해 실제 OOM을
        유발하지 않습니다. RTX 3080 16GB에서도 전체 VRAM을 목표로 하지 않고 다른 앱과 CUDA context를
        위해 여유를 둡니다.

        완료 기준: device 확인 → 재현 가능한 loader → non-blocking 이동 → AMP 학습 → peak memory →
        안전 batch 후보 → checkpoint 재개 → 조건부 `torch.compile` → 평가를 모두 통과하는 것입니다.
        """
    ),
    code(
        """
        # 준비 셀: 그대로 실행합니다.
        import math
        import os
        import random
        import tempfile
        import time
        from contextlib import nullcontext
        from pathlib import Path
        
        import torch
        from torch import Tensor, nn
        from torch.utils.data import DataLoader, Dataset
        from llm_engineering_lab.acceleration import get_accelerator
        
        SEED = 21
        ACCELERATOR = get_accelerator()
        device = ACCELERATOR.device
        print(f"torch={torch.__version__}, {ACCELERATOR.summary()}")
        """,
        """
        import math
        import os
        import random
        import tempfile
        import time
        from contextlib import nullcontext
        from pathlib import Path
        
        import torch
        from torch import Tensor, nn
        from torch.utils.data import DataLoader, Dataset
        from llm_engineering_lab.acceleration import get_accelerator
        
        SEED = 21
        ACCELERATOR = get_accelerator()
        device = ACCELERATOR.device
        print(f"torch={torch.__version__}, {ACCELERATOR.summary()}")
        """,
    ),
    markdown(
        """
        ## 1) Hardware inventory와 메모리 예산

        코드에서 device를 문자열로 가정하지 말고 실제 가용성을 확인합니다. CUDA일 때 GPU 이름,
        compute capability, 총 VRAM을 기록하세요. 총 VRAM은 곧 사용 가능 VRAM이 아닙니다. Driver,
        display, 다른 프로세스, CUDA context가 함께 사용하므로 이 랩은 총량의 65%만 탐색 예산으로 둡니다.
        """
    ),
    code(
        """
        # TODO 21-1: CUDA면 get_device_properties로 이름/VRAM/capability를 읽으세요.
        if device.type == "cuda":
            properties = ...
            hardware = ...
        else:
            hardware = ...
        print(hardware)
        """,
        """
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            hardware = {
                "name": properties.name,
                "total_vram_gb": properties.total_memory / 1024**3,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "safe_budget_gb": properties.total_memory * 0.65 / 1024**3,
            }
        else:
            hardware = {
                "name": "CPU fallback",
                "total_vram_gb": 0.0,
                "compute_capability": None,
                "safe_budget_gb": 0.5,
            }
        print(hardware)
        """,
    ),
    markdown(
        """
        ## 2) 재현성 설정

        Python과 PyTorch RNG를 함께 고정하고, CUDA라면 모든 GPU seed를 설정합니다. Deterministic
        algorithm은 느려지거나 일부 연산에서 경고가 날 수 있어 `warn_only=True`로 학습 실습을 막지
        않습니다. 완전한 재현에는 데이터·코드·라이브러리·driver 버전까지 함께 기록해야 합니다.
        """
    ),
    code(
        """
        def seed_everything(seed: int) -> None:
            # TODO 21-2: random, torch, CUDA seed와 deterministic 옵션을 설정하세요.
            raise NotImplementedError("TODO 21-2")
        
        
        seed_everything(SEED)
        """,
        """
        def seed_everything(seed: int) -> None:
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = False
        
        
        seed_everything(SEED)
        print("seeded:", SEED)
        """,
    ),
    markdown(
        """
        ## 3) 학습 가능한 synthetic dataset

        입력 feature에 고정된 teacher matrix를 곱해 class를 만듭니다. Train/validation이 같은 teacher를
        공유하므로 일반화를 측정할 수 있고, 파일 I/O 없이 GPU 파이프라인만 빠르게 반복합니다. Dataset은
        CPU tensor를 반환합니다. Worker가 GPU tensor를 만드는 패턴은 CUDA context 충돌의 원인이 됩니다.
        """
    ),
    code(
        """
        class SyntheticClassificationDataset(Dataset):
            def __init__(
                self,
                size: int,
                input_dim: int,
                teacher: Tensor,
                seed: int,
            ) -> None:
                # TODO 21-3: CPU feature와 teacher 기반 label을 미리 생성하세요.
                raise NotImplementedError("TODO 21-3")
        
            def __len__(self) -> int:
                raise NotImplementedError("TODO 21-3")
        
            def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
                raise NotImplementedError("TODO 21-3")
        """,
        """
        class SyntheticClassificationDataset(Dataset):
            def __init__(
                self,
                size: int,
                input_dim: int,
                teacher: Tensor,
                seed: int,
            ) -> None:
                generator = torch.Generator().manual_seed(seed)
                self.features = torch.randn(size, input_dim, generator=generator)
                noise = 0.15 * torch.randn(size, teacher.shape[1], generator=generator)
                self.labels = (self.features @ teacher + noise).argmax(dim=1)
        
            def __len__(self) -> int:
                return len(self.features)
        
            def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
                return self.features[index], self.labels[index]
        """,
    ),
    markdown(
        """
        ## 4) Windows/Jupyter에 안전한 DataLoader

        CUDA일 때 `pin_memory=True`인 CPU batch를 `non_blocking=True`로 복사하면 compute와 transfer를
        겹칠 가능성이 생깁니다. Windows Notebook에서는 `num_workers>0`가 셀 코드를 다시 import하거나
        worker 시작 문제를 만들 수 있어 기본은 0입니다. 별도 `.py` 진입점에서만 worker 수를 조정하세요.
        """
    ),
    code(
        """
        input_dim, hidden_dim, num_classes = 128, 256, 8
        teacher_generator = torch.Generator().manual_seed(999)
        teacher = torch.randn(input_dim, num_classes, generator=teacher_generator)
        train_dataset = SyntheticClassificationDataset(2048, input_dim, teacher, seed=101)
        validation_dataset = SyntheticClassificationDataset(512, input_dim, teacher, seed=202)
        
        batch_size = 128
        loader_generator = torch.Generator().manual_seed(SEED)
        # TODO 21-4: CUDA 여부에 맞춰 pin_memory를 정하고 두 DataLoader를 만드세요.
        pin_memory = ...
        train_loader = ...
        validation_loader = ...
        """,
        """
        input_dim, hidden_dim, num_classes = 128, 256, 8
        teacher_generator = torch.Generator().manual_seed(999)
        teacher = torch.randn(input_dim, num_classes, generator=teacher_generator)
        train_dataset = SyntheticClassificationDataset(2048, input_dim, teacher, seed=101)
        validation_dataset = SyntheticClassificationDataset(512, input_dim, teacher, seed=202)
        
        batch_size = 128
        loader_generator = torch.Generator().manual_seed(SEED)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            pin_memory=pin_memory,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        print("batches:", len(train_loader), "pin_memory:", pin_memory)
        """,
    ),
    markdown(
        """
        ## 5) Host-to-device 경계 한곳에 모으기

        Batch 이동을 helper 하나로 모으면 model 코드에 `.cuda()`가 흩어지지 않고 CPU fallback도
        유지됩니다. `non_blocking=True`는 source가 pinned memory일 때 비동기 전송 가능성을 열어 주며,
        즉시 동기화된다는 보장은 아닙니다.
        """
    ),
    code(
        """
        def move_batch(batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            # TODO 21-5: feature/label을 device로 옮기세요. CUDA일 때 non_blocking을 사용하세요.
            raise NotImplementedError("TODO 21-5")
        
        
        sample_features, sample_labels = move_batch(next(iter(train_loader)))
        assert sample_features.device.type == sample_labels.device.type == device.type
        """,
        """
        def move_batch(batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            non_blocking = device.type == "cuda"
            return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)
        
        
        sample_features, sample_labels = move_batch(next(iter(train_loader)))
        assert sample_features.device.type == sample_labels.device.type == device.type
        print(tuple(sample_features.shape), sample_features.device)
        """,
    ),
    markdown(
        """
        ## 6) 작은 모델과 sanity check

        먼저 한 batch에서 forward/loss/backward shape를 확인합니다. GPU 최적화 전에 잘못된 label dtype,
        output class 수, device mismatch를 잡는 가장 싼 테스트입니다. 이 모델은 의도적으로 작아 노트북을
        막지 않으며, 동일한 학습 loop를 CNN/Transformer로 교체할 수 있습니다.
        """
    ),
    code(
        """
        class TinyClassifier(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
                super().__init__()
                # TODO 21-6: Linear-GELU-LayerNorm-Dropout-Linear를 구성하세요.
                self.network = ...
        
            def forward(self, features: Tensor) -> Tensor:
                return self.network(features)
        
        
        model = TinyClassifier(input_dim, hidden_dim, num_classes).to(device)
        parameter_count = ...
        """,
        """
        class TinyClassifier(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes),
                )
        
            def forward(self, features: Tensor) -> Tensor:
                return self.network(features)
        
        
        model = TinyClassifier(input_dim, hidden_dim, num_classes).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"parameters={parameter_count:,}")
        """,
    ),
    code(
        """
        sanity_logits = model(sample_features)
        sanity_loss = nn.functional.cross_entropy(sanity_logits, sample_labels)
        sanity_loss.backward()
        assert sanity_logits.shape == (len(sample_labels), num_classes)
        assert torch.isfinite(sanity_loss)
        model.zero_grad(set_to_none=True)
        print("sanity loss:", round(sanity_loss.item(), 4))
        """,
        """
        sanity_logits = model(sample_features)
        sanity_loss = nn.functional.cross_entropy(sanity_logits, sample_labels)
        sanity_loss.backward()
        assert sanity_logits.shape == (len(sample_labels), num_classes)
        assert torch.isfinite(sanity_loss)
        model.zero_grad(set_to_none=True)
        print("sanity loss:", round(sanity_loss.item(), 4))
        """,
    ),
    markdown(
        """
        ## 7) GPU memory 측정

        `allocated`는 tensor가 실제 사용 중인 메모리, `reserved`는 allocator가 재사용하려고 확보한
        메모리, `peak`는 reset 이후 최대 allocated입니다. `empty_cache()`는 살아 있는 tensor를 지우지
        않으며 매 step 호출하면 오히려 느려질 수 있습니다. 측정 경계에서만 synchronize합니다.
        """
    ),
    code(
        """
        def reset_peak_memory() -> None:
            # TODO 21-7: CUDA에서 synchronize 후 peak 통계를 reset하세요.
            raise NotImplementedError("TODO 21-7")
        
        
        def memory_snapshot() -> dict[str, float]:
            # TODO: CUDA allocated/reserved/peak MB를 반환하고 CPU면 0을 반환하세요.
            raise NotImplementedError("TODO 21-7")
        """,
        """
        def reset_peak_memory() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
        
        
        def memory_snapshot() -> dict[str, float]:
            if device.type != "cuda":
                return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
            torch.cuda.synchronize(device)
            scale = 1024**2
            return {
                "allocated_mb": torch.cuda.memory_allocated(device) / scale,
                "reserved_mb": torch.cuda.memory_reserved(device) / scale,
                "peak_mb": torch.cuda.max_memory_allocated(device) / scale,
            }
        
        
        print("before training:", memory_snapshot())
        """,
    ),
    markdown(
        """
        ## 8) AMP autocast + GradScaler

        Autocast 영역에는 forward와 loss만 넣습니다. Backward는 autocast 밖에서 scaler를 통해 실행합니다.
        Clipping 전에 반드시 `scaler.unscale_(optimizer)`를 호출해야 실제 gradient norm을 봅니다.
        CPU fallback에서는 `nullcontext`와 disabled scaler로 같은 코드를 유지합니다.
        """
    ),
    code(
        """
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        
        
        def amp_context():
            # TODO 21-8: CUDA면 float16 autocast, CPU면 nullcontext를 반환하세요.
            raise NotImplementedError("TODO 21-8")
        """,
        """
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        
        
        def amp_context():
            if amp_enabled:
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            return nullcontext()
        
        
        print("AMP enabled:", amp_enabled)
        """,
    ),
    markdown(
        """
        ## 9) Accumulation + clipping + throughput 학습 함수

        Effective batch는 `micro batch × accumulation_steps`입니다. Loss를 accumulation 수로 나누고,
        optimizer/scheduler/zero_grad는 update 시점에만 호출합니다. 마지막 남은 micro-batch도 update하며
        clipping 전 norm과 examples/sec를 함께 기록해 안정성과 성능을 비교합니다.
        """
    ),
    code(
        """
        def train_one_epoch(
            model: nn.Module,
            loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scaler,
            accumulation_steps: int = 2,
            clip_norm: float = 1.0,
        ) -> dict[str, float]:
            # TODO 21-9: AMP/accumulation/unscale/clipping/update와 지표 집계를 구현하세요.
            raise NotImplementedError("TODO 21-9")
        """,
        """
        def train_one_epoch(
            model: nn.Module,
            loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scaler,
            accumulation_steps: int = 2,
            clip_norm: float = 1.0,
        ) -> dict[str, float]:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            reset_peak_memory()
            start = time.perf_counter()
            total_loss = 0.0
            correct = 0
            examples = 0
            updates = 0
            last_grad_norm = 0.0
        
            for micro_step, batch in enumerate(loader):
                features, labels = move_batch(batch)
                with amp_context():
                    logits = model(features)
                    raw_loss = nn.functional.cross_entropy(logits, labels)
                    loss = raw_loss / accumulation_steps
                scaler.scale(loss).backward()
                total_loss += raw_loss.detach().item() * len(labels)
                correct += logits.detach().argmax(dim=1).eq(labels).sum().item()
                examples += len(labels)
        
                update_now = (micro_step + 1) % accumulation_steps == 0
                update_now |= micro_step + 1 == len(loader)
                if update_now:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    last_grad_norm = float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
        
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            return {
                "loss": total_loss / examples,
                "accuracy": correct / examples,
                "examples_per_second": examples / max(elapsed, 1e-9),
                "updates": float(updates),
                "last_grad_norm": last_grad_norm,
                **memory_snapshot(),
            }
        """,
    ),
    code(
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        first_epoch_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, accumulation_steps=2, clip_norm=1.0
        )
        assert first_epoch_metrics["updates"] == math.ceil(len(train_loader) / 2)
        print({key: round(value, 3) for key, value in first_epoch_metrics.items()})
        """,
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        first_epoch_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, accumulation_steps=2, clip_norm=1.0
        )
        assert first_epoch_metrics["updates"] == math.ceil(len(train_loader) / 2)
        print({key: round(value, 3) for key, value in first_epoch_metrics.items()})
        """,
    ),
    markdown(
        """
        ## 10) OOM 없는 batch-size 탐색

        후보 batch를 실제로 할당하며 OOM을 잡는 코드는 allocator fragmentation과 Notebook kernel 불안정을
        만들 수 있습니다. 먼저 parameter/optimizer/gradient와 activation을 보수적으로 추정해 후보를
        거릅니다. 이 값은 **상한 보장치가 아닌 1차 필터**이며, 실제 모델에서는 profiler와 작은 증가
        실험으로 확인하세요. Transformer activation은 sequence length 제곱 항도 포함해야 합니다.
        """
    ),
    code(
        """
        bytes_per_parameter = next(model.parameters()).element_size()
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        
        
        def estimate_training_bytes(candidate_batch: int) -> int:
            # TODO 21-10: weights+grads+Adam state와 보수적 activation 메모리를 추정하세요.
            raise NotImplementedError("TODO 21-10")
        
        
        candidate_batches = [64, 128, 256, 512, 1024, 2048]
        if device.type == "cuda":
            budget_bytes = int(torch.cuda.get_device_properties(device).total_memory * 0.65)
        else:
            budget_bytes = 512 * 1024**2
        safe_candidates = ...
        """,
        """
        bytes_per_parameter = next(model.parameters()).element_size()
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        
        
        def estimate_training_bytes(candidate_batch: int) -> int:
            model_gradient_optimizer = parameter_bytes * 4  # weight + grad + Adam m/v
            # 여러 layer의 저장 activation과 framework overhead를 보수적으로 확대합니다.
            activation_values = candidate_batch * (input_dim + hidden_dim * 6 + num_classes)
            activation_bytes = activation_values * bytes_per_parameter * 8
            fixed_overhead = 128 * 1024**2
            return model_gradient_optimizer + activation_bytes + fixed_overhead
        
        
        candidate_batches = [64, 128, 256, 512, 1024, 2048]
        if device.type == "cuda":
            budget_bytes = int(torch.cuda.get_device_properties(device).total_memory * 0.65)
        else:
            budget_bytes = 512 * 1024**2
        safe_candidates = [
            batch
            for batch in candidate_batches
            if estimate_training_bytes(batch) < budget_bytes
        ]
        estimate_table = {
            batch: round(estimate_training_bytes(batch) / 1024**2, 1)
            for batch in candidate_batches
        }
        assert safe_candidates
        print("estimated MB:", estimate_table)
        print("analysis-only safe candidates:", safe_candidates, "(실제 할당하지 않음)")
        """,
    ),
    markdown(
        """
        ## 11) Checkpoint 저장과 resume

        Model만 저장하면 정확한 재개가 아닙니다. Optimizer, GradScaler, epoch/global step, RNG, DataLoader
        generator 상태를 함께 저장합니다. 새 객체에 로드한 뒤 실제로 다음 epoch를 진행하여 resume 경로를
        검증합니다. 실무에서는 scheduler, tokenizer/config, git commit, metric도 포함하세요.
        """
    ),
    code(
        """
        checkpoint_path = Path(tempfile.gettempdir()) / "ai_lab21_gpu_resume.pt"
        # TODO 21-11: model/optimizer/scaler/RNG/loader generator 상태를 저장하세요.
        checkpoint_payload = ...
        torch.save(checkpoint_payload, checkpoint_path)
        assert checkpoint_path.exists()
        """,
        """
        checkpoint_path = Path(tempfile.gettempdir()) / "ai_lab21_gpu_resume.pt"
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": 1,
            "global_updates": int(first_epoch_metrics["updates"]),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "loader_generator_state": loader_generator.get_state(),
            "metrics": first_epoch_metrics,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        assert checkpoint_path.exists() and checkpoint_path.stat().st_size > 0
        print("checkpoint bytes:", checkpoint_path.stat().st_size)
        """,
    ),
    code(
        """
        # TODO 21-12: 새 model/optimizer/scaler를 만들고 모든 상태를 복구한 뒤 한 epoch 재개하세요.
        payload = ...
        resumed_model = ...
        resumed_optimizer = ...
        resumed_scaler = ...
        second_epoch_metrics = ...
        """,
        """
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        resumed_model = TinyClassifier(input_dim, hidden_dim, num_classes).to(device)
        resumed_model.load_state_dict(payload["model_state"])
        resumed_optimizer = torch.optim.AdamW(
            resumed_model.parameters(), lr=2e-3, weight_decay=0.01
        )
        resumed_optimizer.load_state_dict(payload["optimizer_state"])
        resumed_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        resumed_scaler.load_state_dict(payload["scaler_state"])
        torch.set_rng_state(payload["rng_cpu"].cpu())
        if device.type == "cuda" and payload["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng_cuda"]])
        loader_generator.set_state(payload["loader_generator_state"].cpu())
        
        second_epoch_metrics = train_one_epoch(
            resumed_model,
            train_loader,
            resumed_optimizer,
            resumed_scaler,
            accumulation_steps=2,
            clip_norm=1.0,
        )
        total_updates = payload["global_updates"] + int(second_epoch_metrics["updates"])
        assert total_updates == 2 * math.ceil(len(train_loader) / 2)
        print(
            "resumed epoch metrics:", {k: round(v, 3) for k, v in second_epoch_metrics.items()}
        )
        """,
    ),
    markdown(
        """
        ## 12) `torch.compile`은 명시적 opt-in

        Compile은 첫 호출 비용, graph break, backend/driver 조합 이슈가 있어 작은 Notebook에서는 오히려
        느릴 수 있습니다. 기본은 꺼 두고 CUDA, API 존재, 사용자의 명시적 선택을 모두 만족할 때만
        실행합니다. 먼저 eager correctness를 통과하고 고정 shape의 반복 workload에서 warmup 이후를
        benchmark하세요. 문제가 나면 원본 model로 즉시 fallback합니다.
        """
    ),
    code(
        """
        RUN_TORCH_COMPILE = False  # GPU에서 원할 때만 True
        inference_model = resumed_model
        compile_status = "eager"
        
        # TODO 21-13: 안전 조건을 확인하고 try/except로 torch.compile을 적용하세요.
        """,
        """
        RUN_TORCH_COMPILE = False
        inference_model = resumed_model
        compile_status = "eager"
        
        if RUN_TORCH_COMPILE and device.type == "cuda" and hasattr(torch, "compile"):
            try:
                inference_model = torch.compile(resumed_model, mode="reduce-overhead")
                with torch.inference_mode():
                    _ = inference_model(sample_features)  # compile warmup
                compile_status = "compiled"
            except Exception as error:
                inference_model = resumed_model
                compile_status = f"fallback: {type(error).__name__}"
        print("compile status:", compile_status)
        """,
    ),
    code(
        """
        @torch.inference_mode()
        def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
            model.eval()
            start = time.perf_counter()
            correct = 0
            examples = 0
            for batch in loader:
                features, labels = move_batch(batch)
                logits = model(features)
                correct += logits.argmax(dim=1).eq(labels).sum().item()
                examples += len(labels)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            return {"accuracy": correct / examples, "examples_per_second": examples / elapsed}
        
        
        final_metrics = evaluate(inference_model, validation_loader)
        assert 0.0 <= final_metrics["accuracy"] <= 1.0
        assert math.isfinite(final_metrics["examples_per_second"])
        assert second_epoch_metrics["loss"] <= first_epoch_metrics["loss"] + 0.5
        checkpoint_path.unlink(missing_ok=True)
        print("validation:", {k: round(v, 3) for k, v in final_metrics.items()})
        print("checkpoint cleaned:", not checkpoint_path.exists())
        """,
        """
        @torch.inference_mode()
        def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
            model.eval()
            start = time.perf_counter()
            correct = 0
            examples = 0
            for batch in loader:
                features, labels = move_batch(batch)
                logits = model(features)
                correct += logits.argmax(dim=1).eq(labels).sum().item()
                examples += len(labels)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            return {"accuracy": correct / examples, "examples_per_second": examples / elapsed}
        
        
        final_metrics = evaluate(inference_model, validation_loader)
        assert 0.0 <= final_metrics["accuracy"] <= 1.0
        assert math.isfinite(final_metrics["examples_per_second"])
        assert second_epoch_metrics["loss"] <= first_epoch_metrics["loss"] + 0.5
        checkpoint_path.unlink(missing_ok=True)
        print("validation:", {k: round(v, 3) for k, v in final_metrics.items()})
        print("checkpoint cleaned:", not checkpoint_path.exists())
        """,
    ),
    markdown(
        """
        ## 마무리 — RTX 3080 실험 체크리스트

        - DataLoader worker보다 먼저 `num_workers=0 + pin_memory` 기준선을 측정합니다.
        - AMP, batch size, accumulation을 한 번에 바꾸지 말고 throughput/peak VRAM/metric을 함께 기록합니다.
        - `reserved > allocated`는 정상일 수 있으며 반복적인 `empty_cache()`를 최적화로 오해하지 않습니다.
        - Batch-size 추정표는 1차 필터이고 sequence model은 길이 제곱 activation을 별도 반영합니다.
        - Resume 테스트는 새 객체에서 수행하고 RNG/loader/scheduler/tokenizer까지 저장합니다.
        - `torch.compile`은 eager 정답 검증 뒤 반복 workload에서만 opt-in benchmark합니다.

        이 학습 함수를 06번 classifier, 08번 RNN, 16번 Transformer에 적용하며 모델별 peak 메모리와
        examples(or tokens)/sec를 비교하면 노트북 사양에 맞는 실전 기준선을 만들 수 있습니다.
        """
    ),
]


def validate_pair(filename: str, expected_cells: int) -> None:
    exercise = nbformat.read(EXERCISES / filename, as_version=4)
    solution = nbformat.read(SOLUTIONS / filename, as_version=4)
    nbformat.validate(exercise)
    nbformat.validate(solution)
    assert len(exercise.cells) == len(solution.cells) == expected_cells
    assert [cell.cell_type for cell in exercise.cells] == [
        cell.cell_type for cell in solution.cells
    ]
    assert [cell.id for cell in exercise.cells] == [cell.id for cell in solution.cells]
    for left, right in zip(exercise.cells, solution.cells, strict=True):
        if left.cell_type == "markdown":
            assert left.source == right.source
    exercise_code = "\n".join(
        cell.source for cell in exercise.cells if cell.cell_type == "code"
    )
    assert "TODO" in exercise_code and "NotImplementedError" in exercise_code


def execute_solution(filename: str) -> None:
    """Execute solution code cells in one shared namespace, like a fresh kernel."""

    path = SOLUTIONS / filename
    notebook = nbformat.read(path, as_version=4)
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path}#cell-{index:02d}", "exec"), namespace)
    print(f"executed {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="execute all solution notebooks"
    )
    args = parser.parse_args()
    build_pair(
        "15_transformer_architecture_lab.ipynb",
        "Transformer 구조 해부 실습",
        LAB15,
    )
    build_pair(
        "16_transformer_training_lab.ipynb",
        "Tiny Transformer 학습 실습",
        LAB16,
    )
    build_pair(
        "21_gpu_training_and_optimization.ipynb",
        "GPU 학습·최적화 실습",
        LAB21,
    )
    validate_pair("15_transformer_architecture_lab.ipynb", len(LAB15))
    validate_pair("16_transformer_training_lab.ipynb", len(LAB16))
    validate_pair("21_gpu_training_and_optimization.ipynb", len(LAB21))
    print(
        f"built lab 15 ({len(LAB15)} cells), lab 16 ({len(LAB16)} cells), "
        f"and lab 21 ({len(LAB21)} cells)"
    )
    if args.execute:
        execute_solution("15_transformer_architecture_lab.ipynb")
        execute_solution("16_transformer_training_lab.ipynb")
        execute_solution("21_gpu_training_and_optimization.ipynb")
