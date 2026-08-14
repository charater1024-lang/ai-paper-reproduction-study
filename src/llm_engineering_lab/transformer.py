"""A compact decoder-only Transformer for learning LLM mechanics on CPU.

This is intentionally not a replacement for Hugging Face Transformers.  The
attention, causal mask, residual blocks, loss shift, sampling, and KV cache are
kept visible so learners can set breakpoints and inspect every tensor shape.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .torch_text import Vocabulary, resolve_device


class KeyValueCache(NamedTuple):
    """Cached attention keys/values, both shaped ``[B, heads, past, head_dim]``."""

    key: Tensor
    value: Tensor


@dataclass(slots=True)
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[KeyValueCache, ...] | None = None


@dataclass(frozen=True, slots=True)
class TinyTransformerConfig:
    vocab_size: int
    pad_id: int
    max_seq_length: int = 96
    model_dim: int = 96
    num_heads: int = 4
    num_layers: int = 2
    feed_forward_dim: int = 256
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must be a valid token id")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.num_layers < 1 or self.feed_forward_dim < 1:
            raise ValueError("num_layers/feed_forward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask and KV cache."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.query_key_value = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.output = nn.Linear(config.model_dim, config.model_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KeyValueCache | None]:
        batch_size, query_length, model_dim = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, query_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            expected_prefix = (batch_size, self.num_heads)
            if (
                past_key.shape[:2] != expected_prefix
                or past_value.shape != past_key.shape
            ):
                raise ValueError("Malformed past_key_value cache")
            past_length = past_key.shape[2]
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        key_length = key.shape[2]

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # Query i at absolute position past_length+i may see keys <= that position.
        query_positions = past_length + torch.arange(
            query_length, device=hidden_states.device
        )
        key_positions = torch.arange(key_length, device=hidden_states.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        allowed = causal_mask.view(1, 1, query_length, key_length)

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (
                batch_size,
                key_length,
            ):
                raise ValueError(
                    "attention_mask must cover all cached/current keys and have shape "
                    f"{(batch_size, key_length)}, got {tuple(attention_mask.shape)}"
                )
            allowed = allowed & attention_mask.to(torch.bool).view(
                batch_size, 1, 1, key_length
            )

        # finfo.min is safer than -inf for some reduced-precision kernels.
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        probabilities = self.attention_dropout(probabilities)
        context = probabilities @ value
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_length, model_dim)
        )
        hidden_states = self.residual_dropout(self.output(context))
        present = KeyValueCache(key, value) if use_cache else None
        return hidden_states, present


class TransformerBlock(nn.Module):
    """Pre-layer-normalization decoder block."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.model_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.model_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KeyValueCache | None]:
        attention_output, present = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states, present


class TinyCausalTransformer(nn.Module):
    """Small GPT-like model with teacher-forced loss and cached generation."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.model_dim,
            padding_idx=config.pad_id,
        )
        self.position_embedding = nn.Embedding(config.max_seq_length, config.model_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_id].zero_()

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: Sequence[KeyValueCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, time], got {input_ids.shape}"
            )
        batch_size, query_length = input_ids.shape
        if query_length < 1:
            raise ValueError("input_ids cannot have an empty time dimension")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError(
                "past_key_values must have one entry per Transformer block"
            )
        past_length = 0 if past_key_values is None else past_key_values[0].key.shape[2]
        total_length = past_length + query_length
        if total_length > self.config.max_seq_length:
            raise ValueError(
                f"Sequence length {total_length} exceeds "
                f"max_seq_length={self.config.max_seq_length}"
            )
        if attention_mask is None:
            if past_length:
                attention_mask = torch.ones(
                    (batch_size, total_length),
                    dtype=torch.bool,
                    device=input_ids.device,
                )
            else:
                attention_mask = input_ids.ne(self.config.pad_id)
        else:
            attention_mask = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        if attention_mask.shape != (batch_size, total_length):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, total_length)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        # cumsum position ids make left-padded batched inference agree with
        # running each prompt separately.
        positions = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        positions = positions[:, -query_length:]
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(
            positions
        )
        hidden_states = self.embedding_dropout(hidden_states)

        presents: list[KeyValueCache] = []
        for layer_index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[layer_index]
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if present is not None:
                presents.append(present)

        logits = self.lm_head(self.final_norm(hidden_states))
        assert logits.shape == (batch_size, query_length, self.config.vocab_size)
        loss: Tensor | None = None
        if labels is not None:
            if past_length:
                raise ValueError("labels are only supported without past_key_values")
            if labels.shape != input_ids.shape:
                raise ValueError("labels and input_ids must have identical shapes")
            if query_length < 2:
                raise ValueError(
                    "At least two tokens are required to compute causal LM loss"
                )
            # Token t predicts token t+1; padding and explicit -100 labels are ignored.
            shifted_labels = labels[:, 1:].contiguous()
            shifted_labels = shifted_labels.masked_fill(
                shifted_labels.eq(self.config.pad_id), -100
            )
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(presents) if use_cache else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 24,
        temperature: float = 0.8,
        top_k: int | None = 20,
        top_p: float = 0.95,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively extend one or more (left-padded) prompts."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, non-empty time]")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_length:
            raise ValueError("prompt + max_new_tokens exceeds model context length")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")
        self.eval()
        generated = input_ids
        if attention_mask is None:
            attention_mask = generated.ne(self.config.pad_id)
        else:
            attention_mask = attention_mask.to(
                device=generated.device, dtype=torch.bool
            )
        if attention_mask.shape != generated.shape:
            raise ValueError("attention_mask must match input_ids before generation")
        finished = torch.zeros(
            generated.shape[0], dtype=torch.bool, device=generated.device
        )
        past: tuple[KeyValueCache, ...] | None = None
        current_input = generated

        for _ in range(max_new_tokens):
            output = self(
                current_input,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=use_cache,
            )
            next_logits = output.logits[:, -1, :]
            if repetition_penalty > 1.0:
                # Repeated indices are harmless here; each occurrence writes
                # the same penalized value back to the corresponding row.
                previous_ids = generated
                previous_logits = next_logits.gather(1, previous_ids)
                penalized = torch.where(
                    previous_logits < 0,
                    previous_logits * repetition_penalty,
                    previous_logits / repetition_penalty,
                )
                next_logits.scatter_(1, previous_ids, penalized)
            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            if eos_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                finished |= next_token.eq(eos_id)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (generated.shape[0], 1),
                        dtype=torch.bool,
                        device=generated.device,
                    ),
                ),
                dim=1,
            )
            if eos_id is not None and bool(finished.all()):
                break
            if use_cache:
                past = output.past_key_values
                current_input = next_token[:, None]
            else:
                current_input = generated
        return generated


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply greedy/temperature, top-k, and nucleus sampling to ``[B, vocab]`` logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0:
        return logits.argmax(dim=-1)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    filtered = logits.float() / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive or None")
        top_k = min(top_k, filtered.shape[-1])
        threshold = filtered.topk(top_k, dim=-1).values[:, -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = filtered.sort(descending=True, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        # Always retain the most likely token, including for very small top_p.
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_indices, sorted_logits)
    probabilities = filtered.softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(
        1
    )


def build_lm_token_stream(texts: Iterable[str], vocabulary: Vocabulary) -> list[int]:
    """Pack documents with BOS/EOS boundary markers into one training stream."""

    stream: list[int] = []
    for text in texts:
        stream.extend(vocabulary.encode(text, add_bos=True, add_eos=True))
    if len(stream) < 2:
        raise ValueError("At least two tokens are required to train a language model")
    return stream


class LanguageModelDataset(Dataset[dict[str, Tensor]]):
    """Fixed-size windows over a packed token stream; the final window is padded."""

    def __init__(
        self,
        token_ids: Sequence[int],
        *,
        block_size: int,
        pad_id: int,
        stride: int | None = None,
    ) -> None:
        if len(token_ids) < 2:
            raise ValueError("token_ids must contain at least two tokens")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.token_ids = list(token_ids)
        self.block_size = block_size
        self.pad_id = pad_id
        self.stride = stride or block_size
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.starts = list(range(0, len(self.token_ids) - 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = self.starts[index]
        ids = self.token_ids[start : start + self.block_size]
        ids += [self.pad_id] * (self.block_size - len(ids))
        input_ids = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_id),
            "labels": input_ids.clone(),
        }


@dataclass(frozen=True, slots=True)
class LanguageModelMetrics:
    loss: float
    perplexity: float
    predicted_tokens: int


def _safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def train_tiny_lm_epoch(
    model: TinyCausalTransformer,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
) -> LanguageModelMetrics:
    model.train()
    weighted_loss = 0.0
    predicted_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        assert output.loss is not None
        output.loss.backward()
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        token_count = labels[:, 1:].ne(model.config.pad_id).sum().item()
        weighted_loss += output.loss.item() * token_count
        predicted_tokens += token_count
    if predicted_tokens == 0:
        raise ValueError("Loader has no non-padding target tokens")
    mean_loss = weighted_loss / predicted_tokens
    return LanguageModelMetrics(
        mean_loss, _safe_perplexity(mean_loss), predicted_tokens
    )


@torch.inference_mode()
def evaluate_tiny_lm(
    model: TinyCausalTransformer,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
) -> LanguageModelMetrics:
    model.eval()
    weighted_loss = 0.0
    predicted_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        assert output.loss is not None
        token_count = labels[:, 1:].ne(model.config.pad_id).sum().item()
        weighted_loss += output.loss.item() * token_count
        predicted_tokens += token_count
    if predicted_tokens == 0:
        raise ValueError("Loader has no non-padding target tokens")
    mean_loss = weighted_loss / predicted_tokens
    return LanguageModelMetrics(
        mean_loss, _safe_perplexity(mean_loss), predicted_tokens
    )


@dataclass(frozen=True, slots=True)
class LMTrainingConfig:
    epochs: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0


def fit_tiny_lm(
    model: TinyCausalTransformer,
    train_loader: DataLoader[dict[str, Tensor]],
    *,
    config: LMTrainingConfig | None = None,
    device: torch.device | None = None,
    validation_loader: DataLoader[dict[str, Tensor]] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train a tiny LM and checkpoint the lowest validation (or train) loss."""

    config = config or LMTrainingConfig()
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    selected_device = device or resolve_device()
    model.to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_tiny_lm_epoch(
            model,
            train_loader,
            optimizer,
            device=selected_device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_metrics = (
            evaluate_tiny_lm(model, validation_loader, device=selected_device)
            if validation_loader is not None
            else None
        )
        monitored_loss = (
            validation_metrics.loss if validation_metrics else train_metrics.loss
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics) if validation_metrics else None,
        }
        history.append(record)
        if checkpoint_path is not None and monitored_loss < best_loss:
            best_loss = monitored_loss
            destination = Path(checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model.config),
                    "training_config": asdict(config),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": record,
                    "extra": dict(checkpoint_extra or {}),
                },
                destination,
            )
    return history


def load_tiny_lm_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TinyCausalTransformer, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    model = TinyCausalTransformer(TinyTransformerConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model, payload


# TODO(연습): attention probability를 반환해 token-to-token heatmap을 그려 보세요.
# TODO(심화): learned positional embedding을 RoPE로 바꾸고 긴 문맥에서 비교하세요.
# TODO(현업): KV cache의 메모리 사용량을 dtype/layer/context 길이별로 측정하세요.
