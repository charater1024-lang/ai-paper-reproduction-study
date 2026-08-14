"""Practice production-shaped LLM inference without an external model/API.

The script loads the tiny LM created by ``04_train_tiny_lm.py`` and exposes the
same ideas used in real serving code: prompt templates, tokenization caching,
dynamic batching, left padding, temperature/top-k/top-p sampling, and a real
per-layer KV cache.  Generated quality is intentionally modest; inspect the
pipeline and tensor behavior rather than treating it as a support chatbot.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.torch_text import (  # noqa: E402
    Vocabulary,
    resolve_device,
    seed_everything,
)
from llm_engineering_lab.transformer import (  # noqa: E402
    TinyCausalTransformer,
    load_tiny_lm_checkpoint,
)

DEFAULT_PROMPTS = [
    "결제가 두 번 처리되었습니다. 환불 방법을 알려주세요.",
    "비밀번호를 재설정했는데도 로그인이 되지 않습니다.",
    "배송 조회에는 완료라고 나오지만 상품을 받지 못했습니다.",
]


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    template: str

    def __post_init__(self) -> None:
        if "{message}" not in self.template:
            raise ValueError("Prompt template must contain {message}")

    def render(self, message: str) -> str:
        return self.template.format(message=message.strip())


TEMPLATES = {
    "support": PromptTemplate(
        "support",
        "역할: 고객 지원 상담원\n고객 문의: {message}\n간결한 답변:",
    ),
    "instruction": PromptTemplate(
        "instruction",
        "### 지시\n고객 문의를 해결하세요.\n### 입력\n{message}\n### 응답\n",
    ),
    "plain": PromptTemplate("plain", "{message}"),
}


class TokenizationCache:
    """A tiny LRU cache illustrating reusable prompt-prefix preprocessing."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: OrderedDict[tuple[str, int], tuple[int, ...]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, int]) -> tuple[int, ...] | None:
        value = self._values.get(key)
        if value is None:
            self.misses += 1
            return None
        self._values.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: tuple[str, int], value: Sequence[int]) -> None:
        self._values[key] = tuple(value)
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    message: str
    rendered_prompt: str
    completion: str
    prompt_tokens: int
    generated_tokens: int


class BatchedTinyLMRunner:
    """Prepare variable-length requests and invoke one batched model call."""

    def __init__(
        self,
        model: TinyCausalTransformer,
        vocabulary: Vocabulary,
        template: PromptTemplate,
        device: torch.device,
        *,
        token_cache_entries: int = 128,
    ) -> None:
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.template = template
        self.device = device
        self.token_cache = TokenizationCache(token_cache_entries)

    def _encode(self, rendered_prompt: str, max_prompt_tokens: int) -> list[int]:
        key = (rendered_prompt, max_prompt_tokens)
        cached = self.token_cache.get(key)
        if cached is not None:
            return list(cached)
        ids = self.vocabulary.encode(rendered_prompt, add_bos=True)
        # Preserve BOS plus the most recent instructions if the prompt is long.
        if len(ids) > max_prompt_tokens:
            ids = (
                [self.vocabulary.bos_id]
                if max_prompt_tokens == 1
                else [self.vocabulary.bos_id, *ids[-(max_prompt_tokens - 1) :]]
            )
        self.token_cache.put(key, ids)
        return ids

    def generate_batch(
        self,
        messages: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
        repetition_penalty: float,
        use_kv_cache: bool,
    ) -> tuple[list[GenerationResult], float]:
        if not messages:
            return [], 0.0
        max_prompt_tokens = self.model.config.max_seq_length - max_new_tokens
        if max_prompt_tokens < 1:
            raise ValueError("max_new_tokens leaves no room for a prompt")
        rendered = [self.template.render(message) for message in messages]
        encoded = [self._encode(prompt, max_prompt_tokens) for prompt in rendered]
        width = max(len(ids) for ids in encoded)

        # Left padding keeps every row's last column at its final prompt token.
        input_ids = torch.full(
            (len(messages), width),
            self.vocabulary.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, ids in enumerate(encoded):
            input_ids[row, -len(ids) :] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
            attention_mask[row, -len(ids) :] = True

        _synchronize(self.device)
        started = time.perf_counter()
        generated = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=self.vocabulary.eos_id,
            repetition_penalty=repetition_penalty,
            use_cache=use_kv_cache,
        )
        _synchronize(self.device)
        elapsed = time.perf_counter() - started

        results: list[GenerationResult] = []
        for row, (message, prompt, prompt_ids) in enumerate(
            zip(messages, rendered, encoded, strict=True)
        ):
            completion_ids = generated[row, width:].tolist()
            if self.vocabulary.eos_id in completion_ids:
                completion_ids = completion_ids[
                    : completion_ids.index(self.vocabulary.eos_id)
                ]
            results.append(
                GenerationResult(
                    message=message,
                    rendered_prompt=prompt,
                    completion=self.vocabulary.decode(completion_ids),
                    prompt_tokens=len(prompt_ids),
                    generated_tokens=len(completion_ids),
                )
            )
        return results, elapsed

    def generate(
        self,
        messages: Sequence[str],
        *,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
        repetition_penalty: float,
        use_kv_cache: bool,
    ) -> tuple[list[GenerationResult], float]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        all_results: list[GenerationResult] = []
        total_elapsed = 0.0
        for start in range(0, len(messages), batch_size):
            batch_results, elapsed = self.generate_batch(
                messages[start : start + batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_kv_cache=use_kv_cache,
            )
            all_results.extend(batch_results)
            total_elapsed += elapsed
        return all_results, total_elapsed


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "artifacts" / "tiny_lm.pt"
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Input message; repeat this flag to exercise batching.",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=None, help="UTF-8 text, one prompt per line"
    )
    parser.add_argument("--template", choices=sorted(TEMPLATES), default="support")
    parser.add_argument("--show-rendered-prompt", action="store_true")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="0 selects greedy decoding"
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="0 disables top-k filtering"
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument(
        "--benchmark-cache",
        action="store_true",
        help="Compare greedy decoding with and without the layer KV cache.",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompt_file is not None:
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts or DEFAULT_PROMPTS


def print_results(
    results: Sequence[GenerationResult],
    *,
    elapsed: float,
    show_rendered_prompt: bool,
    uses_cache: bool,
) -> None:
    total_new_tokens = sum(result.generated_tokens for result in results)
    throughput = total_new_tokens / elapsed if elapsed else 0.0
    print(
        f"requests={len(results)} kv_cache={uses_cache} elapsed={elapsed:.4f}s "
        f"new_tokens/s={throughput:.1f}"
    )
    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] message: {result.message}")
        if show_rendered_prompt:
            print(f"rendered prompt:\n{result.rendered_prompt}")
        print(f"tiny LM completion: {result.completion or '<EOS/empty>'}")
        print(
            f"tokens: prompt={result.prompt_tokens}, generated={result.generated_tokens}"
        )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "먼저 `python scripts/04_train_tiny_lm.py`를 실행하세요."
        )
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, payload = load_tiny_lm_checkpoint(args.checkpoint, map_location=device)
    vocabulary = Vocabulary.from_dict(payload["extra"]["vocabulary"])
    runner = BatchedTinyLMRunner(
        model,
        vocabulary,
        TEMPLATES[args.template],
        device,
    )
    prompts = collect_prompts(args)
    top_k = None if args.top_k == 0 else args.top_k
    uses_cache = not args.no_kv_cache
    results, elapsed = runner.generate(
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_kv_cache=uses_cache,
    )
    print(
        f"device={device} template={args.template} checkpoint_epoch={payload['epoch']}"
    )
    print_results(
        results,
        elapsed=elapsed,
        show_rendered_prompt=args.show_rendered_prompt,
        uses_cache=uses_cache,
    )
    print(
        f"tokenization_cache hits={runner.token_cache.hits} misses={runner.token_cache.misses}"
    )

    if args.benchmark_cache:
        print("\n--- greedy KV-cache micro benchmark (quality is not measured) ---")
        timings: dict[bool, float] = {}
        for cache_enabled in (False, True):
            _, benchmark_elapsed = runner.generate(
                prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_k=None,
                top_p=1.0,
                repetition_penalty=1.0,
                use_kv_cache=cache_enabled,
            )
            timings[cache_enabled] = benchmark_elapsed
            print(f"kv_cache={cache_enabled}: {benchmark_elapsed:.4f}s")
        if timings[True] > 0:
            print(f"no-cache/cache ratio: {timings[False] / timings[True]:.2f}x")


if __name__ == "__main__":
    main()


# TODO(연습): batch_size별 tokens/s와 p95 latency를 측정해 표로 저장하세요.
# TODO(심화): system prompt의 layer KV cache를 요청 사이에 재사용하는 prefix cache를 설계하세요.
# TODO(현업): request id, timeout, cancellation, token budget를 갖는 async queue를 구현하세요.
