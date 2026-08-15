"""Add first-use Korean API and formula explanations to learner notebooks.

The registry deliberately explains library mechanics, not a notebook's answer.
Builders can therefore use solution code as the reference while giving exercise
and solution notebooks the same API guidance.
"""

from __future__ import annotations

import ast
import html
import re
from dataclasses import dataclass
from typing import Any

import nbformat

API_EXPLANATIONS_START = "<!-- api-explanations:start -->"
API_EXPLANATIONS_END = "<!-- api-explanations:end -->"
_KEYS_PREFIX = "<!-- api-explanations:keys="
_BLOCK_PATTERN = re.compile(
    rf"\n?{re.escape(API_EXPLANATIONS_START)}.*?"
    rf"{re.escape(API_EXPLANATIONS_END)}\n?",
    flags=re.DOTALL,
)
_KEYS_PATTERN = re.compile(r"<!-- api-explanations:keys=([^>]*) -->")


@dataclass(frozen=True, slots=True)
class ApiExplanation:
    """Learner-facing explanation for one external API."""

    signature: str
    role: str
    inputs_return: str
    reason: str
    caution: str
    docs_url: str | None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FormulaExplanation:
    """Mathematical bridge shown below an API explanation when it is useful."""

    equation: str
    symbols: str
    code_bridge: str
    intuition: str
    shape: str | None = None


API_REGISTRY: dict[str, ApiExplanation] = {}
API_FORMULAS: dict[str, FormulaExplanation] = {}


def _add(
    key: str,
    signature: str,
    role: str,
    inputs_return: str,
    reason: str,
    caution: str,
    docs_url: str | None,
    *aliases: str,
) -> None:
    API_REGISTRY[key] = ApiExplanation(
        signature=signature,
        role=role,
        inputs_return=inputs_return,
        reason=reason,
        caution=caution,
        docs_url=docs_url,
        aliases=tuple(dict.fromkeys((key, *aliases))),
    )


def _add_formula(
    keys: str | tuple[str, ...],
    equation: str,
    symbols: str,
    code_bridge: str,
    intuition: str,
    shape: str | None = None,
) -> None:
    """Attach one equation explanation to one or more canonical API keys."""

    target_keys = (keys,) if isinstance(keys, str) else keys
    note = FormulaExplanation(
        equation=equation,
        symbols=symbols,
        code_bridge=code_bridge,
        intuition=intuition,
        shape=shape,
    )
    for key in target_keys:
        if key not in API_REGISTRY:
            raise KeyError(f"수식 해설 대상 API가 등록되지 않았습니다: {key}")
        API_FORMULAS[key] = note


def _register_comparison_apis() -> None:
    _add(
        "np.allclose",
        "np.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)",
        "두 배열의 모든 원소가 허용 오차 안에서 가까운지 검사합니다.",
        "배열 `a`, 기준 배열 `b`, 오차 설정을 받고 Python `bool` 하나를 반환합니다.",
        "부동소수점 계산 결과를 정확한 값이 아니라 수치적으로 같은지 확인합니다.",
        "조건은 `abs(a-b) <= atol + rtol*abs(b)`입니다. `==`와 달리 오차를 "
        "허용하고, `b`가 기준이라 비대칭일 수 있으며 1보다 매우 작은 값에는 기본 "
        "`atol`이 너무 클 수 있습니다.",
        "https://numpy.org/doc/stable/reference/generated/numpy.allclose.html",
        "numpy.allclose",
        "allclose",
    )
    _add(
        "np.isclose",
        "np.isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)",
        "각 원소 쌍이 허용 오차 안에서 가까운지 검사합니다.",
        "브로드캐스트 가능한 두 배열을 받고 원소별 불리언 배열을 반환합니다.",
        "어느 위치에서 수치 오차가 생겼는지 원소 단위로 확인합니다.",
        "`allclose`와 달리 결과가 배열이며 같은 비대칭 오차 공식을 사용합니다.",
        "https://numpy.org/doc/stable/reference/generated/numpy.isclose.html",
        "numpy.isclose",
    )
    _add(
        "np.isfinite",
        "np.isfinite(x)",
        "각 값이 NaN이나 무한대가 아닌 유한수인지 검사합니다.",
        "배열형 입력을 받고 같은 shape의 불리언 배열을 반환합니다.",
        "학습 값이나 전처리 결과가 수치적으로 정상인지 검증합니다.",
        "유한하다는 사실만 확인하며 값의 크기가 적절하다는 뜻은 아닙니다.",
        "https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html",
        "numpy.isfinite",
    )
    _add(
        "np.array_equal",
        "np.array_equal(a1, a2, equal_nan=False)",
        "shape과 모든 원소가 정확히 같은지 검사합니다.",
        "두 배열을 받고 Python `bool` 하나를 반환합니다.",
        "정수 인덱스나 마스크처럼 오차 허용이 없어야 하는 결과를 확인합니다.",
        "부동소수점 연산 결과에는 보통 `allclose`가 더 적합합니다.",
        "https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html",
        "numpy.array_equal",
        "array_equal",
    )
    _add(
        "np.equal",
        "np.equal(x1, x2)",
        "두 입력을 broadcasting한 뒤 원소별 정확한 동등성을 검사합니다.",
        "배열형 입력 둘을 받고 원소별 불리언 배열을 반환합니다.",
        "정수·문자열·마스크 값이 같은 위치를 벡터화해 찾습니다.",
        "부동소수점 근삿값 비교에는 허용 오차가 있는 `isclose`가 더 적합합니다.",
        "https://numpy.org/doc/stable/reference/generated/numpy.equal.html",
        "numpy.equal",
    )
    _add(
        "torch.allclose",
        "torch.allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False)",
        "두 텐서의 모든 원소가 허용 오차 안에서 가까운지 검사합니다.",
        "두 텐서와 오차 설정을 받고 Python `bool` 하나를 반환합니다.",
        "장치나 구현을 바꾼 뒤 수치 결과가 사실상 같은지 확인합니다.",
        "정확한 동일성 검사가 아니며 작은 값에서는 `atol`을 직접 정해야 합니다.",
        "https://docs.pytorch.org/docs/stable/generated/torch.allclose.html",
    )
    _add(
        "torch.isclose",
        "torch.isclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False)",
        "텐서 원소마다 허용 오차 이내인지 검사합니다.",
        "두 텐서를 받고 원소별 불리언 텐서를 반환합니다.",
        "오차가 발생한 위치를 텐서 단위로 진단합니다.",
        "전체 성공 여부가 필요하면 결과를 모두 축약하거나 `allclose`를 사용합니다.",
        "https://docs.pytorch.org/docs/stable/generated/torch.isclose.html",
    )
    _add(
        "torch.isfinite",
        "torch.isfinite(input)",
        "텐서의 각 원소가 유한수인지 검사합니다.",
        "텐서를 받고 같은 shape의 불리언 텐서를 반환합니다.",
        "loss나 gradient의 NaN·무한대 발생을 빠르게 찾습니다.",
        "검사 결과는 미분 대상이 아니며 원인 자체를 고치지는 않습니다.",
        "https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html",
    )
    _add(
        "torch.equal",
        "torch.equal(input, other)",
        "두 텐서의 크기와 원소가 정확히 같은지 검사합니다.",
        "두 텐서를 받고 Python `bool` 하나를 반환합니다.",
        "토큰 ID나 정수 마스크처럼 완전 일치가 필요한 결과를 확인합니다.",
        "dtype이 달라도 값이 같으면 참일 수 있고 NaN은 자기 자신과 같지 않습니다.",
        "https://docs.pytorch.org/docs/stable/generated/torch.equal.html",
    )


def _register_state_apis() -> None:
    rows = (
        (
            "random.seed",
            "random.seed(a=None)",
            "Python 표준 난수 생성기의 상태를 초기화합니다.",
            "정수 등의 시드를 받고 반환값은 없습니다.",
            "같은 Python 난수 수열을 다시 만들어 실험을 재현합니다.",
            "NumPy와 PyTorch의 난수 상태는 별도로 고정해야 합니다.",
            "https://docs.python.org/3/library/random.html#random.seed",
            (),
        ),
        (
            "np.random.seed",
            "np.random.seed(seed=None)",
            "NumPy의 기존 전역 난수 생성기 상태를 초기화합니다.",
            "시드를 받고 반환값은 없습니다.",
            "레거시 `np.random.*` 호출의 수열을 재현합니다.",
            "새 코드에서는 독립 상태를 가진 `default_rng`가 권장됩니다.",
            "https://numpy.org/doc/stable/reference/random/generated/"
            "numpy.random.seed.html",
            ("numpy.random.seed",),
        ),
        (
            "np.random.default_rng",
            "np.random.default_rng(seed=None)",
            "독립적인 NumPy 난수 생성기 `Generator`를 만듭니다.",
            "선택적 시드를 받고 `Generator` 객체를 반환합니다.",
            "전역 상태를 오염시키지 않고 데이터나 실험 난수를 재현합니다.",
            "생성기를 다시 만들지 않으면 호출할 때마다 내부 상태가 진행됩니다.",
            "https://numpy.org/doc/stable/reference/random/generator.html",
            ("numpy.random.default_rng", "default_rng"),
        ),
        (
            "torch.manual_seed",
            "torch.manual_seed(seed)",
            "PyTorch 난수 생성기의 시드를 설정합니다.",
            "정수 시드를 받고 `Generator`를 반환합니다.",
            "가중치 초기화와 텐서 샘플링을 반복 가능하게 만듭니다.",
            "완전한 결정성에는 알고리즘·장치 설정도 추가로 필요할 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html",
            (),
        ),
        (
            "torch.cuda.manual_seed_all",
            "torch.cuda.manual_seed_all(seed)",
            "사용 가능한 모든 CUDA GPU의 난수 시드를 설정합니다.",
            "정수 시드를 받고 반환값은 없습니다.",
            "여러 GPU에서도 초기화와 샘플링을 최대한 반복 가능하게 합니다.",
            "CPU 시드와 결정론 설정을 대신하지 않습니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.cuda.manual_seed_all.html",
            (),
        ),
        (
            "seed_everything",
            "seed_everything(seed)",
            "프로젝트가 쓰는 여러 난수 생성기를 한 번에 고정하는 도우미입니다.",
            "정수 시드를 받고 구현에 따라 설정 요약 또는 `None`을 반환합니다.",
            "Python·NumPy·PyTorch의 실험 시작 상태를 한곳에서 맞춥니다.",
            "같은 시드만으로 모든 GPU 연산의 완전한 결정성이 보장되지는 않습니다.",
            None,
            (),
        ),
        (
            "torch.no_grad",
            "torch.no_grad()",
            "블록 안에서 autograd 기록을 끄는 context manager입니다.",
            "보통 `with`에 사용하며 블록의 텐서는 기본적으로 gradient를 기록하지 않습니다.",
            "평가나 고정 teacher 추론의 메모리와 연산 부담을 줄입니다.",
            "`model.eval()`과 역할이 다르므로 평가 모드가 필요하면 둘 다 사용합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html",
            ("no_grad",),
        ),
        (
            "torch.inference_mode",
            "torch.inference_mode(mode=True)",
            "추론 전용으로 autograd 관련 추적을 더 강하게 비활성화합니다.",
            "불리언 모드를 받고 context manager 또는 decorator로 동작합니다.",
            "순수 평가 구간의 오버헤드를 줄입니다.",
            "이 모드에서 만든 텐서는 이후 gradient 계산에 쓰기 어려울 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.inference_mode.html",
            ("inference_mode",),
        ),
        (
            "Tensor.detach",
            "tensor.detach()",
            "같은 저장공간을 공유하면서 현재 계산 그래프에서 분리한 텐서를 만듭니다.",
            "텐서를 받아 gradient를 추적하지 않는 텐서 view를 반환합니다.",
            "로그·teacher target·NumPy 변환이 역전파 경로를 만들지 않게 합니다.",
            "원본과 저장공간을 공유하므로 in-place 변경은 양쪽에 영향을 줄 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html",
            (".detach",),
        ),
        (
            "Module.train",
            "model.train(mode=True)",
            "모듈과 하위 모듈을 학습 모드로 전환합니다.",
            "불리언 모드를 받고 모듈 자신을 반환합니다.",
            "Dropout과 BatchNorm이 학습 시 동작을 사용하도록 합니다.",
            "gradient를 켜는 함수가 아니므로 `no_grad`와 독립적입니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html",
            (".train",),
        ),
        (
            "Module.eval",
            "model.eval()",
            "모듈과 하위 모듈을 평가 모드로 전환합니다.",
            "인자 없이 호출하고 모듈 자신을 반환합니다.",
            "Dropout을 끄고 BatchNorm의 저장 통계를 사용하게 합니다.",
            "gradient 기록은 끄지 않으므로 필요하면 `no_grad`도 사용합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html",
            (".eval",),
        ),
        (
            "Optimizer.zero_grad",
            "optimizer.zero_grad(set_to_none=True)",
            "파라미터에 누적된 gradient를 비웁니다.",
            "초기화 방식을 받고 반환값은 없습니다.",
            "현재 mini-batch의 gradient만으로 update하도록 학습 단계를 시작합니다.",
            "호출하지 않으면 PyTorch gradient가 기본적으로 누적됩니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.optim.Optimizer.zero_grad.html",
            (".zero_grad",),
        ),
        (
            "Tensor.backward",
            "loss.backward()",
            "스칼라 loss에서 계산 그래프를 거슬러 gradient를 계산합니다.",
            "스칼라 텐서에서는 보통 인자 없이 호출하며 반환값은 없습니다.",
            "optimizer update에 필요한 각 파라미터의 `.grad`를 채웁니다.",
            "같은 그래프를 다시 역전파하려면 별도 보존 설정이 필요합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html",
            (".backward",),
        ),
        (
            "Optimizer.step",
            "optimizer.step(closure=None)",
            "현재 gradient와 optimizer 규칙으로 파라미터를 갱신합니다.",
            "선택적 closure를 받고 optimizer에 따라 loss 또는 `None`을 반환합니다.",
            "역전파로 계산한 gradient를 실제 학습 update에 반영합니다.",
            "일반적으로 `zero_grad`, forward, `backward` 다음에 호출합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html",
            (".step",),
        ),
        (
            "torch.optim.Adam",
            "torch.optim.Adam(params, lr=1e-3, ...)",
            "1·2차 gradient 모멘트를 추적하는 Adam optimizer를 만듭니다.",
            "파라미터와 학습률 등을 받고 optimizer 객체를 반환합니다.",
            "노트북의 trainable parameter를 반복 update합니다.",
            "weight decay의 해석은 AdamW와 다르며 학습률이 결과에 크게 작용합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html",
            ("Adam",),
        ),
        (
            "torch.optim.AdamW",
            "torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2, ...)",
            "weight decay를 gradient update와 분리한 AdamW optimizer를 만듭니다.",
            "파라미터와 hyperparameter를 받고 optimizer 객체를 반환합니다.",
            "Transformer 계열에서 흔한 최적화 규칙을 재현합니다.",
            "bias·정규화 파라미터를 decay에서 제외하는 설정도 자주 사용합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html",
            ("AdamW",),
        ),
    )
    for row in rows:
        _add(*row[:-1], *row[-1])


def _register_tensor_apis() -> None:
    rows = (
        (
            "Tensor.to",
            "tensor.to(device=None, dtype=None)",
            "텐서를 지정 장치나 dtype으로 변환합니다.",
            "장치·dtype을 받고 변환된 텐서를 반환합니다.",
            "모델과 입력의 장치·정밀도를 일치시킵니다.",
            "원본과 조건이 같으면 복사 없이 같은 텐서를 돌려줄 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html",
            (".to",),
        ),
        (
            "Tensor.cpu",
            "tensor.cpu()",
            "텐서를 CPU 메모리로 옮깁니다.",
            "텐서를 받아 CPU 텐서를 반환합니다.",
            "NumPy 변환이나 CPU 기반 평가 도구에 값을 전달합니다.",
            "GPU 텐서 이동은 동기화·복사 비용을 만들 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html",
            (".cpu",),
        ),
        (
            "Tensor.item",
            "tensor.item()",
            "원소 하나짜리 텐서에서 Python 숫자를 꺼냅니다.",
            "단일 원소 텐서를 받고 Python 숫자를 반환합니다.",
            "loss나 metric을 출력·기록 가능한 값으로 바꿉니다.",
            "여러 원소 텐서에는 쓸 수 없고 GPU에서는 동기화를 일으킬 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html",
            (".item",),
        ),
        (
            "torch.tensor",
            "torch.tensor(data, dtype=None, device=None)",
            "데이터를 복사해 새 PyTorch 텐서를 만듭니다.",
            "배열형 데이터와 선택적 dtype·device를 받고 텐서를 반환합니다.",
            "상수·label·축소 실험 입력을 명시적으로 구성합니다.",
            "기존 텐서에서 호출하면 복사가 생기므로 경우에 따라 `as_tensor`가 낫습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.tensor.html",
            (),
        ),
        (
            "torch.from_numpy",
            "torch.from_numpy(ndarray)",
            "NumPy 배열과 메모리를 공유하는 CPU 텐서를 만듭니다.",
            "NumPy `ndarray`를 받고 CPU 텐서를 반환합니다.",
            "복사 없이 준비된 NumPy 데이터를 모델 입력으로 바꿉니다.",
            "한쪽의 in-place 변경이 다른 쪽에도 반영될 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html",
            ("from_numpy",),
        ),
        (
            "torch.as_tensor",
            "torch.as_tensor(data, dtype=None, device=None)",
            "가능하면 데이터를 공유해 텐서로 해석합니다.",
            "배열형 데이터를 받고 텐서를 반환합니다.",
            "불필요한 복사를 줄이며 입력을 텐서 계약으로 맞춥니다.",
            "원본과 저장공간을 공유할 수 있어 변경 부작용을 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html",
            ("as_tensor",),
        ),
        (
            "torch.arange",
            "torch.arange(start=0, end, step=1, ...)",
            "일정 간격의 1차원 텐서를 만듭니다.",
            "범위와 간격을 받고 1차원 텐서를 반환합니다.",
            "position·index·time step을 벡터화해 만듭니다.",
            "부동소수점 step은 반올림 때문에 원소 수가 예상과 다를 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.arange.html",
            (),
        ),
        (
            "torch.zeros",
            "torch.zeros(*size, dtype=None, device=None)",
            "모든 원소가 0인 텐서를 만듭니다.",
            "shape·dtype·device를 받고 텐서를 반환합니다.",
            "초기 상태·mask·accumulator를 명시적으로 준비합니다.",
            "dtype과 device를 생략하면 주변 텐서와 달라질 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.zeros.html",
            (),
        ),
        (
            "torch.ones",
            "torch.ones(*size, dtype=None, device=None)",
            "모든 원소가 1인 텐서를 만듭니다.",
            "shape·dtype·device를 받고 텐서를 반환합니다.",
            "초기 weight·mask·스케일을 구성합니다.",
            "주변 연산과 dtype·device가 같은지 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.ones.html",
            (),
        ),
        (
            "torch.randn",
            "torch.randn(*size, generator=None, ...)",
            "표준정규분포에서 난수 텐서를 샘플링합니다.",
            "shape와 선택적 생성기를 받고 텐서를 반환합니다.",
            "합성 입력·noise·초기 실험값을 만듭니다.",
            "재현하려면 해당 PyTorch 생성기의 시드를 먼저 고정해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.randn.html",
            (),
        ),
        (
            "torch.randint",
            "torch.randint(low=0, high, size, ...)",
            "정수 구간에서 균등하게 난수 텐서를 뽑습니다.",
            "상·하한과 shape을 받고 정수 텐서를 반환합니다.",
            "토큰 ID·class label·무작위 index를 만듭니다.",
            "`high` 값은 포함되지 않으며 dtype과 seed를 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.randint.html",
            (),
        ),
    )
    for row in rows:
        _add(*row[:-1], *row[-1])

    dtype_rows = (
        ("torch.float16", "16비트 부동소수점 dtype", "정밀도와 메모리를 절충합니다."),
        ("torch.float32", "32비트 부동소수점 dtype", "일반적인 학습 실수 dtype입니다."),
        (
            "torch.float64",
            "64비트 부동소수점 dtype",
            "높은 수치 정밀도가 필요할 때 씁니다.",
        ),
        ("torch.int32", "32비트 정수 dtype", "정수 누산이나 양자화 값을 표현합니다."),
        ("torch.int64", "64비트 정수 dtype", "class·token index를 표현합니다."),
        ("torch.bool", "불리언 dtype", "mask와 조건 결과를 표현합니다."),
    )
    for key, role, reason in dtype_rows:
        _add(
            key,
            key,
            role,
            "값이 아니라 텐서 원소의 저장 형식으로 사용됩니다.",
            reason,
            "연산마다 허용 dtype이 다르고 변환 시 정밀도나 메모리가 달라집니다.",
            "https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype",
        )

    shape_rows = (
        (
            "Tensor.reshape",
            "tensor.reshape(*shape)",
            "원소 수를 유지하며 텐서 shape을 바꿉니다.",
            "새 shape을 받고 가능하면 view, 아니면 복사된 텐서를 반환합니다.",
            "batch·head·sequence 축을 논문 수식의 shape에 맞춥니다.",
            "원소 수가 같아야 하며 메모리 공유 여부에 의존하면 안 됩니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html",
            (".reshape",),
        ),
        (
            "Tensor.view",
            "tensor.view(*shape)",
            "같은 저장공간을 보는 새로운 shape의 텐서를 만듭니다.",
            "새 shape을 받고 view 텐서를 반환합니다.",
            "연속 메모리의 축을 복사 없이 재해석합니다.",
            "stride가 호환되지 않으면 실패하므로 `reshape`나 `contiguous`가 필요합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html",
            (".view",),
        ),
        (
            "Tensor.flatten",
            "tensor.flatten(start_dim=0, end_dim=-1)",
            "지정한 연속 축들을 하나의 축으로 합칩니다.",
            "축 범위를 받고 평탄화된 텐서를 반환합니다.",
            "feature map을 classifier 입력이나 토큰 열로 바꿉니다.",
            "batch 축까지 실수로 합치지 않도록 `start_dim`을 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.flatten.html",
            (".flatten",),
        ),
        (
            "Tensor.transpose",
            "tensor.transpose(dim0, dim1)",
            "두 축의 위치를 맞바꿉니다.",
            "두 축 번호를 받고 축이 교환된 view를 반환합니다.",
            "attention 행렬곱 등에서 축 순서를 맞춥니다.",
            "반환 텐서는 비연속일 수 있어 이후 `view` 전에 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.transpose.html",
            (".transpose",),
        ),
        (
            "Tensor.permute",
            "tensor.permute(*dims)",
            "모든 축을 지정한 순서로 재배열합니다.",
            "축 순열을 받고 재배열된 view를 반환합니다.",
            "batch·channel·height·width 또는 head 축 계약을 변환합니다.",
            "모든 축을 정확히 한 번씩 적어야 하고 결과는 비연속일 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.permute.html",
            (".permute",),
        ),
        (
            "Tensor.contiguous",
            "tensor.contiguous(memory_format=torch.contiguous_format)",
            "필요하면 연속 메모리 배치의 텐서 복사본을 만듭니다.",
            "선택적 메모리 형식을 받고 연속 텐서를 반환합니다.",
            "transpose·permute 뒤 `view`가 요구하는 stride 조건을 맞춥니다.",
            "이미 연속이면 같은 텐서를 반환할 수 있고 복사 비용이 생길 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html",
            (".contiguous",),
        ),
    )
    for row in shape_rows:
        _add(*row[:-1], *row[-1])

    op_rows = (
        ("torch.reshape", "torch.reshape(input, shape)", "텐서 shape을 바꿉니다."),
        (
            "torch.flatten",
            "torch.flatten(input, start_dim=0, end_dim=-1)",
            "축을 합칩니다.",
        ),
        ("torch.transpose", "torch.transpose(input, dim0, dim1)", "두 축을 바꿉니다."),
        ("torch.cat", "torch.cat(tensors, dim=0)", "기존 축을 따라 텐서를 연결합니다."),
        (
            "torch.stack",
            "torch.stack(tensors, dim=0)",
            "새 축을 만들어 텐서를 쌓습니다.",
        ),
        (
            "torch.einsum",
            "torch.einsum(equation, *operands)",
            "축 표기로 텐서 연산을 정의합니다.",
        ),
        (
            "torch.matmul",
            "torch.matmul(input, other)",
            "벡터·행렬·batch 행렬곱을 수행합니다.",
        ),
        ("torch.bmm", "torch.bmm(input, mat2)", "3차원 batch 행렬곱을 수행합니다."),
    )
    for key, signature, role in op_rows:
        docs_name = key.removeprefix("torch.")
        _add(
            key,
            signature,
            role,
            "텐서와 축·연산 설정을 받고 변환된 텐서를 반환합니다.",
            "수식의 축 관계와 batch 연산을 코드로 직접 표현합니다.",
            "입력 shape과 broadcasting 규칙을 셀의 shape assertion으로 확인해야 합니다.",
            f"https://docs.pytorch.org/docs/stable/generated/torch.{docs_name}.html",
        )


def _register_model_apis() -> None:
    activation_rows = (
        (
            "torch.softmax",
            "torch.softmax(input, dim, dtype=None)",
            "지정 축의 값을 합이 1인 확률형 가중치로 변환합니다.",
            "텐서와 축을 받고 같은 shape의 텐서를 반환합니다.",
            "attention score나 class logit을 정규화합니다.",
            "`dim`을 잘못 고르면 엉뚱한 축이 정규화되고 큰 logit은 포화될 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.softmax.html",
            ("F.softmax", "torch.nn.functional.softmax"),
        ),
        (
            "Tensor.masked_fill",
            "tensor.masked_fill(mask, value)",
            "mask가 참인 위치를 지정 값으로 바꿉니다.",
            "불리언 mask와 값을 받고 새 텐서를 반환합니다.",
            "padding이나 미래 token의 attention score를 선택에서 제외합니다.",
            "mask는 입력에 broadcast 가능해야 하고 softmax 전 값은 충분히 작아야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html",
            (".masked_fill",),
        ),
        (
            "Tensor.clamp",
            "tensor.clamp(min=None, max=None)",
            "텐서 값을 지정 범위 안으로 제한합니다.",
            "하한·상한을 받고 같은 shape의 텐서를 반환합니다.",
            "확률·분모·양자화 범위를 안전한 구간에 둡니다.",
            "경계 밖에서는 gradient가 0이 될 수 있어 학습 영향에 주의합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.clamp.html",
            (".clamp", "torch.clamp"),
        ),
        (
            "Tensor.argmax",
            "tensor.argmax(dim=None, keepdim=False)",
            "지정 축에서 최댓값의 index를 찾습니다.",
            "텐서와 축을 받고 정수 index 텐서를 반환합니다.",
            "logit을 예측 class나 선택 action으로 바꿉니다.",
            "미분 불가능하며 동점에서는 첫 index가 선택됩니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.argmax.html",
            (".argmax", "torch.argmax"),
        ),
        (
            "torch.topk",
            "torch.topk(input, k, dim=None, largest=True, sorted=True)",
            "지정 축에서 상위 k개 값과 index를 찾습니다.",
            "텐서·k·축을 받고 `(values, indices)` named tuple을 반환합니다.",
            "검색 후보·추천 결과·beam 후보를 제한합니다.",
            "동점 index 순서는 실행마다 안정적이지 않을 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.topk.html",
            (".topk",),
        ),
    )
    for row in activation_rows:
        _add(*row[:-1], *row[-1])

    layer_rows = (
        (
            "nn.Linear",
            "nn.Linear(in_features, out_features, bias=True)",
            "완전연결 affine 변환을 만듭니다.",
        ),
        (
            "nn.Conv2d",
            "nn.Conv2d(in_channels, out_channels, kernel_size, ...)",
            "2차원 convolution 층을 만듭니다.",
        ),
        (
            "nn.ConvTranspose2d",
            "nn.ConvTranspose2d(in_channels, out_channels, kernel_size, ...)",
            "학습 가능한 2차원 전치 convolution을 만듭니다.",
        ),
        (
            "nn.Embedding",
            "nn.Embedding(num_embeddings, embedding_dim, ...)",
            "정수 ID를 학습 가능한 벡터로 조회합니다.",
        ),
        (
            "nn.GRU",
            "nn.GRU(input_size, hidden_size, ...)",
            "gated recurrent unit 층을 만듭니다.",
        ),
        (
            "nn.LayerNorm",
            "nn.LayerNorm(normalized_shape, ...)",
            "마지막 feature 축들을 표준화합니다.",
        ),
        ("nn.Sequential", "nn.Sequential(*modules)", "모듈을 순서대로 연결합니다."),
        (
            "nn.Parameter",
            "nn.Parameter(data, requires_grad=True)",
            "텐서를 학습 파라미터로 등록합니다.",
        ),
        (
            "nn.ModuleList",
            "nn.ModuleList(modules=None)",
            "하위 모듈 목록을 등록 가능한 형태로 보관합니다.",
        ),
        (
            "nn.Dropout",
            "nn.Dropout(p=0.5)",
            "학습 중 일부 activation을 무작위로 0으로 만듭니다.",
        ),
        (
            "nn.ReLU",
            "nn.ReLU(inplace=False)",
            "음수 값을 0으로 만드는 활성함수를 만듭니다.",
        ),
        (
            "nn.GELU",
            "nn.GELU(approximate='none')",
            "부드러운 GELU 활성함수를 만듭니다.",
        ),
    )
    for key, signature, role in layer_rows:
        qualified = key.replace("nn.", "torch.nn.")
        _add(
            key,
            signature,
            role,
            "구조 hyperparameter를 받고 호출 가능한 `nn.Module` 또는 파라미터를 반환합니다.",
            "논문의 학습 가능한 변환을 명시적인 구성 요소로 조립합니다.",
            "입·출력 shape, bias 여부, 등록 상태가 논문 수식과 같은지 확인해야 합니다.",
            f"https://docs.pytorch.org/docs/stable/generated/{qualified}.html",
            qualified,
            key.removeprefix("nn."),
        )
    _add(
        "Module.register_buffer",
        "module.register_buffer(name, tensor, persistent=True)",
        "학습하지 않지만 state와 장치 이동을 따르는 텐서를 등록합니다.",
        "이름·텐서·저장 여부를 받고 반환값은 없습니다.",
        "position 상수나 running state를 파라미터와 구분해 보관합니다.",
        "일반 attribute와 달리 state dict 포함 여부가 `persistent`에 좌우됩니다.",
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html",
        ".register_buffer",
    )

    loss_rows = (
        (
            "F.cross_entropy",
            "F.cross_entropy(input, target, ...)",
            "logit에 log-softmax와 음의 로그우도를 결합합니다.",
        ),
        (
            "F.binary_cross_entropy_with_logits",
            "F.binary_cross_entropy_with_logits(input, target, ...)",
            "sigmoid와 binary cross entropy를 안정적으로 결합합니다.",
        ),
        (
            "F.mse_loss",
            "F.mse_loss(input, target, reduction='mean')",
            "예측과 target의 제곱 오차를 계산합니다.",
        ),
        (
            "F.l1_loss",
            "F.l1_loss(input, target, reduction='mean')",
            "예측과 target의 절댓값 오차를 계산합니다.",
        ),
        (
            "F.kl_div",
            "F.kl_div(input, target, reduction='mean', log_target=False)",
            "두 분포 사이 KL divergence 항을 계산합니다.",
        ),
        (
            "F.normalize",
            "F.normalize(input, p=2.0, dim=1, eps=1e-12)",
            "지정 축의 벡터 norm을 1로 정규화합니다.",
        ),
        (
            "F.one_hot",
            "F.one_hot(tensor, num_classes=-1)",
            "정수 class ID를 one-hot 텐서로 변환합니다.",
        ),
        (
            "nn.CrossEntropyLoss",
            "nn.CrossEntropyLoss(...)",
            "다중 class cross entropy loss 모듈을 만듭니다.",
        ),
        (
            "nn.MSELoss",
            "nn.MSELoss(reduction='mean')",
            "평균제곱오차 loss 모듈을 만듭니다.",
        ),
        (
            "nn.L1Loss",
            "nn.L1Loss(reduction='mean')",
            "절댓값오차 loss 모듈을 만듭니다.",
        ),
    )
    for key, signature, role in loss_rows:
        if key.startswith("F."):
            qualified = key.replace("F.", "torch.nn.functional.")
        else:
            qualified = key.replace("nn.", "torch.nn.")
        _add(
            key,
            signature,
            role,
            "예측·target과 설정을 받고 축약된 loss 또는 같은 shape의 loss를 반환합니다.",
            "논문의 objective를 수치화해 역전파 시작점으로 사용합니다.",
            "입력 형식, target dtype·shape, `reduction` 의미를 반드시 확인해야 합니다.",
            f"https://docs.pytorch.org/docs/stable/generated/{qualified}.html",
            qualified,
            key.removeprefix("F.").removeprefix("nn."),
        )
    _add(
        "DataLoader",
        "DataLoader(dataset, batch_size=1, shuffle=False, ...)",
        "dataset에서 mini-batch를 반복 생성합니다.",
        "dataset과 batching 설정을 받고 iterable loader를 반환합니다.",
        "학습 데이터를 batch 단위로 섞고 모델에 공급합니다.",
        "Windows 다중 worker는 진입점 보호가 필요하며 마지막 batch 크기도 확인합니다.",
        "https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader",
        "torch.utils.data.DataLoader",
    )


def _register_numpy_and_data_apis() -> None:
    numpy_rows = (
        (
            "np.load",
            "np.load(file, allow_pickle=False, ...)",
            "NumPy 배열이나 archive를 파일에서 읽습니다.",
        ),
        (
            "np.save",
            "np.save(file, arr, allow_pickle=True)",
            "한 NumPy 배열을 `.npy` 파일로 저장합니다.",
        ),
        (
            "np.asarray",
            "np.asarray(a, dtype=None, order=None)",
            "가능하면 복사 없이 입력을 NumPy 배열로 해석합니다.",
        ),
        (
            "np.array",
            "np.array(object, dtype=None, copy=True, ...)",
            "입력 데이터로 새 NumPy 배열을 만듭니다.",
        ),
        (
            "np.reshape",
            "np.reshape(a, newshape, order='C')",
            "원소 수를 유지하며 배열 shape을 바꿉니다.",
        ),
        ("np.stack", "np.stack(arrays, axis=0)", "새 축을 만들어 배열들을 쌓습니다."),
        (
            "np.concatenate",
            "np.concatenate(arrays, axis=0)",
            "기존 축을 따라 배열들을 연결합니다.",
        ),
        (
            "np.argmax",
            "np.argmax(a, axis=None, keepdims=False)",
            "최댓값의 index를 찾습니다.",
        ),
        (
            "np.argsort",
            "np.argsort(a, axis=-1, kind=None)",
            "정렬 결과를 만드는 index 순서를 반환합니다.",
        ),
        (
            "np.linalg.norm",
            "np.linalg.norm(x, ord=None, axis=None, ...)",
            "벡터나 행렬의 norm을 계산합니다.",
        ),
        (
            "np.mean",
            "np.mean(a, axis=None, dtype=None, ...)",
            "지정 축의 산술평균을 계산합니다.",
        ),
        (
            "np.std",
            "np.std(a, axis=None, ddof=0, ...)",
            "지정 축의 표준편차를 계산합니다.",
        ),
        (
            "np.sum",
            "np.sum(a, axis=None, dtype=None, ...)",
            "지정 축의 합을 계산합니다.",
        ),
        (
            "np.clip",
            "np.clip(a, a_min=None, a_max=None)",
            "배열 값을 범위 안으로 제한합니다.",
        ),
        ("np.where", "np.where(condition, x, y)", "조건에 따라 원소를 선택합니다."),
        ("np.exp", "np.exp(x)", "원소별 자연지수 함수를 계산합니다."),
        ("np.log", "np.log(x)", "원소별 자연로그를 계산합니다."),
        (
            "np.einsum",
            "np.einsum(subscripts, *operands)",
            "축 표기로 배열 연산을 정의합니다.",
        ),
    )
    special_docs = {
        "np.linalg.norm": "https://numpy.org/doc/stable/reference/generated/"
        "numpy.linalg.norm.html",
    }
    special_aliases = {
        "np.concatenate": ("np.concat", "numpy.concat"),
    }
    for key, signature, role in numpy_rows:
        name = key.removeprefix("np.")
        docs = special_docs.get(
            key,
            f"https://numpy.org/doc/stable/reference/generated/numpy.{name}.html",
        )
        _add(
            key,
            signature,
            role,
            "배열과 축·dtype 등의 설정을 받고 배열 또는 수치 결과를 반환합니다.",
            "데이터 shape과 수치 연산을 논문 실험에 맞게 구성·검증합니다.",
            "축, dtype, 복사·view 여부와 빈 배열에서의 동작을 확인해야 합니다.",
            docs,
            key.replace("np.", "numpy."),
            *special_aliases.get(key, ()),
        )

    pandas_rows = (
        (
            "pd.read_csv",
            "pd.read_csv(filepath_or_buffer, ...)",
            "CSV 표 데이터를 DataFrame으로 읽습니다.",
        ),
        (
            "pd.DataFrame",
            "pd.DataFrame(data=None, index=None, columns=None, ...)",
            "행·열 구조의 DataFrame을 만듭니다.",
        ),
        (
            "DataFrame.groupby",
            "frame.groupby(by, ...)",
            "key가 같은 행을 그룹으로 묶습니다.",
        ),
        (
            "Series.value_counts",
            "series.value_counts(normalize=False, ...)",
            "고유값별 빈도를 계산합니다.",
        ),
        ("DataFrame.isna", "frame.isna()", "결측값 위치를 불리언 표로 표시합니다."),
        (
            "DataFrame.drop_duplicates",
            "frame.drop_duplicates(subset=None, ...)",
            "중복 행을 제거합니다.",
        ),
    )
    pandas_docs = {
        "pd.read_csv": "pandas.read_csv",
        "pd.DataFrame": "pandas.DataFrame",
        "DataFrame.groupby": "pandas.DataFrame.groupby",
        "Series.value_counts": "pandas.Series.value_counts",
        "DataFrame.isna": "pandas.DataFrame.isna",
        "DataFrame.drop_duplicates": "pandas.DataFrame.drop_duplicates",
    }
    for key, signature, role in pandas_rows:
        if key.startswith("pd."):
            qualified = key.replace("pd.", "pandas.")
            aliases = (qualified, key.removeprefix("pd."))
        else:
            qualified = key
            aliases = (f".{key.split('.')[-1]}",)
        doc_name = pandas_docs[key]
        _add(
            key,
            signature,
            role,
            "표 데이터와 열·그룹 설정을 받고 DataFrame, Series 또는 집계 객체를 반환합니다.",
            "학습 전 데이터 분포·결측·중복과 분할 기준을 점검합니다.",
            "dtype 추론, 결측값, index 보존과 grouping 후 shape 변화를 확인해야 합니다.",
            f"https://pandas.pydata.org/docs/reference/api/{doc_name}.html",
            *aliases,
        )

    sklearn_rows = (
        (
            "train_test_split",
            "train_test_split(*arrays, test_size=None, random_state=None, ...)",
            "배열들을 같은 index 기준으로 train과 test로 나눕니다.",
            "하나 이상의 배열을 받고 각 배열의 train·test 조각을 순서대로 반환합니다.",
            "평가 데이터 누수를 막고 동일 분할을 재현합니다.",
            "분류에서는 `stratify`와 `random_state` 필요 여부를 확인해야 합니다.",
            "https://scikit-learn.org/stable/modules/generated/"
            "sklearn.model_selection.train_test_split.html",
            ("sklearn.model_selection.train_test_split",),
        ),
        (
            "Estimator.fit",
            "estimator.fit(X, y=None, **fit_params)",
            "데이터에서 estimator의 파라미터나 통계를 학습합니다.",
            "feature `X`와 선택적 target `y`를 받고 보통 estimator 자신을 반환합니다.",
            "학습 split만 사용해 모델·전처리기를 적합합니다.",
            "test 데이터에 `fit`하면 데이터 누수가 생기며 estimator별 입력 계약이 다릅니다.",
            "https://scikit-learn.org/stable/glossary.html#term-fit",
            (".fit",),
        ),
        (
            "Estimator.predict",
            "estimator.predict(X)",
            "학습된 estimator로 sample별 예측을 만듭니다.",
            "feature 행렬을 받고 sample 축 길이의 예측 배열을 반환합니다.",
            "검증·test split에서 최종 label이나 값을 평가합니다.",
            "먼저 `fit`되어야 하며 분류 확률이 필요한 경우와 구분해야 합니다.",
            "https://scikit-learn.org/stable/glossary.html#term-predict",
            (".predict",),
        ),
        (
            "Estimator.predict_proba",
            "estimator.predict_proba(X)",
            "sample별 class 확률 추정치를 반환합니다.",
            "feature 행렬을 받고 보통 `[samples, classes]` 확률 배열을 반환합니다.",
            "threshold·ROC·확률 기반 metric을 계산합니다.",
            "모든 estimator가 지원하지 않으며 class 열 순서는 `classes_`를 따릅니다.",
            "https://scikit-learn.org/stable/glossary.html#term-predict-proba",
            (".predict_proba",),
        ),
    )
    for row in sklearn_rows:
        _add(*row[:-1], *row[-1])


def _register_io_and_project_apis() -> None:
    rows = (
        (
            "Path",
            "Path(*pathsegments)",
            "운영체제에 맞는 파일 경로 객체를 만듭니다.",
            "경로 조각을 받고 `Path` 객체를 반환합니다.",
            "Windows에서도 문자열 결합 없이 데이터·결과 경로를 구성합니다.",
            "상대 경로는 현재 작업 디렉터리를 기준으로 해석됩니다.",
            "https://docs.python.org/3/library/pathlib.html#pathlib.Path",
            ("pathlib.Path",),
        ),
        (
            "Path.read_text",
            "path.read_text(encoding=None, errors=None)",
            "텍스트 파일 전체를 문자열로 읽습니다.",
            "인코딩 설정을 받고 문자열을 반환합니다.",
            "작은 설정·말뭉치·메타데이터 파일을 명시적 인코딩으로 읽습니다.",
            "학습용 대용량 파일에는 전체 메모리 적재가 부담될 수 있습니다.",
            "https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text",
            (".read_text",),
        ),
        (
            "Path.write_text",
            "path.write_text(data, encoding=None, errors=None, newline=None)",
            "문자열을 텍스트 파일에 쓰고 기존 내용을 교체합니다.",
            "문자열·인코딩 설정을 받고 쓴 문자 수를 반환합니다.",
            "생성 데이터나 실험 결과를 재현 가능한 형식으로 저장합니다.",
            "기존 파일을 덮어쓰며 Windows 줄바꿈 차이는 `newline`으로 고정할 수 있습니다.",
            "https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_text",
            (".write_text",),
        ),
        (
            "json.load",
            "json.load(fp, ...)",
            "열린 텍스트 파일의 JSON을 Python 객체로 역직렬화합니다.",
            "읽기 가능한 파일 객체를 받고 dict·list 등의 객체를 반환합니다.",
            "구조화된 학습 데이터나 manifest를 읽습니다.",
            "파일 인코딩과 신뢰할 수 없는 입력의 크기·구조를 검증해야 합니다.",
            "https://docs.python.org/3/library/json.html#json.load",
            ("load",),
        ),
        (
            "json.dump",
            "json.dump(obj, fp, ensure_ascii=True, indent=None, ...)",
            "Python 객체를 JSON으로 직렬화해 열린 파일에 씁니다.",
            "객체와 쓰기 가능한 파일을 받고 반환값은 없습니다.",
            "실험 설정·metric·데이터를 사람이 확인 가능한 형식으로 저장합니다.",
            "NaN 허용, 한글 escaping, key 정렬과 개행 정책을 명시해야 재현성이 좋아집니다.",
            "https://docs.python.org/3/library/json.html#json.dump",
            ("dump",),
        ),
        (
            "json.loads",
            "json.loads(s, ...)",
            "JSON 문자열을 Python 객체로 역직렬화합니다.",
            "문자열·bytes를 받고 dict·list 등의 객체를 반환합니다.",
            "메모리에 있는 JSON line이나 응답을 구조화합니다.",
            "입력이 한 문서인지 JSON Lines의 한 줄인지 구분해야 합니다.",
            "https://docs.python.org/3/library/json.html#json.loads",
            ("loads",),
        ),
        (
            "json.dumps",
            "json.dumps(obj, ensure_ascii=True, indent=None, ...)",
            "Python 객체를 JSON 문자열로 직렬화합니다.",
            "객체를 받고 JSON 문자열을 반환합니다.",
            "metric이나 sample을 파일·로그에 넣기 전 안정된 문자열로 만듭니다.",
            "한글 보존에는 `ensure_ascii=False`가 필요하고 객체 종류가 제한됩니다.",
            "https://docs.python.org/3/library/json.html#json.dumps",
            ("dumps",),
        ),
        (
            "get_accelerator",
            "get_accelerator(prefer_gpu=True)",
            "프로젝트 환경에서 사용할 CPU·GPU 가속기를 선택합니다.",
            "선호 설정을 받고 프로젝트 `Accelerator` 객체를 반환합니다.",
            "같은 실습이 GPU가 있으면 사용하고 없으면 CPU로 안전하게 실행되게 합니다.",
            "프로젝트 도우미이므로 반환 객체의 device·dtype 정책을 함께 확인해야 합니다.",
            None,
            (),
        ),
        (
            "ACCELERATOR.move",
            "ACCELERATOR.move(value)",
            "텐서나 모듈을 선택된 실행 장치로 옮기는 프로젝트 도우미입니다.",
            "이동할 값을 받고 같은 논리 값을 장치에 배치해 반환합니다.",
            "모델과 batch가 서로 다른 장치에 놓이는 오류를 막습니다.",
            "컨테이너 재귀 처리 여부와 dtype 변경 여부는 프로젝트 구현을 확인해야 합니다.",
            None,
            ("Accelerator.move", ".move"),
        ),
        (
            "summary",
            "summary(model, input_size=..., ...)",
            "모델 층별 출력 shape과 파라미터 수를 요약합니다.",
            "모델과 예시 입력 정보를 받고 요약 객체 또는 출력 결과를 반환합니다.",
            "구현된 구조가 논문의 shape·규모 계약과 맞는지 빠르게 확인합니다.",
            "라이브러리별 signature가 다르고 실제 forward가 실행될 수 있습니다.",
            None,
            (),
        ),
        (
            "sys.path.insert",
            "sys.path.insert(index, path)",
            "Python 모듈 검색 경로의 지정 위치에 경로를 추가합니다.",
            "index와 경로 문자열을 받고 반환값은 없습니다.",
            "노트북에서 프로젝트 `src` 모듈을 import할 수 있게 합니다.",
            "전역 import 상태를 바꾸므로 중복·우선순위와 실행 위치에 주의합니다.",
            "https://docs.python.org/3/library/sys.html#sys.path",
            (),
        ),
        (
            "dataclass",
            "@dataclass(frozen=False, slots=False, ...)",
            "type annotation을 바탕으로 초기화·표현 메서드 등을 생성합니다.",
            "class decorator 설정을 받고 변환된 class를 반환합니다.",
            "실험 설정과 결과 record의 필드를 명시적으로 정의합니다.",
            "mutable 기본값은 `default_factory`를 쓰고 equality·frozen 의미를 확인합니다.",
            "https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass",
            ("dataclasses.dataclass",),
        ),
        (
            "argparse.ArgumentParser",
            "argparse.ArgumentParser(...)",
            "명령행 인자 규칙과 도움말을 관리하는 parser를 만듭니다.",
            "프로그램 설명 등의 설정을 받고 parser 객체를 반환합니다.",
            "학습률·seed·장치 설정을 코드 수정 없이 실행 시점에 바꿉니다.",
            "notebook kernel 인자와 충돌할 수 있어 script 진입점에서 주로 사용합니다.",
            "https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser",
            ("ArgumentParser",),
        ),
        (
            "perf_counter",
            "perf_counter()",
            "짧은 구간 측정에 적합한 고해상도 단조 시계를 읽습니다.",
            "인자 없이 호출하고 초 단위 부동소수점 값을 반환합니다.",
            "학습·추론 latency의 시작과 끝 차이를 측정합니다.",
            "절댓값에는 의미가 없고 GPU 시간은 실행 동기화 뒤 측정해야 합니다.",
            "https://docs.python.org/3/library/time.html#time.perf_counter",
            ("time.perf_counter",),
        ),
    )
    for row in rows:
        _add(*row[:-1], *row[-1])


def _register_extended_math_apis() -> None:
    """Register frequent tensor operations whose mathematics matters to learners."""

    rows = (
        (
            "Tensor.mean",
            "tensor.mean(dim=None, keepdim=False, dtype=None)",
            "지정 축의 산술평균을 계산합니다.",
            "텐서와 축 설정을 받고 축약된 텐서를 반환합니다.",
            "batch loss, feature 통계, metric을 하나의 대표값으로 축약합니다.",
            "어느 축을 없애는지와 빈 텐서에서 NaN이 나올 수 있음을 확인합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.mean.html",
            (".mean", "torch.mean"),
        ),
        (
            "Tensor.sum",
            "tensor.sum(dim=None, keepdim=False, dtype=None)",
            "지정 축의 원소 합을 계산합니다.",
            "텐서와 축 설정을 받고 축약된 텐서를 반환합니다.",
            "확률 질량, mask 개수, message aggregation을 계산합니다.",
            "정수 overflow와 축약 뒤 shape 변화를 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.sum.html",
            (".sum", "torch.sum"),
        ),
        (
            "Tensor.std",
            "tensor.std(dim=None, correction=1, keepdim=False)",
            "지정 축의 표준편차를 계산합니다.",
            "텐서와 축·보정값을 받고 축약된 텐서를 반환합니다.",
            "표현의 퍼짐, 데이터 scale, 학습 안정성을 진단합니다.",
            "표본 수와 `correction`에 따라 작은 batch에서는 NaN이 될 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.std.html",
            (".std", "torch.std"),
        ),
        (
            "torch.linalg.norm",
            "torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False)",
            "벡터의 크기인 norm을 계산합니다.",
            "텐서·차수·축을 받고 norm 텐서를 반환합니다.",
            "거리, gradient 크기, 정규화 분모를 계산합니다.",
            "`dim`과 norm 차수에 따라 의미와 반환 shape이 달라집니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html",
            ("torch.linalg.vector_norm", ".norm", "torch.norm"),
        ),
        (
            "torch.sqrt",
            "torch.sqrt(input)",
            "각 원소의 제곱근을 계산합니다.",
            "0 이상 텐서를 받고 같은 shape의 텐서를 반환합니다.",
            "표준편차, 거리, diffusion schedule의 scale을 계산합니다.",
            "음수 입력은 NaN을 만들며 0 근처 gradient가 매우 커질 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.sqrt.html",
            (".sqrt",),
        ),
        (
            "torch.exp",
            "torch.exp(input)",
            "각 원소에 자연지수함수 $e^x$를 적용합니다.",
            "실수 텐서를 받고 같은 shape의 양수 텐서를 반환합니다.",
            "logit, 확률비, diffusion coefficient를 양수 scale로 바꿉니다.",
            "큰 양수는 overflow, 큰 음수는 0에 가까운 underflow를 만들 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.exp.html",
            (".exp",),
        ),
        (
            "torch.log",
            "torch.log(input)",
            "각 원소에 자연로그를 적용합니다.",
            "양수 텐서를 받고 같은 shape의 텐서를 반환합니다.",
            "곱을 합으로 바꾸고 log-likelihood를 계산합니다.",
            "0은 음의 무한대, 음수는 NaN이므로 작은 epsilon이나 clamp가 필요합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.log.html",
            (".log",),
        ),
        (
            "torch.sigmoid",
            "torch.sigmoid(input)",
            "실수를 0과 1 사이 값으로 바꿉니다.",
            "logit 텐서를 받고 같은 shape의 확률형 텐서를 반환합니다.",
            "이진 확률, gate, attention gate를 표현합니다.",
            "큰 절댓값에서는 gradient가 거의 0이 되며 BCE 학습은 logits 버전이 안정적입니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html",
            (".sigmoid", "F.sigmoid", "torch.nn.functional.sigmoid"),
        ),
        (
            "torch.tanh",
            "torch.tanh(input)",
            "실수를 -1과 1 사이 값으로 바꿉니다.",
            "실수 텐서를 받고 같은 shape의 텐서를 반환합니다.",
            "RNN hidden state나 범위가 제한된 출력을 만듭니다.",
            "큰 절댓값에서 포화되어 gradient가 작아질 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.tanh.html",
            (".tanh",),
        ),
        (
            "torch.cumprod",
            "torch.cumprod(input, dim)",
            "지정 축을 따라 누적곱을 계산합니다.",
            "텐서와 축을 받고 같은 shape의 누적곱 텐서를 반환합니다.",
            "diffusion의 누적 noise schedule처럼 단계별 곱을 한 번에 계산합니다.",
            "긴 수열에서는 underflow가 생길 수 있어 log-domain 계산을 고려합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.cumprod.html",
            (".cumprod",),
        ),
        (
            "torch.where",
            "torch.where(condition, input, other)",
            "조건에 따라 두 텐서의 원소를 선택합니다.",
            "불리언 조건과 두 입력을 받고 broadcast된 결과를 반환합니다.",
            "mask 기반 분기와 piecewise 수식을 벡터화합니다.",
            "세 입력의 broadcasting과 두 branch의 dtype·device를 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.where.html",
            (".where",),
        ),
        (
            "torch.multinomial",
            "torch.multinomial(input, num_samples, replacement=False, ...)",
            "각 행의 가중치에 비례해 범주 index를 샘플링합니다.",
            "음이 아닌 가중치와 표본 수를 받고 정수 index 텐서를 반환합니다.",
            "정책 action, negative sample, token을 확률적으로 선택합니다.",
            "입력은 합이 1일 필요는 없지만 음수·NaN이 없어야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html",
            (".multinomial",),
        ),
        (
            "F.relu",
            "F.relu(input, inplace=False)",
            "음수 activation을 0으로 만드는 ReLU를 적용합니다.",
            "텐서를 받고 같은 shape의 텐서를 반환합니다.",
            "선형 층 사이에 비선형성을 추가합니다.",
            "음수 구간 gradient는 0이며 in-place 연산은 autograd 값을 덮어쓸 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html",
            ("torch.nn.functional.relu", "torch.relu"),
        ),
        (
            "F.gelu",
            "F.gelu(input, approximate='none')",
            "입력을 확률적으로 gate하는 형태의 부드러운 GELU를 적용합니다.",
            "텐서를 받고 같은 shape의 텐서를 반환합니다.",
            "Transformer FFN의 비선형 변환을 구현합니다.",
            "근사 방식에 따라 작은 수치 차이가 생길 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html",
            ("torch.nn.functional.gelu",),
        ),
        (
            "F.log_softmax",
            "F.log_softmax(input, dim, dtype=None)",
            "softmax의 로그를 수치적으로 안정된 한 연산으로 계산합니다.",
            "logit과 축을 받고 같은 shape의 log-probability를 반환합니다.",
            "NLL, KL divergence, log-likelihood 계산에 사용합니다.",
            "`log(softmax(x))`를 따로 계산하면 overflow나 underflow가 생길 수 있습니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.nn.functional.log_softmax.html",
            ("torch.nn.functional.log_softmax", "torch.log_softmax"),
        ),
        (
            "F.cosine_similarity",
            "F.cosine_similarity(x1, x2, dim=1, eps=1e-8)",
            "두 벡터의 방향 유사도를 cosine 값으로 계산합니다.",
            "broadcast 가능한 두 텐서와 축을 받고 해당 축이 제거된 텐서를 반환합니다.",
            "embedding·표현 벡터의 크기보다 방향 일치를 비교합니다.",
            "0에 가까운 norm은 `eps`에 민감하고 출력 범위는 보통 -1에서 1입니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.nn.functional.cosine_similarity.html",
            ("torch.nn.functional.cosine_similarity",),
        ),
        (
            "F.pad",
            "F.pad(input, pad, mode='constant', value=None)",
            "텐서 가장자리에 값을 덧붙여 공간 또는 길이 축을 확장합니다.",
            "텐서와 뒤쪽 축부터 적는 padding 설정을 받고 확장된 텐서를 반환합니다.",
            "convolution 크기 보존, sequence 정렬, patch 분할 조건을 맞춥니다.",
            "`pad` 순서는 마지막 축부터이며 mode마다 허용 shape이 다릅니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html",
            ("torch.nn.functional.pad",),
        ),
        (
            "F.interpolate",
            "F.interpolate(input, size=None, scale_factor=None, mode='nearest', ...)",
            "feature map이나 신호를 지정 크기로 다시 샘플링합니다.",
            "입력과 목표 크기 또는 배율을 받고 resized 텐서를 반환합니다.",
            "U-Net skip 연결과 segmentation mask의 공간 크기를 맞춥니다.",
            "mode와 `align_corners` 선택은 값과 gradient에 영향을 줍니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.nn.functional.interpolate.html",
            ("torch.nn.functional.interpolate",),
        ),
        (
            "clip_grad_norm_",
            "clip_grad_norm_(parameters, max_norm, norm_type=2.0, ...)",
            "전체 gradient norm이 한계를 넘으면 같은 비율로 축소합니다.",
            "파라미터와 최대 norm을 받고 clipping 전 전체 norm을 반환합니다.",
            "RNN·Transformer·RL 학습의 폭주하는 update를 제한합니다.",
            "`backward()` 뒤 `step()` 전에 호출해야 하며 근본 원인을 없애지는 않습니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.nn.utils.clip_grad_norm_.html",
            ("torch.nn.utils.clip_grad_norm_",),
        ),
        (
            "Module.parameters",
            "model.parameters(recurse=True)",
            "모듈에 등록된 학습 파라미터 iterator를 반환합니다.",
            "하위 모듈 포함 여부를 받고 `Parameter` iterator를 반환합니다.",
            "optimizer가 갱신할 weight와 bias를 전달합니다.",
            "iterator는 한 번 소비되며 buffer와 일반 tensor attribute는 포함하지 않습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html",
            (".parameters",),
        ),
        (
            "Tensor.requires_grad_",
            "tensor.requires_grad_(requires_grad=True)",
            "텐서의 gradient 추적 여부를 in-place로 바꿉니다.",
            "불리언을 받고 같은 텐서를 반환합니다.",
            "teacher 동결이나 새로 학습할 파라미터 범위를 명시합니다.",
            "이미 만들어진 계산 그래프를 소급해 바꾸지 않으며 leaf 여부를 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.Tensor.requires_grad_.html",
            (".requires_grad_",),
        ),
        (
            "Tensor.softmax",
            "tensor.softmax(dim, dtype=None)",
            "지정 축의 값을 합이 1인 확률형 가중치로 변환합니다.",
            "텐서와 축을 받고 같은 shape의 텐서를 반환합니다.",
            "method 문법으로 attention score나 class logit을 정규화합니다.",
            "`dim`이 확률 합을 낼 축이며 입력은 일반적으로 raw logit이어야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.softmax.html",
            (".softmax",),
        ),
        (
            "Tensor.float",
            "tensor.float()",
            "텐서를 32비트 부동소수점 dtype으로 변환합니다.",
            "텐서를 받고 `torch.float32` 텐서를 반환합니다.",
            "정수 입력이나 NumPy 데이터를 신경망 weight와 계산 가능한 dtype으로 맞춥니다.",
            "class index는 보통 정수 dtype이어야 하므로 모든 텐서에 적용하면 안 됩니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.float.html",
            (".float",),
        ),
        (
            "F.linear",
            "F.linear(input, weight, bias=None)",
            "명시적으로 전달한 weight와 bias로 affine 변환을 계산합니다.",
            "입력·weight·선택적 bias를 받고 마지막 feature 축이 바뀐 텐서를 반환합니다.",
            "별도 `nn.Linear` 객체 없이 공유·재사용 weight의 선형 변환을 드러냅니다.",
            "weight shape은 `[out_features, in_features]`이고 transpose 방향에 주의합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html",
            ("torch.nn.functional.linear",),
        ),
        (
            "nn.BatchNorm1d",
            "nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, ...)",
            "mini-batch의 channel별 통계로 1D 또는 sequence feature를 정규화합니다.",
            "feature 수와 설정을 받고 학습·평가 상태를 가진 `nn.Module`을 반환합니다.",
            "activation scale을 안정화하고 running mean·variance를 학습합니다.",
            "학습은 batch 통계, 평가는 running 통계를 사용하므로 `train()`·`eval()`이 중요합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html",
            ("torch.nn.BatchNorm1d", "BatchNorm1d"),
        ),
        (
            "nn.BatchNorm2d",
            "nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, ...)",
            "CNN feature map의 channel별 mini-batch 통계로 값을 정규화합니다.",
            "channel 수와 설정을 받고 학습·평가 상태를 가진 `nn.Module`을 반환합니다.",
            "batch와 공간 위치에서 channel 통계를 내어 CNN 학습을 안정화합니다.",
            "작은 batch에서는 통계가 불안정하고 평가 전에 `eval()` 전환이 필요합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html",
            ("torch.nn.BatchNorm2d", "BatchNorm2d"),
        ),
        (
            "torch.optim.SGD",
            "torch.optim.SGD(params, lr, momentum=0, weight_decay=0, ...)",
            "gradient 방향으로 파라미터를 이동하는 SGD optimizer를 만듭니다.",
            "파라미터와 학습률·momentum 등을 받고 optimizer 객체를 반환합니다.",
            "고전 CNN과 최적화 기초 실험의 update 규칙을 명시합니다.",
            "momentum·weight decay·Nesterov 설정에 따라 실제 update 식이 달라집니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html",
            ("SGD",),
        ),
        (
            "torch.round",
            "torch.round(input, decimals=0)",
            "각 원소를 지정 자릿수로 반올림합니다.",
            "텐서와 자릿수를 받고 같은 shape의 텐서를 반환합니다.",
            "양자화에서 연속값을 이산 정수 격자에 대응시킵니다.",
            "절반값은 half-to-even이며 거의 모든 위치에서 gradient가 0입니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.round.html",
            (),
        ),
        (
            "torch.cdist",
            "torch.cdist(x1, x2, p=2.0, compute_mode=...)",
            "두 점 집합의 모든 쌍별 p-norm 거리를 계산합니다.",
            "두 batch 점 집합을 받고 pairwise distance 행렬을 반환합니다.",
            "clustering, nearest neighbor, representation separation을 벡터화합니다.",
            "두 입력의 feature 차원과 batch broadcasting 조건이 같아야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.cdist.html",
            (),
        ),
        (
            "F.scaled_dot_product_attention",
            "F.scaled_dot_product_attention(query, key, value, attn_mask=None, ...)",
            "scaled dot-product attention을 최적화된 kernel로 계산합니다.",
            "query·key·value와 mask를 받고 query 길이별 context를 반환합니다.",
            "Transformer의 score, mask, softmax, value 가중합을 한 API로 수행합니다.",
            "mask 불리언 의미와 dropout이 평가에서도 인자값대로 적용됨을 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/"
            "torch.nn.functional.scaled_dot_product_attention.html",
            ("torch.nn.functional.scaled_dot_product_attention",),
        ),
        (
            "nn.MultiheadAttention",
            "nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=False, ...)",
            "여러 attention head를 병렬 계산하는 모듈을 만듭니다.",
            "embedding·head 설정을 받고 self/cross-attention용 `nn.Module`을 반환합니다.",
            "표준 MHA를 기준 구현으로 사용하거나 논문 고유 블록과 비교합니다.",
            "기본 입력 축, mask shape, 반환 weight 평균 여부를 반드시 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html",
            ("torch.nn.MultiheadAttention", "MultiheadAttention"),
        ),
        (
            "nn.RNN",
            "nn.RNN(input_size, hidden_size, nonlinearity='tanh', ...)",
            "이전 hidden state와 현재 입력을 결합하는 기본 recurrent 층을 만듭니다.",
            "입력·hidden 크기와 설정을 받고 sequence module을 반환합니다.",
            "단순 RNN의 시간축 상태 전달을 GRU·LSTM과 비교합니다.",
            "긴 수열에서는 gradient 소실·폭주가 쉬우며 hidden 축 계약을 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html",
            ("torch.nn.RNN", "RNN"),
        ),
        (
            "nn.LSTM",
            "nn.LSTM(input_size, hidden_size, ...)",
            "input·forget·output gate와 cell state를 가진 recurrent 층을 만듭니다.",
            "입력·hidden 크기와 설정을 받고 sequence module을 반환합니다.",
            "장기 정보를 cell state로 보존하는 sequence encoder·decoder를 구현합니다.",
            "hidden state와 cell state 두 개를 전달하며 `batch_first`가 둘의 shape은 바꾸지 않습니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html",
            ("torch.nn.LSTM", "LSTM"),
        ),
        (
            "F.max_pool2d",
            "F.max_pool2d(input, kernel_size, stride=None, padding=0, ...)",
            "각 공간 window의 최댓값으로 feature map을 축소합니다.",
            "4D 텐서와 window 설정을 받고 축소된 feature map을 반환합니다.",
            "강한 지역 반응을 보존하면서 공간 해상도와 계산량을 줄입니다.",
            "stride·padding·ceil_mode가 출력 크기와 경계 window를 바꿉니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html",
            ("torch.nn.functional.max_pool2d",),
        ),
        (
            "F.avg_pool2d",
            "F.avg_pool2d(input, kernel_size, stride=None, padding=0, ...)",
            "각 공간 window의 평균으로 feature map을 축소합니다.",
            "4D 텐서와 window 설정을 받고 축소된 feature map을 반환합니다.",
            "지역 정보를 평균내어 downsampling하거나 global average pooling을 구성합니다.",
            "padding 값을 평균 분모에 포함할지와 출력 공간 크기를 확인해야 합니다.",
            "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html",
            ("torch.nn.functional.avg_pool2d",),
        ),
    )
    for row in rows:
        _add(*row[:-1], *row[-1])


def _register_formula_explanations() -> None:
    """Connect mathematical APIs to equations, symbols, code, and tensor shapes."""

    _add_formula(
        ("np.allclose", "np.isclose", "torch.allclose", "torch.isclose"),
        r"\lvert a_i-b_i\rvert \le "
        r"\mathrm{atol}+\mathrm{rtol}\,\lvert b_i\rvert",
        r"$a_i$는 계산값, $b_i$는 기준값, `atol`은 절대 오차, "
        r"`rtol`은 상대 오차 허용치입니다.",
        "`isclose`는 이 부등식을 원소마다 계산하고, `allclose`는 모든 결과에 "
        "논리 AND를 적용합니다.",
        "완전히 같은 비트인지가 아니라 실험 목적에 충분히 가까운지를 묻는 검사입니다.",
        "두 입력은 broadcasting 가능해야 합니다. `isclose`는 broadcast shape, "
        "`allclose`는 scalar `bool`을 반환합니다.",
    )
    _add_formula(
        "torch.optim.Adam",
        r"\begin{aligned}"
        r"m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\"
        r"v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\"
        r"\theta_t&=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}"
        r"\end{aligned}",
        r"$g_t$는 gradient, $m_t,v_t$는 1·2차 모멘트, $\eta$는 학습률입니다.",
        "`backward()`가 만든 gradient를 읽어 모멘트를 갱신한 뒤 `step()`에서 "
        "파라미터에 적용합니다.",
        "최근 gradient 방향과 크기를 함께 누적해 파라미터별 적응형 보폭을 만듭니다.",
        "각 trainable parameter와 같은 shape의 모멘트 상태가 optimizer에 저장됩니다.",
    )
    _add_formula(
        "torch.optim.AdamW",
        r"\theta_t=(1-\eta\lambda)\theta_{t-1}"
        r"-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}",
        r"$\lambda$는 weight decay이고 나머지 기호는 Adam과 같습니다.",
        "코드의 `weight_decay`가 gradient 기반 Adam update와 분리된 축소 항으로 "
        "적용됩니다.",
        "손실 gradient와 무관하게 큰 weight를 조금씩 줄이는 것이 Adam과의 핵심 차이입니다.",
        "파라미터와 모멘트 tensor의 shape은 변하지 않습니다.",
    )
    _add_formula(
        "Tensor.backward",
        r"\frac{\partial L}{\partial x}="
        r"\frac{\partial L}{\partial y}\frac{\partial y}{\partial x},"
        r"\qquad x.\mathrm{grad}\leftarrow x.\mathrm{grad}+\nabla_xL",
        r"$L$은 scalar loss이고 $x$는 계산 그래프의 leaf parameter입니다.",
        "`backward()`는 연쇄법칙으로 gradient를 계산해 각 parameter의 `.grad`에 누적합니다.",
        "파라미터를 직접 바꾸는 호출이 아니라 update에 필요한 미분값을 준비하는 호출입니다.",
        "각 `.grad`는 대응 parameter와 같은 shape이며 scalar가 아닌 출력에는 seed gradient가 "
        "필요합니다.",
    )
    _add_formula(
        "Optimizer.step",
        r"\theta_{t+1}=\mathcal{U}(\theta_t,\nabla_\theta L_t,s_t;\eta)",
        r"$\mathcal U$는 optimizer update, $s_t$는 momentum·moment 상태입니다.",
        "`step()`은 이미 `.grad`에 저장된 값을 읽어 선택한 optimizer 규칙을 적용합니다.",
        "gradient 계산과 parameter 갱신을 분리하면 학습 lifecycle을 정확히 추적할 수 있습니다.",
        "parameter shape은 유지되고 값과 optimizer state만 바뀝니다.",
    )
    _add_formula(
        "torch.optim.SGD",
        r"g_t=\nabla_\theta L_t,\qquad "
        r"v_t=\mu v_{t-1}+g_t,\qquad\theta_{t+1}=\theta_t-\eta v_t",
        r"$\eta$는 학습률, $\mu$는 momentum 계수입니다. $\mu=0$이면 기본 SGD입니다.",
        "`momentum`이 0이면 현재 gradient만, 양수이면 이전 update 방향도 함께 사용합니다.",
        "손실이 가장 빠르게 증가하는 gradient의 반대 방향으로 parameter를 이동합니다.",
        "각 momentum buffer는 대응 parameter와 같은 shape입니다.",
    )
    _add_formula(
        (
            "Tensor.reshape",
            "Tensor.view",
            "Tensor.flatten",
            "torch.reshape",
            "torch.flatten",
            "np.reshape",
        ),
        r"\prod_{k=1}^{K} d_k=\prod_{j=1}^{J} d'_j",
        r"$d_k$는 원래 각 축 크기, $d'_j$는 바꾼 뒤 각 축 크기입니다.",
        "`reshape`·`view`·`flatten`은 원소의 총개수를 보존하면서 축 해석만 바꿉니다.",
        "데이터 값을 계산하는 연산이 아니라 같은 원소를 다른 좌표계로 보는 연산입니다.",
        "예: `[B, H, T, D] -> [B, T, H*D]`; 왼쪽과 오른쪽 원소 수가 같아야 합니다.",
    )
    _add_formula(
        ("Tensor.transpose", "Tensor.permute", "torch.transpose"),
        r"Y_{i_{\pi(1)},\ldots,i_{\pi(K)}}=X_{i_1,\ldots,i_K}",
        r"$\pi$는 축의 새 순서를 나타내는 permutation입니다.",
        "`transpose`는 두 축만 바꾸고 `permute`는 모든 축의 새 순서를 명시합니다.",
        "원소 값을 바꾸지 않고 각 index가 의미하는 축의 위치만 다시 배치합니다.",
        "예: `[B,T,H,D] -> [B,H,T,D]`; 결과는 비연속 view일 수 있습니다.",
    )
    _add_formula(
        ("torch.matmul", "torch.bmm", "torch.einsum", "np.einsum"),
        r"C_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}",
        r"$k$는 곱한 뒤 사라지는 축이고 $i,j$는 결과에 남는 축입니다.",
        "`matmul`은 마지막 두 축에 이 합을 적용하고, `bmm`은 batch별로, "
        "`einsum`은 문자열로 축 관계를 지정합니다.",
        "공통 축 $k$의 정보를 가중합해 두 표현 사이의 상호작용을 만듭니다.",
        "`[..., I, K] @ [..., K, J] -> [..., I, J]`이며 앞쪽 batch 축은 "
        "broadcast될 수 있습니다.",
    )
    _add_formula(
        ("torch.cat", "np.concatenate"),
        r"d_{\mathrm{out}}^{(a)}=\sum_{r=1}^{R}d_r^{(a)}",
        r"$a$는 연결할 축이고 $R$은 입력 tensor 개수입니다.",
        "`dim=a` 축의 길이만 더하고 나머지 축은 그대로 유지합니다.",
        "이미 존재하는 축을 길게 이어 붙이는 연산입니다.",
        "연결 축을 제외한 모든 shape과 dtype·device가 같아야 합니다.",
    )
    _add_formula(
        ("torch.stack", "np.stack"),
        r"\operatorname{shape}(Y)=(d_1,\ldots,d_{a-1},R,d_a,\ldots,d_K)",
        r"$R$은 쌓는 tensor 개수이고 $a$는 새 축이 들어갈 위치입니다.",
        "`stack`은 각 입력을 새 index 하나로 배치해 새로운 축을 만듭니다.",
        "연결할 기존 축을 고르는 `cat`과 달리 축의 개수가 하나 늘어납니다.",
        "모든 입력 shape이 완전히 같아야 합니다.",
    )
    _add_formula(
        ("torch.softmax", "Tensor.softmax", "F.log_softmax"),
        r"p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},"
        r"\qquad \log p_i=z_i-\log\sum_j e^{z_j}",
        r"$z_i$는 logit, $p_i$는 선택한 축에서 합이 1인 확률입니다.",
        "코드의 `dim`이 분모의 합을 계산할 $j$축을 결정합니다.",
        "각 logit을 다른 후보와 비교 가능한 상대 확률로 바꿉니다.",
        "입력과 출력 shape은 같고 선택한 축의 softmax 값 합은 1입니다.",
    )
    _add_formula(
        "Tensor.masked_fill",
        r"s'_{ij}=\begin{cases}s_{ij},&M_{ij}=0\\-\infty,&M_{ij}=1\end{cases}"
        r"\quad\Rightarrow\quad\operatorname{softmax}(s')_{ij}=0\;\text{if }M_{ij}=1",
        r"$s$는 score, $M$은 제외할 위치가 참인 mask입니다.",
        "attention에서는 mask 위치를 매우 작은 값으로 채운 뒤 softmax를 적용합니다.",
        "확률을 나중에 지우는 대신 정규화 분모에도 들어가지 못하게 합니다.",
        "mask는 score에 broadcast 가능해야 하며 보통 `[B, H, T_q, T_k]`를 사용합니다.",
    )
    _add_formula(
        ("Tensor.argmax", "np.argmax"),
        r"k^\star=\operatorname*{arg\,max}_{k}x_k",
        r"$k^\star$는 가장 큰 값 자체가 아니라 그 값의 index입니다.",
        "`dim` 또는 `axis`가 어느 후보 집합에서 최댓값 index를 고를지 정합니다.",
        "연속 score를 이산 class·action 결정으로 바꾸는 선택 연산입니다.",
        "선택 축이 제거되며 미분 가능한 gradient 경로는 만들지 않습니다.",
    )
    _add_formula(
        "torch.topk",
        r"\mathcal I_K=\operatorname{TopKIndices}(\{x_i\}),\qquad v_r=x_{\mathcal I_{K,r}}",
        r"$\mathcal I_K$는 상위 $K$개 index 집합이고 $v_r$은 대응 값입니다.",
        "API가 선택 축에서 값과 index를 함께 반환합니다.",
        "모든 후보를 유지하지 않고 가장 유망한 K개만 다음 단계에 전달합니다.",
        "입력의 선택 축 길이가 `K`로 바뀐 `(values, indices)` 두 텐서를 반환합니다.",
    )
    _add_formula(
        ("Tensor.clamp", "np.clip"),
        r"y=\min\!\left(\max(x,\ell),u\right)",
        r"$\ell$은 하한, $u$는 상한입니다.",
        "코드의 `min`·`max` 바깥 값은 각각 경계값으로 치환됩니다.",
        "수치적으로 위험한 범위를 잘라내지만 경계 밖의 차이는 모두 잃습니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        ("nn.Linear", "F.linear"),
        r"y=xW^{\mathsf T}+b",
        r"$x$는 입력, $W$는 weight, $b$는 bias, $y$는 출력입니다.",
        "`in_features`와 `out_features`가 각각 $W$의 두 축을 결정합니다.",
        "각 출력 feature는 모든 입력 feature의 학습 가능한 가중합입니다.",
        "`[..., D_in] -> [..., D_out]`, weight는 `[D_out, D_in]`입니다.",
    )
    _add_formula(
        "nn.Conv2d",
        r"y_{n,o,h,w}=b_o+\sum_{c,i,j}W_{o,c,i,j}"
        r"x_{n,c,h\cdot s_h+i-p_h,w\cdot s_w+j-p_w}",
        r"$n,o,c$는 batch·출력·입력 channel, $i,j$는 kernel 위치입니다.",
        "`kernel_size`, `stride`, `padding`이 수식의 합 범위와 입력 좌표를 정합니다.",
        "같은 작은 filter를 모든 공간 위치에 공유해 지역 패턴을 찾습니다.",
        "`[N,C_in,H,W] -> [N,C_out,H_out,W_out]`; 공간 크기는 stride·padding에 "
        "따라 달라집니다.",
    )
    _add_formula(
        "nn.ConvTranspose2d",
        r"H_{out}=(H_{in}-1)s-2p+d(k-1)+o+1",
        r"$s,p,d,k,o$는 stride, padding, dilation, kernel, output padding입니다.",
        "입력 위치의 값을 kernel로 펼쳐 겹치는 출력 위치에 더합니다.",
        "convolution의 선형 연산을 전치한 것으로 단순한 역함수는 아닙니다.",
        "`[N,C_in,H,W] -> [N,C_out,H_out,W_out]`입니다.",
    )
    _add_formula(
        ("F.max_pool2d", "F.avg_pool2d"),
        r"H_{out}=\left\lfloor\frac{H_{in}+2p-d(k-1)-1}{s}+1\right\rfloor",
        r"$k,s,p,d$는 kernel, stride, padding, dilation입니다.",
        "각 window에서 max 또는 평균을 계산하고 stride만큼 이동합니다.",
        "channel 수는 유지하면서 공간 정보를 지역 대표값으로 압축합니다.",
        "`[N,C,H,W] -> [N,C,H_out,W_out]`입니다.",
    )
    _add_formula(
        "nn.Embedding",
        r"e_i=E[i],\qquad E\in\mathbb{R}^{V\times D}",
        r"$i$는 정수 ID, $V$는 vocabulary 크기, $D$는 embedding 차원입니다.",
        "forward는 one-hot 행렬곱을 만들지 않고 weight 행 $E[i]$를 직접 조회합니다.",
        "범주 ID를 학습 가능한 연속 벡터 좌표로 바꿉니다.",
        "입력 `[...] -> 출력 [..., D]`; 입력 dtype은 보통 `torch.int64`입니다.",
    )
    _add_formula(
        "nn.GRU",
        r"\begin{aligned}"
        r"r_t&=\sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr}),\\"
        r"z_t&=\sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz}),\\"
        r"n_t&=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\\"
        r"h_t&=(1-z_t)\odot n_t+z_t\odot h_{t-1}"
        r"\end{aligned}",
        r"$r_t,z_t$는 reset·update gate, $n_t$는 후보, $h_t$는 hidden state입니다.",
        "`nn.GRU`가 time step마다 이 네 식을 계산하고 hidden state를 다음 step으로 넘깁니다.",
        "gate가 과거 정보를 얼마나 지우고 유지할지 학습합니다.",
        "`batch_first=True`이면 입력 `[B,T,D]`, 출력 `[B,T,H]`, 마지막 hidden "
        "`[L,B,H]`입니다.",
    )
    _add_formula(
        "nn.RNN",
        r"h_t=\tanh(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh})",
        r"$x_t$는 현재 입력, $h_{t-1},h_t$는 이전·현재 hidden state입니다.",
        "RNN module이 sequence 순서대로 같은 weight를 반복 적용합니다.",
        "현재 상태가 이전 상태와 새 입력을 압축해 다음 시간으로 전달됩니다.",
        "`batch_first=True`이면 입력 `[B,T,D]`, 출력 `[B,T,H]`, hidden `[L,B,H]`입니다.",
    )
    _add_formula(
        "nn.LSTM",
        r"\begin{aligned}"
        r"i_t&=\sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi}),\\"
        r"f_t&=\sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf}),\\"
        r"g_t&=\tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg}),\\"
        r"o_t&=\sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho}),\\"
        r"c_t&=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)"
        r"\end{aligned}",
        r"$i,f,o$는 input·forget·output gate, $c_t$는 cell state입니다.",
        "LSTM module이 각 time step에서 네 gate와 cell·hidden update를 계산합니다.",
        "cell state가 정보를 더하기 경로로 전달해 기본 RNN보다 긴 의존성을 보존합니다.",
        "입력·출력은 RNN과 같고 마지막 상태는 `(h_n, c_n)` 두 `[L,B,H]` 텐서입니다.",
    )
    _add_formula(
        "nn.LayerNorm",
        r"y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i",
        r"$\mu,\sigma^2$는 한 sample의 지정 feature 축 평균·분산입니다.",
        "`normalized_shape`의 마지막 축들에서 통계를 내고 학습 가능한 scale과 bias를 "
        "적용합니다.",
        "batch 크기와 무관하게 각 token·sample의 feature scale을 안정화합니다.",
        r"입력과 출력 shape은 같고 $\gamma,\beta$는 `normalized_shape`를 따릅니다.",
    )
    _add_formula(
        ("nn.BatchNorm1d", "nn.BatchNorm2d"),
        r"\mu_c=\frac{1}{m}\sum_{i=1}^{m}x_{i,c},\qquad "
        r"\sigma_c^2=\frac{1}{m}\sum_{i=1}^{m}(x_{i,c}-\mu_c)^2,\qquad "
        r"y_{i,c}=\gamma_c\frac{x_{i,c}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta_c",
        r"$c$는 channel, $m$은 batch와 필요하면 공간 위치를 합친 표본 수입니다.",
        "학습 모드는 현재 batch 통계를 쓰고 running 통계를 갱신하며, 평가 모드는 저장된 "
        "running 통계를 사용합니다.",
        "channel마다 값의 scale을 맞춘 뒤 학습 가능한 scale과 bias로 표현력을 복원합니다.",
        r"입력과 출력 shape은 같고 $\gamma,\beta$는 `[C]`입니다.",
    )
    _add_formula(
        "nn.Dropout",
        r"y_i=\frac{m_i}{1-p}x_i,\qquad m_i\sim\operatorname{Bernoulli}(1-p)",
        r"$p$는 제거 확률, $m_i$는 학습 중 표본화한 mask입니다.",
        "`model.train()`일 때 mask와 $1/(1-p)$ scale을 적용하고 `eval()`에서는 그대로 둡니다.",
        "일부 activation에 의존하지 못하게 하면서 기댓값은 유지합니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        ("nn.ReLU", "F.relu"),
        r"\operatorname{ReLU}(x)=\max(0,x)",
        r"$x$는 선형 변환 뒤 activation입니다.",
        "코드가 음수 원소를 0으로 치환하고 양수는 그대로 통과시킵니다.",
        "단순하지만 여러 선형 층을 비선형 함수로 조합할 수 있게 합니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        ("nn.GELU", "F.gelu"),
        r"\operatorname{GELU}(x)=x\Phi(x)"
        r"\approx\frac{x}{2}\left[1+\tanh\!\left(\sqrt{\frac{2}{\pi}}"
        r"(x+0.044715x^3)\right)\right]",
        r"$\Phi(x)$는 표준정규분포의 누적분포함수입니다.",
        "`approximate` 설정에 따라 정확한 CDF 기반 값 또는 tanh 근사를 계산합니다.",
        "입력 크기에 비례해 부드럽게 gate하여 ReLU보다 경계가 매끄럽습니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        ("F.cross_entropy", "nn.CrossEntropyLoss"),
        r"\mathcal{L}_{CE}=-\frac{1}{N}\sum_{n=1}^{N}"
        r"\log\frac{e^{z_{n,y_n}}}{\sum_{c=1}^{C}e^{z_{n,c}}}",
        r"$z_{n,c}$는 class logit, $y_n$은 정답 class ID, $C$는 class 수입니다.",
        "함수가 내부에서 `log_softmax`와 정답 class의 음의 로그우도를 결합합니다.",
        "정답 class 확률이 낮을수록 큰 벌점을 줍니다.",
        "일반적으로 logits `[N,C,...]`, class-index target `[N,...]`을 사용합니다.",
    )
    _add_formula(
        "F.binary_cross_entropy_with_logits",
        r"\mathcal{L}_{BCE}=-\frac{1}{N}\sum_i"
        r"\left[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\right]",
        r"$z_i$는 logit, $y_i\in[0,1]$은 target, $\sigma$는 sigmoid입니다.",
        "sigmoid와 BCE를 한 함수에서 log-sum-exp 형태로 안정적으로 계산합니다.",
        "각 항을 독립적인 이진 선택으로 보고 정답 쪽 확률의 음의 로그를 최소화합니다.",
        "logit과 target은 같거나 broadcast 가능한 shape이어야 합니다.",
    )
    _add_formula(
        ("F.mse_loss", "nn.MSELoss"),
        r"\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2",
        r"$\hat y_i$는 예측, $y_i$는 target입니다.",
        "`reduction='mean'`이 모든 원소의 제곱 오차를 평균냅니다.",
        "큰 오차를 제곱해 더 강하게 벌점하므로 회귀와 reconstruction에 자주 씁니다.",
        "예측과 target은 같거나 broadcast 가능한 shape이어야 합니다.",
    )
    _add_formula(
        ("F.l1_loss", "nn.L1Loss"),
        r"\mathcal{L}_{L1}=\frac{1}{N}\sum_{i=1}^{N}\lvert\hat y_i-y_i\rvert",
        r"$\hat y_i$는 예측, $y_i$는 target입니다.",
        "`reduction='mean'`이 원소별 절댓값 차이를 평균냅니다.",
        "MSE보다 큰 오차 하나에 덜 민감하며 image 복원에서 선명도를 보존하는 데 쓰입니다.",
        "예측과 target은 같거나 broadcast 가능한 shape이어야 합니다.",
    )
    _add_formula(
        "F.kl_div",
        r"D_{KL}(P\Vert Q)=\sum_i P_i\log\frac{P_i}{Q_i}",
        r"$P$는 target 분포, $Q$는 model 분포입니다.",
        "PyTorch의 기본 계약은 `input=log(Q)`이고 `log_target=False`이면 `target=P`입니다.",
        "두 분포가 같은 곳에 확률 질량을 두도록 하며 일반적으로 방향이 비대칭입니다.",
        "class·token 확률 축이 같아야 하고 batch 평균은 `batchmean` 의미를 확인합니다.",
    )
    _add_formula(
        "F.normalize",
        r"\hat x=\frac{x}{\max(\lVert x\rVert_p,\epsilon)}",
        r"$p$는 norm 차수, $\epsilon$은 0으로 나누지 않기 위한 하한입니다.",
        "코드의 `dim` 축에서 norm을 구한 뒤 모든 원소를 같은 값으로 나눕니다.",
        "벡터 크기를 제거해 방향이나 상대 패턴에 집중하게 합니다.",
        "입력과 출력 shape은 같고 선택 축의 norm은 약 1입니다.",
    )
    _add_formula(
        "F.one_hot",
        r"H_{i,c}=\mathbb{1}[x_i=c]",
        r"$x_i$는 class ID이고 $H_{i,c}$는 ID가 $c$일 때만 1인 indicator입니다.",
        "API가 정수 index를 class 축이 추가된 indicator tensor로 바꿉니다.",
        "범주 번호를 거리 있는 수로 보지 않고 서로 독립적인 축으로 표현합니다.",
        "입력 `[...] -> [...,C]`; 출력은 정수 dtype이므로 loss에 따라 float 변환이 필요합니다.",
    )
    _add_formula(
        "F.cosine_similarity",
        r"\operatorname{cos}(x,y)=\frac{x^{\mathsf T}y}"
        r"{\max(\lVert x\rVert_2,\epsilon)\max(\lVert y\rVert_2,\epsilon)}",
        r"$x^{\mathsf T}y$는 내적이고 분모는 두 벡터의 길이입니다.",
        "`dim` 축에서 내적과 두 norm을 계산하고 그 축을 제거합니다.",
        "벡터 크기가 달라도 같은 방향이면 1에 가까워집니다.",
        "예: `[B,D]` 두 입력과 `dim=-1`이면 출력은 `[B]`입니다.",
    )
    _add_formula(
        ("Tensor.mean", "np.mean"),
        r"\bar x=\frac{1}{N}\sum_{i=1}^{N}x_i",
        r"$N$은 선택한 축의 원소 수입니다.",
        "`dim` 또는 `axis`가 평균에 포함할 원소와 결과에서 사라질 축을 결정합니다.",
        "여러 관측값을 중심 위치 하나로 요약합니다.",
        "`keepdim=False`이면 축약한 축이 제거되고 참이면 크기 1로 남습니다.",
    )
    _add_formula(
        ("Tensor.std", "np.std"),
        r"s=\sqrt{\frac{1}{N-\delta}\sum_{i=1}^{N}(x_i-\bar x)^2}",
        r"$\delta$는 correction이며 NumPy 기본은 0, PyTorch `std` 기본은 1입니다.",
        "API의 `correction` 또는 `ddof`가 분모를 $N$과 $N-1$ 중 무엇으로 할지 정합니다.",
        "평균 주변에 값이 얼마나 퍼져 있는지를 원래 단위로 나타냅니다.",
        "평균과 마찬가지로 지정한 축이 제거되거나 크기 1로 유지됩니다.",
    )
    _add_formula(
        ("Tensor.sum", "np.sum"),
        r"S=\sum_{i=1}^{N}x_i",
        r"$N$은 선택 축의 원소 수입니다.",
        "`dim` 또는 `axis`의 모든 원소를 더해 축약합니다.",
        "count, 확률 질량, 이웃 message처럼 누적량을 계산합니다.",
        "선택한 축은 제거되거나 `keepdim=True`이면 크기 1로 남습니다.",
    )
    _add_formula(
        ("torch.linalg.norm", "np.linalg.norm"),
        r"\lVert x\rVert_p=\left(\sum_i\lvert x_i\rvert^p\right)^{1/p},"
        r"\qquad \lVert x\rVert_2=\sqrt{\sum_i x_i^2}",
        r"$p$는 norm 차수이며 $p=2$가 유클리드 길이입니다.",
        "`ord`와 `dim`·`axis`가 어떤 원소 묶음의 크기를 계산할지 정합니다.",
        "여러 성분을 하나의 거리·크기 값으로 요약합니다.",
        "지정한 축은 결과에서 제거되며 `keepdim=True`이면 크기 1로 남습니다.",
    )
    _add_formula(
        "torch.sqrt",
        r"y_i=\sqrt{x_i},\qquad x_i\ge 0",
        r"$x_i$는 음수가 아닌 입력입니다.",
        "API가 제곱 또는 분산 scale을 원래 단위의 길이로 되돌립니다.",
        "제곱된 크기를 사람이 해석하기 쉬운 원래 scale로 바꿉니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        ("torch.exp", "np.exp"),
        r"y_i=e^{x_i}",
        r"$e$는 자연로그의 밑입니다.",
        "API가 각 원소를 양수 값으로 지수 변환합니다.",
        "더하기 차이를 곱셈 비율로 바꾸며 log-domain 값을 원래 scale로 복원합니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        ("torch.log", "np.log"),
        r"y_i=\log x_i,\qquad x_i>0",
        r"$\log$는 자연로그입니다.",
        "API가 곱셈 관계를 덧셈 관계로 바꾸고 작은 확률의 크기를 안정적으로 표현합니다.",
        "확률 곱을 log-probability 합으로 계산할 수 있게 합니다.",
        "입력과 출력 shape은 같으며 입력은 양수여야 합니다.",
    )
    _add_formula(
        "torch.sigmoid",
        r"\sigma(x)=\frac{1}{1+e^{-x}}",
        r"$x$는 제한되지 않은 logit이고 $\sigma(x)$는 0과 1 사이입니다.",
        "코드가 logit을 이진 확률이나 gate 값으로 변환합니다.",
        "0 근처는 민감하고 큰 절댓값에서는 0 또는 1에 포화됩니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        "torch.tanh",
        r"\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}",
        r"$x$는 실수이고 출력은 -1과 1 사이입니다.",
        "API가 activation을 0 중심의 제한된 범위로 압축합니다.",
        "부호는 유지하면서 큰 값의 영향을 제한합니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        "torch.cumprod",
        r"y_t=\prod_{s=0}^{t}x_s",
        r"$t$는 선택한 축의 현재 위치입니다.",
        "API가 각 위치까지의 곱을 모두 벡터화해 반환합니다.",
        "이전 단계의 효과가 현재 단계까지 계속 누적되는 schedule을 표현합니다.",
        "입력과 출력 shape은 같습니다.",
    )
    _add_formula(
        "torch.where",
        r"y_i=\begin{cases}x_i,&c_i=\mathrm{True}\\z_i,&c_i=\mathrm{False}\end{cases}",
        r"$c_i$는 조건, $x_i,z_i$는 두 후보 값입니다.",
        "API가 Python 분기 대신 원소별로 두 tensor 중 하나를 선택합니다.",
        "piecewise 수식과 mask 기반 update를 병렬 tensor 연산으로 표현합니다.",
        "세 입력을 broadcasting한 공통 shape이 출력 shape입니다.",
    )
    _add_formula(
        "torch.randn",
        r"X\sim\mathcal N(0,1),\qquad "
        r"f(x)=\frac{1}{\sqrt{2\pi}}\exp\!\left(-\frac{x^2}{2}\right)",
        r"$\mathcal N(0,1)$은 평균 0, 분산 1인 표준정규분포입니다.",
        "요청한 모든 위치에서 독립적인 표준정규 난수를 샘플링합니다.",
        "학습 noise와 합성 입력의 분포 가정을 코드로 만듭니다.",
        "인자로 지정한 shape 그대로 반환하며 generator·device가 재현성에 영향을 줍니다.",
    )
    _add_formula(
        "torch.randint",
        r"\Pr(X=k)=\frac{1}{\mathrm{high}-\mathrm{low}},\qquad "
        r"k\in\{\mathrm{low},\ldots,\mathrm{high}-1\}",
        r"$k$는 상한을 포함하지 않는 정수 범주의 한 값입니다.",
        "API가 지정 구간의 각 정수를 같은 확률로 샘플링합니다.",
        "무작위 index나 class ID를 편향 없이 선택합니다.",
        "인자로 지정한 shape의 정수 텐서를 반환합니다.",
    )
    _add_formula(
        "clip_grad_norm_",
        r"g' = g\cdot\min\!\left(1,\frac{c}{\lVert g\rVert_p+\epsilon}\right)",
        r"$g$는 모든 parameter gradient를 합친 벡터, $c$는 `max_norm`입니다.",
        "전체 norm이 한계를 넘을 때만 모든 gradient를 같은 비율로 줄입니다.",
        "방향은 유지하면서 update 크기만 제한합니다.",
        "각 parameter의 gradient shape은 유지됩니다.",
    )
    _add_formula(
        "torch.round",
        r"q=\operatorname{round}(x)",
        r"$x$는 연속값이고 $q$는 가장 가까운 정수 격자의 값입니다.",
        "양자화에서는 scale과 zero-point 변환 사이에 이 연산을 넣어 정수 code를 만듭니다.",
        "연속 공간을 이산 격자로 접어 정보량을 줄입니다.",
        "입력과 출력 shape은 같지만 dtype은 자동으로 정수로 바뀌지 않습니다.",
    )
    _add_formula(
        "torch.cdist",
        r"D_{ij}=\lVert x_i-y_j\rVert_p="
        r"\left(\sum_{k=1}^{D}\lvert x_{ik}-y_{jk}\rvert^p\right)^{1/p}",
        r"$i,j$는 두 점 집합의 index이고 $k$는 feature 축입니다.",
        "API가 모든 $i,j$ 조합의 거리를 Python loop 없이 계산합니다.",
        "가까운 표현은 작은 값, 멀리 떨어진 표현은 큰 값으로 나타납니다.",
        "`[B,P,D]`와 `[B,R,D]`를 받으면 `[B,P,R]`을 반환합니다.",
    )
    _add_formula(
        "F.scaled_dot_product_attention",
        r"\operatorname{Attention}(Q,K,V)="
        r"\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V",
        r"$Q,K,V$는 query·key·value, $d_k$는 head 차원, $M$은 mask입니다.",
        "함수 하나가 score 행렬곱, scale, mask, softmax, value 가중합을 순서대로 수행합니다.",
        "query와 key의 유사도로 value를 얼마나 섞을지 결정합니다.",
        "보통 `[...,T_q,D]`, `[...,T_k,D]`, `[...,T_k,D_v]`에서 "
        "`[...,T_q,D_v]`가 됩니다.",
    )
    _add_formula(
        "nn.MultiheadAttention",
        r"\mathrm{head}_h=\operatorname{Attention}(QW_h^Q,KW_h^K,VW_h^V),\qquad "
        r"\operatorname{MHA}(Q,K,V)=\operatorname{Concat}(\mathrm{head}_1,\ldots,"
        r"\mathrm{head}_H)W^O",
        r"$H$는 head 수이고 $W_h^Q,W_h^K,W_h^V,W^O$는 학습 projection입니다.",
        "모듈이 head 분할, 각 attention, concat, output projection을 묶어 계산합니다.",
        "여러 표현 부분공간에서 서로 다른 관계를 동시에 학습합니다.",
        "`batch_first=True`이면 주 입력은 `[B,T,E]`, 출력도 `[B,T_q,E]`입니다.",
    )


_register_comparison_apis()
_register_state_apis()
_register_tensor_apis()
_register_model_apis()
_register_numpy_and_data_apis()
_register_io_and_project_apis()
_register_extended_math_apis()
_register_formula_explanations()


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {
        "np": "numpy",
        "pd": "pandas",
        "nn": "torch.nn",
        "F": "torch.nn.functional",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local = imported.asname or imported.name.split(".")[0]
                aliases[local] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                if imported.name == "*":
                    continue
                local = imported.asname or imported.name
                aliases[local] = f"{node.module}.{imported.name}"
    return aliases


def _resolve_name(name: str, aliases: dict[str, str]) -> str:
    if not name:
        return ""
    head, separator, tail = name.partition(".")
    resolved_head = aliases.get(head, head)
    return f"{resolved_head}.{tail}" if separator else resolved_head


def _alias_map() -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    exact: dict[str, str] = {}
    suffixes: list[tuple[str, str]] = []
    for key, entry in API_REGISTRY.items():
        for alias in entry.aliases:
            if alias.startswith("."):
                suffixes.append((alias, key))
            else:
                exact[alias] = key
    suffixes.sort(key=lambda item: len(item[0]), reverse=True)
    return exact, tuple(suffixes)


_EXACT_ALIASES, _SUFFIX_ALIASES = _alias_map()


def _match_api(name: str) -> str | None:
    exact = _EXACT_ALIASES.get(name)
    if exact is not None:
        return exact
    for suffix, key in _SUFFIX_ALIASES:
        if name.endswith(suffix):
            return key
    return None


def detect_api_keys(source: str) -> list[str]:
    """Return important API keys in source order, without duplicates."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    aliases = _import_aliases(tree)
    candidates: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _resolve_name(_dotted_name(node.func), aliases)
            candidates.append((node.lineno, node.col_offset, name))
        elif isinstance(node, ast.Attribute):
            name = _resolve_name(_dotted_name(node), aliases)
            candidates.append((node.lineno, node.col_offset, name))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                target = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                name = _resolve_name(_dotted_name(target), aliases)
                candidates.append(
                    (
                        getattr(decorator, "lineno", node.lineno),
                        getattr(decorator, "col_offset", node.col_offset),
                        name,
                    )
                )
    keys: list[str] = []
    seen: set[str] = set()
    for _, _, name in sorted(candidates):
        key = _match_api(name)
        if key is not None and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _render_keys(keys: list[str]) -> str:
    if not keys:
        return ""
    formula_count = len({API_FORMULAS[key] for key in keys if key in API_FORMULAS})
    lines = [
        API_EXPLANATIONS_START,
        f"{_KEYS_PREFIX}{','.join(keys)} -->",
        "<details>",
        "<summary><strong>이 셀에서 처음 만나는 함수·API·수식 해설 "
        f"(API {len(keys)}개 · 수식 {formula_count}개)</strong></summary>",
        "",
        "> 아래 내용은 라이브러리 API의 의미를 설명합니다. 실습 정답 로직은 "
        "포함하지 않습니다.",
        "> 수식은 별도 표기가 없으면 대표적인 기본 설정을 나타냅니다. `dim`, "
        "`reduction`, 가중치 같은 실제 인자에 따라 세부 형태가 달라질 수 있습니다.",
    ]
    formula_owners: dict[FormulaExplanation, str] = {}
    for key in keys:
        entry = API_REGISTRY[key]
        lines.extend(
            [
                "",
                "<details>",
                f"<summary><code>{html.escape(entry.signature)}</code></summary>",
                "",
                f"#### `{entry.signature}`",
                "",
                f"- **역할:** {entry.role}",
                f"- **입력·반환:** {entry.inputs_return}",
                f"- **이 셀에서 쓰는 이유:** {entry.reason}",
                f"- **주의점:** {entry.caution}",
            ]
        )
        formula = API_FORMULAS.get(key)
        if formula is not None:
            owner = formula_owners.get(formula)
            if owner is not None:
                owner_signature = API_REGISTRY[owner].signature
                lines.append(
                    f"- **관련 수식:** 이 셀의 `{owner_signature}`와 같은 수식 계약을 "
                    "사용합니다. 중복 표시는 생략했습니다."
                )
            else:
                formula_owners[formula] = key
                lines.extend(
                    [
                        "",
                        "##### 관련 수식과 코드 연결",
                        "",
                        "$$",
                        formula.equation,
                        "$$",
                        "",
                        f"- **기호:** {formula.symbols}",
                        f"- **수식 → 코드:** {formula.code_bridge}",
                        f"- **직관:** {formula.intuition}",
                    ]
                )
                if formula.shape is not None:
                    lines.append(f"- **shape 확인:** {formula.shape}")
        if entry.docs_url:
            lines.append(f"- **공식 문서:** [{key}]({entry.docs_url})")
        else:
            lines.append(
                "- **공식 문서:** 프로젝트 내부 도우미이므로 정의의 docstring을 확인합니다."
            )
        lines.extend(["", "</details>"])
    lines.extend(["", "</details>", API_EXPLANATIONS_END])
    return "\n".join(lines)


def render_api_notes(source: str, seen: set[str]) -> str:
    """Render explanations for APIs first encountered in ``source``."""

    keys = [key for key in detect_api_keys(source) if key not in seen]
    seen.update(keys)
    return _render_keys(keys)


def _strip_api_notes(markdown: str) -> str:
    return _BLOCK_PATTERN.sub("\n", markdown).strip()


def _existing_keys(markdown: str) -> set[str]:
    match = _KEYS_PATTERN.search(markdown)
    if match is None:
        return set()
    return {key for key in match.group(1).split(",") if key in API_REGISTRY}


def append_api_notes(markdown: str, code_source: str, seen: set[str]) -> str:
    """Replace any generated block, then append the current first-use notes."""

    old_keys = _existing_keys(markdown)
    clean_markdown = _strip_api_notes(markdown)
    detected = detect_api_keys(code_source)
    keys = [key for key in detected if key not in seen or key in old_keys]
    seen.update(detected)
    notes = _render_keys(keys)
    if not notes:
        return clean_markdown
    if not clean_markdown:
        return notes
    return f"{clean_markdown}\n\n{notes}"


def _cell_type(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("cell_type", ""))
    return str(getattr(cell, "cell_type", ""))


def _cell_source(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("source", ""))
    return str(getattr(cell, "source", ""))


def _set_cell_source(cell: Any, source: str) -> None:
    if isinstance(cell, dict):
        cell["source"] = source
    else:
        cell.source = source


def _cell_metadata(cell: Any) -> Any:
    if isinstance(cell, dict):
        return cell.setdefault("metadata", {})
    return cell.metadata


def _notebook_cells(notebook: Any) -> list[Any]:
    if isinstance(notebook, dict):
        return notebook["cells"]
    return notebook.cells


def _new_markdown_cell(source: str, ordinal: int) -> Any:
    return nbformat.v4.new_markdown_cell(
        source=source,
        id=f"api-note-{ordinal:03d}",
        metadata={"api_explanations_inserted": True},
    )


def _remove_generated_blocks(notebook: Any) -> None:
    cells = _notebook_cells(notebook)
    kept: list[Any] = []
    for cell in cells:
        if _cell_type(cell) != "markdown":
            kept.append(cell)
            continue
        clean = _strip_api_notes(_cell_source(cell))
        _set_cell_source(cell, clean)
        metadata = _cell_metadata(cell)
        if metadata.get("api_explanations_inserted") and not clean:
            continue
        kept.append(cell)
    cells[:] = kept


def annotate_notebook(notebook: Any, reference_notebook: Any | None = None) -> Any:
    """Mutate a notebook so each important API is explained at first use."""

    _remove_generated_blocks(notebook)
    reference = reference_notebook if reference_notebook is not None else notebook
    reference_sources = [
        _cell_source(cell)
        for cell in _notebook_cells(reference)
        if _cell_type(cell) == "code"
    ]
    cells = _notebook_cells(notebook)
    seen: set[str] = set()
    code_ordinal = 0
    index = 0
    while index < len(cells):
        cell = cells[index]
        if _cell_type(cell) != "code":
            index += 1
            continue
        own_source = _cell_source(cell)
        source = (
            reference_sources[code_ordinal]
            if code_ordinal < len(reference_sources)
            else own_source
        )
        notes = render_api_notes(source, seen)
        code_ordinal += 1
        if not notes:
            index += 1
            continue
        if index > 0 and _cell_type(cells[index - 1]) == "markdown":
            previous = cells[index - 1]
            clean_markdown = _strip_api_notes(_cell_source(previous))
            updated = f"{clean_markdown}\n\n{notes}" if clean_markdown else notes
            _set_cell_source(previous, updated)
            index += 1
            continue
        cells.insert(index, _new_markdown_cell(notes, code_ordinal))
        index += 2
    return notebook


def annotate_pair(exercise: Any, solution: Any) -> tuple[Any, Any]:
    """Mutate an exercise/solution pair using only solution API usage as reference."""

    annotate_notebook(solution, reference_notebook=solution)
    annotate_notebook(exercise, reference_notebook=solution)
    return exercise, solution
