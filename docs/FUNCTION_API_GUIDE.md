# 함수와 Python 실행 뼈대 읽기 가이드

[← 저장소 학습 안내로 돌아가기](../README.md)

이 문서는 노트북의 **이 셀에서 처음 만나는 함수·API·수식 해설**과 `projects/nlp/01~06`의
`starter.py`·`solution.py`를 읽을 때 공통으로 만나는 Python 문법을 설명합니다.
코드를 외우기보다 아래 네 가지를 먼저 확인하는 것이 목표입니다.

1. 함수에 무엇을 넣는가?
2. 무엇을 반환하며 반환형은 무엇인가?
3. 이 위치에서 왜 필요한가?
4. 기본값이나 부작용 때문에 생길 수 있는 실수는 무엇인가?

## 1. 노트북의 첫 사용 접이식 해설 읽기

생성된 노트북은 중요한 외부 API가 **처음 등장하는 코드 셀 바로 앞**에
`이 셀에서 처음 만나는 함수·API·수식 해설`을 붙입니다. 첫 접기를 열어 API 목록을 보고,
궁금한 API의 두 번째 접기를 열어 다음 순서로 읽습니다.

- **입력과 핵심 인자**: 어떤 값·shape를 받고 `axis`, `dim`, `dtype` 등이 무엇을 정하는가?
- **반환값**: scalar, 배열, 새 객체 중 무엇이며 원본과 메모리를 공유하는가?
- **이 셀에서의 이유**: 같은 일을 하는 여러 API 중 왜 이 함수를 골랐는가?
- **주의점**: tolerance, 기본값, gradient, device처럼 결과를 바꿀 조건은 무엇인가?
- **관련 수식**: 어떤 기호와 기본식을 계산하며 코드 인자와 어떻게 연결되는가?
- **shape 확인**: 입력·중간값·출력의 축이 수식과 일치하는가?

실습 노트북과 정답 노트북의 해설은 **정답 코드에서 발견한 API를 기준으로 동일하게**
생성됩니다. 따라서 TODO 때문에 함수 이름이 가려져도 먼저 계약을 읽고 구현할 수 있습니다.
같은 설명은 노트북마다 첫 사용 위치에만 두어 코드 흐름을 방해하지 않습니다. `print`, `len`
같은 기초 함수가 매번 설명되지 않는 것은 누락이 아니라, 수치·shape·학습 상태를 바꾸는 API에
집중하기 위한 설계입니다.

## 2. 정확 비교와 근사 비교: `==`, `array_equal`, `isclose`, `allclose`

부동소수점 계산에는 반올림 오차가 있으므로 목적에 맞는 비교 함수를 골라야 합니다.

```python
import numpy as np

expected = np.array([0.1 + 0.2, 1.0])
actual = np.array([0.3, 1.0])

print(expected == actual)             # 원소별 정확 비교: [False, True]
print(np.array_equal(expected, actual))  # 배열 전체 정확 비교: False
print(np.isclose(expected, actual))    # 원소별 근사 비교: [True, True]
print(np.allclose(expected, actual))   # 배열 전체 근사 비교: True
```

`np.allclose(a, b, rtol=1e-5, atol=1e-8)`는 모든 원소가 다음 조건을 만족하면 하나의
`bool`을 반환합니다.

$$
|a-b| \leq \mathrm{atol} + \mathrm{rtol}\,|b|
$$

- `atol`은 0 근처에서도 적용되는 절대 허용 오차입니다.
- `rtol`은 기준 배열 `b`의 크기에 비례하는 상대 허용 오차입니다.
- 식에서 `b`가 기준이므로 드문 경우 `allclose(a, b)`와 `allclose(b, a)`가 다를 수 있습니다.
- 아주 작은 수를 비교할 때 기본 `atol`이 너무 관대할 수 있으므로 문제의 단위에 맞게 지정합니다.
- NaN을 같은 위치에서 같다고 취급하려면 `equal_nan=True`가 필요합니다.

예제의 재현성 검사는 `np.allclose(numpy_sample, np.random.random(3))`가 **배열 전체를
허용 오차로 한 번에 검사**하므로 `assert` 조건에 바로 사용할 수 있습니다. 같은 프로세스에서
같은 NumPy 생성기와 호출 순서를 정확히 되풀이했다면 `array_equal`도 통과할 수 있지만,
모델 연산 결과나 다른 하드웨어의 부동소수점 결과에는 보통 `allclose`가 더 적합합니다.
상세 계약은 [NumPy `allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)와
[`isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)를 참고하세요.

## 3. seed는 난수를 멈추는 값이 아니라 수열의 시작점이다

```python
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

seed를 설정하면 각 난수 생성기가 같은 시작 상태로 돌아갑니다. 그 뒤 **같은 함수들을 같은
순서와 횟수로 호출해야** 같은 수열이 나옵니다. 난수를 한 번 더 뽑거나 다른 생성기를 사용하면
이후 값이 달라집니다. Python `random`, NumPy, PyTorch는 서로 다른 생성기이므로 하나만
설정해서는 나머지가 고정되지 않습니다.

`np.random.seed`는 기존 전역 생성기를 다시 설정하는 API입니다. 새 코드에서는 독립적인
`rng = np.random.default_rng(seed)`를 함수나 클래스에 전달하면 전역 상태 간섭을 줄일 수
있습니다. seed는 같은 환경의 재현성을 높이지만 라이브러리 버전, GPU 연산, 병렬 실행까지
항상 비트 단위로 같게 보장하지는 않습니다. PyTorch의 결정적 알고리즘 설정은 성능과 사용
가능한 연산에 영향을 줄 수 있습니다.

공식 설명은 [Python `random.seed`](https://docs.python.org/3/library/random.html#random.seed),
[NumPy random sampling](https://numpy.org/doc/stable/reference/random/),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)를 참고하세요.

## 4. PyTorch gradient 생명주기와 학습 순서

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(features)
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()
optimizer.step()
```

1. `model.train()`은 Dropout과 BatchNorm 같은 모듈을 학습 동작으로 전환합니다.
2. `zero_grad()`는 이전 반복에서 누적된 `.grad`를 지웁니다. PyTorch gradient는 기본적으로
   누적되므로 생략하면 여러 batch의 gradient가 의도치 않게 더해집니다.
3. forward 연산은 gradient 계산 그래프와 `loss`를 만듭니다.
4. `loss.backward()`는 연쇄 법칙으로 leaf parameter의 `.grad`를 계산·누적합니다.
5. `optimizer.step()`은 현재 `.grad`를 사용해 parameter를 갱신합니다.

`set_to_none=True`는 0 텐서를 채우는 대신 `.grad = None` 상태로 되돌려 메모리 쓰기를 줄일
수 있습니다. 일부 사용자 정의 코드가 `.grad`가 항상 텐서라고 가정한다면 차이를 고려해야
합니다. gradient accumulation을 의도한다면 여러 micro-batch에서 `backward()`를 수행한 뒤
한 번만 `step()`하고, loss를 누적 횟수로 나누는 방식이 흔합니다.

학습 그래프의 기본은 [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html),
optimizer 계약은 [`Optimizer.zero_grad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)와
[`Optimizer.step`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html)을
참고하세요.

### `train()`·`eval()`과 `no_grad()`는 서로 대체하지 않는다

```python
model.eval()
with torch.no_grad():
    validation_logits = model(validation_features)
```

- `model.eval()`은 Dropout을 끄고 BatchNorm이 저장된 통계를 쓰게 하는 등 **모듈의 동작
  모드**를 바꿉니다. gradient 기록을 끄지는 않습니다.
- `torch.no_grad()`는 블록 안의 **역전파 그래프 기록**을 끕니다. 모델을 평가 모드로
  바꾸지는 않습니다.
- 따라서 일반적인 검증·추론에는 둘을 함께 사용하고, 다시 학습할 때 `model.train()`을
  호출합니다.

공식 계약은 [`Module.train`/`eval`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)과
[`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)에 있습니다.

## 5. shape와 메모리 배치: `view`, `reshape`, `transpose`, `permute`

```python
x = torch.randn(2, 3, 4)       # [batch, time, feature]
flat = x.reshape(2, -1)        # [2, 12], -1은 한 축만 자동 추론
swapped = x.permute(0, 2, 1)   # [batch, feature, time]
safe_flat = swapped.contiguous().view(2, -1)
```

- `view()`는 같은 저장공간을 다른 shape로 해석하므로 stride가 호환되어야 합니다.
- `reshape()`는 가능하면 view를 반환하고, 불가능하면 복사할 수 있습니다. 따라서 반환값이
  원본과 저장공간을 공유하는지에 의존하는 코드는 피합니다.
- `transpose(dim0, dim1)`는 두 축만 맞바꾸고, `permute(*dims)`는 모든 축의 새 순서를
  지정합니다. 둘은 값의 축 배치를 바꾸며 단순 reshape와 의미가 다릅니다.
- 축 순서를 바꾼 텐서는 non-contiguous일 수 있습니다. `view()` 전에 `contiguous()`로
  연속 메모리 복사본을 만들거나 바로 `reshape()`를 사용합니다.
- reshape 전후 원소 수는 같아야 하며 `-1` 자동 추론은 한 축에만 쓸 수 있습니다.

공식 예시는 [`Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html),
[`torch.reshape`](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html),
[`Tensor.permute`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.permute.html)를
참고하세요.

## 6. device와 그래프 경계: `to`, `detach`, `cpu`, `item`

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch = batch.to(device)

loss_value = loss.detach().cpu().item()
predictions = logits.detach().cpu().numpy()
```

- `.to(device)`는 지정한 장치의 텐서/모듈을 반환합니다. 텐서에서는 `batch = batch.to(device)`처럼
  반환값을 받아야 하며, 모델·입력은 같은 device에 있어야 연산할 수 있습니다.
- `.detach()`는 현재 autograd 그래프에서 분리된 텐서를 반환합니다. 원본과 저장공간을 공유할
  수 있으므로 in-place 수정에는 주의합니다.
- `.cpu()`는 CPU 메모리의 텐서를 반환합니다. NumPy는 일반 CUDA 텐서를 직접 변환할 수
  없으므로 보통 `detach().cpu().numpy()` 순서를 사용합니다.
- `.item()`은 원소 하나짜리 텐서를 Python 숫자로 바꿉니다. 여러 원소 텐서에는 쓸 수 없고,
  GPU 값을 읽을 때 CPU와의 동기화가 생길 수 있어 training loop에서 매 연산마다 남용하지
  않습니다.

공식 계약은 [`Tensor.to`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html),
[`Tensor.detach`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html),
[`Tensor.cpu`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html),
[`Tensor.item`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html)에 있습니다.

## 7. logits, `softmax`, loss의 역할을 구분하기

```python
logits = model(features)  # 정규화되지 않은 점수 [batch, classes]
loss = torch.nn.functional.cross_entropy(logits, targets)
probabilities = torch.softmax(logits, dim=-1)
predictions = logits.argmax(dim=-1)
```

- logits는 합이 1일 필요가 없는 모델의 원시 점수입니다.
- `softmax(logits, dim=-1)`는 지정한 class 축에서 양수이고 합이 1인 확률 형태로 바꿉니다.
  `dim`을 잘못 고르면 batch 사이를 정규화하는 오류가 생깁니다.
- `cross_entropy`는 class logits에 log-softmax와 negative log likelihood를 안정적으로
  결합하므로, 일반적인 단일-label 분류에서는 logits에 softmax를 먼저 적용하지 않습니다.
- class index target은 보통 `[batch]` 정수형이고 logits는 `[batch, classes]`입니다.
- `argmax`는 softmax 전후 순위를 바꾸지 않으므로 class 예측만 필요하면 logits에서 바로
  계산할 수 있습니다.

이진·multi-label 문제의 `binary_cross_entropy_with_logits`도 sigmoid와 BCE를 수치적으로
안정되게 결합합니다. MSE/L1은 회귀 오차의 크기를 재며 분류 확률 손실과 목적이 다릅니다.
공식 문서는 [`softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html),
[`cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html),
[`binary_cross_entropy_with_logits`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)를
참고하세요.

## 8. pandas와 scikit-learn의 상태 있는 API

```python
import pandas
from sklearn.model_selection import train_test_split

frame = pandas.read_csv(data_path)
summary = frame.groupby("label")["score"].mean()

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_valid)
```

- `read_csv()`는 파일을 `DataFrame`으로 읽으며 `encoding`, `dtype`, 결측값 해석이 데이터
  계약을 바꿀 수 있습니다. 읽은 직후 `shape`, columns, dtypes, 결측치를 점검합니다.
- `groupby()`는 즉시 표를 만드는 함수가 아니라 split-apply-combine 연산을 준비한 GroupBy
  객체를 반환합니다. 뒤의 `mean`, `agg`, `transform`에 따라 결과 shape와 index가 달라집니다.
- scikit-learn의 `fit()`은 학습 데이터에서 vocabulary·평균·모델 parameter 같은 상태를
  배웁니다. `transform()`/`predict()`는 그 상태를 새 데이터에 적용합니다.
- `fit_transform()`을 validation/test에 따로 호출하면 평가 데이터에서 상태를 다시 배워
  누수가 생길 수 있습니다. train에만 fit하고 나머지는 transform합니다.
- `train_test_split(..., random_state=42)`는 분할을 재현하고, `stratify=y`는 분류 label 비율을
  보존합니다. 같은 고객·문서의 변형이 있다면 행 단위 split보다 group split이 필요합니다.
- `Pipeline`은 전처리와 estimator를 묶어 교차 검증에서도 train fold에만 fit되도록 돕습니다.

공식 설명은 [pandas `read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html),
[GroupBy](https://pandas.pydata.org/docs/user_guide/groupby.html),
[scikit-learn `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
[`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)을
참고하세요.

## 9. `Path(__file__)`: 실행 파일을 기준으로 경로 찾기

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "customer_support_tickets.csv"
```

- `__file__`은 현재 실행 중인 `.py` 파일의 경로를 담는 Python 특수 변수입니다.
- `Path(__file__)`은 문자열 경로를 `pathlib.Path` 객체로 바꿉니다. `/` 연산자로
  하위 경로를 이어 붙일 수 있어 Windows의 역슬래시를 직접 조립할 필요가 없습니다.
- `.resolve()`는 `..` 등을 정리한 절대 경로를 만듭니다.
- `.parents[3]`은 부모 경로 목록에서 네 번째 항목을 선택합니다. 현재 프로젝트의
  `projects/nlp/번호_프로젝트/solution.py`에서 저장소 루트까지 올라가기 위한 값입니다.

이 방식은 PowerShell의 현재 작업 폴더가 달라도 같은 데이터 파일을 찾게 합니다.
단, 파일 위치를 옮기면 `parents[3]`이 가리키는 곳도 달라집니다. 또한 Jupyter 셀처럼
일반적인 `.py` 파일 밖에서 실행할 때는 `__file__`이 없을 수 있으므로 노트북은
`Path.cwd()` 또는 별도의 루트 탐색 함수를 사용합니다.

공식 API는 [Python `pathlib`](https://docs.python.org/3/library/pathlib.html)을 참고하세요.

## 10. `sys.path.insert(0, ...)`: 로컬 `src`를 먼저 검색하기

```python
import sys

sys.path.insert(0, str(ROOT / "src"))
from llm_engineering_lab.rag import load_knowledge_base
```

Python은 모듈을 가져올 때 `sys.path`에 적힌 폴더를 앞에서부터 검색합니다.
`insert(0, 경로)`는 저장소의 `src`를 검색 목록 맨 앞에 넣어, 아직 패키지를 설치하지
않았더라도 이 저장소의 최신 코드를 우선 불러오게 합니다. `Path`가 아니라 `str`로
변환하는 이유는 import 검색 경로가 문자열 경로를 기준으로 동작하기 때문입니다.

이 코드는 학습용 스크립트를 어디서 실행해도 로컬 모듈을 찾게 하는 작은 부트스트랩입니다.
실제 배포 패키지에서는 `pip install -e .` 같은 설치 방식을 우선하고, 여러 곳에서
무분별하게 `sys.path`를 바꾸지 않는 편이 좋습니다. 맨 앞에 넣은 이름이 설치된 다른
패키지와 같으면 로컬 파일이 그 패키지를 가릴 수 있기 때문입니다.

## 11. `@dataclass(frozen=True, slots=True)`: 설정의 계약 표현하기

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchConfig:
    data_path: Path
    top_k: int = 3
    min_score: float = 0.01
```

`@dataclass`는 타입이 표시된 필드를 바탕으로 `__init__`, `__repr__`, 값 비교 메서드
등을 자동으로 만듭니다. 따라서 여러 인자를 느슨한 `dict`로 전달하는 대신, 실행 조건을
이름과 타입이 있는 하나의 객체로 묶을 수 있습니다.

- `frozen=True`: 생성 후 `config.top_k = 10` 같은 필드 재할당을 막습니다. 실험 도중
  설정이 조용히 바뀌는 것을 줄이지만, 필드 안의 가변 리스트까지 깊게 불변으로 만드는
  것은 아닙니다.
- `slots=True`: 선언한 필드에 필요한 저장 공간만 만들고, 오타로
  `config.topkk = 10` 같은 새 속성을 붙이는 것을 막습니다.
- `field(default_factory=...)`: 리스트나 딕셔너리처럼 가변 기본값이 필요할 때 인스턴스마다
  새 값을 만들게 합니다. `items=[]`를 클래스 선언에 직접 두는 방식은 피합니다.

설정 객체는 보통 `frozen=True`가 잘 맞지만, 학습 과정에서 값이 계속 누적되는 report나
state 객체는 가변 dataclass가 더 자연스러울 수 있습니다.

세부 인자는 [Python `dataclasses`](https://docs.python.org/3/library/dataclasses.html)를
참고하세요.

## 12. `argparse`: 명령줄 문자열을 검증된 값으로 바꾸기

```python
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--show-prompt", action="store_true")
    return parser.parse_args()
```

`ArgumentParser`는 PowerShell에서 전달한 명령줄 옵션을 정의하고 도움말을 생성합니다.
`add_argument()`의 중요한 인자는 다음과 같습니다.

| 인자 | 역할 | 예 |
|---|---|---|
| 옵션 이름 | 사용자가 터미널에서 적는 이름 | `--top-k` |
| `type` | 문자열 인자를 원하는 타입으로 변환하며 실패 시 오류 표시 | `type=int` |
| `default` | 사용자가 옵션을 생략했을 때 사용할 값 | `default=3` |
| `action` | 값을 받는 대신 특별한 동작 수행 | `store_true`는 옵션이 있으면 `True` |
| `help` | `--help`에 표시할 설명 | `help="검색 결과 수"` |

`parser.parse_args()`는 정의에 따라 실제 인자를 읽고 `argparse.Namespace`를 반환합니다.
옵션의 하이픈은 밑줄 속성으로 바뀌므로 `--top-k 5`는 `args.top_k == 5`가 됩니다.
`Namespace`는 최종 설정 객체가 아니라 명령줄 파싱 결과입니다. 이 프로젝트에서는 다음처럼
명시적인 dataclass로 옮겨 검증과 프로젝트 로직을 분리합니다.

```python
args = parse_args()
config = SearchConfig(
    data_path=ROOT / "data" / "knowledge_base.jsonl",
    top_k=args.top_k,
)
project = SearchProject(config)
project.run()
```

`type=int`는 타입 힌트만 붙이는 것이 아니라 실행 시 실제 변환을 수행합니다. 반대로
`default`의 타입은 작성자가 직접 일관되게 맞춰야 합니다. `store_true` 옵션에는 별도의
문자열 값이 필요 없으므로 `--show-prompt true`가 아니라 `--show-prompt`만 적습니다.

옵션별 동작은 [Python `argparse`](https://docs.python.org/3/library/argparse.html)를
참고하세요.

## 13. 함수 설명을 읽고 직접 확인하는 순서

노트북의 접힌 API 설명을 읽은 뒤 작은 입력으로 다음을 확인해 보세요.

```python
result = some_function(input_value)

print(type(result))
print(getattr(result, "shape", None))
print(result)
```

- 반환값이 하나의 `bool`인지, 원소별 Boolean 배열인지 구분합니다.
- 배열 함수라면 입력·출력 `shape`와 연산 축 `axis`/`dim`을 확인합니다.
- 난수 함수라면 seed가 어느 난수 생성기에 적용되는지 확인합니다.
- 학습 함수라면 모델 파라미터나 gradient를 바꾸는 부작용이 있는지 확인합니다.
- `default`, tolerance, threshold를 바꿨을 때 통과 조건이 어떻게 달라지는지 비교합니다.

예를 들어 `np.allclose(a, b)`는 배열 전체가 허용 오차 안에서 가까운지를 **하나의
Boolean 값**으로 반환합니다. 정확한 `==` 비교가 아니며, 원소별 결과가 필요할 때는
`np.isclose(a, b)`를 사용합니다. 이처럼 함수 이름만 번역하지 말고 반환형과 현재 셀에서
그 함수를 선택한 이유까지 함께 읽어야 코드를 안전하게 바꿀 수 있습니다.

## 14. 해설이 빠진 API를 발견했을 때

노트북의 해설은 `tools/notebook_api_explanations.py`에 있는 공통 registry에서 생성됩니다.
생성된 `.ipynb`에 같은 설명을 여러 번 직접 붙이면 다음 재생성 때 사라질 수 있습니다.
새 API를 보강한 뒤에는 Windows 저장소 루트에서 다음 검사를 실행합니다.

```powershell
.\.venv\Scripts\python.exe tools\apply_api_explanations.py
.\.venv\Scripts\python.exe tools\build_paper_reproductions.py
.\.venv\Scripts\python.exe tools\build_field_reproductions.py
.\.venv\Scripts\python.exe tools\validate_api_explanations.py
```

설명에는 함수 이름만 번역하지 말고 signature, 입력·반환값, 현재 셀에서 선택한 이유,
수치·shape·gradient·device 관련 주의점과 공식 문서 링크를 함께 적습니다. 실습본에는 정답
코드가 노출되지 않도록 라이브러리 계약만 설명합니다.

## 15. Jupyter Markdown에서 수식을 읽고 쓰는 법

Jupyter Notebook의 Markdown 셀은 MathJax를 이용해 LaTeX 문법의 수식을 렌더링합니다.
수식을 코드처럼 한 줄로만 적기보다, **본문 안의 짧은 수식**과 **독립된 핵심 수식**을
구분하면 논문의 전개를 훨씬 쉽게 따라갈 수 있습니다.

### 인라인 수식과 블록 수식

문장 안에서 기호를 짧게 설명할 때는 `$...$`를 사용합니다.

```markdown
예측값 $\hat{y}_i$와 정답 $y_i$의 차이를 오차라고 합니다.
```

위 Markdown은 다음과 같이 보입니다.

예측값 $\hat{y}_i$와 정답 $y_i$의 차이를 오차라고 합니다.

논문의 핵심 식이나 여러 항의 관계를 보여줄 때는 빈 줄 사이에 `$$...$$`를 둡니다.

```markdown
평균 제곱 오차는 다음과 같습니다.

$$
\mathcal{L}_{\mathrm{MSE}}
= \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$
```

렌더링 결과는 다음과 같습니다.

$$
\mathcal{L}_{\mathrm{MSE}}
= \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

수식이 여러 줄이면 `aligned` 환경으로 등호나 설명의 시작점을 맞춥니다.

$$
\begin{aligned}
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t, \\
v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^2.
\end{aligned}
$$

다음 표는 이 프로젝트에서 자주 쓰는 LaTeX 표현입니다.

| 목적 | Markdown/LaTeX | 렌더링 예 |
|---|---|---|
| 아래 첨자 | `x_i`, `x_{ij}` | $x_i$, $x_{ij}$ |
| 위 첨자·제곱 | `x^2`, `e^{z_i}` | $x^2$, $e^{z_i}$ |
| 분수 | `\frac{a}{b}` | $\frac{a}{b}$ |
| 합 | `\sum_{i=1}^{N}` | $\sum_{i=1}^{N}$ |
| 제곱근 | `\sqrt{d_k}` | $\sqrt{d_k}$ |
| 절댓값·노름 | `\lvert x\rvert`, `\lVert x\rVert_2` | $\lvert x\rvert$, $\lVert x\rVert_2$ |
| 소속·실수 공간 | `x \in \mathbb{R}^D` | $x \in \mathbb{R}^D$ |
| 편미분·기울기 | `\frac{\partial L}{\partial x}`, `\nabla_\theta L` | $\frac{\partial L}{\partial x}$, $\nabla_\theta L$ |
| 근사·비례 | `\approx`, `\propto` | $\approx$, $\propto$ |
| 조건부 정의 | `\begin{cases}...\end{cases}` | $f(x)=\begin{cases}x&x>0\\0&x\leq0\end{cases}$ |

수식을 백틱 한 개로 감싼 `` `$x$` ``는 코드로 표시되어 렌더링되지 않습니다. 설명용
Markdown 셀에서는 `$x$`를 사용하고, Python 셀이나 함수 이름은 `` `torch.softmax` ``처럼
백틱으로 구분합니다. GitHub와 Jupyter에서 모두 안정적으로 보이게 하려면 블록 수식의
`$$` 앞뒤에 빈 줄을 두고, 한 수식 안에서 `$`와 `$$`를 섞지 않는 편이 좋습니다.

## 16. 수식 기호와 tensor shape를 먼저 해석하기

같은 문자도 논문마다 뜻이 달라질 수 있으므로 **논문이 정의한 기호가 최우선**입니다.
다만 이 저장소에서는 가능하면 다음 관례를 사용합니다.

| 기호 | 일반적인 뜻 | 코드에서 자주 보이는 이름 |
|---|---|---|
| $N$ 또는 $B$ | sample 수 또는 batch 크기 | `num_samples`, `batch_size` |
| $T$ 또는 $L$ | sequence 길이·token 수 | `seq_len`, `num_tokens` |
| $D$, $d_{\mathrm{model}}$ | feature 또는 embedding 차원 | `feature_dim`, `d_model` |
| $C$ | class 수 | `num_classes` |
| $H$ | attention head 수 | `num_heads` |
| $d_k$, $d_v$ | head 하나의 key/value 차원 | `head_dim` |
| $x_i$ | $i$번째 입력 | `inputs[i]` |
| $y_i$, $\hat{y}_i$ | 정답과 예측 | `targets[i]`, `predictions[i]` |
| $z_i$ | 정규화 전 점수인 logit | `logits[..., i]` |
| $\theta$ | 학습할 전체 parameter | `model.parameters()` |
| $\mathcal{L}$ 또는 $L$ | 최소화할 loss | `loss` |
| $g_t=\nabla_\theta\mathcal{L}_t$ | $t$번째 step의 gradient | `parameter.grad` |
| $\eta$ | learning rate | `lr` |
| $\varepsilon$ | 0으로 나누기를 막는 작은 수 | `eps` |

### shape 표기와 행렬 곱

$X\in\mathbb{R}^{B\times T\times D}$는 숫자 하나가 아니라 다음 구조의 tensor입니다.

- $B$: 한 번에 처리하는 문장 수
- $T$: 문장 하나의 token 수
- $D$: token 하나를 표현하는 feature 수

분류 가중치가 $W\in\mathbb{R}^{D\times C}$이고 bias가
$b\in\mathbb{R}^{C}$라면 다음 연산의 shape는 수식만으로도 추적할 수 있습니다.

$$
\underbrace{X}_{[B,T,D]}
\underbrace{W}_{[D,C]}
+\underbrace{b}_{[C]}
=\underbrace{Z}_{[B,T,C]}
$$

행렬 곱에서는 맞닿은 $D$가 같아야 하며 결과에는 바깥쪽 축 $[B,T,C]$가 남습니다.
$b$는 앞의 두 축으로 복제해 더하는 것처럼 동작하는 **broadcasting**을 사용합니다.

```python
logits = hidden_states @ weight + bias
# hidden_states: [batch, tokens, d_model]
# weight:        [d_model, classes]
# bias:          [classes]
# logits:        [batch, tokens, classes]
```

`dim=-1`은 마지막 축을 뜻하므로 `torch.softmax(logits, dim=-1)`은 각 token의 $C$개
class 점수를 확률로 바꿉니다. 반대로 `dim=0`을 사용하면 서로 다른 batch sample을 함께
정규화할 수 있습니다. 수식에 합 기호 $\sum_c$가 있으면 코드에서는 어느 `dim`이 class
축인지 먼저 찾는 습관이 중요합니다.

## 17. 수식을 코드로 옮길 때 확인할 여섯 단계

논문 수식을 만나면 아래 순서로 읽으면 구현 실수를 크게 줄일 수 있습니다.

1. **출력 한 줄 요약**: 이 식이 scalar loss, 확률 벡터, 새로운 tensor 중 무엇을 만드는가?
2. **기호 정의**: 각 문자와 아래 첨자가 sample·token·class·시간 step 중 무엇인가?
3. **shape 계약**: 각 항의 입력 shape와 출력 shape는 무엇이며 어떤 축이 줄어드는가?
4. **연산 순서**: 괄호, 지수, 합, 평균, 정규화 중 무엇을 먼저 계산하는가?
5. **안정성 장치**: $\varepsilon$, log-sum-exp, 최대값 빼기, masking이 왜 필요한가?
6. **학습 연결**: 미분 가능한가, gradient가 어느 parameter까지 전달되는가?

아래 예제들은 모두 **직관 → 수식 → 코드 대응 → 작은 손계산 → 주의점** 순서로 설명합니다.

## 18. 근사 비교: `np.allclose`

### 직관과 수식

부동소수점 두 값이 완전히 같은지를 묻는 대신, 두 값의 차이가 허용 오차보다 작은지를
검사합니다. 배열의 각 원소 $a_i$, $b_i$가 다음 조건을 만족해야 합니다.

$$
\lvert a_i-b_i\rvert
\leq
\mathrm{atol}+\mathrm{rtol}\lvert b_i\rvert
$$

배열 전체 결과는 각 원소 조건의 논리곱입니다.

$$
\operatorname{allclose}(a,b)
=\bigwedge_i
\left(
\lvert a_i-b_i\rvert
\leq \mathrm{atol}+\mathrm{rtol}\lvert b_i\rvert
\right)
$$

### 작은 손계산

$a_i=1.000009$, $b_i=1.0$, `rtol=1e-5`, `atol=1e-8`이면 다음과 같습니다.

$$
\begin{aligned}
\text{실제 차이} &= \lvert1.000009-1.0\rvert=9\times10^{-6},\\
\text{허용 오차} &= 10^{-8}+10^{-5}\lvert1.0\rvert=1.001\times10^{-5}.
\end{aligned}
$$

실제 차이가 허용 오차보다 작으므로 이 원소는 `True`입니다. 하지만 배열 원소 중 하나라도
조건을 넘으면 `np.allclose`의 최종 반환값은 `False`입니다.

```python
is_close_per_element = np.isclose(a, b, rtol=1e-5, atol=1e-8)
all_are_close = np.allclose(a, b, rtol=1e-5, atol=1e-8)
```

`np.isclose`는 원소별 Boolean 배열을, `np.allclose`는 그 결과를 모두 묶은 Boolean 하나를
반환합니다. $b_i$를 상대 오차의 기준으로 사용하므로 식은 일반적으로 대칭이 아닙니다.
0에 가까운 값에서는 상대 오차 항도 작아지므로 데이터 단위에 알맞은 `atol`을 직접 정해야
합니다.

## 19. 평균 제곱 오차: MSE

### 직관과 수식

MSE(mean squared error)는 예측과 정답의 차이를 제곱하고 평균 냅니다. 큰 오차가 제곱으로
더 크게 벌점을 받기 때문에 회귀에서 자주 사용합니다.

$$
\mathcal{L}_{\mathrm{MSE}}
=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

| 기호 | 뜻 |
|---|---|
| $N$ | 평균에 포함된 원소 수 |
| $y_i$ | $i$번째 정답 |
| $\hat{y}_i$ | $i$번째 예측 |
| $\mathcal{L}_{\mathrm{MSE}}$ | 평균된 scalar loss |

예측 $\hat{y}=[1,1]$, 정답 $y=[2,0]$이면 두 오차는 $-1$, $1$입니다.

$$
\mathcal{L}_{\mathrm{MSE}}
=\frac{(-1)^2+(1)^2}{2}=1
$$

각 예측에 대한 gradient는 다음과 같습니다.

$$
\frac{\partial\mathcal{L}_{\mathrm{MSE}}}{\partial\hat{y}_i}
=\frac{2}{N}(\hat{y}_i-y_i)
$$

따라서 예측이 정답보다 크면 gradient가 양수이고, gradient descent는 예측을 낮추는 방향으로
parameter를 움직입니다.

```python
loss = torch.nn.functional.mse_loss(
    predictions,
    targets,
    reduction="mean",
)
```

`reduction="mean"`은 모든 출력 원소까지 평균하므로 batch 크기뿐 아니라 출력 차원도
분모에 포함될 수 있습니다. sample별 loss가 필요하면 `reduction="none"`으로 원소별 오차를
받은 뒤 의도한 축에서 직접 평균합니다. MSE 계약은
[PyTorch `mse_loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html)를
참고하세요.

## 20. Softmax와 교차 엔트로피

### Softmax: 점수를 확률 형태로 바꾸기

class별 logit $z_i$를 양수이면서 합이 1인 값 $p_i$로 바꿉니다.

$$
p_i=\operatorname{softmax}(z)_i
=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
$$

logit $z=[2,1,0]$의 경우 약 $p=[0.6652,0.2447,0.0900]$이 됩니다. 가장 큰 logit의
class가 가장 큰 확률을 가지며, 전체 합은 1입니다.

실제 구현은 큰 지수 때문에 overflow가 생기지 않도록 $m=\max_j z_j$를 먼저 뺍니다.

$$
p_i
=\frac{e^{z_i-m}}{\sum_{j=1}^{C}e^{z_j-m}}
$$

모든 logit에서 같은 값을 빼도 softmax 결과는 변하지 않습니다. `dim=-1`은 분모의 합을
계산할 class 축을 지정합니다.

```python
probabilities = torch.softmax(logits, dim=-1)
```

### 교차 엔트로피: 정답 class에 높은 확률 주기

one-hot 정답 $y_i$를 쓰면 한 sample의 교차 엔트로피는 다음과 같습니다.

$$
\mathcal{L}_{\mathrm{CE}}
=-\sum_{i=1}^{C}y_i\log p_i
$$

정답 class index가 $c$이면 $y_c=1$이고 나머지는 0이므로 더 간단해집니다.

$$
\mathcal{L}_{\mathrm{CE}}=-\log p_c
$$

앞의 $p=[0.6652,0.2447,0.0900]$에서 정답이 첫 번째 class라면
$-\log(0.6652)\approx0.4076$입니다. 정답 확률이 1에 가까우면 loss는 0에 가까워지고,
정답 확률이 0에 가까우면 loss가 매우 커집니다.

PyTorch는 softmax와 log를 따로 계산하는 대신 다음과 같은 log-sum-exp 형태를 내부적으로
사용해 수치 안정성을 높입니다.

$$
\mathcal{L}_{\mathrm{CE}}
=-z_c+\log\left(\sum_{j=1}^{C}e^{z_j}\right)
$$

```python
loss = torch.nn.functional.cross_entropy(logits, targets)
```

따라서 `cross_entropy`에 넘기기 전에 `softmax`를 적용하지 않습니다. 일반적인 class-index
정답에서 `logits.shape == [B, C]`, `targets.shape == [B]`이며 targets의 dtype은
`torch.long`입니다. 확률은 해석이나 시각화가 필요할 때 별도로 계산합니다.

## 21. Cosine similarity와 정규화

### Cosine similarity

두 벡터의 길이보다는 **방향이 얼마나 비슷한지**를 측정합니다.

$$
\operatorname{cos}(x,y)
=\frac{x^\top y}{\lVert x\rVert_2\lVert y\rVert_2}
$$

$x=[1,1]$, $y=[1,0]$이면 $x^\top y=1$, $\lVert x\rVert_2=\sqrt{2}$,
$\lVert y\rVert_2=1$이므로 다음과 같습니다.

$$
\operatorname{cos}(x,y)=\frac{1}{\sqrt{2}}\approx0.7071
$$

일반적인 실수 벡터의 결과 범위는 $[-1,1]$입니다. 1은 같은 방향, 0은 직교, -1은 반대
방향을 뜻합니다. 영벡터는 방향이 없고 분모가 0이 되므로 라이브러리는 작은
$\varepsilon$을 사용하거나 사용자가 영벡터를 별도로 처리해야 합니다.

```python
similarity = torch.nn.functional.cosine_similarity(
    query_embeddings,
    document_embeddings,
    dim=-1,
    eps=1e-8,
)
```

자세한 인자와 broadcast 규칙은
[PyTorch `cosine_similarity`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html)를
참고하세요.

### 서로 다른 세 종류의 정규화

“normalize”라는 말은 문맥에 따라 다른 연산을 뜻합니다.

1. **L2 vector normalization**은 벡터 길이를 1로 만듭니다.

   $$
   \tilde{x}=\frac{x}{\max(\lVert x\rVert_2,\varepsilon)}
   $$

   L2 정규화된 두 벡터에서는 내적 $\tilde{x}^\top\tilde{y}$가 cosine similarity와
   같습니다. embedding 검색에서 행렬 곱으로 cosine 점수를 빠르게 계산할 때 유용합니다.

   ```python
   normalized = torch.nn.functional.normalize(
       embeddings,
       p=2,
       dim=-1,
       eps=1e-12,
   )
   ```

2. **표준화(z-score)**는 feature별 평균을 0, 표준편차를 1에 가깝게 만듭니다.

   $$
   \tilde{x}_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j+\varepsilon}
   $$

   train data에서 구한 $\mu_j$, $\sigma_j$를 validation과 test에도 그대로 사용해야 합니다.
   각 split에서 다시 통계를 구하면 데이터 누수가 됩니다.

3. **Layer Normalization**은 sample의 지정된 feature 축에서 평균과 분산을 구한 뒤 학습
   가능한 scale $\gamma$와 shift $\beta$를 적용합니다.

   $$
   \operatorname{LayerNorm}(x_i)
   =\gamma\odot\frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}}+\beta
   $$

   Transformer에서는 보통 token별 마지막 embedding 축을 정규화합니다. L2 정규화처럼
   벡터 길이를 1로 만드는 연산이 아니며, 입력 전체 dataset의 통계를 저장하는 표준화와도
   다릅니다. 공식 계약은 [PyTorch `normalize`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)와
   [`LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)을
   참고하세요.

## 22. Scaled dot-product attention

### 핵심 수식

Transformer의 attention은 query와 key의 유사도로 value를 가중합합니다.

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
$$

| 항 | 역할 | 대표 shape |
|---|---|---|
| $Q$ | 현재 위치가 찾고 싶은 정보 | $[B,H,T_q,d_k]$ |
| $K$ | 각 위치가 가진 정보의 검색 key | $[B,H,T_k,d_k]$ |
| $V$ | 실제로 섞어 가져올 내용 | $[B,H,T_k,d_v]$ |
| $QK^\top$ | query-key 유사도 점수 | $[B,H,T_q,T_k]$ |
| $1/\sqrt{d_k}$ | 차원이 커질 때 내적 크기가 커지는 것을 완화 | scalar |
| $M$ | padding 또는 미래 token을 가리는 mask | 점수에 broadcast 가능한 shape |
| 최종 출력 | attention-weighted value | $[B,H,T_q,d_v]$ |

$M$은 허용된 위치에는 0, 차단할 위치에는 매우 작은 값 또는 $-\infty$를 더합니다.
softmax 후 차단 위치의 확률은 0이 됩니다. causal language model은 미래 token을 볼 수 없도록
위쪽 삼각 영역을 가립니다.

### 작은 손계산

head 하나, $d_k=1$이고 다음과 같다고 가정합니다.

$$
Q=K=\begin{bmatrix}1\\0\end{bmatrix},\qquad
V=\begin{bmatrix}10&0\\0&10\end{bmatrix}
$$

scale은 $\sqrt{1}=1$이고 score와 행별 softmax는 다음과 같습니다.

$$
QK^\top
=\begin{bmatrix}1&0\\0&0\end{bmatrix},\qquad
A\approx
\begin{bmatrix}0.7311&0.2689\\0.5&0.5\end{bmatrix}
$$

따라서 출력은 다음처럼 value의 가중 평균이 됩니다.

$$
AV\approx
\begin{bmatrix}7.311&2.689\\5&5\end{bmatrix}
$$

첫 번째 query는 첫 번째 key와 더 비슷해서 첫 value에 더 큰 가중치를 줍니다. 두 번째 query의
score는 같으므로 두 value를 절반씩 섞습니다.

```python
scores = query @ key.transpose(-2, -1)
scores = scores / math.sqrt(query.size(-1))
scores = scores.masked_fill(~attention_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
context = weights @ value
```

`transpose(-2, -1)`은 key의 token 축과 feature 축을 맞바꿔 $QK^\top$를 만들고,
마지막 softmax 축은 **각 query가 바라보는 key 위치**입니다. PyTorch의 최적화된 구현은
[`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)을
참고하세요.

## 23. Adam과 AdamW optimizer

Adam은 gradient의 지수 이동 평균과 gradient 제곱의 지수 이동 평균을 함께 사용합니다.
$t$번째 step에서 $g_t=\nabla_\theta\mathcal{L}_t$라 두면 다음과 같습니다.

$$
\begin{aligned}
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t},\\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t},\\
\theta_t &= \theta_{t-1}
-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}.
\end{aligned}
$$

| 기호 | 역할 | 흔한 기본값 |
|---|---|---|
| $m_t$ | gradient의 1차 moment 이동 평균 | $m_0=0$ |
| $v_t$ | gradient 제곱의 2차 moment 이동 평균 | $v_0=0$ |
| $\beta_1$ | 1차 moment의 과거 정보 유지율 | $0.9$ |
| $\beta_2$ | 2차 moment의 과거 정보 유지율 | $0.999$ |
| $\hat{m}_t$, $\hat{v}_t$ | 초기값 0의 편향을 보정한 moment | — |
| $\eta$ | learning rate | 문제마다 조정 |
| $\varepsilon$ | 0으로 나누기 방지와 수치 안정성 | 구현별 기본값 확인 |

예를 들어 첫 step에서 $g_1=0.2$, $\beta_1=0.9$, $\beta_2=0.999$이면
$m_1=0.02$, $v_1=0.00004$입니다. 편향을 보정하면
$\hat{m}_1=0.2$, $\hat{v}_1=0.04$가 됩니다. $\eta=0.001$이고
$\varepsilon$이 매우 작다면 parameter 변화량의 크기는 약 $0.001$입니다.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

`loss.backward()`가 $g_t$를 만들고 `optimizer.step()`이 moment와 parameter를 갱신합니다.
`zero_grad()`는 Adam 수식의 일부가 아니라 다음 mini-batch의 gradient가 이전 gradient에
의도치 않게 누적되는 것을 막는 PyTorch 단계입니다.

AdamW는 weight decay를 gradient 기반 Adam 갱신과 분리해 적용합니다. 따라서 같은
`weight_decay` 값이라도 Adam의 L2 penalty와 AdamW의 decoupled weight decay는 일반적으로
같은 경로를 만들지 않습니다. bias와 LayerNorm parameter에는 decay를 적용하지 않도록
parameter group을 나누는 구현도 많습니다. 공식 인자는
[PyTorch `Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)과
[`AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)를 참고하세요.

## 24. 논문 수식과 구현을 함께 설명하는 Markdown 템플릿

새 해설을 작성할 때는 다음 형식을 복사해 논문과 코드에 맞게 채웁니다. 아래 블록은
**복사용 원문**이므로 수식이 렌더링되지 않는 것이 정상입니다. 실제 Markdown 셀에 붙여
넣으면 `$...$`와 `$$...$$` 부분이 수식으로 표시됩니다.

````markdown
### [수식 또는 모듈 이름]

**한 문장 직관**

이 식이 무엇을 계산하고 왜 필요한지 비전공자도 이해할 수 있는 한 문장으로 적습니다.

**논문의 위치와 역할**

- 논문: [논문 제목]
- 위치: Section X.X, Equation (N), Figure/Table N
- 앞 단계에서 받는 값과 다음 단계로 보내는 값을 적습니다.

**핵심 수식**

$$
[LaTeX 수식]
$$

| 기호 | 뜻 | shape 또는 범위 |
|---|---|---|
| $x$ | 입력 | $[B,T,D]$ |
| $\theta$ | 학습 parameter | 구현에 따라 다름 |
| $y$ | 출력 | $[B,T,C]$ |

**shape 흐름**

$$
[B,T,D]\;\longrightarrow\;[B,T,C]
$$

**코드 대응**

```python
# 수식의 각 항이 어느 변수인지 주석으로 연결합니다.
output = module(inputs)
```

**작은 수치 예제**

2~3개 원소로 직접 계산해 중간값과 최종값을 확인합니다.

**왜 이렇게 구현하는가?**

- 논문 수식을 그대로 옮긴 부분
- broadcasting·vectorization 등 효율을 위해 바꾼 표현
- log-sum-exp·epsilon 등 수치 안정성을 위해 추가한 부분

**주의점과 실패하기 쉬운 경우**

- 잘못 선택하기 쉬운 `dim`/`axis`
- dtype, device, mask, padding, gradient 관련 조건
- mean/sum reduction과 batch 크기에 따른 scale 변화

**직접 확인할 실험**

1. 입력 shape를 바꾸었을 때 출력 shape를 예측합니다.
2. 핵심 인자를 한 개 바꾸고 중간값이 어떻게 달라지는지 관찰합니다.
3. 작은 입력에서 손계산과 코드 결과를 `allclose`로 비교합니다.
````

Markdown 안에 Python fence를 포함하는 템플릿을 그대로 문서화할 때는 바깥 fence와 안쪽
fence의 길이를 다르게 해야 합니다. 개별 노트북에서는 템플릿 전체를 fence로 감싸지 않고
제목·수식·코드 셀로 나누면 더 읽기 좋습니다.

## 25. 부가 설명을 풍부하게 만들되 읽기 어렵지 않게 하는 기준

설명이 많을수록 좋은 것은 **필요할 때 단계적으로 펼쳐 읽을 수 있을 때**입니다. 각 논문
노트북에서는 다음 우선순위를 권장합니다.

1. 셀 앞에는 핵심 직관, 입력·출력 shape, 논문 section/equation 대응을 먼저 둡니다.
2. 핵심 수식은 렌더링된 블록으로 보여 주고, 바로 아래에 모든 기호를 표로 풉니다.
3. 구현 셀에는 수식의 항과 변수 이름을 연결하는 짧은 주석을 둡니다.
4. 긴 유도 과정, 대안 구현, 역사적 배경은 접이식 `<details>` 영역에 둡니다.
5. 손계산 가능한 작은 예제와 실제 mini-batch 실행 예제를 분리합니다.
6. 마지막에는 “무엇을 바꾸어 볼 것인가”와 예상 결과를 질문으로 제공합니다.

특히 다음 부가 정보는 포트폴리오와 복습 모두에 도움이 됩니다.

- 논문에서 제시한 원래 식과 교육용 구현에서 단순화한 부분
- 원 논문 실험 규모와 로컬 GPU용 축소 실험의 차이
- 계산 복잡도와 주요 tensor의 메모리 사용량
- 학습이 실패할 때 확인할 loss, gradient norm, learning rate, mask, dtype
- 같은 목적의 다른 함수나 최신 기법과의 차이
- 결과를 재현할 때 고정해야 할 seed, data split, library version

수식 자체를 외우는 것이 목표가 아닙니다. **기호의 의미, shape 변화, 코드 변수, 수치 안정성,
gradient 경로를 한 묶음으로 설명할 수 있는 상태**가 논문 구현을 이해했다는 가장 실용적인
기준입니다.
