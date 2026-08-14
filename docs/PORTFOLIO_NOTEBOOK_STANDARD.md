# 포트폴리오용 논문 재현 표준

이 문서는 논문 실습을 “결과가 나오는 짧은 예제”에서 “구현 결정을
설명할 수 있는 포트폴리오”로 발전시키기 위한 작성 계약입니다.

구성 깊이는 [Deep-Learning-Paper-Review-and-Practice](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice)의
단계별 논문 실습 방식을 참고했습니다. 코드나 설명을 복제하지 않고, 이 저장소의
Windows·로컬 데이터·exercise/solution·자동 검증 구조에 맞게 다시 설계합니다.

특히 참고 저장소의
[Transformer 실습 노트북](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_%28German_English%29.ipynb)처럼
데이터 준비, tensor shape 확인, layer 구현, 학습과 평가를 단계별로 분리하는 장점을
반영합니다. 여기에 이 저장소만의 한국어 수식 해설, TODO/정답 1:1 pairing, 로컬 합성
데이터, 장치 자동 선택과 회귀 validator를 더합니다.

## 논문 한 편의 필수 구성

### 1. 문제와 배경

- 이전 방법이 어디서 실패했는지
- 논문이 제안한 변화가 무엇인지
- 이 실습이 검증할 수 있는 주장과 검증할 수 없는 주장

### 2. 원문 위치

모든 핵심 셀은 원 논문의 절·식·그림·알고리즘 중 적어도 하나와 연결됩니다.

| 설명 항목 | 기록 예시 |
|---|---|
| 원문 위치 | `§3.2 Scaled Dot-Product Attention, Eq. (1)` |
| 실습 위치 | `Task 2 · scaled_dot_product_attention` |
| 검증 증거 | `attention weight 합=1`, `causal 위치=0` |
| 축소 범위 | `1 head, 짧은 합성 sequence` |

### 3. 수식·기호·코드 대응

수식을 스크린샷으로 붙이지 않고 Markdown LaTeX로 적습니다.

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

- $Q,K,V$: query·key·value tensor
- $d_k$: head 하나의 feature 차원
- $M$: padding 또는 causal mask
- 수식의 역할: dot product의 크기를 보정하고 허용된 위치만 가중 합산
- 코드 대응: projection → score → scale → mask → softmax → value aggregation

수식 아래에는 반드시 각 기호의 의미, tensor shape, 수치 안정성 이유와 관련
코드 위치를 적습니다.

### 4. 데이터 계약

- 원본 file의 schema·dtype·shape
- train/validation 분리 기준과 누수 방지
- normalization·mask·padding·augmentation 과정
- batch 하나의 실제 shape과 범위

외부 대형 데이터셋을 기본 실행 중 자동 다운로드하지 않습니다. 저장소에 포함된 작은
로컬 데이터로 핵심 계산을 먼저 재현합니다.

### 5. 직접 구현하는 컴포넌트

논문에서 제안한 핵심은 하나의 긴 셀이 아니라 이름이 드러나는 class와 함수로 만듭니다.

- `Config`: 재현 가능한 hyperparameter와 seed
- `Dataset` 또는 preprocessing 함수: raw input을 tensor로 변환
- 핵심 `Layer`/`Model`: `__init__` 구성과 `forward` 계산
- objective 함수: 논문 수식을 코드로 직접 표현
- `training_step`: forward, loss, backward, gradient 검사, optimizer step
- `evaluate`: metric·기준선·실패 예시를 분리해 계산

PyTorch의 고수준 layer를 사용해도 됩니다. 다만 논문의 제안 자체를 한 줄 helper 호출로
숨기지 않고, 그 안의 shape 변환·mask·손실·상태 변화를 코드에서 확인할 수 있게
합니다.

클래스는 줄 수를 늘리기 위한 장식물이 아닙니다. 노트북의 주 학습 경로가 해당 클래스의
`forward`·`compute_loss`·`training_step`·`evaluate`를 실제로 호출해야 합니다. 같은 계산을
별도의 절차형 loop에서 중복하고 마지막에 클래스의 설정값만 검사하는 구성은 허용하지
않습니다. 마지막 검증은 loss 변화, parameter update, gradient, metric 또는 핵심 상태 변화
중 논문의 주장과 직접 연결된 증거를 확인합니다.

exercise와 solution은 같은 공개 API를 사용합니다. exercise에는 solution의 class·함수·공개
메서드 이름과 매개변수 순서, docstring, 입력/출력 shape 계약을 그대로 보여 주고, 구현할
body만 TODO로 남깁니다. 따라서 학습자는 정답 파일의 숨겨진 함수 서명을 추측하지 않고
같은 경계 안에서 알고리즘을 완성할 수 있습니다.

자동 검사는 깊이를 완벽히 판단할 수 없지만, 지나치게 얇은 구현으로 되돌아가는 것을
막기 위해 다음 최소 계약을 둡니다.

| 트랙 | 정답 코드 | 명시적 class | 함수·메서드 |
|---|---:|---:|---:|
| 유명 논문 20편 | 비어 있지 않은 Python 100줄 이상 | 1개 이상 | 4개 이상 |
| 분야별 논문 70편 | 비어 있지 않은 Python 110줄 이상 | 1개 이상 | 7개 이상 |

줄 수를 채우기 위한 중복은 허용하지 않습니다. 핵심 layer, objective, update, evaluate를
역할별로 분리한 결과로 위 계약을 만족해야 합니다.

### 6. 구현 이유

각 핵심 컴포넌트 앞의 Markdown에 다음 질문에 답합니다.

1. 왜 이 class나 함수가 필요한가?
2. 이 계산은 논문의 어느 절·식에 해당하는가?
3. 다른 구현 방법 중 왜 이 방법을 선택했는가?
4. 이 축소 실습에서 유지하거나 생략한 가정은 무엇인가?

이 설명은 노트북 앞부분에 한 번만 모아 두지 않습니다. setup을 제외한 model, 핵심 연산,
loss, 학습, 평가 code cell 바로 앞에 다음 네 항목을 반복해 연결합니다.

- 원문의 정확한 절·식·그림·알고리즘 위치
- 해당 셀의 입력/출력 tensor shape
- 이 구현 방식을 선택한 이유
- 셀을 완료했다고 판단할 assertion·metric·상태 변화

### 7. 학습과 평가

학습 loop는 다음 순서를 생략하지 않습니다.

```text
batch 이동 → train mode → gradient 초기화 → forward → loss
→ backward → gradient/finite 검사 → optimizer step → history 기록
```

평가는 train loss 하나로 끝내지 않고 논문의 핵심 주장과 연결된 metric, 정상성
assertion, 간단한 실패 예시를 함께 보여 줍니다.

## 코드 가독성 규칙

- learner-facing notebook Python code는 한 줄 88자를 상한으로 둡니다.
- 생성기·검증기 등 저장소 내부 Python source도 한 줄 100자를 넘지 않습니다.
- semicolon으로 여러 문장을 한 줄에 연결하지 않습니다.
- `class`, `def`, `for`, `if`, `with`의 body를 한 줄에 압축하지 않습니다.
- 변수 이름은 수식 기호와 연결되되 역할을 알 수 있게 적습니다.
- 중요한 tensor 변환 전후에는 shape와 dtype 불변식을 둡니다.
- 설명은 코드를 다시 읽는 주석보다 선택 이유와 주의점을 담습니다.
- 생성 원본인 `*_specs.py`의 문자열 안 코드도 같은 규칙을 지킵니다. 생성 뒤 formatter가
  고쳐 준다는 이유로 source에 semicolon 압축 코드나 수백 자짜리 한 줄을 남기지 않습니다.

생성된 notebook code cell은 Ruff formatter를 거쳐 같은 표준을 유지합니다.

## 포트폴리오에 남길 것

노트북을 실행했다는 사실만으로는 충분하지 않습니다. 다음을 개인 실험 노트에
남기면 면접이나 코드 리뷰에서 구현을 설명하기 쉬워집니다.

1. 논문이 해결하려는 문제를 자신의 말로 세 문장
2. 핵심 수식과 코드 symbol 대응표
3. 구현 중 만난 shape·device·gradient bug와 해결 근거
4. 기준 설정과 한 가지 변수만 바꿔 ablation
5. metric 표와 적어도 하나의 실패 예시
6. 원 논문의 규모로 확장할 때 필요한 데이터·메모리·학습 예산

## “재현”의 범위

이 저장소는 대부분의 논문에서 원 데이터셋·모델 크기·학습 step·분산 학습을 축소합니다.
따라서 포트폴리오에는 “논문 성능을 완전 재현했다”가 아니라 다음과 같이 적습니다.

> 원 논문의 핵심 layer·objective·update rule을 PyTorch로 직접 구현하고, 작은
> 로컬 데이터에서 shape·gradient·상대 지표 방향을 검증했다. 원 규모 성능은 재현 범위에
> 포함하지 않았다.
