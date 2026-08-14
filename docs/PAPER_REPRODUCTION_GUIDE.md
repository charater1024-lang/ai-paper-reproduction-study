# 논문 재현 트랙 학습 가이드

이 문서는 [유명 AI 논문 20편 미니 재현 트랙](../notebooks/paper_reproductions/README.md)을
실습본/정답본 페어로 사용하는 방법을 설명합니다. 목표는 코드를 외우는 것이 아니라 논문의 주장,
수식, 구현, 관찰 증거를 한 줄로 연결하는 것입니다.

## 이 트랙의 학습 계약

각 논문은 같은 파일명의 두 노트북으로 제공됩니다.

- `notebooks/paper_reproductions/exercises/`: TODO가 남아 있는 실습본
- `notebooks/paper_reproductions/solutions/`: 같은 셀 위치의 실행 가능한 정답본

두 파일은 cell ID와 설명 순서가 같습니다. 달라지는 부분은 주로 구현 과제 코드와 마지막 회고입니다.
따라서 화면을 좌우로 놓고 현재 셀끼리 비교할 수 있습니다. 정답본은 복사할 답안지가 아니라 막혔을
때 확인하는 작은 단위의 참고 구현입니다.

## 시작과 페어 학습법

프로젝트 루트에서 다음 명령을 실행합니다.

```bat
Start_Paper_Reproductions.cmd
```

특정 논문을 바로 열려면 번호를 덧붙입니다.

```bat
Start_Paper_Reproductions.cmd 09
```

권장하는 한 논문 학습 순서는 다음과 같습니다.

1. **범위를 먼저 읽습니다.** 헤더의 원문 링크, 미니 재현 목표, 원 논문과 다른 scale을 확인합니다.
2. **mapping 표를 읽습니다.** 아직 코드를 실행하지 말고 해당 수식의 입력·출력 shape를 예상합니다.
3. **공유 setup 셀을 실행합니다.** 합성 데이터의 분포, dtype, label 규칙을 눈으로 확인합니다.
4. **실습본 TODO를 구현합니다.** 정답본은 닫아 두고 함수 계약과 `assert`부터 맞춥니다.
5. **작은 단위로 비교합니다.** 10분 이상 막히면 정답 셀 전체가 아니라 필요한 3~5줄만 봅니다.
6. **검증을 통과시킵니다.** shape, 확률 합, mask 방향, gradient, 유한 loss 등의 assertion을 설명합니다.
7. **metric을 해석합니다.** 값이 좋아졌다는 사실보다 어떤 논문 메커니즘 때문에 변했는지 적습니다.
8. **변수 하나만 바꿉니다.** seed를 유지하고 한 요소만 바꾼 뒤 아래 템플릿으로 결과를 남깁니다.

정답과 다른 코드라도 같은 수학적 계약과 검증을 만족하면 좋은 답입니다. 반대로 출력만 우연히
비슷하고 shape·mask·gradient의 의미를 설명하지 못하면 아직 완료한 것이 아닙니다.

## 수식·section mapping 읽기

각 노트북 앞부분의 mapping 표는 세 열로 되어 있습니다.

| 열 | 읽는 질문 |
|---|---|
| 논문의 위치 | 어느 section, equation, figure, algorithm의 주장인가? |
| 노트북의 대응 구현 | 그 기호가 어떤 함수, tensor, layer로 바뀌었는가? |
| 확인할 증거 | 어떤 assertion, metric, plot이 구현 오류를 드러내는가? |

표를 읽을 때는 다음 순서를 권장합니다.

1. 원문의 section 제목과 수식 번호를 함께 찾습니다. PDF 판본에 따라 페이지가 달라도 제목과 번호는
   비교적 안정적입니다.
2. 수식의 각 기호 옆에 코드 이름을 적습니다. 예: `Q,K,V → q,k,v`, `p_data → real_samples`.
3. 축소하면서 유지한 조건과 바꾼 조건을 나눕니다. 예를 들어 attention의 `1/√d_k`는 유지하지만
   layer 수, token 수, dataset은 줄일 수 있습니다.
4. plot을 장식으로 보지 말고 반증 도구로 읽습니다. causal mask의 위쪽 삼각형이 열리거나 확률 합이
   1이 아니면, loss가 감소하더라도 구현은 잘못된 것입니다.
5. 노트북 출력과 논문 결과표를 직접 등치하지 않습니다. 미니 재현의 metric은 메커니즘 검사용입니다.

논문 버전마다 수식 번호가 바뀌었다면 section 제목과 수식의 첫 항을 함께 기록하세요. 단순히
“논문 3쪽”이라고 적는 것보다 재확인하기 쉽습니다.

## 하드웨어 인식·no-download 설계

이 트랙의 노트북은 다음 제약을 의도적으로 지킵니다.

- 신경망 학습은 CUDA → Apple MPS → CPU를 자동 선택하며 GPU가 없어도 됩니다.
- 데이터는 셀 안에서 생성한 작은 synthetic dataset입니다.
- seed를 고정해 같은 환경에서 assertion과 추세를 재현할 수 있게 합니다.
- 노트북 안에서 `pip`, `wget`, dataset download 또는 외부 API를 호출하지 않습니다.
- 학습 step, 해상도, vocabulary, channel 수를 줄여 핵심 계산만 남깁니다.
- 저장소에 commit된 노트북에는 실행 output을 남기지 않습니다.

의존성 설치 자체는 최초 환경 준비 때 필요할 수 있지만, 준비가 끝난 뒤 각 노트북은 네트워크 없이
실행됩니다. 작은 합성 데이터는 실제 데이터의 다양성이나 benchmark 난도를 대표하지 않습니다.

## 검증 명령

아래 명령은 프로젝트 루트의 CMD 또는 PowerShell에서 실행합니다.

페어 구조, cell ID, TODO, 메타데이터, 금지된 다운로드 코드, Python 문법만 빠르게 검사합니다.

```bat
.venv\Scripts\python.exe tools\validate_paper_reproductions.py
.venv\Scripts\python.exe tools\validate_portfolio_quality.py --track paper
```

논문 하나의 정답본까지 실제 실행합니다.

```bat
.venv\Scripts\python.exe tools\validate_paper_reproductions.py --only 09 --execute-solutions
```

20개 정답본을 모두 실행합니다. 메모리가 넉넉하지 않으면 `--jobs 1`을 유지하세요.

```bat
.venv\Scripts\python.exe tools\validate_paper_reproductions.py --execute-solutions --jobs 2
```

각 code cell의 기본 제한 시간은 240초입니다. 느린 CPU에서만 명시적으로 늘립니다.

```bat
.venv\Scripts\python.exe tools\validate_paper_reproductions.py --only 18 --execute-solutions --timeout 480
```

명세 파일을 수정해 노트북 페어를 다시 만들 때만 다음 생성 명령을 사용합니다. 이 명령은 생성된
실습본과 정답본을 덮어쓰므로, 학습 중 작성한 답은 먼저 별도 파일이나 Git diff로 보존하세요.

```bat
.venv\Scripts\python.exe tools\build_paper_reproductions.py
```

## 실험 기록 템플릿

아래 블록을 노트북 마지막이나 별도 실험 노트에 복사해 사용하세요. 한 실험에서는 독립 변수 하나만
바꾸는 것이 핵심입니다.

```markdown
## 실험: <논문 번호 / 짧은 이름>

- 날짜:
- 원문 위치: <section 제목, equation/figure 번호>
- 확인할 주장:
- 기준 seed:
- 환경: CPU / Python 버전 / PyTorch 버전

### 가설
<변수를 어떻게 바꾸면 어떤 관찰값이 왜 변할지 한 문장>

### 통제와 변경

| 항목 | 기준 실험 | 변경 실험 |
|---|---|---|
| 독립 변수 | | |
| 고정한 조건 | | |
| 반복 횟수 | | |

### 결과

| metric / 관찰 | 기준 | 변경 | 차이 |
|---|---:|---:|---:|
| | | | |

### 해석
- 논문의 주장과 일치한 점:
- 일치하지 않거나 판단할 수 없는 점:
- mini scale 때문에 생긴 가능한 교란 요인:
- 다음에 바꿀 변수 하나:
```

loss 한 개만 기록하지 말고 논문 메커니즘에 맞는 보조 증거도 남기세요. 예를 들어 attention은
mask와 row sum, BatchNorm은 feature별 mean/variance, GAN은 두 loss와 sample 분포, CLIP은
similarity matrix의 대각선과 retrieval accuracy를 함께 보는 편이 좋습니다.

## 한계와 올바른 해석

이 자료로 확인할 수 있는 것은 “핵심 연산이 이 작은 조건에서 예상한 불변식과 추세를 보이는가”입니다.
다음은 확인할 수 없습니다.

- 원 논문의 benchmark 수치와 통계적 유의성
- 대규모 데이터에서의 일반화, robustness, fairness
- 원래 optimizer schedule과 장시간 학습의 효과
- 여러 GPU/TPU에서의 처리량과 분산 학습 특성
- 논문에 보고된 ablation 전체 또는 후속 연구의 결론

한 seed의 loss 감소를 “논문의 주장을 입증했다”고 쓰지 마세요. 대신 “합성 데이터와 이 seed에서
해당 연산의 예상 shape/gradient/metric 추세를 확인했다”고 범위를 정확히 적습니다. 결과가 기대와
다르더라도 assertion, scale 차이, initialization을 분석했다면 유효한 재현 학습입니다.

## 원문 인용 원칙

1. 각 노트북 헤더의 primary paper 링크를 먼저 사용합니다.
2. 아이디어나 결과를 설명할 때 저자, 논문명, 발표 연도와 section/equation/figure를 함께 적습니다.
3. 원문의 주장과 자신의 미니 실험 관찰을 문장에서 분리합니다.
4. 직접 인용은 짧게 따옴표로 표시하고 판본의 페이지 또는 section을 기록합니다. 가능하면 자신의
   말로 정확히 요약합니다.
5. 블로그나 구현 문서는 보조 자료로만 쓰고, 논문의 수식·실험 결과를 뒷받침하는 근거는 원문에서
   확인합니다.
6. 원 논문과 다르게 단순화한 부분—synthetic data, channel 수, loss proxy, 평가 metric—을 숨기지
   않습니다.
7. 다른 저장소의 코드를 사용했다면 해당 라이선스와 출처를 별도로 기록합니다. 논문 인용만으로
   구현 코드의 라이선스 의무가 대체되지는 않습니다.

좋은 기록의 문장 구조는 다음과 같습니다.

> 논문 §3.2의 scaled dot-product attention에는 `1/√d_k`가 있다. 노트북에서는 logits를 scaling한
> 경우와 하지 않은 경우의 entropy를 비교했다. 이 작은 합성 실험에서는 scaling한 쪽의 entropy가
> 더 높았지만, 이는 원 논문의 번역 성능을 재현한 결과는 아니다.

이처럼 **원문의 주장 → 대응 코드 → 관찰 증거 → 해석 한계**를 한 묶음으로 남기는 것이 이 트랙의
최종 학습 목표입니다.
