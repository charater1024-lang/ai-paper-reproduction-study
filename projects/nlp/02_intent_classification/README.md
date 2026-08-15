# 02. 고객 문의 의도 분류

이 실습에서는 고객 문의 문장을 다섯 가지 업무 의도로 자동 분류하는 기준선 모델을 만듭니다.
문자 n-gram TF-IDF, 범주형 메타데이터, 로지스틱 회귀를 하나의 scikit-learn `Pipeline`으로
묶고, 같은 고객 사례의 유사 문장이 학습과 검증 양쪽에 섞이지 않도록 `case_id` 단위로
분리합니다.

단순히 accuracy 숫자 하나를 높이는 것이 목표는 아닙니다. **데이터 누수 없이 평가하고,
클래스별 실패를 읽고, 새 문장 예측까지 같은 전처리로 재현하는 전체 흐름**을 익히는 것이
핵심입니다. API 키나 외부 모델 다운로드는 필요하지 않습니다.

## 공통 Python 실행 뼈대

`starter.py`와 `solution.py`를 읽기 전에 다음 공통 장치를 확인하세요.

- `Path(__file__)`은 현재 작업 폴더가 아니라 실행 파일을 기준으로 저장소 루트를 찾습니다.
- `sys.path.insert(0, ...)`는 로컬 `src`를 import 검색 경로 맨 앞에 둡니다.
- `@dataclass(frozen=True, slots=True)`는 설정 필드의 재할당과 임의 속성 추가를 막습니다.
- `argparse`의 `type`·`default`·`action`은 명령줄 입력을 해석하고, `parse_args()`는
  그 결과를 `Namespace`로 반환합니다.

각 함수의 입력·반환값·주의점과 코드 흐름은
[함수와 Python 실행 뼈대 읽기 가이드](../../../docs/FUNCTION_API_GUIDE.md)에 정리했습니다.

## 학습 목표

실습을 마치면 다음을 할 수 있습니다.

- 행 단위 무작위 분할이 유사 문장 데이터에서 왜 과대평가를 만들 수 있는지 설명한다.
- `case_id` 그룹을 보존하면서 클래스 비율도 유지하는 train/validation split을 만든다.
- 문자 n-gram TF-IDF와 범주형 메타데이터를 `ColumnTransformer`로 결합한다.
- 전처리와 `LogisticRegression`을 하나의 `Pipeline`으로 학습해 vocabulary 누수를 막는다.
- accuracy, macro F1, weighted F1, 클래스별 precision·recall·F1을 함께 해석한다.
- 신뢰도가 높은 오분류부터 읽고 특징 문제와 데이터 문제를 구분한다.
- 학습한 모델로 새 문장을 예측하고, 확률을 “보정된 확신”으로 오해하지 않는다.

## 문제와 데이터 이해하기

기본 파일은 `data/customer_support_tickets.csv`이며, 현재 데이터는 100개 티켓과 50개
`case_id`로 구성됩니다. 한 사례마다 표현이 비슷한 티켓 두 개가 있고, 다섯 클래스는 각각
20개 행으로 균형을 이룹니다.

| 열 | 의미 | 모델 사용 여부 |
|---|---|---|
| `ticket_id` | 티켓 고유 ID | 오류 추적에 사용 |
| `case_id` | 같은 고객 사례를 묶는 그룹 ID | 안전한 분할에 사용 |
| `created_at` | 생성 시각 | 현재 모델에는 미사용 |
| `channel` | `email`, `chat`, `web`, `phone` 등 | one-hot 특징 |
| `priority` | 우선순위 | one-hot 특징 |
| `customer_tier` | 고객 등급 | one-hot 특징 |
| `text` | 고객 문의 본문 | 문자 n-gram TF-IDF 특징 |
| `label` | 정답 의도 | 학습 목표값 |

예측 대상은 다음 다섯 가지입니다.

- `account_access`: 로그인, 계정 잠금, 인증·복구 문제
- `billing`: 결제, 청구, 환불, 영수증 문제
- `cancellation`: 구독 해지, 갱신 중단 문제
- `delivery`: 배송, 누락·지연 문제
- `technical_issue`: 오류, 업로드, 앱·서비스 기술 문제

## 핵심 개념과 필요한 이유

### 1. `case_id` 그룹 분할로 누수를 막는다

같은 사례에서 만든 두 문장은 핵심 표현이 매우 비슷합니다. 한 문장이 train에, 다른 문장이
validation에 들어가면 모델은 새로운 의도를 일반화하기보다 거의 같은 문장을 기억해 맞힐 수
있습니다. 이때 검증 점수는 실제 성능보다 높게 보입니다.

`stratified_group_split`은 먼저 `case_id`와 `label`의 고유 조합을 만들고, 클래스별로 사례를
섞은 뒤 사례 전체를 한쪽에만 배치합니다. 기본 `validation_size=0.2`에서는 각 클래스의 10개
사례 중 2개 사례, 즉 4개 행이 validation으로 갑니다. 전체는 train 80행, validation
20행입니다.

```text
잘못된 행 분할                 올바른 그룹 분할
C-AC-006 문장 A → train       C-AC-006 문장 A → validation
C-AC-006 문장 B → validation  C-AC-006 문장 B → validation
        ↑ 거의 같은 문장 누수          ↑ 사례 전체가 한쪽에만 존재
```

`DatasetSplit.assert_no_group_leakage()`는 두 프레임의 `case_id` 교집합이 비어 있지 않으면
`AssertionError`를 발생시킵니다. 점수를 보기 전에 이 검사를 통과하는 것이 우선입니다.

### 2. 문자 n-gram TF-IDF는 한국어 기준선으로 강하다

TF-IDF는 한 문서에서 자주 나오지만 모든 문서에 흔하지는 않은 특징에 큰 가중치를 줍니다.
이 예제는 단어 대신 `analyzer="char_wb"`와 `ngram_range=(2, 5)`를 사용합니다. 즉 단어 경계
안에서 2~5글자 조각을 특징으로 만듭니다.

예를 들어 `로그인`, `로그인이`, `로그인을`은 완전히 같은 단어 토큰은 아니지만 `로그`, `로그인`
같은 문자 조각을 공유합니다. 형태 변화가 많은 한국어, 오탈자, 띄어쓰기 변형에 비교적 견고하고
별도 형태소 분석기 없이 쓸 수 있다는 장점이 있습니다. 반면 특징 수가 많아지고, 긴 문맥이나
문장 의미를 직접 이해하지는 못합니다.

현재 `TrainingConfig`의 주요 기본값은 다음과 같습니다.

| 설정 | 기본값 | 의미 |
|---|---:|---|
| `validation_size` | `0.2` | 검증 사례 비율 |
| `max_features` | `20_000` | TF-IDF 최대 특징 수 |
| `ngram_range` | `(2, 5)` | 문자 조각 길이 |
| `min_df` | `1` | 한 문서만 등장한 특징도 허용 |
| `regularization_c` | `4.0` | 로지스틱 회귀 규제의 역강도 |
| `max_iter` | `1_000` | 최적화 최대 반복 수 |
| `class_weight` | `"balanced"` | 클래스 빈도를 반영한 가중치 |
| `seed` | `42` | 분할·학습 재현성 |

### 3. 메타데이터는 one-hot으로 결합한다

본문 `text`는 TF-IDF로, `channel`, `priority`, `customer_tier`는 `OneHotEncoder`로
변환합니다. 학습 때 없던 범주가 예측 때 들어와도 실패하지 않도록
`handle_unknown="ignore"`를 사용합니다.

`ColumnTransformer`의 `transformer_weights`는 텍스트에 `1.0`, 메타데이터에 `0.05`를
적용합니다. 작은 데이터에서 채널 같은 우연한 상관관계가 본문보다 지나치게 큰 영향을 주지
않게 하는 선택입니다. 이 값이 최선이라고 가정하지 말고 검증 실험으로 확인해야 합니다.

### 4. 하나의 `Pipeline`은 학습과 예측의 규칙을 묶는다

TF-IDF vocabulary와 one-hot 범주는 **train 데이터에만 `fit`**되어야 합니다. validation까지
미리 보고 vocabulary를 만들면 작은 누수가 생깁니다. `Pipeline` 안에 전처리와 분류기를 함께
넣으면 `fit`, `predict`, 저장·복원에서 같은 변환이 자동으로 적용됩니다.

`normalize_ticket_text`는 NFKC, 소문자화, 공백 정리에 더해 이메일과 5자리 이상 긴 숫자를
각각 `<email>`, `<number>`로 마스킹합니다. 프로젝트 01의 더 세밀한 `mask_pii`와 목적은
비슷하지만, 이 분류 기준선에 맞춘 별도 전처리 함수입니다.

### 5. accuracy만으로는 클래스별 실패를 볼 수 없다

- **accuracy**: 전체 예측 중 정답 비율
- **precision**: 특정 클래스로 예측한 것 중 실제로 그 클래스인 비율
- **recall**: 실제 특정 클래스 중 모델이 찾아낸 비율
- **macro F1**: 클래스별 F1을 동일한 비중으로 평균
- **weighted F1**: 각 클래스의 표본 수를 가중치로 사용한 F1 평균

현재 데이터는 균형이므로 macro와 weighted F1이 비슷합니다. 불균형 데이터에서는 다수 클래스를
잘 맞혀 accuracy가 높아도 소수 클래스 recall이 0일 수 있으므로 macro F1과 클래스별 지표가
특히 중요합니다.

## 데이터와 코드 흐름

```text
customer_support_tickets.csv
    ↓ load_ticket_data
스키마·결측·중복 ID·case-label 관계 검증
    ↓ TicketClassifierTrainer.train
case_id + label 기준 stratified_group_split
    ├─ train 80행 ─→ Pipeline.fit
    │                 ├─ text → normalize_ticket_text → char_wb TF-IDF
    │                 ├─ metadata → OneHotEncoder × 0.05
    │                 └─ LogisticRegression
    └─ validation 20행
          ├─ evaluate_model → EvaluationReport
          └─ build_error_table → 신뢰도순 오분류 DataFrame
    ↓
TrainingResult(model, report, errors, split)
    ↓ predict_tickets
새 문장 클래스와 클래스별 확률
```

폴더의 파일 역할은 다음과 같습니다.

- `starter.py`: `train_baseline` 하나의 함수 안에서 로드·분할·학습·평가를 연결하는 연습 파일
- `solution.py`: 재사용 모듈의 `TicketClassifierTrainer`를 사용한 완성 CLI
- `src/llm_engineering_lab/data.py`: 데이터 계약, `stratified_group_split`, `DatasetSplit`
- `src/llm_engineering_lab/ml.py`: 모델 구축, 평가, 오류 분석, 예측, 저장·복원

## 실행 방법

프로젝트 루트에서 실행합니다.

```powershell
# TODO 구현 후 starter 실행
.\.venv\Scripts\python.exe projects\nlp\02_intent_classification\starter.py

# 기본 seed와 예측 문장으로 완성 예제 실행
.\.venv\Scripts\python.exe projects\nlp\02_intent_classification\solution.py

# 분할 seed와 새 문의를 바꿔 실행
.\.venv\Scripts\python.exe projects\nlp\02_intent_classification\solution.py `
  --seed 17 --predict "파일 업로드에서 500 오류가 납니다"

# 다른 데이터 사용
.\.venv\Scripts\python.exe projects\nlp\02_intent_classification\solution.py `
  --data data\customer_support_tickets.csv
```

TODO를 구현하기 전 starter의 `NotImplementedError`는 정상입니다.

## `starter.py` TODO 단계별 힌트

`IntentClassificationProject.load_frame()`이 CSV 계약을 담당하고,
`train_baseline(frame, seed) -> BaselineTrainingResult`가 분할·학습·평가를 담당합니다.
starter 결과에는 `model`, `metrics`, `errors`, train/validation case ID가 들어 있습니다.
solution의 `TrainingResult`는 같은 책임을 더 풍부한 타입으로 표현합니다. 아래 순서를 하나씩
구현하세요.

### TODO 1. CSV 로드와 스키마 검증

- 단순 `pd.read_csv`로 시작할 수 있지만, 완성 모듈의 `load_ticket_data(data_path)`를 사용하면
  다음 계약을 한 번에 검사할 수 있습니다.
  - 필수 8개 열 존재
  - 빈 값 없음
  - `ticket_id` 중복 없음
  - 하나의 `case_id`가 하나의 `label`에만 대응
  - 기대한 다섯 label이 모두 존재
  - `created_at`이 유효한 UTC datetime으로 변환 가능
- 로드 직후 `shape`, `label`별 행 수, `case_id`별 행 수를 출력해 데이터를 먼저 이해하세요.
- 기본 경로는 `ROOT / "data/customer_support_tickets.csv"`로 계산됩니다.

힌트가 너무 직접적이라 느껴지면 먼저 pandas만으로 위 검사를 작성한 후
`validate_ticket_frame` 구현과 비교하세요.

### TODO 2. label 균형을 보존한 `case_id` 분할

- 행이 아니라 `frame[["case_id", "label"]].drop_duplicates()`에서 분할 대상을 만드세요.
- 클래스별 case 목록을 정렬한 뒤 하나의 seed가 있는 난수 생성기로 섞으면 재현하기 쉽습니다.
- 각 클래스에 train case와 validation case가 최소 하나씩 남도록 경계를 처리하세요.
- 선택한 validation case 집합으로 원본 **전체 행**을 나눕니다.
- 마지막에 train/validation의 `case_id` 교집합이 빈 집합인지 assert하세요.

직접 분할을 구현한 뒤 완성 함수 `stratified_group_split(frame, validation_size=..., seed=...)`의
결과 계약과 비교할 수 있습니다. 기본 데이터·seed 42에서는 80/20행이며 validation의 각
클래스 support는 4입니다.

### TODO 3. TF-IDF + 로지스틱 회귀 `Pipeline` 학습

두 가지 학습 경로 중 하나를 선택할 수 있습니다.

1. **원리 학습 경로**: `TfidfVectorizer`, `OneHotEncoder`, `ColumnTransformer`,
   `LogisticRegression`, `Pipeline`을 직접 조합합니다.
2. **재사용 경로**: `TrainingConfig`와 `TicketClassifierTrainer`를 사용해 완성된 구성요소의
   연결과 결과 해석에 집중합니다.

직접 구성할 때 확인할 점은 다음과 같습니다.

- 텍스트 transformer에 열 이름 문자열 `"text"`를 전달해 1차원 문자열 시리즈가
  `TfidfVectorizer`로 들어가게 합니다.
- 메타데이터 transformer에는 세 열의 리스트를 전달합니다.
- 모델 입력 열은 `TicketSchema.model_feature_columns`, 즉
  `("text", "channel", "priority", "customer_tier")`입니다.
- validation에는 `fit_transform`을 호출하지 말고, 학습된 pipeline의 `predict`만 사용하세요.
- 고차원 희소 특징과 잘 맞는 `LogisticRegression`으로 기준선을 만듭니다.

처음에는 solution의 기본 설정과 동일하게 시작한 뒤 한 설정씩 바꾸세요. 여러 설정을 동시에
바꾸면 점수 변화의 원인을 알기 어렵습니다.

### TODO 4. 모델·지표·분할·오류 예시 반환

- 예측값으로 accuracy와 macro F1을 계산하되, 클래스별 precision·recall·F1도 남기세요.
- `evaluate_model`을 사용하면 `EvaluationReport`의 `accuracy`, `macro_f1`, `weighted_f1`,
  `per_class`, `confusion_matrix`를 얻습니다.
- 틀린 행에는 최소한 원문, 실제 label, 예측 label을 넣으세요.
- `build_error_table`은 여기에 `confidence`, `second_choice`, `second_probability`,
  `confidence_margin`을 추가하고 높은 confidence 순서로 정렬합니다.
- 참고 구조에서는 `TrainingResult` 하나를 반환하므로 호출자가 정확한 split과 모델을 함께
  보관할 수 있습니다.

### TODO 5. 새 문장 추론

`IntentClassificationProject.predict_new_text()`에서 학습된 pipeline에 새 문장을 전달하세요.
학습에 쓰인 `text`, `channel`, `priority`, `customer_tier` 열을 정확히 만들고, 예측 label과
최대 클래스 확률을 하나의 `DataFrame`으로 반환합니다. 전처리 객체를 새로 fit하면 안 됩니다.

작은 함수들을 한꺼번에 완성하려 하지 말고 다음 순서로 확인하세요.

```text
로드 성공 → case 교집합 0 → fit 성공 → 예측 길이 20
→ 전체 지표 생성 → 오분류 표 생성 → 새 문장 예측
```

## 수식에서 코드로

문서 \(d\)의 문자 n-gram \(t\)에 대한 TF-IDF와 다중 클래스 확률은 다음 형태입니다.

$$
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)
\log\frac{N+1}{\operatorname{df}(t)+1},\qquad
p(y=c\mid x)=\frac{e^{w_c^\top x+b_c}}
{\sum_j e^{w_j^\top x+b_j}}
$$

`TfidfVectorizer`가 첫 항의 희소 특징 \(x\)를 만들고 `LogisticRegression`이 두 번째 식의
가중치 \(w_c,b_c\)를 학습합니다. 두 객체를 하나의 `Pipeline`에 둔 이유는 validation이나
새 문장으로 TF-IDF vocabulary가 다시 fit되는 누수를 구조적으로 막기 위해서입니다.

## starter와 `solution.py` 1:1 코드 지도

| 단계 | starter/solution 코드 | 데이터 shape·계약 | 구현 선택 이유 |
|---|---|---|---|
| 설정 | `IntentProjectConfig` | CSV 경로·seed·예측 문장 | 실험 재현에 필요한 값을 한 객체로 남긴다 |
| 로드 | `load_frame()` | CSV `(100, 8)` → `DataFrame` | split 전에 `case_id`, `text`, `label`을 검증한다 |
| 학습 | `train()` | train 80행 / validation 20행 → `TrainingResult` | 사례 그룹 교집합 0을 점수보다 먼저 검사한다 |
| 지표 | `print_metrics()` | 전체·클래스별 지표 → 표 | accuracy 하나가 숨기는 소수 클래스 실패를 드러낸다 |
| 오류 분석 | `print_errors()` | 오분류 frame → 상위 5행 | 자신 있게 틀린 구체적 문장을 포트폴리오 증거로 남긴다 |
| 추론 | `predict_new_text()` | `str` → label·confidence 표 | validation과 동일한 fitted pipeline을 재사용한다 |
| 실행 | `run()` | config → `TrainingResult` | 로드·학습·진단·추론 순서를 고정한다 |

starter의 `BaselineTrainingResult`는 model, metrics, 오류 행, train/validation case ID를
명시하고 `assert_no_group_leakage()`를 제공합니다. solution은 더 풍부한 공용
`TrainingResult`를 사용하지만 책임과 호출 순서는 같습니다. starter의 `train_baseline()`에
그룹 분할과 Pipeline을 직접 구현한 뒤 각 행을 solution의 `train()`과 비교하세요.

solution의 `load_frame()`은 `load_ticket_data()`가 전체 스키마·결측·case-label 관계를
검증한 뒤 필수 열을 한 번 더 확인합니다. `train()`은 `TrainingConfig(seed=...)`와
`TicketClassifierTrainer.train()`을 호출하고 곧바로 leakage assertion을 수행합니다.
`predict_new_text()`는 `predict_tickets()`를 통해 누락된 metadata를 `"unknown"`으로 채우고
클래스별 확률을 반환합니다. confidence는 calibration된 실제 정답 확률이 아니므로 운영
임계값으로 사용하기 전 별도 검증이 필요합니다.

CLI의 `--data`, `--seed`, `--predict`는 모두 `IntentProjectConfig`에 기록됩니다. starter는
TODO가 남으면 `연습 대기: ...`를 출력하고, solution은 완성된 전체 실험을 실행합니다.

## 예상 출력과 해석

기본 데이터와 seed 42에서 현재 참고 구현의 핵심 출력은 다음과 같습니다. 라이브러리의 세부
버전이 달라지면 확률의 마지막 소수 자리는 조금 달라질 수 있습니다.

```text
train=80, validation=20
accuracy=0.800, macro_f1=0.793, weighted_f1=0.793

클래스별 지표
  account_access     precision=1.000 recall=0.500 f1=0.667 n=4
  billing            precision=0.500 recall=0.500 f1=0.500 n=4
  cancellation       precision=0.667 recall=1.000 f1=0.800 n=4
  delivery           precision=1.000 recall=1.000 f1=1.000 n=4
  technical_issue    precision=1.000 recall=1.000 f1=1.000 n=4

오분류 수: 4
...

새 문장 예측
                 text predicted_label  confidence ...
카드에서 같은 금액이 두 번 결제됐어요         billing    0.607068 ...
```

해석할 때 다음을 확인하세요.

- 20개 validation 중 16개를 맞혀 accuracy가 0.8입니다.
- 각 클래스의 support가 모두 4이므로 그룹 분할 뒤에도 클래스 균형이 유지되었습니다.
- `account_access` recall 0.5는 실제 계정 접근 티켓 4개 중 2개만 맞혔다는 뜻입니다.
- `billing`은 precision과 recall이 모두 0.5여서 이 split의 가장 약한 클래스입니다.
- `delivery`, `technical_issue`의 F1 1.0은 이 작은 validation split에서 모두 맞혔다는 뜻일
  뿐, 실제 일반화 성능이 완벽하다는 뜻은 아닙니다.
- macro F1과 weighted F1이 거의 같은 것은 클래스 support가 동일하기 때문입니다.

현재 오답에는 “해외 결제 수수료” billing 사례를 cancellation로 예측한 두 문장과, “인증 앱
복구 코드” account_access 사례를 billing으로 예측한 두 문장이 포함됩니다. 같은 `case_id`의
두 문장이 함께 틀린 점은 그룹이 온전히 validation에 배치되었다는 증거이자, 학습에서 보지 못한
사례 표현에 일반화하지 못했다는 신호입니다. 오답 confidence가 약 0.28~0.31로 낮고 2순위와의
차이도 작으므로 사람 검토 임계값을 둘 여지도 보입니다.

## 실험 과제

모든 비교는 같은 split seed를 고정한 상태에서 한 설정씩 바꾸세요.

1. **문자 vs 단어 n-gram**: 현재 `char_wb` 2~5글자와 word unigram/bigram을 비교하세요.
   accuracy와 macro F1뿐 아니라 어떤 사례의 예측이 바뀌었는지 기록하세요.
2. **메타데이터 제거 실험**: `channel`, `priority`, `customer_tier` transformer를 제거하거나
   가중치 `0.05`를 바꿔 점수와 오답을 비교하세요. 우연한 메타데이터 의존이 있는지 확인합니다.
3. **n-gram 범위·특징 수 실험**: `(2, 4)`, `(3, 5)`, 최대 특징 5천·2만 등을 비교해
   성능, 학습 시간, 희소 행렬 크기의 trade-off를 기록하세요.
4. **규제와 클래스 가중치**: `regularization_c`와 `class_weight=None`을 바꿔 보세요. 균형
   데이터에서 `balanced`가 실제로 도움이 되는지 검증합니다.
5. **seed 민감도**: seed 5개 이상에서 macro F1의 평균·최솟값·최댓값을 계산하세요. 작은
   validation 하나의 점수만 보고 결론 내리는 위험을 확인할 수 있습니다.
6. **오류 유형 태깅**: 오답을 “학습에 없는 표현”, “두 의도에 걸친 문장”, “메타데이터 영향”,
   “정답 label 의심” 등으로 수작업 분류하고 다음 데이터 수집 우선순위를 정하세요.
7. **검토 임계값**: 최고 확률 또는 `confidence_margin`이 낮으면 `human_review`로 보내는 규칙을
   만들고, 자동 처리 비율과 자동 처리 정확도의 관계를 표로 만드세요.
8. **저장·복원 검증**: `save_model`, `load_model`로 pipeline을 joblib 파일에 저장한 뒤 같은
   문장의 `predict_tickets` 결과가 완전히 같은지 확인하세요. 신뢰할 수 없는 joblib 파일은
   역직렬화하지 마세요.

## 자주 생기는 오류와 디버깅

### 점수는 높은데 실제 새 문장에서 성능이 나쁘다

먼저 train과 validation의 `case_id` 교집합을 확인하세요. 행 단위 `train_test_split`을 사용하면
유사 문장 누수로 점수가 부풀 수 있습니다. `result.split.assert_no_group_leakage()`를 평가 직전에
반드시 실행하세요.

### `missing required columns` 오류가 발생한다

CSV 열 이름이 `TicketSchema.required_columns`의 8개 이름과 정확히 일치하는지 확인하세요.
특히 `text`, `label`, `case_id`의 철자·앞뒤 공백을 점검합니다. 데이터 행의 값 공백은 로더가
정리하지만 열 이름은 임의로 바꾸지 않습니다.

### `label contract mismatch` 오류가 발생한다

기본 `load_ticket_data`는 다섯 label의 집합이 정확히 일치하는지 검사합니다. 일부 클래스만
추출한 작은 실험 데이터라면 계약을 바꿀지, 데이터를 보완할지 명시적으로 결정해야 합니다.
조용히 무시하면 평가 해석이 달라집니다.

### `one case_id maps to multiple labels` 오류가 발생한다

같은 사례의 두 행에 서로 다른 label이 들어 있습니다. split 전에 정답 데이터를 수정해야 합니다.
한 case가 양쪽 의도에 실제로 걸친다면 label 체계나 multi-label 문제 정의를 재검토하세요.

### `label ... needs at least two case groups` 오류가 발생한다

어떤 클래스에 고유 `case_id`가 하나뿐이라 train과 validation 양쪽에 배치할 수 없습니다. 행을
복제해 해결하지 말고 독립된 사례를 더 수집하거나 해당 클래스를 이번 평가에서 제외하는 정책을
명시하세요.

### `ValueError: empty vocabulary`가 발생한다

텍스트가 비었거나 `min_df`가 데이터 크기에 비해 너무 크거나, analyzer 설정이 모든 특징을
제거했을 수 있습니다. 정규화 후 첫 문장과 TF-IDF 설정을 출력해 확인하세요.

### `ConvergenceWarning`이 발생한다

`max_iter`를 늘리기 전에 특징 스케일, 지나치게 큰 `C`, 데이터와 특징 수를 확인하세요. 기본
설정은 `max_iter=1000`입니다. 경고를 숨기기보다 모델이 수렴했는지 확인해야 합니다.

### 새 문장을 예측할 때 열이 없다는 오류가 발생한다

직접 `model.predict`를 호출하면 pipeline은 네 feature 열이 있는 `DataFrame`을 기대합니다.
문자열만 예측하려면 `predict_tickets(model, text)`를 사용하거나 `channel`, `priority`,
`customer_tier`를 포함한 프레임을 만드세요.

### validation에 없는 label 때문에 지표가 사라진다

그룹 분할에서 클래스별 validation case를 최소 하나 보장했는지 확인하세요. 현재
`stratified_group_split`은 클래스마다 최소 한 그룹, train에도 최소 한 그룹을 남깁니다.

### 결과가 실행할 때마다 달라진다

분할 seed와 `LogisticRegression(random_state=...)`를 모두 고정했는지 확인하세요. 비교 실험에서는
seed, 데이터, split을 고정하고 바꾼 설정 하나만 기록합니다.

### confidence가 높은데 틀린다

분류 확률은 자동으로 calibration되지 않습니다. 최고 확률만 보지 말고 실제·예측 label,
두 번째 선택, `confidence_margin`, 원문을 함께 읽으세요. 운영 임계값은 별도 holdout에서
조정해야 합니다.

## 완료 체크리스트

- [ ] CSV의 필수 열, 결측, 중복 `ticket_id`, case-label 관계를 검증했다.
- [ ] train/validation을 행이 아니라 `case_id` 단위로 분할했다.
- [ ] `assert_no_group_leakage()`를 통과하고 두 쪽의 case 교집합이 0이다.
- [ ] 각 label이 train과 validation에 모두 존재하며 기본 split은 80/20행이다.
- [ ] TF-IDF와 one-hot encoder를 train에만 fit했다.
- [ ] 전처리, 특징 변환, 로지스틱 회귀가 하나의 `Pipeline`에 들어 있다.
- [ ] accuracy와 macro F1, weighted F1, 클래스별 recall을 모두 출력했다.
- [ ] confusion matrix 또는 오분류 표를 확인하고 실패 예시를 직접 읽었다.
- [ ] 가장 자신 있게 틀린 사례의 원인을 최소 두 가지 기록했다.
- [ ] word n-gram과 char n-gram을 동일한 split에서 비교했다.
- [ ] 새 문장을 `predict_tickets`로 예측하고 confidence의 한계를 설명할 수 있다.
- [ ] 모델을 저장했다 다시 읽었을 때 같은 입력의 예측이 동일함을 확인했다.

완료 기준은 **case 누수가 없고, 저장·복원한 모델이 같은 예측을 만들며, macro F1과 클래스별
지표뿐 아니라 실제 실패 예시까지 함께 보고하는 것**입니다.
