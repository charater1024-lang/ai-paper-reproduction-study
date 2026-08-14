# 학습 경로

이 과정은 문법 암기보다 **실행 → 관찰 → 작은 수정 → 검증**을 반복하도록 설계했습니다. 하루 60~90분 기준 약 3~5주 분량이며, 이미 익숙한 단계는 건너뛰어도 됩니다. 각 단계가 끝날 때 `artifacts/`의 결과와 짧은 실험 노트를 남기세요.

## 0. 준비: 실행 가능한 기준점 만들기

먼저 가상환경과 의존성을 설치하고 테스트를 실행합니다.

```powershell
python -m pytest -q
python scripts/00_generate_data.py
python scripts/00_python_engineering_patterns.py
python scripts/01_inspect_data.py --data data/customer_support_tickets.csv
```

`00_python_engineering_patterns.py`에서는 불변 `Ticket` 데이터 계약, 호출 가능한 `Step` Protocol, 합성 가능한 `Pipeline`, streaming batch generator, 지연 시간 decorator, 실험 context manager를 한 흐름으로 실행합니다. 출력된 마스킹 결과와 `artifacts/python_patterns_demo.json`을 코드와 대조해 보세요.

관찰 포인트:

- 현재 Python, PyTorch 버전과 선택된 device
- seed를 같게 했을 때 데이터 split과 첫 배치가 같은지
- 원본 데이터의 필드, 클래스 수, 결측/중복/레이블 분포
- 잘못된 `Ticket`이나 batch size가 데이터 경계에서 즉시 실패하는지

완료 조건: 데이터 한 행의 생명주기(파일 → 파싱 → 검증 → split)를 말로 설명할 수 있습니다.

## 1. 실무형 Python 설계

`src/llm_engineering_lab/`의 설정, 데이터, 모델, 학습 컴포넌트를 읽습니다. 함수만 늘어놓지 않고 책임을 나누는 이유를 파악하세요.

핵심 주제:

- `dataclass` 기반 설정과 유효성 검사
- `Protocol`/추상 인터페이스 또는 공통 메서드를 이용한 교체 가능한 컴포넌트
- `pathlib`, 타입 힌트, context manager, 사용자 정의 예외
- 상태를 가진 `Trainer`와 상태 없는 metric 함수의 구분
- dependency injection: 데이터·모델·optimizer를 클래스 내부에서 숨기지 않고 전달하기

실행하며 확인할 것:

1. 잘못된 경로나 음수 batch size를 넣어 실패 메시지를 확인합니다.
2. 동일 설정 객체를 출력·직렬화하고 실험 결과 옆에 저장합니다.
3. 데이터 로더를 가짜 객체로 바꿔 테스트가 모델 학습 없이 실행되는지 봅니다.

현업 연결: 설정, 로깅, 오류 경계가 명확하면 notebook 코드를 배치 작업이나 서비스 코드로 옮기기 쉽습니다.

## 2. 머신러닝 기준선과 평가

`02_train_ml_baseline.py`로 텍스트 분류 기준선을 학습합니다. 복잡한 신경망보다 먼저 “이 데이터가 실제로 학습 가능한가?”를 확인하는 단계입니다.

```powershell
python scripts/02_train_ml_baseline.py --data data/customer_support_tickets.csv
```

관찰 포인트:

- stratified split 전후의 클래스 비율
- accuracy와 macro F1이 다르게 움직이는 이유
- confusion matrix에서 자주 섞이는 의도
- TF-IDF vocabulary에서 지나치게 강한 단어(레이블 누수 신호)
- 오분류 문장을 직접 읽었을 때 데이터 문제인지 모델 한계인지

기본 실행은 `artifacts/ml/`에 직렬화한 모델, `metrics.json`, `errors.csv`를 남깁니다. metric 숫자만 확인하지 말고 `errors.csv`에서 확신도가 높은 오분류부터 읽으세요.

작은 실험:

- n-gram 범위를 바꾸고 숫자/기호가 많은 문의의 결과를 비교합니다.
- class weight 적용 전후 macro F1과 소수 클래스 recall을 비교합니다.
- 중복 제거 전후 validation 성능 차이로 누수를 점검합니다.

완료 조건: 기준선 결과표와 오분류 5건에 대한 원인 가설을 남깁니다.

## 3. PyTorch 분류기: 텐서에서 checkpoint까지

같은 분류 문제를 PyTorch로 해결합니다.

```powershell
python scripts/03_train_torch_classifier.py --epochs 5 --device auto
```

코드를 다음 순서로 추적하세요.

```text
JSONL row → tokenizer/vectorizer → Dataset.__getitem__
→ DataLoader batch → model.forward → logits
→ loss → backward → optimizer.step → validation → checkpoint
```

관찰 포인트:

- 단일 sample과 batch의 shape 차이
- target이 `long` dtype이어야 하는 이유
- `model.train()`과 `model.eval()`이 dropout 등에 미치는 영향
- `optimizer.zero_grad()`를 빼면 gradient가 누적되는 현상
- validation에서 `torch.no_grad()`를 쓰는 이유
- epoch별 train/validation loss, macro F1, gradient norm

권장 디버깅 순서:

1. 데이터 8~32개만으로 의도적으로 과적합시킵니다.
2. 한 배치 forward와 loss 계산을 확인합니다.
3. backward 후 한 파라미터의 gradient가 존재하는지 봅니다.
4. 전체 train split으로 확장합니다.
5. 저장한 checkpoint를 새 모델에 불러와 예측을 비교합니다.

기본 checkpoint 위치는 `artifacts/ticket_classifier.pt`입니다. CPU로 복원할 때 저장 당시의 장치와 무관하게 `map_location`이 올바르게 적용되는지 확인하세요.

완료 조건: 학습 루프의 각 줄이 어떤 상태를 바꾸는지 설명하고, 저장/복원 테스트를 통과시킵니다.

## 4. 작은 언어 모델: 다음 토큰 예측

작은 corpus와 토크나이저로 causal LM을 훈련합니다. 품질 좋은 챗봇이 목적이 아니라 LLM 내부 데이터 흐름을 손으로 추적하는 것이 목적입니다.

```powershell
python scripts/04_train_tiny_lm.py --epochs 3 --device auto
```

관찰 포인트:

- 텍스트가 token id가 되고 고정 길이 sequence로 잘리는 과정
- `input_ids[:, :-1]`와 `target_ids[:, 1:]`의 한 칸 이동
- padding token이 loss에서 제외되는지
- attention score의 shape가 `(batch, heads, time, time)`인지
- causal mask 위쪽 삼각 영역이 미래 토큰을 차단하는지
- vocabulary 차원의 logits가 확률 분포로 변환되는 과정

생성 실험:

- greedy, temperature, top-k 설정으로 같은 prompt를 여러 번 생성합니다.
- temperature가 낮을 때 반복이 늘고, 너무 높을 때 문맥이 무너지는지 봅니다.
- context length를 넘었을 때 앞 토큰을 자르는 정책을 확인합니다.
- `model.eval()`과 고정 seed로 결과 재현성을 확인합니다.

학습 뒤에는 별도 추론 패턴 실습으로 batching, padding, sampling 설정이 결과와 처리량에 미치는 영향을 비교합니다.

```powershell
python scripts/05_llm_inference_patterns.py --checkpoint artifacts/tiny_lm.pt --device auto
```

이 스크립트는 앞 단계에서 생성한 `artifacts/tiny_lm.pt`를 사용합니다. checkpoint가 없다면 추론 문제로 오해하지 말고 먼저 학습 명령의 저장 경로와 완료 여부를 확인하세요.

```powershell
# --prompt를 반복해 실제 batch padding과 처리량을 관찰
python scripts/05_llm_inference_patterns.py --device auto --prompt "환불하고 싶어요" --prompt "로그인이 안 돼요" --show-rendered-prompt

# 같은 greedy 생성에서 KV cache 사용 전후 비교
python scripts/05_llm_inference_patterns.py --device auto --temperature 0 --benchmark-cache
```

prompt template이 사용자 메시지를 어떻게 감싸는지, batch 내 길이가 다른 요청이 어떻게 padding되는지, cache가 결과 token은 유지하면서 처리량에 어떤 영향을 주는지를 확인합니다. 이 tiny LM의 문장 품질은 낮은 것이 정상이며 여기서는 추론 코드의 계약과 비용을 관찰하는 데 초점을 둡니다.

CPU 팁: sequence length와 embedding/hidden dimension을 먼저 줄이세요. batch size만 줄이면 optimizer step 수가 늘어 오히려 오래 걸릴 수 있습니다. 전체 학습 전에 1~2개 batch로 forward/backward smoke test를 수행하세요.

완료 조건: 하나의 prompt가 token id, logits, sampling을 거쳐 새 텍스트가 되는 과정을 shape와 함께 설명합니다.

## 5. 검색과 RAG: 생성보다 근거 먼저

`knowledge_base.jsonl`의 문서를 읽고 질의와 가까운 문서를 검색합니다. 기본 구현은 문서 한 건을 하나의 검색 단위로 사용하는 문자 n-gram TF-IDF/cosine similarity 방식이라 외부 API 없이 내부 점수를 추적할 수 있습니다. [실습 문제](EXERCISES.md)의 E1에서 긴 문서를 chunk 단위로 확장합니다.

```powershell
python scripts/06_rag_demo.py --evaluate --show-prompt
```

파이프라인:

```text
JSONL 로드/검증 → 검색용 텍스트 구성 → TF-IDF index 구축 → query 변환
→ top-k 검색 → 근거와 grounded prompt 조립 → 출처 표시
```

기본 데모는 외부 LLM API를 호출하지 않고 **LLM에 전달하기 직전의 prompt까지** 보여 줍니다. 덕분에 비용과 네트워크 없이 retrieval을 재현할 수 있으며, 이후 원하는 로컬 모델이나 API를 마지막 단계에 연결할 수 있습니다.

관찰 포인트:

- 제목·category·keyword·본문을 함께 색인했을 때와 본문만 썼을 때의 차이
- top-k가 너무 작거나 클 때의 precision/context noise
- 질문과 top-1 문서의 score, 문서 id, 실제 근거 문장
- 정답 문서가 검색되지 않은 실패와, 검색됐지만 답을 잘못 만든 실패의 구분
- 근거가 약한 질문에 “알 수 없음”으로 답하는 threshold

완료 조건: 최소 10개의 평가 질의로 recall@k 또는 hit@k를 계산하고, 최종 답변과 별도로 retrieval 지표를 보고합니다.

심화 단계에서 문서 chunker를 추가한 뒤에는 chunk 크기와 overlap이 검색 결과에 미치는 영향도 같은 평가 세트로 비교합니다.

## 6. 통합: 대화형 관찰과 운영 관점

`app.py`에서 입력을 바꾸며 분류 확률, 실제 PyTorch tensor shape, Tiny LM 생성 설정,
검색 문서와 점수, 학습 산출물을 즉시 확인합니다. 사이드바에서 데이터, PyTorch 해부,
분류 추론, Tiny LM 생성, RAG 검색, 산출물 메뉴를 오가며 코드 한 부분을 수정한 뒤
관련 결과를 바로 비교할 수 있습니다.

```powershell
python -m streamlit run app.py
```

각 메뉴에서 확인할 정보:

- **데이터:** 레이블 분포, 원문 예시, 결측/중복
- **PyTorch 해부:** token/id, padding, attention mask, input/logit shape
- **분류 추론:** 사용 모델, 입력, 클래스별 확률, 예측 레이블
- **Tiny LM 생성:** seed, temperature, top-k, KV cache, 생성 지연 시간
- **RAG 검색:** query, retrieval 문서 id, score, 근거 본문
- **산출물:** checkpoint와 JSON metric 파일, 학습 결과의 생성 여부
- **공통:** checkpoint/device, 지연 시간, 이해 가능한 오류 메시지

현업형 확장:

- 모델 로딩을 요청마다 반복하지 않고 캐시합니다.
- 입력 길이와 허용 문자를 검증합니다.
- 로그에 원문 개인정보를 남기지 않는 정책을 둡니다.
- timeout, 빈 검색 결과, 손상된 checkpoint 같은 실패 경로를 테스트합니다.
- latency, throughput, 모델 크기, 품질을 함께 비교합니다.

완료 조건: 처음 보는 입력 10개로 데모를 실행하고, 실패 유형·개선 우선순위·예상 trade-off를 짧은 보고서로 정리합니다.

## 추천 주간 계획

| 주차 | 초점 | 결과물 |
|---|---|---|
| 1주 | Python 설계, 데이터 검증, ML 기준선 | 데이터 리포트, baseline 표, 실패 예시 |
| 2주 | PyTorch Dataset/모델/학습 루프 | checkpoint, metric 로그, 복원 테스트 |
| 3주 | tokenizer, attention, 작은 causal LM | 생성 샘플 비교표, shape 추적 노트 |
| 4주 | retrieval/RAG, 통합 데모 | 검색 평가표, 동작 데모, 오류 분석 |
| 5주(선택) | 성능·테스트·리팩터링 | profiling 결과, 추가 테스트, 설계 문서 |

## 실험 노트 템플릿

매 실험마다 아래 다섯 줄만 남겨도 무작위 튜닝을 크게 줄일 수 있습니다.

```text
가설:
변경한 한 가지:
고정한 조건/seed:
결과(metric + 실패 예시):
다음 결정:
```
