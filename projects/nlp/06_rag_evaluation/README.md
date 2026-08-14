# 06. RAG 검색·보류 평가

이 실습에서는 고정된 JSONL 평가셋으로 RAG의 **검색 품질**과 **no-answer 보류 품질**을
수치화합니다. 좋은 데모 답변 몇 개를 눈으로 보는 것만으로는 검색기가 실제로 개선됐는지,
우연히 한 질문만 잘 맞힌 것인지 구분하기 어렵습니다. 같은 평가셋과 같은 지표를 반복해서
사용하면 임베딩, 청크, `top_k`, score threshold 변경 전후를 비교할 수 있습니다.

여기서는 LLM의 문장 품질을 평가하지 않습니다. 검색 지표와 생성 지표를 한 점수로 섞으면
“정답 근거를 못 찾은 검색 문제”와 “좋은 근거를 잘못 요약한 생성 문제”를 구분하기 어렵기
때문입니다. 먼저 검색과 보류를 안정화한 뒤 별도의 groundedness, 사실성, 인용 정확도 평가를
추가하는 순서가 좋습니다.

## 학습 목표

- answerable 질문과 no-answer 질문을 같은 방식으로 채점하면 안 되는 이유를 설명한다.
- hit@k, Recall@k, MRR@k의 정의와 차이를 손으로 계산한다.
- coverage가 정확도가 아니며, 잘못된 문서를 반환해도 coverage에는 포함됨을 이해한다.
- score threshold가 Recall과 보류 정확도 사이에 만드는 trade-off를 측정한다.
- 전체 평균뿐 아니라 실패한 `query_id`, 기대 문서, 실제 순위를 남겨 회귀 원인을 찾는다.

## 준비와 실행

저장소 루트에서 실행합니다. API 키나 네트워크는 필요하지 않습니다.

```powershell
.\.venv\Scripts\python.exe projects\nlp\06_rag_evaluation\solution.py
```

실패 사례까지 보려면 다음 옵션을 사용합니다.

```powershell
.\.venv\Scripts\python.exe projects\nlp\06_rag_evaluation\solution.py --show-failures
```

주요 옵션은 다음과 같습니다.

```text
--top-k         질문마다 평가할 최대 검색 결과 수(기본 3)
--min-score     이 점수보다 낮은 결과를 반환하지 않는 검색 threshold(기본 0.01)
--show-failures 기대 문서를 못 찾았거나 no-answer에서 문서를 반환한 사례 출력
```

## 평가 데이터 이해하기

평가셋은 `data/practice/rag_queries.jsonl`이며 한 줄이 한 사례입니다.

```json
{"query_id":"q-010","query":"429 응답을 안전하게 재시도하는 방법","relevant_ids":["kb-010"]}
{"query_id":"q-013","query":"회사 주소가 어디인가요?","relevant_ids":[]}
```

- `query_id`: 결과를 회귀 추적할 고유 ID입니다. 중복되면 어느 사례가 실패했는지 모호해져
  로더가 오류를 냅니다.
- `query`: 검색기에 전달할 사용자 질문입니다.
- `relevant_ids`: 이 질문의 정답 근거로 인정할 문서 ID 목록입니다. 정답이 여러 개인
  질문도 있으므로 문자열 하나가 아니라 배열입니다.
- 빈 `relevant_ids`는 지식베이스 안에 정답이 없는 **no-answer 질문**입니다. 이 프로젝트의
  정답 행동은 관련 없는 문서를 붙이는 것이 아니라 빈 검색 결과를 반환하는 것입니다.

현재 데이터는 총 13건이며 answerable 12건, no-answer 1건(`q-013`)입니다. no-answer가
한 건뿐이므로 보류 정확도는 0% 또는 100%로 크게 움직입니다. 실무 평가셋에서는 여러 유형의
도메인 밖 질문, 모호한 질문, 공격적 질문을 더 넣어야 안정적인 추정이 됩니다.

`load_evaluation_cases()`는 다음을 검증합니다.

- 각 줄이 올바른 JSON 객체인지
- `query_id`와 `query`가 공백이 아닌 문자열인지
- `relevant_ids`가 문자열 배열이고 내부에 중복이 없는지
- 파일 전체에서 `query_id`가 고유한지

오류 메시지에는 파일 경로와 줄 번호가 포함되므로 데이터를 고칠 때 활용할 수 있습니다.

## 지표 정의

아래 지표에서 `R_q`는 질문 `q`의 정답 문서 ID 집합,
`P_q@k`는 검색 결과 상위 `k`개의 문서 ID 목록입니다. no-answer 사례는 별도로 평가하므로
hit, Recall, MRR의 평균 분모에는 answerable 질문만 들어갑니다.

핵심 세 지표를 한 번에 쓰면 다음과 같습니다. \(A\)는 answerable 질문 집합이고,
\(r_q\)는 top-k에서 처음 정답이 나온 1-based 순위이며 miss일 때는 무한대입니다.

$$
\operatorname{Recall@k}
=\frac{1}{|A|}\sum_{q\in A}
\frac{|R_q\cap P_q@k|}{|R_q|},\qquad
\operatorname{MRR@k}
=\frac{1}{|A|}\sum_{q\in A}\frac{1}{r_q}
$$

코드에서 `evaluate_retrieval()`의 answerable 분기가 첫 합을 누적하고,
`reciprocal_rank()`가 두 번째 합의 각 항을 계산합니다. no-answer를 이 분모에서 제외하는
이유는 빈 정답 집합으로 나누는 수학적 오류와 검색 품질 왜곡을 동시에 피하기 위해서입니다.

### 1. hit@k와 hit rate@k

answerable 질문 하나에 대해 상위 `k`개 안에 정답 문서가 **하나라도** 있으면 hit입니다.

```text
hit@k(q) = 1  if R_q ∩ P_q@k 가 비어 있지 않음
           0  otherwise

hit rate@k = answerable hit 수 / answerable 질문 수
```

hit는 성공 여부만 봅니다. 정답 문서가 두 개 필요한 질문에서 하나만 찾든 둘 다 찾든
`hit=1`입니다. CLI의 `hit@3=...` 표시는 엄밀히 말하면 전체 answerable 사례의
`hit_rate_at_k`입니다.

### 2. Recall@k

Recall은 필요한 정답 문서 중 몇 개를 상위 `k`에서 찾았는지 봅니다.

```text
Recall@k(q) = |R_q ∩ P_q@k| / |R_q|
전체 Recall@k = answerable 질문별 Recall@k의 평균(macro average)
```

예를 들어 정답이 `{kb-001, kb-002}`인데 상위 3개에서 `kb-002`만 찾았다면 hit는 1,
Recall@3은 `1/2 = 0.5`입니다. 현재 평가셋의 `q-002`가 두 정답 문서를 가진 사례라서
hit rate와 Recall이 실제로 달라질 수 있습니다.

이 구현은 모든 정답-질문 쌍을 한꺼번에 세는 micro average가 아니라, 질문별 recall을
계산한 뒤 같은 가중치로 평균냅니다. 정답 문서 수가 많은 질문 하나가 전체 점수를 과도하게
지배하지 않게 하기 위한 선택입니다.

### 3. Reciprocal Rank와 MRR@k

Reciprocal Rank(RR)는 **첫 번째 정답 문서가 나온 순위**만 봅니다.

```text
첫 정답이 1위 -> RR = 1/1 = 1.0
첫 정답이 2위 -> RR = 1/2 = 0.5
첫 정답이 3위 -> RR = 1/3 ≈ 0.333
상위 k에 정답 없음 -> RR = 0

MRR@k = answerable 질문들의 RR 평균
```

사용자는 보통 위 결과부터 읽으므로, 같은 hit 수라면 정답을 더 앞에 배치하는 검색기의
MRR이 높습니다. 반면 두 번째, 세 번째 정답을 더 찾았는지는 MRR에 반영되지 않으므로
다중 근거가 중요한 RAG에서는 Recall과 함께 봐야 합니다.

코드는 `top_k` 이후 결과를 잘라 채점하므로 4위에 정답이 있어도 MRR@3에서는 0입니다.

### 4. Coverage

Coverage는 전체 질문 중 검색 결과가 하나 이상 반환된 비율입니다.

```text
coverage = 결과가 비어 있지 않은 전체 사례 수 / 전체 사례 수
```

coverage는 “시스템이 답을 시도할 수 있는 비율”이지 정확도가 아닙니다. 틀린 문서만
반환한 질문도 covered입니다. 구현의 report에는 다음 세 값이 있습니다.

- `coverage`: answerable과 no-answer를 모두 포함한 전체 반환 비율
- `answerable_coverage`: answerable 질문 중 하나 이상 반환한 비율
- `no_answer_coverage`: no-answer 질문에서 문서를 반환한 비율, 즉 검색 단계의 거짓 양성률

CLI는 표를 간단히 유지하기 위해 전체 `coverage`만 출력하지만, 분석 코드에서는 세 값을
함께 보는 것이 좋습니다.

### 5. No-answer abstention accuracy

no-answer 사례에서는 빈 결과를 반환하면 올바른 보류(correct abstention), 하나라도
반환하면 잘못된 답변 후보(false positive)로 봅니다.

```text
abstention accuracy
  = 빈 결과를 반환한 no-answer 사례 수 / 전체 no-answer 사례 수
```

현재 정의에서는 no-answer 사례마다 “결과 있음/없음”이 서로 보완 관계이므로 다음이
성립합니다.

```text
abstention_accuracy = 1 - no_answer_coverage
```

주의할 점은 “관련 문서를 못 찾았지만 LLM이 우연히 모른다고 말했다”를 올바른 검색으로
채점하지 않는다는 것입니다. 이 실습은 생성 전에 검색 결과가 비었는지를 평가합니다.
보류를 생성 모델의 자유로운 문장에만 맡기면 일관성과 비용 관리가 어려워집니다.

answerable 사례가 하나도 없거나 no-answer 사례가 하나도 없는 평가 묶음에서는 해당 지표의
분모가 0입니다. 구현은 오해를 부르는 `0.0` 대신 `None`을 반환하고 CLI의 `compact()`가
`n/a`로 표시합니다.

## 작은 손계산 예제

`k=3`이고 다음 네 사례가 있다고 가정합니다. `X`, `Y`는 모두 오답 문서입니다.

| 사례 | 정답 `R_q` | 검색 결과 `P_q@3` | hit | Recall | RR | 결과 있음? |
|---|---|---|---:|---:|---:|---:|
| q-a | `{A, B}` | `[X, A, Y]` | 1 | 1/2 | 1/2 | 1 |
| q-b | `{C}` | `[C, X]` | 1 | 1 | 1 | 1 |
| q-c | `{D}` | `[X, Y]` | 0 | 0 | 0 | 1 |
| q-n | `{}` | `[]` | 별도 평가 | 별도 평가 | 별도 평가 | 0 |

answerable 질문은 q-a, q-b, q-c 세 개입니다.

```text
hit rate@3 = (1 + 1 + 0) / 3 = 0.667
Recall@3   = (0.5 + 1 + 0) / 3 = 0.500
MRR@3      = (0.5 + 1 + 0) / 3 = 0.500
coverage   = 3 / 4 = 0.750
abstention accuracy = 1 / 1 = 1.000
no-answer coverage  = 0 / 1 = 0.000
```

q-c는 오답만 반환했지만 coverage에는 포함됩니다. q-a는 hit이지만 정답 두 개 중 하나만
찾았으므로 Recall은 0.5입니다. 이 두 행이 지표를 함께 봐야 하는 이유를 보여 줍니다.

## threshold의 역할과 trade-off

이 프로젝트의 `TfidfRetriever`는 질문-문서 유사도가 `min_score` 이상인 문서만 반환합니다.
threshold는 순위를 만드는 설정이 아니라 **이미 계산한 점수 중 반환할 최소 기준**입니다.

- threshold를 낮추면 answerable Recall과 coverage가 보통 높아지지만, no-answer 질문에도
  억지로 문서를 붙일 가능성이 커집니다.
- threshold를 높이면 no-answer 보류는 쉬워지지만, 점수가 낮은 올바른 근거까지 사라져
  answerable hit/Recall과 coverage가 낮아질 수 있습니다.
- 이 구현에서 threshold는 남은 문서를 재정렬하지 않습니다. 따라서 고정된 점수와 데이터라면
  threshold를 높일 때 MRR은 유지되거나 낮아집니다.
- score의 척도는 검색기마다 다릅니다. TF-IDF에서 고른 `0.20`을 해시 임베딩이나 다른
  벡터 모델에 그대로 적용할 수 없습니다.

따라서 “가장 높은 Recall”만으로 threshold를 고르면 no-answer 정확도가 무너지고,
“가장 높은 보류 정확도”만 보면 유용한 질문을 너무 많이 거절할 수 있습니다. 실제 제품에서는
잘못 답변하는 비용과 불필요하게 보류하는 비용을 정한 뒤, 별도 validation set에서 정책을
고르고 마지막 test set은 한 번만 확인해야 합니다.

## `starter.py` TODO 힌트

`starter.py`는 문자열 ID를 반환하는 간단한 `search(query, top_k)` 인터페이스로 핵심
지표를 직접 구현하게 되어 있습니다.

### TODO 1: `reciprocal_rank()`

순서를 보존해야 하므로 `ranked_ids`를 `set`으로 바꾸면 안 됩니다.

```text
ranked_ids를 enumerate(..., start=1)로 순회
  -> id가 relevant_ids 안에 처음 나타나면 1 / rank 반환
끝까지 없으면 0.0 반환
```

다음 작은 검사를 먼저 통과시켜 보세요.

```python
assert reciprocal_rank(["A", "B"], {"A"}) == 1.0
assert reciprocal_rank(["X", "A"], {"A"}) == 0.5
assert reciprocal_rank(["X", "Y"], {"A"}) == 0.0
```

### TODO 2: 평가 사례 검증

- `top_k`가 `bool`이 아닌 양의 정수인지 확인합니다. Python에서 `bool`은 `int`의 하위
  타입이므로 엄격한 검증에서는 별도 처리가 필요합니다.
- 사례가 하나 이상인지, `query_id`가 고유한지 확인합니다.
- `query`, `query_id`, `relevant_ids`의 타입과 빈 문자열을 확인합니다.
- search 구현이 실수로 더 많이 반환해도 처음 `top_k`개만 채점합니다.

### TODO 3: answerable 지표

각 answerable 질문에서 다음 값을 누적합니다.

```text
retrieved_ids = search(query, top_k)의 처음 top_k ID
matched = set(retrieved_ids) & relevant_ids
hit 누적       += bool(matched)
recall 누적    += len(matched) / len(relevant_ids)
RR 누적        += reciprocal_rank(retrieved_ids, relevant_ids)
```

마지막에 answerable 질문 수로 각각 나눕니다. 전체 13건으로 나누면 no-answer 사례가 검색
지표를 부당하게 낮추므로 주의하세요.

### TODO 4: no-answer와 coverage

`relevant_ids`가 비어 있으면 recall의 분모로 사용하지 말고 별도 분기로 보냅니다.

- `retrieved_ids == []`: correct abstention
- `retrieved_ids != []`: no-answer false positive
- coverage에는 어느 종류의 질문이든 `retrieved_ids != []`이면 1을 더합니다.

반환 딕셔너리에 최소한 `hit_rate_at_k`, `recall_at_k`, `mrr_at_k`, `coverage`,
`abstention_accuracy`와 각 분모의 사례 수를 넣으면 결과를 해석하기 쉽습니다. 완성 후 아래의
손계산 fixture가 앞 절의 값과 일치하는지 먼저 확인하세요.

### TODO 5~8: 평가 프로젝트 구성

1. `load_cases()`에서 JSONL을 읽고 `query_id`, `query`, `relevant_ids` 계약을 검사합니다.
2. `build_retriever(min_score)`는 각 threshold 실험마다 새 인덱스를 fit합니다.
3. `RetrievalEvaluationProject.evaluate()`는 결과 객체를 ID 목록으로 바꾸는 adapter를 만든 뒤
   위의 순수 `evaluate()`를 호출합니다.
4. `find_failures()`는 answerable miss와 no-answer false positive를 query ID와 함께
   반환합니다. 평균만 출력하지 말고 실제 순위 ID도 포함하세요.

## starter와 `solution.py` 1:1 코드 지도

| 단계 | starter/solution 코드 | 입력 → 출력 | 왜 이 구현을 택했는가 |
|---|---|---|---|
| 설정 | `EvaluationProjectConfig` | 경로·k·threshold 후보 | 평가 동작점을 결과와 함께 기록한다 |
| 사례 로드 | `load_cases()` | JSONL 13행 → 12 answerable + 1 no-answer | 두 분모가 모두 존재하는지 tuning 전에 확인한다 |
| 검색기 | `build_retriever(min_score)` | 문서 12개 → fitted retriever | threshold 후보마다 독립 객체를 만들어 상태 간섭을 막는다 |
| 단일 평가 | `evaluate()` | retriever·cases → report | adapter 뒤의 순수 지표를 검색기 타입과 분리한다 |
| 정책 비교 | `threshold_sweep()` | 5개 threshold → 5 reports | recall과 abstention의 반대 방향 변화를 같은 표로 본다 |
| 실패 분석 | `find_failures()` | report 근거 → query별 실제 ID | 평균 점수가 숨기는 miss와 false evidence를 공개한다 |
| 표현 | `print_report()`, `print_sweep()` | report → 표 | 계산 결과 객체를 출력 형식과 분리한다 |
| 전체 실행 | `run()` | config → 기준 report | 동일한 순서와 serialization assertion을 재현한다 |

starter에서는 `reciprocal_rank()`와 `evaluate(search, cases, top_k)`를 직접 구현합니다.
바깥의 `RetrievalEvaluationProject`는 solution과 같은 메서드 이름과 실행 순서를 가지므로,
순수 지표와 프로젝트 orchestration을 함께 연습할 수 있습니다. `evaluate()` adapter는 검색
결과 객체에서 ID를 꺼내는 책임만 지고 지표 공식에는 결과 클래스 지식을 넣지 마세요.

재사용 구현의 `EvaluationCase`는 검증된 `query_id`, `query`, `relevant_ids`를 불변
dataclass로 보관하고 `bool(relevant_ids)`로 answerable 여부를 한 곳에서 결정합니다.
`evaluate_retrieval()`은 `search(query, top_k)`와 `get_result_id(result)`를 받아 결과를
`top_k`로 자릅니다. 이 설계 덕분에 TF-IDF, LangChain `Document`, 벡터 DB 결과도 ID
adapter만 바꿔 같은 수식으로 평가할 수 있습니다.

solution의 `load_cases()`는 answerable과 no-answer가 모두 있는지 검사합니다.
`evaluate()`는 report가 입력 사례를 누락하지 않았는지 assertion을 실행합니다.
`threshold_sweep()`은 `0.01`, `0.05`, `0.10`, `0.15`, `0.20`마다 새 검색기를 만들고,
`find_failures()`는 정답 교집합이 없는 answerable 또는 결과가 있는 no-answer를 반환합니다.
마지막 `asdict(report)` 검사는 결과가 JSON 기준선으로 저장 가능한 구조인지 확인합니다.
starter는 TODO가 남으면 `연습 대기: ...`를 출력합니다.

## 예상 출력과 해석

현재 데이터와 기본 설정(`top_k=3`, `min_score=0.01`)의 출력은 다음과 같습니다.

```text
cases=13 (answerable=12, no-answer=1)
hit@3=1.000, recall@3=1.000, MRR@3=0.944
coverage=1.000, no-answer abstention accuracy=0.000

threshold sweep
min_score  recall@k  MRR@k  coverage  no-answer-accuracy
     0.01     1.000  0.944     1.000               0.000
     0.05     1.000  0.944     1.000               0.000
     0.10     0.917  0.917     0.923               0.000
     0.15     0.875  0.917     0.923               0.000
     0.20     0.875  0.917     0.846               1.000
```

해석은 다음과 같습니다.

- 기본값은 answerable 12건의 정답을 모두 top 3 안에서 찾았습니다.
- MRR이 0.944라는 것은 모든 첫 정답이 1위인 것은 아니라는 뜻입니다. Recall이 1이어도
  순위 품질은 별개입니다.
- coverage 1.0은 13건 모두에 무언가 반환했다는 뜻입니다. no-answer `q-013`에도 문서를
  반환했으므로 보류 정확도는 0입니다.
- threshold 0.20에서는 `q-013`이 빈 결과가 되어 보류 정확도가 1.0으로 좋아집니다.
  대신 answerable `q-008`도 빈 결과가 되어 hit가 줄고, `q-002`에서는 두 정답 중 하나만
  남아 Recall이 추가로 낮아집니다.
- threshold 0.20의 coverage 0.846은 13건 중 11건만 결과가 있다는 뜻입니다. 이 값만 보면
  빠진 두 건 중 하나가 올바른 보류이고 하나가 잘못된 보류라는 사실을 알 수 없습니다.

기본 설정에서 실패를 출력하면 다음 no-answer 거짓 양성을 볼 수 있습니다.

```text
실패 사례
q-013: expected=[] retrieved=['kb-007', 'kb-001', 'kb-006'] | 회사 주소가 어디인가요?
```

threshold 0.20으로 실행하면 실패 유형이 바뀝니다.

```powershell
.\.venv\Scripts\python.exe projects\nlp\06_rag_evaluation\solution.py `
  --min-score 0.20 --show-failures
```

```text
hit@3=0.917, recall@3=0.875, MRR@3=0.917
coverage=0.846, no-answer abstention accuracy=1.000

실패 사례
q-008: expected=['kb-008'] retrieved=[] | 받은 제품이 깨져 있는데 어떤 사진이 필요한가요?
```

즉, threshold 하나로 모든 문제가 사라진 것이 아니라 **거짓 답변을 줄이는 대신 유효한
질문 하나를 보류**하도록 정책이 이동했습니다.

## 직접 해볼 실험

### 실험 1: `top_k` 변화

`--top-k 1`, `2`, `3`, `5`를 실행하고 hit, Recall, MRR을 표로 기록하세요.

- `k`가 커지면 Recall은 유지되거나 증가할 수 있습니다.
- 첫 정답이 이미 top-k 안에 있으면 `k`만 늘려도 RR 값은 바뀌지 않습니다.
- 실제 RAG에서는 큰 `k`가 프롬프트 길이와 무관한 근거도 늘리므로 검색 지표만 보고
  무한히 키우면 안 됩니다.

### 실험 2: 더 촘촘한 threshold sweep

`0.08`부터 `0.24`까지 `0.02` 간격으로 평가하고 다음 열을 함께 그려 보세요.

- answerable Recall@3
- answerable coverage
- no-answer abstention accuracy
- 전체 coverage

선택한 threshold와 선택 이유를 “거짓 답변 1건의 비용이 불필요한 보류 몇 건과 같은가?”라는
제품 정책으로 설명해 보세요.

### 실험 3: hit와 Recall이 달라지는 `q-002`

각 threshold에서 `q-002`의 실제 검색 ID와 점수를 출력하세요. 정답 두 개 중 하나만
남을 때 hit는 1이지만 Recall은 0.5가 되는지 확인합니다. 다중 문서 근거를 요구하는 질문을
평가셋에 더 추가해 보세요.

### 실험 4: 순위만 바꾼 가짜 검색기

같은 정답 문서를 1위, 2위, 3위로 돌려주는 작은 `search` 함수를 만들어
`evaluate_retrieval()`에 전달하세요. hit와 Recall은 같은데 MRR만 `1`, `0.5`, `0.333`으로
변하는지 확인하면 지표의 책임이 분명해집니다.

### 실험 5: no-answer 평가셋 확장

`회사 전화번호`, `채용 공고`, `날씨`, `주식 가격`처럼 지식베이스 밖 질문을 최소 10개
추가하세요. 단순히 낯선 단어만 있는 질문뿐 아니라 “주소 변경”처럼 지식 문서의 단어와
겹치지만 실제 의도는 도메인 밖인 어려운 음성 사례도 포함하세요.

### 실험 6: 검색기 회귀 비교

현재 TF-IDF 설정을 기준선으로 저장하고 n-gram 범위, exact keyword bonus, 임베딩 검색,
청크 검색을 각각 바꿔 같은 평가셋으로 비교하세요. 평균이 좋아져도 핵심 질문의 개별 실패가
생길 수 있으므로 `query_id`별 결과도 함께 diff 하세요.

## 자주 발생하는 오류와 디버깅

### JSONL 로드 오류에 경로와 줄 번호가 표시됨

해당 줄이 JSON 객체인지, 쉼표나 따옴표가 올바른지, `relevant_ids`가 문자열이 아닌 배열인지
확인합니다. JSONL은 파일 전체가 하나의 배열이 아니라 **한 줄마다 독립 JSON 객체**입니다.

### `duplicate query_id`

복사한 사례의 `query_id`를 바꾸지 않은 경우입니다. 질문 문장이 달라도 ID가 같으면 회귀
기록이 충돌하므로 새 고유 ID를 부여합니다.

### MRR이 예상과 다름

- 순위를 0부터 세지 말고 `enumerate(..., start=1)`을 사용합니다.
- 첫 정답을 찾는 즉시 멈춥니다. 모든 정답의 reciprocal rank를 더하는 지표가 아닙니다.
- 순위 목록을 `set`으로 바꾸지 않습니다.
- top-k 밖의 정답은 0으로 처리합니다.
- miss도 answerable 평균의 분모에 포함합니다.

### hit rate와 Recall을 같은 공식으로 구현함

정답 문서가 항상 한 개면 두 값이 우연히 같습니다. `relevant_ids={A, B}`이고 하나만 검색된
fixture를 추가하면 hit는 1, Recall은 0.5여야 합니다.

### no-answer 때문에 0으로 나누거나 Recall이 낮아짐

`relevant_ids`가 빈 사례는 Recall 계산 전에 별도 분기합니다. answerable 검색 지표의 분모는
answerable 사례 수이고, 보류 정확도의 분모는 no-answer 사례 수입니다.

### coverage가 높은데 품질이 나쁨

정상입니다. coverage는 결과 존재 여부만 측정합니다. hit/Recall과 `no_answer_coverage`, 실패
목록을 함께 봐야 반환된 결과가 유용한지 알 수 있습니다.

### threshold를 바꿨는데 보류되지 않음

검색기가 threshold 미만 결과를 실제 반환 목록에서 제거하는지 확인합니다. 점수만 출력하고
결과 객체는 그대로 반환하면 평가기는 이를 covered로 계산합니다. 이 실습에서는
`TfidfRetriever.search()`가 `scores[index] >= min_score`인 결과만 반환합니다.

### 평균은 좋아졌는데 중요한 질문이 실패함

집계 지표는 어떤 사례가 변했는지 숨깁니다. `--show-failures`를 켜고 변경 전후의
`query_id`, 정답 ID, 실제 순위와 점수를 비교하세요. 중요한 질문에는 별도 회귀 테스트나
가중 정책을 둘 수 있지만, 일반 평균과 가중 점수의 의미는 구분해 보고해야 합니다.

### `n/a`가 출력됨

오류가 아니라 해당 지표의 분모가 0이라는 뜻일 수 있습니다. 예를 들어 no-answer 사례가
없는 평가셋에는 `abstention_accuracy`를 정의할 수 없습니다. 데이터 구성을 먼저 확인하세요.

## 완료 체크리스트

- [ ] JSONL 스키마, 빈 문자열, 중복 `query_id`를 검증했다.
- [ ] `reciprocal_rank()`가 1위, 2위, miss 사례에서 각각 1, 0.5, 0을 반환한다.
- [ ] hit@k와 Recall@k가 다중 정답 fixture에서 다른 값을 낼 수 있음을 확인했다.
- [ ] MRR이 첫 정답의 순위만 평가한다는 것을 설명할 수 있다.
- [ ] coverage가 정확도가 아니며 오답 결과도 포함한다는 것을 확인했다.
- [ ] answerable 지표와 no-answer abstention accuracy의 분모를 분리했다.
- [ ] 손계산 예제와 코드의 결과가 일치한다.
- [ ] `q-013`을 빈 `relevant_ids`인 no-answer 사례로 처리했다.
- [ ] 최소 세 개 이상의 threshold를 같은 평가셋에서 비교했다.
- [ ] 높은 threshold가 보류 정확도와 answerable Recall에 미치는 반대 효과를 확인했다.
- [ ] `--show-failures`로 변경 전후의 실패 `query_id`를 비교했다.
- [ ] 선택한 검색 설정과 지표를 회귀 기준선으로 재현할 수 있게 기록했다.
