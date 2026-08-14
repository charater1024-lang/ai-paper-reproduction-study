# NLP·검색·RAG 상세 학습 가이드

이 문서는 `projects/nlp/01~06`을 처음 공부하는 사람을 위한 통합 설명서입니다. 각 폴더의
README가 해당 프로젝트의 실습 지침이라면, 이 문서는 여섯 프로젝트가 하나의 NLP 시스템으로
어떻게 이어지는지를 설명합니다.

가장 중요한 원칙은 다음 세 가지입니다.

1. 완성 코드를 바로 외우지 말고 각 단계의 **입력과 출력**을 먼저 확인합니다.
2. 모델 점수만 보지 말고 실패한 실제 문장을 읽습니다.
3. RAG에서는 생성 결과보다 먼저 검색 근거와 출처를 검사합니다.

## 1. 이 트랙에서 만드는 전체 시스템

여섯 프로젝트를 모두 연결하면 아래와 같은 고객지원 NLP 시스템이 됩니다.

```text
사용자 문의
  │
  ├─ 정규화·PII 마스킹·토큰 관찰                       프로젝트 01
  │
  ├─ 문의 의도 분류(account/billing/delivery/...)      프로젝트 02
  │
  └─ 지식 문서 검색
       ├─ TF-IDF/SVD dense 검색                        프로젝트 03
       ├─ 직접 구현한 retrieve → context → prompt      프로젝트 04
       └─ LangChain Document/retriever/LCEL             프로젝트 05
                    │
                    └─ Recall·MRR·보류 임계값 평가      프로젝트 06
```

실제 서비스에서는 분류 결과로 검색 범위를 좁히거나 담당 팀을 선택할 수 있습니다. 예를 들어
`billing`으로 분류된 문의에는 결제 문서를 우선 검색하고, 검색 근거가 약하면 사람에게 넘길 수
있습니다. 이 저장소에서는 각 부품을 따로 관찰할 수 있도록 프로젝트를 분리했습니다.

## 2. 파일을 읽는 순서

각 프로젝트에는 다음 파일이 있습니다.

```text
README.md     개념, 실습 순서, 출력 해석, 실험 과제
starter.py   Config·Project class와 핵심 알고리즘 TODO가 남아 있는 연습용 코드
solution.py  같은 class/method 흐름으로 끝까지 실행되는 참고 코드
```

재사용 가능한 실제 구현은 `src/llm_engineering_lab/`에 있고, 동작 계약은 `tests/`에서 확인할
수 있습니다.

```text
projects/nlp/.../starter.py
        │ 직접 구현
        ▼
projects/nlp/.../solution.py ── 사용 ──> src/llm_engineering_lab/*.py
                                           │
                                           └── 검증 ──> tests/test_*.py
```

권장 순서는 다음과 같습니다.

1. 프로젝트 README를 읽습니다.
2. `starter.py`에서 Config, Project class, 함수의 인자와 반환형을 확인합니다.
3. TODO를 직접 구현하고 작은 입력으로 출력합니다.
4. 막힌 부분만 `solution.py`의 같은 이름 메서드 또는 대응 `src` 모듈에서 확인합니다.
5. 값을 하나 바꿔 보고 결과가 바뀐 이유를 기록합니다.
6. 마지막에 관련 테스트를 실행합니다.

## 3. 사용하는 데이터

### 고객 문의 분류 데이터

`data/customer_support_tickets.csv`에는 합성 고객 문의 100행이 있습니다.

- 고유 `case_id`: 50개
- 레이블: 5개, 각 20행
- 레이블: `account_access`, `billing`, `cancellation`, `delivery`, `technical_issue`
- 추가 정보: channel, priority, customer tier, 생성 시각

같은 `case_id`의 두 문장은 같은 사건을 다르게 표현한 유사 문장입니다. 따라서 행을 무작위로
나누면 거의 같은 문장이 train과 validation에 동시에 들어갈 수 있습니다. 프로젝트 02가
`case_id` 단위 분리를 강조하는 이유입니다.

### RAG 지식 문서

`data/raw/knowledge_base.jsonl`에는 계정, 결제, 배송, 반품, API 오류, 개인정보와 관련된
12개 문서가 있습니다. 한 줄이 하나의 JSON 객체이며 다음 필드를 가집니다.

```json
{
  "id": "kb-010",
  "title": "요청 한도 오류 429",
  "category": "technical",
  "content": "...",
  "keywords": ["429", "rate limit", "재시도"]
}
```

`id`는 검색 평가와 인용에 사용되므로 중복되면 안 됩니다. 제목과 키워드는 검색에는 유용하지만
최종 답변의 실제 근거는 본문인지도 따로 확인해야 합니다.

### RAG 평가 데이터

`data/practice/rag_queries.jsonl`에는 13개 질문이 있습니다.

- 답할 수 있는 질문: 12개
- 지식베이스로 답할 수 없는 질문: 1개(`q-013`, 회사 주소 질문)

`relevant_ids`가 빈 배열이면 검색기가 문서를 억지로 반환하지 않고 보류하는 것이 목표입니다.

## 4. 공통 용어

| 용어 | 이 트랙에서의 의미 |
|---|---|
| corpus | 처리하거나 검색할 전체 문서 모음 |
| document | 하나의 문의 또는 지식 문서 |
| token | 모델이 세는 기본 문자열 단위 |
| feature | 모델에 입력되는 수치 표현 |
| sparse vector | 대부분의 값이 0인 TF-IDF 벡터 |
| dense vector | 대부분의 차원이 실수 값을 갖는 작은 벡터 |
| embedding | 텍스트를 비교 가능한 수치 벡터로 바꾼 결과 |
| chunk | 검색 단위를 작게 만들기 위해 나눈 문서 조각 |
| index | 문서 벡터와 문서 id를 조회할 수 있게 저장한 구조 |
| retriever | 질문을 받아 관련 문서를 반환하는 인터페이스 |
| context | 검색 문서 중 모델 프롬프트에 실제로 넣는 부분 |
| grounded answer | 제공된 근거를 벗어나지 않는 답변 |
| abstention | 근거가 부족할 때 답변을 보류하는 동작 |

## 5. 프로젝트 01: 텍스트 전처리

### 왜 필요한가

컴퓨터에는 눈으로 같아 보이는 문자열이 다른 코드로 저장될 수 있습니다. 전각 영문 `ＮＬＰ`와
일반 영문 `NLP`, 여러 종류의 Unicode 공백이 대표적입니다. 이를 그대로 세면 같은 단어가 서로
다른 feature가 됩니다.

또한 이메일, 전화번호, 주문번호를 그대로 학습·로그에 남기면 개인정보 노출이나 과적합의 원인이
됩니다. 전처리의 목적은 텍스트를 무조건 짧게 만드는 것이 아니라 다음 조건을 만족하는 것입니다.

- 의미 있는 차이는 보존합니다.
- 표현상의 불필요한 차이는 통일합니다.
- 민감한 값은 종류만 남기고 실제 값은 제거합니다.
- 같은 입력은 항상 같은 출력으로 변환합니다.

### 처리 순서

```text
원문
  → mask_pii()        이메일/전화/긴 주문번호를 <EMAIL>/<PHONE>/<ORDER_ID>로 교체
  → normalize_text()  Unicode NFKC, 소문자화, 연속 공백 축소
  → tokenize()        한글·영문·숫자·마스크 토큰 추출
  → 빈도 계산         corpus 전체의 Counter 작성
```

`normalize_text()`는 NFKC를 사용합니다. NFKC는 호환 문자를 통합하는 데 편리하지만 모든 업무
도메인에 항상 정답은 아닙니다. 수학 기호나 법률 문서처럼 표기 형태 자체가 의미를 가질 때는
NFC와 비교해야 합니다.

현재 tokenizer는 형태소 분석기가 아닙니다. 예를 들어 `구독을`, `구독은`, `구독이`를 서로 다른
토큰으로 둘 수 있습니다. 설치 없이 동작하는 기준선이라는 장점이 있지만 조사·어미를 분리하지
못한다는 한계가 있습니다.

### 출력에서 볼 것

```text
[원문] ... Student@example.com / 010-1234-5678
[마스킹] ... <EMAIL> / <PHONE>
[정규화] ... <email> / <phone>
[토큰] ... '<EMAIL>', '<PHONE>'
```

정규화 과정에서 마스크가 소문자로 바뀌어도 tokenizer가 다시 표준 대문자 마스크로 반환합니다.
가장 중요한 검사는 원래 이메일과 전화번호가 마스킹 이후 문자열에 남아 있지 않은지 확인하는
것입니다.

### 해볼 실험

- NFKC 대신 NFC를 썼을 때 전각 문자 결과를 비교합니다.
- 마스킹 전후의 고유 토큰 수를 비교합니다.
- word token 대신 문자 2~5-gram을 세어 조사 변화에 얼마나 강한지 확인합니다.

## 6. 프로젝트 02: 의도 분류

### 문제 정의

입력은 고객 문의 한 문장이고 출력은 다섯 의도 중 하나입니다.

```text
"카드에서 같은 금액이 두 번 결제됐어요"
        ↓
      billing
```

### 학습 파이프라인

```text
CSV 로드·스키마 검증
  → case_id 단위 train/validation 분리
  → 문자 n-gram TF-IDF
  → channel/priority/tier one-hot encoding
  → LogisticRegression 학습
  → accuracy + macro F1 + 클래스별 지표 + 실제 오분류
```

전처리와 분류기를 하나의 scikit-learn `Pipeline`으로 묶으면 validation 데이터로 vocabulary를
미리 학습하는 누수를 막을 수 있고, 저장 후 추론에서도 같은 변환이 적용됩니다.

### 왜 문자 n-gram인가

한국어에서 `결제`, `결제됐어요`, `결제입니다`는 문자열 일부를 공유합니다. 문자 2~5-gram은
형태소 분석기 없이도 이런 공통 부분을 feature로 만들 수 있습니다. 반대로 너무 짧은 n-gram은
우연한 겹침을 많이 만들고, 너무 긴 n-gram은 새로운 표현에 약합니다.

TF-IDF의 직관은 다음과 같습니다.

```text
한 문서에서 자주 나오고(TF)
전체 문서 어디에나 나오지는 않는 단어일수록(IDF)
해당 문서를 구분하는 중요한 feature로 본다.
```

### 왜 macro F1을 보는가

accuracy는 전체 정답 수만 셉니다. macro F1은 각 클래스를 동일한 비중으로 평균하므로 작은
클래스가 무시되는 문제를 더 잘 드러냅니다. 현재 데이터는 균형이지만 실제 고객 데이터는 보통
불균형하므로 습관적으로 함께 확인하는 편이 좋습니다.

현재 기준선에서는 validation 20행에서 대략 다음 정도를 관찰할 수 있습니다.

```text
accuracy ≈ 0.800
macro F1 ≈ 0.793
```

숫자 자체보다 `인증 앱 복구`가 billing으로, `해외 결제 수수료`가 cancellation으로 잘못
분류되는 식의 실제 오류를 읽는 것이 중요합니다. 이는 공통 문구가 강한 feature가 되었거나
학습 예시가 부족하다는 신호일 수 있습니다.

### 해볼 실험

- `ngram_range`를 `(2, 5)`, `(3, 5)`, `(2, 6)`으로 비교합니다.
- metadata feature를 제거하고 macro F1 변화를 봅니다.
- row random split과 group-safe split 점수를 비교하되, 높은 점수가 누수 때문인지 확인합니다.

## 7. 프로젝트 03: 의미 검색

### sparse 검색에서 dense 공간으로

TF-IDF 문서 벡터는 vocabulary 크기만큼 차원이 있고 대부분이 0입니다. 프로젝트 03은
`TruncatedSVD`로 이 행렬을 작은 dense 공간에 투영합니다.

```text
문서 문자열
  → 문자 n-gram TF-IDF       shape = (문서 수, vocabulary 크기)
  → TruncatedSVD             shape = (문서 수, 작은 component 수)
  → L2 normalize
  → query와 cosine similarity
  → top-k 문서
```

문서-단어 행렬에서 함께 등장하는 패턴을 SVD가 낮은 차원의 축으로 압축하기 때문에 단순한 exact
match보다 잠재 관계를 일부 포착할 수 있습니다. 이를 Latent Semantic Analysis라고 부르기도
합니다.

다만 이 예제는 대규모 언어 데이터로 사전학습된 신경망 embedding이 아닙니다. 12개 문서에서
학습한 작은 corpus 전용 투영이므로, 여기서 배우는 핵심은 embedding 모델의 품질보다
`fit → encode → normalize → index → search`의 인터페이스입니다.

### cosine similarity

두 벡터의 방향이 얼마나 비슷한지를 측정합니다.

```text
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

코드가 query와 document를 미리 L2 norm 1로 만들기 때문에 실제 검색에서는 내적 `q @ d`가
cosine과 같습니다. `embedding_norms`가 1에 가까운지 보는 이유입니다.

### component 수 읽기

`--components 64`를 주더라도 12개 문서에서는 실제 차원이 12로 자동 축소됩니다.

```text
requested_components=64
effective_components=12
```

작은 행렬에 존재하지 않는 차원을 억지로 만들 수 없기 때문입니다. 요청값과 실제값을 따로
출력하면 설정이 조용히 바뀌는 일을 피할 수 있습니다.

### 해볼 실험

- component를 2, 4, 8, 12로 바꾸고 top-3와 점수를 기록합니다.
- 같은 질문을 프로젝트 04의 TF-IDF 검색기로 검색해 순위를 비교합니다.
- 문서 본문만 색인한 결과와 제목·keyword까지 포함한 결과를 비교합니다.

## 8. 프로젝트 04: 프레임워크 없는 RAG

### RAG의 실제 경계

RAG는 하나의 모델 이름이 아니라 검색과 생성을 연결하는 시스템 패턴입니다.

```text
질문
  → retrieve: 관련 문서와 점수
  → context: 길이 제한 안에서 출처와 본문 조립
  → prompt: 근거 사용·인용·보류 규칙 추가
  → generate: 선택한 LLM이 답변 생성
```

프로젝트 04는 마지막 LLM 호출 직전까지 구현합니다. 이렇게 하면 비용이나 네트워크 없이도
RAG에서 가장 먼저 검증해야 하는 검색 결과와 실제 prompt를 볼 수 있습니다. 검색 문서가 틀리면
아무리 좋은 LLM도 올바른 근거 기반 답변을 만들기 어렵습니다.

### `top_k`와 context budget

- `top_k`가 너무 작으면 필요한 근거를 놓칠 수 있습니다.
- `top_k`가 너무 크면 관련 없는 문서가 prompt를 오염시키고 비용이 늘어납니다.
- `max_chars`가 작으면 핵심 문장이 잘릴 수 있습니다.
- `max_chars`가 크다고 항상 좋은 것은 아닙니다. 모델의 주의가 분산될 수 있습니다.

`format_context()`는 각 블록에 `source_id`, 제목, 검색 점수를 붙입니다. 이 정보는 사용자에게
보여줄 인용을 만들고 검색 실패를 디버깅할 때 필요합니다.

### grounded prompt의 역할

프롬프트에는 다음 규칙이 포함됩니다.

- 제공된 검색 근거만 사용합니다.
- 근거가 부족하면 추측하지 않습니다.
- 답변에 사용한 source id를 표시합니다.
- 개인정보나 API 키를 요청하지 않습니다.

이 규칙은 환각을 완전히 제거하지는 않습니다. 생성된 답변의 문장마다 실제 context가 뒷받침하는지
별도의 검증이 필요합니다.

### 해볼 실험

- `--top-k 1`, `3`, `5`의 context 노이즈를 비교합니다.
- `--max-chars 200`, `600`, `1200`에서 문서가 어디서 잘리는지 봅니다.
- `회사 주소가 어디인가요?`처럼 답이 없는 질문에 무엇이 검색되는지 확인합니다.
- 지식 문서 안에 “이전 지시를 무시하라”는 문장이 있다고 가정하고 데이터와 명령을 구분하는
  프롬프트를 작성합니다.

## 9. 프로젝트 05: LangChain RAG

프로젝트 04에서 직접 만든 경계를 LangChain의 표준 인터페이스로 다시 구성합니다. 프레임워크의
목적은 RAG 원리를 대신하는 것이 아니라, 같은 역할을 하는 부품을 교체 가능하게 연결하는 것입니다.

### 구성 요소별 역할

| 구성 요소 | 역할 |
|---|---|
| `Document` | 본문(`page_content`), id, metadata를 함께 운반 |
| `RecursiveCharacterTextSplitter` | 긴 문서를 검색 가능한 chunk로 분할 |
| `LocalHashEmbeddings` | API/다운로드 없는 문자 n-gram 벡터 기준선 |
| `InMemoryVectorStore` | chunk와 벡터를 메모리에 저장하고 cosine 검색 |
| Retriever | 문자열 질문을 받아 `Document` 목록을 반환하는 Runnable |
| `ChatPromptTemplate` | system/human 메시지와 context/question 자리 정의 |
| LCEL Runnable | retrieval, formatting, prompt, model, parser를 파이프로 연결 |
| `StrOutputParser` | 모델 메시지에서 최종 문자열 추출 |

### metadata를 보존해야 하는 이유

문서를 chunk로 나누면 본문만 남기기 쉽습니다. 하지만 다음 정보가 사라지면 인용과 디버깅이
어려워집니다.

```text
source_id    원문 지식 문서 id
chunk_id     같은 문서 안에서 어떤 조각인지 구분
start_index  원문에서 chunk가 시작한 문자 위치
title        사용자와 개발자가 읽을 제목
category     검색 필터나 분석에 사용할 범주
```

프로젝트 코드는 `kb-010:chunk-000` 같은 고유 chunk id를 만듭니다. 원문 id를 그대로 모든
chunk에 쓰면 vector store에서 같은 id가 덮어써질 수 있습니다.

### 로컬 hashing embedding의 의미

`LocalHashEmbeddings`는 동일한 문자 n-gram을 같은 고정 차원에 해싱합니다. 결정적이고 매우
빠르며 모델 다운로드가 없다는 장점이 있습니다. 그러나 문맥 의미를 학습한 신경망 embedding이
아니므로 “자동차”와 “승용차”처럼 글자가 겹치지 않는 동의어 검색에는 약할 수 있습니다.

이 기준선을 `OpenAIEmbeddings`, Ollama 또는 Hugging Face embedding으로 교체해도 vector store
사용 코드는 거의 유지됩니다. 단, embedding 모델을 바꾸면 벡터 공간과 차원이 달라지므로 기존
index를 재사용하지 말고 다시 색인해야 합니다.

### LCEL 흐름

```text
question ──┬─> retriever ─> format_documents ─> context ─┐
           └───────────────────────────────> question ────┤
                                                         ▼
                                               ChatPromptTemplate
                                                         │
                                                         ▼
                                                     chat model
                                                         │
                                                         ▼
                                                  StrOutputParser
```

코드에서는 `RunnablePassthrough()`가 원래 질문을 그대로 전달하고, retriever 쪽 분기는 문서
목록을 context 문자열로 바꿉니다.

기본 `ask()`는 실제 LLM인 척하지 않습니다. 검색된 최상위 근거를 그대로 보여 주는 extractive
fallback을 사용합니다. `--openai`를 명시했을 때만 선택형 chat model을 연결합니다.

### 보류 기준

검색 결과가 있어도 `min_score`보다 낮으면 근거로 받아들이지 않습니다. 점수는 절대적인 진실
확률이 아니며 embedding과 corpus가 바뀌면 다시 조정해야 합니다. 대표적인 answerable/no-answer
validation set으로 coverage와 오답률을 함께 측정해야 합니다.

### 해볼 실험

- `--min-score`를 0.05, 0.10, 0.20으로 바꾸고 보류 여부를 비교합니다.
- `chunk_size`와 `chunk_overlap`을 바꾸고 chunk 수와 source metadata를 확인합니다.
- `FakeListChatModel`로 API 없는 LCEL 테스트를 작성합니다.
- 실제 embedding을 연결한 뒤 같은 13개 평가 질의의 순위를 비교합니다.

## 10. 프로젝트 06: RAG 평가

좋아 보이는 답변 몇 개만 읽어서는 검색기의 품질을 알 수 없습니다. 고정 평가 질문과 정답 문서
id를 이용해 설정 변경 전후를 같은 기준으로 비교해야 합니다.

### Hit@k와 Recall@k

정답 문서가 top-k 안에 하나라도 있으면 query hit입니다. relevant 문서가 여러 개라면 recall은
몇 개를 찾았는지까지 봅니다.

```text
정답: [A, B]
검색 top-3: [X, A, C]

Hit@3 = 1                 하나 이상 찾음
Recall@3 = 1 / 2 = 0.5    정답 두 개 중 한 개를 찾음
```

현재 report의 `hit_rate_at_k`는 answerable query 중 hit한 비율이고, `recall_at_k`는 query별
recall을 평균한 값입니다.

### MRR

첫 정답 문서가 위에 있을수록 높은 점수를 줍니다.

```text
첫 정답 순위 1위 → reciprocal rank = 1/1 = 1.0
첫 정답 순위 2위 → reciprocal rank = 1/2 = 0.5
첫 정답 순위 3위 → reciprocal rank = 1/3 ≈ 0.333
정답 없음        → 0
```

모든 answerable query의 reciprocal rank 평균이 MRR입니다. Recall@3가 같아도 정답을 늘 1위에
놓는 검색기의 MRR이 더 높습니다.

### coverage와 no-answer

`coverage`는 전체 질문 중 검색 결과를 하나 이상 반환한 비율입니다. coverage가 높다고 무조건
좋은 것은 아닙니다. 답이 없는 질문에도 관련 없는 문서를 반환하면 환각의 출발점이 됩니다.

```text
abstention accuracy = no-answer 질문 중 빈 결과를 반환한 비율
no_answer_coverage  = no-answer 질문 중 문서를 반환한 비율
```

두 값은 현재 이진 보류 규칙에서는 합이 1입니다.

### 실제 threshold trade-off 읽기

현재 TF-IDF 기준선에서 관찰되는 예시는 다음과 같습니다.

| min_score | Recall@3 | MRR@3 | Coverage | no-answer 보류 정확도 |
|---:|---:|---:|---:|---:|
| 0.01 | 1.000 | 0.944 | 1.000 | 0.000 |
| 0.10 | 0.917 | 0.917 | 0.923 | 0.000 |
| 0.20 | 0.875 | 0.917 | 0.846 | 1.000 |

0.01에서는 모든 정답 문서를 잘 찾지만 `회사 주소` 질문에도 엉뚱한 문서를 반환합니다. 0.20은
그 질문을 올바르게 보류하지만 `파손 상품 사진` 같은 실제 질문도 놓칠 수 있습니다. 따라서
임계값 선택은 “가장 높은 단일 지표”가 아니라 제품에서 허용 가능한 coverage와 위험의 균형입니다.

### generation 평가는 별도다

검색 Recall이 높아도 모델이 근거를 잘못 요약하거나 인용 id를 틀릴 수 있습니다. 반대로 검색이
틀렸는데 문장이 자연스럽다고 좋은 RAG가 아닙니다. 다음을 분리해 측정하세요.

- retrieval: Recall@k, MRR, latency, no-answer 보류
- context: 정답 근거 포함 여부, 길이, 중복
- generation: answer correctness, groundedness, citation validity

## 11. 권장 학습 일정

| 일차 | 내용 | 남길 결과 |
|---:|---|---|
| 1일 | 프로젝트 01 | 원문/마스킹/토큰 비교, 전처리 한계 2개 |
| 2일 | 프로젝트 02 | metric 표, 오분류 3개와 원인 가설 |
| 3일 | 프로젝트 03 | component별 top-k 비교표 |
| 4일 | 프로젝트 04 | top-k/context budget 실험 |
| 5일 | 프로젝트 05 | LangChain component 연결 그림, prompt preview |
| 6일 | 프로젝트 06 | threshold별 Recall/coverage/no-answer 표 |
| 7일 | 통합 | 새 질문 10개, 실패 단계 분류와 개선안 |

시간이 부족하다면 01 → 04 → 05 → 06 순서로 먼저 진행하고, 분류와 dense 검색을 나중에
보충해도 됩니다.

## 12. 실험 기록 예시

```text
가설:
min_score를 높이면 no-answer 보류 정확도는 오르지만 answerable coverage는 떨어질 것이다.

변경한 한 가지:
TfidfRetriever의 min_score를 0.01에서 0.20으로 변경했다.

고정한 조건/seed:
동일한 12개 KB 문서, 13개 평가 질문, top_k=3을 사용했다.

결과(metric + 실패 예시):
no-answer 정확도 0.0 → 1.0, Recall@3 1.0 → 0.875.
q-013은 보류했지만 q-008도 검색 결과가 없어졌다.

다음 결정:
하나의 전역 threshold 대신 category별 threshold 또는 reranker를 검토한다.
```

## 13. 자주 묻는 질문

### 한국어 형태소 분석기를 왜 사용하지 않나요?

설치 부담 없이 Windows/CPU에서 바로 실행하고, 전처리와 검색의 기본 경계를 먼저 관찰하기
위해서입니다. 실제 프로젝트에서는 Kiwi, MeCab 계열 또는 subword tokenizer를 후보로 추가하고
현재 문자 n-gram 기준선과 같은 평가셋에서 비교하세요.

### SVD 검색이 진짜 semantic embedding인가요?

작은 corpus 안의 공기 패턴을 압축한다는 의미에서는 latent semantic 표현이지만, 대규모
사전학습 모델의 언어 이해와 같지는 않습니다. 이 프로젝트에서는 dense index의 구조와 평가법을
익히는 로컬 기준선으로 사용합니다.

### LangChain을 쓰면 직접 구현 RAG는 몰라도 되나요?

아닙니다. 프레임워크는 부품 연결과 교체를 편하게 하지만, 잘못된 chunk·검색 결과·threshold를
자동으로 고쳐 주지 않습니다. 프로젝트 04에서 각 경계를 이해한 뒤 05로 넘어가는 것을 권장합니다.

### API 키가 꼭 필요한가요?

아닙니다. 모든 핵심 실습과 LangChain retrieval/LCEL 테스트는 로컬에서 실행됩니다. 실제 생성
모델을 비교하고 싶을 때만 provider integration과 환경 변수에 저장한 키를 사용합니다.

### 점수가 높으면 답이 맞다는 뜻인가요?

아닙니다. 검색 점수는 특정 vectorizer/embedding 공간에서의 유사도입니다. corpus, chunk 방식,
모델이 바뀌면 분포도 달라집니다. 사람이 검토한 평가셋으로 임계값을 다시 정해야 합니다.

## 14. 완료 체크리스트

- [ ] 원문과 전처리 결과를 비교하고 정보 손실 한 가지를 설명할 수 있다.
- [ ] group-safe split이 필요한 이유를 데이터 예시로 설명할 수 있다.
- [ ] TF-IDF sparse 벡터와 SVD dense 벡터의 차이를 설명할 수 있다.
- [ ] 검색 실패와 생성 실패를 서로 분리해 진단할 수 있다.
- [ ] LangChain의 Document, splitter, vector store, retriever 역할을 구분할 수 있다.
- [ ] context에 source/chunk metadata가 남아 있는지 확인할 수 있다.
- [ ] Recall@k와 MRR을 작은 예제로 직접 계산할 수 있다.
- [ ] threshold가 Recall, coverage, no-answer에 미치는 trade-off를 설명할 수 있다.
- [ ] 적어도 한 프로젝트에서 설정을 하나 바꾸고 실험 기록을 남겼다.
- [ ] 전체 테스트를 실행해 코드 변경이 다른 프로젝트를 깨뜨리지 않았는지 확인했다.

## 15. 다음 확장 순서

기본 트랙을 마친 뒤에는 아래 순서가 자연스럽습니다.

1. 실제 다국어 embedding 모델과 로컬 hashing/SVD 기준선 비교
2. BM25 + dense 검색의 hybrid/RRF
3. cross-encoder reranker
4. PDF/Markdown ingestion과 문서 version 관리
5. 인용 정확도와 answer groundedness 평가
6. LangGraph를 이용한 query rewrite·검색 검증·재검색
7. Streamlit 화면에 검색 점수, context, 인용, 보류 사유 표시

관련 심화 노트북은 17(Embedding과 Vector Index), 18(End-to-End RAG), 22(Hybrid RAG 평가)입니다.
