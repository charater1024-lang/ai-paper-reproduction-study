# 03. 의미 검색과 벡터 인덱스

이 프로젝트는 한국어 고객지원 문서를 **문자 TF-IDF 희소 벡터**로 표현한 뒤,
`TruncatedSVD`로 낮은 차원의 **밀집(dense) 벡터**로 투영하고 cosine similarity로
검색합니다. 실제 임베딩 API나 벡터 데이터베이스를 쓰기 전에
`fit → transform → reduce → normalize → top-k` 경계를 눈으로 확인하는 실습입니다.

> 이 예제의 밀집 벡터는 신경망 임베딩이 아니라 TF-IDF에 잠재 의미 분석(LSA)을
> 적용해 만든 로컬 벡터입니다. 따라서 “밀집 벡터 검색의 구조”를 익히는 데 초점이
> 있으며, 범용 임베딩 모델과 같은 의미 이해 능력을 기대하면 안 됩니다.

예상 소요 시간은 60~90분이며 외부 API나 API 키는 필요하지 않습니다.

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

실습을 마치면 다음을 설명하고 직접 구현할 수 있어야 합니다.

- 희소 벡터와 밀집 벡터가 무엇이며, 메모리와 표현력 측면에서 어떻게 다른지 설명한다.
- 문자 n-gram TF-IDF 행렬을 만들고 `TruncatedSVD`로 안전하게 차원을 줄인다.
- 문서와 질문에 **같은** 전처리·벡터화·SVD 파이프라인을 적용한다.
- L2 정규화된 벡터의 내적이 cosine similarity와 같음을 코드로 확인한다.
- `top_k`, `min_score`, SVD 차원, seed가 검색 순위에 미치는 영향을 실험한다.
- 벡터 shape, norm, 문서 ID와 행 인덱스의 매핑을 점검해 검색 오류를 디버깅한다.

## 먼저 알아둘 핵심 개념

### 1. 희소 벡터와 밀집 벡터

TF-IDF는 말뭉치에서 만든 어휘의 각 항목을 한 차원으로 사용합니다. 현재 구현은
형태소 대신 `char_wb` 문자 n-gram `(2, 5)`를 쓰므로, 예를 들어 “중복 결제”에서
2~5글자 조각을 특징으로 만듭니다. 한 문서는 전체 특징 중 일부만 포함하므로 대부분의
값이 0인 **희소 행렬**이 됩니다.

`TruncatedSVD`는 이 큰 행렬을 더 작은 잠재 축으로 투영합니다. 변환된 벡터는 대부분의
차원에 0이 아닌 값이 들어가는 **밀집 벡터**입니다.

| 구분 | TF-IDF 희소 벡터 | SVD 밀집 벡터 |
|---|---|---|
| 차원의 의미 | 문자 n-gram 하나하나 | 여러 n-gram이 섞인 잠재 축 |
| 차원 수 | 최대 20,000 | `effective_components` |
| 0의 비율 | 높음 | 대체로 낮음 |
| 해석 | 어떤 문자열 특징이 겹쳤는지 비교적 명확 | 각 축의 직접 해석은 어려움 |
| 이 실습의 역할 | 원본 lexical 신호 | 압축된 검색용 표현 |

이 방식은 표면 문자열이 완전히 같지 않아도 함께 등장하는 특징을 잠재 축에서 묶을 수
있지만, 작은 데이터에서는 SVD가 우연한 상관관계도 함께 학습할 수 있습니다.

### 2. SVD 차원을 안전하게 정하기

요청한 `n_components`가 문서 수나 TF-IDF 특징 수보다 크면 작은 연습 말뭉치에서
분해가 실패할 수 있습니다. 현재 `DenseSemanticRetriever.fit()`은 다음처럼 실제
차원을 낮춥니다.

```python
effective_components = min(
    requested_components,
    document_count,
    vocabulary_size,
)
```

기본 데이터는 문서 12개이므로 `--components 64`를 지정해도 실제 벡터는 12차원입니다.
`requested_components`와 `effective_components`를 함께 출력하는 이유가 바로 이것입니다.

### 3. cosine similarity와 L2 정규화

두 벡터 `a`, `b`의 cosine similarity는 방향이 얼마나 비슷한지 측정합니다.

$$
\operatorname{cos}(a,b)=
\frac{a^\top b}{\lVert a\rVert_2\lVert b\rVert_2}
$$

```text
cosine(a, b) = (a · b) / (||a||₂ × ||b||₂)
```

문서 벡터와 질문 벡터의 L2 norm을 미리 1로 만들면 분모가 1이므로 단순 내적
`document_embeddings @ query_embedding`이 cosine 점수가 됩니다. 점수 범위는
이론상 -1~1이며, **확률이 아닙니다**. 현재 검색기는 부동소수점 오차를 막기 위해 점수를
`[-1, 1]`로 clip합니다.

### 4. 한국어를 문자 n-gram으로 다루는 이유

형태소 분석기 설치 없이도 “결제”, “결제됐어요”, “중복결제” 사이의 문자 조각을 공유할
수 있기 때문입니다. 현재 전처리 함수 `normalize_semantic_text()`는 다음을 수행합니다.

- Unicode를 NFKC로 정규화하고 소문자로 바꾼다.
- 앞뒤 공백을 제거하고 연속 공백을 하나로 합친다.
- `429`, `401`처럼 실제 3자리 HTTP 상태 코드로 보이는 숫자를 한 번 더 붙여
  짧은 오류 코드가 긴 문서에서 묻히지 않게 한다.
- `TfidfVectorizer`는 `analyzer="char_wb"`, `ngram_range=(2, 5)`,
  `sublinear_tf=True`, `max_features=20_000`을 사용한다.

## 데이터와 코드 흐름

완성 예제의 흐름은 다음과 같습니다.

```text
data/raw/knowledge_base.jsonl
  ↓ load_knowledge_base(): 필수 필드와 중복 ID 검증
KnowledgeArticle 12개
  ↓ searchable_text = title + category + keywords + content
문자 TF-IDF 희소 행렬 (문서 수 × 어휘 수)
  ↓ TruncatedSVD(random_state=42, n_iter=7)
밀집 행렬 (문서 수 × effective_components)
  ↓ 행별 L2 정규화
문서 임베딩 + 문서 행/ID 매핑

질문 문자열
  ↓ 같은 vectorizer.transform()
  ↓ 같은 reducer.transform()
  ↓ L2 정규화
질문 임베딩
  ↓ 문서 행렬 @ 질문 벡터
stable 내림차순 정렬 → min_score 필터 → SearchResult top-k
```

각 `SearchResult`에는 `article`, `score`, `rank`가 들어 있습니다. `article`은 다시
`id`, `title`, `category`, `content`, `keywords`를 가집니다. 문서 행 순서와
`self.articles` 순서를 함께 보존해야 점수와 실제 문서가 뒤섞이지 않습니다.

관련 파일은 다음과 같습니다.

- `starter.py`: 문자열 목록으로 작은 `DenseSearchIndex`를 직접 구현하는 연습용 뼈대
- `solution.py`: 재사용 모듈 `DenseSemanticRetriever`를 데이터셋에 적용하는 실행 예제
- `src/llm_engineering_lab/semantic_search.py`: 전처리, SVD, 진단, 검색의 참고 구현
- `src/llm_engineering_lab/retrieval.py`: `KnowledgeArticle`, `SearchResult`, JSONL 로더
- `data/raw/knowledge_base.jsonl`: 12개의 고객지원 지식 문서

## 실행 방법

저장소 루트에서 실행합니다.

```powershell
# 기본 질문
.\.venv\Scripts\python.exe projects\nlp\03_semantic_search\solution.py

# 질문, 검색 개수, 요청 SVD 차원 변경
.\.venv\Scripts\python.exe projects\nlp\03_semantic_search\solution.py `
  --query "API 요청에서 429 오류가 나요" `
  --top-k 3 `
  --components 8

# 옵션 확인
.\.venv\Scripts\python.exe projects\nlp\03_semantic_search\solution.py --help
```

## `starter.py` TODO 단계별 힌트

`starter.py`는 완성 모듈을 그대로 복사하는 문제가 아닙니다. `fit(document_ids,
documents)`가 같은 길이의 두 시퀀스를 받고, `search()`가 `(source_id, 점수)` 튜플 목록을
반환하는 더 작은 인덱스를 직접 설계하는 연습입니다. `vectorizer`, `reducer`, ID 목록,
문서 임베딩 상태는 skeleton에 명시되어 있습니다.

### TODO 1. 문자 TF-IDF 행렬 학습

1. 입력을 `list(documents)`로 materialize하고 빈 목록을 거부합니다.
2. `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), ...)`를 만듭니다.
3. `fit_transform()`으로 문서-특징 희소 행렬을 얻습니다.
4. 빈 문자열뿐인 문서나 빈 어휘가 들어왔을 때 어떤 오류를 보여 줄지도 정합니다.

확인할 값은 `term_matrix.shape == (문서 수, 어휘 수)`입니다. 질문에는 절대로 다시
`fit_transform()`을 호출하지 않습니다.

### TODO 2. 안전한 `TruncatedSVD`

요청 차원을 고정 숫자로 쓰지 말고 `문서 수`, `특징 수`, `요청 차원`의 최솟값으로
계산합니다. 재현 가능한 결과를 위해 `random_state`도 고정합니다. 그런 다음
`fit_transform(term_matrix)`로 문서 밀집 행렬을 만듭니다.

### TODO 3. 정규화와 문서 매핑 보존

`sklearn.preprocessing.normalize(..., norm="l2")`로 각 **행**을 정규화합니다.
문서 벡터의 `axis=1` norm이 1에 가까운지 확인하세요. 원래 문서 목록도 같은 순서로
저장해야 검색 결과 행 인덱스를 source ID로 되돌릴 수 있습니다. `fit()` 마지막에는
메서드 체이닝이 가능하도록 `self`를 반환합니다.

### TODO 4. 질문을 같은 파이프라인으로 변환

먼저 `fit()`이 호출되었는지 확인합니다. 질문은 이미 학습된
`vectorizer.transform([query]) → reducer.transform(...) → L2 normalize` 순서로
변환합니다. 질문에서 새 어휘를 학습하면 문서와 질문의 좌표계가 달라집니다.

### TODO 5. cosine 순위와 안정적인 top-k

정규화된 문서 행렬과 질문 벡터를 내적해 점수를 얻습니다. 내림차순 인덱스를 만든 뒤
`top_k`개만 고르고, 저장한 문서 목록과 점수를 튜플로 묶습니다. 동점일 때 입력 문서
순서를 보존하려면 stable 정렬을 사용하세요. `top_k <= 0`, 빈 질문, 모든 특징이
OOV(out-of-vocabulary)인 질문에 대한 정책도 테스트해 보세요.

구현 중에는 다음 작은 점검 코드를 추가하면 좋습니다.

```python
assert document_embeddings.shape[0] == len(documents)
assert document_embeddings.shape[1] == reducer.n_components
```

## starter와 `solution.py` 1:1 코드 지도

| 단계 | starter/solution 코드 | 입력 → 출력 shape | 왜 분리했는가 |
|---|---|---|---|
| 설정 | `SearchProjectConfig` | 경로·질문·k·요청 차원 | 재현할 검색 조건을 한 레코드에 고정한다 |
| 데이터 | `load_articles()` | JSONL 12행 → article 12개 | ID 중복을 벡터 행 매핑 전에 차단한다 |
| 인덱스 | starter `DenseSearchIndex.fit()` / solution `build_index()` | `X:(12,V)` → `Z:(12,r)` | fit 상태와 원본 ID 순서를 같은 객체가 소유한다 |
| 진단 | `print_diagnostics()` | shape·norm·V·요청/실제 r | 작은 데이터에서 차원이 자동 축소된 사실을 숨기지 않는다 |
| 질문 변환 | `embed_query()` | 질문 1개 → `q:(r,)` | 문서에 fit한 vectorizer와 SVD를 그대로 재사용한다 |
| 순위 | `search()` | `Zq` → `SearchResult[k]` | 점수와 source ID를 함께 반환하고 동률 순서를 고정한다 |
| 실행 | `run()` | config → ranked results | 진단을 결과보다 먼저 출력해 검색 오류를 추적한다 |

starter의 `DenseSearchIndex`에는 `vectorizer`, `reducer`, `document_embeddings`,
`document_ids` 상태가 명시되어 있습니다. TODO 1~5에서 이 상태를 직접 채우면 solution의
`DenseSemanticRetriever` 내부 단계와 대응합니다. 특히 `embed_query()`에서 새 TF-IDF를
fit하지 말아야 문서 행렬의 열 \(V\)와 질문 벡터의 열이 같은 의미를 갖습니다.

solution의 `build_index()`는 `DenseSemanticRetriever.fit()` 뒤
`embedding_diagnostics.shape[0] == len(articles)`와 모든 norm이 1인지 assertion으로
검사합니다. `search()`는 점수가 실제 내림차순인지 다시 검사합니다. 단지 실행되는 코드가
아니라 **shape·정규화·순위 불변 조건을 스스로 증명하는 코드**로 만든 이유입니다.

`Path(__file__).resolve().parents[3]`으로 저장소 루트를 계산하므로 Windows의 어느 작업
디렉터리에서도 같은 JSONL을 찾습니다. `--query`, `--top-k`, `--components`를 바꾸면 모두
`SearchProjectConfig`에 기록됩니다. starter는 TODO가 남아 있으면 해당 번호를 출력하고
정상 종료합니다.

## 예상 출력과 해석

기본 명령의 대표 출력은 다음과 같습니다. 라이브러리 버전에 따라 마지막 소수점은 조금
달라질 수 있습니다.

```text
documents=12, dimensions=12, vocabulary=2673
requested_components=64, effective_components=12
embedding_norm: min=1.000, max=1.000
query_shape=(12,), query_norm=1.000

질문: 카드가 이중으로 결제됐어요
1. [kb-003] 이중 결제 확인 및 환불 cosine=0.869
2. [kb-005] 환불 처리 기간 cosine=0.587
3. [kb-008] 파손 상품 교환 cosine=0.366
```

- 문서가 12개이므로 요청 차원 64가 실제 12차원으로 조정되었습니다.
- 문서와 질문 norm이 모두 1이므로 출력 점수는 정규화된 벡터의 내적입니다.
- `kb-003`이 정답에 가까운 1위인 것은 좋지만, 2·3위가 모두 질문에 필요한 근거라는
  뜻은 아닙니다. 점수는 확률이나 정답 보장이 아니라 **현재 인덱스 안에서의 유사도**입니다.
- 작은 말뭉치를 최대 12차원으로 투영했으므로 우연한 잠재 상관도 순위에 영향을 줍니다.
  이후 RAG에서는 `min_score`, 재순위화, 평가 데이터로 노이즈를 관리해야 합니다.

`429` 질문에서는 보통 다음 순서가 나옵니다.

```text
1. [kb-010] 요청 한도 오류 429 cosine=0.864
2. [kb-009] API 인증 오류 401 cosine=0.591
3. [kb-011] 서비스 장애 확인 절차 cosine=0.097
```

전처리에서 3자리 상태 코드를 한 번 더 강조했기 때문에 `kb-010`이 강하게 검색됩니다.

## 실험 과제

각 실험은 한 번에 한 조건만 바꾸고 질문·seed·평가 기준을 기록하세요.

1. **SVD 차원 비교**: `--components 2`, `4`, `8`, `64`를 비교합니다. top-3 순위와
   1위/2위 점수 차이가 어떻게 변하는지 기록하고, 64가 실제 몇 차원이 되는지 설명합니다.
2. **질문 표현 변화**: “중복 결제”, “같은 돈이 두 번 빠져나감”, “duplicate charge”를
   검색합니다. 문자 n-gram이 잘 처리하는 변화와 실패하는 변화를 분류합니다.
3. **오류 코드 보존 효과**: `429`, `401`, “요청 한도” 질문을 비교합니다.
   `normalize_semantic_text()`가 숫자 코드를 재첨부하지 않는 가상 버전과 순위를 비교해도
   좋습니다.
4. **희소 검색과 비교**: 프로젝트 04의 `TfidfRetriever`로 같은 질문을 검색하고
   top-k 문서 ID를 나란히 적습니다. 어느 쪽의 실패가 lexical mismatch 때문인지,
   어느 쪽이 SVD 압축 때문인지 추론합니다.
5. **임계값 실험**: `DenseSemanticRetriever(min_score=0.1/0.3/0.6)`를 직접 만들어
   반환 문서 수와 정답 누락을 관찰합니다. 높은 임계값이 항상 좋은 것은 아닌 이유를
   설명합니다.
6. **재현성 확인**: 같은 `random_state`로 두 번 학습해 벡터와 순위를 비교하고, seed를
   바꿔 작은 말뭉치에서 변화가 있는지 확인합니다.

## 자주 생기는 오류와 디버깅

### `RuntimeError: call fit() before ...`

`search()`, `embed_query()`, 진단 속성보다 먼저 `fit()`을 호출해야 합니다. 온라인 질문
처리 때마다 다시 fit하지 말고, 문서가 바뀔 때 인덱스를 다시 만드는 구조가 일반적입니다.

### `ValueError: ... components ...` 또는 SVD shape 오류

요청 차원을 그대로 쓰지 않았는지 확인합니다. 차원은 문서 수와 특징 수 이하로 안전하게
조정해야 하며, 현재 참고 구현은 그 값을 `effective_components`로 보여 줍니다.

### 빈 질문에서 오류가 남

현재 `DenseSemanticRetriever`는 빈 문자열이나 공백뿐인 질문을 `ValueError`로
거부합니다. 반면 프로젝트 04의 `TfidfRetriever`는 빈 목록을 반환합니다. 두 검색기의
계약이 다르므로 호출부에서 혼동하지 마세요.

### 검색 결과가 빈 목록임

질문의 모든 특징이 학습 어휘에 없으면 정규화된 질문 벡터가 0이 되어 빈 목록이
반환됩니다. 또는 모든 점수가 `min_score`보다 낮을 수 있습니다. 질문 벡터 norm,
`vectorizer.vocabulary_`, 임계값을 순서대로 확인하세요.

### norm이 1이 아님

SVD 전 희소 행렬만 정규화했거나, 문서에는 정규화를 하고 질문에는 빼먹었을 수 있습니다.
`np.linalg.norm(embeddings, axis=1)`과 질문 norm을 각각 출력해 보세요. 완전히 0인 벡터는
정규화해도 norm이 0입니다.

### 문서 제목과 점수가 맞지 않음

점수를 정렬한 뒤 정렬 인덱스가 아니라 단순 순번으로 원본 문서를 꺼낸 경우가 많습니다.
`ordered_indices`의 각 값을 문서 목록과 점수 배열 양쪽에 똑같이 적용하세요.

### 실행할 때 모듈이나 데이터 파일을 찾지 못함

저장소 루트에서 `.venv`의 Python으로 실행했는지 확인합니다. 필요하면 먼저
`.\.venv\Scripts\python.exe -m pip install -e ".[all]"`을 실행하세요. 데이터 경로는
현재 작업 폴더가 아니라 `ROOT`를 기준으로 만드는 것이 안전합니다.

## 완료 체크리스트

- [ ] `starter.py`에서 TF-IDF 학습과 질문 `transform`을 구분했다.
- [ ] SVD 실제 차원을 문서 수·어휘 수·요청 차원의 최솟값으로 정했다.
- [ ] 문서와 질문 벡터를 L2 정규화하고 norm을 확인했다.
- [ ] 벡터 shape의 각 축과 문서 행/ID 매핑을 설명할 수 있다.
- [ ] stable top-k 정렬과 `top_k <= 0` 검증을 구현했다.
- [ ] 같은 seed에서 같은 검색 순위가 나오는지 확인했다.
- [ ] `중복 결제`, `429`, `계정 삭제` 질문을 실행해 결과를 해석했다.
- [ ] 프로젝트 04의 희소 검색과 순위가 다른 사례를 하나 이상 기록했다.
- [ ] 유사도 점수가 확률이 아니며, 높은 순위도 정답을 보장하지 않음을 설명할 수 있다.
