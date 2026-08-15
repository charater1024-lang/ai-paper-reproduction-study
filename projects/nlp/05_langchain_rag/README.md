# 05. LangChain v1 기반 2-step RAG

이 실습에서는 고객지원 지식 문서 12개를 검색 가능한 청크로 만들고, 질문과 가까운
근거를 찾아 답변에 주입하는 **2-step RAG(Retrieval-Augmented Generation)** 를
구현합니다. 첫 번째 단계에서 항상 검색하고, 두 번째 단계에서 검색된 문맥으로 답을
만들기 때문에 실행 순서와 비용을 예측하기 쉽습니다.

기본 경로는 완전히 오프라인입니다. 다운로드가 필요 없는 로컬 문자 해시 임베딩과
추출형 답변을 사용하므로 API 키 없이 `Document`부터 검색, 프롬프트, 보류까지 전체
흐름을 관찰할 수 있습니다. 원하면 마지막 생성 단계만 `ChatOpenAI`로 교체할 수
있습니다.

> 이 프로젝트는 LangChain v1의 작은 구성 요소와 Runnable/LCEL을 사용합니다.
> `RetrievalQA`, `LLMChain` 같은 이전의 큰 체인 클래스에 의존하지 않습니다.

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

실습을 마치면 다음 질문에 코드로 답할 수 있어야 합니다.

- `Document.page_content`, `Document.id`, `Document.metadata`는 각각 무엇을 담아야 하는가?
- 원본 문서를 왜 청크로 나누며, `chunk_size`와 `chunk_overlap`은 검색에 어떤 영향을 주는가?
- 임베딩, 벡터 저장소, Retriever의 책임은 어떻게 다른가?
- LCEL의 `|`, `RunnablePassthrough`, `RunnableLambda`가 데이터를 어떻게 전달하는가?
- 검색 점수가 낮을 때 모델을 호출하지 않고 보류해야 하는 이유는 무엇인가?
- 로컬 기준선을 실제 채팅 모델로 바꿔도 나머지 파이프라인을 유지하려면 어떻게 설계해야 하는가?

## 준비와 실행

저장소 루트에서 가상 환경과 의존성을 준비한 뒤 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[rag,dev]"
.\.venv\Scripts\python.exe projects\nlp\05_langchain_rag\solution.py --show-prompt
```

주요 옵션은 다음과 같습니다.

```text
--query       검색하고 답할 질문
--top-k       검색할 최대 청크 수(기본 3)
--min-score   답변 근거로 받아들일 최소 유사도(기본 0.08)
--show-prompt 모델 호출 전 완성된 프롬프트 출력
--openai      로컬 추출형 답변 대신 ChatOpenAI 호출
--model       사용할 채팅 모델 ID
```

## 전체 파이프라인

인덱싱은 보통 문서가 바뀔 때 수행하고, 질문 처리는 사용자 요청마다 수행합니다.

```text
[인덱싱]
knowledge_base.jsonl
  -> Document 12개
  -> RecursiveCharacterTextSplitter
  -> LocalHashEmbeddings
  -> InMemoryVectorStore

[질문 처리]
질문
  -> 질문 임베딩
  -> Retriever / 유사도 검색(top-k)
  -> min_score 기준 근거 선별
  -> 문서와 metadata를 context 문자열로 포맷
  -> ChatPromptTemplate
  -> 로컬 추출형 답변 또는 선택형 ChatOpenAI
  -> 답변 + sources + scores + abstained
```

이 프로젝트의 `solution.py`는 실행용 진입점이고, 재사용 가능한 실제 구현은
`src/llm_engineering_lab/langchain_rag.py`에 있습니다. 두 파일을 나란히 읽으면
“애플리케이션 코드”와 “재사용 모듈”을 분리하는 방식도 익힐 수 있습니다.

## 구성 요소별 상세 설명

### 1. `Document`와 metadata

`load_jsonl_documents()`는 JSONL의 각 지식 문서를 LangChain `Document`로 바꿉니다.

```python
Document(
    id="kb-010",
    page_content="제목: 요청 한도 오류 429\n...\n키워드: 429, rate limit, 재시도",
    metadata={
        "source": ".../knowledge_base.jsonl",
        "source_id": "kb-010",
        "title": "요청 한도 오류 429",
        "category": "technical",
    },
)
```

- `page_content`는 임베딩하고 모델에 전달할 본문입니다. 이 예제는 검색 신호를 늘리기
  위해 제목, 내용, 키워드를 한 문자열에 포함합니다.
- `id`는 원본 객체의 식별자입니다.
- `metadata`는 출처 추적, 필터링, 디버깅에 쓰는 구조화 정보입니다. metadata 자체가
  자동으로 임베딩되는 것은 아니므로, 검색에 반영할 제목이나 키워드는 이 예제처럼
  `page_content`에도 넣어야 합니다.
- `source_id`와 `title`을 끝까지 보존해야 답변의 근거를 사용자에게 보여 주고 검색 실패를
  원문까지 역추적할 수 있습니다.

### 2. `RecursiveCharacterTextSplitter`

`split_documents()`의 기본값은 `chunk_size=360`, `chunk_overlap=60`입니다.
splitter는 `"\n\n"`, `"\n"`, 문장 구분, 공백 순서로 경계를 시도하고, 그래도 큰
텍스트는 더 작은 단위로 나눕니다.

- 너무 큰 청크는 여러 주제가 한 벡터에 섞이고 프롬프트를 낭비합니다.
- 너무 작은 청크는 답에 필요한 앞뒤 맥락을 잃기 쉽습니다.
- overlap은 경계에 걸린 문장을 양쪽 청크가 공유하게 하지만, 너무 크면 중복 검색과
  인덱스 비용이 증가합니다.
- `add_start_index=True`는 각 청크가 원문 몇 번째 문자에서 시작했는지를
  `metadata["start_index"]`에 넣습니다.

`split_documents()`를 사용하면 원본 metadata가 청크에 복사됩니다. 그 뒤 코드는
`kb-010:chunk-000` 같은 고유한 `chunk_id`를 새 `Document.id`와 metadata에 함께
부여합니다. 같은 원문과 같은 분할 설정에서는 추적하기 쉬운 ID이지만, 분할 설정이나
원문이 바뀌면 청크 번호도 바뀔 수 있습니다.

현재 지식 문서는 모두 짧아서 기본 설정에서는 문서당 한 청크, 총 12청크입니다. 따라서
“분할했는데 개수가 늘지 않았다”는 것은 오류가 아닙니다. 작은 `chunk_size`로 바꾸면
분할 효과가 보입니다.

### 3. `LocalHashEmbeddings`

임베딩은 문자열을 고정 길이 숫자 벡터로 바꿉니다. 이 프로젝트의 기본 구현은
scikit-learn `HashingVectorizer`로 문자 2~5-gram을 1,024차원 벡터에 해싱하고 L2
정규화합니다.

- 같은 입력에는 항상 같은 벡터가 나오며 별도의 `fit()`이나 모델 다운로드가 없습니다.
- 한국어 형태소 분석기를 설치하지 않아도 부분 문자열의 겹침을 포착합니다.
- LangChain의 표준 `Embeddings` 인터페이스인 `embed_documents()`와 `embed_query()`를
  구현하므로 다른 임베딩 공급자로 교체하기 쉽습니다.
- 다만 이것은 **어휘적 기준선**입니다. “취소”와 “해지”처럼 글자가 다르지만 뜻이 비슷한
  표현을 사전학습 의미 임베딩만큼 잘 연결하지 못하며, 해시 충돌도 생길 수 있습니다.

검색 점수는 확률이나 정답 신뢰도가 아니라 이 벡터 공간에서의 유사도입니다. 따라서
`0.08` 같은 threshold를 다른 임베딩 모델에 그대로 복사하면 안 되고, 프로젝트 06과
같은 고정 평가셋으로 다시 조정해야 합니다.

### 4. `InMemoryVectorStore`

벡터 저장소는 청크 벡터와 원본 `Document`를 함께 저장하고, 질문 벡터와 가까운 항목을
찾습니다.

```python
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(documents=chunks)
```

`InMemoryVectorStore`는 설정이 간단하고 내부를 관찰하기 좋아 실습에 적합하지만, 프로세스가
끝나면 인덱스가 사라지고 여러 서버가 공유할 수도 없습니다. 대규모·운영 환경에서는 같은
vector store 인터페이스 뒤에 영속 저장소를 연결하고, 재시작·동시성·백업 전략을 별도로
설계해야 합니다.

### 5. Retriever와 점수 검색

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": top_k},
)
```

Retriever는 “질문 문자열을 받아 관련 `Document` 목록을 반환한다”는 표준 인터페이스입니다.
LCEL 안에서는 Runnable처럼 `retriever.invoke(question)`으로 사용할 수 있습니다. 덕분에
벡터 저장소를 바꾸더라도 뒤의 프롬프트 체인은 같은 형태를 유지할 수 있습니다.

한편 `LangChainRAG.search()`는 학습자가 점수를 직접 볼 수 있도록
`similarity_search_with_score()`를 사용합니다. `top_k`는 후보 개수이고,
`min_score`는 그 후보를 답변 근거로 채택할지 결정하는 별도 기준입니다. 둘은 같은 설정이
아닙니다.

### 6. context 포맷과 metadata 보존

`format_documents()`는 검색 결과를 다음 형태로 바꿉니다.

```text
[source_id=kb-010 | chunk_id=kb-010:chunk-000 | title=요청 한도 오류 429]
제목: 요청 한도 오류 429
...
```

청크 경계를 눈에 보이게 만들면 모델이 출처를 표시하기 쉽고, 사람이 잘못 검색된 청크를
찾기도 쉽습니다. 또한 기본 4,000자 예산을 넘으면 뒤 문서를 생략하거나 말줄임표로
자릅니다. `top_k`만 제한하고 전체 문자 수를 제한하지 않으면 긴 청크 몇 개가 모델의
컨텍스트 창과 비용을 과도하게 사용할 수 있습니다.

### 7. LCEL/Runnable 데이터 흐름

`build_lcel_rag_chain()`의 핵심은 다음 v1 패턴입니다.

```python
chain = (
    {
        "context": retriever | RunnableLambda(format_documents),
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | model
    | StrOutputParser()
)
answer = chain.invoke("429 오류를 어떻게 재시도하나요?")
```

입력 질문 하나가 두 갈래로 흐릅니다.

1. `context` 갈래는 질문을 Retriever에 보내 `list[Document]`를 얻고,
   `RunnableLambda(format_documents)`로 프롬프트용 문자열을 만듭니다.
2. `question` 갈래는 `RunnablePassthrough()`가 원래 질문을 그대로 전달합니다.
3. 두 값이 `{"context": ..., "question": ...}` 딕셔너리로 합쳐져
   `ChatPromptTemplate`의 자리표시자를 채웁니다.
4. 채팅 모델의 메시지 결과를 `StrOutputParser()`가 일반 문자열로 바꿉니다.

`|`는 앞 단계의 출력을 다음 단계의 입력으로 연결합니다. 이 조합 전체도 하나의
Runnable이므로 `invoke`, `batch`, `stream` 같은 공통 실행 방식을 사용할 수 있습니다.

주의할 점은 위의 간결한 `build_lcel_rag_chain()` 자체에는 점수 threshold가 없다는
것입니다. 실제 CLI가 호출하는 `LangChainRAG.ask()`는 먼저 점수를 확인하고, 채택된 근거가
있을 때만 `build_generation_chain()`으로 `prompt | model | parser`를 실행합니다. 즉,
**보류 게이트를 관찰 가능한 검색 단계와 생성 단계 사이에 둔 구현**입니다.

### 8. `ChatPromptTemplate`

`RAG_PROMPT`는 system 메시지와 human 메시지를 분리합니다.

- system 메시지: `<context>` 안의 정보만 사용하고, 근거가 없으면 추측하지 않으며,
  답변 끝에 `source_id`를 쓰도록 지시합니다.
- human 메시지: 원래 질문을 `{question}`에 넣습니다.
- `{context}`: 검색된 청크와 출처 정보를 넣습니다.

`prompt_preview()`는 실제 모델을 호출하지 않고 완성된 메시지를 보여 줍니다. 프롬프트
디버깅은 “좋은 문서가 검색되었는가?”, “필요한 metadata가 포함되었는가?”, “질문과
context가 뒤바뀌지 않았는가?”를 생성 결과와 분리해서 확인하는 데 유용합니다.

프롬프트 지시는 생성 모델의 행동을 유도하지만 정확성을 보장하지는 않습니다. 운영
환경에서는 반환된 인용이 실제 `sources` 안에 있는지 검증하고, 민감 정보 필터링과
프롬프트 주입 방어도 추가해야 합니다.

### 9. 보류(abstention)

`LangChainRAG.ask()`는 검색 결과 중 `score >= min_score`인 청크만 받습니다. 하나도
남지 않으면 다음 고정 문구를 반환하고 생성 모델을 호출하지 않습니다.

```text
관련 근거를 찾지 못했습니다. 질문을 더 구체적으로 작성해 주세요.
```

이때 `sources=()`, `context=""`, `abstained=True`가 됩니다. 근거가 있는데 모델이
“모르겠다”고 생성하는 것과, 시스템이 검색 단계에서 보류한 것은 서로 다른 사건이므로
`abstained`를 별도 필드로 남기는 것이 좋습니다.

threshold가 낮으면 답할 수 있는 질문을 놓치지 않지만 무관한 문서로 답하는 거짓 양성이
늘고, 높으면 잘못된 답은 줄지만 답할 수 있는 질문까지 보류할 수 있습니다. 이 균형은
프로젝트 06에서 정량적으로 평가합니다.

### 10. 오프라인 답변과 선택형 OpenAI

모델을 전달하지 않으면 `ask()`는 최상위 청크를 그대로 보여 주는 결정적 추출형 답변을
사용합니다. 이것은 LLM인 척하는 생성기가 아니라, 검색부터 최종 응답까지 API 없이
검증하기 위한 기준선입니다.

OpenAI 연동은 선택 사항입니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[openai]"
$env:OPENAI_API_KEY = "환경에 안전하게 설정한 키"
$env:OPENAI_CHAT_MODEL = "사용 가능한 채팅 모델 ID"
.\.venv\Scripts\python.exe projects\nlp\05_langchain_rag\solution.py --openai
```

또는 `--model "사용 가능한 채팅 모델 ID"`를 직접 전달할 수 있습니다. 키는 소스 코드,
노트북 출력, Git 저장소에 기록하지 마세요.

`--openai`가 바꾸는 것은 **생성 모델만**입니다. 임베딩은 계속
`LocalHashEmbeddings`이고 벡터 저장소도 메모리 방식입니다. OpenAI 임베딩까지 바꾸고
싶다면 `LangChainRAG(..., embeddings=새_Embeddings_구현)`처럼 표준 `Embeddings`
객체를 주입하고, 모델 변경 뒤에는 인덱스를 다시 만들어야 합니다.

## `starter.py` TODO 힌트

먼저 `starter.py`의 세 함수를 직접 완성한 뒤 `solution.py`와 비교해 보세요. 아래는
정답 전체가 아니라 구현 방향을 찾기 위한 힌트입니다.

### TODO 1~2: `build_documents(knowledge_path)`

1. 저장소 루트와 `data/raw/knowledge_base.jsonl` 경로를 구합니다.
2. 각 article을 `Document(page_content=..., metadata=..., id=...)`로 변환합니다.
3. `RecursiveCharacterTextSplitter(..., add_start_index=True)`를 만들고 반드시
   `split_documents()`를 호출합니다. `split_text()`만 쓰면 원본 metadata 연결을 직접
   다시 해야 합니다.
4. 원본 `source_id`는 유지하고, 소스별 순번으로 고유한 `chunk_id`를 만듭니다.
5. 다음 조건을 확인해 보세요.

```python
assert chunks
assert len({chunk.id for chunk in chunks}) == len(chunks)
assert all("source_id" in chunk.metadata for chunk in chunks)
assert all("chunk_id" in chunk.metadata for chunk in chunks)
assert all("start_index" in chunk.metadata for chunk in chunks)
```

### TODO 3: `build_retriever(documents, top_k)`

1. `LocalHashEmbeddings()` 또는 같은 `Embeddings` 인터페이스 구현을 만듭니다.
2. `InMemoryVectorStore(embedding=...)`에 청크를 `add_documents()`합니다.
3. `as_retriever(search_type="similarity", search_kwargs={"k": top_k})`로 변환합니다.
4. `retriever.invoke("429 오류")`의 반환값이 문자열이 아니라 `Document` 목록인지
   확인합니다.

### TODO 4: `build_chain(retriever, chat_model)`

1. Retriever 뒤에 문서 목록을 출처 포함 문자열로 바꾸는 formatter를 연결합니다.
2. 질문은 `RunnablePassthrough()`로 보존합니다.
3. `{"context": ..., "question": ...} | ChatPromptTemplate | chat_model |
   StrOutputParser()` 순서로 연결합니다.
4. 점수 기반 보류까지 넣으려면 Retriever만으로는 점수가 보이지 않으므로, 이 프로젝트의
   `LangChainRAG.search()`처럼 vector store의 점수 API를 먼저 호출해 생성 전 게이트를
   두는 방법을 생각해 보세요.

### TODO 5~7: 프로젝트 실행 경계

- `LangChainRagProject.retrieve()`에서 결과마다 `source_id`, `chunk_id`를 검증합니다.
- `build_optional_model()`은 config가 opt-in일 때만 `ChatOpenAI`를 만듭니다.
- `answer()`는 오프라인 fallback과 선택형 model 모두 같은 response 계약으로 반환합니다.
- 먼저 각 TODO를 작은 fixture로 확인한 뒤 `run()`에서 검색 → 증거 출력 → 답변 순으로
  연결하세요.

## 수식에서 코드로

`LocalHashEmbeddings`가 청크 \(c_i\)와 질문 \(q\)를 1,024차원 L2 정규화 벡터로 만들면
벡터 저장소의 기본 유사도는 다음 내적입니다.

$$
s_i=\frac{e(q)^\top e(c_i)}{\lVert e(q)\rVert_2
\lVert e(c_i)\rVert_2}=e(q)^\top e(c_i)
$$

`LangChainRAG.search()`는 \(s_i\ge\tau\)인 상위 `top_k` 청크만 근거로 채택합니다.
여기서 \(\tau\)가 `min_score`입니다. 이 값은 확률이 아니므로 임베딩 구현이 바뀌면 프로젝트
06의 고정 평가셋으로 다시 조정해야 합니다.

## starter와 `solution.py` 1:1 코드 지도

| 단계 | starter/solution 코드 | 데이터·shape | 왜 이 경계를 두었는가 |
|---|---|---|---|
| 설정 | `LangChainProjectConfig` | 경로·질문·k·threshold·model | 오프라인/온라인 실행 조건을 명시한다 |
| 파이프라인 | `build_pipeline()` | JSONL 12문서 → 기본 12청크 × 1024차원 | 로드·분할·인덱싱은 질문마다 반복하지 않는다 |
| 검색 관찰 | `retrieve()` | 질문 → `ScoredDocument[k]` | 생성 전에 source/chunk provenance를 검증한다 |
| 모델 선택 | `build_optional_model()` | opt-in flag → model 또는 `None` | 네트워크·비용이 드는 생성은 명시적으로 켠다 |
| 답변 | `answer()` | RAG + model → `LangChainRAGResponse` | 로컬 fallback과 외부 model이 같은 응답 계약을 쓴다 |
| 증거 | `print_evidence()` | 결과 → rank·source·chunk·score | 답변의 출처를 원문 청크까지 역추적한다 |
| 실행 | `run()` | config → response | 검색 증거를 답변보다 먼저 출력한다 |

starter의 `build_documents()`, `build_retriever()`, `build_chain()`은 프레임워크가 감추기 쉬운
세부 단계를 직접 조립하는 TODO입니다. 그 바깥에는 solution과 같은
`LangChainRagProject` 메서드가 있어 핵심 구현과 실행 orchestration을 함께 연습합니다.

solution의 `build_pipeline()`은 `LangChainRAG.from_jsonl()` 뒤 청크가 하나 이상인지
assert합니다. `retrieve()`는 모든 청크의 `source_id`와 `chunk_id` 존재를 확인합니다.
`build_optional_model()`은 `--openai`가 없으면 `None`을 반환하고, 켰을 때만
`create_openai_model()`을 호출합니다. `answer()`는 질문이 응답 계약에서 바뀌지 않았는지
검사합니다. 이 검증들은 LangChain 체인이 “실행됐다”는 사실보다 provenance와 입출력 계약이
유지됐음을 보여 주기 위한 선택입니다.

`--show-prompt`는 API 호출 없이 `prompt_preview()`를 출력합니다. `response.sources`는
threshold를 통과한 청크의 source ID를 순서를 유지하며 중복 제거합니다. 로컬 답변 본문은
최상위 청크만 인용할 수 있지만 `sources`에는 채택된 모든 근거가 남는 차이를 확인하세요.
starter는 TODO가 남으면 `연습 대기: ...`를 출력합니다.

## 예상 출력

기본 질문과 현재 데이터에서는 다음과 비슷하게 출력됩니다. 라이브러리 버전이나 임베딩
설정을 바꾸면 소수점 점수와 순위는 달라질 수 있습니다.

```text
documents=12, chunks=12, embedding=LocalHashEmbeddings
질문: 429 오류는 어떻게 안전하게 재시도하나요?

1. [kb-010] score=0.306 chunk=kb-010:chunk-000
2. [kb-011] score=0.255 chunk=kb-011:chunk-000
3. [kb-009] score=0.141 chunk=kb-009:chunk-000

--- prompt preview (API 호출 없음) ---
System: ...
<context>
[source_id=kb-010 | chunk_id=kb-010:chunk-000 | title=요청 한도 오류 429]
...
</context>
Human: 429 오류는 어떻게 안전하게 재시도하나요?

--- answer ---
검색된 최상위 근거:
제목: 요청 한도 오류 429
429 응답은 ... Retry-After 헤더를 존중하고 지수 백오프와 무작위 지연을 적용하세요.
...
출처: [kb-010]
sources=['kb-010', 'kb-011', 'kb-009'], abstained=False
```

보류 동작은 threshold를 높여 확인할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe projects\nlp\05_langchain_rag\solution.py `
  --query "회사 주소가 어디인가요?" --min-score 0.20
```

```text
--- answer ---
관련 근거를 찾지 못했습니다. 질문을 더 구체적으로 작성해 주세요.
sources=[], abstained=True
```

기본 `min_score=0.08`에서는 같은 질문에 무관한 배송 문서가 선택될 수 있습니다. 이 실패는
프롬프트만 잘 써서는 해결되지 않으며 threshold 조정, 더 나은 임베딩, no-answer 평가가
필요하다는 중요한 관찰입니다.

## 직접 해볼 실험

### 실험 1: `top_k`와 context 크기

같은 질문을 `--top-k 1`, `3`, `5`로 실행하고 순위, `sources`, prompt 길이를 기록하세요.
`top_k`를 늘리면 필요한 근거를 찾을 가능성은 커지지만 무관한 문서와 토큰 비용도 늘어납니다.

### 실험 2: threshold와 보류

“회사 주소가 어디인가요?”를 `--min-score 0.05`, `0.08`, `0.12`, `0.20`으로 실행하세요.
어느 값부터 `abstained=True`가 되는지, 답할 수 있는 “429 오류” 질문은 여전히 통과하는지
함께 확인하세요. 한 사례만 보고 최적 threshold라고 결론 내리지 마세요.

### 실험 3: 청크 크기와 overlap

`LangChainRAG`를 `chunk_size=120, chunk_overlap=0`과
`chunk_size=120, chunk_overlap=30`으로 각각 만들고 다음을 비교하세요.

- 총 청크 수
- 최상위 `chunk_id`와 `start_index`
- 경계에 걸린 문장이 두 청크에 보존되는지
- 중복 청크가 top-k를 차지하는지

### 실험 4: 어휘적 표현 차이

“구독을 해지하고 싶어요”와 “멤버십을 그만 쓰고 싶어요”를 비교하세요. 로컬 해시 임베딩이
동의어와 바꿔 말하기에 약한 사례를 모은 뒤, 의미 임베딩으로 교체했을 때 순위가 개선되는지
같은 질문 목록으로 재평가하세요.

### 실험 5: 생성기 교체

같은 검색 설정으로 오프라인 추출형 답변과 `--openai` 답변을 비교하세요. 검색 결과가 같아도
답변의 요약, 인용, 보류 표현이 달라질 수 있습니다. 검색 품질과 생성 품질을 한 문제로
섞지 말고 따로 기록하세요.

## 자주 발생하는 오류와 디버깅

### `ModuleNotFoundError: No module named 'langchain...'`

저장소 루트에서 같은 가상 환경의 Python으로 의존성을 설치했는지 확인합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[rag,dev]"
```

### `--openai 사용 시 --model 또는 OPENAI_CHAT_MODEL을 설정하세요.`

`--model`을 전달하거나 현재 PowerShell 세션에 `OPENAI_CHAT_MODEL`을 설정합니다. 모델 ID는
계정에서 실제 사용할 수 있는 값을 써야 합니다.

### `No module named 'langchain_openai'`

선택 의존성을 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[openai]"
```

### 인증·권한·요금 오류

`OPENAI_API_KEY`가 현재 프로세스 환경에 있는지 확인하되 값을 출력하거나 로그에 남기지
마세요. API 키, 모델 접근 권한, 사용 한도는 서로 별개의 원인일 수 있습니다. 오프라인
경로로 먼저 검색과 프롬프트가 정상인지 확인하면 공급자 오류와 RAG 오류를 분리할 수 있습니다.

### `start_index` 또는 출처 metadata가 사라짐

- splitter에 `add_start_index=True`가 있는지 확인합니다.
- 문자열만 자르는 `split_text()` 대신 `Document`를 받는 `split_documents()`를 썼는지
  확인합니다.
- 새 `Document`를 만들 때 `metadata=dict(raw_chunk.metadata)`처럼 복사했는지 확인합니다.
- `chunk_id`를 만들면서 원본 `source_id`를 덮어쓰지 않았는지 확인합니다.

### 관련 없는 문서가 높은 순위로 나옴

`--show-prompt` 전에 출력되는 검색 점수부터 봅니다. 원인은 낮은 threshold, 문자 해시
임베딩의 한계, 과도한 `top_k`, 부적절한 청크 경계일 수 있습니다. 생성 모델이나 프롬프트를
바꾸기 전에 검색 실패를 먼저 고치세요.

### 모든 문서가 한 청크인 것처럼 보임

현재 12개 원문이 기본 `chunk_size=360`보다 짧기 때문입니다. 작은 `chunk_size`로 실험하되,
운영값은 실제 문서 길이 분포와 평가 결과로 결정해야 합니다.

## 완료 체크리스트

- [ ] 12개 원문을 `Document`로 만들고 `source_id`, `title`, `category`를 보존했다.
- [ ] 분할 뒤 모든 청크에 고유한 `chunk_id`와 `start_index`가 있다.
- [ ] `LocalHashEmbeddings`의 장점과 의미 임베딩이 아니라는 한계를 설명할 수 있다.
- [ ] `InMemoryVectorStore`에 청크를 넣고 Retriever로 질문을 조회했다.
- [ ] `top_k`와 `min_score`의 역할 차이를 설명할 수 있다.
- [ ] `RunnablePassthrough`와 LCEL `|`를 사용해 질문과 context의 흐름을 추적했다.
- [ ] `--show-prompt`로 모델 호출 전 prompt와 metadata를 확인했다.
- [ ] 근거가 threshold를 통과하지 못할 때 모델 호출 없이 보류되는 것을 확인했다.
- [ ] 도메인 안 질문과 no-answer 질문을 모두 시험했다.
- [ ] 선택형 모델을 쓸 경우 API 키를 코드나 Git에 남기지 않았다.

## 공식 문서

- [LangChain retrieval 개요](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangChain semantic search와 `Document`/splitter/vector store](https://docs.langchain.com/oss/python/langchain/knowledge-base)
- [텍스트 splitter 통합](https://docs.langchain.com/oss/python/integrations/splitters/index)
- [LangChain v1 migration](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [OpenAI embedding 통합](https://docs.langchain.com/oss/python/integrations/embeddings/openai)
