# NLP 실습 프로젝트 트랙

한국어 고객지원 데이터를 이용해 전처리부터 RAG 평가까지 한 단계씩 직접 구현하는
미니 프로젝트 모음입니다. 각 폴더에는 두 종류의 파일이 있습니다.

- `starter.py`: Config·Project class와 핵심 함수 TODO를 직접 채우는 연습용
- `solution.py`: 같은 클래스·메서드 흐름으로 끝까지 실행되는 참고 구현

프로젝트는 서로 이어지지만 각각 독립적으로 실행할 수 있습니다. 01~04와 06은 외부 API가
필요 없고, 05 LangChain도 기본 모드는 API 키나 모델 다운로드 없이 실행됩니다.

처음 공부한다면 실행 명령만 따라가기 전에 [NLP·검색·RAG 상세 학습 가이드](DETAILED_GUIDE.md)를
읽어 보세요. 데이터 구조, 핵심 용어, 각 알고리즘이 필요한 이유, 출력 해석, 실제 threshold
trade-off, 7일 학습 일정까지 한 흐름으로 설명합니다.

## 준비

프로젝트 루트에서 한 번만 실행합니다.

```powershell
.\Install_or_Repair.cmd

# 또는 이미 .venv가 있다면
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
```

## 권장 학습 순서

| 번호 | 프로젝트 | 핵심 결과물 | 예상 시간 | 외부 API |
|---:|---|---|---:|---|
| 01 | [텍스트 전처리](01_text_preprocessing/README.md) | 정규화·PII 마스킹·토큰 빈도 | 45~60분 | 불필요 |
| 02 | [의도 분류](02_intent_classification/README.md) | TF-IDF 분류기·오류 분석 | 60~90분 | 불필요 |
| 03 | [의미 검색](03_semantic_search/README.md) | SVD dense index·cosine 검색 | 60~90분 | 불필요 |
| 04 | [프레임워크 없는 RAG](04_rag_from_scratch/README.md) | 검색·context·grounded prompt | 60~90분 | 불필요 |
| 05 | [LangChain RAG](05_langchain_rag/README.md) | Document·splitter·vector store·LCEL | 90~120분 | 기본 불필요 |
| 06 | [RAG 평가](06_rag_evaluation/README.md) | Recall@k·MRR·no-answer 평가 | 60~90분 | 불필요 |

완성 예제를 빠르게 확인하려면 다음 명령을 순서대로 실행하세요.

```powershell
.\.venv\Scripts\python.exe projects\nlp\01_text_preprocessing\solution.py
.\.venv\Scripts\python.exe projects\nlp\02_intent_classification\solution.py
.\.venv\Scripts\python.exe projects\nlp\03_semantic_search\solution.py
.\.venv\Scripts\python.exe projects\nlp\04_rag_from_scratch\solution.py --show-prompt
.\.venv\Scripts\python.exe projects\nlp\05_langchain_rag\solution.py --show-prompt
.\.venv\Scripts\python.exe projects\nlp\06_rag_evaluation\solution.py
```

각 starter도 그대로 실행할 수 있습니다. TODO가 남아 있으면 `연습 대기: TODO ...`를
출력하고 정상 종료하며, 구현을 채우면 대응 solution과 같은 project flow를 수행합니다.
비교할 때는 `Config → load/build → validate/evaluate → print → run`의 메서드 경계와
README의 1:1 코드 지도를 함께 확인하세요.

## 이미 있는 심화 노트북과 연결하기

코드 프로젝트를 마친 뒤에는 아래 짝지어진 Jupyter 노트북으로 실험 범위를 넓힐 수 있습니다.

| 프로젝트 | 이어서 볼 노트북 |
|---|---|
| 전처리·분류 | 03 pandas EDA, 04 scikit-learn, 06 PyTorch 분류 |
| 의미 검색 | 17 Embedding과 Vector Index |
| 직접 구현 RAG | 11 검색과 근거화, 18 End-to-End RAG |
| RAG 평가 | 22 Hybrid RAG와 회귀 평가 |

## 공통 실험 기록

각 프로젝트에서 적어도 한 가지 설정을 바꾼 뒤 아래 다섯 줄을 남겨 보세요.

```text
가설:
변경한 한 가지:
고정한 조건/seed:
결과(metric + 실패 예시):
다음 결정:
```

전체 회귀 테스트는 다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests projects\nlp
```
