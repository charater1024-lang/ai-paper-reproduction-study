# 분야별 AI 논문 재현 실습 가이드

## 구성

- 7개 분야, 분야별 10편, 총 70편
- 논문마다 `exercises/`와 `solutions/` 한 쌍: 총 140개 노트북
- 모든 pair는 동일한 cell ID와 순서를 사용
- 원 논문의 절·수식·그림 ↔ 코드 ↔ 검증 증거를 첫 표에서 연결
- `data/field_curriculum/`의 로컬 합성 데이터 7종만 사용
- `docs/paper_reading_notes/`에 논문별 한국어 배경·선수지식·핵심 요약 제공

## 한 논문을 공부하는 순서

1. 헤더에서 primary paper와 재현 범위·생략 범위를 읽습니다.
2. mapping 표의 원문 위치를 찾아 입력·출력 shape를 메모합니다.
3. 실습본의 데이터 준비 셀을 실행하고 배열 key, dtype, split을 확인합니다.
4. TODO 하나를 구현한 뒤 바로 아래 assertion만 실행합니다.
5. 막히면 정답본의 동일 위치에서 최소한의 코드만 확인합니다.
6. 작은 학습이나 ablation의 metric을 원 논문의 주장과 연결하되, 원 수치와 직접 비교하지 않습니다.
7. 한 변수만 변경한 추가 실험을 기록합니다.

## 데이터

데이터는 모두 고정 seed로 로컬 생성한 CC0 합성 자료입니다. 개인정보·외부 API·런타임 다운로드가
없습니다. 영상 분류뿐 아니라 mask와 box, 문장 분류·번역·검색, 생성 분포, 강화학습 transition,
graph와 user-item interaction, image-text pair, teacher logits·attention·양자화 calibration까지
분야별 입력 구조를 담았습니다.

```powershell
# 데이터 재생성 + SHA-256 manifest 갱신
.\.venv\Scripts\python.exe -X utf8 tools\generate_field_datasets.py
```

## 검증 명령

```powershell
# 데이터 checksum과 70쌍의 구조·문법·TODO·metadata 검사
.\.venv\Scripts\python.exe -X utf8 tools\validate_field_reproductions.py

# class·함수 깊이, 수식 설명, 구현 이유와 코드 가독성 검사
.\.venv\Scripts\python.exe -X utf8 tools\validate_portfolio_quality.py --track field

# 한 분야만 검사
.\.venv\Scripts\python.exe -X utf8 tools\validate_field_reproductions.py --field vision

# 한 편의 정답을 새 커널에서 실행
.\.venv\Scripts\python.exe -X utf8 tools\validate_field_reproductions.py `
  --field nlp_llm --only 03 --execute-solutions

# 70개 정답 전체 실행 (동시 커널 수는 메모리에 맞게 조절)
.\.venv\Scripts\python.exe -X utf8 tools\validate_field_reproductions.py `
  --execute-solutions --jobs 3
```

명세 파일을 수정했을 때만 생성기를 실행하세요. 생성기는 학습자가 작성한 exercise 파일을
덮어쓰므로 답은 먼저 복사하거나 Git diff로 보존합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\build_field_reproductions.py
.\.venv\Scripts\python.exe -X utf8 tools\build_field_catalog.py
```

## 실험 기록 템플릿

```markdown
### 논문 / 원문 위치
- 논문:
- section·equation·figure:
- 핵심 주장:

### 통제
- seed:
- 고정한 데이터·모델·step:
- 바꾼 변수 하나:

| 관찰 | 기준 | 변경 | 차이 |
|---|---:|---:|---:|
| loss/accuracy/invariant | | | |

### 해석
- 원문의 주장과 연결되는 점:
- 이 미니 재현으로 판단할 수 없는 점:
- 다음 반증 실험:
```

## 해석의 한계

이 노트북이 통과했다는 것은 작은 합성 조건에서 핵심 연산과 불변식이 동작한다는 의미입니다.
원 논문의 benchmark 성능, 대규모 일반화, 공정성, robustness, 분산 처리량을 재현한 것이 아닙니다.
각 결과를 인용할 때는 원 논문의 결과와 자신의 미니 실험 관찰을 문장에서 분리하세요.
