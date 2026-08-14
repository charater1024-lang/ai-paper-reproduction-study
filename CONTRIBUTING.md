# 기여 가이드

이 저장소는 Windows에서 AI 기초와 주요 논문을 직접 구현하며 공부하기 위한 한국어 학습 자료입니다. 작은 수정도 환영하지만, 실습 노트북은 여러 빌더가 생성하므로 먼저 아래의 원본과 생성물 관계를 확인해 주세요. 처음 설치한다면 [프로젝트 안내](README.md)와 [Windows 설치 안내](docs/WINDOWS_SETUP.md)를 먼저 읽어 보세요.

## 시작하기

권장 환경은 **Windows 10/11, Python 3.12, PowerShell 7 또는 Windows PowerShell**입니다.

```powershell
git clone <저장소-URL>
cd <저장소-폴더>
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --editable . --no-deps
```

가상환경을 활성화하지 않고 그 안의 Python을 직접 호출하므로 PowerShell 실행 정책을 바꿀 필요가 없습니다. 이후 명령의 `python`은 이 `.venv` Python을 활성화한 상태를 뜻합니다.

작업은 `main`에 직접 쌓지 말고 목적이 드러나는 브랜치에서 시작해 주세요.

```powershell
git switch -c docs/improve-paper-summary
```

## 무엇을 수정해야 하나요?

| 학습 영역 | 원본(source of truth) | 생성 명령 |
|---|---|---|
| 기본 00~22 페어 | `notebooks/00_*.ipynb`~`12_*.ipynb`, `tools/build_legacy_pairs.py`, `tools/build_paired_*.py` | `python tools/rebuild_paired_curriculum.py` |
| 유명 논문 20편 | `tools/paper_curriculum/*_specs.py` | `python tools/build_paper_reproductions.py` |
| 분야별 논문 70편 | `tools/field_curriculum/*_specs.py` | `python tools/build_field_reproductions.py` |
| 분야별 합성 데이터 | `tools/generate_field_datasets.py` | `python tools/generate_field_datasets.py` |
| 한국어 논문 읽기 자료 | `docs/paper_reading_notes/*.md` | 직접 수정 후 테스트 |

`notebooks/exercises`, `notebooks/solutions`, `notebooks/paper_reproductions/*`, `notebooks/field_reproductions/*`의 노트북은 생성물입니다. 생성물을 직접 고치면 다음 빌드에서 덮어써집니다. 대응하는 빌더 또는 `*_specs.py`를 수정한 뒤 생성 명령을 실행하고, 소스와 생성물을 함께 커밋해 주세요.

개인 학습 기록이나 실행 출력은 생성 디렉터리 밖에 복사해 보관하세요. 저장소에 커밋하는 정규 노트북은 재현성과 리뷰 가독성을 위해 출력과 실행 번호를 비워 둡니다. GitHub가 생성 노트북 diff를 접어 표시하면 **Load diff**를 눌러 볼 수 있지만, 리뷰에서는 먼저 빌더와 spec 변경을 확인합니다.

## 포트폴리오용 구현 기준

논문 실습은 단순히 metric이 출력되는 짧은 예제가 아니라, 코드 리뷰에서 설계 결정을
설명할 수 있는 구현이어야 합니다. 자세한 계약은
[포트폴리오용 논문 재현 표준](docs/PORTFOLIO_NOTEBOOK_STANDARD.md)을 따릅니다.

- 논문의 핵심 연산은 이름이 드러나는 class와 함수로 직접 구현합니다.
- 원문 절·식·그림을 노트북의 class·method·assertion과 연결합니다.
- 핵심 수식은 block LaTeX로 쓰고 기호, tensor shape, 코드 실행 순서를 풀이합니다.
- data preparation, objective, update, evaluate를 한 셀이나 한 줄 helper에 숨기지 않습니다.
- 각 단계 앞에서 왜 이 구현을 택했는지와 원 논문 대비 생략 범위를 밝힙니다.
- setup을 제외한 핵심 code cell 바로 앞에는 원문 위치, shape, 구현 이유, 완료 증거를
  설명하는 Markdown을 둡니다.
- exercise는 solution과 같은 public class·함수·메서드 signature를 보여 주고 body만
  TODO로 남깁니다.
- 새 `Lab`·`Agent` class는 설정값만 검사하지 않고 실제 학습·평가 경로에서 호출하며,
  loss·gradient·parameter·metric 변화를 assertion으로 검증합니다.
- 학습자용 notebook Python은 88자를 넘지 않으며, semicolon이나 한 줄짜리
  `class`·`def`·`if`·`for`로 압축하지 않습니다.

노트북은 빌드 시 Ruff formatter를 통과합니다. formatter를 피하려고 긴 표현식을 문자열로
조립하지 말고, 사람이 읽을 수 있는 중간 변수와 명시적 method로 분리하세요.

## 한국어 논문 요약 기여 기준

AI의 도움을 받아 초안을 작성할 수 있지만, 다음 항목은 기여자가 원 논문을 기준으로 직접 확인해야 합니다.

- 논문 제목, 저자, 발표 연도, 학회·저널과 링크가 정확한가
- 핵심 주장이 초록뿐 아니라 본문의 방법·실험·한계와 일치하는가
- 논문에 없는 해석이나 추론은 사실처럼 쓰지 않고 명확히 구분했는가
- 원문의 문장이나 초록을 길게 복사하지 않고 한국어로 새로 설명했는가
- 실습 코드가 논문의 어느 절·수식·알고리즘을 축소 재현하는지 연결했는가
- 재현하지 못한 규모, 데이터, 평가 항목과 알려진 한계를 적었는가

새 요약은 가능하면 다음 순서를 따릅니다.

1. 해결하려는 문제와 등장 배경
2. 읽기 전 필요한 기초 지식
3. 핵심 아이디어와 단계별 동작
4. 중요한 수식 또는 알고리즘의 직관
5. 실험 결과가 의미하는 것
6. 한계와 후속 연구
7. 이 저장소의 실습 코드와 논문 위치 대응표
8. 원 논문과 공식 구현 링크

PR 설명에는 AI를 초안·번역·교정에 사용했는지와 사람이 어떤 항목을 검증했는지 간단히 밝혀 주세요.

## 데이터 기여 기준

- 개인정보, 비공개 자료, 사용 조건이 불명확한 데이터를 추가하지 않습니다.
- 교육용 데이터는 작고 오프라인에서 실행 가능해야 하며, 가능하면 고정 seed로 재생성할 수 있어야 합니다.
- 새 데이터에는 출처 또는 생성 방법, 라이선스, 스키마, 크기, 연결된 실습을 문서화합니다.
- `data/field_curriculum`을 바꾸면 `manifest.json`의 크기와 SHA-256도 생성 스크립트로 갱신합니다.
- 대형 모델 체크포인트와 실행 중 생성된 `artifacts/` 파일은 커밋하지 않습니다.

## 제출 전 빠른 검증

아래 명령은 GPU 없이 Windows CPU 환경에서도 실행되어야 합니다.

```powershell
$env:AI_LAB_DEVICE = "cpu"
python -m ruff check app.py scripts src tests tools projects
python -m pytest
python tools/validate_all_notebooks.py
python tools/validate_paired_notebooks.py
python tools/validate_paper_reproductions.py
python tools/validate_field_reproductions.py
python tools/validate_portfolio_quality.py
```

논문 또는 분야별 spec을 수정했다면 먼저 생성물을 갱신합니다.

```powershell
python tools/build_paper_reproductions.py
python tools/build_field_reproductions.py
git status --short
```

학습 코드의 실행 동작을 바꿨다면 관련 정답 노트북도 실행해 보세요. 일반 문서 PR에서 전체 113개 정답 노트북 실행은 요구하지 않습니다.

```powershell
python tools/validate_paper_reproductions.py --execute-solutions --jobs 2
python tools/validate_field_reproductions.py --field vision --execute-solutions --jobs 2
```

## Pull Request 체크리스트

- 한 PR에는 하나의 명확한 목적만 담습니다.
- 생성 파일만 바뀐 PR이라면 반드시 대응하는 원본 변경도 포함합니다.
- 새 의존성이 필요하면 `requirements.txt`와 `pyproject.toml`의 역할을 함께 검토합니다.
- Windows에서 실행한 명령과 결과를 PR에 적습니다.
- 비밀키, 로컬 절대 경로, 개인정보, 모델 체크포인트가 없는지 확인합니다.
- UI 또는 노트북 표시가 바뀌면 필요한 경우 스크린샷을 첨부합니다.

버그를 발견했다면 재현 가능한 최소 실습 번호, 실행 명령, Windows/Python 버전, 전체 오류 메시지를 Issue에 남겨 주세요.
