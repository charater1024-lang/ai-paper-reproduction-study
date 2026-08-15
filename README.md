# AI 논문·모델 구현 독학 랩

![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D4?logo=windows11&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/JupyterLab-Exercise%20%2B%20Solution-F37626?logo=jupyter&logoColor=white)
![Korean](https://img.shields.io/badge/Docs-한국어-2E8B57)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Python·데이터 분석·머신러닝부터 PyTorch, Transformer, RAG, 컴퓨터 비전, 생성 모델,
강화학습, 그래프, 멀티모달, 지식 증류와 모델 경량화까지 **직접 구현하고 검증하며 공부하는
로컬 학습 저장소**입니다.

> **이 저장소를 만든 이유**  
> AI에게 지시하고 결과를 받는 바이브 코딩에만 익숙해지다 보니, 직접 코드를 읽고 짜며
> 오류를 추적하는 감각과 긴 글·논문을 천천히 읽는 감각이 떨어지는 것 같았습니다. 그래서
> 기초부터 다시 복습하고, 손으로 구현·실행·실패·수정하는 과정을 반복하며 **코딩 감각과
> 글 읽는 감각을 다시 기르기 위해** 이 학습 랩을 만들었습니다. AI는 정답 복사기가 아니라,
> 설명을 비교하고 질문을 더 좋게 만드는 학습 동료로 사용합니다.

이 저장소의 중심은 강의를 보기만 하는 것이 아니라 다음 사이클을 반복하는 데 있습니다.

> 한국어 요약 읽기 → 실행 결과 예측 → exercise의 TODO 구현 → assert로 검증 → solution을
> 최소한만 확인 → 변수 하나 변경 → 결과와 실패 이유 기록

모든 기본 데이터는 로컬 합성 예제라서 다운로드나 API 키 없이 실행됩니다. 신경망 실습은
CUDA → Apple MPS → CPU 순서로 장치를 자동 선택합니다. 논문 노트북은 원 논문의 대규모
benchmark를 그대로 재현한 것이 아니라, 핵심 메커니즘과 주장 하나를 빠르게 확인하는
**교육용 축소 재현**입니다.

**바로가기:** [30초 시작](#windows-30초-시작) · [학습 트랙](#나에게-맞는-트랙-고르기) ·
[AI 코딩 실습 센터](AI_Coding_Practice_Center.ps1) ·
[브라우저 학습 가이드](AI_Coding_Practice_Guide.html) ·
[상세 한국어 논문 해설](AI_Korean_Paper_Notes.html) ·
[함수·API 사전](AI_Function_Glossary.html) ·
[한국어 논문 원본 노트](docs/paper_reading_notes/README.md) ·
[함수·API 해설](docs/FUNCTION_API_GUIDE.md) · [데이터](data/README.md) ·
[Windows 설치](docs/WINDOWS_SETUP.md) · [휴대용 ZIP](docs/PORTABLE_ZIP.md) ·
[GitHub 게시](docs/GITHUB_PUBLISHING_WINDOWS.md) ·
[기여 가이드](CONTRIBUTING.md)

## Windows 30초 시작

GitHub에서 Clone하거나 ZIP을 풀어 저장소 루트로 이동한 뒤 다음 세 파일을
순서대로 실행하세요.

```powershell
.\Install_or_Repair.cmd
.\Verify_Setup.cmd
.\Start_Paired_Learning.cmd 00
```

왼쪽 exercise에서 TODO를 풀고, 막힐 때만 오른쪽 solution의 같은 번호 셀을 확인합니다.
NVIDIA GPU를 사용하려면 기본 설치 후 `Install_GPU_PyTorch.cmd`를 한 번 실행하세요.
Windows 환경 재현은 [`requirements-windows.txt`](requirements-windows.txt), 공통 패키지는
[`requirements.txt`](requirements.txt)에 정리되어 있습니다.
다른 PC에 최신 작업본 전체를 한 파일로 옮기는 방법은
[Windows 휴대용 ZIP 안내](docs/PORTABLE_ZIP.md)를 참고하세요.

## 현재 포함된 학습 자료

| 구분 | 학습 단위 | 파일 구성 | 권장 목적 |
|---|---:|---:|---|
| 기본 AI 엔지니어링 | 23개 | exercise 23 + solution 23 | Python부터 GPU·Hybrid RAG까지 순서대로 학습 |
| 유명 AI 논문 미니 재현 | 20개 | exercise 20 + solution 20 | 분야를 가로지르며 AI 발전 흐름 파악 |
| 7개 분야별 논문 재현 | 70개 | exercise 70 + solution 70 | 관심 분야를 10편 단위로 전문화 |
| NLP 미니 프로젝트 | 6개 | starter 6 + solution 6 | 노트북 밖의 작은 `.py` 프로젝트 연습 |
| 순차 실행 스크립트 | 10개 | `scripts/*.py` | 데이터 → 학습 → 추론 → RAG 파이프라인 실행 |
| 로컬 학습 데이터 | 14종 | 기본 3 + 추가 4 + 분야별 7 | 외부 다운로드 없는 반복 실험 |

기본·논문 트랙을 합치면 **113개 페어 학습 단위와 226개 exercise/solution 노트북**입니다.
00~12 보존 원본 13개까지 포함한 정규 학습 노트북은 239개입니다. 논문 20편 트랙과 분야별
70편 트랙은 제목 기준 17편이 겹치므로, 논문 실습 항목은 90개지만 고유 논문은 73편입니다.
중복 논문은 다른 데이터·과제·설명으로 복습하거나 이미 익숙하면 건너뛰어도 됩니다.

```mermaid
flowchart LR
    A["한국어 배경·선수지식"] --> B["Exercise TODO"]
    B --> C["Assert·Metric 검증"]
    C --> D["Solution 최소 확인"]
    D --> E["변수 하나 변경"]
    E --> F["실험 노트 기록"]
    F --> B
```

## 첫날 5분 시작

Windows에서는 저장소 루트에서 다음 순서로 실행합니다.

```powershell
# 1. 가상환경·의존성·Jupyter 커널 설치 또는 복구
.\Install_or_Repair.cmd

# 2. 환경, 데이터, 노트북 구조, 실행기 점검
.\Verify_Setup.cmd

# 3. 초보 시작점인 기본 00번을 실습/정답 두 창으로 열기
.\Start_Paired_Learning.cmd 00
```

### 하나의 바탕화면 아이콘으로 시작하기

설치가 끝난 뒤 아래 명령을 한 번 실행하면 Windows Desktop에 **`AI 코딩 실습 센터`**
아이콘 하나가 만들어집니다. 이 허브에서 설치·환경 점검·JupyterLab·기본/논문/분야별
실습·NLP 타이핑 랩·읽기 자료·학습 폴더를 모두 열 수 있습니다.

```powershell
# Windows가 등록한 기본 Desktop에 단일 허브 바로가기 생성·업데이트
powershell -ExecutionPolicy Bypass -File .\Create_AI_Coding_Lab_Shortcut.ps1

# 바탕화면 바로가기 없이 허브를 바로 열 때
.\Start_AI_Coding_Lab.cmd
```

허브는 Python이 아직 설치되지 않은 상태에서도 열립니다. 이 경우 먼저 `설치 또는 복구`를
선택하고, 설치가 끝난 뒤 실습 버튼을 사용하세요.

### 예전 개별 바로가기 (선택)

기존의 여러 개 바로가기를 계속 쓰고 싶을 때만 아래 스크립트를 사용하세요.

```powershell
# Windows가 등록한 기본 Desktop에 11개 바로가기 생성·업데이트
powershell -ExecutionPolicy Bypass -File .\tools\create_learning_shortcuts.ps1

# OneDrive Desktop을 쓰는 경우
powershell -ExecutionPolicy Bypass -File .\tools\create_learning_shortcuts.ps1 `
  -OutputDirectory "$env:OneDrive\Desktop"
```

설치된 주요 바로가기는 다음과 같습니다.

- `AI 코딩연습 - 기본 실습과 정답`
- `AI 논문 20편`
- `AI 논문 실습 - 전체 분야`
- 7개 분야별 바로가기
- `AI 논문 한국어 요약`

## 눈으로 읽는 오프라인 학습 자료

Markdown 파일을 편집기로 열지 않아도 되는 읽기 전용 자료를 저장소 루트에 함께 둡니다.
모두 인터넷이나 별도 서버 없이 브라우저에서 열립니다.

| 자료 | 열기 | 용도 |
|---|---|---|
| 브라우저 학습 가이드 | [AI_Coding_Practice_Guide.html](AI_Coding_Practice_Guide.html) | 과정 전체 흐름, 오늘 할 일, 주요 문서로 이동 |
| 인쇄용 가이드 | [AI_Coding_Practice_Guide.pdf](AI_Coding_Practice_Guide.pdf) | 화면 밖에서 읽거나 인쇄할 때 |
| 상세 한국어 논문 해설 | [AI_Korean_Paper_Notes.html](AI_Korean_Paper_Notes.html) | 핵심 70편을 분야·논문별로 읽고, 상세 배경지식 카드를 펼칠 때 |
| 함수·API 한국어 사전 | [AI_Function_Glossary.html](AI_Function_Glossary.html) | 함수의 역할·입력·반환·shape·주의점을 인터넷 검색 없이 확인할 때 |
| NLP 타이핑 워크북 | [projects/nlp/Typing_Practice_Workbook.html](projects/nlp/Typing_Practice_Workbook.html) | starter 코드를 직접 따라 치며 6개 프로젝트를 진행할 때 |

논문 리더의 **`상세 배경지식`**을 열면, 원문에 있던 짧은 선수지식 목록 대신 논문에 필요한
개념을 먼저 읽을 순서와 함께 보여 줍니다. 각 카드는 `무엇인가 → 직관 → 왜 이 논문에
필요한가 → 코드에서 만나는 곳 → 자주 헷갈리는 점`으로 구성되어 있습니다.

## 나에게 맞는 트랙 고르기

| 현재 목표 | 시작 위치 | 권장 경로 | 전체 분량 참고 |
|---|---|---|---:|
| AI 코드를 처음 체계화 | 기본 00 | 기본 00~12 → 13~22 | 00~12 약 38.5~47.5시간 |
| Transformer를 빨리 이해 | 기본 02 | 02 → 05 → 07 → 08 → 09 → 논문 09 | 최소 선수 경로 |
| 논문 흐름을 넓게 훑기 | 논문 20편 00 | 00~19 후 관심 분야 선택 | 약 25.3시간 |
| 한 분야를 깊게 공부 | 한국어 독해 노트 | 해당 분야 00~09 | 분야당 약 10~14시간 |
| NLP/RAG 코드를 프로젝트로 연습 | NLP 프로젝트 01 | 01~06 순서 | 프로젝트당 약 1~2시간 |
| 최신 연구 방향 찾기 | 최신 top-tier 지도 | 기존 핵심 논문 → 2023~2025 후속 논문 | 선택 학습 |

70편 전체를 반드시 선형으로 완주할 필요는 없습니다. 권장 전략은 기본 과정으로 코드 기초를
만든 뒤, 논문 20편으로 전체 지도를 보고, 관심 분야 10편만 먼저 깊게 실습하는 것입니다.

### 최소 경로: 4~6주

`기본 00 → 02 → 03 → 04 → 05 → 07 → 08 → 09 → 10 → 11 → 12`

환경, 수치 계산, 데이터 검증, 기준선, autograd, 신경망, attention, Transformer, tiny LM,
RAG와 통합 프로젝트를 한 번씩 경험합니다.

### 표준 경로: 10~14주

기본 00~22를 순서대로 진행합니다. 이후 논문 20편을 survey로 보고, 관심 분야 10편을
선택합니다. 13~22는 앞 노트북의 개념을 실제 모달리티·학습 루프·GPU·평가로 확장합니다.

### NLP·LLM 집중 경로

`기본 00~06 → 07~10 → 11~12 → 논문 06~11 → 분야 NLP·LLM 00~09 →
증류·경량화 00~09`

## 한 회차를 공부하는 방법

하루 60~90분 기준으로 한 논문이나 한 노트북의 절반 정도가 적당합니다.

1. **5분:** 제목·한 문장 요약·등장 배경을 읽습니다.
2. **10분:** 셀을 실행하기 전에 입력/출력 shape와 metric 방향을 적습니다.
3. **30~40분:** 왼쪽 exercise의 TODO를 직접 구현합니다.
4. **10분:** assert 또는 오류 메시지를 읽고 실패 원인을 말로 설명합니다.
5. **5분:** 막힌 TODO와 같은 번호의 solution 셀만 확인합니다.
6. **10분:** learning rate, temperature, mask, top-k 등 변수 하나만 바꿉니다.
7. **5분:** 가설·변경·결과·실패 예시·다음 결정을 기록합니다.

정답 코드와 다르게 구현했더라도 셀의 계약, shape, assertion과 학습 목표를 만족하면 유효한
풀이일 수 있습니다. 정답을 복사하는 것보다 왜 작동하는지 설명하는 것이 완료 기준입니다.

### 코드 셀의 함수·API 해설 읽는 법

각 노트북은 중요한 라이브러리 함수가 **처음 등장하는 코드 셀 바로 위**에 접이식
`이 셀에서 처음 만나는 함수·API·수식 해설`을 둡니다. 첫 번째 접기를 열면 API 목록이
나오고, 필요한 함수만 한 번 더 펼쳐 signature, 입력·반환값, 사용 이유, 대표 수식, 기호,
수식과 코드의 대응, 직관, tensor shape와 주의점을 확인할 수 있습니다. 같은 설명을 매 셀마다
반복하지 않고 노트북 안의 첫 사용 위치에서만 자세히 다룹니다.

예를 들어 `np.allclose(a, b)`는 배열 두 개가 정확히 같은 객체인지 확인하는 함수가 아닙니다.
각 원소에서 다음 부등식을 검사한 뒤 모든 결과를 하나의 `bool`로 축약합니다.

$$
\lvert a_i-b_i\rvert
\le \mathrm{atol}+\mathrm{rtol}\,\lvert b_i\rvert
$$

따라서 부동소수점 반올림 오차가 있는 결과를 `==`보다 안전하게 검증할 때 사용합니다.
`atol`은 0 근처의 절대 오차, `rtol`은 기준값 크기에 비례하는 상대 오차입니다. 기본
`atol`이 아주 작은 기준값에서는 지나치게 느슨할 수 있고 식에서 $b_i$가 기준이므로,
정밀한 실험에서는 두 허용치를 직접 정해야 합니다.

노트북에서 반복되는 NumPy·PyTorch·pandas·scikit-learn·Python API를 주제별로 다시 찾으려면
[함수·API 한국어 학습 가이드](docs/FUNCTION_API_GUIDE.md)를 사용하세요.

## 1. 기본 AI 엔지니어링 23개

```powershell
# 00~22, 9와 09 모두 입력 가능
.\Start_Paired_Learning.cmd 00
```

번호를 생략하고 Enter만 누르면 현재 런처 기본값은 13이므로, 첫 학습은 반드시 `00`을
지정하세요. `notebooks/00_...`부터 `12_...`까지는 보존용 단일 원본이고, 새로 답을 작성할
때는 `notebooks/exercises/`의 파일을 사용합니다.

| 단계 | 번호 | 핵심 주제 | 대표 결과물 |
|---|---|---|---|
| 환경·데이터·기준선 | 00~04 | Jupyter, Python 설계, NumPy, pandas, scikit-learn | 데이터 리포트, baseline metric, 오류 표 |
| PyTorch 기초 | 05~08 | tensor, autograd, Dataset/DataLoader, MLP, CNN/RNN/attention | 학습 루프, gradient, confusion matrix |
| LLM·RAG 기초 | 09~12 | Transformer, tiny causal LM, 검색, end-to-end 프로젝트 | checkpoint, 생성 비교, grounded prompt |
| 구조·학습 심화 | 13~16 | 모달리티 pipeline, RNN 계열, Transformer 구조와 학습 | scheduler, AMP, clipping, best checkpoint |
| 응용·운영 | 17~22 | embedding, RAG, CNN, tabular/anomaly, GPU 최적화, Hybrid RAG | vector index, 평가표, resume checkpoint |

- 전체 번호·예상 시간: [notebooks/README.md](notebooks/README.md)
- 두 창 학습법: [notebooks/PAIRED_LEARNING.md](notebooks/PAIRED_LEARNING.md)
- TODO와 완료 기준: [docs/EXERCISES.md](docs/EXERCISES.md)
- 상세 학습 순서: [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md)

## 2. 유명 AI 논문 20편

```powershell
# 메뉴에서 선택
.\Start_Paper_Reproductions.cmd

# 09 = Attention Is All You Need
.\Start_Paper_Reproductions.cmd 09
```

<details>
<summary>00~19 논문 목록 펼치기</summary>

| 번호 | 핵심 논문/아이디어 | 번호 | 핵심 논문/아이디어 |
|---:|---|---:|---|
| 00 | LeNet-5 | 10 | GPT-1 |
| 01 | AlexNet | 11 | BERT |
| 02 | U-Net | 12 | VAE |
| 03 | ResNet | 13 | GAN |
| 04 | Dropout | 14 | DQN |
| 05 | Batch Normalization | 15 | GCN |
| 06 | word2vec Negative Sampling | 16 | SimCLR |
| 07 | Sequence to Sequence | 17 | Vision Transformer |
| 08 | Bahdanau Attention | 18 | DDPM |
| 09 | Attention Is All You Need | 19 | CLIP |

</details>

각 노트북 첫 표에는 원 논문의 절·수식·그림과 해당 실습 셀, 관찰할 증거가 연결되어 있습니다.
핵심 연산은 완성된 model 호출 한 줄로 숨기지 않고, layer·model·objective·학습·평가
class와 함수로 나누어 구현합니다. 수식 아래에는 기호, tensor shape, 코드 실행 순서와
해당 방식을 택한 이유를 적었습니다. 이 구조는 논문 성능을 과장하기 위한 것이 아니라,
면접이나 코드 리뷰에서 “무엇을 왜 구현했는가”를 직접 설명할 수 있게 하기 위한 것입니다.

- 전체 목록: [notebooks/paper_reproductions/README.md](notebooks/paper_reproductions/README.md)
- 원문 mapping 읽는 법: [docs/PAPER_REPRODUCTION_GUIDE.md](docs/PAPER_REPRODUCTION_GUIDE.md)
- 포트폴리오 구현 표준: [docs/PORTFOLIO_NOTEBOOK_STANDARD.md](docs/PORTFOLIO_NOTEBOOK_STANDARD.md)

## 3. 7개 분야 × 10편 논문

```powershell
# 분야와 논문을 메뉴로 선택
.\Start_Field_Paper_Labs.cmd

# NLP·LLM 03 = Attention Is All You Need
.\Start_Field_Paper_Labs.cmd nlp_llm 03
```

| 분야와 ID | 10편의 흐름 | 로컬 데이터 | 한국어 독해 노트 |
|---|---|---|---|
| 컴퓨터 비전 `vision` | LeNet → CNN 계보 → detection → ViT/DETR | `vision_shapes.npz` | [VISION](docs/paper_reading_notes/VISION.md) |
| NLP·LLM `nlp_llm` | word2vec → seq2seq → Transformer → GPT/BERT/T5 → LoRA/RAG | `nlp_corpus.json` | [NLP_LLM](docs/paper_reading_notes/NLP_LLM.md) |
| 생성 모델 `generative` | denoising AE → VAE/GAN → DDPM → Score-SDE/LDM | `generative_samples.npz` | [GENERATIVE](docs/paper_reading_notes/GENERATIVE.md) |
| 강화학습 `reinforcement_learning` | REINFORCE → DQN 계열 → actor-critic → PPO/SAC → AlphaZero | `gridworld.json` | [RL](docs/paper_reading_notes/REINFORCEMENT_LEARNING.md) |
| 그래프·추천 `graph_recommendation` | DeepWalk/node2vec → GNN → PinSage/LightGCN | `graph_recommendation.npz` | [GRAPH](docs/paper_reading_notes/GRAPH_RECOMMENDATION.md) |
| 자기지도·멀티모달 `self_supervised_multimodal` | CPC/MoCo/SimCLR → BYOL/DINO/MAE → CLIP/BLIP/Flamingo | `multimodal_pairs.npz` | [MULTIMODAL](docs/paper_reading_notes/SELF_SUPERVISED_MULTIMODAL.md) |
| 증류·경량화 `distillation_compression` | KD → feature/attention distillation → pruning/quantization → mobile/OFA | `compression_bench.npz` | [COMPRESSION](docs/paper_reading_notes/DISTILLATION_COMPRESSION.md) |

논문 전문을 모두 읽기 어렵다면 [한국어 독해 노트 목차](docs/paper_reading_notes/README.md)에서
5분 경로를 먼저 사용하세요. 각 논문에 배경, 선수지식, 핵심 아이디어, 꼭 볼 원문 위치,
수식 직관, TODO 연결, 한계와 자가점검 문항이 있습니다.

분야별 노트북은 `Config → data contract → 핵심 class → objective → training_step →
evaluate` 순서로 읽을 수 있게 구성했습니다. exercise에는 같은 공개 API의 TODO 골격을,
solution에는 shape·gradient·metric assertion까지 포함한 실행 가능한 구현을 둡니다.
`Lab`·`Agent` class는 설명용 장식으로만 두지 않고 주 학습 loop와 평가 경로에서 실제로
호출합니다. setup을 제외한 핵심 code cell 바로 앞에는 원문 위치, 입력/출력 shape,
구현 이유와 완료 증거를 적어 코드와 논문을 위아래로 함께 읽을 수 있게 했습니다.

> **AI 보조 작성 안내:** 한국어 요약은 AI로 초안·구조화를 보조한 2차 교육
> 자료이며 공식 번역이 아닙니다. 수식·실험 조건·인용의 최종 기준은 연결된
> 원 논문입니다. [작성 범위·저작권·정확성 한계](docs/paper_reading_notes/AI_ASSISTED_NOTICE.md)를
> 함께 확인하세요.

- 70편 전체 catalog: [notebooks/field_reproductions/README.md](notebooks/field_reproductions/README.md)
- 분야별 학습 가이드: [docs/FIELD_PAPER_LABS_GUIDE.md](docs/FIELD_PAPER_LABS_GUIDE.md)
- 2023~2025 top-tier 후속 지도: [docs/paper_reading_notes/RECENT_TOP_TIER.md](docs/paper_reading_notes/RECENT_TOP_TIER.md)

## 4. NLP 미니 프로젝트 6개

노트북보다 독립 실행 가능한 Python 파일을 선호하면 [projects/nlp/README.md](projects/nlp/README.md)에서
시작하세요. 각 폴더에는 직접 작성하는 `starter.py`와 실행 가능한 `solution.py`가 있습니다.

| 번호 | 프로젝트 | 핵심 결과 |
|---:|---|---|
| 01 | 텍스트 전처리 | 정규화, PII 마스킹, token 통계 |
| 02 | 의도 분류 | 문자 TF-IDF, 오류 분석 |
| 03 | 의미 검색 | TF-IDF + SVD dense embedding, cosine 검색 |
| 04 | RAG 직접 구현 | 검색 → context → grounded prompt |
| 05 | LangChain RAG | Document, splitter, vector store, retriever, LCEL |
| 06 | RAG 평가 | Recall@k, MRR, coverage, no-answer threshold |

01~04와 06은 외부 API 없이 실행됩니다. 05도 기본은 로컬 embedding과 extractive fallback을
사용합니다. 실제 OpenAI API 연결은 별도 선택 사항이며 `.[openai]` extra 설치와 API 키가
필요합니다.

```powershell
.\.venv\Scripts\python.exe projects\nlp\01_text_preprocessing\solution.py
.\.venv\Scripts\python.exe projects\nlp\05_langchain_rag\solution.py --show-prompt
.\.venv\Scripts\python.exe projects\nlp\06_rag_evaluation\solution.py --show-failures
```

<details>
<summary><strong>코드·스크립트 상세 지도 펼치기</strong></summary>

## 코드 지도

| 경로 | 역할 | 학습자가 할 일 |
|---|---|---|
| `notebooks/exercises/` | 기본 23개 문제 | TODO를 직접 작성 |
| `notebooks/solutions/` | 기본 23개 정답·해설 | 막힌 같은 번호 셀만 확인 |
| `notebooks/paper_reproductions/` | 유명 논문 20편 페어 | 분야 횡단 survey |
| `notebooks/field_reproductions/` | 7개 분야 70편 페어 | 관심 분야 전문화 |
| `projects/nlp/` | 6개 starter/solution 프로젝트 | 노트북 코드를 작은 프로그램으로 전환 |
| `scripts/` | 순차 실행 pipeline 10개 | 인자와 출력 artifact 비교 |
| `src/llm_engineering_lab/` | 재사용 가능한 실제 구현 | notebook 뒤에 함수·클래스 설계 읽기 |
| `data/` | 합성 학습 입력 14종 | schema, split, 누수와 metric 검사 |
| `artifacts/` | checkpoint, metric, log, hardware report | 실행 결과를 실험 노트와 비교 |
| `tests/` | 회귀 테스트 14개 파일 | 코드를 바꾼 뒤 재실행 |
| `tools/` | notebook/data 생성·검증·실행기 | 초반에는 수정하지 말고 validator만 사용 |
| `docs/` | 학습법, GPU, 논문 독해 노트 | 구현 전 선수지식과 완료 기준 확인 |
| `app.py` | Streamlit 관찰 앱 | 학습된 모델·검색 결과를 입력별로 탐색 |

### 순차 실행 스크립트

| 파일 | 하는 일 | 주요 출력/선행 조건 |
|---|---|---|
| `00_generate_data.py` | 기본 합성 데이터 생성 | `data/` |
| `00_python_engineering_patterns.py` | dataclass, Protocol, generator, context manager | 설계 패턴 JSON |
| `01_inspect_data.py` | schema, 결측, 중복, 분포 검사 | 콘솔 데이터 리포트 |
| `02_train_ml_baseline.py` | TF-IDF + scikit-learn 기준선 | model, metrics, errors |
| `03_train_torch_classifier.py` | PyTorch 텍스트 분류 | `artifacts/ticket_classifier.pt` |
| `04_train_tiny_lm.py` | 작은 causal Transformer 학습 | `artifacts/tiny_lm.pt` |
| `05_llm_inference_patterns.py` | batching, sampling, KV cache 비교 | 04 checkpoint 필요 |
| `06_rag_demo.py` | 로컬 검색, 평가, grounded prompt | 지식 문서와 검색 결과 |
| `07_generate_practice_datasets.py` | 추가 모달리티 데이터 4종 생성 | `data/practice/` |
| `08_hardware_report.py` | CPU/GPU/VRAM과 권장 profile 탐지 | `artifacts/hardware_profile.json` |

```powershell
# 전체 pipeline의 대표 실행 순서
.\.venv\Scripts\python.exe scripts\01_inspect_data.py `
  --data data\customer_support_tickets.csv
.\.venv\Scripts\python.exe scripts\02_train_ml_baseline.py
.\.venv\Scripts\python.exe scripts\03_train_torch_classifier.py --epochs 5 --device auto
.\.venv\Scripts\python.exe scripts\04_train_tiny_lm.py --epochs 3 --device auto
.\.venv\Scripts\python.exe scripts\05_llm_inference_patterns.py --device auto
.\.venv\Scripts\python.exe scripts\06_rag_demo.py --evaluate --show-prompt
```

### 재사용 소스 모듈

- `config.py`, `hardware.py`, `acceleration.py`: 설정, seed, 장치 선택, AMP와 하드웨어 profile
- `data.py`, `ml.py`: schema, loader, split, 전통 ML 기준선과 평가
- `torch_text.py`, `transformer.py`: Dataset/DataLoader, 분류기, causal Transformer, 학습·checkpoint
- `nlp_basics.py`, `semantic_search.py`: 정규화·마스킹·token과 dense 검색
- `retrieval.py`, `rag_evaluation.py`, `langchain_rag.py`: 검색, RAG, 평가, LangChain 2-step pipeline

노트북에서 개념을 확인한 뒤 같은 기능이 `src/`에서 어떻게 재사용 가능한 함수·클래스로
분리되었는지 읽으면 실무 코드 구조를 이해하기 쉽습니다.

</details>

## 데이터 지도

전체 schema, shape, 연결 실습과 로드 예시는 [data/README.md](data/README.md)에 모았습니다.

| 그룹 | 파일 수 | 파일 | 생성 seed |
|---|---:|---|---:|
| 기본 | 3 | 고객 문의 CSV, 지식 문서 JSONL, mini corpus | 42 |
| 추가 실습 | 4 | 센서 시계열, 표형 risk, 도형 이미지, RAG 질의 | 2026 |
| 분야별 논문 | 7 | vision, NLP, generative, gridworld, graph, multimodal, compression | 20260814 |

모든 데이터는 합성이며 실제 개인정보가 아닙니다. 분야별 `data/field_curriculum/` 데이터는
CC0-1.0이고 `manifest.json`에 SHA-256이 기록됩니다. 작은 합성 데이터에서 높은 정확도가
나와도 실제 서비스 성능이나 원 논문 성능을 재현한 것으로 해석하면 안 됩니다.

## GPU와 실행 시간

기본값 `auto`는 CUDA → MPS → CPU 순서로 선택합니다. setup 셀에서 다음과 같은 출력이 보이면
GPU가 사용 중입니다.

```text
device=cuda (NVIDIA ...) | AMP policy=enabled
```

```powershell
# GPU PyTorch 설치/복구
.\Install_GPU_PyTorch.cmd

# 현재 하드웨어와 권장 설정 기록
.\.venv\Scripts\python.exe scripts\08_hardware_report.py

# 현재 PowerShell에서 장치 강제 비교
$env:AI_LAB_DEVICE = "cuda"  # 또는 cpu, mps
$env:AI_LAB_AMP = "auto"     # 1, 0도 가능
```

작은 full-batch 모델, Python random walk, tokenization, TF-IDF, 환경 simulation은 GPU 전송·초기화
비용 때문에 CPU가 더 빠를 수 있습니다. 이 경우 CPU에 남겨 두는 것이 정상입니다. 더 자세한
설명은 [GPU 가속 안내](docs/GPU_ACCELERATION.md)를 참고하세요. 이미 열린 커널에서 장치를
바꿨다면 `Kernel → Restart Kernel and Run All Cells`을 실행합니다.

## 설치와 수동 실행

가장 쉬운 방법은 `Install_or_Repair.cmd`를 실행하는 것입니다. 다른 Windows PC에서
수동으로 설치할 때는 **64비트 Python 3.12**와 저장소의 requirements 파일을 사용합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --editable . --no-deps
```

`requirements-windows.txt`는 모든 PC에서 시작할 수 있는 CPU 기준 PyTorch를 설치합니다.
NVIDIA GPU PC에서는 이 설치가 끝난 뒤 `Install_GPU_PyTorch.cmd`를 실행하면 됩니다.
파일별 역할과 오류 복구 방법은 [Windows 설치 안내](docs/WINDOWS_SETUP.md)에 정리했습니다.

Streamlit 앱에서는 데이터 분포, PyTorch tensor shape, 분류 확률, Tiny LM 생성 설정, RAG 검색
문서와 artifact를 한 화면에서 관찰할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 검증 명령

각 validator의 범위가 다르므로 “전체”를 확인할 때는 네 종류를 구분해 실행합니다.

```powershell
# 재사용 코드와 데이터 단위 테스트
.\.venv\Scripts\python.exe -m pytest -q

# 보존 원본 00~12만 검사
.\.venv\Scripts\python.exe tools\validate_all_notebooks.py

# 기본 exercise/solution 23쌍
.\.venv\Scripts\python.exe tools\validate_paired_notebooks.py --execute-solutions

# 유명 논문 20쌍
.\.venv\Scripts\python.exe tools\validate_paper_reproductions.py `
  --execute-solutions --jobs 2

# 분야별 70쌍 + 데이터 manifest/SHA-256
.\.venv\Scripts\python.exe tools\validate_field_reproductions.py `
  --execute-solutions --jobs 2

# 239개 학습 노트북의 함수·API·수식 해설과 페어 일치 검사
.\.venv\Scripts\python.exe tools\validate_api_explanations.py
```

테스트 수와 생성 노트북 수는 자료가 보강되며 달라질 수 있으므로, 고정 숫자보다 위 명령의
현재 결과를 최종 기준으로 사용하세요.

## 학습 체크리스트

### 첫날

- [ ] `Verify_Setup.cmd`가 통과했다.
- [ ] 기본 00번 exercise와 solution이 두 창에 열렸다.
- [ ] setup 셀의 Python, kernel, `device=`를 확인했다.
- [ ] exercise에만 답을 작성하고 solution은 참고용으로 두었다.

### 매 노트북

- [ ] 실행 전에 핵심 tensor shape 또는 예상 metric을 적었다.
- [ ] TODO를 먼저 직접 구현했다.
- [ ] assertion 실패 이유를 설명했다.
- [ ] solution은 막힌 같은 번호 셀만 확인했다.
- [ ] 변수 하나만 바꾸고 기준선과 비교했다.
- [ ] 성공 지표와 실패 예시를 모두 기록했다.
- [ ] 끝난 Jupyter kernel을 종료해 CPU/GPU 메모리를 돌려줬다.

### 매주

- [ ] 완료한 노트북과 아직 설명하지 못하는 개념을 정리했다.
- [ ] 가장 오래 디버깅한 오류 하나와 해결 근거를 기록했다.
- [ ] metric 하나와 실제 실패 사례 하나를 함께 남겼다.
- [ ] 다음 주에 검증할 질문을 한 문장으로 만들었다.

## 실험 노트 템플릿

```text
날짜 / 노트북:
오늘의 질문:
가설:
바꾼 한 가지:
고정한 조건 / seed / device:
결과(metric + 실패 예시):
왜 이런 결과가 나왔다고 생각하는가:
다음 결정:
```

<details>
<summary><strong>자주 묻는 질문 펼치기</strong></summary>

## 자주 묻는 질문

**어디서 시작하나요?**  
`Start_Paired_Learning.cmd 00`에서 시작하세요. 번호를 생략하면 기본값 13이므로 초보는 00을
명시하는 것이 안전합니다.

**exercise와 solution은 어떻게 다른가요?**  
셀 순서와 설명은 같지만 exercise에는 TODO와 빈 구현이 있고 solution에는 실행 가능한 정답과
assertion이 있습니다. 작성은 exercise에만 하세요.

**논문 전문을 모두 읽어야 하나요?**  
아닙니다. 한국어 독해 노트의 5분 요약을 읽고, 안내된 절·수식·그림만 확인한 뒤 실습으로
검증하는 경로를 먼저 권장합니다.

**논문 20편과 분야별 70편을 모두 해야 하나요?**  
17편이 겹칩니다. 20편 트랙으로 지도를 본 뒤 관심 분야 10편을 선택하고, 중복은 복습하거나
건너뛰세요.

**번호는 어떻게 입력하나요?**  
`9`와 `09` 모두 가능합니다. 분야별 런처는 분야를 먼저 고른 뒤 00~09를 선택합니다.

**GPU가 있는데 왜 더 느릴 때가 있나요?**  
아주 작은 모델은 커널 시작과 tensor 전송 비용이 계산보다 큽니다. setup의 `device=`를 확인하고
동일 조건에서 CPU와 CUDA를 측정하세요.

**`device=cpu`만 표시됩니다.**  
`Install_GPU_PyTorch.cmd`를 실행하고 Jupyter kernel을 완전히 재시작하세요. 그래도 같으면
[GPU 안내](docs/GPU_ACCELERATION.md)의 진단 명령을 실행합니다.

**모듈이나 Jupyter kernel을 찾지 못합니다.**  
열려 있는 Jupyter를 닫고 `Install_or_Repair.cmd`와 `Verify_Setup.cmd`를 다시 실행하세요.

**데이터 다운로드나 API 키가 필요한가요?**  
기본 실습에는 필요하지 않습니다. 실제 OpenAI 호출처럼 명시된 선택 기능만 별도 키가 필요합니다.

**노트북 결과가 원 논문 성능인가요?**  
아닙니다. 합성 데이터에서 핵심 수식과 메커니즘을 확인한 교육용 축소 재현입니다.

**내 답이 정답 코드와 다릅니다.**  
입출력 계약, shape, assertion과 학습 목표를 만족하고 이유를 설명할 수 있다면 다른 구현도
가능합니다.

**작성한 exercise가 사라질 수 있나요?**  
`tools/build_*`와 `tools/rebuild_*`는 생성 노트북을 덮어쓸 수 있습니다. 개인 답안은 별도 복사본,
Git branch 또는 실험 노트에 보관하고, 학습 중에는 validator만 실행하세요.

</details>

## 완료 기준

다음을 코드와 자신의 말로 설명할 수 있으면 핵심 목표를 달성한 것입니다.

- 데이터 한 행이 검증·split·batch를 거쳐 모델에 들어가는 과정을 추적한다.
- 기준선과 신경망을 같은 split과 metric으로 공정하게 비교한다.
- forward, loss, backward, optimizer step에서 tensor shape과 상태 변화를 설명한다.
- checkpoint를 새 프로세스에서 복원하고 같은 조건으로 추론한다.
- attention과 causal mask가 미래 정보를 어떻게 막는지 설명한다.
- RAG에서 검색 실패와 생성 실패를 분리하고 Recall@k·MRR로 검색을 평가한다.
- 논문의 핵심 수식·그림과 실습 코드의 대응 관계를 찾는다.
- GPU, latency, memory, 모델 크기와 품질 사이의 trade-off를 측정한다.
- metric뿐 아니라 실패 예시와 재현 조건을 함께 기록한다.

처음부터 모든 자료를 끝내려 하지 말고, **기본 00번 → 작은 성공 → 한 분야 선택 → 반복 실험**
순서로 진행하세요.

## 라이선스와 사용 범위

이 저장소의 코드와 직접 작성한 문서는 [MIT License](LICENSE)로 공개합니다. 개인 복습,
교육, 실험과 포트폴리오 목적으로 자유롭게 활용할 수 있지만, 논문 자체의 저작권은 각 저자와
출판사에 있습니다. 이 저장소는 논문 원문이나 공식 번역을 배포하지 않으며, 원문 링크와
교육용 축소 구현을 제공합니다.

합성 데이터의 세부 생성 방법과 별도 라이선스는 [데이터 안내](data/README.md)를 확인하세요.
AI 보조로 작성한 한국어 요약의 범위와 검증 원칙은
[AI 작성 고지](docs/paper_reading_notes/AI_ASSISTED_NOTICE.md)에 설명되어 있습니다.
