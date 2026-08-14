# AI 핵심 논문 70편 한국어 독해 노트

논문 전문을 처음부터 끝까지 읽기 어려운 독학자가 **연구 배경 → 핵심 아이디어 → 꼭 볼
원문 위치 → 축소 실습** 순서로 공부할 수 있게 만든 한국어 길잡이입니다. 7개 분야의 핵심
논문 70편과 2023–2025년 중심의 후속 논문 37편 지도를 제공합니다.

> **AI 보조 작성 자료입니다.** 요약 초안과 문장 정리에 AI를 활용했으며, 공식 번역이나
> 논문 저자의 해설이 아닙니다. 내용과 인용의 최종 기준은 반드시 연결된 원문입니다.
> [작성 원칙·저작권·정확성 한계](AI_ASSISTED_NOTICE.md)를 먼저 확인하세요.

## 바로 찾기

| 분야 | 핵심 논문 | 한국어 독해 노트 | 실습본·정답본 |
|---|---:|---|---|
| 컴퓨터 비전 | 10편 | [VISION.md](VISION.md) | [노트북](../../notebooks/field_reproductions/vision/) |
| NLP·LLM | 10편 | [NLP_LLM.md](NLP_LLM.md) | [노트북](../../notebooks/field_reproductions/nlp_llm/) |
| 생성 모델 | 10편 | [GENERATIVE.md](GENERATIVE.md) | [노트북](../../notebooks/field_reproductions/generative/) |
| 강화학습·에이전트 | 10편 | [REINFORCEMENT_LEARNING.md](REINFORCEMENT_LEARNING.md) | [노트북](../../notebooks/field_reproductions/reinforcement_learning/) |
| 그래프·추천 | 10편 | [GRAPH_RECOMMENDATION.md](GRAPH_RECOMMENDATION.md) | [노트북](../../notebooks/field_reproductions/graph_recommendation/) |
| 자기지도·멀티모달 | 10편 | [SELF_SUPERVISED_MULTIMODAL.md](SELF_SUPERVISED_MULTIMODAL.md) | [노트북](../../notebooks/field_reproductions/self_supervised_multimodal/) |
| 지식 증류·모델 경량화 | 10편 | [DISTILLATION_COMPRESSION.md](DISTILLATION_COMPRESSION.md) | [노트북](../../notebooks/field_reproductions/distillation_compression/) |

- [2023–2025 최신 top-tier 후속 논문 지도](RECENT_TOP_TIER.md)
- [AI 보조 작성 및 공개 원칙](AI_ASSISTED_NOTICE.md)
- [분야별 실습 전체 안내](../../notebooks/field_reproductions/README.md)

각 핵심 논문에는 다음 내용이 반복됩니다.

1. 한 문장 요약
2. 등장 배경과 이전 방법의 한계
3. 읽기 전에 알아야 할 기초지식
4. 핵심 아이디어
5. 꼭 볼 원문의 절·수식·그림
6. 수식·알고리즘의 한국어 직관
7. 실습본·정답본과 TODO 연결
8. 5분 요약, 재현 한계, 자가점검

## 독학할 때 이렇게 사용하세요

### 5분: 전체 지도를 먼저 잡기

`한 문장 요약 → 배경 → 핵심 아이디어 → 5분 요약`만 읽습니다. “이 논문은 무엇의 어떤
한계를 해결했는가?”를 자신의 말로 한 문장 설명할 수 있으면 다음 논문으로 넘어갑니다.

### 20분: 원문을 선택적으로 확인하기

5분 경로에 `기초지식 → 꼭 볼 원문 위치 → 수식 직관`을 더합니다. 안내된 절·식·그림을
원문에서 직접 확인하고, 주요 텐서의 입력 shape와 출력 shape를 종이에 적어 봅니다. 요약과
원문이 다르게 읽히면 원문을 기준으로 개인 노트를 수정합니다.

### 60분: 수식에서 포트폴리오 코드까지 조립하기

20분 경로 뒤 같은 번호의 `exercises` 노트북을 열어 세 단계로 풉니다.

1. 원문의 핵심 수식·알고리즘에 대응하는 TODO를 구현합니다.
2. 같은 계산을 `Config → 전처리 → 명시적 class → objective/update` 구조로 조립합니다.
3. 각 Task의 assertion으로 shape·finite loss·지표를 확인하고, 마지막 조립·평가 셀에서
   class/config 또는 finite-history/metric contract를 검증합니다.

먼저 정답을 보지 말고, assertion이 실패한 이유를 설명할 수 없을 때만 `solutions`의 같은 번호
셀을 확인합니다. 마지막에 변수 하나를 바꾸고 지표가 왜 달라졌는지 세 문장으로 기록하면 한
회차가 끝납니다.

Windows에서는 저장소 루트에서 다음처럼 분야와 번호를 선택할 수 있습니다.

```powershell
.\Start_Field_Paper_Labs.cmd
```

실행 셀의 `device=cuda` 또는 `device=cpu` 표시는 현재 선택된 연산 장치입니다. 실습은 원 논문의
대규모 학습을 복제하는 것이 아니라, 저장소의 작은 로컬 데이터로 핵심 계산을 확인하도록
설계되어 있습니다.

## 추천 선행 순서

- 비전: `LeNet → AlexNet/VGG → ResNet → ViT → DETR`
- NLP·LLM: `word2vec → seq2seq/attention → Transformer → BERT/GPT → T5/LoRA/RAG`
- 생성: `autoencoder → VAE/GAN → DDPM → Score-SDE/LDM`
- 강화학습: `REINFORCE → DQN 계열 → actor-critic → PPO/SAC → AlphaZero`
- 그래프·추천: `DeepWalk/node2vec → GCN → GraphSAGE/GAT → PinSage/LightGCN`
- 자기지도·멀티모달: `CPC/MoCo → SimCLR/BYOL/DINO → MAE → CLIP/BLIP/Flamingo`
- 경량화: `KD → feature/attention distillation → pruning/quantization → mobile CNN → OFA`

## 읽고 인용할 때의 원칙

- 이 자료는 원문을 대체하지 않습니다. 과제·보고서·논문에는 이 요약이 아니라 원 논문을
  확인하고 원 논문을 인용하세요.
- 원문의 문장, 표, 그림을 장문으로 옮기지 않았습니다. 필요한 원문 위치와 공식 링크를
  안내하고 핵심 내용을 한국어로 풀어 설명합니다.
- 노트북의 수치와 plot은 로컬 합성 데이터에서 핵심 계산을 확인한 결과입니다. 원 논문의
  benchmark, 대규모 일반화, 실제 기기 latency 또는 에너지 효율을 재현한 결과가 아닙니다.
- `top-tier`는 절대적인 논문 순위가 아닙니다. 공식 주요 학회 게재 여부, 교육적 연결성과 분야
  다양성을 기준으로 만든 읽기 목록이며, venue·연도·링크는 시간이 지나며 바뀔 수 있습니다.
- AI가 만든 설명에는 누락이나 오해가 남을 수 있습니다. 수식 번호, 실험 조건, 수치, venue와
  최신성은 공식 논문 및 공식 proceedings에서 다시 확인하세요.

오류를 발견하면 어떤 문서의 어떤 논문인지, 그리고 확인 가능한 원문 절·식·그림을 함께 적어
GitHub Issue 또는 Pull Request로 알려 주세요.
