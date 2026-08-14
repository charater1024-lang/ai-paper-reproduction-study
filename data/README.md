# 학습 데이터 안내

이 폴더에는 이 저장소의 노트북·스크립트·미니 프로젝트가 사용하는 **로컬 합성 데이터 14종**과
분야별 데이터 무결성 manifest가 들어 있습니다. 실제 고객이나 서비스에서 수집한 개인정보는
포함하지 않으며, 논문 원본 benchmark를 복제하기보다 데이터 흐름과 핵심 알고리즘을 빠르게
연습할 수 있도록 작게 설계했습니다.

## 데이터를 공부하는 공통 순서

새 파일을 만나면 모델부터 실행하지 말고 다음 순서로 확인하세요.

1. 행·배열 수, 열·key 이름, shape, dtype을 출력합니다.
2. 입력 feature와 예측 target을 자신의 말로 정의합니다.
3. train/validation/test 분할 기준과 누수 가능성을 확인합니다.
4. 무작위 예측이나 단순 기준선의 metric을 먼저 계산합니다.
5. 모델을 학습한 뒤 성공 사례뿐 아니라 실패 사례를 직접 봅니다.
6. 한 번에 변수 하나만 바꾸고 결과와 해석을 기록합니다.

## 1. 기본 AI·NLP 데이터

| 파일 | 현재 규모 | 사용 위치 | 주요 학습 내용 |
|---|---:|---|---|
| `customer_support_tickets.csv` | 100행 × 8열 | 기본 03·04·06·10·12, scripts 01~04 | EDA, group split, TF-IDF, PyTorch 분류, tiny LM |
| `raw/knowledge_base.jsonl` | 12개 문서 | 기본 11·12·17·18·22, NLP 프로젝트 03~06 | 검색, vector index, RAG, Recall@k·MRR |
| `raw/mini_corpus.txt` | 911자 | tiny LM 확장 실험 | 토큰화, next-token window, causal LM |

`customer_support_tickets.csv`에서는 같은 `case_id`가 분할 양쪽에 들어가 누수를 만들지 않는지,
레이블별 행 수가 작은 상황에서 accuracy만 믿어도 되는지를 확인하세요. `knowledge_base.jsonl`은
최종 답변보다 먼저 검색 문서 ID와 점수를 검사하는 데 사용합니다.

## 2. 모달리티별 추가 실습 데이터

상세 설명은 [`practice/README.md`](practice/README.md)에 있습니다.

| 파일 | 현재 규모 | 연결 실습 | 먼저 확인할 질문 |
|---|---:|---|---|
| `practice/sensor_timeseries.csv` | 3,444행 × 8열, 장비 12개 | 기본 13·14·20 | 시간 순서를 보존했는가? 같은 장비가 split을 가로지르는가? |
| `practice/tabular_risk.csv` | 1,500행 × 10열 | 기본 13·20 | imputer/scaler를 train에만 fit했는가? class imbalance는 어떤가? |
| `practice/image_shapes.npz` | 이미지 1,200개, `28×28`, 3 class | 기본 13·19 | channel 축과 label dtype이 맞는가? augmentation은 train에만 적용되는가? |
| `practice/rag_queries.jsonl` | 평가 질의 13개 | 기본 18·22, RAG 평가 | 정답 문서가 top-k에 드는가? no-answer 질의를 보류하는가? |

## 3. 7개 분야 × 논문 실습 데이터

이 데이터는 `tools/generate_field_datasets.py`가 고정 seed로 생성하며, 분야별 70편의 축소 재현
노트북에서 인터넷 다운로드 없이 사용합니다. 상세 라이선스와 key 설명은
[`field_curriculum/README.md`](field_curriculum/README.md)에 있습니다.

| 파일 | 핵심 shape·구성 | 연결 분야 | 연습할 메커니즘 |
|---|---|---|---|
| `field_curriculum/vision_shapes.npz` | 이미지 480개 `1×16×16`, mask, box, 4 class | 컴퓨터 비전 | 분류, segmentation, detection, ViT |
| `field_curriculum/nlp_corpus.json` | 문장 360개, 번역 pair 60개, 문서 5개, 질의 3개 | NLP·LLM | SGNS, seq2seq, attention, MLM, LoRA, RAG |
| `field_curriculum/generative_samples.npz` | 2D point 600개, 이미지 300개 `1×8×8`, 두 domain | 생성 모델 | AE/VAE, GAN, diffusion, score, latent model |
| `field_curriculum/gridworld.json` | 전이 14개, bandit arm 4개, offline noise 128개 | 강화학습·에이전트 | policy gradient, value learning, actor-critic, search |
| `field_curriculum/graph_recommendation.npz` | node 120개, user-item `40×60`, relation graph `3×12×12` | 그래프·추천 | random walk, message passing, link/recommendation |
| `field_curriculum/multimodal_pairs.npz` | image-text pair 240개, sequence `240×8×4` | 자기지도·멀티모달 | contrastive learning, masking, dual encoder, captioning |
| `field_curriculum/compression_bench.npz` | tabular 512개, image 320개, teacher/attention/calibration 배열 | 증류·경량화 | KD, pruning, quantization, mobile block, supernet |

`field_curriculum/manifest.json`에는 생성 seed, 파일 크기와 SHA-256이 기록됩니다. 분야별 validator는
노트북을 실행하기 전에 이 manifest와 실제 파일을 대조합니다.

## 로드 예시

```python
from pathlib import Path
import json

import numpy as np
import pandas as pd

root = Path("data")

tickets = pd.read_csv(root / "customer_support_tickets.csv")
print(tickets.shape, tickets.dtypes)

with (root / "raw" / "knowledge_base.jsonl").open(encoding="utf-8") as file:
    documents = [json.loads(line) for line in file if line.strip()]
print(len(documents), documents[0].keys())

vision = np.load(root / "field_curriculum" / "vision_shapes.npz")
print({key: vision[key].shape for key in vision.files})
```

노트북 안에서는 저장소 루트를 찾는 setup 셀을 먼저 실행하세요. Jupyter의 현재 폴더가 달라지면
상대 경로 `data/...`가 실패할 수 있습니다.

## 재생성

기존 파일을 보존한 채 기본 데이터를 확인하거나, 명시적으로 다시 만들 수 있습니다.

```powershell
# 기본 고객 문의·지식 문서·미니 corpus: 기존 파일 보존
.\.venv\Scripts\python.exe scripts\00_generate_data.py

# 기본 데이터 강제 재생성
.\.venv\Scripts\python.exe scripts\00_generate_data.py --force

# 시계열·표형·이미지·RAG 평가 데이터
.\.venv\Scripts\python.exe scripts\07_generate_practice_datasets.py

# 7개 분야 논문 데이터와 SHA-256 manifest
.\.venv\Scripts\python.exe tools\generate_field_datasets.py
```

seed나 생성 규칙을 바꿨다면 기존 metric과 직접 비교하지 마세요. 새 데이터 버전, seed, split,
모델 설정을 실험 노트에 함께 기록해야 합니다.

## 데이터 검증

```powershell
# 데이터 로더·생성기의 단위 테스트
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_dataset_generator.py `
  tests\test_practice_data.py `
  tests\test_field_curriculum_data.py

# 분야 데이터 manifest/SHA-256와 노트북 페어 검사
.\.venv\Scripts\python.exe tools\validate_field_reproductions.py
```

## 개인정보와 라이선스

- 모든 파일은 교육을 위해 생성한 합성 예제이며 실제 개인정보가 아닙니다.
- `field_curriculum/`의 데이터는 CC0-1.0으로 제공됩니다.
- 데이터가 작기 때문에 얻은 정확도나 loss를 실제 서비스 성능 또는 원 논문 benchmark로
  해석하면 안 됩니다.
- 자신의 데이터를 추가할 때는 원문 개인정보를 notebook 출력·로그·Git에 남기지 말고,
  train/validation 누수와 사용 권한을 먼저 확인하세요.
