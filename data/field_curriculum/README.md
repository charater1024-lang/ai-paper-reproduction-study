# 분야별 논문 실습용 로컬 데이터

이 폴더의 데이터는 `tools/generate_field_datasets.py`가 고정 seed로 만든 완전 합성 데이터입니다.
사람이나 실제 서비스에서 수집한 정보가 없으며 CC0-1.0으로 사용할 수 있습니다. 목적은 원 논문의
최종 benchmark를 흉내 내는 것이 아니라 각 분야의 입력·정답 구조를 네트워크 없이 재현하는 것입니다.

| 파일 | 분야 | 주요 배열·필드 |
|---|---|---|
| `vision_shapes.npz` | 컴퓨터 비전 | 16×16 영상, class, mask, bounding box |
| `nlp_corpus.json` | NLP·LLM | 감성 문장, 번역 pair, 검색 문서와 질의 |
| `generative_samples.npz` | 생성모델 | 2D 혼합분포, 8×8 영상, paired domain |
| `gridworld.json` | 강화학습 | LineWorld 전이, reward, bandit 평균 |
| `graph_recommendation.npz` | 그래프·추천 | adjacency, node feature/label, user-item 행렬 |
| `multimodal_pairs.npz` | 자기지도·멀티모달 | 영상, image/text feature, token, class prototype |
| `compression_bench.npz` | 증류·모델 경량화 | teacher logits/hidden/attention, calibration tensor, 영상, supernet 후보 |

`manifest.json`에는 생성 seed, 파일 크기, SHA-256이 기록됩니다. 다시 생성하면 같은 버전의 NumPy
환경에서 동일한 논리 데이터가 만들어지며, 노트북은 인터넷 다운로드를 수행하지 않습니다.
