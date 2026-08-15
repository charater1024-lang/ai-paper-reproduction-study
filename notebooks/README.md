# AI Engineering 따라 치기 노트북

> 새 권장 방식은 [실습 코드 + 정답 코드 2창 학습](PAIRED_LEARNING.md)입니다.
> `exercises/`와 `solutions/`에 서로 짝이 되는 별도 파일이 있으며, 기존 00~12 원본도
> 아래 경로에 그대로 유지됩니다.

각 노트북은 **읽기 → 그대로 실행 → 손으로 다시 입력 → 값을 하나 바꾸기 → TODO 해결 →
assert로 검증** 순서로 진행합니다. 처음부터 정답을 외우는 대신 shape, dtype, loss, gradient,
오류 사례를 설명하는 것을 목표로 합니다.

중요한 NumPy·PyTorch·pandas·scikit-learn 함수는 각 노트북에서 처음 사용하는 코드 셀 바로
위의 접이식 **함수·API·수식 해설**에서 signature, 입력·반환값, 사용 이유, 관련 수식과
기호, 수식과 코드의 대응, shape와 주의점을 확인할 수 있습니다. API가 많은 셀은 필요한
함수만 두 번째 접기에서 열면 됩니다. 전체 주제별 색인은
[함수·API 한국어 학습 가이드](../docs/FUNCTION_API_GUIDE.md)를 참고하세요.

노트북보다 작은 독립 실행 파일로 NLP를 먼저 연습하고 싶다면
[6개 NLP 실습 프로젝트](../projects/nlp/README.md)를 01번부터 진행하세요. 전처리·분류·의미
검색·직접 구현 RAG·LangChain RAG·평가가 `starter.py`/`solution.py` 쌍으로 구성되어 있습니다.

논문 중심으로 공부하려면 [AI 7개 분야 × 분야별 10편 실습](field_reproductions/README.md)을
사용하세요. 총 70편의 실습본·정답본 140개와 다운로드 없는 로컬 학습 데이터가 포함되며,
`.\Start_Field_Paper_Labs.cmd`로 원하는 분야와 논문을 두 창에 열 수 있습니다.
논문 전체를 읽기 어렵다면 [한국어 독해 노트](../docs/paper_reading_notes/README.md)를 먼저 보세요.

## 권장 순서

| 번호 | 주제 | 예상 시간 |
|---:|---|---:|
| 00 | [환경, 커널, 재현 가능한 실행](00_environment_and_jupyterlab.ipynb) | 30분 |
| 01 | [중급 Python과 객체 설계](01_intermediate_python_for_ai.ipynb) | 2~3시간 |
| 02 | [NumPy와 수치 계산](02_numpy_for_ml.ipynb) | 2~3시간 |
| 03 | [pandas 데이터 검증과 EDA](03_pandas_eda_and_group_split.ipynb) | 2~3시간 |
| 04 | [scikit-learn 텍스트 기준 모델](04_sklearn_text_baseline.ipynb) | 3시간 |
| 05 | [PyTorch tensor와 autograd 디버깅](05_pytorch_tensor_autograd_debugging.ipynb) | 3시간 |
| 06 | [Dataset/DataLoader와 텍스트 분류](06_pytorch_text_classification_pipeline.ipynb) | 3~4시간 |
| 07 | [신경망 핵심 원리 직접 구현](07_neural_networks_from_scratch.ipynb) | 3~4시간 |
| 08 | [CNN, RNN, Attention 비교](08_cnn_rnn_attention_comparison.ipynb) | 3~4시간 |
| 09 | [Transformer 직접 구현](09_transformer_from_scratch.ipynb) | 4시간 |
| 10 | [Tiny LM 학습, 생성, KV cache](10_tiny_lm_training_and_generation.ipynb) | 3~4시간 |
| 11 | [검색과 RAG 근거화](11_rag_retrieval_and_grounding.ipynb) | 3시간 |
| 12 | [End-to-End AI Engineer 미니 프로젝트](12_end_to_end_ai_engineer_project.ipynb) | 4~6시간 |

2창 심화 자료는 다음 순서로 이어집니다. 각 링크는 실습본이며 같은 이름의 정답본은
`solutions/`에 있습니다.

| 번호 | 심화 주제 | 실습본 |
|---:|---|---|
| 13 | 데이터 모달리티와 파이프라인 | [열기](exercises/13_data_modalities_and_pipelines.ipynb) |
| 14 | RNN/GRU/LSTM 시퀀스 모델링 | [열기](exercises/14_rnn_sequence_modeling.ipynb) |
| 15 | Transformer 구조 직접 구현 | [열기](exercises/15_transformer_architecture_lab.ipynb) |
| 16 | Transformer 학습과 생성 | [열기](exercises/16_transformer_training_lab.ipynb) |
| 17 | Embedding과 Vector Index | [열기](exercises/17_embeddings_and_vector_index.ipynb) |
| 18 | End-to-End RAG | [열기](exercises/18_rag_end_to_end.ipynb) |
| 19 | CNN 이미지 분류 | [열기](exercises/19_cnn_image_classification.ipynb) |
| 20 | 표형 MLP와 Autoencoder 이상 탐지 | [열기](exercises/20_tabular_mlp_and_anomaly_detection.ipynb) |
| 21 | GPU 학습 최적화 | [열기](exercises/21_gpu_training_and_optimization.ipynb) |
| 22 | Hybrid RAG와 회귀 평가 | [열기](exercises/22_hybrid_rag_evaluation.ipynb) |

한 번에 모두 끝내려 하지 말고 하루에 한 노트북의 절반 정도를 권장합니다. 각 TODO의
정답 코드는 Git에 바로 덮어쓰기보다 새 셀에 작성하면 비교하기 쉽습니다.

`Start_JupyterLab.cmd`를 실행하면 00번이 자동으로 열립니다. 오른쪽 위 커널은
`Python (AI Engineering Lab)`을 선택하세요. 전체 자료의 구조 검사는 프로젝트 루트에서
`.venv\Scripts\python.exe tools\validate_all_notebooks.py`로 다시 실행할 수 있습니다.
