# 2023–2025 최신 top-tier 후속 논문 지도

## 선정 기준

여기서 `top-tier`는 보편적인 절대 순위를 뜻하지 않습니다. CVPR·ICCV·ECCV·SIGGRAPH/TOG,
NeurIPS·ICML·ICLR, ACL 계열·COLM, KDD·SIGIR·WSDM·TheWebConf, MLSys 등의 **공식
proceedings에서 venue와 연도가 확인되는 논문** 가운데 기존 70편과 학습 연결성이 큰 연구를
뜻합니다. 단순 arXiv 제출이나 심사 중인 논문을 정식 채택 논문으로 표시하지 않았습니다.

2026년은 출판 기록이 계속 갱신되는 시점이므로, 현재 자료에는 검증이 끝난 2023–2025 논문을
중심으로 넣었습니다. 각 제목의 공식 원문 링크, 주목할 이유와 선수지식은 연결된 분야 문서의
마지막 `후속 학습` 절에 있습니다.

## 분야별 빠른 목록

| 분야 | 추가 후속 논문 | 상세 요약 |
|---|---|---|
| 비전 | Segment Anything, 3D Gaussian Splatting, Depth Anything, Grounding DINO, VGGT | [보기](VISION.md) |
| NLP·LLM | DPO, QLoRA, Self-RAG, Mamba, LongRoPE2 | [보기](NLP_LLM.md) |
| 생성 모델 | Flow Matching, Consistency Models, DiT, EDM2, VAR | [보기](GENERATIVE.md) |
| 강화학습·에이전트 | BBF, RLPD, TD-MPC2, Stop Regressing, WorldCoder | [보기](REINFORCEMENT_LEARNING.md) |
| 그래프·추천 | Exphormer, LightGCL, GraphACL, graph prompt-tuning for recommendation, Subgraphormer | [보기](GRAPH_RECOMMENDATION.md) |
| 자기지도·멀티모달 | I-JEPA, SigLIP, BLIP-2, LLaVA, DINOv2 | [보기](SELF_SUPERVISED_MULTIMODAL.md) |
| 증류·경량화 | SparseGPT, SmoothQuant, AWQ, Wanda, QuaRot, SpinQuant, SVDQuant | [보기](DISTILLATION_COMPRESSION.md) |

## 무엇부터 볼까

- 로컬 LLM을 작게 돌리는 데 관심이 있으면 `SmoothQuant → AWQ → QuaRot/SpinQuant`를 봅니다.
- LLM pruning은 `SparseGPT → Wanda` 순서가 계산비용과 중요도 지표의 차이를 보기 좋습니다.
- 이미지 생성의 현대 흐름은 `DiT → EDM2 → VAR`, 생성 과정 자체의 수학은
  `Flow Matching → Consistency Models`이 좋습니다.
- 멀티모달 응용은 `SigLIP → BLIP-2/LLaVA`, 범용 시각 표현은 `I-JEPA/DINOv2`가 연결됩니다.
- 장문 LLM은 `Mamba → LongRoPE2`, preference alignment는 `DPO`를 먼저 봅니다.

최신 논문은 실험 설정과 공개 구현이 빠르게 바뀝니다. 공식 프로시딩의 본문·supplementary를
먼저 기준으로 삼고, 블로그나 재현 저장소의 수치는 원 논문 결과와 구분해서 기록하세요.
