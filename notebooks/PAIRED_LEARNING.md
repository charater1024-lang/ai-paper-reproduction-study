# 실습 코드 + 정답 코드 2창 학습

`notebooks/exercises/`에는 일부 구현을 비워 둔 **실습 노트북**, `notebooks/solutions/`에는
같은 파일명과 셀 순서를 가진 **정답 노트북**이 있습니다. 바탕화면의
**AI 코딩연습 - 기본 실습과 정답**을 실행하면 번호 선택 후 두 파일이 좌우 창으로 열립니다.

## 사용 순서

1. 왼쪽 실습 창에서 주석, 입력 shape, 기대 출력을 먼저 읽습니다.
2. 오른쪽 정답 창을 가린 상태로 `TODO` 또는 `NotImplementedError` 부분을 작성합니다.
3. 막히면 정답을 한 줄만 확인하고 다시 왼쪽으로 돌아옵니다.
4. 결과가 같아도 값 하나를 바꾸고 왜 달라졌는지 Markdown 셀에 기록합니다.
5. 학습이 끝나면 서버 콘솔에서 `Ctrl+C`를 눌러 GPU와 kernel을 해제합니다.

명령줄에서 번호를 바로 지정할 수도 있습니다.

```powershell
.\Start_Paired_Learning.cmd 14   # RNN/GRU/LSTM
.\Start_Paired_Learning.cmd 16   # Transformer 학습
.\Start_Paired_Learning.cmd 18   # End-to-End RAG
```

## 커리큘럼

| 번호 | 영역 | 핵심 실습 |
|---:|---|---|
| 00~04 | Python·데이터·ML | 환경, 객체 설계, NumPy, pandas, scikit-learn |
| 05~08 | PyTorch·DL | autograd, DataLoader, 신경망, CNN/RNN/Attention |
| 09~12 | LLM·RAG 기초 | Transformer, Tiny LM, 검색, 미니 프로젝트 |
| 13 | 다양한 데이터 파이프라인 | 표형·시계열·텍스트의 split, scaling, 누수 방지 |
| 14 | RNN 시퀀스 모델링 | RNN/GRU/LSTM, padding/mask, 분류·예측, clipping |
| 15 | Transformer 구조 | QKV, multi-head attention, mask, residual, FFN |
| 16 | Transformer 학습 | tokenizer, teacher forcing, AMP, perplexity, generation |
| 17 | Embedding과 Vector Index | sparse/dense embedding, cosine, 저장·로드, recall@k/MRR |
| 18 | End-to-End RAG | chunking, retrieval, prompt, citation, grounding, no-answer |
| 19 | 이미지 CNN | 합성 이미지, augmentation, CNN, confusion matrix |
| 20 | 표형·이상 탐지 | 불균형 분류, threshold, MLP/autoencoder 비교 |
| 21 | GPU 학습 최적화 | AMP, accumulation, clipping, memory, checkpoint/resume |
| 22 | Hybrid RAG 평가 | sparse+dense fusion, RRF, 회귀 평가, index 갱신 |

## 기준 Windows PC에서 검증한 기본값

전체 검증에 사용한 기준 환경은 논리 CPU 16개, RAM 약 32GB, RTX 3080 Laptop GPU
16GB입니다. 다른 Windows PC에서는 `scripts/08_hardware_report.py`의 추천값을 사용하세요. 실습은
브라우저와 Jupyter가 동시에 떠 있어도 안정적으로 실행되도록 작은 모델에서 시작하며,
CUDA에서는 mixed precision을 선택적으로 사용합니다. 권장값은
`scripts/08_hardware_report.py`로 다시 확인할 수 있습니다.

- Transformer 시작점: `d_model=256`, 4 layers, sequence length 256
- 텍스트 batch: 64, 이미지 batch: 256부터 시작
- Windows Jupyter의 DataLoader worker 기본값: 0
- 긴 학습은 AMP + gradient accumulation으로 메모리 여유 확보
- OOM 발생 시 batch를 절반으로 줄이고 kernel을 재시작

## 검증

```powershell
# 짝 구성과 TODO 확인
.\.venv\Scripts\python.exe tools\validate_paired_notebooks.py

# 모든 정답 노트북을 실제 커널에서 실행
.\.venv\Scripts\python.exe -X utf8 tools\validate_paired_notebooks.py --execute-solutions

# 2창 서버 시작만 검사하고 브라우저는 열지 않음
.\.venv\Scripts\python.exe -X utf8 tools\start_paired_lab.py 13 --smoke-test

# 생성기를 수정한 뒤 전체 짝 자료를 다시 만들 때
.\.venv\Scripts\python.exe -X utf8 tools\rebuild_paired_curriculum.py
```

정답 노트북을 직접 수정해도 실습 파일에는 반영되지 않습니다. 원본 00~12 노트북도
`notebooks/` 바로 아래에 그대로 보존되어 있습니다.

실습과 정답에서 생성하는 checkpoint는 파일명에 `_exercise` 또는 `_solution`이 붙어 서로
덮어쓰지 않습니다. 2창 런처를 여러 번 실행할 때도 Jupyter 로그는 port와 process ID별로
분리됩니다. 그래도 GPU 메모리를 아끼려면 한 번에 하나의 학습 셀만 실행하세요.
