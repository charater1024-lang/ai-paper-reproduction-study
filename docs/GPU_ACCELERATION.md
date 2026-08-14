# GPU 자동 가속 사용 안내

이 저장소의 PyTorch 실습은 기본값 `AI_LAB_DEVICE=auto`에서 다음 순서로 장치를 선택합니다.

1. NVIDIA CUDA GPU
2. Apple Metal(MPS)
3. CPU

게시 전 기준 검증 PC에서는 **NVIDIA GeForce RTX 3080 Laptop GPU 16GB**와
`PyTorch 2.13.0+cu130`이 확인되었습니다. 다른 Windows PC에서는 해당 기기의 GPU 이름이
표시됩니다. 노트북 setup 셀에 `device=cuda (...)`가 출력되면 GPU를 사용 중입니다.

## 평소에는 설정하지 않아도 됩니다

바탕화면 바로가기나 `Start_*` 실행 파일을 평소처럼 사용하면 자동 선택됩니다. 노트북의 setup
셀을 먼저 실행한 뒤 출력의 `device=`를 확인하세요. CUDA에서는 TF32와 고정 shape용 cuDNN
benchmark가 활성화되고, `autocast`를 사용하는 큰 학습에는 AMP policy가 적용됩니다. 수치 임계값을
정확히 관찰하는 극소형 논문 재현은 GPU에서도 FP32를 유지할 수 있습니다.

장치를 강제로 비교하려면 JupyterLab을 시작하기 **전** PowerShell에서 설정합니다.

```powershell
$env:AI_LAB_DEVICE = "cuda" # CUDA가 없으면 설명이 포함된 오류
$env:AI_LAB_DEVICE = "cpu"  # CPU 재현성·속도 비교
$env:AI_LAB_DEVICE = "auto" # 기본값 복원

$env:AI_LAB_AMP = "0"       # CUDA mixed precision만 끄기
$env:AI_LAB_CPU_THREADS = "4"
```

현재 커널에서 직접 확인하는 최소 코드는 다음과 같습니다.

```python
from llm_engineering_lab.acceleration import get_accelerator

ACCELERATOR = get_accelerator()
DEVICE = ACCELERATOR.device
print(ACCELERATOR.summary())
```

## GPU를 쓰지 않는 부분도 있습니다

GPU는 큰 행렬곱·convolution·attention처럼 병렬 계산량이 충분할 때 유리합니다. 다음 작업은 CPU가
더 빠르거나 CUDA 연산 자체가 없어 의도적으로 CPU에서 수행합니다.

- NumPy·pandas 전처리, 문자열 tokenization
- 강화학습 환경 step, Python graph random walk와 이웃 sampling
- TF-IDF·희소 검색, 작은 RAG orchestration
- PyTorch CUDA가 지원하지 않는 `int32` 양자화 행렬곱
- 모델보다 kernel 시작·데이터 전송 시간이 더 큰 극소형 불변식 실험

이 경우에도 뒤따르는 neural model 학습이 있으면 준비된 batch를 선택 장치로 한 번 옮겨 학습합니다.
작은 실습 하나의 총시간은 GPU 초기화 때문에 CPU와 비슷할 수 있지만, 큰 모델·batch·반복 횟수로
확장할수록 GPU 이점이 커집니다.

## 설치와 확인

CUDA가 있는데 setup 셀에 `device=cpu`가 나오면 먼저 모든 Jupyter 창을 닫고 다음 파일을
실행합니다.

```bat
Install_GPU_PyTorch.cmd
Verify_Setup.cmd
```

`Verify_Setup.cmd`에서 `CUDA: True`와 GPU 이름을 확인합니다. GPU용 PyTorch는 공식
[PyTorch CUDA wheel index](https://download.pytorch.org/whl/cu130/torch/)에서 설치합니다.

## 느리거나 메모리가 부족할 때

- 같은 GPU에서 정답 노트북 여러 개를 동시에 전체 실행하지 않습니다.
- 먼저 작은 batch로 동작을 확인하고, GPU 사용률이 낮을 때 batch를 늘립니다.
- 노트북 종료 후 `Kernel → Shut Down Kernel`을 눌러 VRAM을 반환합니다.
- out-of-memory가 나면 커널을 재시작하고 batch·sequence length·image resolution 순으로 줄입니다.
- CPU와 GPU 시간을 잴 때는 CUDA 비동기 실행 때문에 반드시 synchronize 전후로 측정합니다.
