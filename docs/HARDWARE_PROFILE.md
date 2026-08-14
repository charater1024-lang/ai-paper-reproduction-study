# 검증에 사용한 Windows 기준 하드웨어

이 저장소의 전체 CUDA 검증에 사용한 기준 PC 사양은 다음과 같습니다. 다른 Windows 기기에서
같은 사양이 필요한 것은 아니며, 실제 권장값은 각 기기에서 `scripts/08_hardware_report.py`를
실행해 다시 생성합니다.

- 논리 CPU: 16개
- 메모리: 약 31.9GB
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU
- VRAM: 16GB
- PyTorch: CUDA 13.0 빌드, 실제 CUDA tensor/optimizer 동작 검증 완료

이 정도면 작은 Transformer를 직접 학습하고 CNN/RNN/embedding index 실습을 수행하기에
충분합니다. 다만 7B 이상 LLM의 전체 파라미터 학습이나 대규모 corpus 학습을 목표로 하지
않습니다. 이 자료는 원리를 빠르게 반복하고 프로파일링하는 크기로 구성했습니다.

## 권장 확장 순서

| 단계 | d_model | layers | sequence | batch | 권장 용도 |
|---|---:|---:|---:|---:|---|
| 빠른 확인 | 96~128 | 2 | 64~128 | 16~32 | CPU 또는 오류 디버깅 |
| 기본 실습 | 256 | 4 | 256 | 64 | RTX 3080에서 AMP 학습 |
| 확장 실험 | 384~512 | 6 | 256~512 | 16~32 | accumulation 2~4회 사용 |

학습 안정성을 위해 다음 순서를 지키세요.

1. 작은 batch 한 번으로 forward/backward와 shape를 검증합니다.
2. CUDA에서 `autocast`와 `GradScaler`를 켭니다.
3. batch를 늘리기 전에 `torch.cuda.max_memory_allocated()`를 기록합니다.
4. VRAM이 부족하면 batch를 절반으로 줄이고 accumulation을 두 배로 늘립니다.
5. 실험 사이에는 불필요한 tensor 참조를 지우고 kernel을 재시작합니다.

사용 중인 Windows 환경을 측정하려면 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\08_hardware_report.py
```

결과는 `artifacts/hardware_profile.json`에 저장됩니다. 코드에서는
`llm_engineering_lab.hardware.recommend_training_profile()`을 사용할 수 있습니다.
