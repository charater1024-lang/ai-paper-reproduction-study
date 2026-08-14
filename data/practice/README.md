# 추가 실습 데이터

이 폴더의 파일은 `scripts/07_generate_practice_datasets.py`로 동일하게 다시 만들 수 있는
합성 데이터입니다. 실제 개인정보를 포함하지 않습니다.

| 파일 | 형태 | 주요 실습 |
|---|---|---|
| `sensor_timeseries.csv` | 12개 장비의 5분 간격 센서 시계열 | windowing, RNN/GRU/LSTM, forecasting, anomaly |
| `tabular_risk.csv` | 범주형·수치형·결측치·불균형 label | preprocessing, tabular MLP, threshold tuning |
| `image_shapes.npz` | 28×28 원·사각형·십자 이미지 1,200개 | Dataset, augmentation, CNN, confusion matrix |
| `rag_queries.jsonl` | 질의와 관련 문서 ID | recall@k, MRR, no-answer, RAG 회귀 평가 |

재생성:

```powershell
.\.venv\Scripts\python.exe scripts\07_generate_practice_datasets.py
```

원본과 다른 실험 세트를 만들려면 `--seed` 값을 변경하되, train/validation/test split의
group·시간 순서·class balance를 다시 확인하세요.
