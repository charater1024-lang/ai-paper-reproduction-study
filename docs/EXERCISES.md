# 실습 문제와 TODO

정답을 베끼기보다 작은 테스트를 먼저 만들고 TODO를 한 개씩 해결하세요. 각 문제의 “검증”은 완료 여부를 스스로 판정하기 위한 최소 기준입니다. ★는 난이도이며, ★★★는 포트폴리오 확장에 가깝습니다.

## A. Python과 설계

### A0. 파이프라인 단계 확장 ★

TODO:

- `00_python_engineering_patterns.py`의 `Step[Ticket]`을 구현하는 `UnicodeNormalizer`를 추가합니다.
- 기존 `Pipeline`에 새 단계를 끼워 넣되 다른 단계 코드는 수정하지 않습니다.
- 원본 `Ticket`이 변하지 않고 새 객체가 반환되는지 테스트합니다.

검증: Unicode 조합형/분해형 입력이 같은 결과가 되고 기존 정규화·마스킹 테스트도 통과합니다.

### A1. 안전한 실험 설정 ★

TODO:

- batch size, learning rate, epoch, seed, device를 가진 `TrainingConfig` 클래스를 만듭니다.
- 음수/0 값과 지원하지 않는 device를 생성 시점에 거부합니다.
- JSON으로 저장하고 다시 불러오는 메서드를 추가합니다.

검증: round-trip 후 객체가 같고, 잘못된 값에는 의미 있는 예외가 발생합니다.

### A2. 레지스트리 기반 컴포넌트 생성 ★★

TODO:

- 문자열 이름으로 tokenizer/model/retriever를 생성하는 registry를 작성합니다.
- 등록되지 않은 이름에는 가능한 선택지를 포함한 오류를 냅니다.
- 새 구현을 기존 분기문 수정 없이 등록할 방법을 고민합니다.

검증: fake 컴포넌트를 테스트에서 등록하고 실제 학습 없이 생성할 수 있습니다.

### A3. 리소스 수명과 로깅 ★★

TODO:

- 실험 시작/종료 시간, 설정, metric을 JSON Lines로 남기는 context manager를 구현합니다.
- 예외로 종료되어도 상태가 `failed`로 기록되게 합니다.
- 비밀번호·토큰·개인정보 필드는 기록 전에 마스킹합니다.

검증: 성공/실패 두 경우 모두 로그가 유효한 JSON이며 종료 상태가 남습니다.

## B. 데이터와 머신러닝

### B1. 데이터 계약 검사 ★

TODO:

- `data/customer_support_tickets.csv`의 필수 열, 타입, 빈 문자열, 허용 레이블을 검사합니다.
- 중복 id와 거의 같은 본문을 별도로 보고합니다.
- 오류를 모두 모아 행 번호와 함께 보여 줍니다.

검증: 일부러 손상한 fixture에서 각 오류가 정확한 행과 함께 검출됩니다.

### B2. 누수 없는 split ★★

TODO:

- 클래스 비율을 보존하는 train/validation/test split을 만듭니다.
- 같은 고객 또는 유사 문장이 서로 다른 split에 들어가지 않도록 group 기준을 추가합니다.
- seed와 split 통계를 파일에 저장합니다.

검증: 세 split의 id/group 교집합이 비어 있고 같은 seed에서 결과가 같습니다.

### B3. 기준선 실험표 ★★

TODO:

- majority class, word TF-IDF, char TF-IDF 기준선을 비교합니다.
- accuracy, macro/weighted F1, 클래스별 recall, 추론 시간을 기록합니다.
- 가장 자신 있게 틀린 예시 5개를 출력합니다.

검증: 하나의 표에서 품질과 비용을 함께 비교하고 채택한 기준선의 이유를 설명합니다.

### B4. threshold와 calibration ★★★

TODO:

- 최대 확률이 threshold보다 낮으면 `unknown`으로 보류합니다.
- threshold별 coverage와 accepted accuracy를 계산합니다.
- 가능하다면 calibration curve 또는 expected calibration error를 추가합니다.

검증: “더 많이 답하기”와 “답한 것의 정확도” 사이 trade-off가 수치로 드러납니다.

## C. PyTorch 핵심

### C1. Dataset/DataLoader 추적 ★

TODO:

- 한 sample과 collated batch의 key, shape, dtype을 출력하는 debug 함수를 만듭니다.
- 가변 길이 입력을 padding하고 attention mask를 함께 반환합니다.
- 길이 0 입력을 명확히 처리합니다.

검증: batch size와 최대 길이를 바꿔도 shape invariant 테스트가 통과합니다.

### C2. 한 배치 과적합 ★

TODO:

- 고정된 작은 batch를 반복 학습해 loss를 충분히 낮춥니다.
- 파라미터 하나의 값과 gradient norm 변화를 기록합니다.
- `zero_grad()`를 제거한 실험과 비교합니다.

검증: 정상 루프는 한 batch를 외우며, 잘못된 루프의 차이를 그래프나 표로 설명합니다.

### C3. 재사용 가능한 Trainer ★★

TODO:

- train/evaluate 단계를 분리하고 callback 또는 hook을 지원합니다.
- early stopping, gradient clipping, best checkpoint 저장을 추가합니다.
- metric 계산 시 logits와 targets를 CPU로 안전하게 모읍니다.

검증: tiny fake dataset에서 2~3초 내 smoke test가 끝나고 best 모델이 복구됩니다.

### C4. 재현 가능한 checkpoint ★★

TODO:

- model, optimizer, scheduler, epoch, config, random state를 저장합니다.
- 중단한 epoch 다음부터 재개합니다.
- `map_location="cpu"`로 GPU 없이도 로드할 수 있게 합니다.

검증: 연속 4 epoch 학습과 2 epoch + 재개 2 epoch 결과가 허용 오차 내에서 같습니다.

### C5. 혼합정밀도와 성능 측정 ★★★

TODO:

- CUDA가 있을 때만 autocast/GradScaler를 활성화하고 CPU에서는 안전하게 비활성화합니다.
- dataloader, forward, backward 시간을 따로 측정합니다.
- batch size별 samples/sec와 메모리를 비교합니다.

검증: 장치별 기능 분기가 테스트되고 병목이 숫자로 식별됩니다.

## D. 작은 언어 모델

### D1. 토크나이저 round-trip ★

TODO:

- special token을 포함한 문자 또는 단어 단위 tokenizer를 구현합니다.
- `encode → decode`가 가능한 입력과 불가능한 입력을 정의합니다.
- unknown token과 저장/불러오기를 지원합니다.

검증: 한글, 영문, 숫자, 줄바꿈, 미등록 문자의 round-trip 테스트가 있습니다.

### D2. shifted target 만들기 ★

TODO:

- causal LM의 input/target을 한 칸 어긋나게 생성합니다.
- padding 위치는 ignore index로 바꿉니다.
- 사람이 읽을 수 있는 표로 token과 target의 대응을 출력합니다.

검증: 첫 1개 sequence의 각 위치가 “다음 토큰”을 가리키는지 눈으로 확인할 수 있습니다.

### D3. causal mask 검증 ★★

TODO:

- 길이 `T`의 causal mask를 만들고 장치/dtype을 입력과 맞춥니다.
- attention weight 일부를 저장하거나 hook으로 관찰합니다.
- 미래 토큰을 바꿔도 이전 위치의 logit이 변하지 않는 테스트를 작성합니다.

검증: eval 모드에서 미래 정보 누수가 수치 허용 오차 내 0입니다.

### D4. 생성 전략 비교 ★★

TODO:

- greedy, temperature, top-k sampling을 공통 인터페이스로 구현합니다.
- 빈 후보, 0 이하 temperature, context 초과를 처리합니다.
- prompt/seed별 반복률, 고유 token 비율, 생성 시간을 기록합니다.

검증: 고정 seed에서 sampling 결과가 재현되고 설정별 특성이 비교표에 나타납니다.

### D5. KV cache 검증과 계측 ★★★

TODO:

- 기존 `KeyValueCache`가 layer별 이전 key/value를 어떻게 재사용하는지 shape를 기록합니다.
- cache 사용/미사용 logits가 같은지 확인합니다.
- sequence 길이에 따른 생성 지연 시간을 비교합니다.
- cache가 차지하는 byte 수를 dtype, layer 수, context 길이별로 계산합니다.

검증: 품질은 유지하면서 긴 생성의 반복 연산이 줄어듭니다.

## E. 검색과 RAG

### E1. 문서 chunker ★

TODO:

- 문단 우선 분할과 최대 길이/overlap을 지원합니다.
- 원문 id, chunk index, 문자 범위를 metadata로 보존합니다.
- 빈 문서와 최대 길이보다 긴 단일 문장을 처리합니다.

검증: chunk를 순서대로 합치면 overlap을 제외하고 원문의 정보를 잃지 않습니다.

### E2. 검색 평가 세트 ★★

TODO:

- 질문, 기대 문서 id, 답을 알 수 없는 질문으로 구성된 평가 JSONL을 만듭니다.
- hit@1, hit@3, MRR을 구현합니다.
- 실패 질의를 query mismatch, chunking, vocabulary 문제로 분류합니다.

검증: 최소 10개 질문에 대한 지표와 실패 목록이 한 번의 명령으로 생성됩니다.

### E3. 근거 기반 답변과 보류 ★★

TODO:

- 검색 문서의 id와 인용 구간을 답변에 연결합니다.
- top score가 threshold보다 낮으면 답변을 보류합니다.
- 답변에 없는 사실을 추가했는지 확인할 간단한 휴리스틱을 작성합니다.

검증: 답이 없는 질문에서 근거를 꾸며내지 않고 명시적으로 보류합니다.

### E4. hybrid retrieval ★★★

TODO:

- lexical 점수와 dense 점수를 정규화해 합칩니다.
- 가중치 또는 reciprocal rank fusion을 설정으로 선택합니다.
- 약어/고유명사 질의와 의미 유사 질의를 나눠 비교합니다.

검증: 단일 retriever 대비 좋아진/나빠진 질문이 모두 보고됩니다.

## F. 통합과 운영

### F1. 대화형 디버그 패널 ★★

TODO:

- 입력, 전처리 결과, 모델 출력, 검색 결과, 총 지연 시간을 한 화면/세션에 보여 줍니다.
- seed, temperature, top-k, checkpoint를 바꿀 수 있게 합니다.
- 오류 stack trace는 개발 모드에서만 노출합니다.

검증: 사용자가 같은 입력과 설정을 다시 넣어 결과를 재현할 수 있습니다.

### F2. 모델 서비스 경계 ★★★

TODO:

- 요청/응답 schema와 입력 길이 제한을 정의합니다.
- 모델은 프로세스 시작 시 한 번만 로드합니다.
- 동시에 여러 요청이 들어올 때 mutable state가 섞이지 않게 합니다.

검증: 잘못된 요청은 4xx 성격의 오류로, 내부 실패는 안전한 메시지로 구분됩니다.

### F3. 최소 회귀 테스트 세트 ★★

TODO:

- 데이터 schema, tokenizer, model shape, checkpoint, retrieval metric을 테스트합니다.
- 무거운 학습 없이 작은 fixture와 1-step update만 사용합니다.
- 느린 통합 테스트에 marker를 붙입니다.

검증: CPU에서 빠른 테스트가 안정적으로 통과하며 핵심 계약이 깨지면 실패합니다.

### F4. 실험 리포트 ★★

TODO:

- baseline, PyTorch 모델, 검색 파이프라인의 설정·metric·latency를 한 표로 정리합니다.
- 성공 사례뿐 아니라 대표 실패와 윤리/개인정보 위험을 기록합니다.
- 다음 개선의 예상 효과, 비용, 검증 방법을 우선순위화합니다.

검증: 다른 사람이 저장소와 리포트만 보고 결과를 재현할 수 있습니다.

## 최종 미니 프로젝트 제안

고객 지원 문의 도우미를 만드세요. 입력 문의에 대해 먼저 의도를 분류하고, 지식 문서에서 관련 근거를 검색한 뒤, 근거가 충분할 때만 답변 초안을 만듭니다.

필수 조건:

- 기준선과 PyTorch 모델을 동일 split에서 비교
- 클래스 확률과 `unknown` 보류 정책
- top-k 문서 id/score/근거 표시
- 답을 알 수 없는 평가 질문 포함
- 모델/checkpoint/config 버전 표시
- CPU에서 전체 smoke test 가능
- 실패 예시 10건과 개선 우선순위 작성

리뷰 질문:

1. 성능 향상이 데이터 누수 때문이 아님을 어떻게 보였나요?
2. 분류 실패와 검색 실패 중 무엇이 최종 품질을 더 제한하나요?
3. threshold를 바꾸면 coverage와 위험이 어떻게 변하나요?
4. 실제 운영 데이터에서 drift를 무엇으로 감지할 건가요?
5. 더 큰 LLM이나 외부 임베딩 API로 교체해도 유지되는 인터페이스는 무엇인가요?
