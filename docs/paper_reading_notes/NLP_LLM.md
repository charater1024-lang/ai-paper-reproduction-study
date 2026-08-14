# NLP·LLM 핵심 논문 10편 독해 노트

> **AI 보조 작성 안내:** 이 한국어 요약은 공식 번역이 아닙니다. 내용과 인용의 최종 기준은
> 연결된 원문입니다. [작성 원칙과 한계](AI_ASSISTED_NOTICE.md) · [70편 목차](README.md)

이 문서는 word representation에서 검색 증강 생성까지의 흐름을 원문 위치와 실습으로 연결한다. 각 노트북은 로컬 합성 corpus를 사용해 핵심 계산을 CPU에서 검증하며, 원 논문의 대규모 성능 재현을 주장하지 않는다. 위치와 TODO는 [NLP/LLM 사양 파일](../../tools/field_curriculum/nlp_specs.py)과 일치한다.

## 00. word2vec SGNS — Distributed Representations of Words and Phrases and their Compositionality (2013)

**① 한 문장 요약.** 중심 단어가 실제 주변 단어에는 높은 점수를, noise 단어에는 낮은 점수를 주도록 학습해 의미·문법 관계를 담는 조밀한 word vector를 효율적으로 얻는다.

**② 왜 나왔나 / 이전 한계.** one-hot vector는 단어 간 유사성을 표현하지 못하고 vocabulary가 커질수록 full softmax 계산이 비싸다. 기존 neural language model도 모든 단어 출력을 정규화해야 했다. Skip-gram은 주변 단어 예측에 집중하고 negative sampling으로 작은 수의 이진 분류 문제로 바꿨다.

**③ 읽기 전 기초지식.** distributional hypothesis, center/context window, embedding lookup, sigmoid/log-sigmoid, negative sampling, unigram distribution.

**④ 핵심 아이디어.**

- sequence의 각 중심 단어에서 일정 window 안의 context 단어 쌍을 만든다.
- input embedding과 output embedding을 별도로 둔다.
- 실제 pair의 내적은 높이고 sampled noise pair의 내적은 낮춘다.
- 빈도 분포를 `3/4` 제곱한 noise distribution이 너무 잦은 단어의 지배를 완화한다.
- 자주 등장하는 단어 subsampling과 phrase token도 제안한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1310.4546)의 §2 Fig. 1과 Eq. (1)에서 Skip-gram objective를 본다. §2.2 Eq. (4)는 negative sampling, 같은 절의 `Pn(w)=U(w)^(3/4)/Z`는 noise 선택이다. §5는 vector addition의 compositionality를 다룬다.

**⑥ 수식·알고리즘 직관.** positive pair `(center, context)`의 내적 `s`에는 `log σ(s)`, negative pair에는 `log σ(-s)`를 최대화한다. 이는 “실제 문맥인가, noise인가”를 판별하는 logistic regression이다. 모든 vocabulary 점수를 계산하지 않아 한 update 비용이 `V` 대신 negative sample 수 `k`에 비례한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/00_word2vec_sgns.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/00_word2vec_sgns.ipynb). TODO 1은 window=1 pair와 unigram^0.75 distribution을 만든다. TODO 2는 Eq. (4)의 SGNS loss를 구현한다. TODO 3은 실제 pair와 noise의 평균 dot score, loss를 그린다. 작은 반복 corpus이므로 analogy 품질은 평가하지 않는다.

**⑧ 5분 요약 / 한계 / 자가점검.** 요약은 “문맥 예측을 positive-vs-noise 판별로 싸게 학습”이다. 한계는 단어마다 고정 vector 하나라 문맥별 의미가 변하지 않고, 순서/긴 의존성을 직접 모델링하지 않는다는 점이다.

1. input embedding과 output embedding을 분리하는 이유는?
2. negative sample 수가 너무 작거나 너무 크면 어떤 trade-off가 생기는가?
3. `unigram^0.75`가 uniform과 raw unigram의 중간 성질을 갖는 이유는?

## 01. Sequence to Sequence Learning with Neural Networks (2014)

**① 한 문장 요약.** 한 LSTM이 가변 길이 입력을 고정 차원 state로 압축하고 다른 LSTM이 그 state를 조건으로 출력 sequence 확률을 순차적으로 분해한 범용 encoder-decoder다.

**② 왜 나왔나 / 이전 한계.** 일반 DNN은 입력·출력 차원이 고정되어 길이가 다른 번역 문장을 직접 대응시키기 어렵다. 고전 통계 번역은 여러 독립 구성요소와 feature engineering이 필요했다. seq2seq는 정렬 가정 없이 source 전체를 읽고 target을 생성하는 end-to-end likelihood 모델을 제시했다.

**③ 읽기 전 기초지식.** RNN/LSTM hidden state, encoder-decoder, autoregressive factorization, teacher forcing, begin/end token, token cross-entropy.

**④ 핵심 아이디어.**

- encoder final state를 source sentence 표현으로 사용한다.
- decoder는 이전 target token과 hidden state로 다음 token 확률을 낸다.
- 전체 target 확률을 token conditional probability의 곱으로 분해한다.
- source 단어 순서를 뒤집어 source 시작과 target 시작 사이의 최소 dependency 거리를 줄인다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1409.3215)의 §2 Eq. (1)에서 conditional sequence probability, Fig. 1에서 encoder/decoder 흐름을 본다. §3.3은 source reversal의 최적화 동기, §4는 translation evaluation 결과를 설명한다.

**⑥ 수식·알고리즘 직관.** `p(y|x)=∏t p(y_t | y_<t, v)`이며 `v`가 encoder의 고정 context다. 학습 때는 실제 이전 token을 decoder에 넣는 teacher forcing으로 각 조건부 분포를 동시에 계산한다. inference 때는 자기 예측을 다시 넣으므로 exposure bias가 생길 수 있다. source reversal은 모델 능력을 바꾸기보다 gradient가 지나야 할 시간 거리를 줄인다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/01_seq2seq.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/01_seq2seq.ipynb). TODO 1은 reverse source와 target의 거리를 비교한다. TODO 2는 GRU encoder final state로 decoder를 초기화한다. TODO 3은 60개의 3-token reverse pair에서 token CE와 accuracy를 확인한다. 원 논문의 4-layer LSTM, beam search, BLEU는 생략한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “입력을 state 하나로 압축하고 target 확률을 왼쪽부터 분해”하는 것이다. 한계는 긴 입력의 모든 정보를 고정 vector 하나에 담아야 하는 bottleneck과 순차 계산이다.

1. teacher forcing과 실제 생성 시 decoder 입력은 어떻게 다른가?
2. source reversal이 의미를 바꾸지 않으면서 학습을 도울 수 있는 이유는?
3. 고정 context vector가 긴 문장에서 병목이 되는 이유는?

## 02. Bahdanau Attention — Jointly Learning to Align and Translate (2014)

**① 한 문장 요약.** decoder가 매 출력 시점마다 source annotation 전부를 점수화하고 가중합 context를 만들어, 고정 vector 병목을 완화하면서 번역과 soft alignment를 함께 학습한다.

**② 왜 나왔나 / 이전 한계.** 초기 seq2seq는 source 전체를 final state 하나에 압축해 긴 문장에서 성능이 약해졌다. 전통 번역의 alignment는 별도 모델/단계였다. Bahdanau attention은 decoder state와 각 source 위치의 호환도를 학습해 필요한 source 부분을 동적으로 읽는다.

**③ 읽기 전 기초지식.** bidirectional RNN, encoder annotation, softmax attention weight, weighted sum context, additive alignment score, teacher forcing.

**④ 핵심 아이디어.**

- bidirectional encoder가 각 source 위치의 양방향 annotation을 만든다.
- 이전 decoder state와 각 annotation으로 alignment energy를 계산한다.
- source 위치에 softmax를 적용해 attention weight를 얻는다.
- weight를 사용한 annotation 가중합이 현재 decoder context가 된다.
- alignment에 별도 정답을 주지 않고 translation loss로 함께 학습한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1409.0473)의 §2.2.1 Eq. (5)는 `c_i=Σ_j α_ij h_j`, Eq. (6)은 alignment softmax, Eq. (7)은 `a(s_{i-1},h_j)` score다. §3.1 Eq. (8–9)는 bidirectional annotation을 정의한다.

**⑥ 수식·알고리즘 직관.** additive attention은 대략 `e_ij=vᵀ tanh(Ws_{i-1}+Uh_j)`다. softmax한 `α_ij`는 source 위치별 합이 1이어서 현재 target token이 어디를 얼마나 참고하는지 나타낸다. context는 hard하게 위치 하나를 고르는 대신 모든 annotation의 미분 가능한 기대값이다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/02_bahdanau_attention.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/02_bahdanau_attention.ipynb). TODO 1은 additive score, normalized weight와 context를 구현한다. TODO 2는 bidirectional encoder와 step별 attention decoder를 만든다. TODO 3은 reverse translation을 학습하고 3×3 alignment heatmap을 그린다.

**⑧ 5분 요약 / 한계 / 자가점검.** 요약은 “매 target step마다 source를 다시 검색하는 미분 가능한 정렬”이다. 한계는 RNN의 순차성, source와 target 길이에 비례하는 attention 계산, attention weight를 곧바로 인간적 설명으로 해석하기 어렵다는 점이다.

1. attention weight의 source 방향 합이 1이어야 하는 이유는?
2. context가 fixed vector 방식과 달리 target step마다 바뀌는 이유는?
3. alignment label 없이도 attention이 학습되는 신호는 어디서 오는가?

## 03. Transformer — Attention Is All You Need (2017)

**① 한 문장 요약.** recurrence와 convolution을 제거하고 multi-head attention, feed-forward network와 위치 정보를 쌓아 sequence를 병렬 처리하는 encoder-decoder 구조다.

**② 왜 나왔나 / 이전 한계.** RNN은 token을 순서대로 처리해 병렬화가 어렵고 먼 위치 사이 정보가 여러 state를 거쳐야 했다. attention을 보조로 써도 encoder/decoder 본체는 recurrent했다. Transformer는 모든 token pair를 직접 연결하는 self-attention을 중심 연산으로 삼았다.

**③ 읽기 전 기초지식.** query/key/value, scaled dot-product attention, causal mask, multi-head, residual+layer normalization, sinusoidal position encoding.

**④ 핵심 아이디어.**

- `QKᵀ/√d_k`로 token 간 호환도를 계산하고 value를 가중합한다.
- 여러 head가 서로 다른 projection 공간의 관계를 병렬로 본다.
- encoder self-attention, decoder masked self-attention, encoder-decoder attention을 구분한다.
- 위치 encoding이 순서 정보를 제공한다.
- 각 attention/FFN sublayer를 residual과 layer normalization으로 감싼다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1706.03762)의 §3.2.1 Eq. (1)에서 scaled dot-product attention, §3.2.2에서 multi-head를 본다. §3.4는 embedding/softmax, §3.5 Eq. (3)은 sinusoidal position encoding이다. Fig. 1은 세 attention 경로를 구분하는 데 필수다.

**⑥ 수식·알고리즘 직관.** dot product 분산은 차원이 커질수록 커져 softmax가 포화될 수 있으므로 `√d_k`로 나눈다. causal mask는 미래 위치 score를 `-∞`로 바꿔 softmax 확률을 0으로 만든다. 각 token이 다른 모든 token까지 한 attention step에 도달하지만, 길이 `n`에서 score matrix가 `n²` 크기다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/03_attention_is_all_you_need.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/03_attention_is_all_you_need.ipynb). TODO 1은 Eq. (1)과 sinusoidal encoding을 구현한다. TODO 2는 4-head, 1-layer encoder-decoder와 causal target mask를 만든다. TODO 3은 reverse translation accuracy/loss를 확인한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “모든 token 간 직접 attention + 위치 정보 + 병렬 학습”이다. 한계는 긴 sequence의 quadratic memory/compute, position extrapolation, 많은 데이터/계산 요구다.

1. `√d_k` scaling이 없으면 큰 key 차원에서 softmax는 어떻게 변할 수 있는가?
2. encoder self-attention과 decoder self-attention의 mask 차이는?
3. position encoding 없이 self-attention이 단어 순서를 구분할 수 있는가?

## 04. GPT-1 — Improving Language Understanding by Generative Pre-Training (2018)

**① 한 문장 요약.** 큰 unlabeled corpus에서 causal language modeling으로 Transformer decoder를 사전학습한 뒤, 작은 supervised dataset의 task head와 보조 LM loss로 전체 모델을 미세조정한다.

**② 왜 나왔나 / 이전 한계.** supervised NLP task마다 많은 label과 task-specific architecture가 필요했다. word embedding은 transfer되지만 문장 수준의 문맥 표현은 제한적이었다. GPT-1은 생성 사전학습으로 범용 문맥 표현을 얻고 최소한의 입력 변환/head만 추가하는 통합 transfer 방식을 제시했다.

**③ 읽기 전 기초지식.** decoder-only Transformer, causal language modeling, next-token cross-entropy, pretraining/fine-tuning, auxiliary loss, task-aware input transformation.

**④ 핵심 아이디어.**

- unlabeled token sequence의 autoregressive likelihood를 최대화한다.
- masked multi-head self-attention을 쌓은 decoder를 사용한다.
- downstream input을 delimiter 등으로 하나의 token sequence로 바꾼다.
- supervised objective에 LM objective를 보조항으로 더해 fine-tuning한다.

**⑤ 꼭 볼 원문 위치.** [OpenAI 원문 PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)의 §2 Eq. (1)에서 LM objective, Eq. (2)에서 decoder block을 본다. Eq. (3)은 supervised objective, Eq. (4)와 Fig. 1은 joint objective와 task input transformations다.

**⑥ 수식·알고리즘 직관.** causal LM은 `p(u_i|u_{i-k},…,u_{i-1})`의 log probability 합을 최대화한다. downstream에서는 마지막 token representation 등을 classification head에 넣는다. `L=L_supervised+λL_LM`은 task label을 맞히는 동안 사전학습된 language structure가 급격히 무너지는 것을 완화하는 보조 신호로 볼 수 있다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/04_gpt1.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/04_gpt1.ipynb). TODO 1은 strict causal mask를 검증한다. TODO 2는 동일 decoder에 LM/classification head를 단다. TODO 3은 next-token pretraining 후 supervised+auxiliary LM fine-tuning loss를 이어 그린다. 실제 BooksCorpus 규모와 task별 입력 변환은 축소했다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “생성 사전학습 하나를 여러 이해 task에 전이”하는 것이다. 한계는 단방향 문맥, task마다 parameter 전체를 복제하는 fine-tuning 비용, pretraining data의 편향이다.

1. causal mask가 미래 token을 가려야 하는 이유는?
2. fine-tuning 중 LM 보조 loss가 할 수 있는 역할은?
3. GPT-1의 transfer와 고정 word2vec 사용은 무엇이 다른가?

## 05. BERT — Bidirectional Encoder Representations from Transformers (2018)

**① 한 문장 요약.** 일부 입력 token을 가리고 양쪽 문맥으로 복원하는 masked language model과 sentence-pair objective로 Transformer encoder를 사전학습해 다양한 이해 task에 미세조정한다.

**② 왜 나왔나 / 이전 한계.** GPT식 causal LM은 각 위치에서 왼쪽 문맥만 보므로 token-level 이해에서 오른쪽 정보가 사전학습 표현에 직접 반영되지 않는다. ELMo는 양방향 RNN을 결합했지만 깊은 층 전체에서 좌우를 동시에 조건화하지 않았다. BERT는 masking으로 정답 누출을 막으면서 양방향 encoder를 학습했다.

**③ 읽기 전 기초지식.** Transformer encoder, bidirectional self-attention, `[CLS]`/`[SEP]`, token·position·segment embedding, masked LM(MLM), next sentence prediction(NSP).

**④ 핵심 아이디어.**

- token/position/segment embedding을 더해 sentence pair를 표현한다.
- 선택 token의 80%는 `[MASK]`, 10%는 random, 10%는 그대로 둔다.
- loss는 선택 위치의 원래 token에만 적용한다.
- `[CLS]` representation으로 NSP와 downstream classification을 수행한다.
- pretrained encoder 전체를 작은 task head와 함께 fine-tune한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1810.04805)의 §3과 Fig. 2에서 input representation을 본다. §3.1 Task #1은 MLM과 80/10/10 corruption, Task #2는 NSP다. Fig. 1은 BERT의 깊은 양방향 조건화와 GPT/ELMo를 비교한다.

**⑥ 수식·알고리즘 직관.** 일반 bidirectional encoder가 원 token을 그대로 보면 복사만 할 수 있으므로 입력을 손상시키고 원래 token을 label로 둔다. `[MASK]`만 쓰지 않는 이유는 fine-tuning 때 `[MASK]`가 나타나지 않는 불일치를 줄이기 위해서다. 선택되지 않은 위치에는 MLM loss를 주지 않는다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/05_bert.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/05_bert.ipynb). TODO 1은 10개 token에 8/1/1 corruption과 `-100` ignore label을 만든다. TODO 2는 causal mask 없는 token+position encoder와 MLM head를 구현한다. TODO 3은 선택 token accuracy와 loss를 그린다. segment embedding과 NSP는 데이터 규모상 설명에만 남기고 실습에서는 생략했다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “입력을 일부 가려 양방향 문맥으로 복원”이다. 한계는 pretrain/fine-tune 입력 불일치, 선택 token만 예측하는 계산 비효율, 자유 생성에 바로 쓰기 어려운 encoder 구조와 NSP 효용 논쟁이다.

1. MLM이 모든 입력 token에 loss를 주지 않는 이유는?
2. 선택 token 중 일부를 unchanged로 두면서도 label loss를 주는 이유는?
3. BERT와 GPT의 self-attention mask는 어떻게 다른가?

## 06. T5 — Text-to-Text Transfer Transformer (2019)

**① 한 문장 요약.** 분류·번역·요약을 모두 “text 입력→text 출력”으로 통일하고, span corruption으로 encoder-decoder를 사전학습해 transfer learning 요소를 체계적으로 비교한 연구다.

**② 왜 나왔나 / 이전 한계.** BERT, GPT 등은 architecture와 objective가 달랐고 downstream마다 별도 head/출력 형식이 필요했다. 데이터·모델·학습 목표가 함께 바뀌어 어떤 선택이 중요한지 비교하기도 어려웠다. T5는 공통 framework에서 objective, corpus, architecture, scale을 비교한다.

**③ 읽기 전 기초지식.** encoder-decoder Transformer, task prefix, text-to-text formulation, denoising objective, span corruption, sentinel token, relative position bias.

**④ 핵심 아이디어.**

- 모든 task의 정답을 token sequence로 생성한다.
- 연속된 여러 token span을 고유 sentinel token 하나로 치환한다.
- target에는 sentinel과 삭제된 span들을 순서대로 둔다.
- 절대 위치 vector 대신 attention logit에 learned relative-position bias를 더한다.
- C4 corpus와 여러 transfer 설정을 통제된 조건에서 비교한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/1910.10683)의 §2와 Fig. 1에서 text-to-text framework를, §2.1에서 encoder-decoder/relative position embedding을 본다. §3.1.4와 Fig. 2는 corruption variants, §3.3.4는 span-corruption 결과를 다룬다.

**⑥ 수식·알고리즘 직관.** 원문 `[a,b,c]`에서 `[b,c]`를 가리면 encoder input은 `[a,<X>]`, decoder target은 `[<X>,b,c,EOS]`가 된다. 개별 token masking보다 긴 span의 관계를 복원하게 한다. relative bias는 query-key 위치 차이를 bucket으로 묶어 attention score에 더하며 길이 변화에 더 자연스럽게 대응한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/06_t5.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/06_t5.ipynb). TODO 1은 3-token sequence의 연속 span을 sentinel input/target으로 바꾼다. TODO 2는 clipped relative distance와 encoder-decoder를 구현한다. TODO 3은 denoising token accuracy/loss를 그린다. 실습의 relative distance는 개념 확인용이며 full learned bucket bias로 attention에 주입하지 않는다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “모든 task를 생성 형식으로 통일 + span denoising”이다. 한계는 출력 vocabulary 전체 softmax 비용, 분류도 autoregressive decoding해야 하는 지연, 거대한 corpus/compute 의존성이다.

1. span corruption에서 sentinel token은 어떤 경계를 알려 주는가?
2. text-to-text 통일이 task-specific head보다 주는 이점은?
3. relative position bias와 absolute position embedding의 차이는?

## 07. GPT-3 — Language Models are Few-Shot Learners (2020)

**① 한 문장 요약.** 매우 큰 causal language model은 gradient update 없이 prompt 안의 지시와 몇 개 예시만으로 새로운 task 형식을 따라갈 수 있음을 대규모로 평가했다.

**② 왜 나왔나 / 이전 한계.** pretrain-fine-tune은 task마다 labeled data와 별도 weight 사본이 필요하고, 좁은 training distribution에 과적합할 수 있다. 인간은 몇 개 예시나 자연어 설명만으로 새 task를 이해한다. GPT-3는 parameter scale이 이러한 in-context learning을 얼마나 강화하는지 zero/one/few-shot protocol에서 조사했다.

**③ 읽기 전 기초지식.** causal LM, prompt/context window, zero-shot·one-shot·few-shot, in-context learning, demonstration, no-gradient evaluation, contamination.

**④ 핵심 아이디어.**

- task 설명과 labeled demonstration을 하나의 token context로 직렬화한다.
- query 뒤 정답 continuation의 확률로 prediction한다.
- task마다 weight를 업데이트하지 않는다.
- model scale에 따라 in-context 성능이 어떻게 바뀌는지 폭넓게 측정한다.
- 데이터 contamination, bias와 사회적 영향도 별도로 논의한다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/2005.14165)의 §1 Fig. 1.1에서 zero/one/few-shot 차이를 본다. §2.1 Eq. (2.1)은 autoregressive objective, §2.2는 evaluation setting, §3은 task별 shot 수 결과다. 표의 최고 수치만 보지 말고 setting과 baseline을 함께 읽는다.

**⑥ 수식·알고리즘 직관.** fine-tuning은 dataset gradient로 weight를 바꾸지만 in-context learning은 `p(answer | instruction, examples, query)`의 조건 context만 바꾼다. 따라서 prompt 순서·표현·example 선택이 입력 분포가 된다. 능력을 얻는 정확한 내부 학습 메커니즘을 단순 nearest-neighbor로 동일시하면 안 된다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/07_gpt3_few_shot.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/07_gpt3_few_shot.ipynb). TODO 1은 demonstration과 unlabeled query prompt를 직렬화한다. TODO 2는 parameter가 없는 token-overlap predictor로 “weight update 없음” protocol을 격리한다. TODO 3은 shot 수에 따른 정확도를 그린다. 이 알고리즘은 GPT-3를 재현하지 않고 평가 형식만 재현한다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “예시를 학습 데이터가 아니라 context로 제공”하는 것이다. 한계는 huge compute, prompt 민감성, contamination, 환각·편향, few-shot이 전용 fine-tuning보다 항상 낫지는 않다는 점이다.

1. few-shot prompting과 few-shot gradient fine-tuning의 차이는?
2. demonstration 순서가 결과에 영향을 줄 수 있는 이유는?
3. 실습의 nearest-example predictor 결과를 GPT-3 능력으로 해석하면 안 되는 이유는?

## 08. LoRA — Low-Rank Adaptation of Large Language Models (2021)

**① 한 문장 요약.** pretrained weight는 고정하고 그 변화량을 낮은 rank 행렬 두 개의 곱으로 제한해, 훨씬 적은 trainable parameter와 메모리로 task adaptation을 수행한다.

**② 왜 나왔나 / 이전 한계.** full fine-tuning은 task마다 전체 model weight와 optimizer state를 저장해야 해 LLM이 커질수록 비용이 막대하다. adapter는 효율적이지만 inference 경로에 추가 layer/latency를 만들 수 있다. LoRA는 기존 linear projection에 병렬 low-rank update만 더하고 나중에 weight에 병합할 수 있다.

**③ 읽기 전 기초지식.** matrix rank, low-rank factorization, frozen parameter, linear projection, parameter-efficient fine-tuning(PEFT), optimizer state memory.

**④ 핵심 아이디어.**

- 원 weight `W0`의 gradient를 끈다.
- update를 `ΔW=BA`로 parameterize하며 rank `r`을 작게 둔다.
- forward는 `h=W0x+(α/r)BAx`다.
- 보통 한 factor를 0으로 초기화해 시작 출력이 base model과 같게 한다.
- attention projection 등 선택한 linear layer에만 적용하고 inference 때 병합할 수 있다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/2106.09685)의 §4.1 Eq. (3)에서 `W0+BA`, frozen/trainable 구분을 본다. §4.2는 Transformer에 적용하는 위치, §7.1은 adaptation update의 낮은 intrinsic rank를 분석한다.

**⑥ 수식·알고리즘 직관.** `W0∈R^{d×k}` 전체를 바꾸면 `dk`개가 필요하지만 `B∈R^{d×r}`, `A∈R^{r×k}`면 `r(d+k)`개만 학습한다. `r≪min(d,k)`일 때 절약이 크다. 이는 실제 최적 update가 반드시 정확히 low rank라는 보장이 아니라 유용한 제약/근사 가정이다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/08_lora.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/08_lora.ipynb). TODO 1은 Eq. (3)의 `LoRALinear`와 zero-B 초기 동일성을 구현한다. TODO 2는 bag-of-words classifier의 frozen projection에 rank-2 update를 학습한다. TODO 3은 trainable/full parameter와 loss/accuracy를 비교한다. 실제 Transformer attention에 삽입하지 않은 축소 실습이다.

**⑧ 5분 요약 / 한계 / 자가점검.** 요약은 “전체 weight가 아니라 low-rank 변화량만 저장”이다. 한계는 rank/적용 layer 선택, task가 큰 full-rank 변화를 요구할 때의 표현 제한, 여러 adapter 운영 복잡성이다.

1. B를 0으로 초기화하면 첫 forward가 base model과 같은 이유는?
2. `r(d+k)<dk`가 되려면 rank는 어느 정도여야 하는가?
3. frozen weight에 gradient가 없다는 것과 activation memory가 0이라는 것은 같은가?

## 09. RAG — Retrieval-Augmented Generation (2020)

**① 한 문장 요약.** query로 외부 문서를 확률적으로 검색하고, retrieved passage를 조건으로 생성한 답의 확률을 문서들에 대해 marginalize해 parametric LM과 updatable knowledge store를 결합한다.

**② 왜 나왔나 / 이전 한계.** 생성 모델의 지식은 weight에 암묵적으로 저장되어 출처 확인·갱신이 어렵고, 드문 사실에서 환각할 수 있다. extractive QA는 근거를 찾지만 자유로운 응답 생성이 제한된다. RAG는 dense retriever의 non-parametric memory와 seq2seq generator를 end-to-end로 연결한다.

**③ 읽기 전 기초지식.** dense passage retrieval, query/document embedding, maximum inner product search, latent document, marginalization, seq2seq likelihood, recall@k.

**④ 핵심 아이디어.**

- query encoder와 문서 embedding 내적으로 `pη(z|x)`를 만든다.
- top-k 문서 각각을 query와 generator 입력에 넣는다.
- RAG-Sequence는 답 전체에서 하나의 latent document를 공유한다.
- RAG-Token은 token마다 문서를 다시 marginalize한다.
- 문서 index를 바꾸면 generator를 다시 사전학습하지 않고도 지식 갱신이 가능하다.

**⑤ 꼭 볼 원문 위치.** [원문](https://arxiv.org/abs/2005.11401)의 §2.1 Eq. (1)에서 retriever distribution을 본다. §2.2 Eq. (2)는 RAG-Sequence, Eq. (3)은 RAG-Token이고 Fig. 1은 retriever와 generator의 결합을 보여 준다.

**⑥ 수식·알고리즘 직관.** RAG-Sequence의 답 확률은 `Σ_z pη(z|x) pθ(y|x,z)`다. 어떤 문서가 정답 근거인지 latent variable로 두고 가능한 문서별 생성 확률을 검색 확률로 가중 평균한다. 검색이 실패하면 generator가 올바른 근거를 볼 수 없으므로 generation metric만이 아니라 retrieval recall도 따로 측정해야 한다.

**⑦ 우리 노트북 실습 연결.** [실습본](../../notebooks/field_reproductions/nlp_llm/exercises/09_rag.ipynb) / [정답본](../../notebooks/field_reproductions/nlp_llm/solutions/09_rag.ipynb). TODO 1은 로컬 5문서에서 TF-IDF cosine과 softmax retrieval distribution을 만든다. TODO 2는 toy generator likelihood를 Eq. (2)처럼 marginalize한다. TODO 3은 recall@1과 `[document id]`가 붙은 답, similarity heatmap을 만든다. dense neural retriever/BART 학습은 재현하지 않는다.

**⑧ 5분 요약 / 한계 / 자가점검.** 핵심은 “먼저 근거를 찾고, 문서를 latent variable로 생성 확률에 결합”하는 것이다. 한계는 retrieval miss, 잘못된 문서에 대한 과신, index/latency 비용, 인용이 사실성을 자동 보장하지 않는다는 점이다.

1. retriever recall@1이 0이면 generator가 맞힐 가능성이 반드시 0인가?
2. RAG-Sequence와 RAG-Token에서 latent document가 공유되는 범위는?
3. 검색 점수 softmax와 generator likelihood를 곱하는 의미는?

## 2023–2026 후속 학습 후보 5편

선정 기준은 (1) 2023년 이후 정식 conference/proceedings 기록이 확인될 것, (2) 위 과정의 alignment·PEFT·RAG·대체 sequence architecture·long context를 한 단계씩 확장할 것, (3) 주제가 중복되지 않을 것, (4) 공식 프로시딩/OpenReview/PMLR에서 venue와 연도를 확인할 수 있을 것이다. 2026년 제출작은 아직 정식 채택 여부와 후속 검증이 불안정할 수 있어 포함하지 않았고, 2023–2025의 확정 논문만 제시한다.

### A. Direct Preference Optimization: Your Language Model is Secretly a Reward Model — NeurIPS 2023

- **왜 주목할까:** 별도 reward model 학습과 online RL 없이 chosen/rejected preference pair에 대한 단일 classification형 objective로 policy를 reference model에 상대적으로 정렬한다. GPT fine-tuning 다음의 preference alignment를 이해하기 좋은 출발점이다.
- **선수지식:** RLHF, Bradley–Terry preference model, KL-regularized reward maximization, policy/reference log probability, binary cross-entropy.
- **공식 근거/원문:** NeurIPS 공식 프로시딩이 Main Conference Track, NeurIPS 2023과 reward model/RL 단계를 제거한 접근을 명시한다. [NeurIPS 2023 공식 페이지](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html).

### B. QLoRA: Efficient Finetuning of Quantized LLMs — NeurIPS 2023

- **왜 주목할까:** frozen 4-bit base model을 통해 gradient를 전달하면서 LoRA만 학습하고, NF4·double quantization·paged optimizer로 fine-tuning memory를 줄인다. LoRA 실습 직후 읽으면 “trainable parameter”와 “base-weight storage” 절약을 구분할 수 있다.
- **선수지식:** LoRA, affine/block-wise quantization, 4-bit NormalFloat, quantization scale, optimizer memory와 paging.
- **공식 근거/원문:** NeurIPS 공식 페이지는 2023 Main Conference Track과 4-bit frozen model+LoRA, NF4, double quantization, paged optimizer를 명시한다. [NeurIPS 2023 공식 페이지](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html).

### C. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection — ICLR 2024

- **왜 주목할까:** 모든 query에서 고정 개수 passage를 무조건 가져오는 RAG 대신, reflection token으로 검색 필요성·근거 관련성·응답 품질을 모델이 평가하고 inference behavior를 조절한다.
- **선수지식:** RAG-Sequence/RAG-Token, instruction tuning, special control token, passage relevance, factuality와 citation evaluation.
- **공식 근거/원문:** OpenReview의 accepted PDF 상단에 “Published as a conference paper at ICLR 2024”가 명시되고, 공식 record는 ICLR 2024 poster로 분류한다. [ICLR 2024 공식 OpenReview 원문](https://openreview.net/pdf?id=hSyW5go0v8).

### D. Mamba: Linear-Time Sequence Modeling with Selective State Spaces — COLM 2024

- **왜 주목할까:** 입력에 따라 state-space parameter를 선택적으로 바꾸어 중요한 정보를 전파/망각하고, attention의 quadratic sequence 비용과 다른 linear-time backbone을 제시한다. Transformer의 대안을 구조적으로 공부하기 좋다.
- **선수지식:** recurrent/state-space model, discretization, convolutional SSM, input-dependent gating/selection, parallel scan, attention complexity.
- **공식 근거/원문:** OpenReview 공식 PDF는 “Published as a conference paper at COLM 2024”를 명시하고 COLM record에 논문이 등재되어 있다. [COLM 2024 공식 OpenReview 원문](https://openreview.net/pdf?id=tEYskw1VY2).

### E. LongRoPE2: Near-Lossless LLM Context Window Scaling — ICML 2025

- **왜 주목할까:** RoPE 차원별 학습 부족과 out-of-distribution 위치 문제를 다루고, search 기반 rescaling과 mixed-context training으로 긴 문맥 확장과 원래 짧은 문맥 성능 보존을 함께 겨냥한다.
- **선수지식:** rotary position embedding(RoPE), positional interpolation, perplexity, needle retrieval evaluation, context-window fine-tuning, evolutionary search.
- **공식 근거/원문:** PMLR 공식 페이지가 Proceedings of the 42nd ICML, volume 267, 2025 및 논문의 세 기여를 명시한다. [ICML 2025 PMLR 공식 페이지](https://proceedings.mlr.press/v267/shang25a.html).
