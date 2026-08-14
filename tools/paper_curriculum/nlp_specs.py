"""Paper specifications for the NLP foundations track (papers 06--11).

Every solution is deliberately a mechanism-level mini reproduction.  Tensor
workloads use the shared accelerator contract with a CPU fallback; tiny Python
protocols stay on CPU.  The synthetic corpora require no network access and are
not substitutes for the original papers' experiments.
"""

from __future__ import annotations

from .common import (
    PaperSpec,
    code,
    markdown,
    shared_code,
)

SPECS: tuple[PaperSpec, ...] = (
    PaperSpec(
        number=6,
        slug="word2vec_negative_sampling",
        short_title="word2vec: Skip-gram과 Negative Sampling",
        paper_title="Distributed Representations of Words and Phrases and their Compositionality",
        authors="Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean",
        year=2013,
        primary_url="https://arxiv.org/abs/1310.4546",
        venue="NeurIPS 2013",
        difficulty="중급",
        expected_minutes=65,
        prerequisites="PyTorch 텐서, 임베딩, 이진 교차 엔트로피, 코사인 유사도",
        reproduction_goal=(
            "작은 문장 모음에서 Skip-gram 중심어-문맥 쌍을 만들고, unigram 분포의 "
            "3/4승으로 음성 표본을 뽑아 식 (4)의 SGNS 목적함수를 최적화한다. "
            "손실 감소와 학습된 단어 공간의 코사인 구조를 확인한다."
        ),
        original_scale=(
            "논문은 최대 약 10억 단어 규모 말뭉치와 큰 어휘, 다수의 음성 표본으로 "
            "단어·구 벡터 및 analogy 정확도를 평가했다. 여기서는 8개 미니 문장과 "
            "수십 개 파라미터만 사용하므로 원 논문의 품질 수치는 재현하지 않는다."
        ),
        mappings=(
            (
                "§2, 식 (1): Skip-gram이 주변 단어의 로그확률을 최대화",
                "`make_skipgram_pairs`가 윈도 안의 (중심어, 문맥어) 양성 쌍을 생성",
                "쌍 개수와 자기 자신이 문맥에 들어가지 않는다는 assert",
            ),
            (
                "§2.2, 식 (4): Negative Sampling 목적함수",
                "`sgns_loss`가 positive `log σ`와 negative `log σ(-·)`를 계산",
                "스칼라·유한 손실 및 모든 임베딩에 대한 gradient assert",
            ),
            (
                "§2.2: noise distribution에 unigram의 3/4승 사용",
                "`noise_probs = count**0.75 / Z`의 noise distribution",
                "확률 합 1과 고빈도 단어 확률의 완화 여부를 검사",
            ),
            (
                "§4: 학습된 단어·구 벡터의 의미적 규칙성 평가",
                "학습 embedding의 코사인 유사도 행렬과 2차원 SVD 투영",
                "손실 감소율, 유사도 범위, 시각화로 미니 재현을 진단",
            ),
        ),
        cells=(
            markdown(
                r"""
                ## 핵심 아이디어

                Skip-gram은 중심 단어 $w_I$로 주변 단어 $w_O$를 예측한다. 전체 어휘
                softmax 대신 Negative Sampling은 실제 쌍에는
                $\log\sigma(v'_{w_O}{}^\top v_{w_I})$, 잡음 단어 $w_i$에는
                $\log\sigma(-v'_{w_i}{}^\top v_{w_I})$를 준다(논문 식 (4)).

                아래 실습에서는 논문이 보고한 `unigram_frequency ** 0.75` 잡음 분포도
                그대로 구현한다. 작은 데이터이므로 최종 analogy 점수보다 목적함수와
                임베딩 기하가 올바르게 연결되는지가 핵심이다.
                """,
                "paper-theory",
            ),
            shared_code(
                """
                import math
                from collections import Counter

                import matplotlib.pyplot as plt
                import torch
                from torch import nn
                from torch.nn import functional as F
                from llm_engineering_lab.acceleration import get_accelerator

                SEED = 606
                torch.manual_seed(SEED)
                ACCELERATOR = get_accelerator()
                DEVICE = ACCELERATOR.device
                print(f"torch={torch.__version__}, seed={SEED}; {ACCELERATOR.summary()}")
                """,
                "setup",
            ),
            code(
                """
                sentences = [
                    "king royal palace queen royal palace".split(),
                    "queen royal crown king royal crown".split(),
                    "man human adult woman human adult".split(),
                    "woman human person man human person".split(),
                    "paris france capital berlin germany capital".split(),
                    "berlin germany capital paris france capital".split(),
                    "cat pet animal dog pet animal".split(),
                    "dog pet animal cat pet animal".split(),
                ]
                vocab = sorted({token for sentence in sentences for token in sentence})
                stoi = {token: index for index, token in enumerate(vocab)}

                # TODO 06-1: window 안의 (중심어, 문맥어) 쌍을 만드세요.
                def make_skipgram_pairs(rows, window=2):
                    raise NotImplementedError

                # TODO 06-2: count의 3/4승을 정규화해 noise_probs를 만드세요.
                pairs = make_skipgram_pairs(sentences, window=2)
                counts = torch.tensor([Counter(token for row in sentences for token in row)[w]
                                       for w in vocab], dtype=torch.float32)
                noise_probs = ...
                pairs, counts, noise_probs = ACCELERATOR.move(pairs, counts, noise_probs)
                """,
                """
                sentences = [
                    "king royal palace queen royal palace".split(),
                    "queen royal crown king royal crown".split(),
                    "man human adult woman human adult".split(),
                    "woman human person man human person".split(),
                    "paris france capital berlin germany capital".split(),
                    "berlin germany capital paris france capital".split(),
                    "cat pet animal dog pet animal".split(),
                    "dog pet animal cat pet animal".split(),
                ]
                vocab = sorted({token for sentence in sentences for token in sentence})
                stoi = {token: index for index, token in enumerate(vocab)}

                def make_skipgram_pairs(rows, window=2):
                    result = []
                    for row in rows:
                        ids = [stoi[token] for token in row]
                        for center_position, center_id in enumerate(ids):
                            left = max(0, center_position - window)
                            right = min(len(ids), center_position + window + 1)
                            for context_position in range(left, right):
                                if context_position != center_position:
                                    result.append((center_id, ids[context_position]))
                    return torch.tensor(result, dtype=torch.long)

                pairs = make_skipgram_pairs(sentences, window=2)
                token_counts = Counter(token for row in sentences for token in row)
                counts = torch.tensor([token_counts[word] for word in vocab], dtype=torch.float32)
                noise_probs = counts.pow(0.75)
                noise_probs = noise_probs / noise_probs.sum()
                pairs, counts, noise_probs = ACCELERATOR.move(pairs, counts, noise_probs)

                assert pairs.ndim == 2 and pairs.shape[1] == 2
                assert torch.all(pairs[:, 0] != pairs[:, 1])
                assert torch.isclose(noise_probs.sum(), noise_probs.new_tensor(1.0))
                most_frequent = int(counts.argmax())
                assert noise_probs[most_frequent] < counts[most_frequent] / counts.sum()
                print(f"vocab={len(vocab)}, positive_pairs={len(pairs)}")
                """,
                "graded",
                "skipgram-data",
            ),
            code(
                """
                # TODO 06-3: 입력/출력 임베딩을 가진 SGNS 모형을 완성하세요.
                class SGNS(nn.Module):
                    def __init__(self, vocab_size, embedding_dim):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, center_ids, positive_ids, negative_ids):
                        raise NotImplementedError

                # TODO 06-4: 논문 식 (4)의 음의 로그우도를 반환하세요.
                def sgns_loss(positive_scores, negative_scores):
                    raise NotImplementedError
                """,
                """
                class SGNS(nn.Module):
                    def __init__(self, vocab_size, embedding_dim):
                        super().__init__()
                        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
                        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
                        bound = 0.5 / embedding_dim
                        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
                        nn.init.zeros_(self.output_embeddings.weight)

                    def forward(self, center_ids, positive_ids, negative_ids):
                        center = self.input_embeddings(center_ids)                 # [P, D]
                        positive = self.output_embeddings(positive_ids)            # [P, D]
                        negatives = self.output_embeddings(negative_ids)           # [P, K, D]
                        positive_scores = (center * positive).sum(dim=-1)           # [P]
                        negative_scores = torch.einsum("pd,pkd->pk", center, negatives)
                        return positive_scores, negative_scores

                def sgns_loss(positive_scores, negative_scores):
                    positive_term = F.logsigmoid(positive_scores)
                    negative_term = F.logsigmoid(-negative_scores).sum(dim=1)
                    return -(positive_term + negative_term).mean()

                probe = SGNS(len(vocab), embedding_dim=12).to(DEVICE)
                probe_negatives = torch.zeros((4, 3), dtype=torch.long, device=DEVICE)
                probe_positive, probe_negative = probe(pairs[:4, 0], pairs[:4, 1], probe_negatives)
                probe_loss = sgns_loss(probe_positive, probe_negative)
                probe_loss.backward()
                assert probe_positive.shape == (4,) and probe_negative.shape == (4, 3)
                assert probe_loss.ndim == 0 and torch.isfinite(probe_loss)
                assert probe.input_embeddings.weight.grad is not None
                assert probe.output_embeddings.weight.grad is not None
                """,
                "graded",
                "sgns-objective",
            ),
            code(
                """
                # TODO 06-5: 고정된 음성 표본으로 SGNS를 학습하고 loss_history를 채우세요.
                torch.manual_seed(SEED)
                model = SGNS(len(vocab), embedding_dim=12).to(DEVICE)
                generator = torch.Generator().manual_seed(SEED + 1)
                negative_ids = ...
                optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
                loss_history = []
                raise NotImplementedError
                """,
                """
                torch.manual_seed(SEED)
                model = SGNS(len(vocab), embedding_dim=12).to(DEVICE)
                generator = torch.Generator().manual_seed(SEED + 1)
                negative_ids = torch.multinomial(
                    noise_probs.detach().cpu(),
                    num_samples=len(pairs) * 5,
                    replacement=True,
                    generator=generator,
                ).view(len(pairs), 5).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
                loss_history = []

                for _ in range(160):
                    optimizer.zero_grad()
                    positive_scores, negative_scores = model(
                        pairs[:, 0], pairs[:, 1], negative_ids
                    )
                    loss = sgns_loss(positive_scores, negative_scores)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(float(loss.detach()))

                reduction = 1.0 - loss_history[-1] / loss_history[0]
                assert loss_history[-1] < 0.75 * loss_history[0]
                assert math.isfinite(reduction)
                print(f"initial_loss={loss_history[0]:.3f}, final_loss={loss_history[-1]:.3f}, "
                      f"reduction={reduction:.1%}")
                """,
                "graded",
                "training",
            ),
            code(
                """
                # TODO 06-6: 입력/출력 임베딩 평균으로 코사인 행렬과 2D 투영을 그리세요.
                raise NotImplementedError
                """,
                """
                with torch.inference_mode():
                    word_vectors = (
                        model.input_embeddings.weight
                        + model.output_embeddings.weight
                    ) / 2
                    word_vectors = F.normalize(word_vectors, dim=1)
                    focus_words = ["king", "queen", "man", "woman", "cat", "dog"]
                    focus_ids = torch.tensor([stoi[word] for word in focus_words], device=DEVICE)
                    focus_vectors = word_vectors[focus_ids]
                    cosine_matrix = focus_vectors @ focus_vectors.T
                    centered = focus_vectors - focus_vectors.mean(dim=0, keepdim=True)
                    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                    coordinates = centered @ vh[:2].T

                assert cosine_matrix.shape == (len(focus_words), len(focus_words))
                assert torch.allclose(
                    cosine_matrix.diag(),
                    cosine_matrix.new_ones(len(focus_words)),
                    atol=1e-5,
                )
                assert torch.all((-1.0001 <= cosine_matrix) & (cosine_matrix <= 1.0001))

                fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
                axes[0].plot(loss_history)
                axes[0].set(title="SGNS objective", xlabel="step", ylabel="loss")
                axes[1].scatter(coordinates[:, 0].detach().cpu(), coordinates[:, 1].detach().cpu())
                for word, (x, y) in zip(focus_words, coordinates.tolist()):
                    axes[1].annotate(word, (x, y), xytext=(4, 4), textcoords="offset points")
                axes[1].set_title("learned word vectors (SVD projection)")
                plt.tight_layout()
                plt.show()
                """,
                "graded",
                "visualization",
            ),
            markdown(
                """
                ## 해석 체크

                - 왜 출력 임베딩이 입력 임베딩과 별도로 필요한가?
                - `0.75`를 `1.0`으로 바꾸면 자주 등장하는 단어가 음성 표본에서 어떻게 달라지는가?
                - 이 작은 투영이 원 논문의 analogy 정확도를 재현했다고 말할 수 없는 이유를 적어 보세요.
                """,
                "interpretation",
            ),
        ),
    ),
    PaperSpec(
        number=7,
        slug="sequence_to_sequence_lstm",
        short_title="LSTM Sequence to Sequence",
        paper_title="Sequence to Sequence Learning with Neural Networks",
        authors="Ilya Sutskever, Oriol Vinyals, Quoc V. Le",
        year=2014,
        primary_url="https://arxiv.org/abs/1409.3215",
        venue="NeurIPS 2014",
        difficulty="중급",
        expected_minutes=75,
        prerequisites="LSTM, teacher forcing, cross entropy, autoregressive decoding",
        reproduction_goal=(
            "LSTM encoder의 마지막 hidden/cell state를 LSTM decoder의 초기 상태로 넘기고, "
            "BOS 입력과 한 칸 이동한 target으로 조건부 다음-token 확률을 학습한다. 논문의 "
            "source 순서 뒤집기 기법과 greedy autoregressive decoding까지 작은 복사 과제로 확인한다."
        ),
        original_scale=(
            "원 논문은 WMT'14 약 1,200만 문장 쌍, 160k/80k source/target vocabulary와 "
            "4층 1000-unit LSTM(약 3.8억 파라미터)을 여러 GPU에서 학습했다. 여기서는 "
            "8개 길이-3 수열을 한 층 LSTM으로 암기해 정보 흐름만 재현한다."
        ),
        mappings=(
            (
                "§2, 식 (1): 조건부 확률을 token별 확률의 곱으로 분해",
                "`decoder_inputs`/`next_token_targets` shift와 teacher-forced NLL",
                "decoder input과 target이 정확히 한 칸 어긋나는지 assert",
            ),
            (
                "§2: 첫 LSTM의 마지막 state가 고정 길이 표현 v를 구성",
                "`Seq2Seq.forward`가 encoder `(h, c)`를 decoder 초기 상태로 전달",
                "logit shape와 encoder/decoder state shape 검사",
            ),
            (
                "§2: 두 번째 LSTM이 v에 조건부로 출력 문장을 생성",
                "`greedy_decode`가 BOS부터 예측 token을 다시 입력",
                "token accuracy와 exact-sequence accuracy를 모두 출력",
            ),
            (
                "§3.2: source sentence의 단어 순서를 뒤집으면 minimal time lag 감소",
                "`build_seq2seq_batch`의 `encoder_ids = flip(source_ids)`",
                "첫 행이 정확히 역순인지 assert",
            ),
        ),
        cells=(
            markdown(
                r"""
                ## 핵심 아이디어

                논문의 기본 모형은 한 LSTM이 source를 읽고 마지막 상태로 표현 $v$를 만든 뒤,
                다른 LSTM이
                $p(T'\mid S)=\prod_t p(y_t\mid v,y_1,\ldots,y_{t-1})$를 모델링한다.
                학습 중에는 실제 이전 token을 넣지만(teacher forcing), 평가 때는 자신의 예측을
                다시 입력해야 한다. 논문 §3.2처럼 encoder 입력만 뒤집고 정답 순서는 유지한다.
                """,
                "paper-theory",
            ),
            shared_code(
                """
                import matplotlib.pyplot as plt
                import torch
                from torch import nn
                from torch.nn import functional as F
                from llm_engineering_lab.acceleration import get_accelerator

                SEED = 707
                torch.manual_seed(SEED)
                ACCELERATOR = get_accelerator()
                DEVICE = ACCELERATOR.device
                print(ACCELERATOR.summary())
                PAD, BOS, EOS = 0, 1, 2
                VOCAB_SIZE = 8
                """,
                "setup",
            ),
            code(
                """
                source_ids = torch.tensor([
                    [3, 4, 5], [4, 5, 6], [5, 6, 7], [7, 6, 5],
                    [3, 6, 4], [4, 7, 3], [5, 3, 7], [6, 4, 3],
                ])
                # TODO 07-1: 역순 encoder input과 [BOS, source, EOS] target을 만드세요.
                def build_seq2seq_batch(source):
                    raise NotImplementedError

                encoder_ids, target_ids = build_seq2seq_batch(source_ids)
                source_ids, encoder_ids, target_ids = ACCELERATOR.move(
                    source_ids, encoder_ids, target_ids
                )
                """,
                """
                source_ids = torch.tensor([
                    [3, 4, 5], [4, 5, 6], [5, 6, 7], [7, 6, 5],
                    [3, 6, 4], [4, 7, 3], [5, 3, 7], [6, 4, 3],
                ])

                def build_seq2seq_batch(source):
                    if source.ndim != 2:
                        raise ValueError("source must have shape [batch, length]")
                    encoder_input = torch.flip(source, dims=(1,))
                    batch = source.shape[0]
                    bos_column = torch.full((batch, 1), BOS, dtype=torch.long)
                    eos_column = torch.full((batch, 1), EOS, dtype=torch.long)
                    decoder_target = torch.cat(
                        (bos_column, source, eos_column),
                        dim=1,
                    )
                    return encoder_input, decoder_target

                encoder_ids, target_ids = build_seq2seq_batch(source_ids)
                source_ids, encoder_ids, target_ids = ACCELERATOR.move(
                    source_ids, encoder_ids, target_ids
                )

                assert torch.equal(encoder_ids[0], encoder_ids.new_tensor([5, 4, 3]))
                assert torch.all(target_ids[:, 0] == BOS) and torch.all(target_ids[:, -1] == EOS)
                print("encoder:", encoder_ids[0].tolist(), "target:", target_ids[0].tolist())
                """,
                "graded",
                "data",
            ),
            code(
                """
                # TODO 07-3: encoder final (h, c)를 decoder 초기 상태로 넘기세요.
                class Seq2Seq(nn.Module):
                    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
                        super().__init__()
                        raise NotImplementedError

                    def encode(self, source):
                        raise NotImplementedError

                    def decode_teacher_forced(self, decoder_inputs, state):
                        raise NotImplementedError

                    def forward(self, source, decoder_inputs):
                        raise NotImplementedError
                """,
                """
                class Seq2Seq(nn.Module):
                    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
                        super().__init__()
                        self.source_embedding = nn.Embedding(vocab_size, embedding_dim)
                        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
                        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                        self.output = nn.Linear(hidden_dim, vocab_size)

                    def encode(self, source):
                        _, state = self.encoder(self.source_embedding(source))
                        return state

                    def decode_teacher_forced(self, decoder_inputs, state):
                        decoded, next_state = self.decoder(
                            self.target_embedding(decoder_inputs), state
                        )
                        return self.output(decoded), next_state

                    def forward(self, source, decoder_inputs):
                        state = self.encode(source)
                        logits, _ = self.decode_teacher_forced(decoder_inputs, state)
                        return logits

                torch.manual_seed(SEED)
                model = Seq2Seq(VOCAB_SIZE).to(DEVICE)
                probe_logits = model(encoder_ids, target_ids[:, :-1])
                assert probe_logits.shape == (len(source_ids), target_ids.shape[1] - 1, VOCAB_SIZE)
                """,
                "graded",
                "model",
            ),
            code(
                """
                # TODO 07-4: teacher input/target shift와 한 step update를 구현하세요.
                decoder_inputs = ...
                next_token_targets = ...
                optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
                loss_history = []

                def train_seq2seq_step(model, source, inputs, targets, optimizer):
                    raise NotImplementedError
                """,
                """
                decoder_inputs = target_ids[:, :-1]
                next_token_targets = target_ids[:, 1:]
                assert torch.equal(decoder_inputs[:, 1:], next_token_targets[:, :-1])

                optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
                loss_history = []

                def train_seq2seq_step(model, source, inputs, targets, optimizer):
                    model.train()
                    optimizer.zero_grad()
                    logits = model(source, inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE),
                        targets.reshape(-1),
                    )
                    loss.backward()
                    gradient_norm = nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )
                    optimizer.step()
                    return float(loss.detach()), float(gradient_norm)

                gradient_norms = []
                for _ in range(180):
                    step_loss, gradient_norm = train_seq2seq_step(
                        model,
                        encoder_ids,
                        decoder_inputs,
                        next_token_targets,
                        optimizer,
                    )
                    loss_history.append(step_loss)
                    gradient_norms.append(gradient_norm)

                assert loss_history[-1] < 0.15 * loss_history[0]
                assert torch.isfinite(torch.tensor(gradient_norms)).all()
                print(f"teacher-forced loss: {loss_history[0]:.3f} -> {loss_history[-1]:.3f}")
                """,
                "graded",
                "training",
            ),
            code(
                """
                # TODO 07-5: BOS에서 시작해 직전 예측을 재입력하는 greedy decoder를 구현하세요.
                @torch.inference_mode()
                def greedy_decode(model, source, steps):
                    raise NotImplementedError

                predicted_ids = greedy_decode(model, encoder_ids, target_ids.shape[1] - 1)
                """,
                """
                @torch.inference_mode()
                def greedy_decode(model, source, steps):
                    model.eval()
                    state = model.encode(source)
                    current = torch.full(
                        (len(source), 1),
                        BOS,
                        dtype=torch.long,
                        device=source.device,
                    )
                    predictions = []
                    for _ in range(steps):
                        logits, state = model.decode_teacher_forced(current, state)
                        current = logits[:, -1:].argmax(dim=-1)
                        predictions.append(current)
                    return torch.cat(predictions, dim=1)

                predicted_ids = greedy_decode(model, encoder_ids, target_ids.shape[1] - 1)
                expected_ids = target_ids[:, 1:]
                token_accuracy = (predicted_ids == expected_ids).float().mean().item()
                exact_accuracy = (predicted_ids == expected_ids).all(dim=1).float().mean().item()
                assert token_accuracy >= 0.95 and exact_accuracy >= 0.75
                print(
                    f"token_accuracy={token_accuracy:.1%}, "
                    f"exact_sequence_accuracy={exact_accuracy:.1%}"
                )
                print(
                    "prediction:",
                    predicted_ids[0].tolist(),
                    "expected:",
                    expected_ids[0].tolist(),
                )
                """,
                "graded",
                "decoding",
            ),
            code(
                """
                # TODO 07-6: loss curve와 수열별 exact-match 지표를 시각화하세요.
                raise NotImplementedError
                """,
                """
                per_sequence = (predicted_ids == expected_ids).all(dim=1).float()
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
                axes[0].plot(loss_history)
                axes[0].set(title="teacher-forced NLL", xlabel="step", ylabel="loss")
                axes[1].bar(range(len(per_sequence)), per_sequence.tolist())
                axes[1].set(title="autoregressive exact match", xlabel="sequence", ylim=(0, 1.05))
                plt.tight_layout()
                plt.show()
                """,
                "graded",
                "visualization",
            ),
            markdown(
                """
                ## 해석 체크

                teacher-forced loss가 작아도 autoregressive exact match가 낮을 수 있는 이유와,
                source를 뒤집는 것이 장거리 의존성의 *시간 간격*을 어떻게 줄이는지 적어 보세요.
                """,
                "interpretation",
            ),
        ),
    ),
    PaperSpec(
        number=8,
        slug="bahdanau_additive_attention",
        short_title="Bahdanau Additive Attention",
        paper_title="Neural Machine Translation by Jointly Learning to Align and Translate",
        authors="Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio",
        year=2015,
        primary_url="https://arxiv.org/abs/1409.0473",
        venue="ICLR 2015",
        difficulty="중급",
        expected_minutes=65,
        prerequisites="encoder-decoder, softmax, masking, batched matrix multiplication",
        reproduction_goal=(
            "decoder query와 모든 encoder annotation 사이 additive energy를 계산하고, "
            "padding을 softmax 전에 가린 뒤 정렬 확률의 가중합으로 context를 만든다. "
            "작은 단조 정렬 과제에서 attention weight가 올바른 위치를 찾는지 측정한다."
        ),
        original_scale=(
            "원 논문은 양방향 RNN encoder와 gated decoder를 WMT'14 영어-프랑스어 번역에 "
            "학습해 BLEU와 alignment를 평가했다. 여기서는 이미 만들어진 6차원 annotation과 "
            "5개 query만 사용하며 번역기 전체나 BLEU를 재현하지 않는다."
        ),
        mappings=(
            (
                "§3.1, 식 (6): context vector는 annotation의 가중합",
                "`AdditiveAttention.forward`의 `context = bmm(alpha, annotations)`",
                "context shape와 attention 행 합 1을 assert",
            ),
            (
                "§3.1, 식 (7): energy를 softmax로 정규화해 α_ij 계산",
                "`AdditiveAttention.forward`의 masked energy와 `softmax(dim=-1)`",
                "padding 열의 확률이 정확히 0인지 검사",
            ),
            (
                "§3.1, 식 (8) 및 Appendix A.1.2: feed-forward alignment model a(s,h)",
                "`AdditiveAttention.score`의 `v(tanh(W_s s + W_h h))`",
                "energy shape 및 모든 파라미터 gradient 확인",
            ),
            (
                "Figure 3: target 위치와 source annotation 사이 soft alignment",
                "합성 monotonic alignment 정확도와 attention heatmap",
                "argmax alignment accuracy 및 padding mass로 정량 검증",
            ),
        ),
        cells=(
            markdown(
                r"""
                ## 핵심 아이디어

                하나의 고정 길이 벡터 대신 decoder step $i$마다
                $e_{ij}=v_a^\top\tanh(W_a s_{i-1}+U_a h_j)$를 계산한다. 이를 source 축으로
                softmax한 $\alpha_{ij}$가 soft alignment이고,
                $c_i=\sum_j\alpha_{ij}h_j$가 현재 step의 context다(식 (6)--(8)).

                논문에는 padding mask가 수식으로 명시되지 않지만, minibatch 구현에서는 가짜
                annotation이 확률 질량을 받지 않도록 energy를 softmax **전에** 가려야 한다.
                """,
                "paper-theory",
            ),
            shared_code(
                """
                import matplotlib.pyplot as plt
                import torch
                from torch import nn
                from torch.nn import functional as F
                from llm_engineering_lab.acceleration import get_accelerator

                SEED = 808
                torch.manual_seed(SEED)
                ACCELERATOR = get_accelerator()
                DEVICE = ACCELERATOR.device
                print(ACCELERATOR.summary())

                # 다섯 decoder step, 여섯 encoder slot(마지막은 padding)
                STEPS, SOURCE_SLOTS, STATE_DIM = 5, 6, 6
                annotations = torch.eye(SOURCE_SLOTS).unsqueeze(0).expand(STEPS, -1, -1).clone()
                decoder_queries = torch.eye(STATE_DIM)[:STEPS]
                source_mask = torch.ones((STEPS, SOURCE_SLOTS), dtype=torch.bool)
                source_mask[:, -1] = False
                alignment_targets = torch.arange(STEPS)
                annotations, decoder_queries, source_mask, alignment_targets = ACCELERATOR.move(
                    annotations, decoder_queries, source_mask, alignment_targets
                )
                """,
                "setup",
            ),
            code(
                """
                # TODO 08-1: 식 (6)--(8)의 additive attention을 구현하세요.
                class AdditiveAttention(nn.Module):
                    def __init__(self, query_dim, annotation_dim, attention_dim):
                        super().__init__()
                        raise NotImplementedError

                    def score(self, query, encoder_annotations):
                        raise NotImplementedError

                    def normalize(self, energy, valid_mask):
                        raise NotImplementedError

                    def compose_context(self, alpha, encoder_annotations):
                        raise NotImplementedError

                    def forward(self, query, encoder_annotations, valid_mask):
                        raise NotImplementedError
                """,
                """
                class AdditiveAttention(nn.Module):
                    def __init__(self, query_dim, annotation_dim, attention_dim):
                        super().__init__()
                        self.query_projection = nn.Linear(query_dim, attention_dim, bias=False)
                        self.annotation_projection = nn.Linear(
                            annotation_dim,
                            attention_dim,
                            bias=False,
                        )
                        self.energy_projection = nn.Linear(attention_dim, 1, bias=False)

                    def score(self, query, encoder_annotations):
                        if query.ndim != 2 or encoder_annotations.ndim != 3:
                            raise ValueError("expected query [B,D] and annotations [B,S,D]")
                        query_term = self.query_projection(query).unsqueeze(1)
                        annotation_term = self.annotation_projection(encoder_annotations)
                        joint_state = torch.tanh(query_term + annotation_term)
                        return self.energy_projection(joint_state).squeeze(-1)

                    def normalize(self, energy, valid_mask):
                        if energy.shape != valid_mask.shape:
                            raise ValueError("energy and valid_mask shapes must match")
                        if not valid_mask.any(dim=1).all():
                            raise ValueError("every sequence needs at least one valid annotation")
                        masked_energy = energy.masked_fill(
                            ~valid_mask,
                            torch.finfo(energy.dtype).min,
                        )
                        alpha = torch.softmax(masked_energy, dim=-1)
                        return alpha, masked_energy

                    def compose_context(self, alpha, encoder_annotations):
                        weighted = torch.bmm(alpha.unsqueeze(1), encoder_annotations)
                        return weighted.squeeze(1)

                    def forward(self, query, encoder_annotations, valid_mask):
                        energy = self.score(query, encoder_annotations)
                        alpha, masked_energy = self.normalize(energy, valid_mask)
                        context = self.compose_context(alpha, encoder_annotations)
                        return context, alpha, masked_energy

                torch.manual_seed(SEED)
                attention = AdditiveAttention(STATE_DIM, STATE_DIM, attention_dim=12).to(DEVICE)
                context, alpha, energy = attention(decoder_queries, annotations, source_mask)
                assert context.shape == (STEPS, STATE_DIM)
                assert alpha.shape == (STEPS, SOURCE_SLOTS)
                assert torch.allclose(alpha.sum(dim=1), alpha.new_ones(STEPS), atol=1e-6)
                assert torch.all(alpha[:, -1] == 0)
                """,
                "graded",
                "attention",
            ),
            code(
                """
                # TODO 08-2: energy에 cross entropy를 적용해 단조 정렬을 학습하세요.
                optimizer = torch.optim.Adam(attention.parameters(), lr=0.05)
                loss_history = []
                raise NotImplementedError
                """,
                """
                optimizer = torch.optim.Adam(attention.parameters(), lr=0.05)
                loss_history = []
                for _ in range(140):
                    optimizer.zero_grad()
                    _, _, energy = attention(decoder_queries, annotations, source_mask)
                    loss = F.cross_entropy(energy, alignment_targets)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(float(loss.detach()))

                assert loss_history[-1] < 0.2 * loss_history[0]
                for parameter in attention.parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                """,
                "graded",
                "alignment-training",
            ),
            code(
                """
                # TODO 08-3: 최종 alignment accuracy와 padding probability mass를 계산하세요.
                raise NotImplementedError
                """,
                """
                with torch.inference_mode():
                    learned_context, learned_alpha, _ = attention(
                        decoder_queries, annotations, source_mask
                    )
                predicted_alignment = learned_alpha.argmax(dim=1)
                alignment_accuracy = (
                    (predicted_alignment == alignment_targets)
                    .float()
                    .mean()
                    .item()
                )
                padding_mass = learned_alpha[:, -1].sum().item()
                identity_context = torch.eye(STEPS, device=DEVICE)
                context_error = (
                    (learned_context[:, :STEPS] - identity_context)
                    .abs()
                    .mean()
                    .item()
                )

                assert alignment_accuracy >= 0.8
                assert padding_mass == 0.0
                assert context_error < 0.1
                print(
                    f"alignment_accuracy={alignment_accuracy:.1%}, "
                    f"padding_mass={padding_mass:.1e}, "
                    f"context_MAE={context_error:.4f}"
                )
                """,
                "graded",
                "metrics",
            ),
            code(
                """
                # TODO 08-4: Figure 3처럼 decoder×encoder attention heatmap을 그리세요.
                raise NotImplementedError
                """,
                """
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
                axes[0].plot(loss_history)
                axes[0].set(title="alignment NLL", xlabel="step", ylabel="loss")
                image = axes[1].imshow(
                    learned_alpha.detach().cpu(),
                    vmin=0,
                    vmax=1,
                    cmap="Blues",
                    aspect="auto",
                )
                axes[1].set(
                    title="soft alignment alpha(i,j)",
                    xlabel="encoder position j (last = PAD)",
                    ylabel="decoder step i",
                    xticks=range(SOURCE_SLOTS),
                    yticks=range(STEPS),
                )
                fig.colorbar(image, ax=axes[1], fraction=0.046)
                plt.tight_layout()
                plt.show()
                """,
                "graded",
                "visualization",
            ),
            markdown(
                """
                ## 해석 체크

                1. 왜 `softmax` 뒤가 아니라 앞에서 padding energy를 가려야 할까요?
                2. context가 decoder step마다 달라지는 것이 고정 길이 encoder-decoder 병목을
                   어떻게 완화하는지 설명하세요.
                """,
                "interpretation",
            ),
        ),
    ),
    PaperSpec(
        number=9,
        slug="attention_is_all_you_need",
        short_title="Attention Is All You Need",
        paper_title="Attention Is All You Need",
        authors="Ashish Vaswani et al.",
        year=2017,
        primary_url="https://arxiv.org/abs/1706.03762",
        venue="NeurIPS 2017",
        difficulty="중상급",
        expected_minutes=80,
        prerequisites="선형대수, softmax, broadcasting, attention mask",
        reproduction_goal=(
            "식 (1)의 scaled dot-product attention, 식 (2)--(3)의 multi-head projection, "
            "§3.3의 position-wise FFN, §3.1의 residual connection과 post-LayerNorm, "
            "§3.5의 sinusoidal positional encoding을 직접 구현한다. 같은 block을 encoder처럼 "
            "mask 없이, decoder self-attention처럼 causal mask와 함께 실행해 미래 token "
            "불변성, 확률 정규화, scale이 entropy에 미치는 효과를 검증한다."
        ),
        original_scale=(
            "원 논문의 base Transformer는 6층 encoder/decoder, d_model=512, 8 heads로 "
            "WMT'14 En-De 약 450만 문장 쌍을 학습했다. 이 노트북은 d_model=16, 4 heads인 "
            "단일 self-attention+FFN block을 사용한다. encoder-decoder cross-attention, "
            "6층 stack, label smoothing, 번역 학습과 beam search는 재현하지 않는다."
        ),
        mappings=(
            (
                "§3.2.1, 식 (1): Attention(Q,K,V)=softmax(QKᵀ/√d_k)V",
                "`scaled_dot_product_attention`의 score→mask→softmax→weighted sum",
                "attention 행 합, shape, 유한값을 assert",
            ),
            (
                "§3.2.2, 식 (2)--(3): head별 projection 후 concat과 W^O",
                "`MultiHeadSelfAttention`의 projection/split/transpose/merge",
                "출력과 head별 weight shape 검사",
            ),
            (
                "§3.1: decoder self-attention에서 subsequent position 차단",
                "`causal_attention_mask`와 `TransformerBlock(..., allowed=causal)`",
                "미래 attention mass=0 및 prefix 출력 불변성 assert",
            ),
            (
                "§3.5: sinusoidal positional encoding 수식",
                "`sinusoidal_positions`의 짝수 sin/홀수 cos 구현",
                "position 0 패턴과 shape 검사",
            ),
            (
                "§3.2.1: 1/√d_k scaling이 큰 내적으로 인한 작은 gradient를 완화",
                "`attention_entropy`의 scaled/unscaled score 비교",
                "scaled entropy가 더 큰지 정량 비교하고 막대그래프로 표시",
            ),
            (
                "§3.1: Sublayer 출력은 LayerNorm(x + Sublayer(x))",
                "`TransformerBlock`의 attention/FFN residual 뒤 각각 post-LayerNorm",
                "residual shape와 token별 평균·분산을 검사",
            ),
            (
                "§3.3, 식 FFN(x)=max(0,xW1+b1)W2+b2",
                "`PositionWiseFeedForward`의 두 Linear와 ReLU",
                "동일 module이 모든 token 위치에 독립 적용되는 shape를 검사",
            ),
        ),
        cells=(
            markdown(
                r"""
                ## 핵심 아이디어

                Transformer는 recurrence 없이 attention만으로 위치 간 정보를 섞는다.
                dot product의 분산은 차원 $d_k$와 함께 커지므로 식 (1)은 점수를
                $\sqrt{d_k}$로 나눈다. 여러 projection 공간에서 병렬로 attention한 결과를
                이어 붙이는 것이 multi-head attention이다. 순서는 고정 sin/cos 위치 벡터를
                token 표현에 더해 제공하며, decoder에는 미래를 볼 수 없는 causal mask가 필요하다.

                §3.1의 원 논문 block은 각 sublayer 뒤에
                $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$를 적용하는 **post-LN** 구조다.
                여기서도 이 순서를 그대로 드러내며, §3.3의 position-wise FFN도 고수준
                Transformer module 안에 숨기지 않는다. `allowed=None`은 encoder self-attention,
                lower-triangular mask는 decoder의 첫 masked self-attention sublayer에 대응한다.
                단, decoder의 encoder-decoder attention은 이 축소 실험 범위 밖이다.
                """,
                "paper-theory",
            ),
            shared_code(
                """
                import math

                import matplotlib.pyplot as plt
                import torch
                from torch import nn
                from llm_engineering_lab.acceleration import get_accelerator

                SEED = 909
                torch.manual_seed(SEED)
                ACCELERATOR = get_accelerator()
                DEVICE = ACCELERATOR.device
                print(ACCELERATOR.summary())
                """,
                "setup",
            ),
            code(
                """
                # TODO 09-1: 논문 §3.5의 짝수 sin/홀수 cos positional encoding을 구현하세요.
                def sinusoidal_positions(length, d_model):
                    raise NotImplementedError

                positions = sinusoidal_positions(length=6, d_model=16)
                """,
                """
                def sinusoidal_positions(length, d_model):
                    if d_model % 2:
                        raise ValueError("d_model must be even in this compact implementation")
                    position = torch.arange(length, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                    inverse_scale = torch.exp(
                        torch.arange(0, d_model, 2, dtype=torch.float32, device=DEVICE)
                        * (-math.log(10000.0) / d_model)
                    )
                    encoding = torch.zeros(length, d_model, device=DEVICE)
                    encoding[:, 0::2] = torch.sin(position * inverse_scale)
                    encoding[:, 1::2] = torch.cos(position * inverse_scale)
                    return encoding

                positions = sinusoidal_positions(length=6, d_model=16)
                assert positions.shape == (6, 16)
                assert torch.allclose(positions[0, 0::2], positions.new_zeros(8))
                assert torch.allclose(positions[0, 1::2], positions.new_ones(8))
                assert not torch.allclose(positions[1], positions[2])
                """,
                "graded",
                "position-encoding",
            ),
            code(
                """
                # TODO 09-2: 식 (1)의 scale, mask, softmax, value 가중합을 구현하세요.
                def scaled_dot_product_attention(query, key, value, allowed=None):
                    raise NotImplementedError
                """,
                """
                def scaled_dot_product_attention(query, key, value, allowed=None):
                    if query.shape[:-2] != key.shape[:-2]:
                        raise ValueError("query and key batch/head axes must match")
                    if key.shape[-2] != value.shape[-2]:
                        raise ValueError("key and value sequence lengths must match")
                    scale = math.sqrt(query.shape[-1])
                    scores = query @ key.transpose(-2, -1) / scale
                    if allowed is not None:
                        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
                    weights = torch.softmax(scores, dim=-1)
                    output = weights @ value
                    return output, weights

                q = torch.randn(2, 4, 6, 4, device=DEVICE)
                probe_output, probe_weights = scaled_dot_product_attention(q, q, q)
                assert probe_output.shape == q.shape
                expected_row_sums = probe_weights.new_ones(2, 4, 6)
                assert torch.allclose(
                    probe_weights.sum(dim=-1),
                    expected_row_sums,
                    atol=1e-6,
                )
                """,
                "graded",
                "scaled-attention",
            ),
            code(
                """
                # TODO 09-3: 식 (2)--(3)의 head split, attention, concat, W^O를 구현하세요.
                class MultiHeadSelfAttention(nn.Module):
                    def __init__(self, d_model, num_heads):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, hidden, allowed=None):
                        raise NotImplementedError
                """,
                """
                class MultiHeadSelfAttention(nn.Module):
                    def __init__(self, d_model, num_heads):
                        super().__init__()
                        if d_model % num_heads:
                            raise ValueError("d_model must be divisible by num_heads")
                        self.num_heads = num_heads
                        self.head_dim = d_model // num_heads
                        self.query = nn.Linear(d_model, d_model, bias=False)
                        self.key = nn.Linear(d_model, d_model, bias=False)
                        self.value = nn.Linear(d_model, d_model, bias=False)
                        self.output = nn.Linear(d_model, d_model, bias=False)

                    def _split(self, tensor):
                        batch, length, _ = tensor.shape
                        split = tensor.view(
                            batch,
                            length,
                            self.num_heads,
                            self.head_dim,
                        )
                        return split.transpose(1, 2)

                    def _merge(self, tensor):
                        batch, _, length, _ = tensor.shape
                        transposed = tensor.transpose(1, 2).contiguous()
                        return transposed.view(
                            batch,
                            length,
                            self.num_heads * self.head_dim,
                        )

                    def forward(self, hidden, allowed=None):
                        q = self._split(self.query(hidden))
                        k = self._split(self.key(hidden))
                        v = self._split(self.value(hidden))
                        attended, weights = scaled_dot_product_attention(q, k, v, allowed)
                        merged = self._merge(attended)
                        return self.output(merged), weights

                torch.manual_seed(SEED)
                attention = MultiHeadSelfAttention(d_model=16, num_heads=4).to(DEVICE)
                token_vectors = torch.randn(1, 6, 16, device=DEVICE)
                hidden = token_vectors + positions.unsqueeze(0)

                def causal_attention_mask(length, device):
                    allowed = torch.ones(
                        length,
                        length,
                        dtype=torch.bool,
                        device=device,
                    ).tril()
                    return allowed.view(1, 1, length, length)

                causal = causal_attention_mask(hidden.shape[1], DEVICE)
                attended, attention_weights = attention(hidden, allowed=causal)

                future = ~causal
                assert attended.shape == (1, 6, 16)
                assert attention_weights.shape == (1, 4, 6, 6)
                assert torch.all(attention_weights.masked_select(future) == 0)
                """,
                "graded",
                "multi-head",
            ),
            code(
                """
                # TODO 09-4: §3.3의 FFN과 §3.1의 post-LN residual block을 구현하세요.
                class PositionWiseFeedForward(nn.Module):
                    def __init__(self, d_model, d_ff):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, hidden):
                        raise NotImplementedError

                class TransformerBlock(nn.Module):
                    def __init__(self, d_model, num_heads, d_ff):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, hidden, allowed=None):
                        raise NotImplementedError

                block = TransformerBlock(d_model=16, num_heads=4, d_ff=32).to(DEVICE)
                """,
                """
                class PositionWiseFeedForward(nn.Module):
                    def __init__(self, d_model, d_ff):
                        super().__init__()
                        self.input_projection = nn.Linear(d_model, d_ff)
                        self.output_projection = nn.Linear(d_ff, d_model)

                    def forward(self, hidden):
                        expanded = self.input_projection(hidden)
                        activated = torch.relu(expanded)
                        return self.output_projection(activated)

                class TransformerBlock(nn.Module):
                    def __init__(self, d_model, num_heads, d_ff):
                        super().__init__()
                        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
                        self.attention_norm = nn.LayerNorm(d_model)
                        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
                        self.feed_forward_norm = nn.LayerNorm(d_model)

                    def forward(self, hidden, allowed=None):
                        attention_output, weights = self.self_attention(hidden, allowed)
                        after_attention = self.attention_norm(
                            hidden + attention_output
                        )
                        feed_forward_output = self.feed_forward(after_attention)
                        output = self.feed_forward_norm(
                            after_attention + feed_forward_output
                        )
                        return output, weights, after_attention

                torch.manual_seed(SEED)
                block = TransformerBlock(
                    d_model=16,
                    num_heads=4,
                    d_ff=32,
                ).to(DEVICE)
                encoder_output, encoder_weights, _ = block(hidden, allowed=None)
                decoder_output, decoder_weights, residual_state = block(
                    hidden,
                    allowed=causal,
                )

                assert encoder_output.shape == decoder_output.shape == hidden.shape
                assert encoder_weights.shape == decoder_weights.shape == (1, 4, 6, 6)
                assert residual_state.shape == hidden.shape
                assert torch.allclose(
                    decoder_output.mean(dim=-1),
                    decoder_output.new_zeros(1, 6),
                    atol=1e-5,
                )
                assert torch.all(decoder_weights.masked_select(future) == 0)
                """,
                "graded",
                "residual-ffn-block",
            ),
            code(
                """
                # TODO 09-5: full block에서도 미래 token 불변성을 검증하세요.
                changed_hidden = hidden.clone()
                changed_hidden[:, -1] += 100.0
                raise NotImplementedError

                # TODO 09-6: scaled/unscaled score entropy를 비교하고 attention을 그리세요.
                """,
                """
                changed_hidden = hidden.clone()
                changed_hidden[:, -1] += 100.0
                with torch.inference_mode():
                    changed_output, _, _ = block(changed_hidden, allowed=causal)
                prefix_delta = decoder_output[:, :-1] - changed_output[:, :-1]
                prefix_max_change = prefix_delta.abs().max().item()
                assert prefix_max_change < 1e-6

                entropy_q = torch.randn(32, 16, device=DEVICE)
                entropy_k = torch.randn(32, 16, device=DEVICE)
                raw_scores = entropy_q @ entropy_k.T
                unscaled_weights = raw_scores.softmax(dim=-1)
                scaled_weights = (raw_scores / math.sqrt(16)).softmax(dim=-1)

                def attention_entropy(weights):
                    return float(-(weights * weights.clamp_min(1e-9).log()).sum(dim=-1).mean())

                unscaled_entropy = attention_entropy(unscaled_weights)
                scaled_entropy = attention_entropy(scaled_weights)
                assert scaled_entropy > unscaled_entropy
                print(
                    f"prefix_max_change={prefix_max_change:.2e}, "
                    f"entropy: unscaled={unscaled_entropy:.3f}, "
                    f"scaled={scaled_entropy:.3f}"
                )

                fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
                image = axes[0].imshow(
                    decoder_weights[0, 0].detach().cpu(),
                    vmin=0,
                    vmax=1,
                    cmap="magma",
                )
                axes[0].set(title="head 0 causal attention", xlabel="key", ylabel="query")
                fig.colorbar(image, ax=axes[0], fraction=0.046)
                axes[1].bar(["unscaled", "scaled"], [unscaled_entropy, scaled_entropy])
                axes[1].set(title="mean attention entropy", ylabel="nats")
                plt.tight_layout()
                plt.show()
                """,
                "graded",
                "causality-and-visualization",
            ),
            markdown(
                """
                ## 해석 체크

                1. `sqrt(d_k)`로 나누지 않으면 softmax와 gradient에 어떤 일이 생기나요?
                2. 원 논문의 post-LN 수식과 최신 pre-LN 구현의 연산 순서는 어떻게 다른가요?
                3. causal mask 검증에서 마지막 token을 크게 바꾼 이유는 무엇인가요?
                4. 서로 다른 head가 유용할 수 있는 이유를 projection 관점에서 설명하세요.
                """,
                "interpretation",
            ),
        ),
    ),
    PaperSpec(
        number=10,
        slug="gpt1_generative_pretraining",
        short_title="GPT-1 Generative Pre-Training",
        paper_title="Improving Language Understanding by Generative Pre-Training",
        authors="Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever",
        year=2018,
        primary_url=(
            "https://cdn.openai.com/research-covers/language-unsupervised/"
            "language_understanding_paper.pdf"
        ),
        venue="OpenAI Technical Report",
        difficulty="중상급",
        expected_minutes=85,
        prerequisites="Transformer decoder, causal LM loss, fine-tuning, 분류 cross entropy",
        reproduction_goal=(
            "작은 decoder-only Transformer를 causal language modeling으로 사전학습하고, "
            "마지막 token 표현의 분류 head와 보조 LM 손실을 함께 미세조정한다. 미래 token "
            "불변성과 사전학습/미세조정 metric을 통해 GPT-1의 두 단계 절차를 재현한다."
        ),
        original_scale=(
            "GPT-1은 BooksCorpus(7천 권 이상)의 연속 텍스트로 12층, 768-hidden, 12-head, "
            "약 1.17억 파라미터 Transformer를 사전학습한 뒤 여러 NLP benchmark에 미세조정했다. "
            "여기서는 8개 고정 수열과 1층 24-hidden 모형을 사용한다."
        ),
        mappings=(
            (
                "§3.1, 식 (1): corpus token의 autoregressive likelihood L1",
                "`lm_inputs`/`lm_targets` shift와 causal LM cross entropy",
                "LM loss 감소율을 assert하고 곡선으로 표시",
            ),
            (
                "§3.1, 식 (2): multi-layer Transformer decoder h_l",
                "`CausalSelfAttention`, `GPTBlock`, token+position embedding",
                "logit shape와 미래 token에 대한 prefix 불변성 검사",
            ),
            (
                "§3.1, 식 (2): P(u)=softmax(h_n W_e^T)의 embedding weight 재사용",
                "`TinyGPT.lm_logits`의 `F.linear(hidden, token_embedding.weight)`",
                "LM vocabulary 축과 embedding parameter identity를 코드로 확인",
            ),
            (
                "§3.2, 식 (3): 마지막 Transformer activation을 선형 분류기로 전달",
                "`classification_head(hidden[:, -1])`",
                "합성 분류 accuracy로 평가",
            ),
            (
                "§3.2, 식 (4): L3 = L2 + λ·L1 보조 언어모델 목적",
                "`classification_loss + 0.2 * language_model_loss`",
                "두 loss가 유한한지와 최종 accuracy를 assert",
            ),
            (
                "§3.3 및 Figure 1: 구조화 입력을 ordered token sequence로 변환",
                "`sequences`가 BOS/content/EOS를 하나의 연속 token sequence로 구성",
                "고정 길이와 경계 token을 검사",
            ),
        ),
        cells=(
            markdown(
                r"""
                ## 핵심 아이디어

                GPT-1의 절차는 (1) 레이블 없는 연속 텍스트에서 왼쪽 문맥만 보는 language model
                likelihood $L_1$을 최대화하고, (2) 같은 Transformer를 task input에 맞게 미세조정하는
                것이다. 지도 목적 $L_2$에 보조 LM 목적을 더한 $L_3=L_2+\lambda L_1$도 사용한다.
                코드는 likelihood 최대화 대신 부호를 바꾼 cross-entropy를 최소화한다.

                `nn.TransformerEncoder` 한 줄로 숨기지 않고, 식 (2)의 causal self-attention,
                residual, LayerNorm, GELU feed-forward를 `CausalSelfAttention`과 `GPTBlock`으로
                분리한다. GPT-1은 원 Transformer를 바탕으로 하므로 여기서는 residual 뒤에
                LayerNorm을 두는 post-LN 순서를 쓴다. 언어모델 출력은 식 (2)처럼 token
                embedding weight를 전치해 재사용한다.
                """,
                "paper-theory",
            ),
            shared_code(
                """
                import math

                import matplotlib.pyplot as plt
                import torch
                from torch import nn
                from torch.nn import functional as F
                from llm_engineering_lab.acceleration import get_accelerator

                SEED = 1010
                torch.manual_seed(SEED)
                ACCELERATOR = get_accelerator()
                DEVICE = ACCELERATOR.device
                print(ACCELERATOR.summary())
                BOS, EOS, VOCAB_SIZE = 1, 2, 16
                """,
                "setup",
            ),
            code(
                """
                content = torch.tensor([
                    [3, 4, 5, 6], [3, 4, 5, 7], [3, 5, 4, 6], [3, 5, 4, 7],
                    [8, 9, 10, 11], [8, 9, 10, 12], [8, 10, 9, 11], [8, 10, 9, 12],
                ])
                labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
                # TODO 10-1: Figure 1의 ordered token sequence처럼 BOS/content/EOS를 연결하세요.
                sequences = ...
                # TODO 10-2: causal LM의 입력과 한 칸 앞 target을 만드세요.
                lm_inputs, lm_targets = ..., ...
                content, labels, sequences, lm_inputs, lm_targets = ACCELERATOR.move(
                    content, labels, sequences, lm_inputs, lm_targets
                )
                """,
                """
                content = torch.tensor([
                    [3, 4, 5, 6], [3, 4, 5, 7], [3, 5, 4, 6], [3, 5, 4, 7],
                    [8, 9, 10, 11], [8, 9, 10, 12], [8, 10, 9, 11], [8, 10, 9, 12],
                ])
                labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
                bos = torch.full((len(content), 1), BOS, dtype=torch.long)
                eos = torch.full((len(content), 1), EOS, dtype=torch.long)
                sequences = torch.cat((bos, content, eos), dim=1)
                lm_inputs, lm_targets = sequences[:, :-1], sequences[:, 1:]
                content, labels, sequences, lm_inputs, lm_targets = ACCELERATOR.move(
                    content, labels, sequences, lm_inputs, lm_targets
                )

                assert sequences.shape == (8, 6)
                assert torch.all(sequences[:, 0] == BOS) and torch.all(sequences[:, -1] == EOS)
                assert torch.equal(lm_inputs[:, 1:], lm_targets[:, :-1])
                """,
                "graded",
                "task-transform",
            ),
            code(
                """
                # TODO 10-3: causal attention과 post-LN GPT block을 직접 구현하세요.
                class CausalSelfAttention(nn.Module):
                    def __init__(self, d_model, num_heads):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, hidden):
                        raise NotImplementedError

                class GPTBlock(nn.Module):
                    def __init__(self, d_model, num_heads, d_ff):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, hidden):
                        raise NotImplementedError

                # TODO 10-4: embedding, block, tied LM head, task head를 조립하세요.
                class TinyGPT(nn.Module):
                    def __init__(self, vocab_size, max_length, d_model=24, num_heads=4):
                        super().__init__()
                        raise NotImplementedError

                    def encode(self, token_ids):
                        raise NotImplementedError

                    def lm_logits(self, token_ids):
                        raise NotImplementedError

                    def classification_logits(self, token_ids):
                        raise NotImplementedError
                """,
                """
                class CausalSelfAttention(nn.Module):
                    def __init__(self, d_model, num_heads):
                        super().__init__()
                        if d_model % num_heads:
                            raise ValueError("d_model must be divisible by num_heads")
                        self.num_heads = num_heads
                        self.head_dim = d_model // num_heads
                        self.query_key_value = nn.Linear(
                            d_model,
                            3 * d_model,
                            bias=False,
                        )
                        self.output_projection = nn.Linear(
                            d_model,
                            d_model,
                            bias=False,
                        )

                    def _split_heads(self, tensor):
                        batch, length, width = tensor.shape
                        split = tensor.view(
                            batch,
                            length,
                            self.num_heads,
                            width // self.num_heads,
                        )
                        return split.transpose(1, 2)

                    def _merge_heads(self, tensor):
                        batch, _, length, _ = tensor.shape
                        merged = tensor.transpose(1, 2).contiguous()
                        return merged.view(
                            batch,
                            length,
                            self.num_heads * self.head_dim,
                        )

                    def forward(self, hidden):
                        query, key, value = self.query_key_value(hidden).chunk(3, dim=-1)
                        query = self._split_heads(query)
                        key = self._split_heads(key)
                        value = self._split_heads(value)
                        scores = query @ key.transpose(-2, -1)
                        scores = scores / math.sqrt(self.head_dim)
                        length = hidden.shape[1]
                        blocked = torch.ones(
                            length,
                            length,
                            dtype=torch.bool,
                            device=hidden.device,
                        ).triu(diagonal=1)
                        scores = scores.masked_fill(
                            blocked.view(1, 1, length, length),
                            torch.finfo(scores.dtype).min,
                        )
                        weights = torch.softmax(scores, dim=-1)
                        context = self._merge_heads(weights @ value)
                        return self.output_projection(context), weights

                class GPTBlock(nn.Module):
                    def __init__(self, d_model, num_heads, d_ff):
                        super().__init__()
                        self.attention = CausalSelfAttention(d_model, num_heads)
                        self.attention_norm = nn.LayerNorm(d_model)
                        self.feed_forward_input = nn.Linear(d_model, d_ff)
                        self.feed_forward_output = nn.Linear(d_ff, d_model)
                        self.feed_forward_norm = nn.LayerNorm(d_model)

                    def forward(self, hidden):
                        attention_output, weights = self.attention(hidden)
                        hidden = self.attention_norm(hidden + attention_output)
                        expanded = self.feed_forward_input(hidden)
                        feed_forward_output = self.feed_forward_output(F.gelu(expanded))
                        hidden = self.feed_forward_norm(hidden + feed_forward_output)
                        return hidden, weights

                class TinyGPT(nn.Module):
                    def __init__(self, vocab_size, max_length, d_model=24, num_heads=4):
                        super().__init__()
                        self.token_embedding = nn.Embedding(vocab_size, d_model)
                        self.position_embedding = nn.Embedding(max_length, d_model)
                        self.blocks = nn.ModuleList(
                            [GPTBlock(d_model, num_heads, d_ff=2 * d_model)]
                        )
                        self.classification_head = nn.Linear(d_model, 2)

                    def encode(self, token_ids):
                        length = token_ids.shape[1]
                        positions = torch.arange(
                            length,
                            device=token_ids.device,
                        ).unsqueeze(0)
                        hidden = self.token_embedding(token_ids)
                        hidden = hidden + self.position_embedding(positions)
                        for block in self.blocks:
                            hidden, _ = block(hidden)
                        return hidden

                    def lm_logits(self, token_ids):
                        hidden = self.encode(token_ids)
                        return F.linear(hidden, self.token_embedding.weight)

                    def classification_logits(self, token_ids):
                        return self.classification_head(self.encode(token_ids)[:, -1])

                torch.manual_seed(SEED)
                model = TinyGPT(VOCAB_SIZE, max_length=sequences.shape[1]).to(DEVICE)
                assert model.lm_logits(lm_inputs).shape == (*lm_inputs.shape, VOCAB_SIZE)
                assert model.classification_logits(sequences).shape == (len(sequences), 2)
                assert len(model.blocks) == 1
                """,
                "graded",
                "model",
            ),
            code(
                """
                # TODO 10-5: 식 (1)의 next-token cross entropy로 사전학습하세요.
                pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
                pretrain_losses = []
                raise NotImplementedError

                # TODO 10-6: 한 수열의 미래 suffix를 바꾸고 prefix logits 불변성을 검사하세요.
                """,
                """
                pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
                pretrain_losses = []
                for _ in range(70):
                    model.train()
                    pretrain_optimizer.zero_grad()
                    logits = model.lm_logits(lm_inputs)
                    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), lm_targets.reshape(-1))
                    loss.backward()
                    pretrain_optimizer.step()
                    pretrain_losses.append(float(loss.detach()))

                assert pretrain_losses[-1] < 0.45 * pretrain_losses[0]

                model.eval()
                original = lm_inputs[:1].clone()
                changed = original.clone()
                changed[:, 3:] = changed.new_tensor([[13, 14]])
                with torch.inference_mode():
                    original_logits = model.lm_logits(original)
                    changed_logits = model.lm_logits(changed)
                prefix_change = (original_logits[:, :3] - changed_logits[:, :3]).abs().max().item()
                assert prefix_change < 1e-6
                print(f"pretrain_loss={pretrain_losses[0]:.3f}->{pretrain_losses[-1]:.3f}, "
                      f"prefix_change={prefix_change:.2e}")
                """,
                "graded",
                "pretraining",
            ),
            code(
                """
                # TODO 10-7: 식 (4)의 classification CE + 0.2 * LM CE를 구현하세요.
                finetune_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                supervised_losses, auxiliary_lm_losses = [], []
                raise NotImplementedError
                """,
                """
                finetune_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                supervised_losses, auxiliary_lm_losses = [], []
                for _ in range(60):
                    model.train()
                    finetune_optimizer.zero_grad()
                    class_logits = model.classification_logits(sequences)
                    lm_logits = model.lm_logits(lm_inputs)
                    supervised_loss = F.cross_entropy(class_logits, labels)
                    auxiliary_lm_loss = F.cross_entropy(
                        lm_logits.reshape(-1, VOCAB_SIZE), lm_targets.reshape(-1)
                    )
                    joint_loss = supervised_loss + 0.2 * auxiliary_lm_loss
                    joint_loss.backward()
                    finetune_optimizer.step()
                    supervised_losses.append(float(supervised_loss.detach()))
                    auxiliary_lm_losses.append(float(auxiliary_lm_loss.detach()))

                model.eval()
                with torch.inference_mode():
                    predictions = model.classification_logits(sequences).argmax(dim=-1)
                classification_accuracy = (predictions == labels).float().mean().item()
                assert classification_accuracy >= 0.875
                assert math.isfinite(supervised_losses[-1])
                assert math.isfinite(auxiliary_lm_losses[-1])
                print(f"fine-tune accuracy={classification_accuracy:.1%}")
                """,
                "graded",
                "fine-tuning",
            ),
            code(
                """
                # TODO 10-8: 사전학습/미세조정 loss와 최종 accuracy를 시각화하세요.
                raise NotImplementedError
                """,
                """
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
                axes[0].plot(pretrain_losses, label="pretrain LM")
                axes[0].plot(auxiliary_lm_losses, label="fine-tune auxiliary LM")
                axes[0].set(title="generative objectives", xlabel="step", ylabel="cross entropy")
                axes[0].legend()
                axes[1].bar(
                    ["error", "accuracy"],
                    [1 - classification_accuracy, classification_accuracy],
                )
                axes[1].set(title="downstream classification", ylim=(0, 1.05))
                plt.tight_layout()
                plt.show()
                """,
                "graded",
                "visualization",
            ),
            markdown(
                """
                ## 해석 체크

                보조 LM loss가 미세조정 중 어떤 regularizer 역할을 하는지, 그리고 이 실습의
                `classification_logits`가 GPT-1 Figure 1의 task-aware input transform 전체를
                재현하지 않는 이유를 적어 보세요.
                """,
                "interpretation",
            ),
        ),
    ),
    PaperSpec(
        number=11,
        slug="bert_pretraining",
        short_title="BERT: MLM과 NSP",
        paper_title=(
            "BERT: Pre-training of Deep Bidirectional Transformers for "
            "Language Understanding"
        ),
        authors="Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        year=2018,
        primary_url="https://arxiv.org/abs/1810.04805",
        venue="NAACL 2019",
        difficulty="중상급",
        expected_minutes=90,
        prerequisites="Transformer encoder, masked loss, segment embedding, multi-task learning",
        reproduction_goal=(
            "token·position·segment embedding을 더한 양방향 Transformer encoder를 만들고, "
            "선택 token의 80/10/10 corruption을 거친 Masked Language Model과 [CLS] 기반 "
            "Next Sentence Prediction을 공동 학습한다. 두 metric과 오른쪽 문맥 민감도를 검증한다."
        ),
        original_scale=(
            "BERT Base/Large는 각각 12/24층, 110M/340M 파라미터로 BooksCorpus와 English "
            "Wikipedia(총 33억 단어)에서 100만 step 사전학습됐다. 여기서는 10개 문장 쌍, "
            "1층 24-hidden encoder를 100여 step 암기시키는 구조 검증이다."
        ),
        mappings=(
            (
                "§3: input representation = token + segment + position embedding",
                "`TinyBERT.embed`의 세 embedding 합과 LayerNorm",
                "hidden 및 두 head logit shape 검사",
            ),
            (
                "§3.1, Task #1: Masked LM에서 입력 token의 15%를 선택",
                "특수 token을 제외한 60개 중 10개를 고정 선택(미니배치 근사 16.7%)",
                "선택 수와 특수 token 비선택을 assert",
            ),
            (
                "§3.1, MLM 선택 token의 80% [MASK], 10% random, 10% unchanged",
                "`corrupted_ids`의 deterministic 8/1/1 corruption",
                "각 corruption 개수를 정확히 검사",
            ),
            (
                "§3.1, Task #2: [CLS]로 IsNext/NotNext 예측",
                "`nsp_head(hidden[:, 0])`와 binary CE",
                "NSP accuracy 출력 및 assert",
            ),
            (
                "Figure 1: GPT와 달리 모든 층에서 좌우 문맥을 함께 조건으로 사용",
                "mask 없는 `BidirectionalSelfAttention`과 오른쪽 token 교란 실험",
                "right-context sensitivity가 0보다 큰지 assert",
            ),
            (
                "§3 및 Figure 1: Transformer encoder의 양방향 self-attention",
                "`BidirectionalSelfAttention`과 `BertEncoderBlock`의 post-LN residual",
                "attention `[B,H,L,L]`과 양방향 확률 질량을 검사",
            ),
        ),
        cells=(
            markdown(
                r"""
                ## 핵심 아이디어

                BERT는 causal mask가 없는 Transformer encoder로 모든 층에서 왼쪽과 오른쪽 문맥을
                함께 사용한다. 그대로 다음 token을 예측하면 정답을 보게 되므로, 입력 token 일부를
                가리고 원 token을 맞히는 MLM을 쓴다. 선택된 token은 80% `[MASK]`, 10% random,
                10% unchanged로 바꾸며, 문장 쌍의 `[CLS]` 표현에는 NSP 이진 목적을 함께 적용한다.

                구현에서는 `nn.TransformerEncoder`를 한 줄로 호출하지 않는다. query/key/value
                projection, head 분할, scaled score, 전체 위치 softmax를
                `BidirectionalSelfAttention`으로 드러낸다. `BertEncoderBlock`은 BERT의 원래
                post-LN 순서인 `LayerNorm(x + sublayer(x))`를 attention과 GELU FFN 뒤에 각각
                적용한다. causal mask가 없다는 사실이 코드에서 바로 보여야 GPT와의 차이를
                설명할 수 있다.
                """,
                "paper-theory",
            ),
            shared_code(
                """
                import math

                import matplotlib.pyplot as plt
                import torch
                from torch import nn
                from torch.nn import functional as F
                from llm_engineering_lab.acceleration import get_accelerator

                SEED = 1111
                torch.manual_seed(SEED)
                ACCELERATOR = get_accelerator()
                DEVICE = ACCELERATOR.device
                print(ACCELERATOR.summary())
                PAD, CLS, SEP, MASK = 0, 1, 2, 3
                VOCAB_SIZE = 24

                sentence_pairs = [
                    ([4, 5, 6], [7, 8, 9], 1),
                    ([7, 8, 9], [13, 14, 15], 1),
                    ([10, 11, 12], [16, 17, 18], 1),
                    ([16, 17, 18], [19, 20, 21], 1),
                    ([4, 6, 5], [8, 9, 7], 1),
                    ([4, 5, 6], [16, 17, 18], 0),
                    ([7, 8, 9], [19, 20, 21], 0),
                    ([10, 11, 12], [7, 8, 9], 0),
                    ([16, 17, 18], [13, 14, 15], 0),
                    ([4, 6, 5], [19, 21, 20], 0),
                ]
                input_ids = torch.tensor([[CLS, *a, SEP, *b, SEP] for a, b, _ in sentence_pairs])
                segment_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1]] * len(sentence_pairs))
                nsp_labels = torch.tensor([label for _, _, label in sentence_pairs])
                input_ids, segment_ids, nsp_labels = ACCELERATOR.move(
                    input_ids, segment_ids, nsp_labels
                )
                """,
                "setup",
            ),
            code(
                """
                # TODO 11-1: 한 행당 content token 하나를 골라 MLM label(-100=무시)을 만드세요.
                selected_positions = torch.tensor([1, 2, 3, 5, 6, 7, 1, 2, 5, 6], device=DEVICE)
                corrupted_ids = input_ids.clone()
                mlm_labels = torch.full_like(input_ids, -100)
                raise NotImplementedError

                # TODO 11-2: 선택 10개를 8개 MASK / 1개 random / 1개 unchanged로 손상하세요.
                """,
                """
                selected_positions = torch.tensor([1, 2, 3, 5, 6, 7, 1, 2, 5, 6], device=DEVICE)
                selected_rows = torch.arange(len(input_ids), device=DEVICE)
                corrupted_ids = input_ids.clone()
                mlm_labels = torch.full_like(input_ids, -100)
                mlm_labels[selected_rows, selected_positions] = input_ids[
                    selected_rows, selected_positions
                ]

                corrupted_ids[selected_rows[:8], selected_positions[:8]] = MASK
                corrupted_ids[selected_rows[8], selected_positions[8]] = 22  # random token
                # row 9 is deliberately unchanged

                chosen = mlm_labels != -100
                mask_count = int((corrupted_ids[chosen] == MASK).sum())
                random_count = int(
                    (
                        (corrupted_ids[chosen] != MASK)
                        & (corrupted_ids[chosen] != input_ids[chosen])
                    ).sum()
                )
                unchanged_count = int((corrupted_ids[chosen] == input_ids[chosen]).sum())
                assert int(chosen.sum()) == 10
                assert (mask_count, random_count, unchanged_count) == (8, 1, 1)
                assert not chosen[:, 0].any() and not chosen[:, 4].any() and not chosen[:, 8].any()
                print(f"selected={int(chosen.sum())}/60 content tokens; corruption=8/1/1")
                """,
                "graded",
                "mlm-corruption",
            ),
            code(
                """
                # TODO 11-3: mask 없는 multi-head self-attention을 직접 구현하세요.
                class BidirectionalSelfAttention(nn.Module):
                    def __init__(self, d_model, num_heads):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, hidden):
                        raise NotImplementedError

                class BertEncoderBlock(nn.Module):
                    def __init__(self, d_model, num_heads, d_ff):
                        super().__init__()
                        raise NotImplementedError

                    def forward(self, hidden):
                        raise NotImplementedError

                # TODO 11-4: 세 embedding, encoder block, MLM/NSP head를 조립하세요.
                class TinyBERT(nn.Module):
                    def __init__(self, vocab_size, max_length, d_model=24, num_heads=4):
                        super().__init__()
                        raise NotImplementedError

                    def embed(self, token_ids, token_type_ids):
                        raise NotImplementedError

                    def forward(self, token_ids, token_type_ids):
                        raise NotImplementedError
                """,
                """
                class BidirectionalSelfAttention(nn.Module):
                    def __init__(self, d_model, num_heads):
                        super().__init__()
                        if d_model % num_heads:
                            raise ValueError("d_model must be divisible by num_heads")
                        self.num_heads = num_heads
                        self.head_dim = d_model // num_heads
                        self.query = nn.Linear(d_model, d_model, bias=False)
                        self.key = nn.Linear(d_model, d_model, bias=False)
                        self.value = nn.Linear(d_model, d_model, bias=False)
                        self.output = nn.Linear(d_model, d_model, bias=False)

                    def _split_heads(self, tensor):
                        batch, length, _ = tensor.shape
                        split = tensor.view(
                            batch,
                            length,
                            self.num_heads,
                            self.head_dim,
                        )
                        return split.transpose(1, 2)

                    def _merge_heads(self, tensor):
                        batch, _, length, _ = tensor.shape
                        merged = tensor.transpose(1, 2).contiguous()
                        return merged.view(
                            batch,
                            length,
                            self.num_heads * self.head_dim,
                        )

                    def forward(self, hidden):
                        query = self._split_heads(self.query(hidden))
                        key = self._split_heads(self.key(hidden))
                        value = self._split_heads(self.value(hidden))
                        scores = query @ key.transpose(-2, -1)
                        scores = scores / math.sqrt(self.head_dim)
                        weights = torch.softmax(scores, dim=-1)
                        context = self._merge_heads(weights @ value)
                        return self.output(context), weights

                class BertEncoderBlock(nn.Module):
                    def __init__(self, d_model, num_heads, d_ff):
                        super().__init__()
                        self.attention = BidirectionalSelfAttention(
                            d_model,
                            num_heads,
                        )
                        self.attention_norm = nn.LayerNorm(d_model)
                        self.intermediate = nn.Linear(d_model, d_ff)
                        self.output = nn.Linear(d_ff, d_model)
                        self.output_norm = nn.LayerNorm(d_model)

                    def forward(self, hidden):
                        attended, weights = self.attention(hidden)
                        hidden = self.attention_norm(hidden + attended)
                        transformed = self.output(F.gelu(self.intermediate(hidden)))
                        hidden = self.output_norm(hidden + transformed)
                        return hidden, weights

                class TinyBERT(nn.Module):
                    def __init__(self, vocab_size, max_length, d_model=24, num_heads=4):
                        super().__init__()
                        self.token_embedding = nn.Embedding(vocab_size, d_model)
                        self.position_embedding = nn.Embedding(max_length, d_model)
                        self.segment_embedding = nn.Embedding(2, d_model)
                        self.embedding_norm = nn.LayerNorm(d_model)
                        self.encoder = BertEncoderBlock(
                            d_model,
                            num_heads,
                            d_ff=2 * d_model,
                        )
                        self.mlm_transform = nn.Linear(d_model, d_model)
                        self.mlm_norm = nn.LayerNorm(d_model)
                        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
                        self.nsp_head = nn.Linear(d_model, 2)

                    def embed(self, token_ids, token_type_ids):
                        batch, length = token_ids.shape
                        positions = torch.arange(
                            length,
                            device=token_ids.device,
                        ).unsqueeze(0)
                        positions = positions.expand(batch, length)
                        return self.embedding_norm(
                            self.token_embedding(token_ids)
                            + self.position_embedding(positions)
                            + self.segment_embedding(token_type_ids)
                        )

                    def masked_language_model_logits(self, hidden):
                        transformed = F.gelu(self.mlm_transform(hidden))
                        transformed = self.mlm_norm(transformed)
                        return F.linear(
                            transformed,
                            self.token_embedding.weight,
                            self.mlm_bias,
                        )

                    def forward(self, token_ids, token_type_ids):
                        embedded = self.embed(token_ids, token_type_ids)
                        hidden, weights = self.encoder(embedded)
                        mlm_logits = self.masked_language_model_logits(hidden)
                        nsp_logits = self.nsp_head(hidden[:, 0])
                        return mlm_logits, nsp_logits, hidden, weights

                torch.manual_seed(SEED)
                model = TinyBERT(VOCAB_SIZE, max_length=input_ids.shape[1]).to(DEVICE)
                probe_mlm, probe_nsp, probe_hidden, probe_attention = model(
                    corrupted_ids,
                    segment_ids,
                )
                assert probe_mlm.shape == (*input_ids.shape, VOCAB_SIZE)
                assert probe_nsp.shape == (len(input_ids), 2)
                assert probe_hidden.shape[:2] == input_ids.shape
                assert probe_attention.shape == (len(input_ids), 4, 9, 9)
                expected_attention_sums = probe_attention.new_ones(
                    len(input_ids),
                    4,
                    9,
                )
                assert torch.allclose(
                    probe_attention.sum(dim=-1),
                    expected_attention_sums,
                    atol=1e-6,
                )
                """,
                "graded",
                "model",
            ),
            code(
                """
                # TODO 11-5: 선택 위치 MLM CE와 [CLS] NSP CE를 더해 공동 학습하세요.
                optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
                total_losses, mlm_losses, nsp_losses = [], [], []
                raise NotImplementedError
                """,
                """
                optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
                total_losses, mlm_losses, nsp_losses = [], [], []
                for _ in range(110):
                    model.train()
                    optimizer.zero_grad()
                    mlm_logits, nsp_logits, _, _ = model(
                        corrupted_ids,
                        segment_ids,
                    )
                    mlm_loss = F.cross_entropy(
                        mlm_logits.reshape(-1, VOCAB_SIZE),
                        mlm_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
                    total_loss = mlm_loss + nsp_loss
                    total_loss.backward()
                    optimizer.step()
                    total_losses.append(float(total_loss.detach()))
                    mlm_losses.append(float(mlm_loss.detach()))
                    nsp_losses.append(float(nsp_loss.detach()))

                model.eval()
                with torch.inference_mode():
                    mlm_logits, nsp_logits, _, _ = model(
                        corrupted_ids,
                        segment_ids,
                    )
                mlm_predictions = mlm_logits[chosen].argmax(dim=-1)
                mlm_accuracy = (mlm_predictions == mlm_labels[chosen]).float().mean().item()
                nsp_accuracy = (nsp_logits.argmax(dim=-1) == nsp_labels).float().mean().item()

                assert total_losses[-1] < 0.2 * total_losses[0]
                assert mlm_accuracy >= 0.8 and nsp_accuracy >= 0.8
                print(f"MLM accuracy={mlm_accuracy:.1%}, NSP accuracy={nsp_accuracy:.1%}")
                """,
                "graded",
                "pretraining-objectives",
            ),
            code(
                """
                # TODO 11-6: 오른쪽 token을 바꿔 양방향 문맥 민감도를 측정하세요.
                raise NotImplementedError
                """,
                """
                context_a = corrupted_ids[:1].clone()
                context_b = context_a.clone()
                masked_position = int(selected_positions[0])
                assert context_a[0, masked_position] == MASK
                context_b[0, masked_position + 1] = 22  # only right context changes
                with torch.inference_mode():
                    logits_a, _, _, attention_a = model(
                        context_a,
                        segment_ids[:1],
                    )
                    logits_b, _, _, _ = model(
                        context_b,
                        segment_ids[:1],
                    )
                right_context_sensitivity = (
                    logits_a[0, masked_position] - logits_b[0, masked_position]
                ).abs().max().item()
                assert right_context_sensitivity > 1e-5
                right_context_mass = attention_a[
                    0,
                    :,
                    masked_position,
                    masked_position + 1 :,
                ].sum()
                assert right_context_mass > 0
                print(f"right-context logit sensitivity={right_context_sensitivity:.4f}")

                fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
                axes[0].plot(mlm_losses, label="MLM")
                axes[0].plot(nsp_losses, label="NSP")
                axes[0].set(title="BERT pre-training objectives", xlabel="step", ylabel="CE")
                axes[0].legend()
                axes[1].bar(["MLM", "NSP"], [mlm_accuracy, nsp_accuracy])
                axes[1].set(title="training-set metrics", ylabel="accuracy", ylim=(0, 1.05))
                plt.tight_layout()
                plt.show()
                """,
                "graded",
                "bidirectionality-and-visualization",
            ),
            markdown(
                """
                ## 해석 체크

                1. MLM loss를 모든 위치가 아니라 선택된 위치에만 적용하는 이유는 무엇인가요?
                2. 선택 token의 10%를 그대로 두는 경우에도 정답 loss를 계산하는 이유는 무엇인가요?
                3. 이 노트북의 training accuracy가 실제 downstream 일반화를 뜻하지 않는 이유를 적으세요.
                """,
                "interpretation",
            ),
        ),
    ),
)


assert tuple(spec.number for spec in SPECS) == tuple(range(6, 12))
