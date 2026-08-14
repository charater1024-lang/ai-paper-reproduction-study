"""Portfolio-grade paired reproductions for ten landmark NLP and LLM papers."""

from __future__ import annotations

from .common import FieldPaperSpec, code, markdown, shared_code

FIELD_ID = "nlp_llm"
FIELD_TITLE = "자연어 처리와 LLM"
DATASET = "data/field_curriculum/nlp_corpus.json"


def _setup():
    return shared_code(
        f"""
        from collections import Counter
        from dataclasses import dataclass
        from pathlib import Path
        import json
        import math
        import re
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        from torch.nn import functional as F
        from llm_engineering_lab.acceleration import get_accelerator

        torch.manual_seed(20260814)
        ACCELERATOR = get_accelerator()
        DEVICE = ACCELERATOR.device

        def locate(relative_path):
            for base in (Path.cwd(), *Path.cwd().parents):
                candidate = base / relative_path
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(relative_path)

        dataset_path = locate("{DATASET}")
        corpus = json.loads(dataset_path.read_text(encoding="utf-8"))
        sentences = corpus["sentences"]
        translation_pairs = corpus["translation_pairs"]
        documents = corpus["documents"]
        queries = corpus["queries"]
        SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<mask>", "<sentinel>"]
        sentence_words = {{
            word.lower()
            for sentence in sentences
            for word in sentence.split()
        }}
        translation_words = {{
            word.lower()
            for pair in translation_pairs
            for side in ("source", "target")
            for word in pair[side].split()
        }}
        words = sorted(sentence_words | translation_words)
        itos = SPECIAL_TOKENS + words
        stoi = {{word: index for index, word in enumerate(itos)}}
        PAD, BOS, EOS, MASK, SENTINEL = range(5)
        VOCAB_SIZE = len(itos)

        def encode(text, length=4):
            token_ids = [stoi[word.lower()] for word in text.split()]
            token_ids = token_ids[:length]
            return token_ids + [PAD] * (length - len(token_ids))

        sentence_ids = torch.tensor([encode(text) for text in sentences])
        sentiment_labels = torch.tensor(corpus["sentiment_labels"], dtype=torch.long)
        source_ids = torch.tensor(
            [encode(pair["source"], 3) for pair in translation_pairs]
        )
        target_ids = torch.tensor(
            [encode(pair["target"], 3) for pair in translation_pairs]
        )
        sentence_ids, sentiment_labels = ACCELERATOR.move(
            sentence_ids,
            sentiment_labels,
        )
        source_ids, target_ids = ACCELERATOR.move(source_ids, target_ids)
        assert sentence_ids.shape == (360, 4)
        assert source_ids.shape == target_ids.shape == (60, 3)
        print(ACCELERATOR.summary())
        print(dataset_path.name, f"sentences={{len(sentences)}}, vocab={{VOCAB_SIZE}}")
        """,
        "setup",
        "local-data",
    )


def _concept(equation: str, symbols: str, roles: str, shapes: str, limits: str):
    return markdown(
        f"""
        ## 핵심 아이디어와 수식

        $${equation}$$

        **기호 해설:** {symbols}

        **코드 대응:** {roles}

        **입력과 출력 shape:** {shapes}

        **포트폴리오 구현 범위:** {limits}

        각 단계는 데이터 전처리와 shape 확인부터 논문 고유 class, `forward`, 목적함수,
        `train_step`, 평가까지 이어진다. 핵심 수식을 한 번의 고수준 호출로 감추지 않고
        중간 tensor를 확인할 수 있게 분리한다.
        """,
        "paper-reading",
        "equation",
        "implementation-map",
    )


def _implementation_note(task_number: int, mapping):
    paper_part, lab_part, evidence = mapping
    return markdown(
        f"""
        ### 구현 단계 {task_number} — 원문 근거를 코드로 옮기기

        - **논문의 어느 부분인가:** {paper_part}
        - **노트북에서 구현할 부분:** {lab_part}
        - **구현 이유:** 원문의 확률 분해나 tensor 변환을 작은 입력에서 직접
          관찰하고, 고수준 API 뒤에 핵심 알고리즘을 숨기지 않기 위해서다.
        - **완료 증거:** {evidence}

        아래 TODO 전에 입력 token/feature shape와 수식의 축을 먼저 확인한다.
        """,
        "paper-section",
        f"task-{task_number}",
    )


def _paper(**values):
    mappings = values.pop("mappings")
    cells = values.pop("cells")
    mapping_indices = (0, min(1, len(mappings) - 1), len(mappings) - 1)
    annotated_cells = []
    task_number = 0
    for cell in cells:
        is_task = cell.cell_type == "code" and cell.exercise != cell.solution
        if is_task:
            task_number += 1
            mapping_index = mapping_indices[min(task_number - 1, 2)]
            annotated_cells.append(
                _implementation_note(task_number, mappings[mapping_index])
            )
        annotated_cells.append(cell)
    return FieldPaperSpec(
        field_id=FIELD_ID,
        field_title=FIELD_TITLE,
        dataset_file=DATASET,
        difficulty="중급",
        expected_minutes=120,
        prerequisites="Python, PyTorch, 확률·선형대수·sequence model의 기초",
        mappings=mappings,
        cells=tuple(annotated_cells),
        **values,
    )


WORD2VEC = _paper(
    number=0,
    slug="word2vec_sgns",
    short_title="word2vec SGNS",
    paper_title="Distributed Representations of Words and Phrases and their Compositionality",
    authors="Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean",
    year=2013,
    primary_url="https://arxiv.org/abs/1310.4546",
    venue="NeurIPS 2013",
    reproduction_goal="skip-gram context pair, negative sampling, unigram^(3/4) noise를 재현한다.",
    original_scale=(
        "수십억 token과 300차원 embedding 대신 로컬 360문장과 16차원을 "
        "사용하고, 288/72개의 분리된 문장 행으로 학습/평가한다."
    ),
    mappings=(
        (
            "§2 Fig. 1, Eq. (1): nearby-word prediction",
            "Task 1 window pairs",
            "pair 수",
        ),
        ("§2.2 Eq. (4): NEG objective", "Task 2–3", "positive/negative logits"),
        ("§2.2: Pn(w)=U(w)^(3/4)/Z", "Task 1", "distribution sum"),
        ("§5: additive/compositional representations", "Task 3", "embedding scores"),
    ),
    cells=(
        _concept(
            (
                r"\log\sigma(v_{w_O}^{\top}v_{w_I})"
                r"+\sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}"
                r"\log\sigma(-v_{w_i}^{\top}v_{w_I})"
            ),
            "w_I는 중심어, w_O는 실제 문맥어, w_i는 noise, k는 negative 개수다.",
            "Task 1이 pair/noise 분포, Task 2가 두 embedding table, Task 3이 SGNS loss를 구현한다.",
            "center [N], positive [N], negative [N,k] → dot logits [N], [N,k] → scalar loss.",
            "subsampling과 phrase detection은 제외하지만 Eq. (4)와 3/4 noise 분포는 보존한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정, window pair 생성, unigram^0.75 noise 분포를 구현하세요.
            @dataclass
            class SGNSConfig:
                pass

            def build_skipgram_pairs(token_lines, window_size):
                raise NotImplementedError

            def build_noise_distribution(token_lines, vocab_size, exponent):
                raise NotImplementedError
            """,
            """
            @dataclass
            class SGNSConfig:
                embedding_dim: int = 16
                window_size: int = 1
                negatives: int = 3
                noise_exponent: float = 0.75
                learning_rate: float = 0.04
                steps: int = 18

            def build_skipgram_pairs(token_lines, window_size):
                pairs = []
                for line in token_lines:
                    for center_index, center in enumerate(line):
                        left = max(0, center_index - window_size)
                        right = min(len(line), center_index + window_size + 1)
                        for context_index in range(left, right):
                            if context_index != center_index:
                                pairs.append((center, line[context_index]))
                return pairs

            def build_noise_distribution(token_lines, vocab_size, exponent):
                flattened = [token for line in token_lines for token in line]
                counts = torch.bincount(
                    torch.tensor(flattened),
                    minlength=vocab_size,
                ).float()
                powered = counts.pow(exponent)
                return powered / powered.sum()

            config = SGNSConfig()
            token_lines = [
                [stoi[word.lower()] for word in sentence.split()]
                for sentence in sentences
            ]
            split_generator = torch.Generator().manual_seed(17)
            sentence_order = torch.randperm(
                len(token_lines),
                generator=split_generator,
            )
            train_sentence_index = sentence_order[:288]
            test_sentence_index = sentence_order[288:]
            assert not bool(
                torch.isin(train_sentence_index, test_sentence_index).any()
            )
            train_token_lines = [
                token_lines[index]
                for index in train_sentence_index.tolist()
            ]
            test_token_lines = [
                token_lines[index]
                for index in test_sentence_index.tolist()
            ]
            train_pairs = build_skipgram_pairs(
                train_token_lines,
                config.window_size,
            )
            test_pairs = build_skipgram_pairs(
                test_token_lines,
                config.window_size,
            )
            noise_probability = build_noise_distribution(
                train_token_lines,
                VOCAB_SIZE,
                config.noise_exponent,
            )
            train_centers = torch.tensor(
                [pair[0] for pair in train_pairs],
                device=DEVICE,
            )
            train_contexts = torch.tensor(
                [pair[1] for pair in train_pairs],
                device=DEVICE,
            )
            test_centers = torch.tensor(
                [pair[0] for pair in test_pairs],
                device=DEVICE,
            )
            test_contexts = torch.tensor(
                [pair[1] for pair in test_pairs],
                device=DEVICE,
            )
            assert len(train_pairs) > 1000
            assert len(test_pairs) > 100
            assert torch.allclose(
                noise_probability.sum(),
                noise_probability.new_tensor(1.0),
            )
            """,
            "todo",
            "sampling",
        ),
        code(
            """
            # TODO 2: input/output embedding과 score 메서드를 갖는 SGNSModel을 구현하세요.
            class SGNSModel(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def positive_score(self, centers, positives):
                    '''[B] 형태의 center-positive 내적을 반환한다.'''
                    raise NotImplementedError

                def negative_scores(self, centers, negatives):
                    '''[B, K] 형태의 center-negative 내적을 반환한다.'''
                    raise NotImplementedError

                def forward(self, centers, positives, negatives):
                    raise NotImplementedError
            """,
            """
            class SGNSModel(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.input_embedding = nn.Embedding(vocab_size, config.embedding_dim)
                    self.output_embedding = nn.Embedding(vocab_size, config.embedding_dim)
                    nn.init.normal_(self.input_embedding.weight, std=0.1)
                    nn.init.zeros_(self.output_embedding.weight)

                def positive_score(self, centers, positives):
                    center_vectors = self.input_embedding(centers)
                    positive_vectors = self.output_embedding(positives)
                    return (center_vectors * positive_vectors).sum(dim=-1)

                def negative_scores(self, centers, negatives):
                    center_vectors = self.input_embedding(centers)
                    negative_vectors = self.output_embedding(negatives)
                    return torch.einsum("bd,bkd->bk", center_vectors, negative_vectors)

                def forward(self, centers, positives, negatives):
                    positive = self.positive_score(centers, positives)
                    negative = self.negative_scores(centers, negatives)
                    return positive, negative

            train_negative_generator = torch.Generator().manual_seed(7)
            test_negative_generator = torch.Generator().manual_seed(8)
            train_negatives = torch.multinomial(
                noise_probability,
                len(train_centers) * config.negatives,
                replacement=True,
                generator=train_negative_generator,
            ).view(-1, config.negatives)
            test_negatives = torch.multinomial(
                noise_probability,
                len(test_centers) * config.negatives,
                replacement=True,
                generator=test_negative_generator,
            ).view(-1, config.negatives)
            train_negatives = train_negatives.to(DEVICE)
            test_negatives = test_negatives.to(DEVICE)
            model = SGNSModel(VOCAB_SIZE, config).to(DEVICE)
            positive_logits, negative_logits = model(
                train_centers[:8],
                train_contexts[:8],
                train_negatives[:8],
            )
            assert positive_logits.shape == (8,)
            assert negative_logits.shape == (8, 3)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: SGNS loss, train step, positive-vs-noise 평가를 구현하세요.
            def sgns_loss(positive_logits, negative_logits):
                raise NotImplementedError

            def train_sgns_step(model, optimizer, centers, contexts, negatives):
                raise NotImplementedError

            def evaluate_sgns(model, centers, contexts, negatives):
                raise NotImplementedError
            """,
            """
            def sgns_loss(positive_logits, negative_logits):
                positive_term = F.logsigmoid(positive_logits)
                negative_term = F.logsigmoid(-negative_logits).sum(dim=1)
                return -(positive_term + negative_term).mean()

            def train_sgns_step(model, optimizer, centers, contexts, negatives):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                positive, negative = model(centers, contexts, negatives)
                loss = sgns_loss(positive, negative)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_sgns(model, centers, contexts, negatives):
                model.eval()
                with torch.no_grad():
                    positive, negative = model(centers, contexts, negatives)
                return float(positive.mean()), float(negative.mean())

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(
                    train_sgns_step(
                        model,
                        optimizer,
                        train_centers,
                        train_contexts,
                        train_negatives,
                    )
                )
            heldout_positive_score, heldout_noise_score = evaluate_sgns(
                model,
                test_centers,
                test_contexts,
                test_negatives,
            )
            assert min(losses[3:]) < losses[0]
            assert heldout_positive_score > heldout_noise_score
            figure, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(losses)
            axes[0].set_title("SGNS loss")
            axes[1].bar(
                ["positive", "noise"],
                [heldout_positive_score, heldout_noise_score],
            )
            axes[1].set_title("held-out dot scores")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


SEQ2SEQ = _paper(
    number=1,
    slug="seq2seq",
    short_title="Sequence to Sequence",
    paper_title="Sequence to Sequence Learning with Neural Networks",
    authors="Ilya Sutskever, Oriol Vinyals, Quoc V. Le",
    year=2014,
    primary_url="https://arxiv.org/abs/1409.3215",
    venue="NeurIPS 2014",
    reproduction_goal="encoder fixed vector, teacher forcing decoder, reversed source를 재현한다.",
    original_scale=(
        "4-layer 1,000-unit LSTM과 WMT 12M sentence pairs 대신 1-layer "
        "GRU/60쌍을 쓰고, 48쌍은 학습에 12쌍은 평가에만 사용한다."
    ),
    mappings=(
        (
            "§2 Eq. (1): conditional sequence probability",
            "Task 2 teacher forcing",
            "token logits",
        ),
        ("§2 Fig. 1: encoder vector→decoder", "Task 1–2", "hidden shape"),
        ("§3.3: reverse source sentence", "Task 1", "순서 assert"),
        ("§4: translation evaluation", "Task 3", "token accuracy"),
    ),
    cells=(
        _concept(
            r"p(y_1,\ldots,y_{T'}\mid x)=\prod_{t=1}^{T'}p(y_t\mid y_{<t},v)",
            "x는 source, v는 encoder 마지막 state, y_<t는 이전 target token이다.",
            "Task 1이 reverse/decoder input, Task 2가 encoder-decoder, Task 3이 token CE를 구현한다.",
            "source [B,3] → hidden [1,B,24] → teacher-forced logits [B,3,V].",
            "beam search와 깊은 LSTM은 제외하지만 고정 context와 조건부 factorization은 보존한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정, source reverse, BOS를 붙인 decoder input을 구현하세요.
            @dataclass
            class Seq2SeqConfig:
                pass

            def reverse_non_padding(sequences, pad_id):
                raise NotImplementedError

            def make_decoder_inputs(targets, bos_id):
                raise NotImplementedError
            """,
            """
            @dataclass
            class Seq2SeqConfig:
                embedding_dim: int = 16
                hidden_dim: int = 24
                learning_rate: float = 0.025
                steps: int = 24

            def reverse_non_padding(sequences, pad_id):
                reversed_rows = []
                for row in sequences:
                    valid = row[row != pad_id].flip(dims=(0,))
                    padding = row.new_full((len(row) - len(valid),), pad_id)
                    reversed_rows.append(torch.cat((valid, padding)))
                return torch.stack(reversed_rows)

            def make_decoder_inputs(targets, bos_id):
                bos = targets.new_full((len(targets), 1), bos_id)
                return torch.cat((bos, targets[:, :-1]), dim=1)

            config = Seq2SeqConfig()
            split_generator = torch.Generator().manual_seed(21)
            translation_order = torch.randperm(
                len(source_ids),
                generator=split_generator,
            ).to(source_ids.device)
            train_translation_index = translation_order[:48]
            test_translation_index = translation_order[48:]
            assert not bool(
                torch.isin(
                    train_translation_index,
                    test_translation_index,
                ).any()
            )
            train_reversed_source = reverse_non_padding(
                source_ids[train_translation_index],
                PAD,
            )
            test_reversed_source = reverse_non_padding(
                source_ids[test_translation_index],
                PAD,
            )
            train_targets = target_ids[train_translation_index]
            test_targets = target_ids[test_translation_index]
            train_decoder_inputs = make_decoder_inputs(train_targets, BOS)
            test_decoder_inputs = make_decoder_inputs(test_targets, BOS)
            assert train_reversed_source.shape == (48, 3)
            assert test_reversed_source.shape == (12, 3)
            assert torch.equal(
                train_decoder_inputs[:, 0],
                train_decoder_inputs.new_full((48,), BOS),
            )
            assert torch.equal(
                test_decoder_inputs[:, 0],
                test_decoder_inputs.new_full((12,), BOS),
            )
            """,
            "todo",
            "preprocessing",
        ),
        code(
            """
            # TODO 2: encoder context를 decoder initial state로 넘기는 Seq2SeqModel을 구현하세요.
            class Seq2SeqModel(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def encode(self, source):
                    '''[B, S] source에서 [1, B, H] context를 만든다.'''
                    raise NotImplementedError

                def decode(self, decoder_inputs, context):
                    '''Context로 초기화한 decoder logits [B, T, V]를 만든다.'''
                    raise NotImplementedError

                def forward(self, source, decoder_inputs):
                    raise NotImplementedError
            """,
            """
            class Seq2SeqModel(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
                    self.encoder = nn.GRU(
                        config.embedding_dim,
                        config.hidden_dim,
                        batch_first=True,
                    )
                    self.decoder = nn.GRU(
                        config.embedding_dim,
                        config.hidden_dim,
                        batch_first=True,
                    )
                    self.output_projection = nn.Linear(config.hidden_dim, vocab_size)

                def encode(self, source):
                    embedded = self.embedding(source)
                    _, context = self.encoder(embedded)
                    return context

                def decode(self, decoder_inputs, context):
                    embedded = self.embedding(decoder_inputs)
                    decoded, _ = self.decoder(embedded, context)
                    return self.output_projection(decoded)

                def forward(self, source, decoder_inputs):
                    context = self.encode(source)
                    return self.decode(decoder_inputs, context)

            model = Seq2SeqModel(VOCAB_SIZE, config).to(DEVICE)
            context = model.encode(train_reversed_source)
            logits = model(train_reversed_source, train_decoder_inputs)
            assert context.shape == (1, 48, 24)
            assert logits.shape == (48, 3, VOCAB_SIZE)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: sequence CE, train step, token accuracy를 구현하세요.
            def sequence_cross_entropy(logits, targets, pad_id):
                raise NotImplementedError

            def train_seq2seq_step(model, optimizer, source, decoder_inputs, targets):
                raise NotImplementedError

            def token_accuracy(model, source, decoder_inputs, targets):
                raise NotImplementedError
            """,
            """
            def sequence_cross_entropy(logits, targets, pad_id):
                return F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=pad_id,
                )

            def train_seq2seq_step(model, optimizer, source, decoder_inputs, targets):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(source, decoder_inputs)
                loss = sequence_cross_entropy(logits, targets, PAD)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def token_accuracy(model, source, decoder_inputs, targets):
                model.eval()
                with torch.no_grad():
                    predictions = model(source, decoder_inputs).argmax(dim=-1)
                valid = targets != PAD
                return float((predictions[valid] == targets[valid]).float().mean())

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(
                    train_seq2seq_step(
                        model,
                        optimizer,
                        train_reversed_source,
                        train_decoder_inputs,
                        train_targets,
                    )
                )
            heldout_accuracy = token_accuracy(
                model,
                test_reversed_source,
                test_decoder_inputs,
                test_targets,
            )
            assert min(losses[3:]) < losses[0]
            assert 0.0 <= heldout_accuracy <= 1.0
            plt.figure(figsize=(5, 3))
            plt.plot(losses)
            plt.title(f"Seq2Seq held-out token accuracy={heldout_accuracy:.1%}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


BAHDANAU = _paper(
    number=2,
    slug="bahdanau_attention",
    short_title="Bahdanau Attention",
    paper_title="Neural Machine Translation by Jointly Learning to Align and Translate",
    authors="Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio",
    year=2014,
    primary_url="https://arxiv.org/abs/1409.0473",
    venue="ICLR 2015",
    reproduction_goal="additive alignment energy, softmax weights, context vector를 재현한다.",
    original_scale=(
        "bidirectional 1,000-unit GRU와 WMT 대신 24-unit GRU와 60 sentence "
        "pairs를 쓰고, 고정된 48/12 train/evaluation split을 둔다."
    ),
    mappings=(
        ("§2.2.1 Eq. (5): ci=Σ αij hj", "Task 1", "weights sum=1"),
        ("§2.2.1 Eq. (6): alignment softmax", "Task 1", "probability assert"),
        ("§2.2.1 Eq. (7): a(si-1,hj)", "Task 1 AdditiveAttention", "energy shape"),
        ("§3.1 Eq. (8–9): bidirectional annotations", "Task 2–3", "attention heatmap"),
    ),
    cells=(
        _concept(
            (
                r"e_{ij}=v_a^{\top}\tanh(W_as_{i-1}+U_ah_j),\quad "
                r"\alpha_{ij}=\frac{\exp e_{ij}}{\sum_k\exp e_{ik}},\quad "
                r"c_i=\sum_j\alpha_{ij}h_j"
            ),
            "s는 decoder state, h는 encoder annotation, e는 energy, α는 alignment, c는 context다.",
            "Task 1의 AdditiveAttention이 세 식, Task 2가 step decoder, Task 3이 CE/heatmap을 맡는다.",
            "encoder [B,3,24], query [B,24] → weights [B,3] → context [B,24].",
            "양방향 encoder 대신 단방향 GRU를 쓰되 동적 context와 alignment normalization은 같다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정과 Eq. (5–7)을 수행하는 AdditiveAttention class를 구현하세요.
            @dataclass
            class AttentionConfig:
                pass

            class AdditiveAttention(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()

                def energies(self, query, annotations):
                    '''Bahdanau Eq. (4)의 energy [B, S]를 계산한다.'''
                    raise NotImplementedError

                def forward(self, query, annotations, valid_mask=None):
                    raise NotImplementedError
            """,
            """
            @dataclass
            class AttentionConfig:
                embedding_dim: int = 16
                hidden_dim: int = 24
                learning_rate: float = 0.025
                steps: int = 26

            class AdditiveAttention(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
                    self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
                    self.energy_projection = nn.Linear(hidden_dim, 1, bias=False)

                def energies(self, query, annotations):
                    projected_query = self.query_projection(query)[:, None]
                    projected_keys = self.key_projection(annotations)
                    return self.energy_projection(
                        torch.tanh(projected_query + projected_keys)
                    ).squeeze(-1)

                def forward(self, query, annotations, valid_mask=None):
                    energy = self.energies(query, annotations)
                    if valid_mask is not None:
                        energy = energy.masked_fill(~valid_mask, float("-inf"))
                    weights = energy.softmax(dim=-1)
                    context = torch.bmm(weights[:, None], annotations).squeeze(1)
                    return context, weights

            config = AttentionConfig()
            attention = AdditiveAttention(config.hidden_dim).to(DEVICE)
            query_probe = torch.randn(2, 24, device=DEVICE)
            annotation_probe = torch.randn(2, 3, 24, device=DEVICE)
            context_probe, weight_probe = attention(query_probe, annotation_probe)
            assert context_probe.shape == (2, 24)
            assert torch.allclose(weight_probe.sum(dim=1), weight_probe.new_ones(2))
            """,
            "todo",
            "attention",
            "equation",
        ),
        code(
            """
            # TODO 2: 매 decoding step에서 context를 다시 계산하는 AttentiveSeq2Seq를 구현하세요.
            class AttentiveSeq2Seq(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def encode(self, source):
                    '''Encoder annotations [B, S, H]를 반환한다.'''
                    raise NotImplementedError

                def decode_step(self, token, state, annotations, valid_mask):
                    '''Context를 다시 계산한 one-step logits를 반환한다.'''
                    raise NotImplementedError

                def forward(self, source, decoder_inputs):
                    raise NotImplementedError
            """,
            """
            class AttentiveSeq2Seq(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
                    self.encoder = nn.GRU(
                        config.embedding_dim,
                        config.hidden_dim,
                        batch_first=True,
                    )
                    self.attention = AdditiveAttention(config.hidden_dim)
                    decoder_input_dim = config.embedding_dim + config.hidden_dim
                    self.decoder_cell = nn.GRUCell(decoder_input_dim, config.hidden_dim)
                    self.output = nn.Linear(config.hidden_dim * 2, vocab_size)

                def encode(self, source):
                    annotations, final_state = self.encoder(self.embedding(source))
                    return annotations, final_state.squeeze(0)

                def decode_step(self, token, state, annotations, valid_mask):
                    context, weights = self.attention(state, annotations, valid_mask)
                    embedded = self.embedding(token)
                    state = self.decoder_cell(torch.cat((embedded, context), dim=-1), state)
                    logits = self.output(torch.cat((state, context), dim=-1))
                    return logits, state, weights

                def forward(self, source, decoder_inputs):
                    annotations, state = self.encode(source)
                    valid_mask = source != PAD
                    logits_by_step = []
                    weights_by_step = []
                    for step in range(decoder_inputs.shape[1]):
                        logits, state, weights = self.decode_step(
                            decoder_inputs[:, step],
                            state,
                            annotations,
                            valid_mask,
                        )
                        logits_by_step.append(logits)
                        weights_by_step.append(weights)
                    return torch.stack(logits_by_step, 1), torch.stack(weights_by_step, 1)

            split_generator = torch.Generator().manual_seed(22)
            translation_order = torch.randperm(
                len(source_ids),
                generator=split_generator,
            ).to(source_ids.device)
            train_translation_index = translation_order[:48]
            test_translation_index = translation_order[48:]
            assert not bool(
                torch.isin(
                    train_translation_index,
                    test_translation_index,
                ).any()
            )
            train_source = source_ids[train_translation_index]
            test_source = source_ids[test_translation_index]
            train_targets = target_ids[train_translation_index]
            test_targets = target_ids[test_translation_index]
            train_decoder_inputs = torch.cat(
                (train_targets.new_full((48, 1), BOS), train_targets[:, :-1]),
                dim=1,
            )
            test_decoder_inputs = torch.cat(
                (test_targets.new_full((12, 1), BOS), test_targets[:, :-1]),
                dim=1,
            )
            model = AttentiveSeq2Seq(VOCAB_SIZE, config).to(DEVICE)
            logits, alignment = model(train_source, train_decoder_inputs)
            assert logits.shape == (48, 3, VOCAB_SIZE)
            assert alignment.shape == (48, 3, 3)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: attention seq2seq loss, train step, token 평가/heatmap을 구현하세요.
            def attentive_loss(logits, targets):
                raise NotImplementedError

            def train_attention_step(model, optimizer, source, decoder_inputs, targets):
                raise NotImplementedError

            def evaluate_attention(model, source, decoder_inputs, targets):
                raise NotImplementedError
            """,
            """
            def attentive_loss(logits, targets):
                return F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=PAD,
                )

            def train_attention_step(model, optimizer, source, decoder_inputs, targets):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(source, decoder_inputs)
                loss = attentive_loss(logits, targets)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_attention(model, source, decoder_inputs, targets):
                model.eval()
                with torch.no_grad():
                    logits, weights = model(source, decoder_inputs)
                valid = targets != PAD
                accuracy = (logits.argmax(-1)[valid] == targets[valid]).float().mean()
                return float(accuracy), weights

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(
                    train_attention_step(
                        model,
                        optimizer,
                        train_source,
                        train_decoder_inputs,
                        train_targets,
                    )
                )
            heldout_accuracy, heldout_weights = evaluate_attention(
                model,
                test_source,
                test_decoder_inputs,
                test_targets,
            )
            assert min(losses[3:]) < losses[0]
            assert torch.allclose(
                heldout_weights.sum(dim=-1),
                heldout_weights.new_ones(12, 3),
            )
            plt.figure(figsize=(4, 3))
            plt.imshow(heldout_weights[0].detach().cpu())
            plt.xlabel("source position")
            plt.ylabel("target position")
            plt.title(f"held-out alignment, token acc={heldout_accuracy:.1%}")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


TRANSFORMER = _paper(
    number=3,
    slug="attention_is_all_you_need",
    short_title="Transformer",
    paper_title="Attention Is All You Need",
    authors="Ashish Vaswani et al.",
    year=2017,
    primary_url="https://arxiv.org/abs/1706.03762",
    venue="NeurIPS 2017",
    reproduction_goal="scaled dot-product, multi-head 분할, sinusoidal position을 직접 재현한다.",
    original_scale="6-layer encoder-decoder, WMT 대신 1 encoder block과 sentiment 분류를 사용한다.",
    mappings=(
        ("§3.2.1 Eq. (1): softmax(QKᵀ/√dk)V", "Task 1–2", "row sum/mask"),
        ("§3.2.2: Multi-Head Attention", "Task 2", "4 heads"),
        ("§3.4: embeddings", "Task 2", "shared sequence tensors"),
        ("§3.5 Eq. (3): positional encoding", "Task 1", "sin/cos values"),
    ),
    cells=(
        _concept(
            (
                r"\operatorname{Attention}(Q,K,V)="
                r"\operatorname{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V"
            ),
            "Q/K/V는 query/key/value, d_k는 head 차원, softmax는 key 축 정규화다.",
            "Task 1이 attention/position 함수, Task 2가 MultiHeadAttention과 encoder를 구현한다.",
            "token [B,4] → embedding [B,4,24] → 4 heads [B,4,4,6] → logits [B,2].",
            "decoder/cross-attention은 제외하고 논문의 encoder self-attention 핵심을 직접 구현한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정, scaled dot-product attention, sinusoidal position을 구현하세요.
            @dataclass
            class TransformerConfig:
                pass

            def scaled_dot_product_attention(query, key, value, mask=None):
                raise NotImplementedError

            def sinusoidal_positions(length, dimension, device):
                raise NotImplementedError
            """,
            """
            @dataclass
            class TransformerConfig:
                model_dim: int = 24
                heads: int = 4
                feedforward_dim: int = 48
                num_classes: int = 2
                learning_rate: float = 0.018
                steps: int = 18

            def scaled_dot_product_attention(query, key, value, mask=None):
                scale = math.sqrt(query.shape[-1])
                scores = torch.matmul(query, key.transpose(-2, -1)) / scale
                if mask is not None:
                    scores = scores.masked_fill(~mask, float("-inf"))
                weights = scores.softmax(dim=-1)
                return torch.matmul(weights, value), weights

            def sinusoidal_positions(length, dimension, device):
                positions = torch.arange(length, device=device).float()[:, None]
                even_indices = torch.arange(0, dimension, 2, device=device).float()
                frequencies = torch.exp(-math.log(10000.0) * even_indices / dimension)
                encoding = torch.zeros(length, dimension, device=device)
                encoding[:, 0::2] = torch.sin(positions * frequencies)
                encoding[:, 1::2] = torch.cos(positions * frequencies)
                return encoding

            config = TransformerConfig()
            position = sinusoidal_positions(4, config.model_dim, DEVICE)
            assert position.shape == (4, 24)
            assert torch.allclose(position[0, 0::2], position.new_zeros(12))
            """,
            "todo",
            "equation",
        ),
        code(
            """
            # TODO 2: Q/K/V projection, head split/merge, residual encoder를 구현하세요.
            class MultiHeadSelfAttention(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def split_heads(self, tensor):
                    '''[B, T, D]를 [B, H, T, d_h]로 변환한다.'''
                    raise NotImplementedError

                def merge_heads(self, tensor):
                    '''[B, H, T, d_h]를 [B, T, D]로 병합한다.'''
                    raise NotImplementedError

                def forward(self, inputs, mask=None):
                    raise NotImplementedError

            class TransformerClassifier(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def encode(self, token_ids):
                    '''Residual encoder hidden states [B, T, D]를 반환한다.'''
                    raise NotImplementedError

                def forward(self, token_ids):
                    raise NotImplementedError
            """,
            """
            class MultiHeadSelfAttention(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.heads = config.heads
                    self.head_dim = config.model_dim // config.heads
                    self.query = nn.Linear(config.model_dim, config.model_dim)
                    self.key = nn.Linear(config.model_dim, config.model_dim)
                    self.value = nn.Linear(config.model_dim, config.model_dim)
                    self.output = nn.Linear(config.model_dim, config.model_dim)

                def split_heads(self, tensor):
                    batch, length, _ = tensor.shape
                    tensor = tensor.view(batch, length, self.heads, self.head_dim)
                    return tensor.transpose(1, 2)

                def merge_heads(self, tensor):
                    batch, _, length, _ = tensor.shape
                    tensor = tensor.transpose(1, 2).contiguous()
                    return tensor.view(batch, length, self.heads * self.head_dim)

                def forward(self, inputs, mask=None):
                    query = self.split_heads(self.query(inputs))
                    key = self.split_heads(self.key(inputs))
                    value = self.split_heads(self.value(inputs))
                    attended, weights = scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        mask,
                    )
                    return self.output(self.merge_heads(attended)), weights

            class TransformerClassifier(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, config.model_dim)
                    self.attention = MultiHeadSelfAttention(config)
                    self.norm1 = nn.LayerNorm(config.model_dim)
                    self.feedforward = nn.Sequential(
                        nn.Linear(config.model_dim, config.feedforward_dim),
                        nn.ReLU(),
                        nn.Linear(config.feedforward_dim, config.model_dim),
                    )
                    self.norm2 = nn.LayerNorm(config.model_dim)
                    self.classifier = nn.Linear(config.model_dim, config.num_classes)

                def encode(self, token_ids):
                    embedded = self.embedding(token_ids)
                    embedded = embedded + sinusoidal_positions(
                        token_ids.shape[1],
                        embedded.shape[-1],
                        token_ids.device,
                    )
                    attended, weights = self.attention(embedded)
                    hidden = self.norm1(embedded + attended)
                    hidden = self.norm2(hidden + self.feedforward(hidden))
                    return hidden, weights

                def forward(self, token_ids):
                    hidden, _ = self.encode(token_ids)
                    return self.classifier(hidden.mean(dim=1))

            model = TransformerClassifier(VOCAB_SIZE, config).to(DEVICE)
            hidden, attention_weights = model.encode(sentence_ids[:2])
            assert hidden.shape == (2, 4, 24)
            assert attention_weights.shape == (2, 4, 4, 4)
            assert model(sentence_ids[:2]).shape == (2, 2)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: 분류 손실, train step, test accuracy를 구현하세요.
            def transformer_loss(logits, targets):
                raise NotImplementedError

            def train_transformer_step(model, optimizer, inputs, targets):
                raise NotImplementedError

            def evaluate_transformer(model, inputs, targets):
                raise NotImplementedError
            """,
            """
            def transformer_loss(logits, targets):
                return F.cross_entropy(logits, targets)

            def train_transformer_step(model, optimizer, inputs, targets):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = transformer_loss(model(inputs), targets)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_transformer(model, inputs, targets):
                model.eval()
                with torch.no_grad():
                    predictions = model(inputs).argmax(dim=1)
                return float((predictions == targets).float().mean())

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(
                    train_transformer_step(
                        model,
                        optimizer,
                        sentence_ids[:240],
                        sentiment_labels[:240],
                    )
                )
            accuracy = evaluate_transformer(
                model,
                sentence_ids[240:],
                sentiment_labels[240:],
            )
            assert min(losses[3:]) < losses[0]
            assert 0.0 <= accuracy <= 1.0
            plt.figure(figsize=(5, 3))
            plt.plot(losses)
            plt.title(f"Transformer accuracy={accuracy:.1%}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


GPT1 = _paper(
    number=4,
    slug="gpt1",
    short_title="GPT-1",
    paper_title="Improving Language Understanding by Generative Pre-Training",
    authors="Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever",
    year=2018,
    primary_url=(
        "https://cdn.openai.com/research-covers/language-unsupervised/"
        "language_understanding_paper.pdf"
    ),
    venue="OpenAI technical report",
    reproduction_goal="causal LM pretraining과 supervised+auxiliary LM fine-tuning을 재현한다.",
    original_scale=(
        "12-layer decoder와 BooksCorpus 대신 1-block decoder와 360개 행을 "
        "사용한다. 로컬 corpus는 6개 문장 template을 반복하므로 240/120 "
        "row split은 입력 행을 격리하지만 새로운 언어 분포의 일반화를 뜻하지 않는다."
    ),
    mappings=(
        ("§2 Eq. (1): L1(U) autoregressive LM", "Task 2–3", "LM loss"),
        ("§2 Eq. (2): Transformer decoder", "Task 1–2", "causal mask"),
        ("§2 Eq. (3): supervised objective L2", "Task 3 classifier", "CE loss"),
        ("§2 Eq. (4), Fig. 1: L2+λL1", "Task 3", "두 loss plot"),
    ),
    cells=(
        _concept(
            r"L_1(U)=\sum_i\log P(u_i\mid u_{i-k},\ldots,u_{i-1}),\quad L_3=L_2+\lambda L_1",
            "L1은 causal LM, L2는 supervised loss, λ는 fine-tuning 중 LM 보조 가중치다.",
            "Task 1이 causal mask, Task 2가 decoder/두 head, Task 3이 pretrain→fine-tune을 구현한다.",
            "LM input [B,4] → hidden [B,4,24] → LM [B,4,V], classifier [B,2].",
            (
                "대규모 corpus와 12-layer는 축소하지만 두 단계 objective와 causal "
                "제약은 보존한다. 여섯 template 반복이라는 holdout 한계도 함께 기록한다."
            ),
        ),
        _setup(),
        code(
            """
            # TODO 1: GPT 설정과 future token을 가리는 causal mask를 구현하세요.
            @dataclass
            class GPTConfig:
                pass

            def causal_attention_mask(length, device):
                raise NotImplementedError

            def shift_right(tokens, bos_id):
                raise NotImplementedError
            """,
            """
            @dataclass
            class GPTConfig:
                model_dim: int = 24
                heads: int = 4
                feedforward_dim: int = 48
                auxiliary_weight: float = 0.2
                pretrain_steps: int = 14
                finetune_steps: int = 12
                learning_rate: float = 0.02

            def causal_attention_mask(length, device):
                positions = torch.arange(length, device=device)
                return positions[None, None, :, None] >= positions[None, None, None, :]

            def shift_right(tokens, bos_id):
                bos = tokens.new_full((len(tokens), 1), bos_id)
                return torch.cat((bos, tokens[:, :-1]), dim=1)

            config = GPTConfig()
            downstream_train_index = torch.arange(
                240,
                device=sentence_ids.device,
            )
            downstream_test_index = torch.arange(
                240,
                len(sentence_ids),
                device=sentence_ids.device,
            )
            pretrain_index = downstream_train_index.clone()
            assert not bool(
                torch.isin(pretrain_index, downstream_test_index).any()
            )
            assert not bool(
                torch.isin(
                    downstream_train_index,
                    downstream_test_index,
                ).any()
            )
            pretrain_tokens = sentence_ids[pretrain_index]
            finetune_tokens = sentence_ids[downstream_train_index]
            evaluation_tokens = sentence_ids[downstream_test_index]
            pretrain_inputs = shift_right(pretrain_tokens, BOS)
            finetune_inputs = shift_right(finetune_tokens, BOS)
            evaluation_inputs = shift_right(evaluation_tokens, BOS)
            finetune_labels = sentiment_labels[downstream_train_index]
            evaluation_labels = sentiment_labels[downstream_test_index]
            causal_mask = causal_attention_mask(sentence_ids.shape[1], DEVICE)
            assert causal_mask.shape == (1, 1, 4, 4)
            assert not bool(causal_mask[0, 0, 0, 1])
            assert pretrain_inputs.shape == finetune_inputs.shape == (240, 4)
            assert evaluation_inputs.shape == (120, 4)
            """,
            "todo",
            "causality",
        ),
        code(
            """
            # TODO 2: causal self-attention과 LM/classification head를 구현하세요.
            class CausalSelfAttention(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def split_heads(self, tensor):
                    '''[B, T, D]를 causal heads [B, H, T, d_h]로 변환한다.'''
                    raise NotImplementedError

                def forward(self, hidden):
                    raise NotImplementedError

            class GPT1Mini(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def hidden_states(self, token_ids):
                    '''공유 Transformer hidden states [B, T, D]를 반환한다.'''
                    raise NotImplementedError

                def forward(self, token_ids):
                    raise NotImplementedError
            """,
            """
            class CausalSelfAttention(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.heads = config.heads
                    self.head_dim = config.model_dim // config.heads
                    self.qkv = nn.Linear(config.model_dim, config.model_dim * 3)
                    self.output = nn.Linear(config.model_dim, config.model_dim)

                def split_heads(self, tensor):
                    batch, length, _ = tensor.shape
                    tensor = tensor.view(batch, length, self.heads, self.head_dim)
                    return tensor.transpose(1, 2)

                def forward(self, hidden):
                    query, key, value = self.qkv(hidden).chunk(3, dim=-1)
                    query = self.split_heads(query)
                    key = self.split_heads(key)
                    value = self.split_heads(value)
                    mask = causal_attention_mask(hidden.shape[1], hidden.device)
                    scores = torch.matmul(query, key.transpose(-2, -1))
                    scores = scores / math.sqrt(self.head_dim)
                    weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
                    attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
                    attended = attended.view(len(hidden), hidden.shape[1], -1)
                    return self.output(attended), weights

            class GPT1Mini(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, config.model_dim)
                    self.position = nn.Embedding(4, config.model_dim)
                    self.attention = CausalSelfAttention(config)
                    self.norm1 = nn.LayerNorm(config.model_dim)
                    self.mlp = nn.Sequential(
                        nn.Linear(config.model_dim, config.feedforward_dim),
                        nn.GELU(),
                        nn.Linear(config.feedforward_dim, config.model_dim),
                    )
                    self.norm2 = nn.LayerNorm(config.model_dim)
                    self.lm_head = nn.Linear(config.model_dim, vocab_size)
                    self.classifier = nn.Linear(config.model_dim, 2)

                def hidden_states(self, token_ids):
                    positions = torch.arange(token_ids.shape[1], device=token_ids.device)
                    hidden = self.embedding(token_ids) + self.position(positions)[None]
                    attended, _ = self.attention(hidden)
                    hidden = self.norm1(hidden + attended)
                    return self.norm2(hidden + self.mlp(hidden))

                def forward(self, token_ids):
                    hidden = self.hidden_states(token_ids)
                    return self.lm_head(hidden), self.classifier(hidden[:, -1])

            model = GPT1Mini(VOCAB_SIZE, config).to(DEVICE)
            lm_logits, class_logits = model(pretrain_inputs)
            assert lm_logits.shape == (240, 4, VOCAB_SIZE)
            assert class_logits.shape == (240, 2)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: LM/supervised loss, pretrain step, fine-tune step, 평가를 구현하세요.
            def language_model_loss(logits, targets):
                raise NotImplementedError

            def pretrain_step(model, optimizer, inputs, targets):
                raise NotImplementedError

            def finetune_step(model, optimizer, inputs, tokens, labels, config):
                raise NotImplementedError

            def evaluate_gpt(model, inputs, labels):
                raise NotImplementedError
            """,
            """
            def language_model_loss(logits, targets):
                return F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )

            def pretrain_step(model, optimizer, inputs, targets):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lm_logits, _ = model(inputs)
                loss = language_model_loss(lm_logits, targets)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def finetune_step(model, optimizer, inputs, tokens, labels, config):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lm_logits, class_logits = model(inputs)
                supervised = F.cross_entropy(class_logits, labels)
                auxiliary = language_model_loss(lm_logits, tokens)
                total = supervised + config.auxiliary_weight * auxiliary
                total.backward()
                optimizer.step()
                return float(supervised.detach())

            def evaluate_gpt(model, inputs, labels):
                model.eval()
                with torch.no_grad():
                    _, class_logits = model(inputs)
                return float((class_logits.argmax(1) == labels).float().mean())

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            pretrain_losses = []
            for step in range(config.pretrain_steps):
                pretrain_losses.append(
                    pretrain_step(
                        model,
                        optimizer,
                        pretrain_inputs,
                        pretrain_tokens,
                    )
                )
            finetune_losses = []
            for step in range(config.finetune_steps):
                finetune_losses.append(
                    finetune_step(
                        model,
                        optimizer,
                        finetune_inputs,
                        finetune_tokens,
                        finetune_labels,
                        config,
                    )
                )
            accuracy = evaluate_gpt(
                model,
                evaluation_inputs,
                evaluation_labels,
            )
            assert min(pretrain_losses[3:]) < pretrain_losses[0]
            assert min(finetune_losses[3:]) < finetune_losses[0]
            plt.figure(figsize=(6, 3))
            plt.plot(pretrain_losses, label="LM pretrain")
            offset = len(pretrain_losses)
            x_axis = range(offset, offset + len(finetune_losses))
            plt.plot(x_axis, finetune_losses, label="supervised")
            plt.title(f"GPT-1 held-out row accuracy={accuracy:.1%}")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


BERT = _paper(
    number=5,
    slug="bert",
    short_title="BERT",
    paper_title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    authors="Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
    year=2018,
    primary_url="https://arxiv.org/abs/1810.04805",
    venue="NAACL 2019",
    reproduction_goal="80/10/10 MLM corruption과 bidirectional encoder를 재현한다.",
    original_scale=(
        "BERT Base/Large와 BooksCorpus+Wikipedia 대신 1 encoder block을 쓰고, "
        "300개 train 행과 60개 held-out 행을 서로 다른 seed로 손상한다."
    ),
    mappings=(
        (
            "§3.1 Task #1: MLM and 80% mask/10% random/10% unchanged",
            "Task 1",
            "train/evaluation별 selected labels와 정확한 80/10/10 corruption",
        ),
        (
            "Fig. 1 and §3 Fig. 2: bidirectional attention and three embeddings",
            "Task 2 explicit unmasked Q/K/V attention",
            "[B,H,T,T] positive attention weights",
        ),
        ("§3.1 Task #1: MLM objective", "Task 3", "loss and masked-token accuracy"),
    ),
    cells=(
        _concept(
            r"L_{MLM}=-\sum_{i\in\mathcal{M}}\log p(x_i\mid x_{\setminus\mathcal{M}})",
            "M은 선택된 position, x_i는 원 token, 양쪽 context가 조건으로 사용된다.",
            (
                "Task 1이 80/10/10 corruption, Task 2가 mask 없는 Q/K/V 양방향 "
                "encoder, Task 3이 MLM CE를 구현한다."
            ),
            (
                "train corruption [300,4], held-out corruption [60,4]이며, "
                "10-row probe는 hidden [10,4,24]와 logits [10,4,V]를 만든다."
            ),
            "NSP와 대규모 corpus는 제외하지만 masking 정책과 비인과적 self-attention은 보존한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: BERT 설정과 선택 position에 8/1/1 corruption을 적용하세요.
            @dataclass
            class BERTConfig:
                pass

            def apply_mlm_corruption(
                tokens,
                mask_id,
                random_token_ids,
                generator,
            ):
                raise NotImplementedError
            """,
            """
            @dataclass
            class BERTConfig:
                model_dim: int = 24
                heads: int = 4
                feedforward_dim: int = 48
                learning_rate: float = 0.025
                steps: int = 32

            def apply_mlm_corruption(
                tokens,
                mask_id,
                random_token_ids,
                generator,
            ):
                corrupted = tokens.clone()
                labels = torch.full_like(corrupted, -100)
                batch_size, sequence_length = tokens.shape
                rows = torch.arange(batch_size, device=tokens.device)
                positions = torch.randint(
                    sequence_length,
                    (batch_size,),
                    generator=generator,
                ).to(tokens.device)
                labels[rows, positions] = corrupted[rows, positions]

                corruption_order = torch.randperm(
                    batch_size,
                    generator=generator,
                ).to(tokens.device)
                mask_count = round(0.8 * batch_size)
                random_count = round(0.1 * batch_size)
                mask_rows = corruption_order[:mask_count]
                random_rows = corruption_order[
                    mask_count:mask_count + random_count
                ]
                corrupted[mask_rows, positions[mask_rows]] = mask_id
                random_offsets = torch.randint(
                    len(random_token_ids),
                    (random_count,),
                    generator=generator,
                ).to(tokens.device)
                corrupted[random_rows, positions[random_rows]] = (
                    random_token_ids[random_offsets]
                )
                return corrupted, labels

            config = BERTConfig()
            split_generator = torch.Generator().manual_seed(31)
            sentence_order = torch.randperm(
                len(sentence_ids),
                generator=split_generator,
            ).to(sentence_ids.device)
            train_sentence_index = sentence_order[:300]
            test_sentence_index = sentence_order[300:]
            assert not bool(
                torch.isin(train_sentence_index, test_sentence_index).any()
            )
            random_token_ids = torch.tensor(
                [stoi[word] for word in words],
                device=sentence_ids.device,
            )
            train_corrupted, train_mlm_labels = apply_mlm_corruption(
                sentence_ids[train_sentence_index],
                MASK,
                random_token_ids,
                torch.Generator().manual_seed(41),
            )
            test_corrupted, test_mlm_labels = apply_mlm_corruption(
                sentence_ids[test_sentence_index],
                MASK,
                random_token_ids,
                torch.Generator().manual_seed(42),
            )
            train_selected = train_mlm_labels != -100
            test_selected = test_mlm_labels != -100
            assert int(train_selected.sum()) == 300
            assert int(test_selected.sum()) == 60
            assert int((train_corrupted[train_selected] == MASK).sum()) == 240
            assert int((test_corrupted[test_selected] == MASK).sum()) == 48
            """,
            "todo",
            "corruption",
        ),
        code(
            """
            # TODO 2: unmasked self-attention encoder와 MLM head를 갖는 BERTMini를 구현하세요.
            class BidirectionalSelfAttention(nn.Module):
                '''BERT Fig. 1의 mask 없는 multi-head self-attention이다.'''

                def __init__(self, config):
                    super().__init__()

                def split_heads(self, tensor):
                    '''[B, T, D]를 [B, H, T, d_h]로 분할한다.'''
                    raise NotImplementedError

                def merge_heads(self, tensor):
                    '''[B, H, T, d_h]를 [B, T, D]로 병합한다.'''
                    raise NotImplementedError

                def forward(self, hidden):
                    '''Mask 없는 attention output과 [B, H, T, T] weights를 반환한다.'''
                    raise NotImplementedError

            class BERTMini(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def embed(self, token_ids):
                    '''Token, position, segment embedding을 [B, T, D]로 합산한다.'''
                    raise NotImplementedError

                def forward(self, token_ids):
                    raise NotImplementedError
            """,
            """
            class BidirectionalSelfAttention(nn.Module):
                '''모든 token pair의 scaled dot-product score를 계산한다.'''

                def __init__(self, config):
                    super().__init__()
                    if config.model_dim % config.heads != 0:
                        raise ValueError("model_dim must be divisible by heads")
                    self.heads = config.heads
                    self.head_dim = config.model_dim // config.heads
                    self.query_projection = nn.Linear(
                        config.model_dim,
                        config.model_dim,
                    )
                    self.key_projection = nn.Linear(
                        config.model_dim,
                        config.model_dim,
                    )
                    self.value_projection = nn.Linear(
                        config.model_dim,
                        config.model_dim,
                    )
                    self.output_projection = nn.Linear(
                        config.model_dim,
                        config.model_dim,
                    )

                def split_heads(self, tensor):
                    batch, length, _ = tensor.shape
                    tensor = tensor.reshape(
                        batch,
                        length,
                        self.heads,
                        self.head_dim,
                    )
                    return tensor.transpose(1, 2)

                def merge_heads(self, tensor):
                    batch, _, length, _ = tensor.shape
                    tensor = tensor.transpose(1, 2).contiguous()
                    return tensor.reshape(
                        batch,
                        length,
                        self.heads * self.head_dim,
                    )

                def forward(self, hidden):
                    queries = self.split_heads(self.query_projection(hidden))
                    keys = self.split_heads(self.key_projection(hidden))
                    values = self.split_heads(self.value_projection(hidden))
                    scale = self.head_dim**-0.5
                    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
                    weights = scores.softmax(dim=-1)
                    context = torch.matmul(weights, values)
                    output = self.output_projection(self.merge_heads(context))
                    return output, weights

            class BERTMini(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.token_embedding = nn.Embedding(vocab_size, config.model_dim)
                    self.position_embedding = nn.Embedding(4, config.model_dim)
                    self.segment_embedding = nn.Embedding(2, config.model_dim)
                    self.attention = BidirectionalSelfAttention(config)
                    self.norm1 = nn.LayerNorm(config.model_dim)
                    self.feedforward = nn.Sequential(
                        nn.Linear(config.model_dim, config.feedforward_dim),
                        nn.GELU(),
                        nn.Linear(config.feedforward_dim, config.model_dim),
                    )
                    self.norm2 = nn.LayerNorm(config.model_dim)
                    self.mlm_head = nn.Linear(config.model_dim, vocab_size)

                def embed(self, token_ids):
                    positions = torch.arange(token_ids.shape[1], device=token_ids.device)
                    segments = torch.zeros_like(token_ids)
                    return (
                        self.token_embedding(token_ids)
                        + self.position_embedding(positions)[None]
                        + self.segment_embedding(segments)
                    )

                def forward(self, token_ids):
                    embedded = self.embed(token_ids)
                    attended, _ = self.attention(embedded)
                    hidden = self.norm1(embedded + attended)
                    hidden = self.norm2(hidden + self.feedforward(hidden))
                    return self.mlm_head(hidden)

            model = BERTMini(VOCAB_SIZE, config).to(DEVICE)
            corruption_probe = train_corrupted[:10]
            assert model.embed(corruption_probe).shape == (10, 4, 24)
            assert model(corruption_probe).shape == (10, 4, VOCAB_SIZE)
            _, bidirectional_weights = model.attention(
                model.embed(corruption_probe)
            )
            assert bidirectional_weights.shape == (10, 4, 4, 4)
            assert torch.all(bidirectional_weights > 0)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: selected-position MLM loss, train step, MLM accuracy를 구현하세요.
            def masked_language_model_loss(logits, labels):
                raise NotImplementedError

            def train_bert_step(model, optimizer, corrupted, labels):
                raise NotImplementedError

            def evaluate_mlm(model, corrupted, labels):
                raise NotImplementedError
            """,
            """
            def masked_language_model_loss(logits, labels):
                return F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            def train_bert_step(model, optimizer, corrupted, labels):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = masked_language_model_loss(model(corrupted), labels)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_mlm(model, corrupted, labels):
                model.eval()
                selected = labels != -100
                with torch.no_grad():
                    predictions = model(corrupted)[selected].argmax(dim=-1)
                return float((predictions == labels[selected]).float().mean())

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(
                    train_bert_step(
                        model,
                        optimizer,
                        train_corrupted,
                        train_mlm_labels,
                    )
                )
            heldout_accuracy = evaluate_mlm(
                model,
                test_corrupted,
                test_mlm_labels,
            )
            assert min(losses[3:]) < losses[0]
            assert heldout_accuracy >= 0.5
            plt.figure(figsize=(5, 3))
            plt.plot(losses)
            plt.title(f"BERT held-out MLM accuracy={heldout_accuracy:.1%}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


T5 = _paper(
    number=6,
    slug="t5",
    short_title="T5",
    paper_title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    authors="Colin Raffel et al.",
    year=2019,
    primary_url="https://arxiv.org/abs/1910.10683",
    venue="JMLR 2020",
    reproduction_goal="text-to-text 형식, span sentinel target, relative position bias를 재현한다.",
    original_scale=(
        "C4와 최대 11B parameter 대신 60개 3-token sequence와 작은 "
        "Transformer를 쓰며, 48개 train/12개 held-out 행을 분리한다."
    ),
    mappings=(
        (
            "§3.1.4 and Fig. 2: span replacement with sentinel tokens",
            "Task 1",
            "sentinel input/target sequence",
        ),
        (
            "§2.1: encoder-decoder attention and relative position embeddings",
            "Task 2 explicit encoder/causal decoder/cross-attention",
            "bias shape, nonzero gradient, and future invariance",
        ),
        ("§3.3.4: span-corruption objective", "Task 3", "denoising accuracy"),
    ),
    cells=(
        _concept(
            r"L_{span}=-\sum_t\log p(y_t\mid y_{<t},\widetilde{x})",
            "x-tilde는 sentinel로 span을 바꾼 입력, y는 sentinel과 제거 span을 잇는 target이다.",
            (
                "Task 1이 span pair, Task 2가 logits에 relative bias를 더하는 "
                "encoder/causal decoder/cross-attention, Task 3이 denoising CE를 맡는다."
            ),
            (
                "train source [48,2], decoder [48,4]와 held-out source [12,2]를 "
                "분리해 logits [B,4,V]와 token accuracy를 계산한다."
            ),
            "여러 span과 bucketed bias 대신 한 span/clip distance를 쓰되 text-to-text 목적은 유지한다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정과 source [a,b,c]의 sentinel span corruption을 구현하세요.
            @dataclass
            class T5Config:
                pass

            def make_span_corruption(source, sentinel_id, eos_id, bos_id):
                raise NotImplementedError
            """,
            """
            @dataclass
            class T5Config:
                model_dim: int = 24
                heads: int = 4
                feedforward_dim: int = 48
                max_distance: int = 4
                learning_rate: float = 0.02
                steps: int = 32

            def make_span_corruption(source, sentinel_id, eos_id, bos_id):
                sentinel = source.new_full((len(source),), sentinel_id)
                eos = source.new_full((len(source),), eos_id)
                corrupted = torch.stack((source[:, 0], sentinel), dim=1)
                target = torch.stack(
                    (sentinel, source[:, 1], source[:, 2], eos),
                    dim=1,
                )
                decoder_inputs = torch.cat(
                    (source.new_full((len(source), 1), bos_id), target[:, :-1]),
                    dim=1,
                )
                return corrupted, target, decoder_inputs

            config = T5Config()
            split_generator = torch.Generator().manual_seed(32)
            translation_order = torch.randperm(
                len(source_ids),
                generator=split_generator,
            ).to(source_ids.device)
            train_translation_index = translation_order[:48]
            test_translation_index = translation_order[48:]
            assert not bool(
                torch.isin(
                    train_translation_index,
                    test_translation_index,
                ).any()
            )
            train_span_input, train_span_target, train_decoder_inputs = (
                make_span_corruption(
                    source_ids[train_translation_index],
                    SENTINEL,
                    EOS,
                    BOS,
                )
            )
            test_span_input, test_span_target, test_decoder_inputs = (
                make_span_corruption(
                    source_ids[test_translation_index],
                    SENTINEL,
                    EOS,
                    BOS,
                )
            )
            assert train_span_input.shape == (48, 2)
            assert test_span_input.shape == (12, 2)
            assert train_span_target.shape == train_decoder_inputs.shape == (48, 4)
            assert test_span_target.shape == test_decoder_inputs.shape == (12, 4)
            """,
            "todo",
            "span-corruption",
        ),
        code(
            """
            # TODO 2: relative bias class와 encoder-decoder T5Mini를 구현하세요.
            class RelativePositionBias(nn.Module):
                '''Clipped relative distance를 head별 attention bias로 바꾼다.'''

                def __init__(self, heads, max_distance):
                    super().__init__()

                def forward(self, query_length, key_length, device):
                    '''Head별 bias [H, T_q, T_k]를 반환한다.'''
                    raise NotImplementedError

            class ExplicitMultiHeadAttention(nn.Module):
                '''Relative bias와 mask를 score에 적용하는 attention이다.'''

                def __init__(self, model_dim, heads):
                    super().__init__()

                def split_heads(self, tensor):
                    '''[B, T, D]를 [B, H, T, d_h]로 분할한다.'''
                    raise NotImplementedError

                def merge_heads(self, tensor):
                    '''[B, H, T, d_h]를 [B, T, D]로 병합한다.'''
                    raise NotImplementedError

                def forward(
                    self,
                    query,
                    key,
                    value,
                    relative_bias=None,
                    attention_mask=None,
                ):
                    '''Bias와 mask가 적용된 attention output과 weights를 반환한다.'''
                    raise NotImplementedError

            class T5EncoderBlock(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def forward(self, hidden, relative_bias):
                    '''Encoder hidden과 self-attention weights를 반환한다.'''
                    raise NotImplementedError

            class T5DecoderBlock(nn.Module):
                def __init__(self, config):
                    super().__init__()

                def forward(self, hidden, memory, relative_bias, causal_mask):
                    '''Decoder hidden과 self/cross attention weights를 반환한다.'''
                    raise NotImplementedError

            class T5Mini(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def causal_mask(self, length, device):
                    '''미래 token을 가리는 bool mask [T, T]를 만든다.'''
                    raise NotImplementedError

                def forward(self, source, decoder_inputs):
                    raise NotImplementedError
            """,
            """
            class RelativePositionBias(nn.Module):
                '''Clipped relative distance를 head별 attention bias로 바꾼다.'''

                def __init__(self, heads, max_distance):
                    super().__init__()
                    self.max_distance = max_distance
                    self.embedding = nn.Embedding(2 * max_distance + 1, heads)

                def forward(self, query_length, key_length, device):
                    query = torch.arange(query_length, device=device)[:, None]
                    key = torch.arange(key_length, device=device)[None, :]
                    relative = (key - query).clamp(
                        -self.max_distance,
                        self.max_distance,
                    )
                    indices = relative + self.max_distance
                    return self.embedding(indices).permute(2, 0, 1)

            class ExplicitMultiHeadAttention(nn.Module):
                '''QK^T/sqrt(d) 위에 bias와 mask를 직접 적용한다.'''

                def __init__(self, model_dim, heads):
                    super().__init__()
                    if model_dim % heads != 0:
                        raise ValueError("model_dim must be divisible by heads")
                    self.heads = heads
                    self.head_dim = model_dim // heads
                    self.query_projection = nn.Linear(model_dim, model_dim)
                    self.key_projection = nn.Linear(model_dim, model_dim)
                    self.value_projection = nn.Linear(model_dim, model_dim)
                    self.output_projection = nn.Linear(model_dim, model_dim)

                def split_heads(self, tensor):
                    batch, length, _ = tensor.shape
                    tensor = tensor.reshape(
                        batch,
                        length,
                        self.heads,
                        self.head_dim,
                    )
                    return tensor.transpose(1, 2)

                def merge_heads(self, tensor):
                    batch, _, length, _ = tensor.shape
                    tensor = tensor.transpose(1, 2).contiguous()
                    return tensor.reshape(
                        batch,
                        length,
                        self.heads * self.head_dim,
                    )

                def forward(
                    self,
                    query,
                    key,
                    value,
                    relative_bias=None,
                    attention_mask=None,
                ):
                    queries = self.split_heads(self.query_projection(query))
                    keys = self.split_heads(self.key_projection(key))
                    values = self.split_heads(self.value_projection(value))
                    scale = self.head_dim**-0.5
                    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
                    if relative_bias is not None:
                        scores = scores + relative_bias[None]
                    if attention_mask is not None:
                        scores = scores.masked_fill(
                            attention_mask[None, None],
                            float("-inf"),
                        )
                    weights = scores.softmax(dim=-1)
                    context = torch.matmul(weights, values)
                    output = self.output_projection(self.merge_heads(context))
                    return output, weights

            class T5EncoderBlock(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(config.model_dim)
                    self.attention = ExplicitMultiHeadAttention(
                        config.model_dim,
                        config.heads,
                    )
                    self.norm2 = nn.LayerNorm(config.model_dim)
                    self.feedforward = nn.Sequential(
                        nn.Linear(config.model_dim, config.feedforward_dim),
                        nn.ReLU(),
                        nn.Linear(config.feedforward_dim, config.model_dim),
                    )

                def forward(self, hidden, relative_bias):
                    normalized = self.norm1(hidden)
                    attended, weights = self.attention(
                        normalized,
                        normalized,
                        normalized,
                        relative_bias=relative_bias,
                    )
                    hidden = hidden + attended
                    hidden = hidden + self.feedforward(self.norm2(hidden))
                    return hidden, weights

            class T5DecoderBlock(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(config.model_dim)
                    self.self_attention = ExplicitMultiHeadAttention(
                        config.model_dim,
                        config.heads,
                    )
                    self.norm2 = nn.LayerNorm(config.model_dim)
                    self.cross_attention = ExplicitMultiHeadAttention(
                        config.model_dim,
                        config.heads,
                    )
                    self.norm3 = nn.LayerNorm(config.model_dim)
                    self.feedforward = nn.Sequential(
                        nn.Linear(config.model_dim, config.feedforward_dim),
                        nn.ReLU(),
                        nn.Linear(config.feedforward_dim, config.model_dim),
                    )

                def forward(self, hidden, memory, relative_bias, causal_mask):
                    normalized = self.norm1(hidden)
                    attended, self_weights = self.self_attention(
                        normalized,
                        normalized,
                        normalized,
                        relative_bias=relative_bias,
                        attention_mask=causal_mask,
                    )
                    hidden = hidden + attended
                    normalized = self.norm2(hidden)
                    attended, cross_weights = self.cross_attention(
                        normalized,
                        memory,
                        memory,
                    )
                    hidden = hidden + attended
                    hidden = hidden + self.feedforward(self.norm3(hidden))
                    return hidden, self_weights, cross_weights

            class T5Mini(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, config.model_dim)
                    self.encoder_relative_bias = RelativePositionBias(
                        config.heads,
                        config.max_distance,
                    )
                    self.decoder_relative_bias = RelativePositionBias(
                        config.heads,
                        config.max_distance,
                    )
                    self.encoder = T5EncoderBlock(config)
                    self.decoder = T5DecoderBlock(config)
                    self.output = nn.Linear(config.model_dim, vocab_size)

                def causal_mask(self, length, device):
                    return torch.ones(
                        length,
                        length,
                        dtype=torch.bool,
                        device=device,
                    ).triu(diagonal=1)

                def forward(self, source, decoder_inputs):
                    source_hidden = self.embedding(source)
                    source_bias = self.encoder_relative_bias(
                        source.shape[1],
                        source.shape[1],
                        source.device,
                    )
                    memory, _ = self.encoder(source_hidden, source_bias)
                    target_hidden = self.embedding(decoder_inputs)
                    target_length = decoder_inputs.shape[1]
                    target_bias = self.decoder_relative_bias(
                        target_length,
                        target_length,
                        source.device,
                    )
                    target_mask = self.causal_mask(target_length, source.device)
                    decoded, _, _ = self.decoder(
                        target_hidden,
                        memory,
                        target_bias,
                        target_mask,
                    )
                    return self.output(decoded)

            model = T5Mini(VOCAB_SIZE, config).to(DEVICE)
            bias = model.encoder_relative_bias(2, 4, DEVICE)
            assert bias.shape == (4, 2, 4)
            train_logits = model(train_span_input, train_decoder_inputs)
            assert train_logits.shape == (48, 4, VOCAB_SIZE)
            probe = train_logits[0, -1, SENTINEL]
            probe.backward()
            encoder_gradient = model.encoder_relative_bias.embedding.weight.grad
            decoder_gradient = model.decoder_relative_bias.embedding.weight.grad
            assert encoder_gradient is not None
            assert decoder_gradient is not None
            assert float(encoder_gradient.abs().sum()) > 0.0
            assert float(decoder_gradient.abs().sum()) > 0.0
            model.zero_grad(set_to_none=True)

            changed_inputs = train_decoder_inputs[:2].clone()
            changed_inputs[:, -1] = EOS
            with torch.no_grad():
                original_logits = model(
                    train_span_input[:2],
                    train_decoder_inputs[:2],
                )
                changed_logits = model(train_span_input[:2], changed_inputs)
            assert torch.allclose(
                original_logits[:, :-1],
                changed_logits[:, :-1],
                atol=1e-6,
            )
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: denoising loss, train step, token accuracy를 구현하세요.
            def denoising_loss(logits, targets):
                raise NotImplementedError

            def train_t5_step(model, optimizer, source, decoder_inputs, targets):
                raise NotImplementedError

            def evaluate_t5(model, source, decoder_inputs, targets):
                raise NotImplementedError
            """,
            """
            def denoising_loss(logits, targets):
                return F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )

            def train_t5_step(model, optimizer, source, decoder_inputs, targets):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = denoising_loss(model(source, decoder_inputs), targets)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_t5(model, source, decoder_inputs, targets):
                model.eval()
                with torch.no_grad():
                    predictions = model(source, decoder_inputs).argmax(dim=-1)
                return float((predictions == targets).float().mean())

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(
                    train_t5_step(
                        model,
                        optimizer,
                        train_span_input,
                        train_decoder_inputs,
                        train_span_target,
                    )
                )
            heldout_accuracy = evaluate_t5(
                model,
                test_span_input,
                test_decoder_inputs,
                test_span_target,
            )
            assert min(losses[3:]) < losses[0]
            assert heldout_accuracy > 0.75
            plt.figure(figsize=(5, 3))
            plt.plot(losses)
            plt.title(f"T5 held-out span accuracy={heldout_accuracy:.1%}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


GPT3 = _paper(
    number=7,
    slug="gpt3_few_shot",
    short_title="GPT-3 Few-shot",
    paper_title="Language Models are Few-Shot Learners",
    authors="Tom B. Brown et al.",
    year=2020,
    primary_url="https://arxiv.org/abs/2005.14165",
    venue="NeurIPS 2020",
    reproduction_goal="zero/one/few-shot prompt와 parameter update 없는 in-context protocol을 재현한다.",
    original_scale=(
        "175B LM 대신 token-overlap surrogate로 protocol과 shot별 평가만 "
        "검증한다. 6개 문장 template이 반복되므로 240/120 row split은 label "
        "접근을 차단하지만 새로운 표현의 일반화 benchmark는 아니다."
    ),
    mappings=(
        ("§1 Fig. 1.1: zero/one/few-shot", "Task 1 prompt", "example count"),
        (
            "§2.1 Eq. (2.1): autoregressive context",
            "Task 1 serialized context",
            "query last",
        ),
        ("§2.2: no gradient updates", "Task 2", "trainable parameters=0"),
        ("§3: task evaluation by number of shots", "Task 3", "accuracy-vs-shots plot"),
    ),
    cells=(
        _concept(
            r"p(x)=\prod_{i=1}^{n}p(x_i\mid x_{<i};\theta),\qquad \theta' = \theta",
            "x_<i에는 demonstrations와 query가 직렬화되고 θ'=θ는 추론 중 update가 없음을 뜻한다.",
            "Task 1이 prompt, Task 2가 parameter-free predictor, Task 3이 shots curve를 구현한다.",
            (
                "demonstrations list + query string → serialized prompt → scalar label; "
                "trainable params=0."
            ),
            (
                "실제 GPT-3 생성 확률은 재현하지 않고 in-context 평가 계약만 "
                "투명한 surrogate로 보존한다. test label은 evaluator만 접근한다."
            ),
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정과 demonstrations 뒤에 query를 직렬화하는 prompt builder를 구현하세요.
            @dataclass
            class InContextConfig:
                pass

            def make_prompt(demo_indices, query_text, texts, labels):
                raise NotImplementedError
            """,
            """
            @dataclass
            class InContextConfig:
                shot_counts: tuple = (0, 2, 4, 6)

            def make_prompt(demo_indices, query_text, texts, labels):
                lines = []
                for index in demo_indices:
                    lines.append(f"Text: {texts[index]}")
                    lines.append(f"Label: {int(labels[index])}")
                lines.append(f"Text: {query_text}")
                lines.append("Label:")
                return "\\n".join(lines)

            config = InContextConfig()
            cpu_labels = sentiment_labels.detach().cpu()
            training_indices = list(range(240))
            test_indices = list(range(240, len(sentences)))
            assert set(training_indices).isdisjoint(test_indices)
            demonstration_labels = {
                index: int(cpu_labels[index])
                for index in training_indices
            }
            training_label_values = cpu_labels[training_indices]
            zero_shot_prior = int(training_label_values.mode().values)
            zero_prompt = make_prompt(
                [],
                sentences[test_indices[0]],
                sentences,
                demonstration_labels,
            )
            few_prompt = make_prompt(
                [0, 1, 2],
                sentences[test_indices[0]],
                sentences,
                demonstration_labels,
            )
            assert zero_prompt.count("Label:") == 1
            assert few_prompt.count("Label:") == 4
            """,
            "todo",
            "prompting",
        ),
        code(
            """
            # TODO 2: parameter 없는 overlap 기반 InContextClassifier를 구현하세요.
            def token_set(text):
                raise NotImplementedError

            def jaccard_similarity(first, second):
                raise NotImplementedError

            class InContextClassifier:
                def __init__(
                    self,
                    texts,
                    demonstration_labels,
                    zero_shot_prior,
                ):
                    raise NotImplementedError

                def nearest_demo(self, demo_indices, query_text):
                    '''Query와 가장 비슷한 demonstration index를 찾는다.'''
                    raise NotImplementedError

                def predict(self, demo_indices, query_text):
                    raise NotImplementedError
            """,
            """
            def token_set(text):
                return set(text.lower().split())

            def jaccard_similarity(first, second):
                first_tokens = token_set(first)
                second_tokens = token_set(second)
                intersection = len(first_tokens & second_tokens)
                union = max(1, len(first_tokens | second_tokens))
                return intersection / union

            class InContextClassifier:
                def __init__(
                    self,
                    texts,
                    demonstration_labels,
                    zero_shot_prior,
                ):
                    self.texts = texts
                    self.demonstration_labels = dict(demonstration_labels)
                    self.zero_shot_prior = int(zero_shot_prior)
                    self.trainable_parameters = 0

                def nearest_demo(self, demo_indices, query_text):
                    return max(
                        demo_indices,
                        key=lambda index: jaccard_similarity(
                            self.texts[index],
                            query_text,
                        ),
                    )

                def predict(self, demo_indices, query_text):
                    if not demo_indices:
                        return self.zero_shot_prior
                    if not set(demo_indices).issubset(self.demonstration_labels):
                        raise ValueError("demonstrations must come from the training split")
                    nearest = self.nearest_demo(demo_indices, query_text)
                    return self.demonstration_labels[nearest]

            model = InContextClassifier(
                sentences,
                demonstration_labels,
                zero_shot_prior,
            )
            assert model.trainable_parameters == 0
            assert not hasattr(model, "labels")
            assert model.predict([], sentences[test_indices[0]]) in (0, 1)
            """,
            "todo",
            "in-context-learning",
        ),
        code(
            """
            # TODO 3: shot 수별 evaluation과 prompt 예시/curve를 구현하세요.
            def select_demonstrations(texts, adjectives, candidate_indices):
                raise NotImplementedError

            def evaluate_in_context(
                model,
                demonstrations,
                test_indices,
                expected_labels,
                shots,
            ):
                raise NotImplementedError
            """,
            """
            def select_demonstrations(texts, adjectives, candidate_indices):
                selected = []
                for adjective in adjectives:
                    selected.append(
                        next(
                            index
                            for index in candidate_indices
                            if adjective in texts[index]
                        )
                    )
                return selected

            def evaluate_in_context(
                model,
                demonstrations,
                test_indices,
                expected_labels,
                shots,
            ):
                active_demos = demonstrations[:shots]
                predictions = torch.tensor(
                    [
                        model.predict(active_demos, model.texts[index])
                        for index in test_indices
                    ]
                )
                expected = expected_labels[test_indices]
                return float((predictions == expected).float().mean())

            adjectives = ["helpful", "clear", "safe", "confusing", "risky", "wrong"]
            demonstrations = select_demonstrations(
                sentences,
                adjectives,
                training_indices,
            )
            assert set(demonstrations).issubset(demonstration_labels)
            assert set(demonstrations).isdisjoint(test_indices)
            accuracies = [
                evaluate_in_context(
                    model,
                    demonstrations,
                    test_indices,
                    cpu_labels,
                    shots,
                )
                for shots in config.shot_counts
            ]
            assert all(0.0 <= value <= 1.0 for value in accuracies)
            assert accuracies[-1] >= accuracies[0]
            plt.figure(figsize=(5, 3))
            plt.plot(config.shot_counts, accuracies, marker="o")
            plt.ylim(0, 1.05)
            plt.xlabel("shots")
            plt.ylabel("accuracy")
            plt.title("in-context demonstrations")
            plt.tight_layout()
            plt.show()
            print(
                make_prompt(
                    demonstrations[:2],
                    sentences[test_indices[0]],
                    sentences,
                    demonstration_labels,
                )
            )
            """,
            "todo",
            "evaluation",
            "visualization",
        ),
    ),
)


LORA = _paper(
    number=8,
    slug="lora",
    short_title="LoRA",
    paper_title="LoRA: Low-Rank Adaptation of Large Language Models",
    authors="Edward J. Hu et al.",
    year=2021,
    primary_url="https://arxiv.org/abs/2106.09685",
    venue="ICLR 2022",
    reproduction_goal="frozen W0 + BA low-rank update와 trainable parameter 절감을 재현한다.",
    original_scale="GPT-3 attention projection 대신 bag-of-words classifier의 rank-2 projection을 쓴다.",
    mappings=(
        ("§4.1 Eq. (3): h=W0x+BAx", "Task 1 LoRALinear", "forward equality"),
        ("§4.1: W0 frozen, A/B trainable", "Task 1–2", "requires_grad assert"),
        (
            "§4.2: applying LoRA to Transformer",
            "Task 2 classifier projection",
            "rank=2",
        ),
        ("§7.1: low intrinsic rank", "Task 3", "parameter ratio/loss plot"),
    ),
    cells=(
        _concept(
            r"h=W_0x+\Delta Wx=W_0x+\frac{\alpha}{r}BAx",
            "W0는 frozen weight, A/B는 rank r factor, α/r은 update scale이다.",
            "Task 1이 LoRALinear, Task 2가 classifier, Task 3이 trainable-only 최적화를 구현한다.",
            "bag-of-words [B,V] → LoRA hidden [B,16] → class logits [B,2].",
            "Transformer Q/V 대신 작은 projection에 적용하지만 low-rank/freeze 계약은 동일하다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정과 frozen base + trainable A/B인 LoRALinear를 구현하세요.
            @dataclass
            class LoRAConfig:
                pass

            class LoRALinear(nn.Module):
                def __init__(self, in_features, out_features, config):
                    super().__init__()

                def low_rank_update(self, inputs):
                    '''alpha/r 곱하기 B(Ax) 형태의 update를 반환한다.'''
                    raise NotImplementedError

                def forward(self, inputs):
                    raise NotImplementedError
            """,
            """
            @dataclass
            class LoRAConfig:
                rank: int = 2
                alpha: float = 2.0
                hidden_dim: int = 16
                learning_rate: float = 0.08
                steps: int = 28

            class LoRALinear(nn.Module):
                def __init__(self, in_features, out_features, config):
                    super().__init__()
                    base = torch.randn(out_features, in_features) * 0.02
                    self.base_weight = nn.Parameter(base, requires_grad=False)
                    self.factor_a = nn.Parameter(
                        torch.randn(config.rank, in_features) * 0.02
                    )
                    self.factor_b = nn.Parameter(torch.zeros(out_features, config.rank))
                    self.scale = config.alpha / config.rank

                def low_rank_update(self, inputs):
                    projected = F.linear(inputs, self.factor_a)
                    return self.scale * F.linear(projected, self.factor_b)

                def forward(self, inputs):
                    base = F.linear(inputs, self.base_weight)
                    return base + self.low_rank_update(inputs)

            config = LoRAConfig()
            layer = LoRALinear(5, 2, config).to(DEVICE)
            probe = torch.randn(3, 5, device=DEVICE)
            assert torch.allclose(layer(probe), F.linear(probe, layer.base_weight))
            assert not layer.base_weight.requires_grad
            """,
            "todo",
            "equation",
        ),
        code(
            """
            # TODO 2: bag-of-words feature와 LoRAClassifier를 구현하세요.
            def build_bag_of_words(texts, vocabulary, word_to_id, device):
                raise NotImplementedError

            class LoRAClassifier(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()

                def forward(self, features):
                    raise NotImplementedError
            """,
            """
            def build_bag_of_words(texts, vocabulary, word_to_id, device):
                features = torch.zeros(len(texts), len(vocabulary), device=device)
                for row, text in enumerate(texts):
                    for word in text.lower().split():
                        features[row, word_to_id[word]] += 1
                return features

            class LoRAClassifier(nn.Module):
                def __init__(self, vocab_size, config):
                    super().__init__()
                    self.adaptation = LoRALinear(vocab_size, config.hidden_dim, config)
                    self.classifier = nn.Linear(config.hidden_dim, 2)

                def forward(self, features):
                    hidden = torch.tanh(self.adaptation(features))
                    return self.classifier(hidden)

            features = build_bag_of_words(sentences, itos, stoi, DEVICE)
            model = LoRAClassifier(VOCAB_SIZE, config).to(DEVICE)
            assert features.shape == (360, VOCAB_SIZE)
            assert model(features[:2]).shape == (2, 2)
            """,
            "todo",
            "model",
        ),
        code(
            """
            # TODO 3: LoRA loss, train step, parameter ratio/test accuracy를 구현하세요.
            def lora_classification_loss(logits, targets):
                raise NotImplementedError

            def train_lora_step(model, optimizer, features, targets):
                raise NotImplementedError

            def evaluate_lora(model, features, targets):
                raise NotImplementedError
            """,
            """
            def lora_classification_loss(logits, targets):
                return F.cross_entropy(logits, targets)

            def train_lora_step(model, optimizer, features, targets):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = lora_classification_loss(model(features), targets)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            def evaluate_lora(model, features, targets):
                model.eval()
                with torch.no_grad():
                    predictions = model(features).argmax(dim=1)
                return float((predictions == targets).float().mean())

            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
            losses = []
            for step in range(config.steps):
                losses.append(
                    train_lora_step(
                        model,
                        optimizer,
                        features[:240],
                        sentiment_labels[:240],
                    )
                )
            accuracy = evaluate_lora(
                model,
                features[240:],
                sentiment_labels[240:],
            )
            trainable_count = sum(parameter.numel() for parameter in trainable)
            full_count = model.adaptation.base_weight.numel()
            assert model.adaptation.base_weight.grad is None
            assert trainable_count < full_count
            assert min(losses[3:]) < losses[0]
            figure, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(losses)
            axes[0].set_title("LoRA adaptation loss")
            axes[1].bar(["trainable", "full W"], [trainable_count, full_count])
            axes[1].set_title(f"parameter count, acc={accuracy:.1%}")
            plt.tight_layout()
            plt.show()
            """,
            "todo",
            "training",
            "evaluation",
        ),
    ),
)


RAG = _paper(
    number=9,
    slug="rag",
    short_title="RAG",
    paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    authors="Patrick Lewis et al.",
    year=2020,
    primary_url="https://arxiv.org/abs/2005.11401",
    venue="NeurIPS 2020",
    reproduction_goal="retriever distribution과 RAG-Sequence marginalization을 재현한다.",
    original_scale="DPR+BART+Wikipedia 21M passages 대신 TF-IDF와 로컬 5문서를 사용한다.",
    mappings=(
        ("§2.1 Eq. (1): pη(z|x)", "Task 1 softmax similarity", "probability sum"),
        ("§2.2 Eq. (2): RAG-Sequence", "Task 2 marginalization", "finite NLL"),
        ("§2.2 Eq. (3): RAG-Token", "Task 2 설명/비교", "per-token distinction"),
        ("Fig. 1: retriever+generator", "Task 3", "recall@1/cited answer"),
    ),
    cells=(
        _concept(
            (
                r"p_{RAG\text{-}Seq}(y\mid x)="
                r"\sum_{z\in top\text{-}k}p_{\eta}(z\mid x)p_{\theta}(y\mid x,z)"
            ),
            "z는 문서, p_eta는 retriever, p_theta는 document-conditioned generator likelihood다.",
            "Task 1이 TF-IDF retriever, Task 2가 RAGSequence class, Task 3이 NLL/recall을 구현한다.",
            "query vectors [3,V], document vectors [5,V] → p(z|x) [3,5] → answer prob [3].",
            "neural DPR/BART 대신 결정적 surrogate를 쓰지만 latent-document marginalization은 정확하다.",
        ),
        _setup(),
        code(
            """
            # TODO 1: 설정, tokenizer/count vector, TF-IDF retriever 분포를 구현하세요.
            @dataclass
            class RAGConfig:
                pass

            def terms(text):
                raise NotImplementedError

            def count_vector(text, vocabulary, term_to_id):
                raise NotImplementedError

            def retrieval_distribution(query_vectors, document_vectors):
                raise NotImplementedError
            """,
            """
            @dataclass
            class RAGConfig:
                generator_match_probability: float = 0.9
                generator_miss_probability: float = 0.05
                top_k: int = 3

            def terms(text):
                return re.findall(r"[a-z0-9]+", text.lower())

            def count_vector(text, vocabulary, term_to_id):
                vector = torch.zeros(len(vocabulary))
                for term, count in Counter(terms(text)).items():
                    vector[term_to_id[term]] = count
                return vector

            def retrieval_distribution(query_vectors, document_vectors):
                similarity = query_vectors @ document_vectors.T
                return similarity.softmax(dim=1), similarity

            config = RAGConfig()
            vocabulary = sorted(
                {term for document in documents for term in terms(document["text"])}
                | {term for query in queries for term in terms(query["query"])}
            )
            term_to_id = {term: index for index, term in enumerate(vocabulary)}
            document_tf = torch.stack(
                [
                    count_vector(document["text"], vocabulary, term_to_id)
                    for document in documents
                ]
            )
            document_frequency = (document_tf > 0).sum(dim=0)
            inverse_frequency = torch.log(
                (1 + len(documents)) / (1 + document_frequency)
            ) + 1
            document_vectors = F.normalize(document_tf * inverse_frequency, dim=1)
            query_tf = torch.stack(
                [count_vector(query["query"], vocabulary, term_to_id) for query in queries]
            )
            query_vectors = F.normalize(query_tf * inverse_frequency, dim=1)
            retrieval_probability, similarity = retrieval_distribution(
                query_vectors,
                document_vectors,
            )
            assert retrieval_probability.shape == (3, 5)
            assert torch.allclose(retrieval_probability.sum(1), torch.ones(3))
            """,
            "todo",
            "retrieval",
        ),
        code(
            """
            # TODO 2: retriever와 generator likelihood를 marginalize하는 RAGSequence를 구현하세요.
            class RAGSequence:
                def __init__(self, documents, config):
                    raise NotImplementedError

                def generator_likelihood(self, queries, document_ids):
                    raise NotImplementedError

                def forward(self, retrieval_probability, queries):
                    raise NotImplementedError

                def grounded_answer(self, document_index):
                    '''Document id가 포함된 grounded answer를 반환한다.'''
                    raise NotImplementedError
            """,
            """
            class RAGSequence:
                def __init__(self, documents, config):
                    self.documents = documents
                    self.document_ids = [document["id"] for document in documents]
                    self.config = config

                def generator_likelihood(self, queries, document_ids):
                    likelihood = torch.empty(len(queries), len(document_ids))
                    for row, query in enumerate(queries):
                        for column, document_id in enumerate(document_ids):
                            if document_id == query["answer_id"]:
                                value = self.config.generator_match_probability
                            else:
                                value = self.config.generator_miss_probability
                            likelihood[row, column] = value
                    return likelihood

                def forward(self, retrieval_probability, queries):
                    generator = self.generator_likelihood(
                        queries,
                        self.document_ids,
                    )
                    return (retrieval_probability * generator).sum(dim=1)

                def grounded_answer(self, document_index):
                    document = self.documents[document_index]
                    return f"[{document['id']}] {document['text']}"

            model = RAGSequence(documents, config)
            sequence_probability = model.forward(retrieval_probability, queries)
            assert sequence_probability.shape == (3,)
            assert bool((sequence_probability > 0).all())
            """,
            "todo",
            "marginalization",
            "model",
        ),
        code(
            """
            # TODO 3: marginal NLL, recall@1, grounded answer 평가를 구현하세요.
            def rag_sequence_loss(sequence_probability):
                raise NotImplementedError

            def evaluate_retrieval(similarity, documents, queries):
                raise NotImplementedError

            def build_grounded_answers(model, top_indices):
                raise NotImplementedError
            """,
            """
            def rag_sequence_loss(sequence_probability):
                return -sequence_probability.clamp_min(1e-8).log().mean()

            def evaluate_retrieval(similarity, documents, queries):
                top_indices = similarity.argmax(dim=1)
                document_ids = [documents[index]["id"] for index in top_indices.tolist()]
                matches = [
                    predicted == query["answer_id"]
                    for predicted, query in zip(document_ids, queries)
                ]
                return sum(matches) / len(matches), top_indices

            def build_grounded_answers(model, top_indices):
                return [
                    model.grounded_answer(document_index)
                    for document_index in top_indices.tolist()
                ]

            loss = rag_sequence_loss(sequence_probability)
            recall, top_indices = evaluate_retrieval(similarity, documents, queries)
            answers = build_grounded_answers(model, top_indices)
            assert loss.isfinite()
            assert recall >= 2 / 3
            assert all(answer.startswith("[") for answer in answers)
            plt.figure(figsize=(7, 3))
            plt.imshow(similarity)
            plt.xticks(range(len(documents)), [item["id"] for item in documents], rotation=30)
            plt.yticks(range(len(queries)), [f"q{index + 1}" for index in range(len(queries))])
            plt.colorbar(label="cosine")
            plt.title(f"RAG retrieval recall@1={recall:.1%}")
            plt.tight_layout()
            plt.show()
            print("\\n".join(answers))
            """,
            "todo",
            "evaluation",
            "visualization",
        ),
    ),
)


SPECS = (WORD2VEC, SEQ2SEQ, BAHDANAU, TRANSFORMER, GPT1, BERT, T5, GPT3, LORA, RAG)
assert len(SPECS) == 10
assert tuple(spec.number for spec in SPECS) == tuple(range(10))
assert len({spec.slug for spec in SPECS}) == 10
