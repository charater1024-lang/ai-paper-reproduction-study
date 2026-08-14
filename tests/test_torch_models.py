from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_engineering_lab.torch_text import (  # noqa: E402
    ClassifierTrainingConfig,
    MeanPoolTextClassifier,
    TextClassificationDataset,
    TextClassifierConfig,
    TicketExample,
    Vocabulary,
    fit_classifier,
    load_classifier_checkpoint,
    make_dataloader,
    split_examples,
)
from llm_engineering_lab.transformer import (  # noqa: E402
    LanguageModelDataset,
    TinyCausalTransformer,
    TinyTransformerConfig,
    sample_next_token,
)

torch.set_num_threads(1)


def test_vocabulary_round_trip_and_unknown_token() -> None:
    vocabulary = Vocabulary.build(["Login failed!", "login reset"], min_frequency=1)

    ids = vocabulary.encode("LOGIN never-seen", add_bos=True, add_eos=True)

    assert ids[0] == vocabulary.bos_id
    assert ids[-1] == vocabulary.eos_id
    assert vocabulary.unk_id in ids
    restored = Vocabulary.from_dict(vocabulary.to_dict())
    assert restored.to_dict() == vocabulary.to_dict()


def test_group_safe_split_keeps_cases_together() -> None:
    examples = [
        TicketExample("login one", "access", "case-a"),
        TicketExample("login followup", "access", "case-a"),
        TicketExample("password", "access", "case-b"),
        TicketExample("invoice one", "billing", "case-c"),
        TicketExample("invoice followup", "billing", "case-c"),
        TicketExample("refund", "billing", "case-d"),
    ]

    train, validation = split_examples(examples, validation_ratio=0.5, seed=7)

    train_groups = {example.group_id for example in train}
    validation_groups = {example.group_id for example in validation}
    assert train_groups.isdisjoint(validation_groups)
    assert {example.label for example in train} == {"access", "billing"}
    assert {example.label for example in validation} == {"access", "billing"}


def test_classifier_forward_ignores_padding() -> None:
    vocabulary = Vocabulary.build(["login failed", "refund payment"])
    model = MeanPoolTextClassifier(
        TextClassifierConfig(
            vocab_size=len(vocabulary),
            num_classes=2,
            pad_id=vocabulary.pad_id,
            embedding_dim=12,
            hidden_dim=8,
            dropout=0.0,
        )
    ).eval()
    short = torch.tensor(
        [vocabulary.encode("login failed", add_bos=True, add_eos=True)]
    )
    padded = torch.cat(
        (short, torch.full((1, 3), vocabulary.pad_id, dtype=torch.long)), dim=1
    )

    short_logits = model(short)
    padded_logits = model(padded, padded.ne(vocabulary.pad_id))

    assert short_logits.shape == (1, 2)
    torch.testing.assert_close(short_logits, padded_logits)


def test_classifier_training_and_checkpoint(tmp_path: Path) -> None:
    examples = [
        TicketExample("cannot login password", "access"),
        TicketExample("reset account login", "access"),
        TicketExample("invoice refund payment", "billing"),
        TicketExample("charged card refund", "billing"),
    ]
    vocabulary = Vocabulary.build(example.text for example in examples)
    label_to_id = {"access": 0, "billing": 1}
    train_dataset = TextClassificationDataset(examples, vocabulary, label_to_id)
    loader = make_dataloader(train_dataset, batch_size=2, shuffle=True)
    model = MeanPoolTextClassifier(
        TextClassifierConfig(
            vocab_size=len(vocabulary),
            num_classes=2,
            pad_id=vocabulary.pad_id,
            embedding_dim=8,
            hidden_dim=8,
            dropout=0.0,
        )
    )
    checkpoint = tmp_path / "classifier.pt"

    history = fit_classifier(
        model,
        loader,
        loader,
        config=ClassifierTrainingConfig(epochs=1, learning_rate=1e-2),
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        checkpoint_extra={"vocabulary": vocabulary.to_dict()},
    )
    restored, payload = load_classifier_checkpoint(checkpoint)

    assert len(history) == 1
    assert checkpoint.exists()
    assert restored.config.num_classes == 2
    assert payload["extra"]["vocabulary"] == vocabulary.to_dict()


def tiny_model(*, dropout: float = 0.0) -> TinyCausalTransformer:
    torch.manual_seed(3)
    return TinyCausalTransformer(
        TinyTransformerConfig(
            vocab_size=24,
            pad_id=0,
            max_seq_length=16,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            feed_forward_dim=32,
            dropout=dropout,
        )
    ).eval()


def test_causal_mask_blocks_future_information_and_computes_loss() -> None:
    model = tiny_model()
    first = torch.tensor([[1, 4, 5, 6]])
    changed_future = torch.tensor([[1, 4, 5, 9]])

    first_output = model(first, labels=first)
    changed_output = model(changed_future)

    assert first_output.logits.shape == (1, 4, 24)
    assert first_output.loss is not None and torch.isfinite(first_output.loss)
    torch.testing.assert_close(
        first_output.logits[:, :-1],
        changed_output.logits[:, :-1],
        atol=1e-6,
        rtol=1e-5,
    )


def test_kv_cache_matches_full_forward_for_next_token() -> None:
    model = tiny_model()
    prefix = torch.tensor([[1, 4, 5]])
    next_token = torch.tensor([[6]])
    prefix_output = model(prefix, use_cache=True)
    assert prefix_output.past_key_values is not None

    cached_output = model(
        next_token,
        attention_mask=torch.ones((1, 4), dtype=torch.bool),
        past_key_values=prefix_output.past_key_values,
        use_cache=True,
    )
    full_output = model(torch.tensor([[1, 4, 5, 6]]))

    assert cached_output.past_key_values is not None
    assert cached_output.past_key_values[0].key.shape[2] == 4
    torch.testing.assert_close(
        cached_output.logits[:, -1],
        full_output.logits[:, -1],
        atol=1e-6,
        rtol=1e-5,
    )


def test_generation_and_sampling_shapes() -> None:
    model = tiny_model()
    prompts = torch.tensor([[1, 4, 5], [0, 1, 7]])
    mask = prompts.ne(0)

    generated = model.generate(
        prompts,
        attention_mask=mask,
        max_new_tokens=3,
        temperature=0.0,
        eos_id=None,
        use_cache=True,
    )
    greedy = sample_next_token(torch.tensor([[0.1, 2.0, 0.3]]), temperature=0.0)

    assert generated.shape == (2, 6)
    assert greedy.tolist() == [1]


def test_language_model_dataset_returns_fixed_windows() -> None:
    dataset = LanguageModelDataset([1, 5, 6, 2, 1], block_size=4, pad_id=0, stride=3)

    first = dataset[0]
    second = dataset[1]

    assert first["input_ids"].tolist() == [1, 5, 6, 2]
    assert second["input_ids"].tolist() == [2, 1, 0, 0]
    assert second["attention_mask"].tolist() == [True, True, False, False]
