import pytest

from llm_engineering_lab.nlp_basics import (
    corpus_term_frequencies,
    mask_pii,
    normalize_text,
    token_ngrams,
    tokenize,
    top_terms,
)


def test_normalize_text_handles_nfkc_case_and_unicode_whitespace() -> None:
    source = "  Ｈｅｌｌｏ\t한국어\u3000ＮＬＰ\n①  "

    assert normalize_text(source) == "hello 한국어 nlp 1"


def test_mask_pii_replaces_supported_values_and_preserves_labels() -> None:
    source = (
        "메일 Test.User+lab@Example.COM, 전화 010-1234-5678, "
        "주문번호: ORD-2026-12345678, 접수값 20260814000123"
    )

    assert mask_pii(source) == (
        "메일 <EMAIL>, 전화 <PHONE>, 주문번호: <ORDER_ID>, 접수값 <ORDER_ID>"
    )


def test_mask_pii_supports_international_and_separator_free_phone_numbers() -> None:
    assert mask_pii("+82 10 9876 5432 / 01098765432") == "<PHONE> / <PHONE>"


def test_tokenize_extracts_mixed_scripts_numbers_and_canonical_masks() -> None:
    source = "한국NLP4 실습, <email> + RAG_2026! <PHONE>"

    assert tokenize(source) == [
        "한국",
        "nlp",
        "4",
        "실습",
        "<EMAIL>",
        "rag",
        "2026",
        "<PHONE>",
    ]


def test_tokenize_can_follow_masking_without_leaking_pii() -> None:
    tokens = tokenize(mask_pii("문의: User@example.com, 02-1234-5678"))

    assert tokens == ["문의", "<EMAIL>", "<PHONE>"]
    assert "user" not in tokens


def test_token_ngrams_preserves_order_and_handles_large_width() -> None:
    tokens = ["한국어", "nlp", "실습"]

    assert token_ngrams(tokens, 2) == [("한국어", "nlp"), ("nlp", "실습")]
    assert token_ngrams(tokens, 4) == []


def test_corpus_frequencies_and_top_terms_are_deterministic() -> None:
    corpus = ["NLP 한국 NLP", "한국 RAG", "rag 생성형 AI"]

    frequencies = corpus_term_frequencies(corpus)

    assert list(frequencies) == ["ai", "nlp", "rag", "생성형", "한국"]
    assert frequencies == {"ai": 1, "nlp": 2, "rag": 2, "생성형": 1, "한국": 2}
    assert top_terms(frequencies, limit=4) == [
        ("nlp", 2),
        ("rag", 2),
        ("한국", 2),
        ("ai", 1),
    ]


def test_corpus_frequency_can_mask_sensitive_values() -> None:
    frequencies = corpus_term_frequencies(
        ["a@example.com", "010-1234-5678"],
        mask_sensitive_data=True,
    )

    assert frequencies == {"<EMAIL>": 1, "<PHONE>": 1}


@pytest.mark.parametrize(
    ("function", "argument", "exception"),
    [
        (normalize_text, None, TypeError),
        (mask_pii, 123, TypeError),
        (tokenize, ["not", "text"], TypeError),
        (lambda value: token_ngrams(value, 2), "not-a-token-sequence", TypeError),
        (lambda value: token_ngrams(["valid", value], 2), 7, TypeError),
        (corpus_term_frequencies, "one document", TypeError),
        (lambda value: top_terms(value), [("nlp", 1)], TypeError),
    ],
)
def test_invalid_container_and_text_inputs_raise(
    function: object,
    argument: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        function(argument)  # type: ignore[operator]


@pytest.mark.parametrize("n", [0, -1])
def test_token_ngrams_rejects_nonpositive_width(n: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        token_ngrams(["nlp"], n)


def test_top_terms_filters_counts_and_validates_frequency_values() -> None:
    assert top_terms({"b": 3, "a": 3, "rare": 1}, min_count=2) == [("a", 3), ("b", 3)]

    with pytest.raises(ValueError, match="negative"):
        top_terms({"invalid": -1})

    with pytest.raises(TypeError, match="integer"):
        top_terms({"invalid": True})
