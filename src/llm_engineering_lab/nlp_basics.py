"""Dependency-free text preprocessing helpers for Korean and English corpora.

The functions in this module intentionally use only Python's standard library.
They provide a small, predictable preprocessing pipeline that is suitable for
introductory NLP exercises before adding a morphological analyzer or tokenizer.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

EMAIL_MASK = "<EMAIL>"
PHONE_MASK = "<PHONE>"
ORDER_ID_MASK = "<ORDER_ID>"
MASK_TOKENS = frozenset({EMAIL_MASK, PHONE_MASK, ORDER_ID_MASK})

_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[a-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+(?![\w.-])",
    flags=re.IGNORECASE,
)

_KOREAN_PHONE_PREFIX = r"(?:1[016789]|2|3[1-3]|4[1-4]|5[1-5]|6[1-4])"
_PHONE_PATTERN = re.compile(
    rf"""
    (?<![\w])
    (?:
        (?:\+82|0082)[ .-]?(?:\(0\)[ .-]?)?{_KOREAN_PHONE_PREFIX}
        |
        (?:\(0{_KOREAN_PHONE_PREFIX}\)|0{_KOREAN_PHONE_PREFIX})
    )
    [ .-]?\d{{3,4}}[ .-]?\d{{4}}
    (?![\w])
    """,
    flags=re.VERBOSE,
)

# Keep a human-readable label such as ``주문번호:`` and replace only its value.
_LABELED_ORDER_PATTERN = re.compile(
    r"(?P<label>"
    r"(?:(?:order|ord)(?:\s*(?:id|no\.?|number))?|주문\s*(?:번호)?)"
    r"[\s:#=_-]*"
    r")"
    r"(?P<value>"
    r"(?=[a-z0-9-]{8,}(?![a-z0-9-]))"
    r"(?=[a-z0-9-]*\d)"
    r"[a-z0-9]+(?:-[a-z0-9]+)*"
    r")",
    flags=re.IGNORECASE,
)

# Phone numbers are masked first, so a remaining run of at least twelve digits
# is treated as a long identifier. Spaces and hyphens are common export formats.
_LONG_NUMBER_PATTERN = re.compile(r"(?<![\w])\d(?:[ -]?\d){11,}(?![\w])")

_TOKEN_PATTERN = re.compile(
    r"<(?:email|phone|order_id)>"
    r"|[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3\ua960-\ua97f\ud7b0-\ud7ff]+"
    r"|[a-z]+"
    r"|\d+",
    flags=re.IGNORECASE,
)
_CANONICAL_MASKS = {mask.lower(): mask for mask in MASK_TOKENS}


def _validate_text(text: object, *, argument: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{argument} must be a string, got {type(text).__name__}")
    return text


def _validate_positive_integer(value: object, *, argument: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{argument} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{argument} must be greater than zero")
    return value


def normalize_text(text: str) -> str:
    """Return NFKC-normalized, lowercased text with collapsed whitespace.

    Unicode whitespace at either end is removed and every internal whitespace
    run is converted to one ASCII space. NFKC also makes compatibility forms,
    including full-width Latin letters and digits, consistent.

    Args:
        text: Source text to normalize.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    validated = _validate_text(text)
    compatible = unicodedata.normalize("NFKC", validated)
    return " ".join(compatible.lower().split())


def mask_pii(text: str) -> str:
    """Replace common email, Korean phone, and long order identifiers.

    The replacement tokens are ``<EMAIL>``, ``<PHONE>``, and ``<ORDER_ID>``.
    Phone numbers are processed before unlabelled long numeric identifiers so a
    separator-free mobile number is not mistaken for an order identifier. The
    function preserves all non-sensitive text and spacing.

    Args:
        text: Text whose sensitive values should be masked.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    masked = _EMAIL_PATTERN.sub(EMAIL_MASK, _validate_text(text))
    masked = _PHONE_PATTERN.sub(PHONE_MASK, masked)
    masked = _LABELED_ORDER_PATTERN.sub(
        lambda match: f"{match.group('label')}{ORDER_ID_MASK}",
        masked,
    )
    return _LONG_NUMBER_PATTERN.sub(ORDER_ID_MASK, masked)


def tokenize(text: str) -> list[str]:
    """Extract canonical mask, Korean, English, and number tokens from text.

    The input is normalized with :func:`normalize_text`. Punctuation is omitted,
    adjacent scripts are split (for example, ``한국NLP4`` becomes ``한국``,
    ``nlp``, ``4``), and known mask tokens are returned in their canonical
    uppercase form. The function does not mask raw PII automatically; call
    :func:`mask_pii` first when that behavior is desired.

    Args:
        text: Source text to tokenize.

    Returns:
        Tokens in their original left-to-right order.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    normalized = normalize_text(text)
    tokens = _TOKEN_PATTERN.findall(normalized)
    return [_CANONICAL_MASKS.get(token, token) for token in tokens]


def token_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Create contiguous token n-grams while preserving input order.

    Args:
        tokens: A sequence of non-empty string tokens.
        n: N-gram width. It must be a positive integer.

    Returns:
        A list of token tuples. The result is empty when ``n`` is larger than
        the token sequence.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``n`` or a token value is invalid.
    """

    width = _validate_positive_integer(n, argument="n")
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise TypeError("tokens must be a sequence of strings")

    materialized = list(tokens)
    for index, token in enumerate(materialized):
        if not isinstance(token, str):
            raise TypeError(
                f"tokens[{index}] must be a string, got {type(token).__name__}"
            )
        if not token:
            raise ValueError(f"tokens[{index}] must not be empty")

    return [
        tuple(materialized[start : start + width])
        for start in range(len(materialized) - width + 1)
    ]


def corpus_term_frequencies(
    corpus: Iterable[str],
    *,
    mask_sensitive_data: bool = False,
) -> dict[str, int]:
    """Count normalized tokens across every document in a corpus.

    Args:
        corpus: An iterable of document strings. A single string is rejected to
            prevent accidentally treating each character as a document.
        mask_sensitive_data: Apply :func:`mask_pii` before tokenizing each
            document when true.

    Returns:
        A token-to-count dictionary inserted in lexical token order for fully
        deterministic display and serialization.

    Raises:
        TypeError: If the corpus, a document, or the option has the wrong type.
    """

    if isinstance(corpus, (str, bytes)) or not isinstance(corpus, Iterable):
        raise TypeError("corpus must be an iterable of document strings")
    if not isinstance(mask_sensitive_data, bool):
        raise TypeError("mask_sensitive_data must be a boolean")

    frequencies: Counter[str] = Counter()
    for index, document in enumerate(corpus):
        validated = _validate_text(document, argument=f"corpus[{index}]")
        prepared = mask_pii(validated) if mask_sensitive_data else validated
        frequencies.update(tokenize(prepared))
    return dict(sorted(frequencies.items()))


def top_terms(
    frequencies: Mapping[str, int],
    *,
    limit: int = 10,
    min_count: int = 1,
) -> list[tuple[str, int]]:
    """Rank a frequency mapping by descending count and then lexical token.

    Args:
        frequencies: Token-to-nonnegative-count mapping, such as the output of
            :func:`corpus_term_frequencies`.
        limit: Maximum number of ranked terms to return.
        min_count: Minimum count required for a term to be included.

    Returns:
        ``(token, count)`` pairs with deterministic tie-breaking.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a limit, threshold, token, or count is invalid.
    """

    if not isinstance(frequencies, Mapping):
        raise TypeError("frequencies must be a token-to-count mapping")
    maximum = _validate_positive_integer(limit, argument="limit")
    threshold = _validate_positive_integer(min_count, argument="min_count")

    validated: list[tuple[str, int]] = []
    for token, count in frequencies.items():
        if not isinstance(token, str):
            raise TypeError(
                f"frequency token must be a string, got {type(token).__name__}"
            )
        if not token:
            raise ValueError("frequency tokens must not be empty")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"frequency for {token!r} must be an integer")
        if count < 0:
            raise ValueError(f"frequency for {token!r} must not be negative")
        if count >= threshold:
            validated.append((token, count))

    return sorted(validated, key=lambda item: (-item[1], item[0]))[:maximum]


__all__ = [
    "EMAIL_MASK",
    "MASK_TOKENS",
    "ORDER_ID_MASK",
    "PHONE_MASK",
    "corpus_term_frequencies",
    "mask_pii",
    "normalize_text",
    "token_ngrams",
    "tokenize",
    "top_terms",
]
