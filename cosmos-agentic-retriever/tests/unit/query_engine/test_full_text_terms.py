"""Exhaustive tests for `cosmos_agentic_retriever.query_engine.full_text_terms`.

Covers FTS tokenization (Unicode, lowering, dedup, stopwords, term cap, the
all-stopword degenerate case) and, critically for security, the escaping in
``fts_literal_args`` that keeps a hostile term from breaking out of the quoted
full-text literal it is embedded in.
"""

from __future__ import annotations

import pytest

from cosmos_agentic_retriever.query_engine.full_text_terms import (
    DEFAULT_MAX_FTS_TERMS,
    fts_literal_args,
    tokenize_for_fts,
)

# ═══════════════════════════ tokenize_for_fts ═════════════════════════════


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "!!! ??? ...", "---"])
def test_tokenize_empty_or_punctuation_only(query: str) -> None:
    assert tokenize_for_fts(query) == []


def test_tokenize_simple() -> None:
    assert tokenize_for_fts("hello world") == ["hello", "world"]


def test_tokenize_lowercases() -> None:
    assert tokenize_for_fts("Hello WORLD FooBar") == ["hello", "world", "foobar"]


def test_tokenize_dedupes_preserving_first_order() -> None:
    assert tokenize_for_fts("bb aa bb cc aa") == ["bb", "aa", "cc"]


def test_tokenize_dedupe_is_case_insensitive() -> None:
    assert tokenize_for_fts("Hello hello HELLO") == ["hello"]


def test_tokenize_splits_on_punctuation() -> None:
    assert tokenize_for_fts("foo, bar. baz! qux?") == ["foo", "bar", "baz", "qux"]


def test_tokenize_keeps_digits_and_underscore() -> None:
    assert tokenize_for_fts("abc 123 foo_bar") == ["abc", "123", "foo_bar"]


def test_tokenize_dedupes_numbers() -> None:
    assert tokenize_for_fts("1 1 2 2 3") == ["1", "2", "3"]


def test_tokenize_removes_stopwords() -> None:
    assert tokenize_for_fts("the cat and the dog") == ["cat", "dog"]


def test_tokenize_apostrophe_splits_and_drops_stopword_half() -> None:
    # "don" is a stopword, apostrophe is a delimiter, so only "t" survives.
    assert tokenize_for_fts("don't") == ["t"]


def test_tokenize_unicode_accented_words() -> None:
    assert tokenize_for_fts("Café Über") == ["café", "über"]


def test_tokenize_unicode_cjk() -> None:
    assert tokenize_for_fts("机器 学习 机器") == ["机器", "学习"]


def test_tokenize_all_stopwords_falls_back_to_original_terms() -> None:
    assert tokenize_for_fts("the and of") == ["the", "and", "of"]
    assert tokenize_for_fts("THE AND OF") == ["the", "and", "of"]


def test_tokenize_caps_at_default_max_terms() -> None:
    assert DEFAULT_MAX_FTS_TERMS == 30
    query = " ".join(f"w{i}" for i in range(DEFAULT_MAX_FTS_TERMS + 10))
    result = tokenize_for_fts(query)
    assert len(result) == DEFAULT_MAX_FTS_TERMS
    assert result == [f"w{i}" for i in range(DEFAULT_MAX_FTS_TERMS)]


def test_tokenize_cap_counts_distinct_only() -> None:
    # Duplicates must not consume the term budget.
    distinct = [f"t{i}" for i in range(DEFAULT_MAX_FTS_TERMS)]
    query = " ".join(distinct + distinct + ["extra_beyond_cap"])
    result = tokenize_for_fts(query)
    assert len(result) == DEFAULT_MAX_FTS_TERMS
    assert "extra_beyond_cap" not in result  # cap already reached by distinct set


@pytest.mark.parametrize("max_terms", [1, 3, 40])
def test_tokenize_accepts_per_call_limit(max_terms: int) -> None:
    terms = [f"term{index}" for index in range(45)]
    query = " ".join(terms)
    assert tokenize_for_fts(query, max_terms=max_terms) == terms[:max_terms]
    assert tokenize_for_fts(query) == terms[:DEFAULT_MAX_FTS_TERMS]


def test_custom_limit_applies_after_filtering_and_deduplication() -> None:
    assert tokenize_for_fts("the battery BATTERY recycling policy", max_terms=2) == [
        "battery",
        "recycling",
    ]


def test_custom_limit_applies_to_stopword_fallback() -> None:
    assert tokenize_for_fts("the and THE of", max_terms=2) == ["the", "and"]


@pytest.mark.parametrize("max_terms", [0, -1, 1.5, "3", None, True])
def test_tokenize_rejects_invalid_limits(max_terms) -> None:
    with pytest.raises(ValueError, match="max_terms must be a positive integer"):
        tokenize_for_fts("battery recycling", max_terms=max_terms)


def test_term_limit_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        tokenize_for_fts("battery recycling", 1)


def test_tokenize_injection_characters_are_stripped() -> None:
    # Quotes / semicolons / brackets are non-word chars -> removed at tokenization.
    assert tokenize_for_fts('drop"; SELECT') == ["drop", "select"]


# ═══════════════════════════ fts_literal_args ═════════════════════════════


def test_fts_literal_args_empty_is_empty_string() -> None:
    assert fts_literal_args([]) == ""


def test_fts_literal_args_single_term() -> None:
    assert fts_literal_args(["foo"]) == '"foo"'


def test_fts_literal_args_multiple_terms_joined() -> None:
    assert fts_literal_args(["foo", "bar", "baz"]) == '"foo", "bar", "baz"'


def test_fts_literal_args_escapes_embedded_quote() -> None:
    # A quote inside a term is backslash-escaped so it can't close the literal.
    assert fts_literal_args(['a"b']) == '"a\\"b"'


def test_fts_literal_args_escapes_backslash() -> None:
    assert fts_literal_args(["a\\b"]) == '"a\\\\b"'


def test_fts_literal_args_escapes_backslash_before_quote() -> None:
    # Backslash is doubled first so it can't neutralize the quote escaping.
    assert fts_literal_args(['\\"']) == '"\\\\\\""'


def test_fts_literal_args_quote_breakout_payload_is_neutralized() -> None:
    result = fts_literal_args(['" OR "1"="1'])
    # every embedded double-quote is preceded by a backslash; no bare '"' survives
    # between the outer wrapping quotes.
    inner = result[1:-1]
    assert '\\"' in inner
    assert inner.replace('\\"', "").count('"') == 0
    assert result == '"\\" OR \\"1\\"=\\"1"'
