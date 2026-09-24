"""Exhaustive tests for `cosmos_agentic_retriever.query_engine.full_text_terms`.

Covers FTS tokenization (Unicode, lowering, dedup, retained stopwords, and term
budget validation) and, critically for security, the escaping in
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


def test_tokenize_preserves_stopwords_for_cosmos() -> None:
    assert tokenize_for_fts("the cat and the dog") == ["the", "cat", "and", "dog"]


def test_tokenize_apostrophe_splits_without_filtering() -> None:
    assert tokenize_for_fts("don't") == ["don", "t"]


def test_tokenize_unicode_accented_words() -> None:
    assert tokenize_for_fts("Café Über") == ["café", "über"]


def test_tokenize_unicode_cjk() -> None:
    assert tokenize_for_fts("机器 学习 机器") == ["机器", "学习"]


def test_tokenize_preserves_all_stopword_queries() -> None:
    assert tokenize_for_fts("the and of") == ["the", "and", "of"]
    assert tokenize_for_fts("THE AND OF") == ["the", "and", "of"]


def test_tokenize_rejects_queries_exceeding_default_budget() -> None:
    assert DEFAULT_MAX_FTS_TERMS == 30
    terms = [f"w{index}" for index in range(DEFAULT_MAX_FTS_TERMS)]
    assert tokenize_for_fts(" ".join(terms)) == terms
    with pytest.raises(ValueError, match="exceeds max_terms=30"):
        tokenize_for_fts(" ".join(terms + ["recycling"]))


def test_tokenize_cap_counts_distinct_only() -> None:
    # Duplicates must not consume the term budget.
    distinct = [f"t{i}" for i in range(DEFAULT_MAX_FTS_TERMS)]
    query = " ".join(distinct + distinct)
    assert tokenize_for_fts(query) == distinct
    with pytest.raises(ValueError, match="exceeds max_terms=30"):
        tokenize_for_fts(query + " extra_beyond_cap")


@pytest.mark.parametrize("max_terms", [1, 3, 40])
def test_tokenize_accepts_per_call_limit(max_terms: int) -> None:
    terms = [f"term{index}" for index in range(max_terms)]
    query = " ".join(terms)
    assert tokenize_for_fts(query, max_terms=max_terms) == terms
    with pytest.raises(ValueError, match=f"exceeds max_terms={max_terms}"):
        tokenize_for_fts(query + " overflow", max_terms=max_terms)


def test_custom_limit_counts_stopwords_after_deduplication() -> None:
    assert tokenize_for_fts("the battery BATTERY recycling", max_terms=3) == [
        "the",
        "battery",
        "recycling",
    ]
    with pytest.raises(ValueError, match="exceeds max_terms=2"):
        tokenize_for_fts("the battery BATTERY recycling", max_terms=2)


def test_custom_limit_applies_to_all_stopword_queries() -> None:
    assert tokenize_for_fts("the and THE", max_terms=2) == ["the", "and"]
    with pytest.raises(ValueError, match="exceeds max_terms=2"):
        tokenize_for_fts("the and THE of", max_terms=2)


def test_stopword_prefix_cannot_silently_discard_search_terms() -> None:
    prefix = "a about above after again against all am an and any are as at be because been before being below"
    query = prefix + " between both but by can did do does doing down battery recycling"
    with pytest.raises(ValueError, match="exceeds max_terms=30"):
        tokenize_for_fts(query)
    assert tokenize_for_fts(query, max_terms=32)[-2:] == ["battery", "recycling"]


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
