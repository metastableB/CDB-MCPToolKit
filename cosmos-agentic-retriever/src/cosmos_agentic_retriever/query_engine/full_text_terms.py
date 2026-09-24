"""Prepare search text for Cosmos DB's FullTextScore function.

tokenize_for_fts splits text into lowercase, unique terms. Cosmos handles
stopwords during indexing and search, so this helper does not filter them:
https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/stopwords

max_terms (30 by default) is an application budget, not a Cosmos limit.
Queries exceeding it raise ValueError instead of being silently truncated.
fts_literal_args quotes and escapes the terms for inclusion in SQL.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_MAX_FTS_TERMS = 30


def tokenize_for_fts(
    query: str, *, max_terms: int = DEFAULT_MAX_FTS_TERMS
) -> list[str]:
    """Prepare search terms, rejecting queries over the positive max_terms budget."""
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    all_terms: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query):
        term = raw.lower()
        if term in seen:
            continue
        seen.add(term)
        all_terms.append(term)
        if len(all_terms) > max_terms:
            raise ValueError(
                f"full-text query exceeds max_terms={max_terms}; "
                "shorten the query or increase the application term budget"
            )
    return all_terms


def fts_literal_args(terms: list[str]) -> str:
    def esc(t: str) -> str:
        return '"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"'

    return ", ".join(esc(t) for t in terms)
