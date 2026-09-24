"""Prepare search text for Cosmos DB's FullTextScore function.

tokenize_for_fts splits text into lowercase, unique terms in first-seen order.
fts_literal_args quotes and escapes the terms for inclusion in SQL.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_for_fts(query: str) -> list[str]:
    """Split the query into lowercase, unique terms in first-seen order."""
    all_terms: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query):
        term = raw.lower()
        if term in seen:
            continue
        seen.add(term)
        all_terms.append(term)
    return all_terms


def fts_literal_args(terms: list[str]) -> str:
    def esc(t: str) -> str:
        return '"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"'

    return ", ".join(esc(t) for t in terms)
