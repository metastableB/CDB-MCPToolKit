"""Prepare search text for Cosmos DB's FullTextScore function.

tokenize_for_fts splits text into lowercase, unique terms and keeps at most
the caller-supplied max_terms. It removes common English words unless that would
leave no terms. fts_literal_args quotes and escapes the terms for inclusion in SQL.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# English-only stopword list. Full-text queries are assumed to be English; this
# list only removes English function words so they don't dominate FullTextScore.
# Non-English queries still work — `_TOKEN_RE` is Unicode-aware, so their tokens
# are tokenized/lower-cased/de-duplicated normally; their function words simply
# aren't stripped (mildly noisier, but RRF/FullTextScore tolerate it). If every
# term is a stopword, the original terms are retained so the query remains valid.
_STOPWORDS = frozenset(
    _TOKEN_RE.findall(
        "a about above after again against all am an and any are as at be because been "
        "before being below between both but by can did do does doing don down during "
        "each few for from further had has have having he her here hers herself him "
        "himself his how i if in into is it its itself just like me more most my myself "
        "no nor not now of off on once only or other our ours ourselves out over own "
        "please same she should so some such tell than that the their theirs them "
        "themselves then there these they this those through to too under until up very "
        "was we were what when where which while who whom why will with would you your "
        "yours yourself yourselves"
    )
)

def tokenize_for_fts(query: str, *, max_terms: int) -> list[str]:
    """Prepare search terms, capped by a positive integer max_terms."""
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    all_terms: list[str] = []
    searchable_terms: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query):
        term = raw.lower()
        if term in seen:
            continue
        seen.add(term)
        all_terms.append(term)
        if term not in _STOPWORDS:
            searchable_terms.append(term)
    return (searchable_terms or all_terms)[:max_terms]


def fts_literal_args(terms: list[str]) -> str:
    def esc(t: str) -> str:
        return '"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"'

    return ", ".join(esc(t) for t in terms)
