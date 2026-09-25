"""Search configured containers concurrently and combine their ranked results.

Adapted from the original cross-collection retriever. Cosmos ranks within each
container; reciprocal-rank combination interleaves those lists, not their BM25
scores. Ties follow configured container order. Each hit keeps its original item
ID and adds a retrieval_id for its database, container, partition key and physical ID.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple
from urllib.parse import quote

import structlog

from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.query_engine.types import RetrievedItem, SearchRequest


class ContainerTarget(NamedTuple):
    database: str
    container: str


class ContainerItem(RetrievedItem):
    """A search item with source fields and an escaped, container-qualified ID."""

    database: str
    container: str
    retrieval_id: str


@dataclass
class MultiSearchResult:
    items: list[ContainerItem] = field(default_factory=list)
    searched: list[ContainerTarget] = field(default_factory=list)
    errors: dict[ContainerTarget, str] = field(default_factory=dict)


def fuse_rrf(
    ranked_lists: Sequence[tuple[ContainerTarget, list[RetrievedItem]]],
    *,
    limit: int,
) -> list[ContainerItem]:
    """Combine by local rank, preserving distinct items from different containers.

    For disjoint lists this yields A1, B1, A2, B2, up to limit. It does not infer
    globally comparable relevance. Inputs are left unchanged.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    scores: dict[tuple[ContainerTarget, str], float] = {}
    chosen: dict[tuple[ContainerTarget, str], ContainerItem] = {}
    for target, items in ranked_lists:
        for position, item in enumerate(items):
            if item.cosmos_identity is None:
                raise ValueError(
                    "physical Cosmos identity is required to combine results"
                )
            identity = item.cosmos_identity.key()
            key = (target, identity)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + position)
            if key not in chosen:
                database, container, physical_key = (
                    quote(value, safe="") for value in (*target, identity)
                )
                chosen[key] = ContainerItem(
                    **item.model_dump(),
                    database=target.database,
                    container=target.container,
                    retrieval_id=f"{database}/{container}:{physical_key}",
                )
    result = []
    for rank, key in enumerate(
        sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    ):
        item = chosen[key]
        item.rank = rank
        result.append(item)
    return result


class MultiContainerRetriever:
    """Run a separate request per configured target with bounded worker threads.

    Return successful targets and sanitized failures separately. The HTTP caller
    decides the status for partial or complete failure. Each underlying retriever
    retains its own schema and partition policy and may share a CosmosExecutor.
    """

    def __init__(
        self, retrievers: Mapping[ContainerTarget, CorpusRetriever], *, max_workers: int
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._retrievers = dict(retrievers)
        self._max_workers = max_workers
        self._logger = structlog.get_logger(__name__)

    def search(
        self, requests: Mapping[ContainerTarget, SearchRequest], *, limit: int
    ) -> MultiSearchResult:
        result = MultiSearchResult()
        if not requests:
            return result
        ranked_lists = []
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(requests))
        ) as pool:
            futures = {
                pool.submit(self._retrievers[target].search, request): target
                for target, request in requests.items()
            }
            for future, target in futures.items():
                try:
                    items = future.result()
                except Exception:
                    self._logger.exception(
                        "container_search_failed",
                        database=target.database,
                        container=target.container,
                    )
                    result.errors[target] = "Search failed."
                    continue
                result.searched.append(target)
                ranked_lists.append((target, items))
        result.items = fuse_rrf(ranked_lists, limit=limit)
        return result
