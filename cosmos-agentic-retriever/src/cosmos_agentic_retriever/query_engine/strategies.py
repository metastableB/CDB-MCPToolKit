"""Ask Cosmos DB to run a search and convert its rows into RetrievedItem objects.

Here, a "strategy" orchestrates a specific kind of search against Cosmos DB. For instance
- Cosmos native full-text, vector and hybrid search calls
- Custom client side search fusion/logic

This module currently implements only full-text search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from azure.cosmos import ContainerProxy

from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.executor import CosmosExecutor
from cosmos_agentic_retriever.query_engine.results_mapping import rows_to_items
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    CrossPartitionQueryDisabled,
    PartitionQueryPolicy,
    RetrievedItem,
    SearchRequest,
)


@dataclass
class RetrievalContext:
    """Carry the configured objects from CorpusRetriever to the search strategy.

    These are reused across calls. The query and result limit arrive separately
    in SearchRequest; this object does not hold per-request results.
    """

    schema: CorpusSchema
    compiler: CosmosQueryCompiler
    executor: CosmosExecutor
    container: ContainerProxy
    policy: PartitionQueryPolicy


def _resolve_cross_partition(
    req_partition_key: Any, policy: PartitionQueryPolicy
) -> bool:
    """Use a supplied partition key, or require permission to search across partitions."""
    if req_partition_key is not None:
        return False
    if not policy.allow_cross_partition_search:
        raise CrossPartitionQueryDisabled(
            "search requires a partition key or cross-partition permission"
        )
    return True


class SearchStrategy(ABC):
    """Give retrieval methods the same request-to-items calling interface.

    The original multi-method implementation used this interface to call whichever
    method its planner selected. The current caller always uses FullTextSearchStrategy.
    """

    name: str = ""
    requires_embedding: bool = False

    @abstractmethod
    def execute(
        self, req: SearchRequest, ctx: RetrievalContext
    ) -> list[RetrievedItem]: ...


class FullTextSearchStrategy(SearchStrategy):
    """Run Cosmos full-text search and return its ranked rows as RetrievedItem objects."""

    name = "full_text"
    requires_embedding = False

    def execute(self, req: SearchRequest, ctx: RetrievalContext) -> list[RetrievedItem]:
        """Check the request's fields and partition access, then compile, run, and map."""
        text_paths = ctx.schema.resolve_text_fields(req.text_fields)
        cross = _resolve_cross_partition(req.partition_key, ctx.policy)
        compiled = ctx.compiler.compile_full_text(
            query=req.query,
            limit=req.limit,
            ignored_item_ids=req.ignored_item_ids,
            filters=req.filters,
            partition_key=req.partition_key,
            cross_partition=cross,
            text_paths=text_paths,
            max_terms=req.max_terms,
        )
        rows = ctx.executor.run(compiled, container=ctx.container)
        return rows_to_items(
            rows,
            strategy=self.name,
            channels=["full_text"],
            projected_aliases=compiled.projected_aliases,
            queried_text_fields=req.text_fields,
        )
