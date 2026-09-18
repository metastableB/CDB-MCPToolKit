from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from azure.cosmos import ContainerProxy

from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.executor import CosmosExecutor
from cosmos_agentic_retriever.query_engine.normalization import normalize_rows
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    CrossPartitionQueryDisabled,
    PartitionQueryPolicy,
    RetrievedItem,
    SearchRequest,
)


@dataclass
class RetrievalContext:
    schema: CorpusSchema
    compiler: CosmosQueryCompiler
    executor: CosmosExecutor
    container: ContainerProxy
    policy: PartitionQueryPolicy


def _resolve_cross_partition(
    req_partition_key: Any, policy: PartitionQueryPolicy
) -> bool:
    if req_partition_key is not None:
        return False
    if not policy.allow_cross_partition_search:
        raise CrossPartitionQueryDisabled(
            "search requires a partition key or cross-partition permission"
        )
    return True


class SearchStrategy(ABC):
    name: str = ""
    requires_embedding: bool = False

    @abstractmethod
    def execute(
        self, req: SearchRequest, ctx: RetrievalContext
    ) -> list[RetrievedItem]: ...


class FullTextSearchStrategy(SearchStrategy):
    name = "full_text"
    requires_embedding = False

    def execute(self, req: SearchRequest, ctx: RetrievalContext) -> list[RetrievedItem]:
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
        return normalize_rows(
            rows,
            strategy=self.name,
            channels=["full_text"],
            projected_aliases=compiled.projected_aliases,
            queried_text_fields=req.text_fields,
        )
