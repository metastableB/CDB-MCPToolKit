"""Search a configured Cosmos container and return ranked items.

The caller supplies a schema, a container with full-text search configured for the
selected fields, and an executor. Reuse the executor across retrievers to share its
concurrency limit. This class selects full-text search explicitly; it does not
discover indexes, generate embeddings, or assemble document chunks.

Usage Example: Assumes container is a configured Cosmos SDK ContainerProxy and
schema maps its item ID and returned text fields. Only selected text fields need
the required full-text policy/index. The caller manages the Cosmos client's lifetime.

    executor = CosmosExecutor(config=QueryEngineConfig())
    retriever = CorpusRetriever(container=container, schema=schema, executor=executor)
    items = retriever.search(SearchRequest(query="battery recycling", limit=5))

Import CosmosExecutor and QueryEngineConfig from cosmos_agentic_retriever.query_engine,
CorpusRetriever from this module, and SearchRequest from query_engine.types.
"""

from __future__ import annotations

from azure.cosmos import ContainerProxy

from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.executor import CosmosExecutor
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.strategies import (
    FullTextSearchStrategy,
    RetrievalContext,
)
from cosmos_agentic_retriever.query_engine.types import (
    PartitionQueryPolicy,
    RetrievedItem,
    SearchRequest,
)


class CorpusRetriever:
    def __init__(
        self,
        *,
        container: ContainerProxy,
        schema: CorpusSchema,
        executor: CosmosExecutor,
        partition_policy: PartitionQueryPolicy | None = None,
    ) -> None:
        self.schema = schema
        self.policy = partition_policy or PartitionQueryPolicy()
        self._compiler = CosmosQueryCompiler(schema)
        self._executor = executor
        self._ctx = RetrievalContext(
            schema=schema,
            compiler=self._compiler,
            executor=self._executor,
            container=container,
            policy=self.policy,
        )

    def search(self, request: SearchRequest) -> list[RetrievedItem]:
        if request.text_fields:
            self.schema.resolve_text_fields(request.text_fields)
        strategy = FullTextSearchStrategy()
        return strategy.execute(request, self._ctx)
