"""Execute a compiled Cosmos query and return its rows as a list.

The caller passes an `azure.cosmos.ContainerProxy` instance and a
`CompiledCosmosQuery` instance that both define the container to access, its
authentication and the query to execute on the container.  The ContainerProxy
should be configured with the target container, its endpoint,
credentials, and retry settings (configured from the parent CosmosClient).  The
credentials must permit queries on that container. The caller keeps the client
open during execution and closes it when finished.

Example Usage:
    config = QueryEngineConfig(max_concurrency=8, slow_query_warning_seconds=4.5)
    executor = CosmosExecutor(config=config)
    rows = executor.run(compiled, container=container)


- run() passes the SQL, parameter values, and partition settings to the
container client and collects the returned rows. It does not create clients or
containers.
- The caller supplies a QueryEngineConfig when creating the executor. Reuse that
executor for calls that must share a query limit, even across different containers.
Additional queries wait until a running query finishes fetching rows or fails.
Separate executors have independent limits, even if they use the same config.
- Configuration defaults to eight concurrent queries and warnings for successful
calls taking over 4.5 seconds, including time waiting to start. The warning does
not stop slow queries. This module reads no environment variables; CLI or environment
settings must be resolved by the caller before constructing the config.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import structlog
from azure.cosmos import ContainerProxy

from cosmos_agentic_retriever.query_engine.types import (
    CompiledCosmosQuery,
    QueryEngineConfig,
)


class CosmosExecutor:
    def __init__(self, *, config: QueryEngineConfig) -> None:
        self._config = config
        self._query_semaphore = threading.BoundedSemaphore(config.max_concurrency)
        self._logger = structlog.get_logger(
            "cosmos_agentic_retriever.query_engine.executor"
        )

    def run(
        self, compiled: CompiledCosmosQuery, *, container: ContainerProxy
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        with self._query_semaphore:
            kwargs: dict[str, Any] = {
                "query": compiled.sql,
                "parameters": compiled.parameters,
            }
            if compiled.partition_key is not None:
                kwargs["partition_key"] = compiled.partition_key
            elif compiled.enable_cross_partition_query:
                kwargs["enable_cross_partition_query"] = True
            result = list(container.query_items(**kwargs))
        elapsed_seconds = time.perf_counter() - start
        if elapsed_seconds > self._config.slow_query_warning_seconds:
            self._logger.warning(
                "slow_cosmos_query",
                elapsed_ms=round(elapsed_seconds * 1000, 1),
                cosmos_max_concurrency=self._config.max_concurrency,
            )
        return result
