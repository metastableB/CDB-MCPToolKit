"""Execute compiled Cosmos queries and collect their rows.

The supplied Cosmos client owns request retries. Errors propagate unchanged;
this module does not restart queries or return partially collected results.
COSMOS_QUERY_MAX_CONCURRENCY, read at import, limits active queries across executor
instances in this process (default 8). The slot stays held while rows are fetched.
Successful calls taking over 4.5 seconds, including time waiting for a slot, are logged.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import structlog
from azure.cosmos import ContainerProxy

from cosmos_agentic_retriever.query_engine.types import CompiledCosmosQuery

logger = structlog.get_logger("cosmos_agentic_retriever.query_engine.executor")


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid_int_env", name=name, value=raw, default=default)
        return default
    if value < 1:
        logger.warning(
            "invalid_positive_int_env", name=name, value=raw, default=default
        )
        return default
    return value


COSMOS_QUERY_MAX_CONCURRENCY = _read_positive_int_env("COSMOS_QUERY_MAX_CONCURRENCY", 8)
_COSMOS_QUERY_SEMAPHORE = threading.BoundedSemaphore(COSMOS_QUERY_MAX_CONCURRENCY)


def _query_items(
    container: ContainerProxy,
    query: str,
    parameters: list[dict[str, Any]],
    *,
    partition_key: Any | None,
    enable_cross_partition_query: bool,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    with _COSMOS_QUERY_SEMAPHORE:
        kwargs: dict[str, Any] = {"query": query, "parameters": parameters}
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        elif enable_cross_partition_query:
            kwargs["enable_cross_partition_query"] = True
        result = list(container.query_items(**kwargs))
    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > 4500:
        logger.warning(
            "slow_cosmos_query",
            elapsed_ms=round(elapsed_ms, 1),
            cosmos_max_concurrency=COSMOS_QUERY_MAX_CONCURRENCY,
        )
    return result


class CosmosExecutor:
    def __init__(self, container: ContainerProxy) -> None:
        self._container = container

    def run(self, compiled: CompiledCosmosQuery) -> list[dict[str, Any]]:
        return _query_items(
            self._container,
            compiled.sql,
            compiled.parameters,
            partition_key=compiled.partition_key,
            enable_cross_partition_query=compiled.enable_cross_partition_query,
        )
