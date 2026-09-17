import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from azure.cosmos import ContainerProxy
from azure.cosmos.exceptions import CosmosHttpResponseError
from structlog.testing import capture_logs

from cosmos_agentic_retriever.query_engine import executor
from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.executor import CosmosExecutor
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    CompiledCosmosQuery,
    EqualsFilter,
)


@pytest.mark.parametrize("rows", [[], [{"item_id": "first"}, {"item_id": "second"}]])
@pytest.mark.parametrize(
    "partition_key, cross_partition, routing",
    [
        ("tenant-a", False, {"partition_key": "tenant-a"}),
        (0, True, {"partition_key": 0}),
        ("", True, {"partition_key": ""}),
        (None, True, {"enable_cross_partition_query": True}),
        (None, False, {}),
    ],
)
def test_runs_compiler_output(partition_key, cross_partition, routing, rows) -> None:
    compiler = CosmosQueryCompiler(
        CorpusSchema(item_id_path="/id", text_paths=["/text"])
    )
    compiled = compiler.compile_structured(
        limit=2,
        filters=[EqualsFilter(logical_field="item_id", value="item-1")],
        ignored_item_ids=[],
        partition_key=partition_key,
        cross_partition=cross_partition,
    )
    container = Mock(spec=ContainerProxy)
    container.query_items.return_value = iter(rows)

    assert CosmosExecutor(container).run(compiled) == rows
    container.query_items.assert_called_once_with(
        query=compiled.sql, parameters=compiled.parameters, **routing
    )


@pytest.mark.parametrize("status", [400, 429, 503, None])
@pytest.mark.parametrize("during_iteration", [False, True])
def test_propagates_failure_without_restarting_query(
    monkeypatch, status, during_iteration
) -> None:
    failure = (
        CosmosHttpResponseError(status_code=status, message="query failed")
        if status is not None
        else RuntimeError("iteration failed")
    )
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(executor, "_COSMOS_QUERY_SEMAPHORE", semaphore)

    def failing_rows():
        yield {"item_id": "partial"}
        raise failure

    container = Mock(spec=ContainerProxy)
    if during_iteration:
        container.query_items.side_effect = lambda **kwargs: failing_rows()
    else:
        container.query_items.side_effect = failure
    compiled = CompiledCosmosQuery(sql="SELECT * FROM c")
    runner = CosmosExecutor(container)
    with pytest.raises(type(failure)) as caught:
        runner.run(compiled)
    assert caught.value is failure
    container.query_items.assert_called_once()
    assert semaphore.acquire(blocking=False)
    semaphore.release()

    container.query_items.side_effect = None
    container.query_items.return_value = iter([{"item_id": "complete"}])
    assert runner.run(compiled) == [{"item_id": "complete"}]


def test_concurrency_limit_covers_iteration_across_executors(monkeypatch) -> None:
    first_page = threading.Event()
    release_first = threading.Event()
    second_attempt = threading.Event()

    class ObservedSemaphore(threading.BoundedSemaphore):
        def __enter__(self):
            if first_page.is_set():
                second_attempt.set()
            return super().__enter__()

    monkeypatch.setattr(executor, "_COSMOS_QUERY_SEMAPHORE", ObservedSemaphore(1))

    def query_items(**kwargs):
        if not first_page.is_set():
            yield {"item_id": "first"}
            first_page.set()
            assert release_first.wait(5), "first query was not released"
        yield {"item_id": "last"}

    container = Mock(spec=ContainerProxy)
    container.query_items.side_effect = query_items
    compiled = CompiledCosmosQuery(sql="SELECT * FROM c")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(CosmosExecutor(container).run, compiled)
        try:
            assert first_page.wait(5), "first query did not start iterating"
            second = pool.submit(CosmosExecutor(container).run, compiled)
            assert second_attempt.wait(5), "second query did not attempt to acquire"
            assert container.query_items.call_count == 1
        finally:
            release_first.set()
        assert first.result(timeout=5) == [{"item_id": "first"}, {"item_id": "last"}]
        assert second.result(timeout=5) == [{"item_id": "last"}]
    assert container.query_items.call_count == 2


@pytest.mark.parametrize("elapsed, slow", [(4.5, False), (4.501, True)])
def test_logs_only_slow_successful_queries(monkeypatch, elapsed, slow) -> None:
    monkeypatch.setattr(executor.time, "perf_counter", Mock(side_effect=[0.0, elapsed]))
    container = Mock(spec=ContainerProxy)
    container.query_items.return_value = iter([])
    with capture_logs() as logs:
        assert (
            CosmosExecutor(container).run(CompiledCosmosQuery(sql="SELECT * FROM c"))
            == []
        )
    assert logs == (
        [
            {
                "event": "slow_cosmos_query",
                "log_level": "warning",
                "elapsed_ms": 4501.0,
                "cosmos_max_concurrency": executor.COSMOS_QUERY_MAX_CONCURRENCY,
            }
        ]
        if slow
        else []
    )


@pytest.mark.parametrize(
    "raw, expected, event",
    [
        (None, 8, None),
        ("2", 2, None),
        ("0", 8, "invalid_positive_int_env"),
        ("-1", 8, "invalid_positive_int_env"),
        ("bad", 8, "invalid_int_env"),
        ("", 8, "invalid_int_env"),
    ],
)
def test_concurrency_environment_validation(monkeypatch, raw, expected, event) -> None:
    name = "COSMOS_QUERY_MAX_CONCURRENCY"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    with capture_logs() as logs:
        assert executor._read_positive_int_env(name, 8) == expected
    assert [entry["event"] for entry in logs] == ([event] if event else [])
