import copy
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from azure.cosmos import ContainerProxy
from azure.cosmos.exceptions import CosmosHttpResponseError
from pydantic import ValidationError
from structlog.testing import capture_logs

from cosmos_agentic_retriever.query_engine import (
    CosmosExecutor,
    QueryEngineConfig,
    executor,
)
from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    CompiledCosmosQuery,
    EqualsFilter,
)


@pytest.fixture
def config() -> QueryEngineConfig:
    return QueryEngineConfig(max_concurrency=1)


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
def test_runs_compiler_output(
    partition_key, cross_partition, routing, rows, config
) -> None:
    compiler = CosmosQueryCompiler(
        CorpusSchema(item_id_path="/id", text_paths=["/text"])
    )
    compiled = compiler.compile_structured(
        limit=2,
        filters=[EqualsFilter(path="/id", value="item-1")],
        ignored_item_ids=[],
        partition_key=partition_key,
        cross_partition=cross_partition,
    )
    container = Mock(spec=ContainerProxy)
    container.query_items.return_value = iter(rows)

    assert CosmosExecutor(config=config).run(compiled, container=container) == rows
    container.query_items.assert_called_once_with(
        query=compiled.sql, parameters=compiled.parameters, **routing
    )


@pytest.mark.parametrize("status", [400, 429, 503, None])
@pytest.mark.parametrize("during_iteration", [False, True])
def test_propagates_failure_without_restarting_query(
    status, during_iteration, config
) -> None:
    failure = (
        CosmosHttpResponseError(status_code=status, message="query failed")
        if status is not None
        else RuntimeError("iteration failed")
    )
    def failing_rows():
        yield {"item_id": "partial"}
        raise failure

    container = Mock(spec=ContainerProxy)
    if during_iteration:
        container.query_items.side_effect = lambda **kwargs: failing_rows()
    else:
        container.query_items.side_effect = failure
    compiled = CompiledCosmosQuery(sql="SELECT * FROM c")
    runner = CosmosExecutor(config=config)
    with pytest.raises(type(failure)) as caught:
        runner.run(compiled, container=container)
    assert caught.value is failure
    container.query_items.assert_called_once()
    assert runner._query_semaphore.acquire(blocking=False)
    runner._query_semaphore.release()

    container.query_items.side_effect = None
    container.query_items.return_value = iter([{"item_id": "complete"}])
    assert runner.run(compiled, container=container) == [{"item_id": "complete"}]


@pytest.mark.parametrize(
    "limit, shared, fail_running",
    [(1, True, False), (1, True, True), (2, True, False), (2, True, True), (1, False, False)],
    ids=["one-slot", "one-slot-failure", "two-slots", "two-slots-failure", "independent"],
)
def test_executor_sharing_limits_queries_across_containers(
    monkeypatch, limit, shared, fail_running
) -> None:
    waiting = threading.Event()
    started = [threading.Event() for _ in range(limit + 1)]
    release = [threading.Event() for _ in started]
    released_index = limit - 1
    failure = CosmosHttpResponseError(status_code=503, message="page fetch failed")

    class ObservedSemaphore(threading.BoundedSemaphore):
        def __enter__(self, blocking: bool = True, timeout: float | None = None):
            if super().acquire(blocking=False):
                return True
            waiting.set()
            assert super().__enter__(blocking, 5 if timeout is None else timeout), (
                "waiting query never acquired a permit"
            )
            return True

    monkeypatch.setattr(executor.threading, "BoundedSemaphore", ObservedSemaphore)

    def rows_for(index):
        yield {"item_id": f"{index}-first"}
        started[index].set()
        assert release[index].wait(5), "running query was not released"
        if fail_running and index == released_index:
            raise failure
        yield {"item_id": f"{index}-last"}

    containers = [Mock(spec=ContainerProxy) for _ in started]
    queries = [
        CompiledCosmosQuery(
            sql=f'SELECT c["field_{index}"] FROM c WHERE c["id"] = @id',
            parameters=[{"name": "@id", "value": str(index)}],
            partition_key=0 if index == 0 else None,
            enable_cross_partition_query=index != 0,
        )
        for index in range(len(started))
    ]
    for index, container in enumerate(containers):
        container.query_items.side_effect = lambda index=index, **kwargs: rows_for(index)
    config = QueryEngineConfig(max_concurrency=limit)
    runner = CosmosExecutor(config=config)
    overflow_runner = runner if shared else CosmosExecutor(config=config)
    with ThreadPoolExecutor(max_workers=len(started)) as pool:
        futures = [
            pool.submit(runner.run, queries[index], container=containers[index])
            for index in range(limit)
        ]
        try:
            for event in started[:limit]:
                assert event.wait(5), "configured capacity did not run concurrently"
            futures.append(pool.submit(overflow_runner.run, queries[-1], container=containers[-1]))
            if shared:
                assert waiting.wait(5), "overflow query did not encounter the limit"
                containers[-1].query_items.assert_not_called()
                release[released_index].set()
            assert started[-1].wait(5), "overflow query did not start"
            if shared and limit > 1:
                assert not futures[0].done(), "the other active query should still be running"
        finally:
            for event in release:
                event.set()
        for index, future in enumerate(futures):
            if fail_running and index == released_index:
                with pytest.raises(CosmosHttpResponseError) as caught:
                    future.result(timeout=5)
                assert caught.value is failure
            else:
                assert future.result(timeout=5) == [
                    {"item_id": f"{index}-first"}, {"item_id": f"{index}-last"}
                ]
    for query, container in zip(queries, containers, strict=True):
        routing = (
            {"partition_key": query.partition_key}
            if query.partition_key is not None
            else {"enable_cross_partition_query": True}
        )
        container.query_items.assert_called_once_with(
            query=query.sql, parameters=query.parameters, **routing
        )


def test_slow_query_warning_includes_waiting_time(monkeypatch) -> None:
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(executor.time, "perf_counter", clock)

    class SimulatedWaitSemaphore(threading.BoundedSemaphore):
        def __enter__(self, blocking: bool = True, timeout: float | None = None):
            acquired = super().__enter__(blocking, timeout)
            clock.return_value = 3.0
            return acquired

    monkeypatch.setattr(executor.threading, "BoundedSemaphore", SimulatedWaitSemaphore)

    def query_items(**kwargs):
        clock.return_value = 3.25
        yield {"item_id": "result"}

    container = Mock(spec=ContainerProxy)
    container.query_items.side_effect = query_items
    runner = CosmosExecutor(config=QueryEngineConfig(slow_query_warning_seconds=2.0))
    with capture_logs() as logs:
        assert runner.run(CompiledCosmosQuery(sql="SELECT * FROM c"), container=container) == [
            {"item_id": "result"}
        ]
    assert len(logs) == 1
    assert logs[0]["event"] == "slow_cosmos_query"
    assert logs[0]["elapsed_ms"] == 3250.0


@pytest.mark.parametrize("threshold", [4.5, 1.25])
@pytest.mark.parametrize("extra_seconds, slow", [(0.0, False), (0.001, True)])
def test_logs_only_slow_successful_queries(
    monkeypatch, threshold, extra_seconds, slow
) -> None:
    config = QueryEngineConfig(
        max_concurrency=1, slow_query_warning_seconds=threshold
    )
    elapsed = threshold + extra_seconds
    monkeypatch.setattr(executor.time, "perf_counter", Mock(side_effect=[0.0, elapsed]))
    container = Mock(spec=ContainerProxy)
    container.query_items.return_value = iter([])
    with capture_logs() as logs:
        assert (
            CosmosExecutor(config=config).run(
                CompiledCosmosQuery(sql="SELECT * FROM c"), container=container
            )
            == []
        )
    assert logs == (
        [
            {
                "event": "slow_cosmos_query",
                "log_level": "warning",
                "elapsed_ms": round(elapsed * 1000, 1),
                "cosmos_max_concurrency": config.max_concurrency,
            }
        ]
        if slow
        else []
    )


def test_config_defaults_ignore_environment(monkeypatch) -> None:
    monkeypatch.setenv("COSMOS_QUERY_MAX_CONCURRENCY", "not-a-number")
    assert QueryEngineConfig().model_dump() == {
        "max_concurrency": 8,
        "slow_query_warning_seconds": 4.5,
    }


@pytest.mark.parametrize(
    "field, value",
    [("max_concurrency", value) for value in (0, -1, True, 1.5, "2")]
    + [
        ("slow_query_warning_seconds", value)
        for value in (0, -1, True, float("nan"), float("inf"), "2")
    ],
)
def test_config_rejects_invalid_values(field, value) -> None:
    with pytest.raises(ValidationError):
        QueryEngineConfig(**{field: value})


def test_config_is_immutable_and_rejects_unknown_fields() -> None:
    config = QueryEngineConfig(max_concurrency=2, slow_query_warning_seconds=1.25)
    assert config.max_concurrency == 2
    assert config.slow_query_warning_seconds == 1.25
    with pytest.raises(ValidationError):
        config.max_concurrency = 4
    with pytest.raises(ValidationError):
        QueryEngineConfig(max_concurency=2)


@pytest.mark.parametrize(
    "round_trip",
    [
        lambda config: QueryEngineConfig.model_validate_json(config.model_dump_json()),
        lambda config: pickle.loads(pickle.dumps(config)),
        copy.deepcopy,
        lambda config: config.model_copy(deep=True),
    ],
    ids=["json", "pickle", "deepcopy", "pydantic-deepcopy"],
)
def test_config_serializes_after_executor_creation(config, round_trip) -> None:
    runner = CosmosExecutor(config=config)
    restored = round_trip(config)
    assert restored == config
    assert restored is not config
    assert not hasattr(config, "_query_semaphore")
    second = CosmosExecutor(config=restored)
    assert second._query_semaphore is not runner._query_semaphore


@pytest.mark.parametrize("limit", [1, 3])
def test_executor_creates_private_limiter_with_configured_capacity(limit) -> None:
    config = QueryEngineConfig(max_concurrency=limit)
    runner = CosmosExecutor(config=config)
    acquired = [runner._query_semaphore.acquire(blocking=False) for _ in range(limit + 1)]
    try:
        assert acquired == [True] * limit + [False]
    finally:
        for held in acquired:
            if held:
                runner._query_semaphore.release()
