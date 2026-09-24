from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from azure.cosmos import ContainerProxy
from azure.cosmos.exceptions import CosmosHttpResponseError
from pydantic import ValidationError

from cosmos_agentic_retriever.query_engine import (
    CorpusSchema,
    CosmosExecutor,
    QueryEngineConfig,
)
from cosmos_agentic_retriever.query_engine import retriever as retr_mod
from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.query_engine.types import (
    CrossPartitionQueryDisabled,
    EqualsFilter,
    PartitionQueryPolicy,
    QueryCompilationError,
    RangeFilter,
    RetrievedItem,
    SearchRequest,
    UnknownField,
)


class _CallRecorder:
    def __init__(self, return_value: object) -> None:
        self.return_value = return_value
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value


class FakeSchema:
    def __init__(self) -> None:
        self.text_calls: list = []
        self.vector_calls: list = []

    def resolve_vector_config(self, name):
        self.vector_calls.append(name)
        return object()

    def resolve_text_fields(self, names):
        self.text_calls.append(names)
        return []


class FakeSearchStrategy:
    def __init__(self, result: list[RetrievedItem]) -> None:
        self._result = result
        self.execute_calls: list[tuple] = []

    def execute(self, req, ctx):
        self.execute_calls.append((req, ctx))
        return self._result


def _item(item_id: str, text: str = "") -> RetrievedItem:
    return RetrievedItem(item_id=item_id, text=text)


def _build(monkeypatch, *, search_strategy=None, policy=None) -> SimpleNamespace:
    schema = FakeSchema()
    compiler_sentinel = object()
    executor_sentinel = object()
    ctx_sentinel = object()
    strategy = search_strategy or FakeSearchStrategy([])
    comp_rec = _CallRecorder(compiler_sentinel)
    ctx_rec = _CallRecorder(ctx_sentinel)
    strategy_rec = _CallRecorder(strategy)
    monkeypatch.setattr(retr_mod, "CosmosQueryCompiler", comp_rec)
    monkeypatch.setattr(retr_mod, "RetrievalContext", ctx_rec)
    monkeypatch.setattr(retr_mod, "FullTextSearchStrategy", strategy_rec)
    container = object()
    retriever = CorpusRetriever(
        container=container,
        schema=schema,
        executor=executor_sentinel,
        partition_policy=policy,
    )
    return SimpleNamespace(
        retriever=retriever,
        schema=schema,
        container=container,
        compiler_sentinel=compiler_sentinel,
        executor_sentinel=executor_sentinel,
        ctx_sentinel=ctx_sentinel,
        comp_rec=comp_rec,
        ctx_rec=ctx_rec,
        strategy=strategy,
        strategy_rec=strategy_rec,
    )


def test_init_defaults_policy_to_partition_query_policy(monkeypatch) -> None:
    built = _build(monkeypatch)
    assert isinstance(built.retriever.policy, PartitionQueryPolicy)


def test_init_uses_provided_policy(monkeypatch) -> None:
    policy = PartitionQueryPolicy()
    built = _build(monkeypatch, policy=policy)
    assert built.retriever.policy is policy


def test_init_stores_schema(monkeypatch) -> None:
    built = _build(monkeypatch)
    assert built.retriever.schema is built.schema


def test_search_uses_injected_executor(monkeypatch) -> None:
    built = _build(monkeypatch)
    assert built.comp_rec.calls == [((built.schema,), {})]
    assert built.retriever._compiler is built.compiler_sentinel
    assert built.retriever._executor is built.executor_sentinel


def test_init_constructs_context_with_keywords(monkeypatch) -> None:
    built = _build(monkeypatch)
    ((_, kwargs),) = built.ctx_rec.calls
    assert kwargs == {
        "schema": built.schema,
        "compiler": built.compiler_sentinel,
        "executor": built.executor_sentinel,
        "container": built.container,
        "policy": built.retriever.policy,
    }
    assert built.retriever._ctx is built.ctx_sentinel


def test_search_skips_vector_resolution_when_absent(monkeypatch) -> None:
    built = _build(monkeypatch)
    built.retriever.search(SearchRequest(query="q"))
    assert built.schema.vector_calls == []


def test_search_resolves_text_fields_when_present(monkeypatch) -> None:
    built = _build(monkeypatch)
    built.retriever.search(SearchRequest(query="q", text_fields=["/a", "/b"]))
    assert built.schema.text_calls == [["/a", "/b"]]


def test_search_skips_text_resolution_when_none_or_empty(monkeypatch) -> None:
    built = _build(monkeypatch)
    built.retriever.search(SearchRequest(query="q", text_fields=None))
    built.retriever.search(SearchRequest(query="q", text_fields=[]))
    assert built.schema.text_calls == []


def test_search_selects_full_text_and_returns_execute_result(monkeypatch) -> None:
    items = [_item("a"), _item("b")]
    built = _build(monkeypatch, search_strategy=FakeSearchStrategy(items))
    req = SearchRequest(query="q")
    assert built.retriever.search(req) is items
    assert built.strategy_rec.calls == [((), {})]
    assert built.strategy.execute_calls == [(req, built.ctx_sentinel)]


def test_search_passes_ctx_identity_to_execute(monkeypatch) -> None:
    built = _build(monkeypatch)
    req = SearchRequest(query="q")
    built.retriever.search(req)
    ((passed_req, passed_ctx),) = built.strategy.execute_calls
    assert passed_req is req
    assert passed_ctx is built.retriever._ctx


def test_search_no_embedding_needed_ignores_missing_embedder(monkeypatch) -> None:
    built = _build(monkeypatch)
    req = SearchRequest(query="q")
    built.retriever.search(req)
    assert built.strategy.execute_calls[0][0] is req


def _search_path(schema=None, policy=None, executor=None):
    container = Mock(spec=ContainerProxy)
    container.query_items.return_value = iter([])
    return CorpusRetriever(
        container=container,
        schema=schema
        if schema is not None
        else CorpusSchema(item_id_path="/id", text_paths=["/text"]),
        executor=executor
        if executor is not None
        else CosmosExecutor(config=QueryEngineConfig()),
        partition_policy=policy,
    ), container


def test_search_request_to_rows() -> None:
    schema = CorpusSchema(
        item_id_path="/id",
        text_paths=["/content/body", "/headline"],
        metadata_paths={"year": "/publication/year"},
    )
    retriever, container = _search_path(schema)
    container.query_items.return_value = iter(
        [
            {"item_id": "a", "txt_0": "A", "txt_1": "Title A", "md_0": 2024},
            {"item_id": "b", "txt_0": "B", "txt_1": "Title B", "md_0": 2021},
        ]
    )
    request = SearchRequest(
        query="battery recycling",
        limit=2,
        text_fields=["/content/body"],
        partition_key=0,
        filters=[
            RangeFilter(logical_field="year", minimum=2020),
            EqualsFilter(logical_field="item_id", value="a"),
        ],
        ignored_item_ids=["skip"],
    )
    before = request.model_dump()
    results = retriever.search(request)
    container.query_items.assert_called_once_with(
        query='SELECT TOP @k0 c["id"] AS item_id, c["content"]["body"] AS txt_0, '
        'c["headline"] AS txt_1, c["publication"]["year"] AS md_0 FROM c '
        'WHERE (c["publication"]["year"] >= @p1) AND c["id"] = @p2 '
        'AND NOT ARRAY_CONTAINS(@p3, c["id"]) '
        'ORDER BY RANK FullTextScore(c["content"]["body"], "battery", "recycling")',
        parameters=[
            {"name": "@k0", "value": 2},
            {"name": "@p1", "value": 2020},
            {"name": "@p2", "value": "a"},
            {"name": "@p3", "value": ["skip"]},
        ],
        partition_key=0,
    )
    assert [item.item_id for item in results] == ["a", "b"]
    assert [item.rank for item in results] == [0, 1]
    assert [item.text for item in results] == ["A", "B"]
    assert [item.metadata for item in results] == [{"year": 2024}, {"year": 2021}]
    assert results[0].text_fields == {"/content/body": "A", "/headline": "Title A"}
    assert all(
        item.retrieval_channels == ["full_text"]
        and item.retrieval_strategy == "full_text"
        for item in results
    )
    assert request.model_dump() == before


@pytest.mark.parametrize("term_count", [31, 100])
def test_search_preserves_all_query_terms(term_count) -> None:
    retriever, container = _search_path()
    terms = [f"term{index}" for index in range(term_count)]
    request = SearchRequest(query=" ".join(terms))
    assert retriever.search(request) == []
    literal_terms = ", ".join(f'"{term}"' for term in terms)
    container.query_items.assert_called_once_with(
        query='SELECT TOP @k0 c["id"] AS item_id, c["text"] AS txt_0 FROM c '
        f'ORDER BY RANK FullTextScore(c["text"], {literal_terms})',
        parameters=[{"name": "@k0", "value": 50}],
        enable_cross_partition_query=True,
    )


@pytest.mark.parametrize(
    "key, allowed",
    [(None, True), (None, False), ("tenant", False), (0, False), ("", False)],
)
def test_search_partition_policy(key, allowed) -> None:
    retriever, container = _search_path(
        policy=PartitionQueryPolicy(allow_cross_partition_search=allowed)
    )
    request = SearchRequest(query="battery", partition_key=key)
    if key is None and not allowed:
        with pytest.raises(CrossPartitionQueryDisabled):
            retriever.search(request)
        container.query_items.assert_not_called()
    else:
        assert retriever.search(request) == []
        arguments = container.query_items.call_args.kwargs
        routing = {
            name: value
            for name, value in arguments.items()
            if name not in ("query", "parameters")
        }
        assert routing == (
            {"partition_key": key}
            if key is not None
            else {"enable_cross_partition_query": True}
        )


@pytest.mark.parametrize(
    "paths, names, expected",
    [
        (["/text"], None, None),
        (["/text"], [], None),
        (["/content/body"], ["/content/body"], None),
        (["/content/body"], ["body"], UnknownField),
        (["/text"], ["text"], UnknownField),
        (["/text"], ["/missing"], UnknownField),
        (["/a/text", "/b/text"], None, UnknownField),
        (["/a/text", "/b/text"], ["/b/text"], None),
        (["/a/text", "/b/text"], ["text"], UnknownField),
        (["/a/text", "/b/text"], ["/b/text#2"], UnknownField),
        ([], None, QueryCompilationError),
    ],
)
def test_search_resolves_text_fields(paths, names, expected) -> None:
    schema = CorpusSchema(item_id_path="/id", text_paths=paths)
    retriever, container = _search_path(schema)
    request = SearchRequest(query="battery", text_fields=names)
    if expected:
        with pytest.raises(expected):
            retriever.search(request)
        container.query_items.assert_not_called()
    else:
        assert retriever.search(request) == []
        selected = schema.resolve_text_fields(names)
        assert (
            f'FullTextScore({selected[0].render()}, "battery")'
            in container.query_items.call_args.kwargs["query"]
        )


@pytest.mark.parametrize(
    "paths",
    [["/article/text", "/summary/text"], ['/"article/text"', "/article/text"]],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_search_uses_full_paths_in_selection_and_results(paths, reverse) -> None:
    configured = list(reversed(paths)) if reverse else paths
    schema = CorpusSchema(item_id_path="/id", text_paths=configured)
    retriever, container = _search_path(schema)
    row = {"item_id": "item-1"}
    for index, path in enumerate(configured):
        row[f"txt_{index}"] = f"content of {path}"
    container.query_items.return_value = iter([row])
    selected = list(reversed(paths))
    result = retriever.search(SearchRequest(query="battery", text_fields=selected))[0]
    assert result.text_fields == {path: f"content of {path}" for path in paths}
    assert result.text == "\n\n".join(
        f"[{path}]\ncontent of {path}" for path in selected
    )
    scores = ", ".join(
        f'FullTextScore({path.render()}, "battery")'
        for path in schema.resolve_text_fields(selected)
    )
    assert container.query_items.call_args.kwargs["query"].endswith(
        f"ORDER BY RANK RRF({scores})"
    )


@pytest.mark.parametrize(
    "arguments",
    [{"limit": value} for value in (True, 0, -1, 1.5, "2", None)]
    + [
        {"max_terms": 0},
        {"max_terms": True},
        {"query": ""},
        {"mode": "vector"},
        {"query_vector": [0.1]},
    ],
)
def test_search_rejects_invalid_request(arguments) -> None:
    retriever, container = _search_path()
    with pytest.raises(ValidationError):
        retriever.search(SearchRequest(**{"query": "battery", **arguments}))
    container.query_items.assert_not_called()


def test_search_propagates_query_failure() -> None:
    retriever, container = _search_path()
    failure = CosmosHttpResponseError(status_code=503, message="unavailable")
    container.query_items.side_effect = failure
    with pytest.raises(CosmosHttpResponseError) as caught:
        retriever.search(SearchRequest(query="battery"))
    assert caught.value is failure
    container.query_items.assert_called_once()


def test_search_rejects_no_term_query_before_execution() -> None:
    retriever, container = _search_path()
    with pytest.raises(QueryCompilationError):
        retriever.search(SearchRequest(query="!!!"))
    container.query_items.assert_not_called()


def test_retrievers_share_executor_without_mixing_containers() -> None:
    executor = CosmosExecutor(config=QueryEngineConfig(max_concurrency=1))
    first, first_container = _search_path(executor=executor)
    second, second_container = _search_path(
        CorpusSchema(item_id_path="/key", text_paths=["/content/body"]),
        executor=executor,
    )
    first_container.query_items.return_value = iter([{"item_id": "a", "txt_0": "A"}])
    second_container.query_items.return_value = iter([{"item_id": "b", "txt_0": "B"}])
    first_items = first.search(SearchRequest(query="battery"))
    second_items = second.search(SearchRequest(query="recycling"))
    assert first._ctx.executor is second._ctx.executor is executor
    assert [(item.item_id, item.text) for item in first_items] == [("a", "A")]
    assert [(item.item_id, item.text) for item in second_items] == [("b", "B")]
    assert 'c["id"] AS item_id' in first_container.query_items.call_args.kwargs["query"]
    assert (
        'c["key"] AS item_id' in second_container.query_items.call_args.kwargs["query"]
    )
    assert (
        'FullTextScore(c["content"]["body"], "recycling")'
        in second_container.query_items.call_args.kwargs["query"]
    )
