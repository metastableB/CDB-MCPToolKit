from __future__ import annotations

from types import SimpleNamespace

from cosmos_agentic_retriever.query_engine import retriever as retr_mod
from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.query_engine.types import (
    PartitionQueryPolicy,
    RetrievedItem,
    SearchRequest,
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
    built.retriever.search(SearchRequest(query="q", text_fields=["a", "b"]))
    assert built.schema.text_calls == [["a", "b"]]


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
