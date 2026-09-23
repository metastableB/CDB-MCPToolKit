from __future__ import annotations

import pytest

from cosmos_agentic_retriever.query_engine import strategies as strat_mod
from cosmos_agentic_retriever.query_engine.strategies import (
    FullTextSearchStrategy,
    RetrievalContext,
    SearchStrategy,
    _resolve_cross_partition,
)
from cosmos_agentic_retriever.query_engine.types import (
    CrossPartitionQueryDisabled,
    SearchRequest,
)


class FakeSchema:
    def __init__(self, text=("T1", "T2")):
        self.text = list(text)
        self.text_calls: list = []

    def resolve_text_fields(self, names):
        self.text_calls.append(names)
        return self.text


class FakeCompiled:
    def __init__(self, aliases="ALIASES"):
        self.projected_aliases = aliases


class FakeCompiler:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def compile_full_text(self, **kwargs):
        self.calls.append(("full_text", kwargs))
        return FakeCompiled()


class FakeExecutor:
    def __init__(self):
        self.ran: list = []

    def run(self, compiled, *, container):
        self.ran.append((compiled, container))
        return [{"row": 1}]


class FakePolicy:
    def __init__(self, cross=True):
        self.allow_cross_partition_search = cross


class FakeNormalize:
    def __init__(self):
        self.calls: list[tuple] = []
        self.default: list = []

    def __call__(self, rows, **kwargs):
        self.calls.append((rows, kwargs))
        return self.default


@pytest.fixture
def norm(monkeypatch) -> FakeNormalize:
    fake = FakeNormalize()
    monkeypatch.setattr(strat_mod, "rows_to_items", fake)
    return fake


def _ctx(schema=None, compiler=None, executor=None, policy=None) -> RetrievalContext:
    return RetrievalContext(
        schema=schema or FakeSchema(),
        compiler=compiler or FakeCompiler(),
        executor=executor or FakeExecutor(),
        container=object(),
        policy=policy or FakePolicy(),
    )


def _req(**kwargs) -> SearchRequest:
    return SearchRequest(**{"query": "q", "limit": 10, **kwargs})


def test_cross_partition_with_key_is_false() -> None:
    assert _resolve_cross_partition("pk", FakePolicy(cross=False)) is False


def test_cross_partition_none_allowed_true() -> None:
    assert _resolve_cross_partition(None, FakePolicy(cross=True)) is True


def test_cross_partition_none_disallowed_raises() -> None:
    with pytest.raises(CrossPartitionQueryDisabled):
        _resolve_cross_partition(None, FakePolicy(cross=False))


def test_search_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        SearchStrategy()


def test_strategy_names_and_embedding_flags() -> None:
    assert (FullTextSearchStrategy.name, FullTextSearchStrategy.requires_embedding) == (
        "full_text",
        False,
    )


def test_full_text_execute_wiring(norm) -> None:
    schema, compiler = FakeSchema(), FakeCompiler()
    ctx = _ctx(schema, compiler)
    FullTextSearchStrategy().execute(_req(query="hello", text_fields=["/body"]), ctx)
    assert schema.text_calls == [["/body"]]
    method, kwargs = compiler.calls[0]
    assert method == "full_text"
    assert kwargs["query"] == "hello" and kwargs["text_paths"] == ["T1", "T2"]
    normalized = norm.calls[0][1]
    assert normalized["strategy"] == "full_text"
    assert normalized["channels"] == ["full_text"]
    assert normalized["queried_text_fields"] == ["/body"]
    assert ctx.executor.ran[0][1] is ctx.container


def test_full_text_cross_partition_false_with_partition_key(norm) -> None:
    compiler = FakeCompiler()
    FullTextSearchStrategy().execute(_req(partition_key="pk"), _ctx(compiler=compiler))
    assert compiler.calls[0][1]["cross_partition"] is False


def test_full_text_cross_partition_disabled_raises() -> None:
    with pytest.raises(CrossPartitionQueryDisabled):
        FullTextSearchStrategy().execute(_req(), _ctx(policy=FakePolicy(cross=False)))
