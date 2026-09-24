from __future__ import annotations

from typing import Any

import pytest

from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.full_text_terms import DEFAULT_MAX_FTS_TERMS
from cosmos_agentic_retriever.query_engine.paths import CosmosPath
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    EqualsFilter,
    InFilter,
    QueryCompilationError,
    RangeFilter,
)

_VEC = CosmosPath.parse("/embedding")
_TEXT = CosmosPath.parse("/text")
_BODY = CosmosPath.parse("/body")


def _schema(*, with_docid: bool = True) -> CorpusSchema:
    return CorpusSchema(
        item_id_path="/id",
        document_id_path="/docid" if with_docid else None,
        chunk_id_path="/id",
        chunk_order_path="/chunk_idx",
        title_path="/title",
        source_path="/source_type",
        text_paths=["/text"],
        metadata_paths={"year": "/year"},
    )


def _compiler(*, with_docid: bool = True) -> CosmosQueryCompiler:
    return CosmosQueryCompiler(_schema(with_docid=with_docid))


def _param(q: Any, name: str) -> dict[str, Any]:
    for p in q.parameters:
        if p["name"] == name:
            return p
    raise AssertionError(
        f"no bound parameter {name!r} in {[p['name'] for p in q.parameters]}"
    )


def _param_values(q: Any) -> list[Any]:
    return [p["value"] for p in q.parameters]


# --- projection -----------------------------------------------------------


def test_projection_emits_logical_columns_and_alias_map() -> None:
    select, aliases = _compiler().projection("@k0")
    assert select.startswith("SELECT TOP @k0 ")
    for col in (
        'c["id"] AS item_id',
        'c["docid"] AS document_id',
        'c["id"] AS chunk_id',
        'c["chunk_idx"] AS chunk_order',
        'c["title"] AS title',
        'c["source_type"] AS source',
        'c["text"] AS txt_0',
        'c["year"] AS md_0',
    ):
        assert col in select
    # text/metadata aliases resolve back to their logical names
    assert aliases["txt_0"] == "text"
    assert aliases["md_0"] == "year"


def test_projection_docstring_example() -> None:
    schema = CorpusSchema(item_id_path="/id", text_paths=["/content/text"])
    assert CosmosQueryCompiler(schema).projection("@k0") == (
        'SELECT TOP @k0 c["id"] AS item_id, c["content"]["text"] AS txt_0 FROM c',
        {"item_id": "item_id", "txt_0": "text"},
    )


def test_projection_does_not_interpolate_metadata_names_into_sql() -> None:
    schema = _schema()
    schema.metadata_paths = {"year AS injected FROM x --": CosmosPath.parse("/year")}
    select, aliases = CosmosQueryCompiler(schema).projection("@k0")
    assert "injected" not in select
    assert aliases["md_0"] == "year AS injected FROM x --"


def test_compiler_preserves_literal_slashes_in_schema_paths() -> None:
    schema = CorpusSchema(
        item_id_path="/id",
        text_paths=['/"document/title"', "/document/title"],
    )
    compiled = CosmosQueryCompiler(schema).compile_full_text(
        query="battery recycling",
        limit=5,
        filters=[],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
        text_paths=schema.text_paths,
    )
    assert 'c["document/title"] AS txt_0' in compiled.sql
    assert 'c["document"]["title"] AS txt_1' in compiled.sql
    assert 'FullTextScore(c["document/title"],' in compiled.sql
    assert 'FullTextScore(c["document"]["title"],' in compiled.sql


# --- structured filters ---------------------------------------------------


def test_structured_equals_filter_is_parameterized() -> None:
    q = _compiler().compile_structured(
        limit=10,
        filters=[EqualsFilter(logical_field="year", value=2020)],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )
    assert q.strategy == "structured"
    assert 'c["year"] = @p1' in q.sql
    assert _param(q, "@p1")["value"] == 2020
    assert _param(q, "@k0")["value"] == 10


def test_range_filter_emits_both_bounds() -> None:
    q = _compiler().compile_structured(
        limit=5,
        filters=[RangeFilter(logical_field="year", minimum=2000, maximum=2020)],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )
    assert '(c["year"] >= @p1 AND c["year"] <= @p2)' in q.sql
    assert _param(q, "@p1")["value"] == 2000
    assert _param(q, "@p2")["value"] == 2020


def test_range_filter_with_only_minimum() -> None:
    q = _compiler().compile_structured(
        limit=5,
        filters=[RangeFilter(logical_field="year", minimum=2000)],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )
    assert '(c["year"] >= @p1)' in q.sql
    assert "<=" not in q.sql


def test_in_filter_uses_array_contains() -> None:
    q = _compiler().compile_structured(
        limit=5,
        filters=[InFilter(logical_field="source", values=["news", "blog"])],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )
    assert 'ARRAY_CONTAINS(@p1, c["source_type"])' in q.sql
    assert _param(q, "@p1")["value"] == ["news", "blog"]


def test_ignored_item_ids_add_not_array_contains() -> None:
    q = _compiler().compile_structured(
        limit=5,
        filters=[],
        ignored_item_ids=["a", "b"],
        partition_key=None,
        cross_partition=True,
    )
    assert 'NOT ARRAY_CONTAINS(@p1, c["id"])' in q.sql
    assert _param(q, "@p1")["value"] == ["a", "b"]


def test_no_filters_emits_no_where_clause() -> None:
    q = _compiler().compile_structured(
        limit=5,
        filters=[],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )
    assert "WHERE" not in q.sql


# --- vector / full-text / hybrid -----------------------------------------


def test_full_text_multiple_paths_uses_rank_rrf() -> None:
    q = _compiler().compile_full_text(
        query="quick brown",
        limit=5,
        ignored_item_ids=[],
        filters=[],
        partition_key=None,
        cross_partition=True,
        text_paths=[_TEXT, _BODY],
    )
    assert "ORDER BY RANK RRF(" in q.sql
    assert 'FullTextScore(c["text"], "quick", "brown")' in q.sql
    assert 'FullTextScore(c["body"], "quick", "brown")' in q.sql


@pytest.mark.parametrize("method", ["compile_full_text", "compile_hybrid"])
@pytest.mark.parametrize("max_terms", [1, 40, 0])
def test_compiler_forwards_per_call_term_limit(method: str, max_terms: int) -> None:
    compile_query = getattr(_compiler(), method)
    terms = [f"term{index}" for index in range(45)]
    arguments = {
        "query": " ".join(terms),
        "limit": 5,
        "ignored_item_ids": [],
        "filters": [],
        "partition_key": None,
        "cross_partition": True,
        "text_paths": [_TEXT, _BODY],
    }
    if method == "compile_hybrid":
        arguments.update(query_vector=[0.1], vector_path=_VEC)
    if max_terms == 0:
        with pytest.raises(ValueError, match="max_terms must be a positive integer"):
            compile_query(**arguments, max_terms=max_terms)
    else:
        with pytest.raises(ValueError, match=f"exceeds max_terms={max_terms}"):
            compile_query(**arguments, max_terms=max_terms)
        arguments["query"] = " ".join(terms[:max_terms])
        result = compile_query(**arguments, max_terms=max_terms)
        expected_terms = ", ".join(f'"{term}"' for term in terms[:max_terms])
        for path in (_TEXT, _BODY):
            assert f"FullTextScore({path.render()}, {expected_terms})" in result.sql
        assert f'"term{max_terms}"' not in result.sql
        assert _param(result, "@k0")["value"] == 5

    assert DEFAULT_MAX_FTS_TERMS == 30
    arguments["query"] = " ".join(terms)
    with pytest.raises(ValueError, match="exceeds max_terms=30"):
        compile_query(**arguments)
    arguments["query"] = " ".join(terms[:DEFAULT_MAX_FTS_TERMS])
    default_result = compile_query(**arguments)
    assert '"term29"' in default_result.sql
    assert '"term30"' not in default_result.sql


@pytest.mark.parametrize("method", ["compile_full_text", "compile_hybrid"])
@pytest.mark.parametrize(
    "query", ["the battery and recycling", "the and of", "not recyclable"]
)
def test_compiler_leaves_stopword_analysis_to_cosmos(method: str, query: str) -> None:
    arguments = {
        "query": query,
        "limit": 5,
        "ignored_item_ids": [],
        "filters": [],
        "partition_key": None,
        "cross_partition": True,
        "text_paths": [_TEXT, _BODY],
    }
    if method == "compile_hybrid":
        arguments.update(query_vector=[0.1], vector_path=_VEC)
    result = getattr(_compiler(), method)(**arguments)
    expected_terms = ", ".join(f'"{term}"' for term in query.split())
    for path in (_TEXT, _BODY):
        assert f"FullTextScore({path.render()}, {expected_terms})" in result.sql


@pytest.mark.parametrize("method", ["full_text", "hybrid"])
def test_full_text_queries_require_searchable_terms(method: str) -> None:
    common = {
        "query": "!!!",
        "limit": 5,
        "ignored_item_ids": [],
        "filters": [],
        "partition_key": None,
        "cross_partition": True,
        "text_paths": [_TEXT],
    }
    with pytest.raises(QueryCompilationError, match="searchable term"):
        if method == "hybrid":
            _compiler().compile_hybrid(query_vector=[0.1], vector_path=_VEC, **common)
        else:
            _compiler().compile_full_text(**common)


@pytest.mark.parametrize("method", ["full_text", "hybrid"])
def test_full_text_queries_require_a_text_path(method: str) -> None:
    common = {
        "query": "query",
        "limit": 5,
        "ignored_item_ids": [],
        "filters": [],
        "partition_key": None,
        "cross_partition": True,
        "text_paths": [],
    }
    with pytest.raises(QueryCompilationError, match="text path"):
        if method == "hybrid":
            _compiler().compile_hybrid(query_vector=[0.1], vector_path=_VEC, **common)
        else:
            _compiler().compile_full_text(**common)


@pytest.mark.parametrize("method", ["vector", "hybrid"])
def test_vector_queries_require_a_nonempty_vector(method: str) -> None:
    common = {
        "query_vector": [],
        "limit": 5,
        "ignored_item_ids": [],
        "filters": [],
        "partition_key": None,
        "cross_partition": True,
        "vector_path": _VEC,
    }
    with pytest.raises(QueryCompilationError, match="vector must not be empty"):
        if method == "hybrid":
            _compiler().compile_hybrid(query="query", text_paths=[_TEXT], **common)
        else:
            _compiler().compile_vector(**common)


def _compile_contract_query(
    method: str,
    limit: Any,
    partition_key: Any = "tenant-a",
    cross_partition: bool = False,
):
    schema = CorpusSchema(
        item_id_path="/id",
        document_id_path="/docid",
        text_paths=["/text"],
        metadata_paths={"year": "/publication/year", "category": "/category"},
    )
    arguments: dict[str, Any] = {
        "partition_key": partition_key,
        "cross_partition": cross_partition,
    }
    if method == "compile_document_read":
        arguments.update(document_id="doc-1", max_chunks=limit)
    else:
        arguments.update(
            limit=limit,
            ignored_item_ids=["skip-1"],
            filters=[
                EqualsFilter(logical_field="category", value="report"),
                RangeFilter(logical_field="year", minimum=2000, maximum=2020),
            ],
        )
    if method in ("compile_vector", "compile_hybrid"):
        arguments.update(query_vector=[0.1, 0.2], vector_path=_VEC)
    if method in ("compile_full_text", "compile_hybrid"):
        arguments.update(query="the battery recycling", text_paths=[_TEXT])
    return getattr(CosmosQueryCompiler(schema), method)(**arguments)


_COMPILE_METHODS = [
    "compile_vector",
    "compile_full_text",
    "compile_hybrid",
    "compile_structured",
    "compile_document_read",
]


@pytest.mark.parametrize("method", _COMPILE_METHODS)
@pytest.mark.parametrize(
    "limit", [0, -1, True, False, 1.5, float("nan"), float("inf"), "5", None]
)
def test_queries_require_a_positive_integer_limit(method: str, limit: Any) -> None:
    with pytest.raises(QueryCompilationError, match="limit must be a positive integer"):
        _compile_contract_query(method, limit)


@pytest.mark.parametrize(
    "partition_key, cross_partition", [("tenant-a", False), (0, True)]
)
@pytest.mark.parametrize(
    "method, condition, ordering, bindings, strategy",
    [
        (
            "compile_vector",
            'c["category"] = @p2 AND (c["publication"]["year"] >= @p3 AND c["publication"]["year"] <= @p4) AND NOT ARRAY_CONTAINS(@p5, c["id"])',
            ' ORDER BY VectorDistance(c["embedding"], @qVec1)',
            [
                ("@k0", 1),
                ("@qVec1", [0.1, 0.2]),
                ("@p2", "report"),
                ("@p3", 2000),
                ("@p4", 2020),
                ("@p5", ["skip-1"]),
            ],
            "vector",
        ),
        (
            "compile_full_text",
            'c["category"] = @p1 AND (c["publication"]["year"] >= @p2 AND c["publication"]["year"] <= @p3) AND NOT ARRAY_CONTAINS(@p4, c["id"])',
            ' ORDER BY RANK FullTextScore(c["text"], "the", "battery", "recycling")',
            [
                ("@k0", 1),
                ("@p1", "report"),
                ("@p2", 2000),
                ("@p3", 2020),
                ("@p4", ["skip-1"]),
            ],
            "full_text",
        ),
        (
            "compile_hybrid",
            'c["category"] = @p2 AND (c["publication"]["year"] >= @p3 AND c["publication"]["year"] <= @p4) AND NOT ARRAY_CONTAINS(@p5, c["id"])',
            ' ORDER BY RANK RRF(VectorDistance(c["embedding"], @qVec1), FullTextScore(c["text"], "the", "battery", "recycling"))',
            [
                ("@k0", 1),
                ("@qVec1", [0.1, 0.2]),
                ("@p2", "report"),
                ("@p3", 2000),
                ("@p4", 2020),
                ("@p5", ["skip-1"]),
            ],
            "native_hybrid",
        ),
        (
            "compile_structured",
            'c["category"] = @p1 AND (c["publication"]["year"] >= @p2 AND c["publication"]["year"] <= @p3) AND NOT ARRAY_CONTAINS(@p4, c["id"])',
            "",
            [
                ("@k0", 1),
                ("@p1", "report"),
                ("@p2", 2000),
                ("@p3", 2020),
                ("@p4", ["skip-1"]),
            ],
            "structured",
        ),
        (
            "compile_document_read",
            'c["docid"] = @doc1',
            "",
            [("@k0", 1), ("@doc1", "doc-1")],
            "document_read",
        ),
    ],
)
def test_complete_compiled_query_contract(
    method, condition, ordering, bindings, strategy, partition_key, cross_partition
) -> None:
    result = _compile_contract_query(method, 1, partition_key, cross_partition)
    expected_select = (
        'SELECT TOP @k0 c["id"] AS item_id, c["docid"] AS document_id, '
        'c["text"] AS txt_0, c["publication"]["year"] AS md_0, '
        'c["category"] AS md_1 FROM c'
    )
    assert result.sql == expected_select + " WHERE " + condition + ordering
    assert result.parameters == [
        {"name": name, "value": value} for name, value in bindings
    ]
    assert result.partition_key == partition_key
    assert result.enable_cross_partition_query is cross_partition
    assert result.strategy == strategy
    assert result.projected_aliases == {
        "item_id": "item_id",
        "document_id": "document_id",
        "txt_0": "text",
        "md_0": "year",
        "md_1": "category",
    }


# --- document read --------------------------------------------------------


def test_document_read_without_document_id_path_raises() -> None:
    with pytest.raises(QueryCompilationError):
        _compiler(with_docid=False).compile_document_read(
            document_id="doc-1",
            max_chunks=50,
            partition_key=None,
            cross_partition=True,
        )


# --- errors & injection safety -------------------------------------------


def test_unknown_logical_field_raises() -> None:
    with pytest.raises(QueryCompilationError):
        _compiler().compile_structured(
            limit=5,
            filters=[EqualsFilter(logical_field="does_not_exist", value=1)],
            ignored_item_ids=[],
            partition_key=None,
            cross_partition=True,
        )


def test_filter_values_are_bound_never_inlined() -> None:
    """User-controlled values must go through @params, not the SQL string."""
    malicious = "2020'; DROP TABLE Foo--"
    q = _compiler().compile_structured(
        limit=5,
        filters=[EqualsFilter(logical_field="year", value=malicious)],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )
    assert "DROP TABLE" not in q.sql
    assert 'c["year"] = @p1' in q.sql
    assert malicious in _param_values(q)
