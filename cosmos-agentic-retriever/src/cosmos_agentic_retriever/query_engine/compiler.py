"""Build Cosmos DB SQL commands for various search types.

In agentic retrieval, an agent receives a natural-language question and chooses
the searches needed to answer it. For example, "How has battery recycling
changed since 2020?" might require a keyword search for "battery recycling", a
vector search for related material, a year filter, and a lookup of the chunks
belonging to a relevant document.

CosmosQueryCompiler provides Python methods that turn search arguments into
Cosmos DB SQL commands. The following methods are supported:
  - compile_vector: rank items by distance from a supplied query vector.
  - compile_full_text: rank items by relevance to text in specified fields.
    - compile_hybrid: combine vector and full-text rankings using reciprocal rank
        fusion (RRF).
  - compile_structured: select items using field filters, without search ranking.
  - compile_document_read: select chunks belonging to a document ID, up to a limit.

The first four methods additionally accept a list of filters:
- EqualsFilter: require a field to equal a value.
- RangeFilter: require a field to fall within inclusive lower or upper bounds.
- InFilter: require a field's value to belong to a supplied list.

Multiple filters are combined with AND.

CorpusSchema maps filter names to stored fields. For example, "year" can refer
to /publication/year. Each method returns a CompiledCosmosQuery containing SQL
and separate parameter values. Filter values, IDs, vectors, and limits use
parameters. Full-text terms are quoted and escaped in the SQL. The compiler
builds commands but does not execute them.
"""

from __future__ import annotations

from typing import Any

from cosmos_agentic_retriever.query_engine.full_text_terms import (
    fts_literal_args,
    tokenize_for_fts,
)
from cosmos_agentic_retriever.query_engine.paths import CosmosPath
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    CompiledCosmosQuery,
    EqualsFilter,
    FilterExpression,
    InFilter,
    QueryCompilationError,
    RangeFilter,
)

_ALIAS = "c"


class _ParamBag:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []
        self._n = 0

    def add(self, value: Any, prefix: str = "p") -> str:
        name = f"@{prefix}{self._n}"
        self._n += 1
        self.params.append({"name": name, "value": value})
        return name


class CosmosQueryCompiler:
    def __init__(self, schema: CorpusSchema) -> None:
        self.schema = schema

    def _resolve_logical(self, name: str) -> CosmosPath:
        s = self.schema
        mapping: dict[str, CosmosPath | None] = {
            "item_id": s.item_id_path,
            "document_id": s.document_id_path,
            "chunk_id": s.chunk_id_path,
            "chunk_order": s.chunk_order_path,
            "title": s.title_path,
            "source": s.source_path,
        }
        path = mapping.get(name)
        if path is not None:
            return path
        if name in s.metadata_paths:
            return CosmosPath.parse(s.metadata_paths[name])
        raise QueryCompilationError(f"unknown logical field {name!r}")

    @staticmethod
    def _limit(bag: _ParamBag, value: int) -> str:
        if value < 1:
            raise QueryCompilationError("query limit must be positive")
        return bag.add(value, prefix="k")

    def projection(self, limit_param: str) -> tuple[str, dict[str, str]]:

        s = CorpusSchema.model_validate(self.schema)
        cols: list[str] = []
        aliases: dict[str, str] = {}

        def add(logical: str, path: CosmosPath | None) -> None:
            if path is None:
                return
            cols.append(f"{path.render(_ALIAS)} AS {logical}")
            aliases[logical] = logical

        add("item_id", s.item_id_path)
        add("document_id", s.document_id_path)
        add("chunk_id", s.chunk_id_path)
        add("chunk_order", s.chunk_order_path)
        add("title", s.title_path)
        add("source", s.source_path)

        for i, (fname, fpath) in enumerate(s.text_field_map().items()):
            alias = f"txt_{i}"
            cols.append(f"{fpath.render(_ALIAS)} AS {alias}")
            aliases[alias] = fname
        for i, (key, path) in enumerate(s.metadata_paths.items()):
            alias = f"md_{i}"
            cols.append(f"{path.render(_ALIAS)} AS {alias}")
            aliases[alias] = key

        select = f"SELECT TOP {limit_param} " + ", ".join(cols) + f" FROM {_ALIAS}"
        return select, aliases

    def _compile_filter(self, f: FilterExpression, bag: _ParamBag) -> str:
        path = self._resolve_logical(f.logical_field).render(_ALIAS)
        if isinstance(f, EqualsFilter):
            return f"{path} = {bag.add(f.value)}"
        if isinstance(f, RangeFilter):
            parts: list[str] = []
            if f.minimum is not None:
                parts.append(f"{path} >= {bag.add(f.minimum)}")
            if f.maximum is not None:
                parts.append(f"{path} <= {bag.add(f.maximum)}")
            return "(" + " AND ".join(parts) + ")" if parts else "true"
        if isinstance(f, InFilter):
            return f"ARRAY_CONTAINS({bag.add(list(f.values))}, {path})"
        raise QueryCompilationError(f"unsupported filter {type(f).__name__}")

    def _where(
        self,
        filters: list[FilterExpression],
        ignored_item_ids: list[str],
        bag: _ParamBag,
    ) -> str:
        clauses = [self._compile_filter(f, bag) for f in filters]
        if ignored_item_ids:
            item_id = self.schema.item_id_path.render(_ALIAS)
            clauses.append(
                f"NOT ARRAY_CONTAINS({bag.add(ignored_item_ids)}, {item_id})"
            )
        return (" WHERE " + " AND ".join(clauses)) if clauses else ""

    @staticmethod
    def _full_text_terms(query: str, text_paths: list[CosmosPath]) -> str:
        if not text_paths:
            raise QueryCompilationError("at least one text path is required")
        terms = tokenize_for_fts(query)
        if not terms:
            raise QueryCompilationError(
                "full-text query must contain a searchable term"
            )
        return fts_literal_args(terms)

    def compile_hybrid(
        self,
        *,
        query: str,
        query_vector: list[float],
        limit: int,
        ignored_item_ids: list[str],
        filters: list[FilterExpression],
        partition_key: Any | None,
        cross_partition: bool,
        vector_path: CosmosPath,
        text_paths: list[CosmosPath],
    ) -> CompiledCosmosQuery:
        if not query_vector:
            raise QueryCompilationError("query vector must not be empty")
        terms = self._full_text_terms(query, text_paths)
        bag = _ParamBag()
        limit_p = self._limit(bag, limit)
        vec_p = bag.add(query_vector, prefix="qVec")
        select, aliases = self.projection(limit_p)
        where = self._where(filters, ignored_item_ids, bag)
        fts = ", ".join(
            f"FullTextScore({tp.render(_ALIAS)}, {terms})" for tp in text_paths
        )
        order = f" ORDER BY RANK RRF(VectorDistance({vector_path.render(_ALIAS)}, {vec_p}), {fts})"
        return CompiledCosmosQuery(
            sql=select + where + order,
            parameters=bag.params,
            partition_key=partition_key,
            enable_cross_partition_query=cross_partition,
            strategy="native_hybrid",
            projected_aliases=aliases,
        )

    def compile_vector(
        self,
        *,
        query_vector: list[float],
        limit: int,
        ignored_item_ids: list[str],
        filters: list[FilterExpression],
        partition_key: Any | None,
        cross_partition: bool,
        vector_path: CosmosPath,
    ) -> CompiledCosmosQuery:
        if not query_vector:
            raise QueryCompilationError("query vector must not be empty")
        bag = _ParamBag()
        limit_p = self._limit(bag, limit)
        vec_p = bag.add(query_vector, prefix="qVec")
        select, aliases = self.projection(limit_p)
        where = self._where(filters, ignored_item_ids, bag)
        order = f" ORDER BY VectorDistance({vector_path.render(_ALIAS)}, {vec_p})"
        return CompiledCosmosQuery(
            sql=select + where + order,
            parameters=bag.params,
            partition_key=partition_key,
            enable_cross_partition_query=cross_partition,
            strategy="vector",
            projected_aliases=aliases,
        )

    def compile_full_text(
        self,
        *,
        query: str,
        limit: int,
        ignored_item_ids: list[str],
        filters: list[FilterExpression],
        partition_key: Any | None,
        cross_partition: bool,
        text_paths: list[CosmosPath],
        strategy: str = "full_text",
    ) -> CompiledCosmosQuery:
        terms = self._full_text_terms(query, text_paths)
        bag = _ParamBag()
        limit_p = self._limit(bag, limit)
        select, aliases = self.projection(limit_p)
        where = self._where(filters, ignored_item_ids, bag)
        if len(text_paths) == 1:
            order = (
                f" ORDER BY RANK FullTextScore({text_paths[0].render(_ALIAS)}, {terms})"
            )
        else:
            fts = ", ".join(
                f"FullTextScore({tp.render(_ALIAS)}, {terms})" for tp in text_paths
            )
            order = f" ORDER BY RANK RRF({fts})"
        return CompiledCosmosQuery(
            sql=select + where + order,
            parameters=bag.params,
            partition_key=partition_key,
            enable_cross_partition_query=cross_partition,
            strategy=strategy,
            projected_aliases=aliases,
        )

    def compile_structured(
        self,
        *,
        limit: int,
        filters: list[FilterExpression],
        ignored_item_ids: list[str],
        partition_key: Any | None,
        cross_partition: bool,
    ) -> CompiledCosmosQuery:
        bag = _ParamBag()
        limit_p = self._limit(bag, limit)
        select, aliases = self.projection(limit_p)
        where = self._where(filters, ignored_item_ids, bag)
        return CompiledCosmosQuery(
            sql=select + where,
            parameters=bag.params,
            partition_key=partition_key,
            enable_cross_partition_query=cross_partition,
            strategy="structured",
            projected_aliases=aliases,
        )

    def compile_document_read(
        self,
        *,
        document_id: str,
        max_chunks: int,
        partition_key: Any | None,
        cross_partition: bool,
    ) -> CompiledCosmosQuery:
        s = self.schema
        if s.document_id_path is None:
            raise QueryCompilationError("document_id_path is not configured")
        bag = _ParamBag()
        limit_p = self._limit(bag, max_chunks)
        select, aliases = self.projection(limit_p)
        doc_p = bag.add(document_id, prefix="doc")
        where = f" WHERE {s.document_id_path.render(_ALIAS)} = {doc_p}"
        return CompiledCosmosQuery(
            sql=select + where,
            parameters=bag.params,
            partition_key=partition_key,
            enable_cross_partition_query=cross_partition,
            strategy="document_read",
            projected_aliases=aliases,
        )
