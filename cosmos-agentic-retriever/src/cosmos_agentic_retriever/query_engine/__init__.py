"""Convert Python search arguments into Cosmos DB for NoSQL SQL queries.

In agentic retrieval, an agent receives a natural-language question and chooses
the search-queries needed to answer it. For example, "How has battery recycling
changed since 2020?" might require a keyword search for "battery recycling", a
vector search for related material, a year filter, and a lookup of the chunks
belonging to a relevant document.

This module (a) exposes a restricted set of searches methods into python and (b)
converts these to safe, valid database queries. Each method returns a
CompiledCosmosQuery containing the SQL command and its parameter values. A
CorpusSchema specifies where fields such as text, document IDs, and metadata are
stored in Cosmos DB items. The compiler constructs the command without sending it.
To execute it, import CosmosExecutor from cosmos_agentic_retriever.query_engine.executor,
construct it with a Cosmos container client, and pass the compiled query to run().

The following query methods and filters are supported. The compile_* methods
belong to CosmosQueryCompiler; the filter types are imported from this package:
- compile_vector: build a query that ranks items by distance from a supplied
    query vector.
- compile_full_text: build a query that ranks items by relevance to terms in
    specified text fields.
- compile_hybrid: build a query that combines vector and full-text rankings
    using reciprocal rank fusion (RRF).
- compile_structured: build a query that selects items using field filters,
    without vector or full-text ranking.
- compile_document_read: build a query that selects chunks belonging to a
    supplied document ID, up to a specified limit.
- EqualsFilter: require a field to equal a value, for example year = 2020.
- RangeFilter: apply an inclusive lower bound, upper bound, or both, for
    example 2020 <= year <= 2025.
- InFilter: require a field's value to be one of a supplied list, for example
    source in ["report", "article"].
- FilterExpression: the type representing any of the three filter types above.

Pass filter objects through the filters argument of compile_vector,
compile_full_text, compile_hybrid, or compile_structured. Multiple filters are
combined with AND. compile_document_read takes a document ID instead.
"""

from __future__ import annotations

from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.paths import CosmosPath
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    CompiledCosmosQuery,
    EqualsFilter,
    FilterExpression,
    InFilter,
    RangeFilter,
)

__all__ = [
    "CompiledCosmosQuery",
    "CorpusSchema",
    "CosmosPath",
    "CosmosQueryCompiler",
    "EqualsFilter",
    "FilterExpression",
    "InFilter",
    "RangeFilter",
]
