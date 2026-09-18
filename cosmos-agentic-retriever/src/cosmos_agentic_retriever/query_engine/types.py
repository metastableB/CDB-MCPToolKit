from __future__ import annotations

from typing import Annotated, Any, Literal

# TODO: Is pydantic justified here? Isn't dataclass cleaner?
from pydantic import BaseModel, ConfigDict, Field


class QueryEngineConfig(BaseModel):
    """Query engine configuration

    Contains only settings, not clients or locks. Each executor creates its own
    concurrency limiter from these values. Reuse an executor to share its limit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrency: int = Field(default=8, strict=True, gt=0)
    slow_query_warning_seconds: float = Field(
        default=4.5,
        strict=True,
        gt=0,
        allow_inf_nan=False,
    )


# Exceptions raised by path validation and query compilation.
class RetrievalError(Exception):
    pass


class UnsafeCosmosPathError(RetrievalError):
    pass


class QueryCompilationError(RetrievalError):
    pass


class UnknownField(RetrievalError):
    pass


class CrossPartitionQueryDisabled(RetrievalError):
    pass


# Field conditions passed to the compiler's filters argument.
class EqualsFilter(BaseModel):
    kind: Literal["equals"] = "equals"
    logical_field: str
    value: Any


class RangeFilter(BaseModel):
    kind: Literal["range"] = "range"
    logical_field: str
    minimum: Any | None = None
    maximum: Any | None = None


class InFilter(BaseModel):
    kind: Literal["in"] = "in"
    logical_field: str
    values: list[Any]


FilterExpression = Annotated[
    EqualsFilter | RangeFilter | InFilter, Field(discriminator="kind")
]


# Compiler output: SQL and the values/settings needed to execute it.
class CompiledCosmosQuery(BaseModel):
    sql: str
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    partition_key: Any | None = None
    enable_cross_partition_query: bool = False
    strategy: str = ""
    projected_aliases: dict[str, str] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    limit: int = 50
    ignored_item_ids: list[str] = Field(default_factory=list)
    filters: list[FilterExpression] = Field(default_factory=list)
    partition_key: Any | None = None
    text_fields: list[str] | None = None


class RetrievedItem(BaseModel):
    item_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    chunk_order: int | None = None
    text: str = ""
    text_fields: dict[str, str] = Field(default_factory=dict)
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str = ""
    retrieval_channels: list[str] = Field(default_factory=list)
    rank: int = 0


class PartitionQueryPolicy(BaseModel):
    allow_cross_partition_search: bool = True
