"""Define the inputs, outputs, settings, and errors used by the query engine."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Final, Literal

# TODO: Is pydantic justified here? Isn't dataclass cleaner?
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
class FieldFilter(BaseModel):
    """Identify a stored field by its Cosmos path, independently of result labels."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always"
    )

    path: str

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> str:
        from cosmos_agentic_retriever.query_engine.paths import CosmosPath

        try:
            return str(CosmosPath.parse(value))
        except UnsafeCosmosPathError as error:
            raise ValueError(str(error)) from error


class EqualsFilter(FieldFilter):
    kind: Literal["equals"] = "equals"
    value: Any


class RangeFilter(FieldFilter):
    kind: Literal["range"] = "range"
    minimum: Any | None = None
    maximum: Any | None = None

    @model_validator(mode="after")
    def _require_bound(self) -> RangeFilter:
        if self.minimum is None and self.maximum is None:
            raise ValueError("range filter requires a minimum or maximum")
        return self


class InFilter(FieldFilter):
    kind: Literal["in"] = "in"
    values: list[Any]


FilterExpression = Annotated[
    EqualsFilter | RangeFilter | InFilter, Field(discriminator="kind")
]


class SQLColumnAliases:
    """SQL column aliases shared by the compiler (encode) and row decoder (decode).

    The compiler projects columns under these names; row_decoding reads the same
    names back into RetrievedItem, so both sides stay in sync through this type.
    """

    ITEM_ID: Final = "item_id"
    PARENT_DOCUMENT_ID: Final = "parent_document_id"
    CHUNK_ID: Final = "chunk_id"
    CHUNK_ORDER: Final = "chunk_order"
    COSMOS_IDENTITY: Final = "_cosmos_identity"
    TEXT_PREFIX: Final = "txt_"
    ADDITIONAL_PREFIX: Final = "add_"


# Compiler output: SQL and the values/settings needed to execute it.
class CompiledCosmosQuery(BaseModel):
    sql: str
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    partition_key: Any | None = None
    enable_cross_partition_query: bool = False
    strategy: str = ""
    projected_aliases: dict[str, str] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Search configured full text paths, such as text_fields=["/article/text"].

    Omit text_fields only when the schema has a single text field.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    query: str = Field(min_length=1)
    # Return at most this many Cosmos items per search: limit=5 returns up to 5.
    # An item may be one chunk of a document, not a complete document.
    limit: int = Field(default=50, strict=True, gt=0)
    ignored_item_ids: list[str] = Field(default_factory=list)
    filters: list[FilterExpression] = Field(default_factory=list)
    partition_key: Any | None = None
    text_fields: list[str] | None = None


class CosmosItemIdentity(BaseModel):
    """A physical Cosmos item and a unique id for that item. 

    In Cosmos DB NoSQL, a document's real unique key is (partition-key value(s),
    id), not id alone. Cosmos's documented addressing identity uses (id +
    partition-key value) for unique addressing. However, partition-key values
    are arbitrary JSON (strings, numbers, bools, null, hierarchical up to 3
    levels, or undefined). To use those values as an equality key we normalize
    them through the _validate_components() private method. 
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(strict=True, min_length=1)
    partition_key: tuple[Any, ...] = Field(min_length=1, max_length=3)

    @field_validator("partition_key")
    @classmethod
    def _validate_components(cls, values: tuple[Any, ...]) -> tuple[Any, ...]:
        normalized = []
        for value in values:
            if value is None or type(value) in (str, bool):
                normalized.append(value)
            elif type(value) in (int, float):
                try:
                    number = float(value)
                except OverflowError as error:
                    raise ValueError("partition key must be a finite number") from error
                if not math.isfinite(number):
                    raise ValueError("partition key must be a finite number")
                normalized.append(int(number) if number.is_integer() else number)
            elif isinstance(value, dict) and not value:
                normalized.append({})
            else:
                raise ValueError("invalid Cosmos partition-key component")
        return tuple(normalized)

    def key(self) -> str:
        """Encode typed components without delimiter collisions."""
        return json.dumps(
            [self.partition_key, self.id],
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class RetrievedItem(BaseModel):
    """A ranked item with text_fields keyed by full paths, such as /article/text."""

    item_id: str
    cosmos_identity: CosmosItemIdentity | None = None
    parent_document_id: str | None = None
    chunk_id: str | None = None
    chunk_order: int | None = None
    text: str = ""
    text_fields: dict[str, str] = Field(default_factory=dict)
    additional_fields: dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str = ""
    retrieval_channels: list[str] = Field(default_factory=list)
    rank: int = 0


class PartitionQueryPolicy(BaseModel):
    allow_cross_partition_search: bool = True
