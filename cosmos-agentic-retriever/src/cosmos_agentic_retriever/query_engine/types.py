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
