from __future__ import annotations

from typing import Annotated, Any, Literal

# TODO: Is pydantic justified here? Isn't dataclass cleaner?
from pydantic import BaseModel, Field


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


class CompiledCosmosQuery(BaseModel):
    sql: str
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    partition_key: Any | None = None
    enable_cross_partition_query: bool = False
    strategy: str = ""
    projected_aliases: dict[str, str] = Field(default_factory=dict)