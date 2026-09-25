"""Configure full-text search over an explicit set of containers in one database.

COSMOS_CONTAINERS is a JSON object keyed by container name. Each entry declares
its schema, selected text fields, and partition rules. All containers share the
account credentials and query concurrency limit. Nothing is auto-discovered.
"""

from typing import Any, Literal, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from cosmos_agentic_retriever.query_engine import CorpusSchema, QueryEngineConfig
from cosmos_agentic_retriever.query_engine.types import PartitionQueryPolicy


class ContainerConfig(BaseModel):
    """Schema and search scope for one container, independent of other containers."""

    model_config = ConfigDict(extra="forbid")

    cosmos_schema: CorpusSchema
    search_text_fields: list[str] | None = None
    partition_key: Any | None = None
    partition_policy: PartitionQueryPolicy = Field(default_factory=PartitionQueryPolicy)

    @model_validator(mode="after")
    def _searchable_fields(self) -> Self:
        if not self.cosmos_schema.partition_key_paths:
            raise ValueError("configure partition_key_paths for physical item identity")
        if not self.cosmos_schema.resolve_text_fields(self.search_text_fields):
            raise ValueError("configure at least one searchable text field")
        if (
            self.partition_key is None
            and not self.partition_policy.allow_cross_partition_search
        ):
            raise ValueError(
                "configure a partition key or allow cross-partition search"
            )
        return self


class RetrieverSettings(BaseSettings):
    """Startup settings. A Cosmos key takes precedence over the identity choice."""

    model_config = SettingsConfigDict(extra="forbid")

    account_uri: AnyHttpUrl
    cosmos_database: str = Field(min_length=1, max_length=256)
    cosmos_containers: dict[str, ContainerConfig] = Field(min_length=1)
    cosmos_key: SecretStr | None = None
    cosmos_credential: Literal["azure_cli", "default"] = "azure_cli"
    query_engine: QueryEngineConfig = Field(default_factory=QueryEngineConfig)
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=9000, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    @field_validator("cosmos_database")
    @classmethod
    def _single_target(cls, value: str) -> str:
        if not value.strip() or "*" in value:
            raise ValueError("configure a database name, not a wildcard")
        return value

    @field_validator("cosmos_containers")
    @classmethod
    def _container_names(
        cls, values: dict[str, ContainerConfig]
    ) -> dict[str, ContainerConfig]:
        if any(not name.strip() or "*" in name or len(name) > 256 for name in values):
            raise ValueError(
                "configure explicit container names, not blanks or wildcards"
            )
        return values

    @field_validator("cosmos_key")
    @classmethod
    def _nonempty_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("cosmos_key must not be empty")
        return value


def get_settings() -> RetrieverSettings:
    """Read the environment explicitly at application startup, not during imports."""
    return RetrieverSettings()
