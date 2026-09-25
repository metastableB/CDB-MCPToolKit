"""Define the small, versioned Cosmos resources shared by setup and live tests.

This module contains fixture data and validation only. It creates no clients and
makes no network calls. Changing an incompatible fixture requires a new version.
"""

from dataclasses import dataclass
from typing import Any

from cosmos_agentic_retriever.config import RetrieverSettings

DEFAULT_DATABASE = "mcp-live-tests-v1"
FIXTURE_VERSION = "mcp-search-v1"


@dataclass(frozen=True)
class ContainerFixture:
    name: str
    schema: dict[str, Any]
    indexed_paths: tuple[str, ...]
    search_fields: tuple[str, ...]
    items: tuple[dict[str, Any], ...]

    def full_text_policy(self) -> dict:
        return {
            "defaultLanguage": "en-US",
            "fullTextPaths": [
                {"path": path, "language": "en-US"} for path in self.indexed_paths
            ],
        }

    def indexing_policy(self) -> dict:
        return {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            # Cosmos excludes _etag from indexing by default. Declare that policy
            # explicitly for round-trip validation; this does not disable ETags.
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "fullTextIndexes": [{"path": path} for path in self.indexed_paths],
        }


def fixtures() -> tuple[ContainerFixture, ...]:
    """Return nine deterministic synthetic documents, not a retrieval corpus.

    Fixed marker words exercise field selection and partition routing. Fresh
    objects keep callers from altering another test's inputs.
    """
    flat, nested, multiple = [], [], []
    for item_id, tenant in (("shared", 0), ("first-only", 0), ("other-only", "other")):
        base = {"id": item_id, "tenant": tenant, "fixture_version": FIXTURE_VERSION}
        text = (
            "battery recycling canaryalpha"
            if tenant == 0
            else "battery recycling canarybeta"
        )
        flat.append({**base, "text": text, "year": 2024})
        nested.append(
            {
                **base,
                "id": f"physical-{item_id}",
                "record": {"id": item_id},
                "content": {"body": text},
                "year": 2025,
            }
        )
        multiple.append(
            {
                **base,
                "title": "canaryheadline" if item_id == "shared" else "canarybody",
                "body": "battery recycling canarybody"
                if item_id == "shared"
                else "battery recycling canaryheadline",
            }
        )
    return (
        ContainerFixture(
            "flat-v1",
            {
                "item_id_path": "/id",
                "partition_key_paths": ["/tenant"],
                "text_paths": ["/text"],
                "metadata_paths": {"year": "/year"},
            },
            ("/text",),
            ("/text",),
            tuple(flat),
        ),
        ContainerFixture(
            "nested-v1",
            {
                "item_id_path": "/record/id",
                "partition_key_paths": ["/tenant"],
                "text_paths": ["/content/body"],
                "metadata_paths": {"year": "/year"},
            },
            ("/content/body",),
            ("/content/body",),
            tuple(nested),
        ),
        ContainerFixture(
            "fields-v1",
            {
                "item_id_path": "/id",
                "partition_key_paths": ["/tenant"],
                "text_paths": ["/title", "/body"],
            },
            ("/title", "/body"),
            ("/body",),
            tuple(multiple),
        ),
    )


def validate_database_name(database: str) -> str:
    """Restrict setup/cleanup to an explicitly named test database namespace."""
    if (
        not database.startswith("mcp-live-tests-")
        or not database.replace("-", "").isalnum()
    ):
        raise ValueError(
            "database must start with mcp-live-tests- and contain only letters, digits, and hyphens"
        )
    return database


def validate_container(properties: dict, fixture: ContainerFixture) -> None:
    """Reject incompatible existing policies rather than modifying shared resources."""
    expected = {
        "partitionKey": {"paths": ["/tenant"], "kind": "Hash"},
        "fullTextPolicy": fixture.full_text_policy(),
        "indexingPolicy": fixture.indexing_policy(),
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            actual = properties.get(section, {}).get(key)
            if (
                section == "indexingPolicy"
                and key == "includedPaths"
                and isinstance(actual, list)
            ):
                # ARM adds indexes:null to included paths. Ignore only that null
                # representation, not explicit index lists or changed paths.
                actual = [
                    {
                        name: setting
                        for name, setting in entry.items()
                        if name != "indexes" or setting is not None
                    }
                    for entry in actual
                ]
            if (
                isinstance(value, list)
                and isinstance(actual, list)
                and key
                in (
                    "fullTextPaths",
                    "fullTextIndexes",
                    "includedPaths",
                    "excludedPaths",
                )
            ):
                actual = sorted(actual, key=lambda entry: entry["path"])
                value = sorted(value, key=lambda entry: entry["path"])
            if actual != value:
                raise ValueError(
                    f"{fixture.name}: incompatible {section}.{key}; use a new fixture database/version"
                )
    if properties.get("defaultTtl") not in (None, -1):
        raise ValueError(f"{fixture.name}: fixture documents must not expire")


def validate_items(
    actual: list[dict], fixture: ContainerFixture, *, allow_missing: bool = False
) -> list[dict]:
    """Return missing seed records, rejecting changed or unexpected records first."""
    expected = {(item["id"], item["tenant"]): item for item in fixture.items}
    found = set()
    # Cosmos generates these values per deployment. Compare application content,
    # not service metadata; this does not alter stored items or concurrency checks.
    system_fields = {"_rid", "_self", "_etag", "_attachments", "_ts"}
    for item in actual:
        key = (item.get("id"), item.get("tenant"))
        record = {
            name: value for name, value in item.items() if name not in system_fields
        }
        if key not in expected or record != expected[key] or key in found:
            raise ValueError(
                f"{fixture.name}: fixture records differ; refusing to overwrite or delete"
            )
        found.add(key)
    missing = [item for key, item in expected.items() if key not in found]
    if missing and not allow_missing:
        raise ValueError(
            f"{fixture.name}: missing fixtures; run setup_cosmos_live_tests.py"
        )
    return missing


def read_fixture_items(container, fixture: ContainerFixture) -> list[dict]:
    """Read at most the expected count plus one, to detect foreign records cheaply."""
    return list(
        container.query_items(
            query="SELECT TOP @limit * FROM c",
            parameters=[{"name": "@limit", "value": len(fixture.items) + 1}],
            enable_cross_partition_query=True,
        )
    )


def service_settings(
    endpoint: str,
    database: str,
    credential: str,
    *,
    selected: tuple[ContainerFixture, ...] | None = None,
) -> RetrieverSettings:
    selected = fixtures() if selected is None else selected
    return RetrieverSettings(
        account_uri=endpoint,
        cosmos_database=validate_database_name(database),
        cosmos_key=None,
        cosmos_credential=credential,
        cosmos_containers={
            fixture.name: {
                "cosmos_schema": fixture.schema,
                "search_text_fields": list(fixture.search_fields),
            }
            for fixture in selected
        },
    )
