"""Provision persistent Cosmos live-test fixtures (synthetic + optional SciFact).

Creates missing databases/containers via Azure CLI and seeds deterministic
records via the Cosmos SDK. It never deletes anything, never mutates an
incompatible existing container, and refuses any database outside the
``mcp-live-tests-`` namespace. Fixtures persist; tests run separately.

SciFact source: https://github.com/beir-cellar/beir (archive March 2021). The
archive stays local and no dataset text is bundled with the application.
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceExistsError
from azure.identity import AzureCliCredential, DefaultAzureCredential

from cosmos_agentic_retriever.config import RetrieverConfig

DEFAULT_DATABASE = "mcp-live-tests-v1"
FIXTURE_VERSION = "mcp-search-v1"
SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
SCIFACT_CONTAINER = "scifact-100-v1"
DOCUMENT_COUNT = 100
QUERY_COUNT = 5
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024


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
            # Cosmos excludes _etag by default; declare it so round-trip validation matches.
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "fullTextIndexes": [{"path": path} for path in self.indexed_paths],
        }


@dataclass(frozen=True)
class CorpusQuery:
    query_id: str
    text: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class RealCorpus:
    fixture: ContainerFixture
    queries: tuple[CorpusQuery, ...]
    selection_sha256: str


def fixtures() -> tuple[ContainerFixture, ...]:
    """Nine deterministic synthetic documents with fixed marker words."""
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
                "additional_return_paths": ["/year"],
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
                "additional_return_paths": ["/year"],
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


def load_scifact(path: Path) -> RealCorpus:
    """Load a checksum-pinned 100-doc SciFact subset covering five test queries."""
    with path.open("rb") as source:
        archive_bytes = source.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("SciFact archive exceeds the 8 MiB input limit")
    if hashlib.sha256(archive_bytes).hexdigest() != SCIFACT_SHA256:
        raise ValueError("SciFact archive SHA-256 mismatch; use the pinned source")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:

        def read_member(name: str) -> str:
            return archive.read(f"scifact/{name}").decode("utf-8")

        corpus = {
            record["_id"]: record
            for line in read_member("corpus.jsonl").splitlines()
            if (record := json.loads(line))
        }
        queries = {
            record["_id"]: record["text"]
            for line in read_member("queries.jsonl").splitlines()
            if (record := json.loads(line))
        }
        qrels: dict[str, set[str]] = {}
        for row in csv.DictReader(
            io.StringIO(read_member("qrels/test.tsv")), delimiter="\t"
        ):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    query_ids = sorted(qrels, key=int)[:QUERY_COUNT]
    if len(query_ids) != QUERY_COUNT or len(corpus) < DOCUMENT_COUNT:
        raise ValueError("SciFact input has fewer documents or queries than expected")
    selected_ids = set().union(*(qrels[query_id] for query_id in query_ids))
    if len(selected_ids) > DOCUMENT_COUNT or not selected_ids <= corpus.keys():
        raise ValueError("SciFact relevance labels do not fit the selected corpus")
    for item_id in sorted(corpus, key=int):
        if len(selected_ids) == DOCUMENT_COUNT:
            break
        selected_ids.add(item_id)
    items = tuple(
        {
            "id": item_id,
            "tenant": "scifact",
            "title": corpus[item_id]["title"],
            "text": corpus[item_id]["text"],
            "source": "beir-scifact",
            "source_sha256": SCIFACT_SHA256,
        }
        for item_id in sorted(selected_ids, key=int)
    )
    selected_queries = tuple(
        CorpusQuery(query_id, queries[query_id], frozenset(qrels[query_id]))
        for query_id in query_ids
    )
    selection = {
        "items": items,
        "queries": [
            {
                "id": query.query_id,
                "text": query.text,
                "relevant_ids": sorted(query.relevant_ids, key=int),
            }
            for query in selected_queries
        ],
    }
    digest = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RealCorpus(
        ContainerFixture(
            SCIFACT_CONTAINER,
            {
                "item_id_path": "/id",
                "partition_key_paths": ["/tenant"],
                "text_paths": ["/title", "/text"],
                "additional_return_paths": ["/source"],
            },
            ("/title", "/text"),
            ("/title", "/text"),
            items,
        ),
        selected_queries,
        digest,
    )


def select_data(
    archive: Path,
) -> tuple[tuple[ContainerFixture, ...], RealCorpus]:
    """Return the synthetic fixtures plus the real SciFact corpus."""
    real = load_scifact(archive)
    return (*fixtures(), real.fixture), real


def validate_database_name(database: str) -> str:
    """Restrict setup to the explicit mcp-live-tests- namespace."""
    if not database.startswith("mcp-live-tests-") or not database.replace(
        "-", ""
    ).isalnum():
        raise ValueError(
            "database must start with mcp-live-tests- and contain only letters, digits, and hyphens"
        )
    return database


def validate_container(properties: dict, fixture: ContainerFixture) -> None:
    """Reject an incompatible existing container rather than modifying it."""
    expected = {
        "partitionKey": {"paths": ["/tenant"], "kind": "Hash"},
        "fullTextPolicy": fixture.full_text_policy(),
        "indexingPolicy": fixture.indexing_policy(),
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            actual = properties.get(section, {}).get(key)
            if section == "indexingPolicy" and key == "includedPaths" and isinstance(
                actual, list
            ):
                # ARM adds indexes:null to included paths; ignore only that null.
                actual = [
                    {n: s for n, s in entry.items() if n != "indexes" or s is not None}
                    for entry in actual
                ]
            if (
                isinstance(value, list)
                and isinstance(actual, list)
                and key
                in ("fullTextPaths", "fullTextIndexes", "includedPaths", "excludedPaths")
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
    found: set = set()
    system_fields = {"_rid", "_self", "_etag", "_attachments", "_ts"}
    for item in actual:
        key = (item.get("id"), item.get("tenant"))
        record = {n: v for n, v in item.items() if n not in system_fields}
        if key not in expected or record != expected[key] or key in found:
            raise ValueError(
                f"{fixture.name}: fixture records differ; refusing to overwrite or delete"
            )
        found.add(key)
    missing = [item for key, item in expected.items() if key not in found]
    if missing and not allow_missing:
        raise ValueError(f"{fixture.name}: missing fixtures; run this setup tool")
    return missing


def read_fixture_items(container, fixture: ContainerFixture) -> list[dict]:
    """Read expected count plus one to detect foreign records cheaply."""
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
) -> RetrieverConfig:
    """Build service settings for the selected fixture containers."""
    selected = fixtures() if selected is None else selected
    return RetrieverConfig(
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


def az_json(arguments: list[str]):
    """Run a noninteractive az management command without a shell or account keys."""
    result = subprocess.run(
        ["az", *arguments, "--only-show-errors", "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def prepare_resources(args, selected: tuple[ContainerFixture, ...]) -> str:
    """Create only missing database/containers; verify existing ones first."""
    validate_database_name(args.database)
    scope = [
        "--subscription", args.subscription,
        "--resource-group", args.resource_group,
        "--account-name", args.account,
    ]
    account = az_json(
        [
            "cosmosdb", "show",
            "--subscription", args.subscription,
            "--resource-group", args.resource_group,
            "--name", args.account,
            "--query", "{endpoint:documentEndpoint,capabilities:capabilities}",
        ]
    )
    endpoint = account["endpoint"]
    databases = az_json(["cosmosdb", "sql", "database", "list", *scope])
    if args.database not in {entry["name"] for entry in databases}:
        options = (
            []
            if any(cap["name"] == "EnableServerless" for cap in account["capabilities"])
            else ["--throughput", "400"]
        )
        az_json(
            ["cosmosdb", "sql", "database", "create", *scope, "--name", args.database, *options]
        )
    listed = az_json(
        ["cosmosdb", "sql", "container", "list", *scope, "--database-name", args.database]
    )
    existing = {entry["name"]: entry["resource"] for entry in listed}
    for fixture in selected:
        if fixture.name in existing:
            validate_container(existing[fixture.name], fixture)
        else:
            created = az_json(
                [
                    "cosmosdb", "sql", "container", "create", *scope,
                    "--database-name", args.database,
                    "--name", fixture.name,
                    "--partition-key-path", "/tenant",
                    "--partition-key-version", "2",
                    "--full-text-policy", json.dumps(fixture.full_text_policy()),
                    "--idx", json.dumps(fixture.indexing_policy()),
                ]
            )
            validate_container(created["resource"], fixture)
    return endpoint


def ensure_records(database, selected: tuple[ContainerFixture, ...]) -> int:
    """Seed missing deterministic records after existing records pass checks."""
    plans = []
    for fixture in selected:
        container = database.get_container_client(fixture.name)
        validate_container(container.read(), fixture)
        missing = validate_items(
            read_fixture_items(container, fixture), fixture, allow_missing=True
        )
        plans.append((container, fixture, missing))
    created = 0
    for container, fixture, missing in plans:
        for item in missing:
            try:
                container.create_item(item)
                created += 1
            except CosmosResourceExistsError:
                found = container.read_item(item=item["id"], partition_key=item["tenant"])
                validate_items([found], fixture, allow_missing=True)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--scifact-archive", required=True, type=Path)
    parser.add_argument(
        "--credential", choices=["azure_cli", "default"], default="azure_cli"
    )
    args = parser.parse_args(argv)
    try:
        validate_database_name(args.database)
        selected, real = select_data(args.scifact_archive)
        print(f"Account: {args.account}; fixture database: {args.database}")
        print(f"Containers: {', '.join(f.name for f in selected)}")
        print(f"Expected records: {sum(len(f.items) for f in selected)}")
        print(f"SciFact selection SHA-256: {real.selection_sha256}")
        endpoint = prepare_resources(args, selected)
        with ExitStack() as stack:
            credential = (
                AzureCliCredential()
                if args.credential == "azure_cli"
                else DefaultAzureCredential()
            )
            stack.callback(credential.close)
            client = CosmosClient(endpoint, credential=credential)
            stack.callback(client.close)
            created = ensure_records(
                client.get_database_client(args.database), selected
            )
        print(f"Fixtures verified; created {created} records. No tests were run.")
        print("Set these non-secret test variables, then run pytest separately:")
        print(
            f"COSMOS_TEST_ENDPOINT={endpoint}\n"
            f"COSMOS_TEST_DATABASE={args.database}\n"
            f"COSMOS_TEST_CREDENTIAL={args.credential}\n"
            f"COSMOS_TEST_SCIFACT_ARCHIVE={args.scifact_archive.resolve()}"
        )
        return 0
    except (ValueError, AzureError, subprocess.SubprocessError, OSError) as error:
        detail = (
            error
            if isinstance(error, ValueError)
            else "Check Azure CLI authentication, permissions, and full-text support."
        )
        print(f"Fixture setup failed ({type(error).__name__}). {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
