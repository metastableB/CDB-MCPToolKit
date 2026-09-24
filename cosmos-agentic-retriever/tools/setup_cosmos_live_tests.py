"""Prepare persistent live-test fixtures in an existing Cosmos account.

Uses Azure CLI management commands for database/container creation and Azure
identity with the Cosmos SDK for records. This script does not run tests, change
account capabilities, grant roles, or delete the database. Fixtures stay by default.
"""

import argparse
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceExistsError
from azure.identity import AzureCliCredential, DefaultAzureCredential
from cosmos_live_corpus import select_data
from cosmos_live_fixtures import (
    DEFAULT_DATABASE,
    ContainerFixture,
    fixtures,
    read_fixture_items,
    validate_container,
    validate_database_name,
    validate_items,
)


def az_json(arguments: list[str]):
    """Run a noninteractive management command without a shell or account keys."""
    result = subprocess.run(
        ["az", *arguments, "--only-show-errors", "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def prepare_resources(
    args, *, selected: tuple[ContainerFixture, ...] | None = None
) -> str:
    """Create only missing resources and verify existing policies before any seeding."""
    selected = fixtures() if selected is None else selected
    validate_database_name(args.database)
    if args.cleanup and args.confirm_cleanup != args.database:
        raise ValueError(
            "cleanup requires --confirm-cleanup with the exact test database name"
        )
    scope = [
        "--subscription",
        args.subscription,
        "--resource-group",
        args.resource_group,
        "--account-name",
        args.account,
    ]
    account = az_json(
        [
            "cosmosdb",
            "show",
            "--subscription",
            args.subscription,
            "--resource-group",
            args.resource_group,
            "--name",
            args.account,
            "--query",
            "{endpoint:documentEndpoint,capabilities:capabilities}",
        ]
    )
    endpoint = account["endpoint"]
    databases = az_json(["cosmosdb", "sql", "database", "list", *scope])
    if args.database not in {entry["name"] for entry in databases}:
        if args.check_only or args.cleanup:
            raise ValueError("test database is missing; run setup without --check-only")
        options = (
            []
            if any(cap["name"] == "EnableServerless" for cap in account["capabilities"])
            else ["--throughput", "400"]
        )
        az_json(
            [
                "cosmosdb",
                "sql",
                "database",
                "create",
                *scope,
                "--name",
                args.database,
                *options,
            ]
        )
    listed = az_json(
        [
            "cosmosdb",
            "sql",
            "container",
            "list",
            *scope,
            "--database-name",
            args.database,
        ]
    )
    existing = {entry["name"]: entry["resource"] for entry in listed}
    for fixture in selected:
        if fixture.name in existing:
            validate_container(existing[fixture.name], fixture)
        elif args.check_only or args.cleanup:
            raise ValueError(f"{fixture.name}: missing container")
    if not args.check_only and not args.cleanup:
        for fixture in selected:
            if fixture.name not in existing:
                created = az_json(
                    [
                        "cosmosdb",
                        "sql",
                        "container",
                        "create",
                        *scope,
                        "--database-name",
                        args.database,
                        "--name",
                        fixture.name,
                        "--partition-key-path",
                        "/tenant",
                        "--partition-key-version",
                        "2",
                        "--full-text-policy",
                        json.dumps(fixture.full_text_policy()),
                        "--idx",
                        json.dumps(fixture.indexing_policy()),
                    ]
                )
                validate_container(created["resource"], fixture)
    return endpoint


def ensure_records(
    database,
    *,
    check_only: bool,
    selected: tuple[ContainerFixture, ...] | None = None,
) -> int:
    """Seed missing deterministic records only after all existing records pass checks."""
    selected = fixtures() if selected is None else selected
    plans = []
    for fixture in selected:
        container = database.get_container_client(fixture.name)
        validate_container(container.read(), fixture)
        missing = validate_items(
            read_fixture_items(container, fixture),
            fixture,
            allow_missing=not check_only,
        )
        plans.append((container, fixture, missing))
    created = 0
    for container, fixture, missing in plans:
        for item in missing:
            try:
                container.create_item(item)
                created += 1
            except CosmosResourceExistsError:
                found = container.read_item(
                    item=item["id"], partition_key=item["tenant"]
                )
                validate_items([found], fixture, allow_missing=True)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument(
        "--data", choices=["synthetic", "scifact", "both"], default="synthetic"
    )
    parser.add_argument("--scifact-archive", type=Path)
    parser.add_argument(
        "--credential", choices=["azure_cli", "default"], default="azure_cli"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check-only", action="store_true", help="Validate fixtures without writes."
    )
    modes.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete only the selected, validated test containers.",
    )
    parser.add_argument(
        "--confirm-cleanup", help="Exact database name required for --cleanup."
    )
    args = parser.parse_args(argv)
    try:
        validate_database_name(args.database)
        selected, real = select_data(args.data, args.scifact_archive)
        print(f"Account: {args.account}; fixture database: {args.database}")
        print(
            f"Data: {args.data}; containers: {', '.join(fixture.name for fixture in selected)}"
        )
        print(f"Expected records: {sum(len(fixture.items) for fixture in selected)}")
        if real is not None:
            print(
                f"SciFact selection SHA-256: {real.selection_sha256}; queries: {len(real.queries)}"
            )
        if not args.check_only and not args.cleanup:
            print(
                "Creates missing fixtures only. A new provisioned database uses 400 shared RU/s (billable); serverless is usage billed."
            )
        endpoint = prepare_resources(args, selected=selected)
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
                client.get_database_client(args.database),
                check_only=args.check_only or args.cleanup,
                selected=selected,
            )
        if args.cleanup:
            for fixture in selected:
                az_json(
                    [
                        "cosmosdb",
                        "sql",
                        "container",
                        "delete",
                        "--subscription",
                        args.subscription,
                        "--resource-group",
                        args.resource_group,
                        "--account-name",
                        args.account,
                        "--database-name",
                        args.database,
                        "--name",
                        fixture.name,
                        "--yes",
                    ]
                )
            print(
                "Fixture containers deleted. Database/account retained; shared throughput may still incur charges."
            )
        else:
            print(
                f"Fixtures verified; created {created} records. Resources retained. No tests were run."
            )
            print("Set these non-secret test variables, then run pytest separately:")
            print(
                f"COSMOS_TEST_ENDPOINT={endpoint}\nCOSMOS_TEST_DATABASE={args.database}\nCOSMOS_TEST_CREDENTIAL={args.credential}"
            )
            print(f"COSMOS_TEST_DATA={args.data}")
            if args.scifact_archive is not None:
                print(f"COSMOS_TEST_SCIFACT_ARCHIVE={args.scifact_archive.resolve()}")
        return 0
    except (ValueError, AzureError, subprocess.SubprocessError, OSError) as error:
        print(
            f"Fixture setup failed ({type(error).__name__}). {error if isinstance(error, ValueError) else 'Check Azure CLI authentication, permissions, and full-text support.'}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
