"""Offline safety checks for the future live-test setup; never contact Azure."""

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cosmos_live_corpus as corpus_loader
import pytest
import setup_cosmos_live_tests as setup
from azure.cosmos.exceptions import CosmosResourceExistsError
from cosmos_live_fixtures import (
    ContainerFixture,
    fixtures,
    service_settings,
    validate_container,
    validate_database_name,
    validate_items,
)
from setup_cosmos_live_tests import ensure_records, prepare_resources


@pytest.fixture
def scifact_archive(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "scifact/corpus.jsonl",
            "\n".join(
                json.dumps(
                    {
                        "_id": str(number),
                        "title": f"Title {number}",
                        "text": f"Text {number}",
                    }
                )
                for number in range(1, 111)
            ),
        )
        output.writestr(
            "scifact/queries.jsonl",
            "\n".join(
                json.dumps({"_id": str(number), "text": f"Query {number}"})
                for number in range(1, 7)
            ),
        )
        output.writestr(
            "scifact/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\n"
            + "\n".join(f"{number}\t{100 + number}\t1" for number in range(1, 7)),
        )
    payload = archive.getvalue()
    path = tmp_path / "scifact.zip"
    path.write_bytes(payload)
    monkeypatch.setattr(
        corpus_loader, "SCIFACT_SHA256", hashlib.sha256(payload).hexdigest()
    )
    return path


def test_scifact_selection_is_repeatable_and_preserves_source(scifact_archive):
    real = corpus_loader.load_scifact(scifact_archive)
    assert real == corpus_loader.load_scifact(scifact_archive)
    assert len(real.fixture.items) == 100
    assert [query.query_id for query in real.queries] == ["1", "2", "3", "4", "5"]
    expected_ids = {str(number) for number in range(1, 96)} | {
        str(number) for number in range(101, 106)
    }
    assert {item["id"] for item in real.fixture.items} == expected_ids
    assert real.fixture.items[-1]["text"] == "Text 105"
    assert all(query.relevant_ids <= expected_ids for query in real.queries)
    assert len(real.selection_sha256) == 64
    selected, loaded = corpus_loader.select_data("both", scifact_archive)
    assert [fixture.name for fixture in selected] == [
        *(fixture.name for fixture in fixtures()),
        "scifact-100-v1",
    ]
    assert loaded == real
    assert len(corpus_loader.select_data("scifact", scifact_archive)[0]) == 1


def test_scifact_bad_input_fails_before_azure_calls(scifact_archive, monkeypatch):
    execute = Mock()
    monkeypatch.setattr(setup, "az_json", execute)
    scifact_archive.write_bytes(b"changed archive")
    assert (
        setup.main(
            [
                "--subscription",
                "sub",
                "--resource-group",
                "rg",
                "--account",
                "account",
                "--data",
                "both",
                "--scifact-archive",
                str(scifact_archive),
            ]
        )
        == 1
    )
    execute.assert_not_called()


@pytest.mark.parametrize(
    "data,archive",
    [("unknown", None), ("both", None), ("synthetic", Path("unused.zip"))],
)
def test_invalid_data_selection_is_rejected(data, archive):
    with pytest.raises(ValueError):
        corpus_loader.select_data(data, archive)


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda fixture: fixture.name)
def test_fixture_schema_and_records_are_consistent(fixture):
    properties = {
        "partitionKey": {"paths": ["/tenant"], "kind": "Hash"},
        "fullTextPolicy": fixture.full_text_policy(),
        "indexingPolicy": fixture.indexing_policy(),
    }
    validate_container(properties, fixture)
    assert validate_items(list(fixture.items), fixture) == []
    assert validate_items([], fixture, allow_missing=True) == list(fixture.items)
    with pytest.raises(ValueError, match="missing"):
        validate_items([], fixture)
    for section, key in (
        ("partitionKey", "paths"),
        ("fullTextPolicy", "fullTextPaths"),
        ("indexingPolicy", "fullTextIndexes"),
    ):
        changed = deepcopy(properties)
        changed[section][key] = []
        with pytest.raises(ValueError, match="incompatible"):
            validate_container(changed, fixture)
    for changed in (
        [{**fixture.items[0], "fixture_version": "foreign"}],
        [*fixture.items, {"id": "foreign", "tenant": 0}],
    ):
        with pytest.raises(ValueError, match="refusing"):
            validate_items(changed, fixture, allow_missing=True)


def test_fixture_names_and_service_settings_are_explicit():
    settings = service_settings(
        "https://example.documents.azure.com", "mcp-live-tests-v1", "azure_cli"
    )
    assert list(settings.cosmos_containers) == ["flat-v1", "nested-v1", "fields-v1"]
    assert settings.cosmos_containers[
        "nested-v1"
    ].cosmos_schema.item_id_path.segments == ("record", "id")
    with pytest.raises(ValueError):
        validate_database_name("production")


def _properties(fixture):
    return {
        "partitionKey": {"paths": ["/tenant"], "kind": "Hash"},
        "fullTextPolicy": fixture.full_text_policy(),
        "indexingPolicy": fixture.indexing_policy(),
    }


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda fixture: fixture.name)
def test_live_policy_defaults_are_accepted_without_mutation(fixture):
    properties = _properties(fixture)
    properties["indexingPolicy"]["includedPaths"] = [{"path": "/*", "indexes": None}]
    properties["indexingPolicy"]["excludedPaths"] = [{"path": '/"_etag"/?'}]
    properties["indexingPolicy"]["compositeIndexes"] = None
    properties["fullTextPolicy"]["defaultSpec"] = {
        "language": "en-US",
        "stopWordListKind": "extended",
    }
    properties["partitionKey"].update(version=2, systemKey=None)
    properties["defaultTtl"] = None
    original = deepcopy(properties)
    validate_container(properties, fixture)
    assert properties == original


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (
            "indexingPolicy",
            "includedPaths",
            [{"path": "/different/*", "indexes": None}],
        ),
        ("indexingPolicy", "includedPaths", [{"path": "/*", "indexes": []}]),
        ("indexingPolicy", "excludedPaths", [{"path": "/text/?"}]),
        ("indexingPolicy", "fullTextIndexes", []),
        (None, "defaultTtl", 60),
        (None, "defaultTtl", 0),
    ],
)
def test_live_policy_normalization_still_rejects_changes(section, key, value):
    fixture = fixtures()[0]
    properties = _properties(fixture)
    target = properties if section is None else properties[section]
    target[key] = value
    with pytest.raises(ValueError, match="incompatible|must not expire"):
        validate_container(properties, fixture)


def _database(records, *, selected=None):
    containers = {}
    for fixture in fixtures() if selected is None else selected:
        container = Mock()
        container.read.return_value = _properties(fixture)
        container.query_items.return_value = records(fixture)
        containers[fixture.name] = container
    database = Mock()
    database.get_container_client.side_effect = containers.__getitem__
    return database, containers


def test_explicit_container_set_controls_setup_records_and_settings(monkeypatch):
    fixture = ContainerFixture(
        "real-v1",
        {"item_id_path": "/id", "text_paths": ["/text"]},
        ("/text",),
        ("/text",),
        ({"id": "source-1", "tenant": "source", "text": "source text"},),
    )
    selected = (fixture,)
    execute = Mock(
        side_effect=[
            {"endpoint": "https://example.documents.azure.com", "capabilities": []},
            [{"name": "mcp-live-tests-v1"}],
            [],
            {"resource": _properties(fixture)},
        ]
    )
    monkeypatch.setattr(setup, "az_json", execute)
    args = SimpleNamespace(
        subscription="sub",
        resource_group="rg",
        account="account",
        database="mcp-live-tests-v1",
        check_only=False,
        cleanup=False,
        confirm_cleanup=None,
    )
    prepare_resources(args, selected=selected)
    assert len(execute.call_args_list) == 4
    command = execute.call_args_list[-1].args[0]
    assert command[command.index("--name") + 1] == "real-v1"
    database, container = Mock(), Mock()
    database.get_container_client.return_value = container
    container.read.return_value = _properties(fixture)
    container.query_items.return_value = []
    assert ensure_records(database, check_only=False, selected=selected) == 1
    database.get_container_client.assert_called_once_with("real-v1")
    container.create_item.assert_called_once_with(fixture.items[0])
    settings = service_settings(
        "https://example.documents.azure.com",
        "mcp-live-tests-v1",
        "azure_cli",
        selected=selected,
    )
    assert list(settings.cosmos_containers) == ["real-v1"]


@pytest.mark.parametrize("check_only", [False, True])
def test_existing_records_are_never_reingested(check_only):
    database, containers = _database(lambda fixture: list(fixture.items))
    assert ensure_records(database, check_only=check_only) == 0
    for container in containers.values():
        container.create_item.assert_not_called()
        container.upsert_item.assert_not_called()


def test_only_missing_records_created():
    database, containers = _database(lambda fixture: list(fixture.items[1:]))
    assert ensure_records(database, check_only=False) == 3
    for fixture in fixtures():
        containers[fixture.name].create_item.assert_called_once_with(fixture.items[0])


def test_mismatch_prevents_all_data_writes():
    database, containers = _database(
        lambda fixture: [] if fixture.name != "fields-v1" else [{"id": "foreign"}]
    )
    with pytest.raises(ValueError, match="refusing"):
        ensure_records(database, check_only=False)
    for container in containers.values():
        container.create_item.assert_not_called()


@pytest.mark.parametrize("check_only", [False, True])
def test_existing_management_resources_are_not_updated(monkeypatch, check_only):
    execute = Mock(
        side_effect=[
            {"endpoint": "https://example.documents.azure.com", "capabilities": []},
            [{"name": "mcp-live-tests-v1"}],
            [
                {"name": fixture.name, "resource": _properties(fixture)}
                for fixture in fixtures()
            ],
        ]
    )
    monkeypatch.setattr("setup_cosmos_live_tests.az_json", execute)
    args = SimpleNamespace(
        subscription="sub",
        resource_group="rg",
        account="account",
        database="mcp-live-tests-v1",
        check_only=check_only,
        cleanup=False,
        confirm_cleanup=None,
    )
    assert prepare_resources(args) == "https://example.documents.azure.com"
    assert len(execute.call_args_list) == 3
    assert all("create" not in call.args[0] for call in execute.call_args_list)


def test_cleanup_confirmation_before_any_azure_call(monkeypatch):
    execute = Mock()
    monkeypatch.setattr("setup_cosmos_live_tests.az_json", execute)
    with pytest.raises(ValueError, match="confirm-cleanup"):
        prepare_resources(
            SimpleNamespace(
                database="mcp-live-tests-v1", cleanup=True, confirm_cleanup=None
            )
        )
    execute.assert_not_called()


@pytest.mark.parametrize("serverless", [False, True])
def test_missing_resources_created_with_full_text_and_partition_policy(
    monkeypatch, serverless
):
    account = {
        "endpoint": "https://example.documents.azure.com",
        "capabilities": [{"name": "EnableServerless"}] if serverless else [],
    }
    execute = Mock(
        side_effect=[
            account,
            [],
            {},
            [],
            *[{"resource": _properties(fixture)} for fixture in fixtures()],
        ]
    )
    monkeypatch.setattr(setup, "az_json", execute)
    args = SimpleNamespace(
        subscription="sub",
        resource_group="rg",
        account="account",
        database="mcp-live-tests-v1",
        check_only=False,
        cleanup=False,
        confirm_cleanup=None,
    )
    prepare_resources(args)
    database_command = execute.call_args_list[2].args[0]
    assert database_command[:4] == ["cosmosdb", "sql", "database", "create"]
    assert ("--throughput" in database_command) is not serverless
    for call, fixture in zip(execute.call_args_list[4:], fixtures(), strict=True):
        command = call.args[0]
        assert command[:4] == ["cosmosdb", "sql", "container", "create"]
        assert command[command.index("--name") + 1] == fixture.name
        assert command[command.index("--partition-key-path") + 1] == "/tenant"
        assert (
            json.loads(command[command.index("--idx") + 1]) == fixture.indexing_policy()
        )
        assert (
            json.loads(command[command.index("--full-text-policy") + 1])
            == fixture.full_text_policy()
        )
        assert "--throughput" not in command


@pytest.mark.parametrize("missing", ["database", "container", "record"])
def test_check_only_missing_resources_never_write(monkeypatch, missing):
    if missing == "record":
        database, containers = _database(lambda fixture: [])
        with pytest.raises(ValueError, match="missing"):
            ensure_records(database, check_only=True)
        for container in containers.values():
            container.create_item.assert_not_called()
        return
    execute = Mock(
        side_effect=[
            {"endpoint": "https://example.documents.azure.com", "capabilities": []},
            [] if missing == "database" else [{"name": "mcp-live-tests-v1"}],
            [],
        ]
    )
    monkeypatch.setattr(setup, "az_json", execute)
    with pytest.raises(ValueError, match="missing"):
        prepare_resources(
            SimpleNamespace(
                subscription="sub",
                resource_group="rg",
                account="account",
                database="mcp-live-tests-v1",
                check_only=True,
                cleanup=False,
                confirm_cleanup=None,
            )
        )
    assert all(
        "create" not in call.args[0] and "delete" not in call.args[0]
        for call in execute.call_args_list
    )


def test_concurrent_seed_conflict_checks_existing_record():
    database, containers = _database(lambda fixture: list(fixture.items[1:]))
    for fixture in fixtures():
        container = containers[fixture.name]
        container.create_item.side_effect = CosmosResourceExistsError(status_code=409)
        container.read_item.return_value = fixture.items[0]
    assert ensure_records(database, check_only=False) == 0
    containers["flat-v1"].read_item.return_value = {
        "id": "shared",
        "tenant": 0,
        "text": "foreign",
    }
    with pytest.raises(ValueError, match="refusing"):
        ensure_records(database, check_only=False)


@pytest.mark.parametrize("foreign", [False, True])
@pytest.mark.parametrize("data", ["synthetic", "scifact", "both"])
def test_cleanup_only_deletes_validated_fixture_containers(
    monkeypatch, foreign, data, scifact_archive
):
    archive = None if data == "synthetic" else scifact_archive
    selected, _ = corpus_loader.select_data(data, archive)
    execute = Mock()
    monkeypatch.setattr(setup, "az_json", execute)
    monkeypatch.setattr(
        setup,
        "prepare_resources",
        Mock(return_value="https://example.documents.azure.com"),
    )
    database, containers = _database(
        lambda fixture: list(fixture.items), selected=selected
    )
    if foreign:
        containers[selected[-1].name].query_items.return_value = [
            {"id": "not-ours", "tenant": 0}
        ]
    client, credential = Mock(), Mock()
    client.get_database_client.return_value = database
    monkeypatch.setattr(setup, "CosmosClient", Mock(return_value=client))
    monkeypatch.setattr(setup, "AzureCliCredential", Mock(return_value=credential))
    status = setup.main(
        [
            "--subscription",
            "sub",
            "--resource-group",
            "rg",
            "--account",
            "account",
            "--data",
            data,
            *(["--scifact-archive", str(archive)] if archive is not None else []),
            "--cleanup",
            "--confirm-cleanup",
            "mcp-live-tests-v1",
        ]
    )
    assert status == (1 if foreign else 0)
    if foreign:
        execute.assert_not_called()
    else:
        assert len(execute.call_args_list) == len(selected)
        for call, fixture in zip(execute.call_args_list, selected, strict=True):
            command = call.args[0]
            assert command[:4] == ["cosmosdb", "sql", "container", "delete"]
            assert command[command.index("--name") + 1] == fixture.name
    client.close.assert_called_once()
    credential.close.assert_called_once()


def test_setup_cli_failure_is_nonzero_and_does_not_run_pytest(monkeypatch, capsys):

    execute = Mock(
        side_effect=subprocess.CalledProcessError(1, "az", stderr="private-credential")
    )
    monkeypatch.setattr(setup, "az_json", execute)
    assert (
        setup.main(
            ["--subscription", "sub", "--resource-group", "rg", "--account", "account"]
        )
        == 1
    )
    assert "private-credential" not in capsys.readouterr().err


@pytest.mark.parametrize("enabled", [False, True])
def test_live_gate_skips_offline_but_fails_missing_configuration(enabled):
    environment = dict(
        os.environ,
        RUN_COSMOS_LIVE="1" if enabled else "0",
        COSMOS_TEST_DATA="synthetic",
    )
    environment.pop("COSMOS_TEST_ENDPOINT", None)
    environment.pop("COSMOS_TEST_SCIFACT_ARCHIVE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/end_to_end/test_cosmos_search_integration.py",
            "-k",
            "test_same_id_survives_in_all_containers",
            "-q",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (1 if enabled else 0), result.stdout + result.stderr
    assert (
        "COSMOS_TEST_ENDPOINT is required" if enabled else "1 skipped"
    ) in result.stdout
