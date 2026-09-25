"""Validate service configuration without credentials or network access."""

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from cosmos_agentic_retriever import __main__ as cli
from cosmos_agentic_retriever.config import (
    ContainerConfig,
    RetrieverSettings,
    get_settings,
)
from cosmos_agentic_retriever.query_engine.types import UnknownField


def _environment(monkeypatch):
    values = {
        "ACCOUNT_URI": "https://example.documents.azure.com",
        "COSMOS_DATABASE": "D",
        "COSMOS_CONTAINERS": json.dumps(
            {
                "C": {
                    "cosmos_schema": {
                        "item_id_path": "/id",
                        "partition_key_paths": ["/tenant"],
                        "text_paths": ["/text"],
                    },
                    "search_text_fields": ["/text"],
                }
            }
        ),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_settings_from_environment(monkeypatch):
    _environment(monkeypatch)
    monkeypatch.setenv("QUERY_ENGINE", '{"max_concurrency":2}')
    monkeypatch.setenv("COSMOS_KEY", "test-secret")
    monkeypatch.setenv("COSMOS_CREDENTIAL", "default")
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "9100")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    settings = get_settings()
    assert settings.cosmos_database == "D"
    assert list(settings.cosmos_containers) == ["C"]
    assert str(settings.cosmos_containers["C"].cosmos_schema.text_paths[0]) == "/text"
    assert settings.query_engine.max_concurrency == 2
    assert settings.cosmos_containers["C"].search_text_fields == ["/text"]
    assert settings.cosmos_key.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(settings)
    assert settings.cosmos_credential == "default"
    assert (settings.host, settings.port, settings.log_level) == (
        "127.0.0.2",
        9100,
        "warning",
    )
    monkeypatch.setenv("COSMOS_DATABASE", "changed")
    assert get_settings().cosmos_database == "changed"
    assert settings.cosmos_database == "D"


@pytest.mark.parametrize(
    "name",
    ["ACCOUNT_URI", "COSMOS_DATABASE", "COSMOS_CONTAINERS"],
)
def test_required_settings_fail_before_launch(monkeypatch, name):
    _environment(monkeypatch)
    monkeypatch.delenv(name)
    with pytest.raises(ValidationError) as error:
        get_settings()
    assert any(
        item["loc"] == (name.lower(),) and item["type"] == "missing"
        for item in error.value.errors()
    )


def test_settings_do_not_implicitly_read_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "ACCOUNT_URI=https://example.documents.azure.com\nCOSMOS_DATABASE=D\n"
        'COSMOS_CONTAINERS={"C":{"cosmos_schema":{"item_id_path":"/id","partition_key_paths":["/tenant"],"text_paths":["/text"]}}}\n'
    )
    with pytest.raises(ValidationError):
        get_settings()


@pytest.mark.parametrize(
    "options",
    [
        {"account_uri": "not-a-url"},
        {"cosmos_database": " "},
        {"cosmos_containers": {}},
        {
            "cosmos_containers": {
                "*": {
                    "cosmos_schema": {
                        "item_id_path": "/id",
                        "partition_key_paths": ["/tenant"],
                        "text_paths": ["/text"],
                    }
                }
            }
        },
        {"port": 0},
        {"port": 65536},
        {"cosmos_key": " "},
        {"cosmos_credential": "unknown"},
        {"cosmos_containers": {"C": {"cosmos_schema": {"item_id_path": "/id"}}}},
        {"query_engine": {"max_concurrency": 0}},
        {"typo": True},
    ],
)
def test_settings_reject_invalid_configuration(options):
    values = {
        "account_uri": "https://example.documents.azure.com",
        "cosmos_database": "D",
        "cosmos_containers": {
            "C": {
                "cosmos_schema": {
                    "item_id_path": "/id",
                    "partition_key_paths": ["/tenant"],
                    "text_paths": ["/text"],
                }
            }
        },
    }
    with pytest.raises(ValidationError):
        RetrieverSettings(**{**values, **options})


def test_settings_require_explicit_selection_for_multiple_fields():
    values = {
        "account_uri": "https://example.documents.azure.com",
        "cosmos_database": "D",
        "cosmos_containers": {
            "C": {
                "cosmos_schema": {
                    "item_id_path": "/id",
                    "partition_key_paths": ["/tenant"],
                    "text_paths": ["/title", "/body"],
                }
            }
        },
    }
    with pytest.raises(UnknownField):
        RetrieverSettings(**values)
    values["cosmos_containers"]["C"]["search_text_fields"] = ["/body"]
    settings = RetrieverSettings(**values)
    assert settings.cosmos_containers["C"].search_text_fields == ["/body"]
    values["cosmos_containers"]["C"]["search_text_fields"] = ["body"]
    with pytest.raises(UnknownField):
        RetrieverSettings(**values)


@pytest.mark.parametrize("key", [0, "", "tenant"])
def test_partition_key_is_per_container_and_falsey_keys_are_valid(key):
    config = ContainerConfig(
        cosmos_schema={
            "item_id_path": "/id",
            "partition_key_paths": ["/tenant"],
            "text_paths": ["/text"],
        },
        partition_key=key,
        partition_policy={"allow_cross_partition_search": False},
    )
    assert config.partition_key == key


def test_unscoped_cross_partition_disabled_is_rejected():
    with pytest.raises(ValidationError, match="partition key"):
        ContainerConfig(
            cosmos_schema={
                "item_id_path": "/id",
                "partition_key_paths": ["/tenant"],
                "text_paths": ["/text"],
            },
            partition_policy={"allow_cross_partition_search": False},
        )


@pytest.mark.parametrize("partition_key", [None, "fixed"])
def test_physical_identity_requires_explicit_partition_paths(partition_key):
    with pytest.raises(ValidationError, match="partition_key_paths"):
        ContainerConfig(
            cosmos_schema={"item_id_path": "/id", "text_paths": ["/text"]},
            partition_key=partition_key,
        )


@pytest.mark.parametrize(
    "arguments, host, port",
    [
        (["serve"], "127.0.0.1", 9000),
        (["serve", "--port", "9100"], "127.0.0.1", 9100),
        (["serve", "--host", "0.0.0.0"], "0.0.0.0", 9000),
        (["serve", "--host", "0.0.0.0", "--port", "9100"], "0.0.0.0", 9100),
    ],
)
def test_serve_command(monkeypatch, arguments, host, port):
    settings = RetrieverSettings(
        account_uri="https://example.documents.azure.com",
        cosmos_database="D",
        cosmos_containers={
            "C": {
                "cosmos_schema": {
                    "item_id_path": "/id",
                    "partition_key_paths": ["/tenant"],
                    "text_paths": ["/text"],
                }
            }
        },
    )
    monkeypatch.setattr(cli, "get_settings", Mock(return_value=settings))
    app = object()
    build = Mock(return_value=app)
    run = Mock()
    monkeypatch.setattr(cli, "create_app", build)
    monkeypatch.setattr(cli.uvicorn, "run", run)
    assert cli.main(arguments) == 0
    run.assert_called_once_with(app, host=host, port=port, log_level="info")
    assert build.call_args.args[0].cosmos_database == "D"
    assert settings.host == "127.0.0.1" and settings.port == 9000


def test_invalid_configuration_does_not_start_or_print_values(monkeypatch, capsys):
    monkeypatch.setenv("ACCOUNT_URI", "private-invalid-endpoint")
    monkeypatch.setenv("COSMOS_KEY", "private-key")
    run = Mock()
    monkeypatch.setattr(cli.uvicorn, "run", run)
    assert cli.main(["serve"]) == 1
    run.assert_not_called()
    error = capsys.readouterr().err
    assert "Invalid service configuration" in error
    assert "private-" not in error


@pytest.mark.parametrize(
    "options",
    [
        {"COSMOS_CONTAINERS": "private-malformed-json"},
        {"QUERY_ENGINE": "private-malformed-json"},
        {
            "COSMOS_CONTAINERS": '{"C":{"cosmos_schema":{"item_id_path":"/id","partition_key_paths":["/tenant"],"text_paths":["/text"]},"search_text_fields":["/unconfigured"]}}'
        },
    ],
)
def test_invalid_environment_never_builds_app(monkeypatch, capsys, options):
    _environment(monkeypatch)
    for name, value in options.items():
        monkeypatch.setenv(name, value)
    build = Mock()
    monkeypatch.setattr(cli, "create_app", build)
    assert cli.main(["serve"]) == 1
    build.assert_not_called()
    assert capsys.readouterr().err == (
        "Invalid service configuration. Check the connection, schema, and server settings.\n"
    )


@pytest.mark.parametrize(
    "arguments", [["--port", "0"], ["--port", "65536"], ["--host", ""]]
)
def test_invalid_cli_overrides_never_build_app(monkeypatch, arguments):
    _environment(monkeypatch)
    build = Mock()
    monkeypatch.setattr(cli, "create_app", build)
    assert cli.main(["serve", *arguments]) == 1
    build.assert_not_called()


def test_cli_overrides_environment_without_changing_target(monkeypatch):
    _environment(monkeypatch)
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "9200")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    build, run = Mock(), Mock()
    monkeypatch.setattr(cli, "create_app", build)
    monkeypatch.setattr(cli.uvicorn, "run", run)
    assert cli.main(["serve", "--port", "9100"]) == 0
    run.assert_called_once_with(
        build.return_value, host="127.0.0.2", port=9100, log_level="warning"
    )
    assert build.call_args.args[0].cosmos_database == "D"
