"""Offline safety checks for the live-test setup tool; never contacts Azure."""

import hashlib
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import setup_cosmos_live_tests as setup
from setup_cosmos_live_tests import (
    fixtures,
    load_scifact,
    select_data,
    validate_container,
    validate_database_name,
    validate_items,
)


@pytest.fixture
def scifact_archive(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "scifact/corpus.jsonl",
            "\n".join(
                json.dumps({"_id": str(n), "title": f"Title {n}", "text": f"Text {n}"})
                for n in range(1, 111)
            ),
        )
        output.writestr(
            "scifact/queries.jsonl",
            "\n".join(json.dumps({"_id": str(n), "text": f"Query {n}"}) for n in range(1, 7)),
        )
        output.writestr(
            "scifact/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\n"
            + "\n".join(f"{n}\t{100 + n}\t1" for n in range(1, 7)),
        )
    payload = archive.getvalue()
    path = tmp_path / "scifact.zip"
    path.write_bytes(payload)
    monkeypatch.setattr(setup, "SCIFACT_SHA256", hashlib.sha256(payload).hexdigest())
    return path


def test_scifact_selection_is_repeatable_and_bounded(scifact_archive):
    real = load_scifact(scifact_archive)
    assert real == load_scifact(scifact_archive)
    assert len(real.fixture.items) == 100
    assert [query.query_id for query in real.queries] == ["1", "2", "3", "4", "5"]
    assert len(real.selection_sha256) == 64


def test_bad_archive_checksum_is_rejected(scifact_archive):
    scifact_archive.write_bytes(b"tampered archive")
    with pytest.raises(ValueError):
        load_scifact(scifact_archive)


def test_bad_archive_aborts_before_any_azure_call(scifact_archive, monkeypatch):
    execute = Mock()
    monkeypatch.setattr(setup, "az_json", execute)
    scifact_archive.write_bytes(b"tampered archive")
    assert (
        setup.main(
            [
                "--subscription", "sub",
                "--resource-group", "rg",
                "--account", "account",
                "--data", "both",
                "--scifact-archive", str(scifact_archive),
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
        select_data(data, archive)


@pytest.mark.parametrize("database", ["prod-db", "mcp-live-tests-bad_name", "other"])
def test_database_name_guard_rejects_foreign_namespaces(database):
    with pytest.raises(ValueError):
        validate_database_name(database)


def test_incompatible_existing_container_is_rejected():
    fixture = fixtures()[0]
    properties = {
        "partitionKey": {"paths": ["/wrong"], "kind": "Hash"},
        "fullTextPolicy": fixture.full_text_policy(),
        "indexingPolicy": fixture.indexing_policy(),
    }
    with pytest.raises(ValueError):
        validate_container(properties, fixture)


def test_validate_items_rejects_tampering_and_reports_missing():
    fixture = fixtures()[0]
    tampered = deepcopy(fixture.items[0])
    tampered["text"] = "changed"
    with pytest.raises(ValueError):
        validate_items([tampered], fixture)
    assert validate_items([], fixture, allow_missing=True) == list(fixture.items)
