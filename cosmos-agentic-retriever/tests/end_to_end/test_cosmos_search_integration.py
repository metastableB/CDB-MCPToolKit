"""Read-only synthetic, real-corpus, and combined-container search checks.

Opt in with RUN_COSMOS_LIVE=1 after running tools/setup_cosmos_live_tests.py.
Tests adapt the full-text, field, limit, and cross-collection assertions from
PR #150 to versioned test data, without embeddings, discovery, or an LLM.
HTTP uses FastAPI's in-process test transport, not the .NET/MCP network path.
"""

import os
import time

import pytest
from cosmos_live_corpus import QUERY_COUNT
from cosmos_live_fixtures import fixtures
from fastapi.testclient import TestClient

from cosmos_agentic_retriever.server import create_app

pytestmark = [
    pytest.mark.cosmos_live,
    pytest.mark.skipif(
        os.environ.get("RUN_COSMOS_LIVE") != "1",
        reason="set RUN_COSMOS_LIVE=1 to use real Cosmos",
    ),
]


def _search(client, *, query="battery recycling", container=None, limit=50):
    body = {"query": query, "maxDocuments": limit}
    if container is not None:
        body["container"] = container
    response = client.post("/search", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["errors"] == [] and result["partial"] is False, result
    return result


def _ready(operation, predicate, *, timeout=60):
    """Retry only successful queries awaiting expected indexing/visibility results."""
    deadline = time.monotonic() + timeout
    while True:
        result = operation()
        if predicate(result):
            return result
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"Fixture results not ready after {timeout}s: {result}"
        time.sleep(min(2, remaining))


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda fixture: fixture.name)
def test_full_text_returns_fixture_ids_text_and_schema(
    synthetic_http, synthetic_settings, fixture
):
    expected = {"shared", "first-only", "other-only"}
    result = _ready(
        lambda: _search(synthetic_http, container=fixture.name),
        lambda result: {item["item_id"] for item in result["documents"]} == expected,
    )
    assert result["searched"] == [
        {"database": synthetic_settings.cosmos_database, "container": fixture.name}
    ]
    for item in result["documents"]:
        physical_id = (
            f"physical-{item['item_id']}"
            if fixture.name == "nested-v1"
            else item["item_id"]
        )
        stored = next(record for record in fixture.items if record["id"] == physical_id)
        expected_fields = {}
        for path in fixture.schema["text_paths"]:
            value = stored
            for segment in path.strip("/").split("/"):
                value = value[segment]
            expected_fields[path] = value
        assert item["text_fields"] == expected_fields
        assert item["text"] == expected_fields[fixture.search_fields[0]]
        assert set(item["text_fields"]) == set(fixture.schema["text_paths"])
        assert item["database"] == synthetic_settings.cosmos_database
        assert item["container"] == fixture.name
        assert item["retrieval_strategy"] == "full_text"
    if fixture.name == "nested-v1":
        assert all(item["metadata"] == {"year": 2025} for item in result["documents"])


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_cross_collection_identity_and_global_limit(synthetic_http, limit):
    result = _ready(
        lambda: _search(synthetic_http, limit=limit),
        lambda result: len(result["documents"]) == limit,
    )
    assert len(result["searched"]) == 3
    assert [item["rank"] for item in result["documents"]] == list(range(limit))
    assert len({item["retrieval_id"] for item in result["documents"]}) == limit


def test_same_id_survives_in_all_containers(synthetic_http):
    result = _ready(
        lambda: _search(synthetic_http), lambda result: len(result["documents"]) == 9
    )
    shared = [item for item in result["documents"] if item["item_id"] == "shared"]
    assert len(shared) == len({item["retrieval_id"] for item in shared}) == 3
    assert {item["container"] for item in shared} == {
        fixture.name for fixture in fixtures()
    }


@pytest.mark.parametrize(
    "key, expected",
    [
        (0, {"shared", "first-only"}),
        ("other", {"other-only"}),
        ("absent-partition", set()),
    ],
)
def test_partition_scope_is_independent_per_container(
    synthetic_settings, key, expected
):
    settings = synthetic_settings.model_copy(deep=True)
    target = settings.cosmos_containers["flat-v1"]
    target.partition_key = key
    target.partition_policy.allow_cross_partition_search = False
    with TestClient(create_app(settings)) as client:
        result = _ready(
            lambda: _search(client),
            lambda result: (
                {
                    item["item_id"]
                    for item in result["documents"]
                    if item["container"] == "flat-v1"
                }
                == expected
                and sum(
                    item["container"] == "nested-v1" for item in result["documents"]
                )
                == 3
            ),
        )
    assert {
        item["item_id"]
        for item in result["documents"]
        if item["container"] == "flat-v1"
    } == expected


@pytest.mark.parametrize(
    "field, expected", [("/body", {"shared"}), ("/title", {"first-only", "other-only"})]
)
def test_selected_text_field_changes_top_match(synthetic_settings, field, expected):
    settings = synthetic_settings.model_copy(deep=True)
    settings.cosmos_containers["fields-v1"].search_text_fields = [field]
    with TestClient(create_app(settings)) as client:
        result = _ready(
            lambda: _search(client, query="canarybody", container="fields-v1", limit=1),
            lambda result: (
                bool(result["documents"])
                and result["documents"][0]["item_id"] in expected
            ),
        )
    item = result["documents"][0]
    assert item["text"] == item["text_fields"][field]


@pytest.mark.parametrize(
    "options",
    [
        {"container": "not-configured"},
        {"overrides": {"schema_override": {"item_id_path": "/other"}}},
    ],
)
def test_request_scope_and_schema_overrides_are_rejected(live_http, options):
    response = live_http.post("/search", json={"query": "battery", **options})
    assert response.status_code == 400 and response.json()["error"]


@pytest.mark.parametrize("query_index", range(QUERY_COUNT))
def test_real_corpus_query_returns_source_documents(
    live_http, live_settings, real_corpus, query_index, record_property
):
    query = real_corpus.queries[query_index]
    fixture = real_corpus.fixture
    result = _ready(
        lambda: _search(live_http, query=query.text, container=fixture.name, limit=5),
        lambda result: bool(result["documents"]),
    )
    assert result["searched"] == [
        {"database": live_settings.cosmos_database, "container": fixture.name}
    ]
    documents = result["documents"]
    assert 1 <= len(documents) <= 5
    assert [item["rank"] for item in documents] == list(range(len(documents)))
    records = {item["id"]: item for item in fixture.items}
    for item in documents:
        stored = records[item["item_id"]]
        assert item["text_fields"] == {
            "/title": stored["title"],
            "/text": stored["text"],
        }
        assert item["text"] and item["retrieval_strategy"] == "full_text"
        assert item["container"] == fixture.name
        assert item["metadata"] == {"source": "beir-scifact"}
    returned = {item["item_id"] for item in documents}
    recall = len(returned & query.relevant_ids) / len(query.relevant_ids)
    record_property("query_id", query.query_id)
    record_property("recall_at_5", recall)
    record_property("returned_ids", ",".join(item["item_id"] for item in documents))
    record_property("selection_sha256", real_corpus.selection_sha256)


@pytest.mark.parametrize("limit", [2, 5])
def test_synthetic_and_real_containers_are_searched_together(
    live_http, live_settings, real_corpus, limit
):
    synthetic_names = {fixture.name for fixture in fixtures()}
    if not synthetic_names <= live_settings.cosmos_containers.keys():
        pytest.skip("select both to test mixed synthetic and real containers")
    result = _ready(
        lambda: _search(live_http, query=real_corpus.queries[0].text, limit=limit),
        lambda result: len(result["documents"]) == limit,
    )
    names = {entry["container"] for entry in result["searched"]}
    assert names == set(live_settings.cosmos_containers)
    assert [item["rank"] for item in result["documents"]] == list(range(limit))
    assert len({item["retrieval_id"] for item in result["documents"]}) == limit
    assert all(item["container"] in names for item in result["documents"])
    if limit == 5:
        assert {item["container"] for item in result["documents"]} == names
