"""Offline HTTP contract tests adapted from PR #150's server tests."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident
from unittest.mock import Mock
from urllib.parse import unquote

import pytest
from azure.cosmos import ContainerProxy
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cosmos_agentic_retriever import server
from cosmos_agentic_retriever.config import RetrieverSettings
from cosmos_agentic_retriever.orchestration import ContainerTarget
from cosmos_agentic_retriever.query_engine import (
    CorpusSchema,
    CosmosExecutor,
    QueryEngineConfig,
)
from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.server import SearchRequest, create_app


def _client(*, paths=None, text_fields=None, schema=None):
    container = Mock(spec=ContainerProxy)
    container.query_items.return_value = iter([])
    retriever = CorpusRetriever(
        container=container,
        schema=schema
        or CorpusSchema(
            item_id_path="/id",
            partition_key_paths=["/tenant"],
            text_paths=paths or ["/text"],
        ),
        executor=CosmosExecutor(config=QueryEngineConfig()),
    )
    return TestClient(
        create_app(
            _settings(
                cosmos_containers={
                    "C": {
                        "cosmos_schema": retriever.schema,
                        "search_text_fields": text_fields,
                    }
                }
            ),
            retrievers={"C": retriever},
        )
    ), container


def _settings(**options):
    return RetrieverSettings(
        **{
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
            **options,
        }
    )


@pytest.mark.parametrize("authentication", ["key", "azure_cli", "default"])
def test_lifespan_builds_once_and_closes_owned_resources(monkeypatch, authentication):
    credential = Mock()
    events = []
    credential.close.side_effect = lambda: events.append(
        ("credential_close", get_ident())
    )
    cli = Mock(return_value=credential)
    default = Mock(return_value=credential)
    monkeypatch.setattr(server, "AzureCliCredential", cli)
    monkeypatch.setattr(server, "DefaultAzureCredential", default)
    client = Mock()
    client.close.side_effect = lambda: events.append(("client_close", get_ident()))

    def make_client(*args, **kwargs):
        events.append(("build", get_ident()))
        return client

    constructor = Mock(side_effect=make_client)
    monkeypatch.setattr(server, "CosmosClient", constructor)
    container = (
        client.get_database_client.return_value.get_container_client.return_value
    )

    def query_items(**kwargs):
        events.append(("query", get_ident()))
        assert 'c["text"] AS txt_0' in kwargs["query"]
        return iter([])

    container.query_items.side_effect = query_items
    options = (
        {"cosmos_key": "test-key"}
        if authentication == "key"
        else {"cosmos_credential": authentication}
    )
    settings = _settings(query_engine={"max_concurrency": 2}, **options)
    app = create_app(settings)
    settings.cosmos_database = "changed-after-app-creation"
    settings.cosmos_containers["C"].cosmos_schema.text_paths.clear()
    constructor.assert_not_called()
    with TestClient(app) as http:
        loop_thread = http.portal.call(get_ident)
        engine = app.state.retriever
        assert (
            engine._retrievers[
                ContainerTarget("D", "C")
            ]._executor._config.max_concurrency
            == 2
        )
        for _ in range(2):
            assert http.post("/search", json={"query": "battery"}).status_code == 200
            assert app.state.retriever is engine
        client.close.assert_not_called()
        assert http.get("/health").status_code == 200
    constructor.assert_called_once_with(
        "https://example.documents.azure.com/",
        credential="test-key" if authentication == "key" else credential,
    )
    client.get_database_client.assert_called_once_with("D")
    client.get_database_client.return_value.get_container_client.assert_called_once_with(
        "C"
    )
    client.close.assert_called_once()
    if authentication == "key":
        cli.assert_not_called()
        default.assert_not_called()
        credential.close.assert_not_called()
    else:
        (cli if authentication == "azure_cli" else default).assert_called_once()
        (default if authentication == "azure_cli" else cli).assert_not_called()
        credential.close.assert_called_once()
    assert app.state.retriever is None
    assert [name for name, _ in events] == [
        "build",
        "query",
        "query",
        "client_close",
    ] + ([] if authentication == "key" else ["credential_close"])
    assert all(thread != loop_thread for _, thread in events)


@pytest.mark.parametrize(
    "stage", ["client", "container", "second_container", "shutdown"]
)
def test_startup_failure_closes_created_resources(monkeypatch, stage):
    credential, client = Mock(), Mock()
    monkeypatch.setattr(server, "AzureCliCredential", Mock(return_value=credential))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(server, "CosmosClient", constructor)
    if stage == "client":
        constructor.side_effect = RuntimeError("startup failed")
    elif stage == "container":
        client.get_database_client.side_effect = RuntimeError("startup failed")
    elif stage == "second_container":
        client.get_database_client.return_value.get_container_client.side_effect = [
            Mock(spec=ContainerProxy),
            RuntimeError("second container failed"),
        ]
    else:
        client.close.side_effect = RuntimeError("shutdown failed")
    settings = _settings()
    settings.cosmos_containers["second"] = settings.cosmos_containers["C"].model_copy(
        deep=True
    )
    app = create_app(settings)
    with (
        pytest.raises(RuntimeError, match="failed"),
        TestClient(app),
    ):
        assert stage == "shutdown"
    credential.close.assert_called_once()
    assert client.close.call_count == (0 if stage == "client" else 1)
    assert app.state.retriever is None


def test_injected_retriever_does_not_create_clients(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(server, "CosmosClient", constructor)
    credentials = Mock()
    monkeypatch.setattr(server, "AzureCliCredential", credentials)
    monkeypatch.setattr(server, "DefaultAzureCredential", credentials)
    load = Mock(
        side_effect=AssertionError("explicit settings must not load the environment")
    )
    monkeypatch.setattr(server, "get_settings", load)
    external_client = Mock()
    container = external_client.get_database_client.return_value.get_container_client.return_value
    container.query_items.return_value = iter([])
    settings = _settings()
    retriever = CorpusRetriever(
        container=container,
        schema=settings.cosmos_containers["C"].cosmos_schema,
        executor=CosmosExecutor(config=QueryEngineConfig()),
    )
    with TestClient(create_app(settings, retrievers={"C": retriever})) as client:
        assert (
            client.app.state.retriever._retrievers[ContainerTarget("D", "C")]
            is retriever
        )
        assert client.post("/search", json={"query": "battery"}).status_code == 200
    external_client.close.assert_not_called()
    constructor.assert_not_called()
    credentials.assert_not_called()
    load.assert_not_called()
    container.query_items.assert_called_once()


def test_injected_schema_mismatch_is_rejected_before_startup(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(server, "CosmosClient", constructor)
    retriever = CorpusRetriever(
        container=Mock(spec=ContainerProxy),
        schema=CorpusSchema(
            item_id_path="/id",
            partition_key_paths=["/tenant"],
            text_paths=["/different"],
        ),
        executor=CosmosExecutor(config=QueryEngineConfig()),
    )
    with pytest.raises(ValueError, match="schema and partition policy must match"):
        create_app(_settings(), retrievers={"C": retriever})
    constructor.assert_not_called()


def test_create_app_falls_back_to_get_settings(monkeypatch):
    settings = _settings()
    load = Mock(return_value=settings)
    monkeypatch.setattr(server, "get_settings", load)
    app = create_app()
    load.assert_called_once()
    assert app.state.retriever is None


def test_requests_before_startup_are_not_ready():
    client, container = _client()
    for _ in range(2):
        health = client.get("/health")
        assert health.status_code == 503 and health.json() == {"status": "unavailable"}
        response = client.post("/search", json={"query": "battery"})
        assert response.status_code == 503 and response.json() == {
            "error": "Service is not ready."
        }
        with client:
            assert client.get("/health").status_code == 200
    container.query_items.assert_not_called()


@pytest.mark.parametrize("overrides", [None, {}])
def test_empty_overrides_use_server_configuration(overrides):
    client, container = _client()
    with client:
        response = client.post(
            "/search",
            json={"query": "battery", "maxDocuments": 50, "overrides": overrides},
        )
    assert response.status_code == 200
    assert container.query_items.call_args.kwargs["parameters"] == [
        {"name": "@k0", "value": 50}
    ]


def test_search_request_defaults() -> None:
    request = SearchRequest(query="q")
    assert request.max_documents == 20
    assert request.database is None and request.container is None
    assert request.container_filters is None
    assert request.overrides is None


def test_search_request_alias_and_field_name() -> None:
    assert (
        SearchRequest.model_validate({"query": "q", "maxDocuments": 5}).max_documents
        == 5
    )
    assert SearchRequest(query="q", max_documents=7).max_documents == 7


@pytest.mark.parametrize("query", ["", "   ", "\n", "q" * 4097])
def test_search_request_query_min_length(query) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=query)


@pytest.mark.parametrize("max_documents", [1, 50])
def test_search_request_max_documents_bounds_ok(max_documents) -> None:
    assert (
        SearchRequest(query="q", max_documents=max_documents).max_documents
        == max_documents
    )


@pytest.mark.parametrize("value", [0, 51, True, 1.5, "5", None])
def test_search_request_max_documents_out_of_range(value) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="q", max_documents=value)


def test_search_happy_path_returns_result_dict() -> None:
    schema = CorpusSchema(
        item_id_path="/id",
        partition_key_paths=["/tenant"],
        text_paths=["/text"],
        document_id_path="/docid",
        chunk_id_path="/chunk",
        chunk_order_path="/position",
        title_path="/title",
        source_path="/source",
        metadata_paths={"year": "/year"},
    )
    client, container = _client(schema=schema)
    container.query_items.return_value = iter(
        [
            {
                "item_id": "b",
                "txt_0": "Battery recycling",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "chunk_order": 2,
                "title": "Title",
                "source": "Source",
                "md_0": 2024,
                "_cosmos_identity": {"id": "b", "partition_key": [0]},
            },
            {
                "item_id": "a",
                "txt_0": "Second",
                "_cosmos_identity": {"id": "a", "partition_key": [0]},
            },
        ]
    )
    with client:
        response = client.post(
            "/search",
            json={
                "query": "battery",
                "database": "D",
                "container": "C",
                "maxDocuments": 5,
            },
        )
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert [item["item_id"] for item in documents] == ["b", "a"]
    assert [item["rank"] for item in documents] == [0, 1]
    assert documents[0] == {
        "database": "D",
        "container": "C",
        "retrieval_id": "D/C:%5B%5B0%5D%2C%22b%22%5D",
        "cosmos_identity": {"id": "b", "partition_key": [0]},
        "item_id": "b",
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "chunk_order": 2,
        "title": "Title",
        "source": "Source",
        "text": "Battery recycling",
        "text_fields": {"/text": "Battery recycling"},
        "metadata": {"year": 2024},
        "rank": 0,
        "retrieval_strategy": "full_text",
        "retrieval_channels": ["full_text"],
    }
    assert documents[1]["document_id"] is None and documents[1]["metadata"] == {}
    container.query_items.assert_called_once_with(
        query='SELECT TOP @k0 c["id"] AS item_id, c["docid"] AS document_id, '
        'c["chunk"] AS chunk_id, c["position"] AS chunk_order, c["title"] AS title, '
        'c["source"] AS source, c["text"] AS txt_0, c["year"] AS md_0, '
        '{"id": c["id"], "partition_key": [IIF(IS_DEFINED(c["tenant"]), c["tenant"], {})]} AS _cosmos_identity FROM c '
        'ORDER BY RANK FullTextScore(c["text"], "battery")',
        parameters=[{"name": "@k0", "value": 5}],
        enable_cross_partition_query=True,
    )


def test_search_omitted_target_uses_configured_container() -> None:
    client, container = _client()
    with client:
        response = client.post("/search", json={"query": "battery"})
    assert response.status_code == 200
    assert response.json() == {
        "documents": [],
        "searched": [{"database": "D", "container": "C"}],
        "errors": [],
        "partial": False,
    }
    assert container.query_items.call_args.kwargs["parameters"] == [
        {"name": "@k0", "value": 20}
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"database": "other"},
        {"container": "other"},
        {"container": "*"},
        {"overrides": {"schema_override": {"item_id_path": "/key"}}},
        {"overrides": {"chat_model": "unused"}},
    ],
)
def test_unsupported_scope_and_overrides_rejected_before_query(options) -> None:
    client, container = _client()
    with client:
        response = client.post("/search", json={"query": "battery", **options})
    assert response.status_code == 400
    assert response.json()["error"]
    container.query_items.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"maxDocuments": 0},
        {"maxDocuments": True},
        {"maxDocuments": 51},
        {"maxDocuments": "5"},
        {"maxDocuments": None},
        {"query": " "},
        {"query": None},
        {"query": 123},
        {"query": "q" * 4097},
        {"database": ""},
        {"container": "c" * 257},
        {"mode": "vector"},
        {"overrides": "none"},
    ],
)
def test_bad_body_rejected_before_query(options) -> None:
    client, container = _client()
    with client:
        response = client.post("/search", json={"query": "battery", **options})
    assert response.status_code == 422
    container.query_items.assert_not_called()


def test_validation_errors_do_not_echo_input():
    client, container = _client()
    with client:
        response = client.post(
            "/search", json={"query": "battery", "api_key": "private-test-value"}
        )
    assert response.status_code == 422
    assert response.json() == {"error": "Invalid search request."}
    container.query_items.assert_not_called()


@pytest.mark.parametrize("body", ["{", "[]", "{}"])
def test_invalid_json_body_is_rejected(body):
    client, container = _client()
    with client:
        response = client.post(
            "/search", content=body, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 422
    assert response.json() == {"error": "Invalid search request."}
    container.query_items.assert_not_called()


def test_no_searchable_terms_rejected_before_query() -> None:
    client, container = _client()
    with client:
        response = client.post("/search", json={"query": "!!!"})
    assert response.status_code == 400
    assert response.json()["type"] == "QueryCompilationError"
    container.query_items.assert_not_called()


@pytest.mark.parametrize("failure", ["query", "iteration", "mapping", "serialization"])
def test_search_engine_exception_returns_500(failure) -> None:
    client, container = _client(
        schema=CorpusSchema(
            item_id_path="/id",
            partition_key_paths=["/tenant"],
            text_paths=["/text"],
            metadata_paths={"value": "/value"},
        )
    )
    if failure == "query":
        container.query_items.side_effect = RuntimeError("private endpoint details")
    elif failure == "iteration":

        def broken_rows():
            yield {"item_id": "partial", "txt_0": "partial result"}
            raise RuntimeError("private endpoint details")

        container.query_items.return_value = broken_rows()
    else:
        container.query_items.return_value = iter(
            [
                {
                    "item_id": None if failure == "mapping" else "a",
                    "md_0": object(),
                    "_cosmos_identity": {"id": "a", "partition_key": [0]},
                },
            ]
        )
    with client:
        response = client.post("/search", json={"query": "battery"})
    assert response.status_code == 500
    if failure == "serialization":
        assert response.json() == {"error": "Search failed."}
    else:
        assert response.json() == {
            "error": "Search failed for all selected containers.",
            "documents": [],
            "searched": [],
            "partial": False,
            "errors": [{"database": "D", "container": "C", "error": "Search failed."}],
        }


def test_health() -> None:
    client, container = _client()
    with client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/config").status_code == 404
        assert client.patch("/config", json={}).status_code == 404
    container.query_items.assert_not_called()


def test_configured_text_fields_are_forwarded() -> None:
    client, container = _client(paths=["/title", "/body"], text_fields=["/body"])
    with client:
        response = client.post("/search", json={"query": "battery"})
    assert response.status_code == 200
    assert container.query_items.call_args.kwargs["query"].endswith(
        'ORDER BY RANK FullTextScore(c["body"], "battery")'
    )


def _multi_app(monkeypatch, *, capacity=2, partition_key=0):
    configs = {
        "A": {
            "cosmos_schema": {
                "item_id_path": "/id",
                "partition_key_paths": ["/tenant"],
                "text_paths": ["/text"],
            }
        },
        "B": {
            "cosmos_schema": {
                "item_id_path": "/key",
                "partition_key_paths": ["/tenant"],
                "text_paths": ["/content/body"],
            },
            "partition_key": partition_key,
            "partition_policy": {"allow_cross_partition_search": False},
        },
    }
    client = Mock()
    containers = {name: Mock(spec=ContainerProxy) for name in configs}
    for name, container in containers.items():
        container.query_items.side_effect = lambda name=name, **kwargs: iter(
            [
                {
                    "item_id": "same",
                    "txt_0": f"{name} first",
                    "_cosmos_identity": {
                        "id": "same",
                        "partition_key": [0 if name == "A" else partition_key],
                    },
                },
                {
                    "item_id": "next",
                    "txt_0": f"{name} next",
                    "_cosmos_identity": {
                        "id": "next",
                        "partition_key": [0 if name == "A" else partition_key],
                    },
                },
            ]
        )
    client.get_database_client.return_value.get_container_client.side_effect = (
        containers.__getitem__
    )
    monkeypatch.setattr(server, "CosmosClient", Mock(return_value=client))
    settings = _settings(
        cosmos_containers=configs,
        cosmos_key="test-key",
        query_engine={"max_concurrency": capacity},
    )
    return create_app(settings), containers, client


@pytest.mark.parametrize("selected", [None, "A", "B"])
def test_container_filters_use_each_targets_stored_paths(monkeypatch, selected):
    app, containers, _ = _multi_app(monkeypatch)
    paths = {"A": "/publication/year", "B": "/publishedYear"}
    names = list(paths) if selected is None else [selected]
    options = {} if selected is None else {"container": selected}
    body = {
        "query": "battery",
        "maxDocuments": 3,
        "container_filters": {
            name: [{"kind": "range", "path": paths[name], "minimum": 2020}]
            for name in names
        },
        **options,
    }
    with TestClient(app) as http:
        response = http.post("/search", json=body)
        assert response.status_code == 200
        assert response.json()["partial"] is False
        assert len(response.json()["documents"]) <= 3
        assert {target["container"] for target in response.json()["searched"]} == set(
            names
        )
        for name, container in containers.items():
            if name not in names:
                container.query_items.assert_not_called()
                continue
            arguments = container.query_items.call_args.kwargs
            expression = (
                'c["publication"]["year"]' if name == "A" else 'c["publishedYear"]'
            )
            assert f"WHERE ({expression} >= @p1)" in arguments["query"]
            assert expression not in arguments["query"].split(" FROM c")[0]
            assert arguments["parameters"] == [
                {"name": "@k0", "value": 3},
                {"name": "@p1", "value": 2020},
            ]
            if name == "B":
                assert arguments["partition_key"] == 0
            container.query_items.reset_mock()
        assert all(item["metadata"] == {} for item in response.json()["documents"])
        assert (
            http.post("/search", json={"query": "battery", **options}).status_code
            == 200
        )
        for name in names:
            assert (
                " WHERE " not in containers[name].query_items.call_args.kwargs["query"]
            )


def test_explicit_empty_filters_and_parameterized_values(monkeypatch):
    app, containers, _ = _multi_app(monkeypatch)
    value = "SciFact' OR true --"
    with TestClient(app) as http:
        response = http.post(
            "/search",
            json={
                "query": "battery",
                "container_filters": {
                    "A": [{"kind": "equals", "path": "/source", "value": value}],
                    "B": [],
                },
            },
        )
    assert response.status_code == 200
    arguments = containers["A"].query_items.call_args.kwargs
    assert 'WHERE c["source"] = @p1' in arguments["query"]
    assert value not in arguments["query"]
    assert arguments["parameters"][-1] == {"name": "@p1", "value": value}
    assert " WHERE " not in containers["B"].query_items.call_args.kwargs["query"]


@pytest.mark.parametrize(
    "options,status",
    [
        ({"container_filters": {}}, 400),
        ({"container_filters": {"A": []}}, 400),
        ({"container_filters": {"A": [], "B": [], "unknown": []}}, 400),
        ({"container": "A", "container_filters": {"A": [], "B": []}}, 400),
        ({"container_filters": {"A": [], "unknown": []}}, 400),
        (
            {
                "container_filters": {
                    "A": [],
                    "B": [{"kind": "equals", "path": "year", "value": 2020}],
                }
            },
            422,
        ),
        (
            {
                "container_filters": {
                    "A": [],
                    "B": [{"kind": "equals", "logical_field": "year", "value": 2020}],
                }
            },
            422,
        ),
        (
            {
                "container_filters": {
                    "A": [],
                    "B": [{"kind": "sql", "path": "/year", "value": 2020}],
                }
            },
            422,
        ),
        (
            {
                "container_filters": {
                    "A": [],
                    "B": [
                        {"kind": "equals", "path": "/year", "value": 2020, "typo": True}
                    ],
                }
            },
            422,
        ),
        ({"container_filters": {"A": [], "B": None}}, 422),
        (
            {"container_filters": {"A": [], "B": [{"kind": "range", "path": "/year"}]}},
            422,
        ),
    ],
)
def test_invalid_container_filters_fail_before_any_query(monkeypatch, options, status):
    app, containers, _ = _multi_app(monkeypatch)
    with TestClient(app) as http:
        response = http.post("/search", json={"query": "battery", **options})
    assert response.status_code == status
    for container in containers.values():
        container.query_items.assert_not_called()


@pytest.mark.parametrize("partition_key", [0, "", "tenant-b"])
@pytest.mark.parametrize("selected", [None, "A", "B"])
def test_multi_container_scope_schema_partition_and_results(
    monkeypatch, selected, partition_key
):
    app, containers, owner = _multi_app(monkeypatch, partition_key=partition_key)
    options = {} if selected is None else {"container": selected}
    with TestClient(app) as http:
        engines = list(app.state.retriever._retrievers.values())
        assert engines[0]._executor is engines[1]._executor
        response = http.post(
            "/search", json={"query": "battery", "maxDocuments": 3, **options}
        )
    assert response.status_code == 200
    body = response.json()
    names = ["A", "B"] if selected is None else [selected]
    assert body["searched"] == [{"database": "D", "container": name} for name in names]
    assert body["partial"] is False and body["errors"] == []
    expected = (
        [("A", "same"), ("B", "same"), ("A", "next")]
        if selected is None
        else [(selected, "same"), (selected, "next")]
    )
    assert [
        (item["container"], item["item_id"]) for item in body["documents"]
    ] == expected
    assert len({item["retrieval_id"] for item in body["documents"]}) == len(expected)
    for item in body["documents"]:
        prefix, encoded = item["retrieval_id"].split(":")
        assert prefix == f"D/{item['container']}"
        assert json.loads(unquote(encoded)) == [
            [0 if item["container"] == "A" else partition_key],
            item["item_id"],
        ]
    assert [item["rank"] for item in body["documents"]] == list(range(len(expected)))
    for name, container in containers.items():
        if name not in names:
            container.query_items.assert_not_called()
            continue
        arguments = container.query_items.call_args.kwargs
        assert arguments["parameters"] == [{"name": "@k0", "value": 3}]
        path = "/text" if name == "A" else "/content/body"
        expression = 'c["text"]' if name == "A" else 'c["content"]["body"]'
        assert f'FullTextScore({expression}, "battery")' in arguments["query"]
        if name == "A":
            assert arguments["enable_cross_partition_query"] is True
            assert "partition_key" not in arguments
        else:
            assert arguments["partition_key"] == partition_key
            assert "enable_cross_partition_query" not in arguments
        for item in body["documents"]:
            if item["container"] == name:
                assert list(item["text_fields"]) == [path]
    owner.close.assert_called_once()


@pytest.mark.parametrize("failed", [[], ["A"], ["B"], ["A", "B"]])
def test_multi_container_failures_are_not_silent(monkeypatch, failed):
    app, containers, _ = _multi_app(monkeypatch)
    for name, container in containers.items():
        if name in failed:
            container.query_items.side_effect = RuntimeError("private-endpoint")
        else:
            container.query_items.side_effect = lambda **kwargs: iter([])
    with TestClient(app) as http:
        response = http.post("/search", json={"query": "battery"})
    body = response.json()
    assert response.status_code == (500 if len(failed) == 2 else 200)
    assert body["documents"] == []
    assert body["partial"] == (len(failed) == 1)
    assert body["errors"] == [
        {"database": "D", "container": name, "error": "Search failed."}
        for name in failed
    ]
    assert body["searched"] == [
        {"database": "D", "container": name}
        for name in containers
        if name not in failed
    ]
    assert ("error" in body) == (len(failed) == 2)
    assert "private-endpoint" not in response.text


def test_invalid_multi_target_request_runs_no_queries(monkeypatch):
    app, containers, _ = _multi_app(monkeypatch)
    with TestClient(app) as http:
        for options in (
            {"container": "unknown"},
            {"database": "other"},
            {"overrides": {"schema_override": {}}},
            {"query": "!!!"},
        ):
            assert (
                http.post("/search", json={"query": "battery", **options}).status_code
                == 400
            )
    for container in containers.values():
        container.query_items.assert_not_called()


@pytest.mark.parametrize("capacity", [1, 2])
def test_concurrent_http_requests_share_query_budget(monkeypatch, capacity):
    app, containers, _ = _multi_app(monkeypatch, capacity=capacity)
    reached_capacity, release = Event(), Event()
    lock = Lock()
    active = maximum = calls = 0

    def rows(**kwargs):
        nonlocal active, maximum, calls
        with lock:
            active += 1
            calls += 1
            maximum = max(maximum, active)
            if active == capacity:
                reached_capacity.set()
        try:
            assert release.wait(5), "query was not released"
            yield {
                "item_id": "same",
                "txt_0": "result",
                "_cosmos_identity": {"id": "same", "partition_key": [0]},
            }
        finally:
            with lock:
                active -= 1

    for container in containers.values():
        container.query_items.side_effect = rows
    with TestClient(app) as http, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(http.post, "/search", json={"query": "battery"})
            for _ in range(2)
        ]
        try:
            assert reached_capacity.wait(5), "fan-out never reached configured capacity"
            assert http.get("/health").status_code == 200
        finally:
            release.set()
        assert all(future.result(timeout=5).status_code == 200 for future in futures)
    assert calls == 4 and active == 0 and maximum == capacity


def test_cross_partition_identity_survives_http_pipeline():
    schema = CorpusSchema(
        item_id_path="/record/id",
        partition_key_paths=["/tenant"],
        text_paths=["/text"],
        metadata_paths={"tenant": "/tenant"},
    )
    client, container = _client(schema=schema)
    container.query_items.side_effect = lambda **kwargs: iter(
        [
            {
                "item_id": "logical",
                "txt_0": text,
                "md_0": tenant,
                "_cosmos_identity": {"id": physical, "partition_key": [tenant]},
            }
            for physical, tenant, text in [
                ("same", "A", "first"),
                ("same", "B", "second"),
                ("other", "A", "third"),
                ("same", "A", "first"),
            ]
        ]
    )
    with client:
        all_results = client.post(
            "/search", json={"query": "battery", "maxDocuments": 5}
        ).json()
        one_target = client.post(
            "/search", json={"query": "battery", "container": "C", "maxDocuments": 5}
        ).json()
    assert all_results["errors"] == [] and all_results["partial"] is False
    assert all_results["documents"] == one_target["documents"]
    items = all_results["documents"]
    assert len(items) == 3 and {item["text"] for item in items} == {
        "first",
        "second",
        "third",
    }
    assert all(item["item_id"] == "logical" for item in items)
    assert len({item["retrieval_id"] for item in items}) == 3
    assert [item["rank"] for item in items] == [0, 1, 2]
    sql = container.query_items.call_args.kwargs["query"]
    assert 'c["record"]["id"] AS item_id' in sql
    assert (
        '{"id": c["id"], "partition_key": [IIF(IS_DEFINED(c["tenant"]), c["tenant"], {})]}'
        in sql
    )


def test_missing_physical_identity_from_backend_is_an_error():
    client, container = _client()
    container.query_items.return_value = iter([{"item_id": "logical", "txt_0": "text"}])
    with client:
        response = client.post("/search", json={"query": "battery"})
    assert response.status_code == 500
    assert response.json()["documents"] == []
