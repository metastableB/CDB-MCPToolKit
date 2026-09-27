"""Check the MCP search request and response contract without cloud calls."""

from __future__ import annotations

import json
from unittest.mock import Mock

import httpx
import pytest
from azure.cosmos import ContainerProxy
from fastapi.testclient import TestClient
from openai import OpenAI

from cosmos_agentic_retriever.agent import chat_client
from cosmos_agentic_retriever.agent.chat_client import ChatClient
from cosmos_agentic_retriever.agent.loop import LlmTurn, ToolCall
from cosmos_agentic_retriever.config import RetrieverConfig
from cosmos_agentic_retriever.orchestration import (
    ContainerItem,
    ContainerTarget,
    MultiSearchResult,
)
from cosmos_agentic_retriever.query_engine import (
    CosmosExecutor,
    QueryEngineConfig,
)
from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.server import create_app


class _FakeChat:
    """A stand-in chat client returning scripted turns for the loop."""

    def __init__(self, turns: list[LlmTurn]) -> None:
        self._turns = list(turns)

    def complete(self, messages, tools) -> LlmTurn:
        return self._turns.pop(0)


class _FakeRetriever:
    """A stand-in retriever returning fixed items for any search."""

    def __init__(self, items: list[ContainerItem]) -> None:
        self._items = items

    def search(self, requests, *, limit) -> MultiSearchResult:
        return MultiSearchResult(items=list(self._items[:limit]), searched=list(requests))


def _document(retrieval_id: str, rank: int, text: str = "") -> ContainerItem:
    return ContainerItem(
        item_id=retrieval_id,
        rank=rank,
        text=text,
        database="D",
        container="A",
        retrieval_id=retrieval_id,
    )


def _settings(**options) -> RetrieverConfig:
    return RetrieverConfig(
        account_uri="https://example.documents.azure.com",
        cosmos_database="D",
        llm_base_url="https://model.example.com/v1",
        llm_model="test-model",
        cosmos_containers={
            "C": {
                "cosmos_schema": {
                    "item_id_path": "/id",
                    "partition_key_paths": ["/tenant"],
                    "text_paths": ["/text"],
                }
            }
        },
        **options,
    )


def _app(settings: RetrieverConfig):
    retrievers = {}
    for name, config in settings.cosmos_containers.items():
        container = Mock(spec=ContainerProxy)
        container.query_items.side_effect = lambda **kwargs: iter([])
        retrievers[name] = CorpusRetriever(
            container=container,
            schema=config.cosmos_schema,
            partition_policy=config.partition_policy,
            executor=CosmosExecutor(config=QueryEngineConfig()),
        )
    return create_app(settings, retrievers=retrievers)


def _search_then_answer(query="hi", call_id="c1") -> list[LlmTurn]:
    return [
        LlmTurn(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "full_text_search",
                            "arguments": json.dumps({"query": query}),
                        },
                    }
                ],
            },
            content="",
            tool_calls=[ToolCall(call_id, "full_text_search", {"query": query})],
        ),
        LlmTurn(message={"role": "assistant", "content": "the answer"}, content="the answer"),
    ]


def test_agent_search_runs_loop_and_returns_answer():
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat(_search_then_answer())
        response = http.post("/agentic_search", json={"query": "what is up?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "the answer",
        "documents": [],
        "terminal_reason": "stop",
        "turns": 2,
        "searched": [{"database": "D", "container": "C"}],
        "errors": [],
        "partial": False,
    }


def test_agent_search_returns_supporting_documents():
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat(_search_then_answer())
        http.app.state.retriever = _FakeRetriever(
            [_document("d1", 0, "recycling one"), _document("d2", 1, "recycling two")]
        )
        response = http.post("/agentic_search", json={"query": "recycling?"})
    body = response.json()
    assert body["answer"] == "the answer"
    assert [doc["retrieval_id"] for doc in body["documents"]] == ["d1", "d2"]
    assert body["documents"][0]["text"] == "recycling one"


def test_agent_search_blank_query_rejected():
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat([])
        response = http.post("/agentic_search", json={"query": "   "})
    assert response.status_code == 422


def test_configured_llm_builds_chat_client():
    with TestClient(_app(_settings())) as http:
        assert isinstance(http.app.state.chat_client, ChatClient)


@pytest.mark.parametrize("container", [None, "C"])
@pytest.mark.parametrize("limit, service_cap", [(1, 10), (50, 1), (8, 10)])
def test_mcp_request_runs_agent_with_scope_filters_and_result_limit(
    container, limit, service_cap
):
    settings = _settings(agent_max_documents=service_cap)
    settings.cosmos_containers["C"].partition_key = "team-a"
    settings.cosmos_containers["other"] = settings.cosmos_containers["C"].model_copy(
        deep=True
    )
    names = [container] if container else list(settings.cosmos_containers)
    targets = [ContainerTarget("D", name) for name in names]
    retriever = Mock()
    retriever.search.return_value = MultiSearchResult(
        items=[_document(f"d{rank}", rank) for rank in range(10)], searched=targets
    )
    chat = _FakeChat([
        _search_then_answer()[0], *_search_then_answer("battery recycling", "c2")
    ])
    filters = [{"kind": "equals", "path": "/tenant", "value": "team-a"}]
    with TestClient(_app(settings)) as http:
        http.app.state.chat_client = chat
        http.app.state.retriever = retriever
        response = http.post(
            "/agentic_search",
            json={
                "query": "Find documents about recycling.",
                "maxDocuments": limit,
                "database": "D",
                "container": container,
                "container_filters": {name: filters for name in names},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "the answer"
    assert body["turns"] == 3 and body["terminal_reason"] == "stop"
    assert len(body["documents"]) == min(limit, service_cap)
    assert body["searched"] == [target._asdict() for target in targets]
    assert retriever.search.call_count == 2
    for call, query in zip(retriever.search.call_args_list, ["hi", "battery recycling"]):
        requests = call.args[0]
        assert list(requests) == targets
        assert call.kwargs["limit"] == min(limit, service_cap)
        for target in targets:
            assert requests[target].query == query
            assert requests[target].partition_key == "team-a"
            assert requests[target].limit == min(limit, service_cap)
            assert requests[target].filters[0].model_dump(mode="json") == filters[0]


def test_failed_model_request_returns_bad_gateway():
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat([
            LlmTurn(
                message={"role": "assistant", "content": ""},
                error="Endpoint failed: private-key private-url",
            )
        ])
        response = http.post("/agentic_search", json={"query": "Find recycling documents."})

    assert response.status_code == 502, response.text
    body = response.json()
    assert body["error"] == "Model request failed."
    assert body["terminal_reason"] == "error"
    assert body["answer"] == "" and body["turns"] == 1
    assert "private-" not in response.text


def test_model_failure_after_search_keeps_documents_but_not_an_unfinished_answer():
    search_turn = _search_then_answer()[0]
    search_turn.message["content"] = "I will search before answering."
    failed_turn = LlmTurn(
        message={"role": "assistant", "content": ""}, error="private endpoint failure"
    )
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat([search_turn, failed_turn])
        http.app.state.retriever = _FakeRetriever([_document("d1", 0, "Evidence.")])
        response = http.post("/agentic_search", json={"query": "Find recycling documents."})

    assert response.status_code == 502, response.text
    body = response.json()
    assert body["answer"] == ""
    assert body["turns"] == 2 and body["terminal_reason"] == "error"
    assert [document["retrieval_id"] for document in body["documents"]] == ["d1"]
    assert "private" not in response.text


@pytest.mark.parametrize("partial", [False, True])
def test_agent_preserves_partial_and_complete_retrieval_failures(partial):
    settings = _settings()
    settings.cosmos_containers["other"] = settings.cosmos_containers["C"].model_copy(
        deep=True
    )
    target, other = ContainerTarget("D", "C"), ContainerTarget("D", "other")
    errors = {other: "Search failed."}
    if not partial:
        errors[target] = "Search failed."
    retriever = Mock()
    retriever.search.return_value = MultiSearchResult(
        items=[_document("d1", 0)] if partial else [],
        searched=[target] if partial else [],
        errors=errors,
    )
    with TestClient(_app(settings)) as http:
        http.app.state.chat_client = _FakeChat(_search_then_answer())
        http.app.state.retriever = retriever
        response = http.post("/agentic_search", json={"query": "Find recycling documents."})

    assert response.status_code == (200 if partial else 500), response.text
    body = response.json()
    assert body["partial"] is partial
    assert len(body["errors"]) == (1 if partial else 2)
    assert body["searched"] == ([target._asdict()] if partial else [])
    if partial:
        assert body["answer"] == "the answer"
        assert len(body["documents"]) == 1
    else:
        assert body["answer"] == "" and body["documents"] == []
        assert body["error"] == "Search failed for all selected containers."


@pytest.mark.parametrize(
    "options, status",
    [
        ({"database": "unknown"}, 400),
        ({"container": "unknown"}, 400),
        ({"container_filters": {}}, 400),
        ({"overrides": {"schema_override": {"text_paths": ["/other"]}}}, 400),
        ({"maxDocuments": 0}, 422),
        ({"maxDocuments": True}, 422),
    ],
)
def test_invalid_mcp_request_never_calls_model_or_retriever(options, status):
    chat, retriever = Mock(), Mock()
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = chat
        http.app.state.retriever = retriever
        response = http.post("/agentic_search", json={"query": "recycling", **options})

    assert response.status_code == status, response.text
    chat.complete.assert_not_called()
    retriever.search.assert_not_called()


def test_exhausted_sdk_retries_return_bad_gateway(monkeypatch):
    requests = []

    def unavailable(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "private endpoint failure"}})

    delay = Mock()
    monkeypatch.setattr("openai._base_client.time.sleep", delay)
    with httpx.Client(transport=httpx.MockTransport(unavailable)) as transport:
        monkeypatch.setattr(
            chat_client, "OpenAI", lambda **options: OpenAI(http_client=transport, **options)
        )
        with TestClient(_app(_settings())) as http:
            response = http.post("/agentic_search", json={"query": "Find recycling documents."})

    assert len(requests) == 3
    assert delay.call_count == 2
    assert response.status_code == 502, response.text
    assert response.json()["error"] == "Model request failed."
    assert response.json()["turns"] == 1
    assert "private" not in response.text


@pytest.mark.parametrize("path", ["/search", "/agent_search"])
def test_removed_agent_routes_return_not_found(path):
    chat = Mock()
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = chat
        response = http.post(path, json={"query": "Find recycling documents."})

    assert response.status_code == 404
    chat.complete.assert_not_called()
