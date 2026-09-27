"""HTTP contract tests for the /agent_search endpoint."""

from __future__ import annotations

from unittest.mock import Mock

from azure.cosmos import ContainerProxy
from fastapi.testclient import TestClient

from cosmos_agentic_retriever.agent.chat_client import ChatClient
from cosmos_agentic_retriever.agent.loop import LlmTurn, ToolCall
from cosmos_agentic_retriever.config import RetrieverConfig
from cosmos_agentic_retriever.orchestration import ContainerItem, MultiSearchResult
from cosmos_agentic_retriever.query_engine import (
    CorpusSchema,
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
        return MultiSearchResult(items=list(self._items[:limit]))


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
    container = Mock(spec=ContainerProxy)
    container.query_items.return_value = iter([])
    retriever = CorpusRetriever(
        container=container,
        schema=CorpusSchema(
            item_id_path="/id",
            partition_key_paths=["/tenant"],
            text_paths=["/text"],
        ),
        executor=CosmosExecutor(config=QueryEngineConfig()),
    )
    return create_app(settings, retrievers={"C": retriever})


def _search_then_answer() -> list[LlmTurn]:
    return [
        LlmTurn(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "full_text_search",
                            "arguments": '{"query": "hi"}',
                        },
                    }
                ],
            },
            content="",
            tool_calls=[ToolCall("c1", "full_text_search", {"query": "hi"})],
        ),
        LlmTurn(message={"role": "assistant", "content": "the answer"}, content="the answer"),
    ]


def test_agent_search_runs_loop_and_returns_answer():
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat(_search_then_answer())
        response = http.post("/agent_search", json={"query": "what is up?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "the answer",
        "documents": [],
        "terminal_reason": "stop",
        "turns": 2,
    }


def test_agent_search_returns_supporting_documents():
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat(_search_then_answer())
        http.app.state.retriever = _FakeRetriever(
            [_document("d1", 0, "recycling one"), _document("d2", 1, "recycling two")]
        )
        response = http.post("/agent_search", json={"query": "recycling?"})
    body = response.json()
    assert body["answer"] == "the answer"
    assert [doc["retrieval_id"] for doc in body["documents"]] == ["d1", "d2"]
    assert body["documents"][0]["text"] == "recycling one"


def test_agent_search_blank_query_rejected():
    with TestClient(_app(_settings())) as http:
        http.app.state.chat_client = _FakeChat([])
        response = http.post("/agent_search", json={"query": "   "})
    assert response.status_code == 422


def test_configured_llm_builds_chat_client():
    with TestClient(_app(_settings())) as http:
        assert isinstance(http.app.state.chat_client, ChatClient)
