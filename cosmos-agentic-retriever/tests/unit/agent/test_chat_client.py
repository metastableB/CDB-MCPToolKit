"""Check model requests and replies without calling a model endpoint."""

from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion

from cosmos_agentic_retriever.agent import chat_client
from cosmos_agentic_retriever.agent.loop import Tool, ToolCall


@pytest.fixture
def client(monkeypatch):
    sdk = Mock()
    monkeypatch.setattr(chat_client, "OpenAI", Mock(return_value=sdk))
    return chat_client.ChatClient(
        base_url="https://model.example.com/v1", model="test-model"
    ), sdk.chat.completions.create


def _reply(*, tool_calls=None):
    return ChatCompletion.model_validate(
        {
            "id": "reply-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": None if tool_calls else "The answer.",
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


def test_complete_sends_tool_descriptions_without_executing_handlers(client):
    chat, create = client
    chat.max_tokens = 256
    create.return_value = _reply()
    messages = [{"role": "user", "content": "Find recycling documents."}]
    handler = Mock()
    tool = Tool(
        name="full_text_search",
        description="Search the configured containers.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=handler,
    )

    turn = chat.complete(messages, [tool])

    create.assert_called_once_with(
        max_tokens=256,
        model="test-model",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
        ],
    )
    handler.assert_not_called()
    assert turn.content == "The answer."
    assert turn.tool_calls == [] and turn.error is None


@pytest.mark.parametrize(
    "arguments, expected",
    [(' {"query": "recycling"} ', {"query": "recycling"}), ("broken", {}), ("[]", {})],
)
def test_complete_returns_tool_calls_for_the_loop(client, arguments, expected):
    chat, create = client
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "full_text_search", "arguments": arguments},
    }
    create.return_value = _reply(tool_calls=[call])

    turn = chat.complete([], [])

    assert turn.error is None
    assert turn.tool_calls == [ToolCall("call-1", "full_text_search", expected)]
    assert turn.message["tool_calls"] == [call]


def test_complete_returns_request_errors_to_the_loop(client):
    chat, create = client
    create.side_effect = RuntimeError("endpoint unavailable")

    turn = chat.complete([], [])

    assert turn.error == "RuntimeError: endpoint unavailable"
    assert turn.tool_calls == []


def test_complete_returns_error_for_missing_reply(client):
    chat, create = client
    response = _reply()
    response.choices = []
    create.return_value = response

    turn = chat.complete([], [])

    assert turn.error == "ValueError: Model returned no choices."
    assert turn.tool_calls == []


def test_complete_reports_unsupported_tool_calls(client):
    chat, create = client
    create.return_value = _reply(
        tool_calls=[
            {
                "id": "call-1",
                "type": "custom",
                "custom": {"name": "unsupported", "input": "query"},
            }
        ]
    )

    turn = chat.complete([], [])

    assert turn.error == "TypeError: Unsupported tool call type: 'custom'."
    assert turn.tool_calls == []
