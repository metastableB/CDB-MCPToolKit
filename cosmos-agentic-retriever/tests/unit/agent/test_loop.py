"""Loop behavior: termination, tool dispatch, and error handling."""

from __future__ import annotations

import json

import pytest

from cosmos_agentic_retriever.agent.loop import (
    LlmTurn,
    Tool,
    ToolCall,
    run_agent_search,
)


def _answer(text: str) -> LlmTurn:
    return LlmTurn(message={"role": "assistant", "content": text}, content=text)


def _call(call_id: str, name: str, arguments: dict) -> LlmTurn:
    return LlmTurn(
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


class _ScriptedModel:
    """A stand-in `complete` that returns pre-scripted turns and records calls."""

    def __init__(self, turns: list[LlmTurn]) -> None:
        self._turns = list(turns)
        self.messages_seen: list[list[dict]] = []

    def __call__(self, messages, tools):
        self.messages_seen.append([dict(message) for message in messages])
        return self._turns.pop(0)


def _echo_tool(record: list[dict]) -> Tool:
    def handler(arguments: dict) -> str:
        record.append(arguments)
        return f"searched {arguments.get('query')!r}"

    return Tool(
        name="search_corpus",
        description="search",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def test_answers_without_searching_stops():
    model = _ScriptedModel([_answer("done")])
    result = run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=5,
    )
    assert result.terminal_reason == "stop"
    assert result.answer == "done"
    assert result.turns == 1


def test_searches_then_answers():
    calls: list[dict] = []
    model = _ScriptedModel(
        [_call("c1", "search_corpus", {"query": "cats"}), _answer("final")]
    )
    result = run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool(calls)],
        system_prompt="sys",
        max_turns=5,
    )
    assert calls == [{"query": "cats"}]
    assert result.terminal_reason == "stop"
    assert result.answer == "final"
    assert result.turns == 2
    roles = [message["role"] for message in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_stops_at_max_turns_when_model_keeps_calling():
    model = _ScriptedModel([_call(f"c{i}", "search_corpus", {"query": "x"}) for i in range(5)])
    result = run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=3,
    )
    assert result.terminal_reason == "max_turns"
    assert result.turns == 3


def test_unknown_tool_yields_error_observation_and_continues():
    model = _ScriptedModel(
        [_call("c1", "does_not_exist", {"query": "x"}), _answer("ok")]
    )
    result = run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=5,
    )
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert "unknown tool" in tool_messages[0]["content"]
    assert result.terminal_reason == "stop"
    assert result.answer == "ok"


def test_tool_exception_becomes_error_observation():
    def boom(_arguments: dict) -> str:
        raise RuntimeError("kaboom")

    tool = Tool(
        name="search_corpus",
        description="search",
        parameters={"type": "object", "properties": {}},
        handler=boom,
    )
    model = _ScriptedModel(
        [_call("c1", "search_corpus", {"query": "x"}), _answer("recovered")]
    )
    result = run_agent_search(
        "q", complete=model, tools=[tool], system_prompt="sys", max_turns=5
    )
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert "RuntimeError: kaboom" in tool_messages[0]["content"]
    assert result.terminal_reason == "stop"


def test_model_error_ends_episode():
    model = _ScriptedModel(
        [LlmTurn(message={"role": "assistant", "content": ""}, error="boom")]
    )
    result = run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=5,
    )
    assert result.terminal_reason == "error"
    assert result.turns == 1


def test_rejects_duplicate_tool_names():
    tool = _echo_tool([])
    with pytest.raises(ValueError, match="unique"):
        run_agent_search(
            "q",
            complete=_ScriptedModel([_answer("x")]),
            tools=[tool, tool],
            system_prompt="sys",
            max_turns=5,
        )


def test_rejects_nonpositive_max_turns():
    with pytest.raises(ValueError, match="max_turns"):
        run_agent_search(
            "q",
            complete=_ScriptedModel([_answer("x")]),
            tools=[_echo_tool([])],
            system_prompt="sys",
            max_turns=0,
        )
