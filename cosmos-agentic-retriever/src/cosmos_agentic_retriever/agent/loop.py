"""The bounded chat loop that drives a model and its tools to an answer.

`run_agent_search` runs one search episode. Each turn it calls the model; if the
model asks for a tool, the loop runs the tool and feeds the result back; if the
model answers instead, the loop stops. It always stops after `max_turns` turns,
so it cannot run forever regardless of what the model does.

The loop does not know about any model provider. It calls the model through a
`complete` callable and appends whatever assistant message that call returns, so
it never has to parse the model's wire format. `llm.ChatClient` provides
`complete` for an OpenAI-compatible endpoint; a test can pass a stand-in.

A `Tool` is a plain capability: a name, a description, a JSON-Schema parameter
spec, and a handler that maps parsed arguments to an observation string. How the
model expresses a call is the model client's concern, not the tool's, so a tool
describes only what it does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

_logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Tool:
    """A capability the model can call: parsed arguments in, observation text out.

    ``parameters`` is a JSON-Schema object describing the arguments. ``handler``
    receives the parsed arguments and returns the text shown back to the model;
    it should return an error string rather than raise for a bad call, though the
    loop also catches exceptions.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ToolCall:
    """One tool request the model made this turn."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LlmTurn:
    """One model reply: its text, any tool calls, and the message to append.

    ``message`` is the assistant message the loop appends to the running
    transcript verbatim, so threading stays correct without the loop parsing the
    model's wire format. ``error`` is set when the model call itself failed.
    """

    message: dict[str, Any]
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None


@dataclass
class AgentResult:
    """The outcome of one episode: the final answer and how the loop ended.

    ``terminal_reason`` is ``"stop"`` (the model answered), ``"max_turns"`` (the
    turn cap was hit), or ``"error"`` (a model call failed). ``messages`` is the
    full transcript, including tool observations, for logging and debugging.
    """

    answer: str
    terminal_reason: str
    turns: int
    messages: list[dict[str, Any]]


def _run_tool_call(tools_by_name: dict[str, Tool], call: ToolCall) -> str:
    """Run one tool call, turning an unknown tool or a raised error into text."""
    tool = tools_by_name.get(call.name)
    if tool is None:
        return f"Error: unknown tool {call.name!r}."
    try:
        return tool.handler(call.arguments)
    except Exception as exc:  # a tool bug should not end the episode
        _logger.exception("tool_call_failed", tool=call.name)
        return f"Error: {type(exc).__name__}: {exc}"


def run_agent_search(
    question: str,
    *,
    complete: Callable[[list[dict[str, Any]], list[Tool]], LlmTurn],
    tools: list[Tool],
    system_prompt: str,
    max_turns: int,
) -> AgentResult:
    """Run the bounded search loop for one question and return its result.

    Args:
        question: the caller's request; the model decides what to search for.
        complete: calls the model with the running transcript and the tools, and
            returns an `LlmTurn`. `ChatClient.complete` is the real one.
        tools: the capabilities offered to the model (one, `search_corpus`, for
            now). Names must be unique.
        system_prompt: the instruction that opens the transcript.
        max_turns: the hard cap on model calls; the loop always stops by here.

    Returns:
        An `AgentResult` with the final answer, why the loop ended, the number of
        turns taken, and the full transcript.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    tools_by_name = {tool.name: tool for tool in tools}
    if len(tools_by_name) != len(tools):
        raise ValueError("tool names must be unique")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    terminal_reason = "max_turns"
    turn = 0
    for turn in range(1, max_turns + 1):
        reply = complete(messages, tools)
        messages.append(reply.message)
        if reply.error is not None:
            terminal_reason = "error"
            break
        if not reply.tool_calls:
            terminal_reason = "stop"
            break
        for call in reply.tool_calls:
            observation = _run_tool_call(tools_by_name, call)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )

    answer = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            answer = str(message["content"])
            break
    return AgentResult(
        answer=answer,
        terminal_reason=terminal_reason,
        turns=turn,
        messages=messages,
    )
