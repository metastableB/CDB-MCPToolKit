"""The model call: the one piece that speaks the OpenAI chat wire format.

`ChatClient.complete` sends the running transcript and the tools to an
OpenAI-compatible chat-completions endpoint and returns a plain `LlmTurn` the loop
understands. Any endpoint with that wire protocol works -- OpenAI, Azure OpenAI,
vLLM, or another gateway -- so the service stays model-agnostic and a user brings
their own model by pointing `base_url`/`model` at it.

Keeping this call in its own file means the loop never imports a model SDK:
changing providers, or the wire format itself, changes this file alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from cosmos_agentic_retriever.agent.loop import LlmTurn, Tool, ToolCall

_logger = structlog.get_logger(__name__)


def to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Serialize tools to the chat-completions ``tools=[...]`` payload."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Parse a tool call's JSON arguments; treat anything malformed as empty.

    A malformed or non-object argument string becomes ``{}`` so the tool's own
    validation reports the problem back to the model instead of raising.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class ChatClient:
    """Call an OpenAI-compatible chat endpoint and return a vendor-neutral turn.

    ``base_url`` points at any endpoint speaking the chat-completions protocol.
    ``model`` names the deployment to call. A failed request is returned as an
    `LlmTurn` with ``error`` set, so the loop ends the episode cleanly rather than
    crashing.
    """

    base_url: str
    model: str
    api_key: str = "unused"
    max_tokens: int = 1024
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self._client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[Tool]) -> LlmTurn:
        """Send one request and map the reply onto an `LlmTurn`."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                tools=to_openai_tools(tools),  # type: ignore[arg-type]
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            _logger.exception("model_call_failed", model=self.model)
            return LlmTurn(
                message={"role": "assistant", "content": ""},
                error=f"{type(exc).__name__}: {exc}",
            )

        choice = response.choices[0].message
        message: dict[str, Any] = {"role": "assistant", "content": choice.content or ""}
        function_calls = [
            call
            for call in (choice.tool_calls or [])
            if isinstance(call, ChatCompletionMessageFunctionToolCall)
        ]
        tool_calls: list[ToolCall] = []
        if function_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in function_calls
            ]
            tool_calls = [
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_parse_arguments(call.function.arguments),
                )
                for call in function_calls
            ]
        return LlmTurn(
            message=message, content=choice.content, tool_calls=tool_calls
        )
