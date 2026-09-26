"""A bounded, single-tool search agent over the configured containers.

The agent runs a short chat loop: it asks a language model what to search for,
runs the search, shows the model the results, and repeats until the model answers
or a turn cap is hit. It exposes one capability, ``search_corpus``, backed by the
in-process multi-container retriever.

The loop (`loop.py`) does not depend on any model provider; the model call
(`llm.py`) is the only piece that knows the OpenAI chat wire format. Splitting
them this way means switching to a different model provider changes one file, not
the loop.
"""

from cosmos_agentic_retriever.agent.llm import ChatClient
from cosmos_agentic_retriever.agent.loop import (
    AgentResult,
    LlmTurn,
    Tool,
    ToolCall,
    run_agent_search,
)
from cosmos_agentic_retriever.agent.search_tool import make_search_corpus_tool

__all__ = [
    "AgentResult",
    "ChatClient",
    "LlmTurn",
    "Tool",
    "ToolCall",
    "make_search_corpus_tool",
    "run_agent_search",
]
