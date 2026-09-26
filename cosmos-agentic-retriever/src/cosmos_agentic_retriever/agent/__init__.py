"""A search agent that answers questions from the configured cosmosDB
containers.

The agent runs a short chat loop: it asks a language model what to search for,
runs the search, shows the model the results, and repeats until the model answers
or a turn cap is hit. It searches with the ``full_text_search`` tool, backed by the
in-process multi-container retriever.
"""

from cosmos_agentic_retriever.agent.llm import ChatClient
from cosmos_agentic_retriever.agent.loop import (
    AgentResult,
    LlmTurn,
    Tool,
    ToolCall,
    run_agent_search,
)
from cosmos_agentic_retriever.agent.search_tool import make_full_text_search_tool

__all__ = [
    "AgentResult",
    "ChatClient",
    "LlmTurn",
    "Tool",
    "ToolCall",
    "make_full_text_search_tool",
    "run_agent_search",
]
