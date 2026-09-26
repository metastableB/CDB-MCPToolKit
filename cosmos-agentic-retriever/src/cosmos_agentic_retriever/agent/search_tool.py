"""The `full_text_search` tool: search the configured containers.

`make_full_text_search_tool` builds the `full_text_search` tool from a `search`
callable that runs one query against the in-process multi-container retriever.
The tool validates the model's arguments, runs the search, and formats the ranked
items as text the model reads and refines against. Each item is shown with its
retrieval id so the model can refer to exact results.
"""

from __future__ import annotations

from collections.abc import Callable

from cosmos_agentic_retriever.agent.loop import Tool
from cosmos_agentic_retriever.orchestration import MultiSearchResult

SearchFn = Callable[[str, int], MultiSearchResult]

_DESCRIPTION = (
    "Search the configured document corpus with a full-text query and get back "
    "the most relevant items. Call it more than once, refining the query from "
    "what you read, until you can answer."
)


def _format_results(result: MultiSearchResult) -> str:
    """Render ranked items as text, one block per item, keyed by retrieval id."""
    if not result.items:
        return "No results. Try a different query."
    blocks = []
    for item in result.items:
        text = item.text.strip() or "(no text)"
        blocks.append(f"[{item.retrieval_id}] ({item.container})\n{text}")
    return "\n\n".join(blocks)


def make_full_text_search_tool(
    search: SearchFn,
    *,
    default_max_documents: int = 10,
    max_documents_cap: int = 50,
) -> Tool:
    """Build the `full_text_search` tool over a `search` callable.

    Args:
        search: runs one query and returns a `MultiSearchResult`; the handler
            owns none of the retrieval, only the argument checks and formatting.
        default_max_documents: how many items to request when the model omits a
            count.
        max_documents_cap: the largest count the model may request; larger
            requests are clamped so one call cannot flood the transcript.
    """

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The full-text search query.",
                "minLength": 1,
            },
            "max_documents": {
                "type": "integer",
                "description": "How many items to return.",
                "minimum": 1,
                "maximum": max_documents_cap,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def handler(arguments: dict) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: 'query' is required and must be a non-empty string."
        requested = arguments.get("max_documents", default_max_documents)
        if not isinstance(requested, int) or isinstance(requested, bool):
            return "Error: 'max_documents' must be an integer."
        count = max(1, min(requested, max_documents_cap))
        result = search(query, count)
        return _format_results(result)

    return Tool(
        name="full_text_search",
        description=_DESCRIPTION,
        parameters=parameters,
        handler=handler,
    )
