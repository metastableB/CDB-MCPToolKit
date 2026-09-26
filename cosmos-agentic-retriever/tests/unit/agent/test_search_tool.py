"""The search_corpus tool: argument checks, clamping, and result formatting."""

from __future__ import annotations

from cosmos_agentic_retriever.agent.search_tool import make_search_corpus_tool
from cosmos_agentic_retriever.orchestration import ContainerItem, MultiSearchResult


def _result(*items: ContainerItem) -> MultiSearchResult:
    return MultiSearchResult(items=list(items))


def _item(item_id: str, text: str, retrieval_id: str) -> ContainerItem:
    return ContainerItem(
        item_id=item_id,
        text=text,
        database="D",
        container="A",
        retrieval_id=retrieval_id,
    )


def test_missing_query_returns_error():
    tool = make_search_corpus_tool(lambda q, n: _result())
    assert tool.handler({}).startswith("Error:")
    assert tool.handler({"query": "  "}).startswith("Error:")


def test_non_integer_max_documents_returns_error():
    tool = make_search_corpus_tool(lambda q, n: _result())
    assert "integer" in tool.handler({"query": "x", "max_documents": "5"})
    assert "integer" in tool.handler({"query": "x", "max_documents": True})


def test_max_documents_is_clamped_to_cap():
    seen: list[int] = []

    def search(query: str, count: int) -> MultiSearchResult:
        seen.append(count)
        return _result()

    tool = make_search_corpus_tool(search, max_documents_cap=20)
    tool.handler({"query": "x", "max_documents": 100})
    tool.handler({"query": "x", "max_documents": 0})
    assert seen == [20, 1]


def test_default_count_used_when_omitted():
    seen: list[int] = []

    def search(query: str, count: int) -> MultiSearchResult:
        seen.append(count)
        return _result()

    make_search_corpus_tool(search, default_max_documents=7).handler({"query": "x"})
    assert seen == [7]


def test_formats_items_with_retrieval_id_and_text():
    tool = make_search_corpus_tool(
        lambda q, n: _result(
            _item("a1", "the answer is 42", "D/A:k1"),
            _item("a2", "more context", "D/A:k2"),
        )
    )
    text = tool.handler({"query": "x"})
    assert "[D/A:k1]" in text and "the answer is 42" in text
    assert "[D/A:k2]" in text and "more context" in text


def test_empty_results_message():
    tool = make_search_corpus_tool(lambda q, n: _result())
    assert "No results" in tool.handler({"query": "x"})
