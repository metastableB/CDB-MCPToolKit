"""Adapted search/fusion tests from PR #150, with identity and error regressions."""

from unittest.mock import Mock

import pytest

from cosmos_agentic_retriever.orchestration import (
    ContainerTarget,
    MultiContainerRetriever,
    fuse_rrf,
)
from cosmos_agentic_retriever.query_engine.types import RetrievedItem, SearchRequest


def _items(prefix):
    return [
        RetrievedItem(item_id=f"{prefix}{index}", rank=index) for index in range(1, 4)
    ]


def test_fuse_rrf_interleaves_and_tags_metadata():
    first, second = ContainerTarget("D", "A"), ContainerTarget("D", "B")
    original = _items("a")
    result = fuse_rrf([(first, original), (second, _items("b"))], limit=3)
    assert [item.item_id for item in result] == ["a1", "b1", "a2"]
    assert [(item.database, item.container, item.rank) for item in result] == [
        ("D", "A", 0),
        ("D", "B", 1),
        ("D", "A", 2),
    ]
    assert original[0].rank == 1 and original[0].metadata == {}


def test_qualified_ids_are_distinct_and_delimiters_escaped():
    result = fuse_rrf(
        [
            (ContainerTarget("D", "A"), [RetrievedItem(item_id="same")]),
            (ContainerTarget("D", "B"), [RetrievedItem(item_id="same")]),
            (ContainerTarget("D/A", "B:C"), [RetrievedItem(item_id="id/%:1")]),
        ],
        limit=3,
    )
    assert [item.retrieval_id for item in result] == [
        "D/A:same",
        "D/B:same",
        "D%2FA/B%3AC:id%2F%25%3A1",
    ]


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_multi_search_fans_out_and_preserves_failures(failures):
    targets = [ContainerTarget("D", "A"), ContainerTarget("D", "B")]
    retrievers = {target: Mock() for target in targets}
    requests = {target: SearchRequest(query="battery", limit=3) for target in targets}
    for index, target in enumerate(targets):
        retrievers[target].search.return_value = _items(target.container)
        if index < failures:
            retrievers[target].search.side_effect = RuntimeError("private-endpoint")
    result = MultiContainerRetriever(retrievers, max_workers=2).search(
        requests, limit=3
    )
    assert result.searched == targets[failures:]
    assert result.errors == {target: "Search failed." for target in targets[:failures]}
    assert len(result.items) == (3 if failures < 2 else 0)
    for target in targets:
        retrievers[target].search.assert_called_once_with(requests[target])


def test_empty_targets():
    assert MultiContainerRetriever({}, max_workers=1).search({}, limit=2).items == []


@pytest.mark.parametrize("limit", [0, -1, True])
def test_fusion_rejects_invalid_limits(limit):
    with pytest.raises(ValueError):
        fuse_rrf([], limit=limit)
