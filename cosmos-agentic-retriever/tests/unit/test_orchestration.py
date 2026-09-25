"""Adapted search/fusion tests from PR #150, with identity and error regressions."""

from unittest.mock import Mock

import pytest

from cosmos_agentic_retriever.orchestration import (
    ContainerTarget,
    MultiContainerRetriever,
    fuse_rrf,
)
from cosmos_agentic_retriever.query_engine.types import (
    CosmosItemIdentity,
    RetrievedItem,
    SearchRequest,
)


def _items(prefix):
    return [
        RetrievedItem(
            item_id=f"{prefix}{index}",
            rank=index,
            cosmos_identity=CosmosItemIdentity(
                id=f"{prefix}{index}", partition_key=(0,)
            ),
        )
        for index in range(1, 4)
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
    import json
    from urllib.parse import unquote

    result = fuse_rrf(
        [
            (ContainerTarget("D", "A"), _items("a")),
            (ContainerTarget("D", "B"), _items("a")),
            (
                ContainerTarget("D/A", "B:C"),
                [
                    RetrievedItem(
                        item_id="logical",
                        cosmos_identity=CosmosItemIdentity(
                            id="id/%:1",
                            partition_key=("A/B:C%",),
                        ),
                    )
                ],
            ),
        ],
        limit=3,
    )
    assert len({item.retrieval_id for item in result}) == 3
    assert result[-1].retrieval_id.startswith("D%2FA/B%3AC:")
    assert json.loads(unquote(result[-1].retrieval_id.split(":")[1])) == [
        ["A/B:C%"],
        "id/%:1",
    ]


def test_same_id_in_different_partitions_survives_and_exact_item_deduplicates():
    target = ContainerTarget("D", "C")
    first = RetrievedItem(
        item_id="logical",
        text="first",
        cosmos_identity=CosmosItemIdentity(id="same", partition_key=("A",)),
    )
    second = RetrievedItem(
        item_id="logical",
        text="second",
        cosmos_identity=CosmosItemIdentity(id="same", partition_key=("B",)),
    )
    third = RetrievedItem(
        item_id="logical",
        text="third",
        cosmos_identity=CosmosItemIdentity(id="different", partition_key=("A",)),
    )
    result = fuse_rrf([(target, [first, second, third]), (target, [first])], limit=5)
    assert len(result) == 3
    assert {item.text for item in result} == {"first", "second", "third"}
    assert all(item.item_id == "logical" for item in result)
    assert len({item.retrieval_id for item in result}) == 3
    assert [item.rank for item in result] == [0, 1, 2]
    assert result[0].text == "first"
    alone = fuse_rrf([(target, [second])], limit=1)[0]
    assert alone.retrieval_id == next(
        item.retrieval_id for item in result if item.text == "second"
    )


def test_partition_component_types_and_hierarchy_are_unambiguous():
    values = [(0,), ("0",), (False,), (None,), ({},), ("",), ("A", "B"), ("B", "A")]
    assert len(
        {CosmosItemIdentity(id="same", partition_key=value).key() for value in values}
    ) == len(values)
    assert (
        CosmosItemIdentity(id="same", partition_key=(1,)).key()
        == CosmosItemIdentity(id="same", partition_key=(1.0,)).key()
    )
    assert (
        CosmosItemIdentity(id="same", partition_key=(-0.0,)).key()
        == CosmosItemIdentity(id="same", partition_key=(0,)).key()
    )


@pytest.mark.parametrize(
    "components",
    [(), (float("nan"),), (float("inf"),), (["A"],), ({"a": 1},), (1, 2, 3, 4)],
)
def test_invalid_physical_identity_is_rejected(components):
    with pytest.raises(ValueError):
        CosmosItemIdentity(id="same", partition_key=components)


def test_missing_identity_does_not_fall_back_to_logical_id():
    with pytest.raises(ValueError, match="physical Cosmos identity"):
        fuse_rrf(
            [(ContainerTarget("D", "C"), [RetrievedItem(item_id="same")])], limit=1
        )


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
