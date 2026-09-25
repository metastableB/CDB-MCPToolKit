from __future__ import annotations

import pytest

from cosmos_agentic_retriever.query_engine.results_mapping import (
    assemble_text,
    row_text_fields,
    rows_to_items,
)


def test_row_text_fields_maps_aliased_txt_keys() -> None:
    row = {"txt_a": "hello", "txt_b": "world"}
    aliases = {"txt_a": "/title", "txt_b": "/body"}
    assert row_text_fields(row, aliases) == {"/title": "hello", "/body": "world"}


def test_row_text_fields_ignores_non_txt_keys() -> None:
    row = {"item_id": "x", "md_foo": "m", "title": "t"}
    aliases = {"item_id": "item", "title": "T"}
    assert row_text_fields(row, aliases) == {}


def test_row_text_fields_ignores_txt_key_absent_from_aliases() -> None:
    assert row_text_fields({"txt_a": "v"}, {}) == {}


def test_row_text_fields_coerces_none_and_falsy_to_empty() -> None:
    row = {"txt_a": None, "txt_b": "", "txt_c": 0}
    aliases = {"txt_a": "/a", "txt_b": "/b", "txt_c": "/c"}
    assert row_text_fields(row, aliases) == {"/a": "", "/b": "", "/c": ""}


def test_row_text_fields_empty_row() -> None:
    assert row_text_fields({}, {"txt_a": "/a"}) == {}


def test_assemble_text_empty_fields_returns_empty() -> None:
    assert assemble_text({}) == ""


def test_assemble_text_single_field_no_header() -> None:
    assert assemble_text({"/body": "hello"}) == "hello"


def test_assemble_text_single_field_none_value() -> None:
    assert assemble_text({"/body": None}) == ""


def test_assemble_text_multiple_fields_joined_with_headers() -> None:
    result = assemble_text({"/title": "T", "/body": "B"})
    assert result == "[/title]\nT\n\n[/body]\nB"


def test_assemble_text_names_filter_to_present_only() -> None:
    fields = {"/title": "T", "/body": "B"}
    assert assemble_text(fields, ["/body"]) == "B"


def test_assemble_text_names_control_order() -> None:
    fields = {"/a": "A", "/b": "B"}
    assert assemble_text(fields, ["/b", "/a"]) == "[/b]\nB\n\n[/a]\nA"


def test_assemble_text_names_none_present_returns_empty() -> None:
    assert assemble_text({"/a": "A"}, ["/missing"]) == ""


def test_assemble_text_names_skip_absent() -> None:
    fields = {"/a": "A", "/b": "B"}
    assert assemble_text(fields, ["/a", "/missing", "/b"]) == "[/a]\nA\n\n[/b]\nB"


def test_rows_to_items_empty() -> None:
    assert rows_to_items([], strategy="s") == []


def test_rows_to_items_full_row() -> None:
    row = {
        "item_id": 42,
        "document_id": 7,
        "chunk_id": 3,
        "chunk_order": 5,
        "txt_a": "hello",
        "md_0": 0.9,
        "md_1": "x",
        "title": "T",
        "source": "S",
    }
    aliases = {"txt_a": "/body", "md_0": "score", "md_1": "source_tag"}
    items = rows_to_items(
        [row],
        strategy="vector",
        channels=["vector"],
        projected_aliases=aliases,
    )
    item = items[0]
    assert item.item_id == "42"
    assert item.document_id == "7" and item.chunk_id == "3"
    assert item.chunk_order == 5
    assert item.text == "hello"
    assert item.text_fields == {"/body": "hello"}
    assert item.title == "T" and item.source == "S"
    assert item.metadata == {"score": 0.9, "source_tag": "x"}
    assert item.retrieval_strategy == "vector"
    assert item.retrieval_channels == ["vector"]
    assert item.rank == 0


def test_rows_to_items_optional_ids_remain_none() -> None:
    row = {"item_id": "x", "document_id": None, "chunk_id": None}
    item = rows_to_items([row], strategy="s")[0]
    assert item.item_id == "x"
    assert item.document_id is None
    assert item.chunk_id is None


def test_rows_to_items_non_int_chunk_order_becomes_none() -> None:
    for bad in ("5", 1.5, None, True, False):
        item = rows_to_items([{"item_id": "x", "chunk_order": bad}], strategy="s")[0]
        assert item.chunk_order is None


def test_rows_to_items_channels_default_empty_and_copied() -> None:
    channels = ["vector"]
    item = rows_to_items([{"item_id": "x"}], strategy="s", channels=channels)[0]
    assert item.retrieval_channels == ["vector"]
    assert item.retrieval_channels is not channels
    second = rows_to_items([{"item_id": "x"}], strategy="s")[0]
    assert second.retrieval_channels == []


def test_rows_to_items_rank_uses_start_rank_offset() -> None:
    rows = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
    items = rows_to_items(rows, strategy="s", start_rank=10)
    assert [item.rank for item in items] == [10, 11, 12]


def test_rows_to_items_queried_text_fields_filter_display() -> None:
    row = {"item_id": "x", "txt_a": "A", "txt_b": "B"}
    aliases = {"txt_a": "/fa", "txt_b": "/fb"}
    item = rows_to_items(
        [row], strategy="s", projected_aliases=aliases, queried_text_fields=["/fb"]
    )[0]
    assert item.text == "B"


def test_rows_to_items_metadata_only_md_prefixed() -> None:
    row = {"item_id": "x", "md_0": 1, "md_unmapped": 3, "b": 2, "txt_c": "c"}
    item = rows_to_items(
        [row], strategy="s", projected_aliases={"txt_c": "/c", "md_0": "a", "b": "b"}
    )[0]
    assert item.metadata == {"a": 1}


@pytest.mark.parametrize("row", [{}, {"item_id": None}])
def test_rows_to_items_missing_item_id_raises(row) -> None:
    with pytest.raises(ValueError, match="item_id"):
        rows_to_items([row], strategy="full_text")


def test_identity_projection_preserves_logical_id_and_metadata():
    from cosmos_agentic_retriever.query_engine import CorpusSchema, CosmosQueryCompiler

    schema = CorpusSchema(
        item_id_path="/record/id",
        text_paths=["/text"],
        partition_key_paths=["/tenant", "/region"],
        metadata_paths={"cosmos_identity": "/label"},
    )
    sql, aliases = CosmosQueryCompiler(schema).projection("@k")
    assert (
        '{"id": c["id"], "partition_key": [IIF(IS_DEFINED(c["tenant"]), c["tenant"], {}), IIF(IS_DEFINED(c["region"]), c["region"], {})]} AS _cosmos_identity'
        in sql
    )
    row = {
        "item_id": "logical",
        "txt_0": "Text",
        "md_0": "label",
        "_cosmos_identity": {"id": "physical", "partition_key": [0, {}]},
    }
    item = rows_to_items([row], strategy="full_text", projected_aliases=aliases)[0]
    assert item.item_id == "logical"
    assert item.cosmos_identity.id == "physical"
    assert item.cosmos_identity.partition_key == (0, {})
    assert item.metadata == {"cosmos_identity": "label"}
    assert item.text_fields == {"/text": "Text"}


@pytest.mark.parametrize(
    "identity",
    [
        None,
        {},
        {"id": "x"},
        {"id": "x", "partition_key": []},
        {"id": None, "partition_key": [0]},
    ],
)
def test_incomplete_projected_identity_is_not_silently_accepted(identity):
    with pytest.raises(ValueError):
        rows_to_items(
            [{"item_id": "logical", "_cosmos_identity": identity}],
            strategy="full_text",
            projected_aliases={"_cosmos_identity": "cosmos_identity"},
        )
