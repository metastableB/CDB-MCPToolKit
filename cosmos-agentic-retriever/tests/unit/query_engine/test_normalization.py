from __future__ import annotations

from cosmos_agentic_retriever.query_engine.normalization import (
    assemble_text,
    normalize_rows,
    row_text_fields,
)


def test_row_text_fields_maps_aliased_txt_keys() -> None:
    row = {"txt_a": "hello", "txt_b": "world"}
    aliases = {"txt_a": "title", "txt_b": "body"}
    assert row_text_fields(row, aliases) == {"title": "hello", "body": "world"}


def test_row_text_fields_ignores_non_txt_keys() -> None:
    row = {"item_id": "x", "md_foo": "m", "title": "t"}
    aliases = {"item_id": "item", "title": "T"}
    assert row_text_fields(row, aliases) == {}


def test_row_text_fields_ignores_txt_key_absent_from_aliases() -> None:
    assert row_text_fields({"txt_a": "v"}, {}) == {}


def test_row_text_fields_coerces_none_and_falsy_to_empty() -> None:
    row = {"txt_a": None, "txt_b": "", "txt_c": 0}
    aliases = {"txt_a": "a", "txt_b": "b", "txt_c": "c"}
    assert row_text_fields(row, aliases) == {"a": "", "b": "", "c": ""}


def test_row_text_fields_empty_row() -> None:
    assert row_text_fields({}, {"txt_a": "a"}) == {}


def test_assemble_text_empty_fields_returns_empty() -> None:
    assert assemble_text({}) == ""


def test_assemble_text_single_field_no_header() -> None:
    assert assemble_text({"body": "hello"}) == "hello"


def test_assemble_text_single_field_none_value() -> None:
    assert assemble_text({"body": None}) == ""


def test_assemble_text_multiple_fields_joined_with_headers() -> None:
    result = assemble_text({"title": "T", "body": "B"})
    assert result == "[title]\nT\n\n[body]\nB"


def test_assemble_text_names_filter_to_present_only() -> None:
    fields = {"title": "T", "body": "B"}
    assert assemble_text(fields, ["body"]) == "B"


def test_assemble_text_names_control_order() -> None:
    fields = {"a": "A", "b": "B"}
    assert assemble_text(fields, ["b", "a"]) == "[b]\nB\n\n[a]\nA"


def test_assemble_text_names_none_present_returns_empty() -> None:
    assert assemble_text({"a": "A"}, ["missing"]) == ""


def test_assemble_text_names_skip_absent() -> None:
    fields = {"a": "A", "b": "B"}
    assert assemble_text(fields, ["a", "missing", "b"]) == "[a]\nA\n\n[b]\nB"


def test_normalize_rows_empty() -> None:
    assert normalize_rows([], strategy="s") == []


def test_normalize_rows_full_row() -> None:
    row = {
        "item_id": 42,
        "document_id": 7,
        "chunk_id": 3,
        "chunk_order": 5,
        "txt_a": "hello",
        "md_score": 0.9,
        "md_source_tag": "x",
        "title": "T",
        "source": "S",
    }
    aliases = {"txt_a": "body"}
    items = normalize_rows(
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
    assert item.text_fields == {"body": "hello"}
    assert item.title == "T" and item.source == "S"
    assert item.metadata == {"score": 0.9, "source_tag": "x"}
    assert item.retrieval_strategy == "vector"
    assert item.retrieval_channels == ["vector"]
    assert item.rank == 0


def test_normalize_rows_none_ids_passthrough() -> None:
    row = {"item_id": None, "document_id": None, "chunk_id": None}
    item = normalize_rows([row], strategy="s")[0]
    assert item.item_id == "None"
    assert item.document_id is None
    assert item.chunk_id is None


def test_normalize_rows_non_int_chunk_order_becomes_none() -> None:
    for bad in ("5", 1.5, None):
        item = normalize_rows([{"item_id": "x", "chunk_order": bad}], strategy="s")[0]
        assert item.chunk_order is None


def test_normalize_rows_channels_default_empty_and_copied() -> None:
    channels = ["vector"]
    item = normalize_rows([{"item_id": "x"}], strategy="s", channels=channels)[0]
    assert item.retrieval_channels == ["vector"]
    assert item.retrieval_channels is not channels
    second = normalize_rows([{"item_id": "x"}], strategy="s")[0]
    assert second.retrieval_channels == []


def test_normalize_rows_rank_uses_start_rank_offset() -> None:
    rows = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
    items = normalize_rows(rows, strategy="s", start_rank=10)
    assert [item.rank for item in items] == [10, 11, 12]


def test_normalize_rows_queried_text_fields_filter_display() -> None:
    row = {"item_id": "x", "txt_a": "A", "txt_b": "B"}
    aliases = {"txt_a": "fa", "txt_b": "fb"}
    item = normalize_rows(
        [row], strategy="s", projected_aliases=aliases, queried_text_fields=["fb"]
    )[0]
    assert item.text == "B"


def test_normalize_rows_metadata_only_md_prefixed() -> None:
    row = {"item_id": "x", "md_a": 1, "b": 2, "txt_c": "c"}
    item = normalize_rows([row], strategy="s", projected_aliases={"txt_c": "c"})[0]
    assert item.metadata == {"a": 1}
