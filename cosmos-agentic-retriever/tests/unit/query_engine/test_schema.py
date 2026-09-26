from itertools import permutations

import pytest
from pydantic import ValidationError

from cosmos_agentic_retriever.query_engine import (
    CorpusSchema,
    CosmosPath,
    CosmosQueryCompiler,
    EqualsFilter,
)
from cosmos_agentic_retriever.query_engine.types import UnsafeCosmosPathError


def _compile(schema: CorpusSchema):
    return CosmosQueryCompiler(schema).compile_structured(
        limit=5,
        filters=[],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )


def test_minimal_schema_and_independent_defaults() -> None:
    first = CorpusSchema(item_id_path="/id")
    second = CorpusSchema(item_id_path="/id")
    assert _compile(first).sql == 'SELECT TOP @k0 c["id"] AS item_id FROM c'
    assert first.parent_document_id_path is None
    assert first.text_paths_by_string() == {}
    first.text_paths.append(CosmosPath.parse("/text"))
    first.additional_return_paths.append(CosmosPath.parse("/year"))
    assert second.text_paths == []
    assert second.additional_return_paths == []
    first.partition_key_paths.append(CosmosPath.parse("/tenant"))
    assert second.partition_key_paths == []


@pytest.mark.parametrize(
    "paths", [["/tenant", "/tenant"], ["/a", "/b", "/c", "/d"], ["tenant"], [None]]
)
def test_partition_identity_paths_are_validated(paths):
    with pytest.raises((ValueError, UnsafeCosmosPathError)):
        CorpusSchema(item_id_path="/id", partition_key_paths=paths)


def test_partition_identity_paths_revalidated_before_projection():
    schema = CorpusSchema(item_id_path="/id", partition_key_paths=["/tenant"])
    schema.partition_key_paths.append(CosmosPath.parse("/tenant"))
    with pytest.raises(ValueError, match="duplicates"):
        _compile(schema)


def test_additional_return_paths_reject_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        CorpusSchema(item_id_path="/id", additional_return_paths=["/year", "/year"])


@pytest.mark.parametrize(
    "field",
    [
        "item_id_path",
        "parent_document_id_path",
        "chunk_id_path",
        "chunk_order_path",
    ],
)
def test_path_fields_accept_strings_and_objects(field: str) -> None:
    path = CosmosPath.parse("/nested/value")
    from_string = CorpusSchema(**{"item_id_path": "/id", field: str(path)})
    from_object = CorpusSchema(**{"item_id_path": "/id", field: path})
    assert getattr(from_string, field) == path
    assert getattr(from_object, field) is path
    if field != "item_id_path":
        assert (
            getattr(CorpusSchema(**{"item_id_path": "/id", field: None}), field) is None
        )


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"item_id_path": None},
        {"item_id_path": "id"},
        {"item_id_path": "/id", "text_paths": None},
        {"item_id_path": "/id", "text_paths": "/text"},
        {"item_id_path": "/id", "text_paths": [None]},
        {"item_id_path": "/id", "additional_return_paths": ["year"]},
    ],
)
def test_invalid_schema_inputs_are_rejected(settings: dict) -> None:
    with pytest.raises((ValidationError, UnsafeCosmosPathError)):
        CorpusSchema(**settings)


def test_misspelled_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="text_path"):
        CorpusSchema(item_id_path="/id", text_path="/text")


def test_text_names_never_overwrite_a_distinct_path() -> None:
    paths = [
        CosmosPath.parse(raw)
        for raw in ['/"/b/text"', '/"/b/text#2"', "/a/text", "/b/text"]
    ]
    for order in permutations(paths):
        schema = CorpusSchema(item_id_path="/id", text_paths=list(order))
        assert schema.text_paths_by_string() == {str(path): path for path in paths}
        assert schema.resolve_text_fields([str(paths[0])]) == [paths[0]]
        compiled = _compile(schema)
        for path in paths:
            assert f"{path.render()} AS txt_" in compiled.sql


def test_text_paths_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        CorpusSchema(item_id_path="/id", text_paths=["/a/text", "/b/text", "/b/text"])


def test_assignment_coerces_paths_and_rejects_invalid_updates() -> None:
    schema = CorpusSchema(item_id_path="/id")
    schema.text_paths = ["/content/text"]
    schema.additional_return_paths = ["/publication/year"]
    assert schema.text_paths == [CosmosPath.parse("/content/text")]
    assert schema.additional_return_paths == [CosmosPath.parse("/publication/year")]
    with pytest.raises(UnsafeCosmosPathError):
        schema.additional_return_paths = ["invalid"]
    assert schema.additional_return_paths == [CosmosPath.parse("/publication/year")]
    with pytest.raises(UnsafeCosmosPathError):
        schema.text_paths = ["invalid"]
    assert schema.text_paths == [CosmosPath.parse("/content/text")]
    with pytest.raises(UnsafeCosmosPathError):
        schema.item_id_path = None
    assert schema.item_id_path == CosmosPath.parse("/id")
    with pytest.raises(ValidationError):
        schema.text_path = "/typo"


def test_compiler_revalidates_in_place_edits() -> None:
    schema = CorpusSchema(item_id_path="/id")
    compiler = CosmosQueryCompiler(schema)
    assert "txt_0" not in compiler.projection("@k0")[0]
    schema.text_paths.append("/content/text")
    schema.additional_return_paths.append("/publication/year")
    assert schema.text_paths_by_string() == {
        "/content/text": CosmosPath.parse("/content/text")
    }
    compiled = compiler.compile_structured(
        limit=5,
        filters=[EqualsFilter(path="/publication/year", value=2020)],
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )
    assert 'c["content"]["text"] AS txt_0' in compiled.sql
    assert 'WHERE c["publication"]["year"] = @p1' in compiled.sql
    assert compiler.schema is schema


@pytest.mark.parametrize("edit", ["invalid_text", "invalid_additional_path"])
def test_compiler_rejects_invalid_in_place_edits(edit: str) -> None:
    schema = CorpusSchema(item_id_path="/id")
    if edit == "invalid_text":
        schema.text_paths.append("invalid")
    else:
        schema.additional_return_paths.append("invalid")
    with pytest.raises((ValidationError, UnsafeCosmosPathError)):
        _compile(schema)


@pytest.mark.parametrize(
    "method",
    [
        "compile_vector",
        "compile_full_text",
        "compile_hybrid",
        "compile_structured",
        "compile_document_read",
    ],
)
def test_every_query_checks_schema_before_emitting_sql(method: str) -> None:
    schema = CorpusSchema(item_id_path="/id", parent_document_id_path="/docid")
    compiler = CosmosQueryCompiler(schema)
    schema.additional_return_paths.append("invalid")
    arguments = {"partition_key": None, "cross_partition": True}
    if method == "compile_document_read":
        arguments.update(document_id="report", max_chunks=5)
    else:
        arguments.update(limit=5, filters=[], ignored_item_ids=[])
    if method in ("compile_vector", "compile_hybrid"):
        arguments.update(query_vector=[0.1], vector_path=CosmosPath.parse("/embedding"))
    if method in ("compile_full_text", "compile_hybrid"):
        arguments.update(query="battery", text_paths=[CosmosPath.parse("/text")])
    with pytest.raises(UnsafeCosmosPathError, match="path must start"):
        getattr(compiler, method)(**arguments)
