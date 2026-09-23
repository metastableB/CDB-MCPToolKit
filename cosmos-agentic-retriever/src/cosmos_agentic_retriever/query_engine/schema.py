"""Tell the query compiler where IDs and text are stored in each JSON record.

Consider the following Cosmos DB item stored as a JSON record.
        {"id": "report-7-0", "docid": "report-7",
         "content": {"text": "Battery recycling..."}}

A field path, such as `/id`, is used to describe a location inside that record:
`/id` means the top-level id field, and `/content/text` means the text field
inside the content object. The CorpusSchema allows us to map arbitrary
cosmos DB schema into one that the query engine understands.

For the item above, a CorpusSchema can be:
- item_id_path="/id": the location of this item's identifier, "report-7-0".
- text_paths=["/content/text"]: the locations of text to include in query
    results, here "Battery recycling...". This is a list because an item may have
    several text fields, such as a title and a body.
- document_id_path="/docid": the location of the source document's identifier,
    "report-7". Other chunks of that document can share this value.

A source document, such as a report, may be split into chunks stored as separate
items. chunk_id_path identifies the field holding a chunk's ID; it can be /id
when the item ID already identifies the chunk. chunk_order_path identifies the
field holding that chunk's position in the report, such as /chunk_idx.

Only item_id_path is required to construct CorpusSchema. Omitting text_paths
means no text fields are selected for output, not that text is discovered
automatically. These settings describe stored data; they do not change it.

Unknown settings are rejected. Replacing a field validates its new value;
in-place list or dictionary edits are checked again before query compilation.
Metadata names cannot reuse item_id, document_id, chunk_id, chunk_order, title,
or source, because those names already have a meaning in compiler filters.

TODO: The schema here feels adhoc and non-generalizable. Revisit this design.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from cosmos_agentic_retriever.query_engine.paths import CosmosPath, coerce_path
from cosmos_agentic_retriever.query_engine.types import UnknownField

PathField = Annotated[CosmosPath, BeforeValidator(coerce_path)]


class CorpusSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, revalidate_instances="always"
    )

    # Path to the item's identifier field, e.g. /id.
    item_id_path: PathField
    # Paths to text fields to return, e.g. ["/title", "/content/text"].
    text_paths: list[PathField] = Field(default_factory=list)
    # Path to the source document ID shared by its chunks, e.g. /docid.
    document_id_path: PathField | None = None
    # Path to a chunk's ID, e.g. /chunk_id; may be the same as item_id_path.
    chunk_id_path: PathField | None = None
    # Path to a chunk's position within its source document, e.g. /chunk_idx.
    chunk_order_path: PathField | None = None
    # Path to the title to return with the result, e.g. /title.
    title_path: PathField | None = None
    # Path to a source reference or label to return, e.g. /url or /source_type.
    source_path: PathField | None = None
    # Extra fields to return and filter on, e.g. {"year": "/publication/year"}.
    metadata_paths: dict[str, PathField] = Field(default_factory=dict)

    @field_validator("metadata_paths")
    @classmethod
    def _check_metadata_names(
        cls, paths: dict[str, CosmosPath]
    ) -> dict[str, CosmosPath]:
        reserved = {
            "item_id",
            "document_id",
            "chunk_id",
            "chunk_order",
            "title",
            "source",
        }
        conflicts = reserved.intersection(paths)
        if conflicts:
            raise ValueError(
                f"metadata names are reserved: {', '.join(sorted(conflicts))}"
            )
        return paths

    def text_field_map(self) -> dict[str, CosmosPath]:
        """Return a lookup for the text paths configured in this schema.

        Our configuration and search requests use strings like "/article/text"
        to refer to fields.  Cosmos SQL needs c["article"]["text"] instead. The
        schema parses each configured path into a CosmosPath object, whose
        render() method produces that SQL expression.

        render() uses json.dumps() to quote each field name and escape any
        double quotes, backslashes, or control characters inside it. This keeps
        those characters part of the field name instead of SQL syntax.

        For text_paths=["/article/text"], return:
            {"/article/text": CosmosPath(segments=("article", "text"))}

        This lookup lets resolve_text_fields check that a requested path is
        configured and return its parsed object. The compiler also uses the
        string keys to label returned text. This method does not generate SQL
        or read documents. Repeated paths appear only once in the dictionary.
        """
        paths = (coerce_path(path) for path in self.text_paths)
        return {str(path): path for path in paths}

    def resolve_text_fields(self, names: list[str] | None) -> list[CosmosPath]:
        """Select configured full paths, or the sole text field if none are given."""
        mapping = self.text_field_map()
        if not names:
            if len(mapping) == 1:
                return [next(iter(mapping.values()))]
            if not mapping:
                return []
            raise UnknownField(
                "multiple text fields are available; specify one or more full paths: "
                f"{sorted(mapping)}"
            )
        paths: list[CosmosPath] = []
        for name in names:
            if name not in mapping:
                raise UnknownField(
                    f"unknown text field path {name!r}; available: {sorted(mapping)}"
                )
            paths.append(mapping[name])
        return paths
