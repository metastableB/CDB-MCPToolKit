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

partition_key_paths declares the actual container key paths in order. The compiler
projects those values and physical /id separately from item_id_path. The HTTP
service requires these paths to distinguish items across partitions. A standalone
compiler schema may omit them, but its rows cannot be combined by physical identity.

Unknown settings are rejected. Replacing a field validates its new value;
in-place list or dictionary edits are checked again before query compilation.
Metadata names label fields inside the result's metadata object. They may match
standard result names such as source. Filters use stored paths, not these names.

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
    # Container partition-key paths in their declared order, e.g. ["/tenant"].
    partition_key_paths: list[PathField] = Field(default_factory=list, max_length=3)
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
    # Extra fields to return, e.g. {"year": "/publication/year"}.
    metadata_paths: dict[str, PathField] = Field(default_factory=dict)

    @field_validator("partition_key_paths")
    @classmethod
    def _unique_partition_paths(cls, paths: list[CosmosPath]) -> list[CosmosPath]:
        if len(set(paths)) != len(paths):
            raise ValueError("partition_key_paths must not contain duplicates")
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
