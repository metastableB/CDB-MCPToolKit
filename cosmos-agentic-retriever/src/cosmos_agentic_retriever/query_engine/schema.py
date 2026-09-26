"""Tell the query compiler where IDs and text are stored in each JSON record.

Retrieval needs to identify each stored item and locate the text to search and
return. item_id_path gives the item's identity; text_paths give the content
fields. Field names and structure vary by container, so CorpusSchema is a
configurable BaseModel that maps a container's actual layout onto what the
compiler needs.

Usage example:
Consider the following Cosmos DB item stored as a JSON record.

        {"id": "report-7-0", "docid": "report-7",
         "content": {"text": "Battery recycling..."}}

For the item above, a CorpusSchema can be:

```python
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema

schema = CorpusSchema(
    item_id_path="/id",
    parent_document_id_path="/docid",
    text_paths=["/content/text"]
)
```

Chunked documents: A source document, such as a report, may be split into chunks
stored as separate items. parent_document_id_path identifies the field holding
the parent document's ID, shared by all of that document's chunks (e.g. /docid);
it lets the service gather a whole document from its chunks. chunk_id_path
identifies the field holding a chunk's own ID; it can be /id when the item ID
already identifies the chunk. chunk_order_path identifies the field holding that
chunk's position in the document, such as /chunk_idx.

Partition keys: partition_key_paths lists where the container's partition-key
field(s) live, in Cosmos's declared order (e.g. ["/tenant"], or ["/tenant",
"/region"] for a hierarchical key). A Cosmos id is unique only within a
partition, so an item's true physical identity is (partition-key values, id).
The compiler projects that identity alongside your logical item_id_path. The
HTTP service needs it to de-duplicate items that share an id across partitions,
so it requires these paths when querying across partitions.

Unknown settings are rejected. Replacing a field validates its new value;
in-place list or dictionary edits are checked again before query compilation.
additional_return_paths lists extra fields to return alongside the text; each is
keyed by its path in the result, not by a label. Filters also use stored paths.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

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
    # Note: (partition-key value, physical /id) together uniquely identify an item;
    # /id alone does not guarantee uniqueness across partitions.
    partition_key_paths: list[PathField] = Field(default_factory=list, max_length=3)
    # Paths to text fields to return, e.g. ["/title", "/content/text"].
    text_paths: list[PathField] = Field(default_factory=list)
    # Chunked documents: The following are only required when documents are chunked.
    # Path to the parent document ID shared by its chunks, e.g. /docid.
    parent_document_id_path: PathField | None = None
    # Path to a chunk's ID, e.g. /chunk_id; may be the same as item_id_path.
    chunk_id_path: PathField | None = None
    # Path to a chunk's position within its source document, e.g. /chunk_idx.
    chunk_order_path: PathField | None = None
    # Extra fields to return alongside text, keyed by path in the result,
    # e.g. ["/publication/year", "/url"].
    additional_return_paths: list[PathField] = Field(default_factory=list)

    @field_validator("partition_key_paths", "additional_return_paths", "text_paths")
    @classmethod
    def _reject_duplicate_paths(
        cls, paths: list[CosmosPath], info: ValidationInfo
    ) -> list[CosmosPath]:
        if len(set(paths)) != len(paths):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return paths

    def text_paths_by_string(self) -> dict[str, CosmosPath]:
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
        or read documents. Duplicate paths are rejected by the field validator.
        """
        paths = (coerce_path(path) for path in self.text_paths)
        return {str(path): path for path in paths}

    def additional_paths_by_string(self) -> dict[str, CosmosPath]:
        """Return additional_return_paths keyed by path string.

        Uniqueness is enforced by the field validator, so paths are already
        distinct here. Keys are the same path strings used elsewhere.
        """
        paths = (coerce_path(path) for path in self.additional_return_paths)
        return {str(path): path for path in paths}

    def resolve_text_fields(self, names: list[str] | None) -> list[CosmosPath]:
        """Select configured full paths, or the sole text field if none are given."""
        mapping = self.text_paths_by_string()
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
