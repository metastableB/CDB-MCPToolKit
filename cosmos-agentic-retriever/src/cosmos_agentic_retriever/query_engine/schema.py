"""Tell the query compiler where IDs and text are stored in each JSON record.

A Cosmos DB item is one stored JSON record. For example:
        {"id": "report-7-0", "docid": "report-7",
         "content": {"text": "Battery recycling..."}}

A field path describes a location inside that record: /id means the top-level
id field, and /content/text means the text field inside the content object.
It is not a file path or the value stored in the field.

For the item above, configure CorpusSchema with:
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
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from cosmos_agentic_retriever.query_engine.paths import CosmosPath, coerce_path

PathField = Annotated[CosmosPath, BeforeValidator(coerce_path)]


class CorpusSchema(BaseModel):
    # Path to the item's identifier field, e.g. /id, not an ID like "report-7-0".
    item_id_path: PathField
    # Paths to text fields to return, e.g. ["/title", "/content/text"], not text values.
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

    @staticmethod
    def _seg_name(path: CosmosPath) -> str:
        return path.segments[-1]

    def text_field_map(self) -> dict[str, CosmosPath]:
        out: dict[str, CosmosPath] = {}
        for p in self.text_paths:
            name = self._seg_name(p)
            if name in out and str(out[name]) != str(p):
                name = str(p)
            out[name] = p
        return out
