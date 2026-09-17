"""Map stored Cosmos DB fields to the names the query compiler understands.

Different corpora can store the same information under different field names.
CorpusSchema tells the compiler where to find item IDs, text, and optional
document, chunk, title, source, and metadata fields. Only item_id_path is
required to construct this schema; text_paths defaults to an empty list.

An item is one JSON record stored in Cosmos DB. A source document, such as a
report, may be split into chunks stored as separate items. Each chunk item has
its own item ID and may share a source document ID with the other chunks.
A chunk ID identifies a particular piece; its chunk order specifies its
position in the source document. If item IDs also identify chunks, both ID
paths can point to the same field.

These settings are field locations, not ID values. For example,
document_id_path="/docid" names the field containing an ID such as "report-7".
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from cosmos_agentic_retriever.query_engine.paths import CosmosPath, coerce_path

PathField = Annotated[CosmosPath, BeforeValidator(coerce_path)]


class CorpusSchema(BaseModel):
    # ID of the stored Cosmos item (usually /id); one chunk can be one item.
    item_id_path: PathField
    # Text fields to include in query results, e.g. ["/title", "/content/text"].
    text_paths: list[PathField] = Field(default_factory=list)
    # Source document ID shared by its chunks, e.g. /docid for document lookup.
    document_id_path: PathField | None = None
    # Individual chunk ID, e.g. /chunk_id; may use the same path as item_id_path.
    chunk_id_path: PathField | None = None
    # Chunk's position within its source document, e.g. /chunk_idx.
    chunk_order_path: PathField | None = None
    # Human-readable title to return with the result, e.g. /title.
    title_path: PathField | None = None
    # Source reference or label to return, e.g. /url or /source_type.
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
