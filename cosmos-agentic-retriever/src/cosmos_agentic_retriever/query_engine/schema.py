"""Describe where the query compiler should find fields in Cosmos DB items.

Different corpora can store the same information under different field names.
For example, document_id_path="/source/document_id" tells the compiler where
to find the source document ID. metadata_paths={"year": "/publication/year"}
lets a filter refer to "year" while the SQL uses c["publication"]["year"].

CorpusSchema records paths for the item ID, text fields, and optional document,
chunk, title, source, and metadata fields. PathField converts supplied path
strings into CosmosPath objects; existing CosmosPath objects are also accepted.

text_field_map() names each text field using the last part of its path. If a
later path has the same name, it uses that full path as the name instead. The
compiler uses this map to label the text fields returned by a query.

This schema describes the caller's field layout, not a schema imposed by
Cosmos DB. It does not change stored documents or check whether fields exist.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from cosmos_agentic_retriever.query_engine.paths import CosmosPath, coerce_path

PathField = Annotated[CosmosPath, BeforeValidator(coerce_path)]


class CorpusSchema(BaseModel):
    item_id_path: PathField
    text_paths: list[PathField] = Field(default_factory=list)
    document_id_path: PathField | None = None
    chunk_id_path: PathField | None = None
    chunk_order_path: PathField | None = None
    title_path: PathField | None = None
    source_path: PathField | None = None
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
