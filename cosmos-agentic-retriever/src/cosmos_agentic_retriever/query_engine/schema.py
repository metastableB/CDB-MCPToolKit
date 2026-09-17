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
