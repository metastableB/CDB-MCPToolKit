"""Convert document field paths into Cosmos DB SQL field references.

For example, /document/title refers to the title field inside an item's document
object. The query compiler needs that location written as c["document"]["title"]
in SQL.

- CosmosPath.parse("/document/title") checks the path and stores its parts as
    ("document", "title"). Passing an existing CosmosPath returns it unchanged.
- path.render() produces c["document"]["title"]. A different SQL table alias
    can be supplied instead of c; quotes and backslashes in field names are escaped.
- str(path) converts the stored parts back to /document/title.
- coerce_path(value) accepts either a path string or an existing CosmosPath,
    allowing schema fields to accept both forms.

Creating a CosmosPath directly also requires at least one field name and
rejects empty field names. A slash inside a quoted name is part of that name:
/"document/title" refers to c["document/title"], while /document/title refers
to c["document"]["title"]. str(path) quotes names when needed to preserve this
distinction, and parse() accepts those quoted names.

When given a string, CosmosPath.parse() checks these rules:
- Start the path with /, as in /document/title. document/title is rejected.
- Include at least one field name. A path containing only / is rejected.
- Put a field name between slashes. /document//title is rejected.
- Do not end the path with /. /document/title/ is rejected.
- Start each unquoted field name with A-Z, a-z, or an underscore (_).
- After the first character, use only those letters, underscores, numbers,
  spaces, dots, or hyphens.

Names outside the unquoted format must use double quotes, such as /"2020_sales".
Quoted names use JSON string escaping for quotes and backslashes. This helper
accepts individual field paths, not indexing wildcards such as /* or /[]/?.
It does not interpret JSON Pointer escapes (~0 and ~1) used by Cosmos Patch.

TODO: Check whether these field-name restrictions are necessary and compatible
with the paths accepted by other MCP Toolkit tools.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from cosmos_agentic_retriever.query_engine.types import UnsafeCosmosPathError

_ALLOWED_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_ .\-]*$")


class CosmosPath(BaseModel):
    model_config = ConfigDict(frozen=True)

    segments: tuple[str, ...]

    @field_validator("segments")
    @classmethod
    def _validate_segments(cls, segments: tuple[str, ...]) -> tuple[str, ...]:
        if not segments or any(not segment for segment in segments):
            raise UnsafeCosmosPathError("path and field names must not be empty")
        return segments

    @classmethod
    def parse(cls, raw: str | CosmosPath) -> CosmosPath:

        if isinstance(raw, CosmosPath):
            return raw
        if not isinstance(raw, str):
            raise UnsafeCosmosPathError(
                f"path must be a string, got {type(raw).__name__}"
            )
        if not raw.startswith("/"):
            raise UnsafeCosmosPathError(f"path must start with '/': {raw!r}")
        if len(raw) < 2 or raw.endswith("/"):
            raise UnsafeCosmosPathError(f"path is empty or has a trailing '/': {raw!r}")

        segments: list[str] = []
        decoder = json.JSONDecoder()
        position = 1
        while position < len(raw):
            if raw[position] == '"':
                try:
                    segment, position = decoder.raw_decode(raw, position)
                except json.JSONDecodeError as error:
                    raise UnsafeCosmosPathError(
                        f"invalid quoted field name in {raw!r}"
                    ) from error
                if position < len(raw) and raw[position] != "/":
                    raise UnsafeCosmosPathError(
                        f"expected '/' after quoted field name in {raw!r}"
                    )
            else:
                end = raw.find("/", position)
                if end == -1:
                    end = len(raw)
                segment = raw[position:end]
                if not _ALLOWED_SEGMENT.fullmatch(segment):
                    raise UnsafeCosmosPathError(
                        f"unsafe path segment {segment!r} in {raw!r}"
                    )
                position = end
            segments.append(segment)
            position += 1
        return cls(segments=tuple(segments))

    def render(self, alias: str = "c") -> str:

        out = alias
        for seg in self.segments:
            out += f"[{json.dumps(seg, ensure_ascii=False)}]"
        return out

    def __str__(self) -> str:
        return "/" + "/".join(
            segment
            if _ALLOWED_SEGMENT.fullmatch(segment)
            else json.dumps(segment, ensure_ascii=False)
            for segment in self.segments
        )


def coerce_path(value: Any) -> CosmosPath:

    if isinstance(value, CosmosPath):
        return value
    return CosmosPath.parse(value)
