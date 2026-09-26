"""Turn raw rows returned by Cosmos query into RetrievedItem results.

CosmosQueryCompiler projects (encodes) columns under safe SQL aliases, so Cosmos
returns rows keyed by those aliases: For example, a text field "/article/text"
might be encoded as txt_0 and it will therefore be returned as txt_0.  This
module is the reverse of that projection. It uses the alias mapping the compiler
maintains internally to put each value back under its configured path and
packages the row as a RetrievedItem: text fields keyed by path, extra fields
keyed by path, the physical Cosmos identity, and the result's rank. It does not
query Cosmos, rebuild the original nested document, or change the result order.

For example:
    >>> rows = [{"item_id": "item-1", "txt_0": "Battery recycling", "add_0": 2024}]
    >>> aliases = {"txt_0": "/article/text", "add_0": "/publication/year"}
    >>> item = rows_to_items(rows, strategy="full_text", projected_aliases=aliases)[0]
    >>> item.text_fields
    {'/article/text': 'Battery recycling'}
    >>> item.additional_fields
    {'/publication/year': 2024}
"""

from __future__ import annotations

from typing import Any

from cosmos_agentic_retriever.query_engine.types import (
    CosmosItemIdentity,
    RetrievedItem,
    SQLColumnAliases,
)


def row_text_fields(row: dict[str, Any], aliases: dict[str, str]) -> dict[str, str]:
    """Decode a row's txt_ columns back to their configured text paths.

    For example, {"txt_0": "hello"} with {"txt_0": "/body"} becomes
    {"/body": "hello"}. Ignore columns without both a txt_ prefix and a mapping.
    None and other false values become empty strings. Other values pass through.
    """
    out: dict[str, str] = {}
    for key, value in row.items():
        if key.startswith(SQLColumnAliases.TEXT_PREFIX) and key in aliases:
            out[aliases[key]] = value or ""
    return out


def assemble_text(text_fields: dict[str, str], names: list[str] | None = None) -> str:
    """Combine selected text fields from one item into a display string.

    names contains full paths to include, in the requested order. None or an
    empty list includes all fields in dictionary order. Missing paths are skipped.
    One field returns just its text. Multiple fields get headings like [/body],
    separated by blank lines. No selected fields returns an empty string.
    This combines fields within an item, not chunks from separate items.
    """
    selected = (
        [name for name in names if name in text_fields] if names else list(text_fields)
    )
    if not selected:
        return ""
    if len(selected) == 1:
        return text_fields.get(selected[0], "") or ""
    return "\n\n".join(
        f"[{name}]\n{text_fields.get(name, '') or ''}" for name in selected
    )


def rows_to_items(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    channels: list[str] | None = None,
    start_rank: int = 0,
    projected_aliases: dict[str, str] | None = None,
    queried_text_fields: list[str] | None = None,
) -> list[RetrievedItem]:
    """Decode each Cosmos row into a RetrievedItem, keyed by configured paths.

    projected_aliases maps txt_ columns to their text paths and add_ columns to
    their configured paths; columns with those prefixes but no mapping are
    ignored. queried_text_fields chooses which paths appear in item.text, while
    item.text_fields retains every mapped text field. strategy and channels are
    copied through as labels describing the search.

    IDs are coerced to strings. A missing or null item_id raises ValueError; a
    missing parent_document_id or chunk_id stays None. chunk_order is kept only
    when it is an integer (booleans excluded). rank is the row's position plus
    start_rank, not a relevance score computed here.

    When partition paths were configured, the _cosmos_identity column must hold
    the physical id and partition values; it is kept separate from the logical
    item_id.
    """
    aliases = projected_aliases or {}
    items: list[RetrievedItem] = []
    for index, row in enumerate(rows):
        if row.get(SQLColumnAliases.ITEM_ID) is None:
            raise ValueError("query result must contain a non-null item_id")
        identity = (
            CosmosItemIdentity.model_validate(row.get(SQLColumnAliases.COSMOS_IDENTITY))
            if SQLColumnAliases.COSMOS_IDENTITY in aliases
            else None
        )
        additional_fields = {
            aliases[key]: value
            for key, value in row.items()
            if key.startswith(SQLColumnAliases.ADDITIONAL_PREFIX) and key in aliases
        }
        text_fields = row_text_fields(row, aliases)
        display = assemble_text(text_fields, queried_text_fields)
        chunk_order = row.get(SQLColumnAliases.CHUNK_ORDER)
        items.append(
            RetrievedItem(
                item_id=str(row.get(SQLColumnAliases.ITEM_ID)),
                cosmos_identity=identity,
                parent_document_id=(
                    str(row[SQLColumnAliases.PARENT_DOCUMENT_ID])
                    if row.get(SQLColumnAliases.PARENT_DOCUMENT_ID) is not None
                    else None
                ),
                chunk_id=(
                    str(row[SQLColumnAliases.CHUNK_ID])
                    if row.get(SQLColumnAliases.CHUNK_ID) is not None
                    else None
                ),
                chunk_order=chunk_order if type(chunk_order) is int else None,
                text=display,
                text_fields=text_fields,
                additional_fields=additional_fields,
                retrieval_strategy=strategy,
                retrieval_channels=list(channels or []),
                rank=start_rank + index,
            )
        )
    return items
