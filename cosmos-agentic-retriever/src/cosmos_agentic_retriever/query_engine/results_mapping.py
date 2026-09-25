"""Replace internal SQL column names and build RetrievedItem results.

The compiler writes SQL such as c["article"]["text"] AS txt_0, so Cosmos returns
{"txt_0": "hello"}. The compiler also supplies a mapping from "txt_0" to
"/article/text". This module uses that mapping to label the returned text with
its configured path. Metadata columns such as md_0 become configured names
such as "year".

rows_to_items puts these values into RetrievedItem objects, along with IDs,
display text, and result positions. It does not rebuild the original nested
document, query Cosmos, or change the result order.

For example:
    >>> rows = [{"item_id": "item-1", "txt_0": "Battery recycling", "md_0": 2024}]
    >>> aliases = {"txt_0": "/article/text", "md_0": "year"}
    >>> item = rows_to_items(rows, strategy="full_text", projected_aliases=aliases)[0]
    >>> item.text_fields
    {'/article/text': 'Battery recycling'}
    >>> item.metadata
    {'year': 2024}
"""

from __future__ import annotations

from typing import Any

from cosmos_agentic_retriever.query_engine.types import (
    CosmosItemIdentity,
    RetrievedItem,
)


def row_text_fields(row: dict[str, Any], aliases: dict[str, str]) -> dict[str, str]:
    """Replace one row's txt_ column names with their configured text paths.

    For example, {"txt_0": "hello"} with {"txt_0": "/body"} becomes
    {"/body": "hello"}. Ignore columns without both a txt_ prefix and a mapping.
    None and other false values become empty strings. Other values pass through.
    """
    out: dict[str, str] = {}
    for key, value in row.items():
        if key.startswith("txt_") and key in aliases:
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
    """Replace SQL column names and package each row as a RetrievedItem.

    projected_aliases maps txt_ columns to full paths and md_ columns to metadata
    names. Unmapped columns with these prefixes are ignored. queried_text_fields
    chooses which paths appear in item.text; item.text_fields retains all mapped
    text fields. Copy strategy and channels as labels describing the search.

    IDs become strings. A missing or null item_id raises ValueError; missing
    document_id and chunk_id stay None. Keep chunk_order only if it is an integer,
    excluding booleans. Rank is the row's position plus start_rank (zero by default),
    not a relevance score calculated here.

    When partition paths were configured, _cosmos_identity must contain the
    physical id and partition values. Keep them separate from logical item_id.
    """
    aliases = projected_aliases or {}
    items: list[RetrievedItem] = []
    for index, row in enumerate(rows):
        if row.get("item_id") is None:
            raise ValueError("query result must contain a non-null item_id")
        identity = (
            CosmosItemIdentity.model_validate(row.get("_cosmos_identity"))
            if "_cosmos_identity" in aliases
            else None
        )
        metadata = {
            aliases[key]: value
            for key, value in row.items()
            if key.startswith("md_") and key in aliases
        }
        text_fields = row_text_fields(row, aliases)
        display = assemble_text(text_fields, queried_text_fields)
        chunk_order = row.get("chunk_order")
        items.append(
            RetrievedItem(
                item_id=str(row.get("item_id")),
                cosmos_identity=identity,
                document_id=(
                    str(row["document_id"])
                    if row.get("document_id") is not None
                    else None
                ),
                chunk_id=(
                    str(row["chunk_id"]) if row.get("chunk_id") is not None else None
                ),
                chunk_order=chunk_order if type(chunk_order) is int else None,
                text=display,
                text_fields=text_fields,
                title=row.get("title"),
                source=row.get("source"),
                metadata=metadata,
                retrieval_strategy=strategy,
                retrieval_channels=list(channels or []),
                rank=start_rank + index,
            )
        )
    return items
