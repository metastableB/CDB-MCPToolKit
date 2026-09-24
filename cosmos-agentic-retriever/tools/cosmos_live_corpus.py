"""Load a checksum-pinned, bounded SciFact subset without network or Azure calls.

Source: https://github.com/beir-cellar/beir (SciFact archive, March 2021).
The archive stays local. No dataset text is bundled with the application.
"""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from cosmos_live_fixtures import ContainerFixture, fixtures

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
SCIFACT_CONTAINER = "scifact-100-v1"
DOCUMENT_COUNT = 100
QUERY_COUNT = 5
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class CorpusQuery:
    query_id: str
    text: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class RealCorpus:
    fixture: ContainerFixture
    queries: tuple[CorpusQuery, ...]
    selection_sha256: str


def load_scifact(path: Path) -> RealCorpus:
    """Keep all positives for five test queries, then fill to 100 by numeric ID.

    This relevance-enriched subset is an integration check, not the full BEIR
    benchmark. Changing the selection or mapping requires a new container version.
    """
    with path.open("rb") as source:
        archive_bytes = source.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("SciFact archive exceeds the 8 MiB input limit")
    if hashlib.sha256(archive_bytes).hexdigest() != SCIFACT_SHA256:
        raise ValueError("SciFact archive SHA-256 mismatch; use the pinned source")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:

        def read_member(name: str) -> str:
            member = archive.getinfo(f"scifact/{name}")
            if member.file_size > MAX_MEMBER_BYTES:
                raise ValueError("SciFact archive member exceeds the 16 MiB limit")
            return archive.read(member).decode("utf-8")

        corpus = {
            record["_id"]: record
            for line in read_member("corpus.jsonl").splitlines()
            if (record := json.loads(line))
        }
        queries = {
            record["_id"]: record["text"]
            for line in read_member("queries.jsonl").splitlines()
            if (record := json.loads(line))
        }
        qrels: dict[str, set[str]] = {}
        for row in csv.DictReader(
            io.StringIO(read_member("qrels/test.tsv")), delimiter="\t"
        ):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    query_ids = sorted(qrels, key=int)[:QUERY_COUNT]
    if len(query_ids) != QUERY_COUNT or len(corpus) < DOCUMENT_COUNT:
        raise ValueError("SciFact input has fewer documents or queries than expected")
    selected_ids = set().union(*(qrels[query_id] for query_id in query_ids))
    if len(selected_ids) > DOCUMENT_COUNT or not selected_ids <= corpus.keys():
        raise ValueError("SciFact relevance labels do not fit the selected corpus")
    for item_id in sorted(corpus, key=int):
        if len(selected_ids) == DOCUMENT_COUNT:
            break
        selected_ids.add(item_id)
    items = tuple(
        {
            "id": item_id,
            "tenant": "scifact",
            "title": corpus[item_id]["title"],
            "text": corpus[item_id]["text"],
            "source": "beir-scifact",
            "source_sha256": SCIFACT_SHA256,
        }
        for item_id in sorted(selected_ids, key=int)
    )
    selected_queries = tuple(
        CorpusQuery(query_id, queries[query_id], frozenset(qrels[query_id]))
        for query_id in query_ids
    )
    selection = {
        "items": items,
        "queries": [
            {
                "id": query.query_id,
                "text": query.text,
                "relevant_ids": sorted(query.relevant_ids, key=int),
            }
            for query in selected_queries
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            selection, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return RealCorpus(
        ContainerFixture(
            SCIFACT_CONTAINER,
            {
                "item_id_path": "/id",
                "text_paths": ["/title", "/text"],
                "metadata_paths": {"source": "/source"},
            },
            ("/title", "/text"),
            ("/title", "/text"),
            items,
        ),
        selected_queries,
        digest,
    )


def select_data(
    data: str, archive: Path | None
) -> tuple[tuple[ContainerFixture, ...], RealCorpus | None]:
    """Select explicit synthetic, real, or combined containers for setup/tests."""
    if data not in ("synthetic", "scifact", "both"):
        raise ValueError("data must be synthetic, scifact, or both")
    if data == "synthetic":
        if archive is not None:
            raise ValueError("select --data scifact or both when supplying an archive")
        return fixtures(), None
    if archive is None:
        raise ValueError(
            "SciFact data requires a local --scifact-archive / COSMOS_TEST_SCIFACT_ARCHIVE"
        )
    real = load_scifact(archive)
    synthetic = fixtures() if data == "both" else ()
    return (*synthetic, real.fixture), real


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the pinned SciFact subset locally. No Azure calls."
    )
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    real = load_scifact(args.archive)
    print(
        json.dumps(
            {
                "source": SCIFACT_URL,
                "source_sha256": SCIFACT_SHA256,
                "container": real.fixture.name,
                "document_count": len(real.fixture.items),
                "document_ids": [item["id"] for item in real.fixture.items],
                "queries": [
                    {
                        "id": query.query_id,
                        "text": query.text,
                        "relevant_ids": sorted(query.relevant_ids, key=int),
                    }
                    for query in real.queries
                ],
                "selection_sha256": real.selection_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
