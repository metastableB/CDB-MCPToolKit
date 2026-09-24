"""Read-only Cosmos comparison. Opt in only after the PR5 fixtures are ready."""

import json
import os

import pytest

from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
from cosmos_agentic_retriever.query_engine.paths import CosmosPath
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema


@pytest.mark.skipif(
    os.environ.get("RUN_COSMOS_STOPWORD_LIVE") != "1",
    reason="requires an explicitly enabled Cosmos fixture account",
)
def test_stopword_policy_on_cosmos() -> None:
    from azure.cosmos import CosmosClient
    from azure.identity import AzureCliCredential

    endpoint = os.environ["COSMOS_TEST_ENDPOINT"]
    database = os.environ.get("COSMOS_TEST_DATABASE", "mcp-live-tests-v1")
    assert database.startswith("mcp-live-tests-")
    compiler = CosmosQueryCompiler(
        CorpusSchema(item_id_path="/id", text_paths=["/text"])
    )
    with (
        AzureCliCredential() as credential,
        CosmosClient(endpoint, credential=credential) as client,
    ):
        container = client.get_database_client(database).get_container_client("flat-v1")
        properties = container.read()
        assert properties["partitionKey"]["paths"] == ["/tenant"]
        assert {"path": "/text", "language": "en-US"} in properties["fullTextPolicy"][
            "fullTextPaths"
        ]
        assert {"path": "/text"} in properties["indexingPolicy"]["fullTextIndexes"]
        records = list(
            container.query_items(
                query="SELECT TOP 4 c.id, c.tenant, c.text, c.fixture_version FROM c",
                enable_cross_partition_query=True,
            )
        )
        expected = [
            {
                "id": item_id,
                "tenant": tenant,
                "text": f"battery recycling {canary}",
                "fixture_version": "mcp-search-v1",
            }
            for item_id, tenant, canary in (
                ("shared", 0, "canaryalpha"),
                ("first-only", 0, "canaryalpha"),
                ("other-only", "other", "canarybeta"),
            )
        ]
        assert sorted(records, key=lambda item: item["id"]) == sorted(
            expected, key=lambda item: item["id"]
        )

        def search(query: str) -> list[str]:
            compiled = compiler.compile_full_text(
                query=query,
                limit=3,
                ignored_item_ids=[],
                filters=[],
                partition_key=None,
                cross_partition=True,
                text_paths=[CosmosPath.parse("/text")],
            )
            return [
                row["item_id"]
                for row in container.query_items(
                    query=compiled.sql,
                    parameters=compiled.parameters,
                    enable_cross_partition_query=True,
                )
            ]

        baseline = search("canarybeta")
        mixed = search("the canarybeta and")
        assert baseline[0] == mixed[0] == "other-only"
        assert set(baseline) == set(mixed)

        observations = {
            "baseline": baseline,
            "with_stopwords": mixed,
            "all_stopwords": search("the and of"),
            "contraction": search("don't canarybeta"),
        }
        assert all(
            set(ids) <= {item["id"] for item in expected}
            for ids in observations.values()
        )
        print(json.dumps(observations, sort_keys=True))
