# cosmos-agentic-retriever

This submodule is the HTTP service that backs the Azure Cosmos DB MCP Toolkit's
`agentic_search` tool. Given a natural-language query, it searches a set of
explicitly configured Cosmos DB for NoSQL containers and returns a ranked set of
relevant documents.

## Setting up and Running the Service

Install into your Python environment:

```bash
python -m pip install .
```

Configure the account, database, and containers through environment variables,
then start the HTTP service. It only reads your existing containers — it never
creates or changes databases, containers, indexes, or documents.

```bash
export ACCOUNT_URI='https://YOUR-ACCOUNT.documents.azure.com:443/'
export COSMOS_DATABASE='YOUR-DATABASE'
export COSMOS_CONTAINERS='{"articles":{"cosmos_schema":{"item_id_path":"/id","partition_key_paths":["/tenant"],"text_paths":["/text"]}}}'
export COSMOS_CREDENTIAL='azure_cli'
az login
python -m cosmos_agentic_retriever serve
```

Replace the `YOUR-*` values with your own. `COSMOS_CONTAINERS` is a JSON object
keyed by container name; each entry declares that container's schema and search
scope. 

The signed-in identity needs Cosmos data-plane permission to query the
containers, and the configured text paths must already have a full-text policy
and index. The HTTP API has no caller authentication and is intended to be bound
loopback (local) or behind an authenticated private gateway.

## Connect the MCP toolkit

Run the service, then point the .NET toolkit at it:

```bash
export COSMOS_RETRIEVER_URL='http://127.0.0.1:9000'
export COSMOS_RETRIEVER_TIMEOUT_S=600
```

Call `agentic_search` without `schemaOverride` (or with `"none"`); the schema is
configured at service startup, not per request.

## Tests

Install the development extras (test dependencies) and run the unit tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/unit -q
```

Unit tests use fake SDK clients and make no cloud calls. For live validation
against a real account, follow the [live-test guide](tests/end_to_end/README.md).
