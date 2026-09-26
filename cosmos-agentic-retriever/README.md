# cosmos-agentic-retriever

This submodule is the HTTP service that backs the Azure Cosmos DB MCP Toolkit's
`agentic_search` tool. Given a natural-language question, it searches a set of
explicitly configured Cosmos DB for NoSQL containers and returns an answer
together with the ranked documents that support it.

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

### Configure with a YAML file

For more than a container or two, set `COSMOS_RETRIEVER_CONFIG` to a YAML file
instead of packing every container into the `COSMOS_CONTAINERS` variable:

```yaml
account_uri: https://YOUR-ACCOUNT.documents.azure.com:443/
cosmos_database: YOUR-DATABASE
cosmos_credential: azure_cli
cosmos_containers:
  articles:
    cosmos_schema:
      item_id_path: /id
      partition_key_paths: ['/tenant']
      text_paths: ['/text']
```

```bash
export COSMOS_RETRIEVER_CONFIG='./config.yaml'
az login
python -m cosmos_agentic_retriever serve
```

Values in the file win; any field it omits falls back to that field's environment
variable, so keep secrets like `COSMOS_KEY` in the environment — or use
`azure_cli` credentials and keep no secrets at all.

## Agentic search

The service can answer a natural-language question instead of running a single
search. It runs a short loop: it asks a language model what to search for, runs
the search, reads the results, and searches again until it can answer or a turn
cap is reached. Bring your own model — any OpenAI-compatible chat-completions
endpoint works (OpenAI, Azure OpenAI, a local vLLM server, or another gateway).

Point the service at your model before starting it:

```bash
export LLM_BASE_URL='https://YOUR-ENDPOINT/v1'
export LLM_MODEL='YOUR-MODEL'
export LLM_API_KEY='YOUR-KEY'          # only if your endpoint requires one
```

Then ask a question:

```bash
curl --fail-with-body http://127.0.0.1:9000/agent_search \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does battery recycling work?"}'
```

The response has the model's `answer`, the ranked `documents` that support it —
pooled from the searches the agent ran, deduplicated, each with its
`retrieval_id` and source container — and how the loop ended (`terminal_reason`
is `stop`, `max_turns`, or `error`; `turns` counts the model calls made). Set
`AGENT_MAX_TURNS` (default 6) to bound the loop and `AGENT_MAX_DOCUMENTS`
(default 10) to cap items per search and in the returned set. Without
`LLM_BASE_URL` and `LLM_MODEL` the service still serves plain search, but
`/agent_search` returns 503.

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
