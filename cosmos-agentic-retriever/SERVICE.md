# Run the Search Service

Install from this directory into your Python environment:

```bash
python -m pip install -e '.[dev]'
```

Configure existing Cosmos DB for NoSQL containers in one account and database. No databases, containers,
indexes, or documents are created or changed by the service.

```bash
export ACCOUNT_URI='https://YOUR-ACCOUNT.documents.azure.com:443/'
export COSMOS_DATABASE='YOUR-DATABASE'
export COSMOS_CONTAINERS='{"articles":{"cosmos_schema":{"item_id_path":"/id","text_paths":["/text"]}},"reports":{"cosmos_schema":{"item_id_path":"/id","text_paths":["/content/body"]}}}'
export COSMOS_CREDENTIAL='azure_cli'
az login
python -m cosmos_agentic_retriever serve
```

The signed-in identity needs Cosmos data-plane permission to query the containers.
Azure subscription Contributor alone is not sufficient. The selected text paths
must already have compatible full-text policy and indexing configured.

## Settings

Settings are read from the process environment at startup, not from a registry or
automatically discovered `.env` file. Restart the service to change configuration.

- `ACCOUNT_URI`, `COSMOS_DATABASE`, and `COSMOS_CONTAINERS` are required.
- `COSMOS_CONTAINERS` is a JSON object keyed by container name. This is the allowed
  search scope; containers are not discovered automatically. All use full-text search.
- Each entry requires `cosmos_schema`, using the existing `CorpusSchema` fields.
  Its text paths identify returned fields, not necessarily fields to search.
- Each entry can set `search_text_fields`, a list such as `["/content/body"]`.
  Supply it when the schema contains multiple text fields. Select only indexed fields.
- Each entry can set `partition_key` and `partition_policy`, for example
  `"partition_key":0,"partition_policy":{"allow_cross_partition_search":false}`.
  Without a key, cross-partition search defaults to allowed. Disabling it requires
  a key. Keys such as `0` and `""` are valid and are not replaced by defaults.
  These rules apply independently within each container, not across containers.
- `COSMOS_CREDENTIAL=azure_cli` uses the Azure CLI login. `default` uses
  `DefaultAzureCredential`, including managed identity where available.
- `COSMOS_KEY`, if supplied securely through the environment, takes precedence
  over identity authentication. Do not put keys in source files or command arguments.
- `QUERY_ENGINE` optionally contains JSON settings, for example
  `{"max_concurrency":4,"slow_query_warning_seconds":4.5}`. The shared executor
  limits simultaneous Cosmos queries across containers and concurrent HTTP calls.
  Each request also limits its fan-out worker count to this value. It does not
  impose a request deadline or limit the total number of waiting HTTP requests.
- `HOST`, `PORT`, and `LOG_LEVEL` default to `127.0.0.1`, `9000`, and `info`.
  `serve --host ... --port ...` overrides the first two.

The HTTP API has no caller authentication. Keep it on loopback or behind an
authenticated private gateway. The Cosmos credential authenticates the service
to Cosmos, not callers to this API. Do not expose the port publicly.

## Request and Response

```bash
curl --fail-with-body http://127.0.0.1:9000/health
curl --fail-with-body http://127.0.0.1:9000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"battery recycling","maxDocuments":5}'
```

`POST /search` accepts `query`, `maxDocuments` (1-50, default 20), and optional
`database` and `container`. Omit `container` to search all configured containers,
or name one to restrict the search. Omitted `database` uses the configured database.
Other databases, unconfigured containers, and `*` are rejected. Nonempty `overrides`
objects are rejected. The old single-container environment settings are replaced
by `COSMOS_CONTAINERS`; there is no implicit fallback to those settings.

The response contains `documents`, `searched`, `errors`, and `partial`:

```json
{
  "documents": [],
  "searched": [{"database":"YOUR-DATABASE","container":"articles"}],
  "errors": [{"database":"YOUR-DATABASE","container":"reports","error":"Search failed."}],
  "partial": true
}
```

Every item includes `database`, `container`, and `retrieval_id`, alongside `RetrievedItem` fields:
`item_id`, `text`, `text_fields`, metadata, optional document/chunk/title/source
fields, and a zero-based `rank`. Text-field keys are full paths such as `/text`.
`item_id` remains the original Cosmos ID. Use `retrieval_id` to distinguish items
across containers: it is `database/container:item_id` with each part percent-escaped.
Source fields are separate from metadata so they cannot overwrite stored metadata.
Entries are stored items, potentially passages, not assembled documents.

Cosmos ranks each container separately. Python combines those lists using reciprocal
rank: A1, B1, A2, B2, with ties following configuration order. This is not a global
BM25 relevance comparison. Each container fetches up to `maxDocuments` items and
the combined response also has at most `maxDocuments`. There is no LLM loop.

Successful empty searches appear in `searched` with an empty documents list.
If some targets fail, HTTP200 returns successful results with `partial:true` and
sanitized per-target errors. If all targets fail, HTTP500 includes a top-level
`error` plus the per-target errors; it never looks like successful empty search.

Invalid request bodies return 422, unsupported scopes/overrides and invalid search
terms return 400, backend failures return 500, and unavailable app state returns
503. Errors are JSON objects with an `error` field. `/health` reports application
readiness; it is not a live index or permission check.

## MCP Connection

Start the service separately and configure the .NET toolkit process:

```bash
export COSMOS_RETRIEVER_URL='http://127.0.0.1:9000'
export COSMOS_RETRIEVER_TIMEOUT_S=600
```

Call `agentic_search` without `schemaOverride`, or with `"none"` (the toolkit
omits it on the wire). Configure the schema at service startup instead. The toolkit
passes the service's JSON response through. This endpoint performs one full-text
search; it does not yet provide the multi-turn workflow described by the agentic tool.

The app creates one Cosmos client and executor, with one retriever per configured
container, at startup. It closes owned resources at shutdown or after failed startup.
Injected retrievers remain
caller-owned. Blocking work runs off the HTTP event loop. An HTTP caller timeout
does not cancel an already-running synchronous SDK query.

## Checks

```bash
python -m pytest tests/unit -q
```

Unit tests use fake SDK clients and make no cloud calls. For a live smoke, use an
authorized configured containers, run the requests above, then call `agentic_search`
from the toolkit. Check source identity, text paths, per-target partition routing,
one/all target selection, limits, partial/all failure, and rejection of unconfigured
targets. Do not write fixture records into a shared corpus.