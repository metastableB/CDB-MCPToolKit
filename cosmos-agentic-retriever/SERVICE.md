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
export COSMOS_CONTAINERS='{"articles":{"cosmos_schema":{"item_id_path":"/id","partition_key_paths":["/tenant"],"text_paths":["/text"]}},"reports":{"cosmos_schema":{"item_id_path":"/id","partition_key_paths":["/tenant"],"text_paths":["/content/body"]}}}'
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
- `cosmos_schema.partition_key_paths` must match the container's actual partition
  key definition, in order. For example, `["/tenant"]`, or `["/tenant", "/region"]`
  for a hierarchical key. These are field locations, not the `partition_key`
  value used to restrict a query. Omitted paths fail configuration validation.
  The service does not discover or verify the container definition automatically.
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
`database`, `container`, and `container_filters`. Omit `container` to search all configured containers,
or name one to restrict the search. Omitted `database` uses the configured database.
Other databases, unconfigured containers, and `*` are rejected. Nonempty `overrides`
objects are rejected. The old single-container environment settings are replaced
by `COSMOS_CONTAINERS`; there is no implicit fallback to those settings.

### Per-Container Filters

Use stored Cosmos paths for each container's filters. For the two containers
configured above, a request can apply different publication-year fields:

```json
{
  "query": "battery recycling",
  "maxDocuments": 5,
  "container_filters": {
    "articles": [{"kind": "range", "path": "/publication/year", "minimum": 2020}],
    "reports": [{"kind": "range", "path": "/publishedYear", "minimum": 2020}]
  }
}
```

Supported conditions are `equals` with `value`, `range` with inclusive `minimum`
and/or `maximum`, and `in` with `values`. Conditions within one container are ANDed.
A range needs at least one bound. Values are passed as SQL parameters.

Omit `container_filters` or use null for an unfiltered search. When supplied, the
map must name every selected container and no others. Use `[]` to explicitly leave
one selected container unfiltered. An empty map or a missing/extra target returns
400 before any query. A malformed path or filter returns 422 before any query.
If `container` selects one target, the map must contain only that target.

Paths use the same slash/JSON-quoted segment syntax as the schema. They are not
logical aliases, raw SQL, indexing wildcards, or schema overrides. The service
validates syntax but does not discover whether the field exists or what it means.
Missing fields follow Cosmos comparison semantics. A filter can reference a field
that is not projected. Output mappings such as `source_path` and `metadata_paths`
do not affect filters, and `source` may also be a metadata label. No additional
startup filter mapping is required. Configured partition restrictions still apply.

Python query-engine callers use `EqualsFilter(path="/source", value="SciFact")`
and equivalent `RangeFilter`/`InFilter` inputs. The old `logical_field` argument
is rejected rather than interpreted as a path or silently ignored.

### Results

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
`item_id` is the string value at the configured `item_id_path`, which need not be
Cosmos's physical `id`. `cosmos_identity` separately contains the physical `id`
and ordered `partition_key` values. For example:

```json
{"item_id":"paper-7","cosmos_identity":{"id":"physical-42","partition_key":["tenant-A"]}}
```

`retrieval_id` is `database/container:identity`, with each part percent-escaped.
The identity part is compact JSON `[partition_key_values, physical_id]`, such as
`[["tenant-A"],"physical-42"]`. Treat the complete string as an opaque identifier.
Different partition values or physical IDs produce distinct IDs even when logical
`item_id` values match. Repeated appearances of the same physical item deduplicate.
IDs do not depend on rank or which other containers were searched.

Strings, numbers, booleans, null and undefined partition components remain distinct.
An empty object `{}` denotes an undefined component. Numeric values use Cosmos's
double precision semantics (1 and 1.0 identify the same partition). The projected
partition paths must be correct; configuring an unrelated field is not safe.
Incomplete physical identity from a query is an error, not a logical-ID fallback.
This replaces the earlier `database/container:item_id` format. Existing clients
must not construct or cache that older format as physical identity.
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

The existing .NET caller does not yet forward `container_filters`. The filter
contract above is available to direct HTTP/Python callers. Agent tool arguments
and field information from explicit service configuration require separate wiring.
There is no automatic schema discovery in this service.

## Checks

```bash
python -m pytest tests/unit -q
```

Unit tests use fake SDK clients and make no cloud calls. For a live smoke, use an
authorized configured containers, run the requests above, then call `agentic_search`
from the toolkit. Check source identity, text paths, per-target partition routing,
one/all target selection, limits, partial/all failure, and rejection of unconfigured
targets. Do not write fixture records into a shared corpus.

For repeatable live validation, follow the [live-test guide](tests/end_to_end/README.md).
The setup script prepares persistent fixtures; a separate opt-in pytest suite
checks them read-only. Setup and test execution can run as separate CI jobs.