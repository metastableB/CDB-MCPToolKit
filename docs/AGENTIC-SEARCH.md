# `agentic_search` HTTP integration

Use `agentic_search` to send a search request from an MCP client to an external
retrieval service. The .NET toolkit validates the tool arguments, sends an HTTP
`POST /search` request, and returns the service's response body.

**A separately running, compatible retrieval service is required.** Its
implementation is not included in this version of the toolkit. Registering the
MCP tool does not provide a working end-to-end retrieval service on its own.
The .NET caller does not execute Cosmos queries, run an LLM loop, or rerank results.

## Architecture

```text
  MCP client
       |
       | MCP tool call: agentic_search
       v
  MCPToolKit (.NET): AgenticSearch -> AgenticSearchExecutor
       |
       | HTTP POST /search       <-- Response body
       v
  External retrieval service (not included)
```

## Configuration

Set these optional variables on the .NET toolkit process. The values shown are
the defaults. They are read on each call. Commented entries in `.env.example`
indicate optional overrides, not a disabled API.

```bash
# Base URL of the external retrieval service
COSMOS_RETRIEVER_URL=http://127.0.0.1:9000

# HTTP request timeout in seconds
COSMOS_RETRIEVER_TIMEOUT_S=600
```

The default URL must be reachable from the toolkit process. It does not start a
service. Missing or invalid timeout values fall back to 600 seconds.

## Tool arguments

| Argument | Requirement | Forwarding behavior |
| --- | --- | --- |
| `query` | Required, nonempty string | Sent as `query`. |
| `maxDocuments` | Optional integer, 1-50; default 20 | Sent as the requested result limit. |
| `database` | Optional string | Sent when nonblank. |
| `container` | Optional string | Sent when nonblank. |
| `schemaOverride` | Optional JSON object | Sent as `overrides.schema_override`. Omit for no override. |

The service must interpret the targeting, result limit, and schema override.
The caller neither discovers a schema nor chooses a default corpus when a target
is omitted. Those behaviors depend on the external service.

For example, these tool arguments:

```json
{"query": "battery recycling", "maxDocuments": 5, "database": "research", "container": "articles", "schemaOverride": {"item_id_path": "/id"}}
```

produce this JSON body for `POST /search`:

```json
{"query": "battery recycling", "maxDocuments": 5, "database": "research", "container": "articles", "overrides": {"schema_override": {"item_id_path": "/id"}}}
```

## Responses and errors

On a successful HTTP response, the caller returns the nonempty response body
without interpreting document fields or enforcing a result schema. The external
service defines that schema; the toolkit does not guarantee a list of documents
or fields such as `text` and `justification`.

Connection failures, request timeouts, and empty successful responses produce
JSON error objects containing `error` and, where applicable, `hint`. For a
non-success HTTP status, a body beginning with `{` or `[` is passed through;
other bodies are wrapped in a JSON error object. Without a compatible service,
the tool cannot return search results.

