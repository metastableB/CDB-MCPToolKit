# `agentic_search` — multi-turn agentic retrieval tool

`agentic_search` uses an LLM search agent to answer natural language queries
against an Azure Cosmos DB corpus. Given a query, the agent issues hybrid
(vector + full-text) RRF searches, optionally reranks and fetches full
documents. Even though we use an LLM for `agentic_search`, from the MCP client's
perspective it's a single tool call. Internally, the agent can use multiple
rounds of retrieval to fetch these documents while respecting a configurable
token budget. Sophisticated multi-turn queries can take upwards of 30s for
completion.


## Architecture

```text
  MCP client  (Claude Desktop · AI Foundry · VS Code Copilot)
       │
       │  MCP HTTP  (one tool call)
       ▼
  MCPToolKit (.NET)
       [McpServerTool] AgenticSearch  →  AgenticSearchExecutor
       │
       │  HTTP POST /search    ◄── JSON body returned ──
       ▼
  cosmos-retriever  (Python · FastAPI · uvicorn, kept warm)
       └─ multi-turn retrieval loop (token-budgeted)
            ├─ tools: search / grep / read / prune
            ├─ retriever model
            ├─ embedding model
            └─ corpus on Cosmos DB (vector + full-text)
```

The retriever runs as a separate Python process; the .NET tool calls its
`POST /search` endpoint per request and passes the JSON response back verbatim.

Returns a JSON list of ranked documents (each with `text` and a short
`justification`).

By default the tool searches the retriever's default corpus; pass the optional
`database` and `container` arguments to target a different Cosmos corpus per
call.

## Configuration

The tool reads two optional environment variables. If `COSMOS_RETRIEVER_URL`
doesn't point at a running retriever, it returns a clean JSON
`{"error":"...","hint":"..."}` envelope instead of crashing.

```bash
# Base URL of the cosmos-retriever service (default shown)
COSMOS_RETRIEVER_URL=http://127.0.0.1:9000

# Per-request wall-clock cap in seconds; the request is abandoned past this
COSMOS_RETRIEVER_TIMEOUT_S=600
```

The retriever service has its **own** separate configuration (Cosmos account,
models, corpora). See the retriever service configuration docs for that setup.

