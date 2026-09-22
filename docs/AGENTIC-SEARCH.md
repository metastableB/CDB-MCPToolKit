# `agentic_search` HTTP integration

Use `agentic_search` to send a search request from an MCP client to an external
retrieval service. The .NET toolkit validates the tool arguments, sends an HTTP
`POST /search` request, and returns the service's response body.
Connection failures and request timeouts produce JSON errors.

*A separately running, compatible retrieval service is required.* (TODO: Update this as the retrieval service lands).

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

The tool reads two optional environment variables. The values below are the defaults.

```bash
# Base URL of the external retrieval service
COSMOS_RETRIEVER_URL=http://127.0.0.1:9000

# HTTP request timeout in seconds
COSMOS_RETRIEVER_TIMEOUT_S=600
```

