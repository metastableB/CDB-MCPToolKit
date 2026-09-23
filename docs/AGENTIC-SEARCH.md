# `agentic_search` HTTP integration

Use `agentic_search` to find relevant documents in a Cosmos DB corpus. A retrieval
agent uses an LLM to refine the query through multiple rounds of searching and
document reading.

The .NET toolkit validates the tool arguments, sends an HTTP
`POST /search` request, and returns the service's response body.
Connection failures and request timeouts produce JSON errors.

*A separately running, compatible retrieval service is required.* Configure the
agent's retrieval behavior, Cosmos access, and models on that service.

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
     External retrieval service
```

## Configuration

The tool reads two optional environment variables. The values below are the defaults.

```bash
# Base URL of the external retrieval service
COSMOS_RETRIEVER_URL=http://127.0.0.1:9000

# HTTP request timeout in seconds
COSMOS_RETRIEVER_TIMEOUT_S=600
```

