# cosmos-agentic-retriever

This submodule provides multi-turn agentic retrieval over Azure Cosmos DB for
NoSQL. It powers the Azure Cosmos DB MCP Toolkit's `agentic_search` tool.

Given a natural-language query, the service runs full-text search (lexical) over
a set of explicitly configured Cosmos DB containers and returns a ranked set of
relevant documents.  Search scope and per-container schemas are configured at
startup; containers are not discovered automatically.
