# cosmos-agentic-retriever

This package is being built to provide multi-turn agentic retrieval over Azure
Cosmos DB for NoSQL for the MCP Toolkit's `agentic_search` tool.

Given a natural-language query, the service searches a configured Cosmos DB
corpus over multiple turns and returns a ranked set of relevant documents. It
supports hybrid vector and full-text search, optional reranking, and
configurable document schemas.

Currently available: query compilation and execution through a caller-supplied
Cosmos container client. Import `CosmosExecutor` from
`cosmos_agentic_retriever.query_engine.executor` and call `run(compiled_query)`
to collect rows. The client owns retries; query failures propagate to the caller.
The HTTP service and multi-turn agent are not included yet.
