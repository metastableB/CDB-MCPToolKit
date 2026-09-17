# cosmos-agentic-retriever

This submodule provides multi-turn agentic retrieval over Azure Cosmos DB for
NoSQL. It powers the Azure Cosmos DB MCP Toolkit's `agentic_search` tool.

Given a natural-language query, the service searches a configured Cosmos DB
corpus over multiple turns and returns a ranked set of relevant documents. It
supports hybrid vector and full-text search, optional reranking, and
configurable document schemas.
