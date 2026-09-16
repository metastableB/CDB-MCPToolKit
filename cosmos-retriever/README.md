# cosmos-retriever

Retrieval library for the Azure Cosmos DB MCP Toolkit's `agentic_search` tool.

The package is being landed incrementally, one reviewable slice at a time. The
current slice provides the **query compiler**: it translates a search request
plus a corpus schema into a Cosmos DB SQL query (vector / full-text / hybrid RRF
/ filter-only / read-document), building parameterised queries only — user
values never enter the SQL text. It does not connect to or execute against
Cosmos DB; that lands in a later slice.
