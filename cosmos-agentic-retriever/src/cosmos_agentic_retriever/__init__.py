"""Multi-turn agentic retrieval over Azure Cosmos DB for NoSQL.

This module implements the `cosmos-retriever` service, providing multi-turn
agentic retrieval over a Cosmos DB for NoSQL database.

Usage: While the module is designed to be used through the cosmos DB MCP
toolkit, we illustrate the main components with a direct usage pattern here

- Assumes the following two containers exists with their own id and text fields. 

	articles: {"id": "a1", "tenant": "team-a", "text": "Battery recycling..."}
	reports: {"id": "r1", "tenant": "team-a", "content": {"body": "Battery recycling..."}}

- Assume we are logged in and our azure identity has permissions to query both containers.
- Assume Cosmos full-text search is configured to index the articles container's
text field and the reports container's content.body field.


	import os
	import uvicorn

	from cosmos_agentic_retriever.config import RetrieverConfig
	from cosmos_agentic_retriever.server import create_app

	settings = RetrieverConfig(
		account_uri=os.environ["ACCOUNT_URI"],
		cosmos_database="example-db",
		cosmos_credential="azure_cli",
		cosmos_key=None,
		cosmos_containers={
			"articles": {"cosmos_schema": {
				"item_id_path": "/id",
				"partition_key_paths": ["/tenant"],
				"text_paths": ["/text"],
			}},
			"reports": {"cosmos_schema": {
				"item_id_path": "/id",
				"partition_key_paths": ["/tenant"],
				"text_paths": ["/content/body"],
			}},
		},
	)
	app = create_app(settings)
	uvicorn.run(app, host="127.0.0.1", port=9000)

From another terminal, search both containers with one request:

	curl --fail-with-body http://127.0.0.1:9000/search \
	  -H 'Content-Type: application/json' \
	  -d '{"query":"battery recycling","maxDocuments":5}'

Add "container":"articles" to the JSON body to search only articles. The response
contains at most five items across both containers, not five from each. Each item
identifies its database and container. If a container's search fails, the response
includes that error. See the README to run the service and connect the MCP toolkit.
"""
