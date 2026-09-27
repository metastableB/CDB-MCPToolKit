# cosmos-agentic-retriever

This submodule is the HTTP service that backs the Azure Cosmos DB MCP Toolkit's
`agentic_search` tool. Given a natural-language question, it searches a set of
explicitly configured Cosmos DB for NoSQL containers and returns an answer
together with the documents that support it.

## Setting up and Running the Service

Install into your Python environment:

```bash
python -m pip install .
```

Configure the account, database, containers, and a model through environment
variables, then start the HTTP service. This service only reads your existing
containers --- it never creates or changes databases, containers, indexes, or
documents.

```bash
export ACCOUNT_URI='https://YOUR-ACCOUNT.documents.azure.com:443/'
export COSMOS_DATABASE='YOUR-DATABASE'
export COSMOS_CONTAINERS='{"articles":{"cosmos_schema":{"item_id_path":"/id","partition_key_paths":["/tenant"],"text_paths":["/text"]}}}'
export COSMOS_CREDENTIAL='azure_cli'
export LLM_BASE_URL='https://YOUR-ENDPOINT/v1'
export LLM_MODEL='YOUR-MODEL'
export LLM_API_KEY='YOUR-KEY'          # only if your endpoint requires one
az login
python -m cosmos_agentic_retriever serve
```

Replace the `YOUR-*` values with your own. `COSMOS_CONTAINERS` is a JSON object
keyed by container name; each entry declares that container's schema and search
scope. The agent needs a model: `LLM_BASE_URL` and `LLM_MODEL` point at any
OpenAI-compatible chat-completions endpoint (OpenAI, Azure OpenAI, a local vLLM
server, or another gateway).

The signed-in identity needs Cosmos data-plane permission to query the
containers, and the configured text paths must already have a full-text policy
and index. The HTTP API has no caller authentication and is intended to be bound
loopback (local) or behind an authenticated private gateway.

### Configure with a YAML file

Use [config.example.yaml](config.example.yaml) as the template for a local
`config.local.yaml`. The example defines two containers with different text
fields. Set your account, database, model, and container schemas in the local
file, then start the service:

```bash
export COSMOS_RETRIEVER_CONFIG_FILE='./config.local.yaml'
az login
python -m cosmos_agentic_retriever serve
```

Local `config*.yaml` and `config*.yml` files are ignored by Git. YAML values
take precedence over environment variables. Leave `COSMOS_KEY` and `LLM_API_KEY`
out of the YAML so they come from the environment.  With `azure_cli`, Cosmos
needs no key; your model endpoint may still require one.

You can keep environment variables in a local `.env` rather than exporting them
one by one.
```bash
set -a
source .env
set +a
```

## Connect the MCP toolkit

Run the service, then point the .NET toolkit at it:

```bash
export COSMOS_RETRIEVER_URL='http://127.0.0.1:9000'
export COSMOS_RETRIEVER_TIMEOUT_S=600
```

This connects the `agentic_search` mcp tool call with this service.

## Tests

Install the development extras (test dependencies) and run the unit tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/unit -q
```

Unit tests use fake SDK clients and make no cloud calls. For live validation
against a real account, follow the [live-test guide](tests/end_to_end/README.md).
