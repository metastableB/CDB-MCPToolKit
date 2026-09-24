# Stopword Policy Check

Compare Cosmos full-text rankings with and without common English words in the
query. The test runs compiler-generated SQL against a dedicated test container.
It only reads data. Queries consume RUs.

## Prerequisites

Use a Cosmos DB for NoSQL account with full-text search enabled and network access
from the test machine. Sign in to Azure CLI with a Cosmos DB Built-in Data Reader
role, or equivalent data access, on the test database. No account keys are used.

The database name must start with `mcp-live-tests-`; the default is
`mcp-live-tests-v1`. Prepare a container named `flat-v1` with:

- Partition key: `/tenant`.
- Full-text policy: `/text` with language `en-US`.
- Full-text index: `/text`.
- Exactly the three records below, each with `fixture_version` set to `mcp-search-v1`.

| `id` | `tenant` | `text` |
| --- | --- | --- |
| `shared` | `0` (number) | `battery recycling canaryalpha` |
| `first-only` | `0` (number) | `battery recycling canaryalpha` |
| `other-only` | `"other"` (string) | `battery recycling canarybeta` |

The test checks these requirements before running ranking queries. It does not
create the database, container, index, or records.

## Run

From the package directory, using its virtual environment:

```bash
python -m pip install -e '.[dev]' 'azure-cosmos>=4.7,<5' 'azure-identity>=1.17,<2'
az login
export COSMOS_TEST_ENDPOINT='https://YOUR-TEST-ACCOUNT.documents.azure.com:443/'
export COSMOS_TEST_DATABASE='mcp-live-tests-v1'
RUN_COSMOS_STOPWORD_LIVE=1 python -m pytest tests/live/test_stopword_policy.py -q -s
```

Without `RUN_COSMOS_STOPWORD_LIVE=1`, the test skips. When enabled, missing
configuration, dependencies, permissions, fixtures, or failed queries fail the run.

## Results

The test checks that `canarybeta` and `the canarybeta and` both rank `other-only`
first and return the same set of IDs. Tied rows are not compared by position.
It also prints results for all-stopword and contraction queries without asserting
their ranking or requiring empty results. This test does not cover HTTP endpoints,
performance, query-size limits, custom stopwords, or multilingual search.