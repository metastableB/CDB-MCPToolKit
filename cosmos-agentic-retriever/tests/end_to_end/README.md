# Live Cosmos Search Tests

These tests exercise the Python HTTP handlers and query engine against real Cosmos.
They use FastAPI's in-process HTTP transport, not a deployed server or the .NET MCP
tool. No embeddings or LLM credentials are needed. Run commands from the Python
package directory after `python -m pip install -e '.[dev]'`.

Synthetic edge cases and real source documents live in separate containers.
Select `--data synthetic`, `--data scifact`, or `--data both` during setup.
The default remains `synthetic`, so an existing setup command cannot silently
download or ingest a new corpus. `both` also enables combined-container tests.

## Prepare Fixtures

Use an existing Cosmos DB for NoSQL account with full-text search available and
network access from your machine. Sign in with `az login`. The setup identity
needs management-plane permission to create database/container resources and
Cosmos data-plane permission to read and create items. Subscription Contributor
alone does not grant data access; Cosmos Data Contributor alone does not grant
management access. The script does not grant roles or enable account features.

```bash
python tools/setup_cosmos_live_tests.py \
  --subscription SUBSCRIPTION-ID \
  --resource-group RESOURCE-GROUP \
  --account COSMOS-ACCOUNT
```

The default database is `mcp-live-tests-v1`. An override must start with
`mcp-live-tests-`. Use a dedicated test database, never an application database.

Setup creates missing resources, validates existing policies, and inserts only
missing records. It never upserts changed records or changes existing indexes.
Mismatched fixtures fail with a request to use a new fixture version/database.
All selected records are checked before seeding begins. Unselected containers
are not read or changed. No tests run during setup.

All containers use a `/tenant` partition key:

| Container | Purpose |
| --- | --- |
| `flat-v1` | 3 synthetic records: `/id`, `/text`, metadata and partitions |
| `nested-v1` | 3 synthetic records: `/record/id` and `/content/body` |
| `fields-v1` | 3 synthetic records: field-selection marker words |
| `scifact-100-v1` | 100 real SciFact abstracts with original IDs and titles |

The synthetic containers share a logical `shared` item ID to check cross-container
identity. Their partition values `0` and `other` check partition routing. SciFact
uses the partition value `scifact` and full-text indexes on `/title` and `/text`.

### Reproduce the Synthetic Data

All nine synthetic documents are defined in [`fixtures()`](../../tools/cosmos_live_fixtures.py).
There is no download, random sampling, or model generation. The text is literal,
including `battery recycling canaryalpha` and the field-selection marker words.
To inspect every document locally without Azure access:

```bash
PYTHONPATH=tools python -c 'import json; from cosmos_live_fixtures import fixtures; print(json.dumps({fixture.name: fixture.items for fixture in fixtures()}, indent=2))'
```

Use the same repository revision for setup and tests. Running setup on a fresh
authorized test database reproduces the application records and policies. Cosmos
assigns its own system metadata, so values such as `_etag` and `_ts` will differ.
Existing fixture records are validated rather than overwritten.

### Prepare the Real Corpus

[BEIR SciFact](https://github.com/beir-cellar/beir#beers-available-datasets) supplies
real scientific abstracts, claims as queries, and published relevance labels.
The [dataset card](https://huggingface.co/datasets/BeIR/scifact) lists CC BY-SA 4.0.
Dataset terms and attribution apply separately from this repository's code license.
The archive is downloaded separately and is not bundled into the package.

```bash
mkdir -p .live-data
curl --fail --location --max-time 90 --max-filesize 8388608 \
  'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip' \
  --output .live-data/scifact.zip
printf '%s  %s\n' \
  '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165' \
  '.live-data/scifact.zip' | sha256sum --check

python tools/cosmos_live_corpus.py --archive .live-data/scifact.zip

python tools/setup_cosmos_live_tests.py \
  --subscription SUBSCRIPTION-ID --resource-group RESOURCE-GROUP \
  --account COSMOS-ACCOUNT --data both \
  --scifact-archive .live-data/scifact.zip
```

The loader verifies that SHA-256 before any Azure call. It selects the first five
test query IDs in numeric order, retains all their positively labelled documents,
then fills to 100 documents in numeric document-ID order. Original titles, text,
IDs and relevance labels are preserved. The source SHA-256 is stored on each
document. Setup prints a second checksum covering the selected documents, queries
and labels. There is no random sampling or model-generated text.

`both` selects four containers and 109 documents total. `scifact` selects only
the real-data container and leaves existing synthetic containers untouched.
Validation reads at most the expected count plus one from each selected container:
four per synthetic container and 101 for the real-data container.

This relevance-enriched subset is for integration testing, not a full SciFact
benchmark. It cannot support claims about corpus-scale recall or performance.
The original SKF tests in PR #150 use preloaded shared containers but supply no
export or ingestion recipe. This setup does not contact or copy those containers.

### Index Policy Validation

The policy explicitly excludes `/"_etag"/?`, matching the
[Cosmos default](https://learn.microsoft.com/en-us/azure/cosmos-db/index-policy#includeexclude-strategy).
This excludes the system version field from the query index, not from storage or
optimistic concurrency checks. The validator still compares all excluded paths.
For included paths, it treats ARM's `indexes: null` as an omitted field, but
rejects explicit index lists and changed paths. Record comparisons omit only
Cosmos-generated metadata and still require exact application content.

**Costs:** resources persist by default. A new provisioned test database requests
400 shared RU/s; serverless accounts are usage-billed. The account must support
that configuration. Existing throughput is not changed. Full-text index/storage
and queries may incur charges. Serialize setup runs for the same fixture database.
If setup fails partway through, it leaves created resources for diagnosis/rerun.

Add `--check-only` to validate without creating or writing anything. This script
still reads management metadata; the tests below need only Cosmos data access.

## Run Read-Only Tests

Setup prints the endpoint and fixture database. Set those values for pytest:

```bash
export COSMOS_TEST_ENDPOINT='https://ACCOUNT.documents.azure.com:443/'
export COSMOS_TEST_DATABASE='mcp-live-tests-v1'
export COSMOS_TEST_CREDENTIAL='azure_cli'
export COSMOS_TEST_DATA='both'
export COSMOS_TEST_SCIFACT_ARCHIVE="$PWD/.live-data/scifact.zip"
RUN_COSMOS_LIVE=1 python -m pytest tests/end_to_end/test_cosmos_search_integration.py \
  -q -o junit_family=legacy --junitxml=artifacts/cosmos-search.xml
```

Use `COSMOS_TEST_DATA=synthetic` and unset `COSMOS_TEST_SCIFACT_ARCHIVE` for the
synthetic-only suite. Use `scifact` for real-only tests. Unselected suites skip.
Selecting real data with a missing or incorrect archive fails before Azure access.
Tests never download data or run setup. Use the same archive and code revision
that prepared the database.

Use Cosmos DB Built-in Data Reader or equivalent scoped access for test execution.
`COSMOS_TEST_CREDENTIAL=default` selects DefaultAzureCredential instead. The tests
explicitly disable account-key use. They generate service settings from fixture
definitions, so no separate COSMOS_CONTAINERS JSON needs to be copied by hand.

Ordinary runs skip these tests. With `RUN_COSMOS_LIVE=1`, missing configuration,
missing/changed fixtures, authentication failures, and database errors fail the
run. They are not converted into skips. Readiness checks retry successful queries
for up to 60 seconds while waiting for expected indexed results; service errors
fail immediately. SDK request/retry time is additional to this polling budget.

Tests verify known content and IDs, field selection, one/all container search,
global limits, qualified IDs, within/cross-partition scope, and rejection of
per-request schema overrides. Startup schemas differ by container. Per-request
schema overrides remain unsupported. Concurrent test-only runs do not write data.
Cosmos ranks even nonmatching items, so an arbitrary unmatched query is not an
empty-result assertion; a nonexistent partition supplies the empty-result canary.

Real-only queries verify returned IDs, original field content, source attribution,
rank order and limits. Each query records Recall@5 against its published labels,
returned IDs and the subset checksum as JUnit properties. Recall is reported,
not used as an unmeasured pass threshold. A zero-recall result must be reviewed
even if the interface checks pass. Combined tests verify all selected containers
are searched, IDs stay distinct and the global cap holds. Scores on disjoint
containers are fused by rank, not globally comparable BM25 scores.

`test_per_container_filters_use_real_stored_paths` applies equals, range and in
filters to the existing four containers. It checks exact item IDs using `/id` in
flat records and `/record/id` in nested records. No additional fixtures are needed.

## Optional Cleanup

```bash
python tools/setup_cosmos_live_tests.py \
  --subscription SUBSCRIPTION-ID --resource-group RESOURCE-GROUP --account COSMOS-ACCOUNT \
  --data both --scifact-archive .live-data/scifact.zip \
  --cleanup --confirm-cleanup mcp-live-tests-v1
```

Cleanup checks selected policies and records before deleting only the selected
test containers. Foreign or changed records stop deletion. It never deletes
the database or account, so shared database throughput may still be billable.
Do not run cleanup or edit fixtures concurrently with setup/tests. No automatic
cleanup happens on test success or failure.

## GitHub Actions

The same pytest command and environment variables work on a runner with network
access to Cosmos. Use a manually triggered job on a trusted branch with a protected
environment, `azure/login` OIDC, and a scoped identity. After Azure CLI login,
`COSMOS_TEST_CREDENTIAL=azure_cli` uses that identity without account-key secrets.
For real-data jobs, download and checksum the archive in a separate step, then
set the same `COSMOS_TEST_DATA` and `COSMOS_TEST_SCIFACT_ARCHIVE` values as above.
Keep write-capable setup in a separately approved job; regular live test jobs can
use read-only access and upload the JUnit report even on failure.

Serialize provisioning/cleanup per account/database; parallel read-only runs are
safe when the fixtures are unchanged. Never run credentialed jobs against untrusted
fork code or use `pull_request_target` to execute untrusted PR contents. No workflow
or Azure environment is created by this change. Full MCP-to-Cosmos testing remains
a separate integration step.