# Live Cosmos Search Tests

These tests run the Python HTTP handlers and query engine against a real Cosmos
DB for NoSQL account. Assumes an account you can provision into and a signed-in
identity with the needed permissions. Run from the package directory after
`python -m pip install -e '.[dev]'` and `az login`.

## Setup

The setup script creates the test containers if they do not exist and inserts
the documents the tests search: a few tiny hand-made records plus 100 real BEIR
SciFact abstracts:

```bash
# Download data
mkdir -p .live-data
curl --fail --location --max-time 90 --max-filesize 8388608 \
  'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip' \
  --output .live-data/scifact.zip
# Verify the download is the exact expected file (fails if the SHA-256 differs)
printf '%s  %s\n' \
  '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165' \
  '.live-data/scifact.zip' | sha256sum --check
# Setup the test containers on cosmos
python tests/live/setup_cosmos_live_tests.py \
  --subscription SUBSCRIPTION-ID --resource-group RESOURCE-GROUP \
  --account COSMOS-ACCOUNT --scifact-archive .live-data/scifact.zip
```

Replace `SUBSCRIPTION-ID`, `RESOURCE-GROUP`, and `COSMOS-ACCOUNT` with your own
Azure details; everything else above is fixed. The database is named
`mcp-live-tests-v1` (an override must start with `mcp-live-tests-`). The script
only adds data — it never overwrites or deletes.

## Run

The live tests are opt in, as they depend on a live database --- set
`RUN_COSMOS_LIVE=1` to turn them on. **The setup command already printed these
four `COSMOS_TEST_*` lines with your real values — paste them exactly as printed;
you do not fill any of them in yourself.** The values here are only examples:

```bash
export COSMOS_TEST_ENDPOINT='https://ACCOUNT.documents.azure.com:443/'  # from setup output
export COSMOS_TEST_DATABASE='mcp-live-tests-v1'                         # from setup output
export COSMOS_TEST_CREDENTIAL='azure_cli'                               # from setup output
export COSMOS_TEST_SCIFACT_ARCHIVE="$PWD/.live-data/scifact.zip"        # from setup output
RUN_COSMOS_LIVE=1 python -m pytest tests/end_to_end/test_cosmos_search_integration.py -q
```

Use the same dataset file and code version you set up with.
