# Stopword Policy Check

This opt-in test sends compiler-generated SQL to a small Cosmos fixture container.
It does not create resources, write records, change indexes, or run the HTTP service.
Wait for PR5's approved fixture setup to create `flat-v1` in `mcp-live-tests-v1`
(or an explicitly authorized database whose name starts with `mcp-live-tests-`).
Do not point it at application data. Queries consume RUs.

From the package directory, using its virtual environment:

```bash
python -m pip install -e '.[dev]' 'azure-cosmos>=4.7,<5' 'azure-identity>=1.17,<2'
az login
export COSMOS_TEST_ENDPOINT='https://YOUR-TEST-ACCOUNT.documents.azure.com:443/'
export COSMOS_TEST_DATABASE='mcp-live-tests-v1'
RUN_COSMOS_STOPWORD_LIVE=1 python -m pytest tests/live/test_stopword_policy.py -q -s
```

Use a Cosmos data-reader identity. No account keys are used. Without the opt-in
flag the test skips; with it, missing configuration, dependencies, permissions,
fixtures, indexes, or failed queries fail the run. No cloud calls have yet been
made to validate this test.

The test checks mixed-stopword ranking against a unique canary. It prints all-stopword and contraction
results without assuming those queries return no rows. Ties are not compared
by position. A passing run would not prove unlimited terms, performance, custom
stopword support, or multilingual equivalence. PR5's HTTP integration tests remain
a separate gate.