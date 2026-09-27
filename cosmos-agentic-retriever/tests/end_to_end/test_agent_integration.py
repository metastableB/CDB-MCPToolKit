"""Live agentic-search check: run the loop against a real model and Cosmos.

Opt in with RUN_COSMOS_LIVE=1 (after tests/live/setup_cosmos_live_tests.py) and a
real OpenAI-compatible model in COSMOS_TEST_LLM_BASE_URL, COSMOS_TEST_LLM_MODEL,
and optionally COSMOS_TEST_LLM_API_KEY. The model is non-deterministic, so this
checks the loop's shape -- an answer, supporting documents, a clean stop -- not an
exact answer. HTTP uses FastAPI's in-process test transport, not the .NET/MCP
network path.
"""

import os

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from cosmos_agentic_retriever.server import create_app

pytestmark = [
    pytest.mark.cosmos_live,
    pytest.mark.skipif(
        os.environ.get("RUN_COSMOS_LIVE") != "1",
        reason="set RUN_COSMOS_LIVE=1 to use real Cosmos",
    ),
    pytest.mark.skipif(
        not os.environ.get("COSMOS_TEST_LLM_BASE_URL")
        or not os.environ.get("COSMOS_TEST_LLM_MODEL"),
        reason="set COSMOS_TEST_LLM_BASE_URL and COSMOS_TEST_LLM_MODEL to run the agent",
    ),
]


@pytest.fixture(scope="session")
def agent_http(synthetic_settings):
    settings = synthetic_settings.model_copy(deep=True)
    settings.llm_base_url = os.environ["COSMOS_TEST_LLM_BASE_URL"]
    settings.llm_model = os.environ["COSMOS_TEST_LLM_MODEL"]
    api_key = os.environ.get("COSMOS_TEST_LLM_API_KEY")
    settings.llm_api_key = SecretStr(api_key) if api_key else None
    with TestClient(create_app(settings)) as client:
        yield client


def test_agent_search_answers_with_supporting_documents(agent_http):
    response = agent_http.post(
        "/agentic_search",
        json={"query": "What do the documents say about battery recycling?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["terminal_reason"] in {"stop", "max_turns"}
    assert body["turns"] >= 1
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert body["documents"], "the agent should return supporting documents"
