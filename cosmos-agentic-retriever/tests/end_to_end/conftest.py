"""Open prepared live fixtures read-only. Never provision, seed, or delete here."""

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential, DefaultAzureCredential
from cosmos_live_corpus import select_data
from cosmos_live_fixtures import (
    DEFAULT_DATABASE,
    fixtures,
    read_fixture_items,
    service_settings,
    validate_container,
    validate_items,
)
from fastapi.testclient import TestClient

from cosmos_agentic_retriever.server import create_app


@pytest.fixture(scope="session")
def live_selection():
    archive = os.environ.get("COSMOS_TEST_SCIFACT_ARCHIVE")
    return select_data(
        os.environ.get("COSMOS_TEST_DATA", "synthetic"),
        Path(archive) if archive else None,
    )


@pytest.fixture(scope="session")
def live_settings(live_selection):
    endpoint = os.environ.get("COSMOS_TEST_ENDPOINT")
    if not endpoint:
        pytest.fail("COSMOS_TEST_ENDPOINT is required when RUN_COSMOS_LIVE=1")
    settings = service_settings(
        endpoint,
        os.environ.get("COSMOS_TEST_DATABASE", DEFAULT_DATABASE),
        os.environ.get("COSMOS_TEST_CREDENTIAL", "azure_cli"),
        selected=live_selection[0],
    )
    with ExitStack() as resources:
        credential = (
            AzureCliCredential()
            if settings.cosmos_credential == "azure_cli"
            else DefaultAzureCredential()
        )
        resources.callback(credential.close)
        client = CosmosClient(str(settings.account_uri), credential=credential)
        resources.callback(client.close)
        database = client.get_database_client(settings.cosmos_database)
        for fixture in live_selection[0]:
            container = database.get_container_client(fixture.name)
            validate_container(container.read(), fixture)
            validate_items(read_fixture_items(container, fixture), fixture)
    return settings


@pytest.fixture(scope="session")
def live_http(live_settings):
    with TestClient(create_app(live_settings)) as client:
        yield client


@pytest.fixture(scope="session")
def synthetic_settings(live_settings):
    names = {fixture.name for fixture in fixtures()}
    settings = live_settings.model_copy(deep=True)
    settings.cosmos_containers = {
        name: config
        for name, config in settings.cosmos_containers.items()
        if name in names
    }
    if not settings.cosmos_containers:
        pytest.skip("synthetic containers were not selected")
    return settings


@pytest.fixture(scope="session")
def synthetic_http(synthetic_settings):
    with TestClient(create_app(synthetic_settings)) as client:
        yield client


@pytest.fixture(scope="session")
def real_corpus(live_selection):
    real = live_selection[1]
    if real is None:
        pytest.skip("select scifact or both to run real-corpus tests")
    return real
