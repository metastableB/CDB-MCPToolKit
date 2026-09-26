"""HTTP layer for the cosmos agentic retriever: the FastAPI app factory, its
routes, and lifecycle.

create_app(settings) builds the app and exposes three endpoints:
- GET /health: readiness only (200 ready, 503 not ready).
- POST /search: search the configured containers and return combined,
  source-tagged results.
- POST /agent_search: run the bounded search agent over a question and return its
  answer with the ranked documents that support it; available only when an LLM
  endpoint is configured.

POST /search names one container to search a single configured target, or omits
it to search all. It returns 200 with each item's source and, on partial failure,
the successful results plus sanitized per-target errors; 500 if every target
fails; 422 for an invalid body; 400 for an unsupported scope or search term; 503
when the app is not ready. Ranking within a container and fusion across
containers are handled by the multi-container retriever, not here. The current
release searches with full text only.

The app owns one Cosmos client and one shared query executor across retrievers;
injected retrievers remain caller-owned. Blocking SDK work runs off the HTTP
event loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from typing import Any

import anyio
import structlog
from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential, DefaultAzureCredential
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cosmos_agentic_retriever.agent import (
    ChatClient,
    make_full_text_search_tool,
    run_agent_search,
)
from cosmos_agentic_retriever.agent.prompts import SYSTEM_PROMPT
from cosmos_agentic_retriever.config import RetrieverConfig, get_config
from cosmos_agentic_retriever.orchestration import (
    ContainerItem,
    ContainerTarget,
    MultiContainerRetriever,
    MultiSearchResult,
    pool_rrf,
)
from cosmos_agentic_retriever.query_engine import CosmosExecutor
from cosmos_agentic_retriever.query_engine.full_text_terms import tokenize_for_fts
from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.query_engine.types import (
    FilterExpression,
    QueryCompilationError,
    RetrievalError,
)
from cosmos_agentic_retriever.query_engine.types import (
    SearchRequest as QuerySearchRequest,
)


class SearchRequest(BaseModel):
    """HTTP input from the MCP caller, with maxDocuments limiting returned items.

    TODO: `container` selects one target, or all when omitted. Support a subset
    (a list of container names) later; the fan-out already operates on an
    arbitrary set of names, so this is an additive input-only change.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    query: str = Field(min_length=1, max_length=4096)
    max_documents: int = Field(
        default=20, strict=True, ge=1, le=50, alias="maxDocuments"
    )
    database: str | None = Field(default=None, min_length=1, max_length=256)
    container: str | None = Field(default=None, min_length=1, max_length=256)
    container_filters: dict[str, list[FilterExpression]] | None = None
    overrides: dict[str, Any] | None = None

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class AgentSearchRequest(BaseModel):
    """HTTP input for the agentic endpoint: a question for the search agent."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4096)

    @field_validator("query")
    @classmethod
    def _reject_blank_agent_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


def create_app(
    settings: RetrieverConfig | None = None,
    *,
    retrievers: dict[str, CorpusRetriever] | None = None,
) -> FastAPI:
    """Serve configured containers using explicit settings or the environment.

    Injected retrievers must match every configured container's schema and policy.
    Their clients remain caller-owned. Otherwise the app owns its client. No discovery,
    container creation, or index changes are performed.
    """
    resolved = (settings or get_config()).model_copy(deep=True)
    if retrievers is not None:
        if retrievers.keys() != resolved.cosmos_containers.keys():
            raise ValueError("retrievers must match the configured containers")
        for name, retriever in retrievers.items():
            config = resolved.cosmos_containers[name]
            if (
                retriever.schema != config.cosmos_schema
                or retriever.policy != config.partition_policy
            ):
                raise ValueError(
                    "retriever schema and partition policy must match configuration"
                )
        retrievers = dict(retrievers)
    selected_fields = {
        name: [
            str(path)
            for path in config.cosmos_schema.resolve_text_fields(
                config.search_text_fields
            )
        ]
        for name, config in resolved.cosmos_containers.items()
    }
    logger = structlog.get_logger("cosmos_agentic_retriever.server")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = ExitStack()

        def build_retrievers() -> dict[str, CorpusRetriever]:
            credential: str | AzureCliCredential | DefaultAzureCredential
            if resolved.cosmos_key is not None:
                credential = resolved.cosmos_key.get_secret_value()
            else:
                credential = (
                    DefaultAzureCredential()
                    if resolved.cosmos_credential == "default"
                    else AzureCliCredential()
                )
                resources.callback(credential.close)
            client = CosmosClient(str(resolved.account_uri), credential=credential)
            resources.callback(client.close)
            database = client.get_database_client(resolved.cosmos_database)
            executor = CosmosExecutor(config=resolved.query_engine)
            return {
                name: CorpusRetriever(
                    container=database.get_container_client(name),
                    schema=config.cosmos_schema,
                    executor=executor,
                    partition_policy=config.partition_policy,
                )
                for name, config in resolved.cosmos_containers.items()
            }

        try:
            engines = (
                retrievers
                if retrievers is not None
                else await anyio.to_thread.run_sync(build_retrievers)
            )
            app.state.retriever = MultiContainerRetriever(
                {
                    ContainerTarget(resolved.cosmos_database, name): engine
                    for name, engine in engines.items()
                },
                max_workers=resolved.query_engine.max_concurrency,
            )
            app.state.chat_client = (
                ChatClient(
                    base_url=str(resolved.llm_base_url),
                    model=resolved.llm_model,
                    api_key=(
                        resolved.llm_api_key.get_secret_value()
                        if resolved.llm_api_key is not None
                        else "unused"
                    ),
                    max_tokens=resolved.llm_max_tokens,
                )
                if resolved.llm_base_url is not None and resolved.llm_model is not None
                else None
            )
            yield
        finally:
            app.state.retriever = None
            app.state.chat_client = None
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(resources.close)

    app = FastAPI(
        title="Cosmos Retriever",
        version="0.1.0",
        description="Search over explicitly configured Cosmos containers.",
        lifespan=lifespan,
    )
    app.state.retriever = None
    app.state.chat_client = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"error": "Invalid search request."}
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        ready = app.state.retriever is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ok" if ready else "unavailable"},
        )

    @app.post("/search")
    async def search(request: SearchRequest) -> JSONResponse:
        active: MultiContainerRetriever | None = app.state.retriever
        if active is None:
            return JSONResponse(
                status_code=503, content={"error": "Service is not ready."}
            )
        if request.database not in (
            None,
            resolved.cosmos_database,
        ) or (
            request.container is not None
            and request.container not in resolved.cosmos_containers
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Only the configured database and containers are supported."
                },
            )
        if request.overrides:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Per-request overrides are not supported. Configure the service instead."
                },
            )
        try:
            names = (
                [request.container]
                if request.container is not None
                else list(resolved.cosmos_containers)
            )
            if request.container_filters is not None and set(
                request.container_filters
            ) != set(names):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "container_filters must name every selected container and no others. Use [] for an unfiltered container."
                    },
                )
            requests = {
                ContainerTarget(resolved.cosmos_database, name): QuerySearchRequest(
                    query=request.query,
                    limit=request.max_documents,
                    text_fields=selected_fields[name].copy(),
                    partition_key=resolved.cosmos_containers[name].partition_key,
                    filters=(
                        request.container_filters[name]
                        if request.container_filters is not None
                        else []
                    ),
                )
                for name in names
            }
            if not tokenize_for_fts(request.query):
                raise QueryCompilationError(
                    "full-text query must contain a searchable term"
                )
            result = await anyio.to_thread.run_sync(
                lambda: active.search(requests, limit=request.max_documents)
            )
            body = {
                "documents": [item.model_dump(mode="json") for item in result.items],
                "searched": [target._asdict() for target in result.searched],
                "errors": [
                    {**target._asdict(), "error": error}
                    for target, error in result.errors.items()
                ],
                "partial": bool(result.errors) and bool(result.searched),
            }
            if not result.searched:
                body["error"] = "Search failed for all selected containers."
                return JSONResponse(status_code=500, content=body)
            return JSONResponse(content=body)
        except RetrievalError as exc:
            return JSONResponse(
                status_code=400, content={"error": str(exc), "type": type(exc).__name__}
            )
        except Exception as exc:
            logger.exception("search_failed", error_type=type(exc).__name__)
            return JSONResponse(status_code=500, content={"error": "Search failed."})

    @app.post("/agent_search")
    async def agent_search(request: AgentSearchRequest) -> JSONResponse:
        active: MultiContainerRetriever | None = app.state.retriever
        client: ChatClient | None = app.state.chat_client
        if active is None:
            return JSONResponse(
                status_code=503, content={"error": "Service is not ready."}
            )
        if client is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Agent search is not configured. Set LLM_BASE_URL and LLM_MODEL."
                },
            )

        runs: list[list[ContainerItem]] = []

        def search(query: str, count: int) -> MultiSearchResult:
            if not tokenize_for_fts(query):
                return MultiSearchResult()
            requests = {
                ContainerTarget(resolved.cosmos_database, name): QuerySearchRequest(
                    query=query,
                    limit=count,
                    text_fields=selected_fields[name].copy(),
                    partition_key=resolved.cosmos_containers[name].partition_key,
                    filters=[],
                )
                for name in resolved.cosmos_containers
            }
            result = active.search(requests, limit=count)
            runs.append(result.items)
            return result

        tool = make_full_text_search_tool(
            search,
            default_max_documents=resolved.agent_max_documents,
            max_documents_cap=resolved.agent_max_documents,
        )
        try:
            result = await anyio.to_thread.run_sync(
                lambda: run_agent_search(
                    request.query,
                    complete=client.complete,
                    tools=[tool],
                    system_prompt=SYSTEM_PROMPT,
                    max_turns=resolved.agent_max_turns,
                )
            )
        except Exception:
            logger.exception("agent_search_failed")
            return JSONResponse(
                status_code=500, content={"error": "Agent search failed."}
            )
        documents = pool_rrf(runs, limit=resolved.agent_max_documents)
        return JSONResponse(
            content={
                "answer": result.answer,
                "documents": [item.model_dump(mode="json") for item in documents],
                "terminal_reason": result.terminal_reason,
                "turns": result.turns,
            }
        )

    return app
