"""Entrypoint for the Cosmos Agentic Retriever service.

Run with:
    python -m cosmos_agentic_retriever serve
"""

import argparse
import sys

import uvicorn
from pydantic import ValidationError
from pydantic_settings import SettingsError

from cosmos_agentic_retriever.config import RetrieverConfig, get_config
from cosmos_agentic_retriever.query_engine.types import RetrievalError
from cosmos_agentic_retriever.server import create_app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosmos-agentic-retriever")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser(
        "serve", help="Run the HTTP service called by the MCP toolkit."
    )
    serve.add_argument(
        "--host", default=None, help="Bind address (default: HOST or 127.0.0.1)."
    )
    serve.add_argument(
        "--port", type=int, default=None, help="Bind port (default: PORT or 9000)."
    )
    return parser


def _cmd_serve(args: argparse.Namespace) -> int:
    settings = get_config()
    overrides = {
        name: getattr(args, name)
        for name in ("host", "port")
        if getattr(args, name) is not None
    }
    if overrides:
        values = {
            name: getattr(settings, name) for name in RetrieverConfig.model_fields
        }
        settings = RetrieverConfig(**{**values, **overrides})
    app = create_app(settings)
    uvicorn.run(
        app, host=settings.host, port=settings.port, log_level=settings.log_level
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _cmd_serve(args)
    except (ValidationError, SettingsError, RetrievalError):
        print(
            "Invalid service configuration. Check the connection, schema, and server settings.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
