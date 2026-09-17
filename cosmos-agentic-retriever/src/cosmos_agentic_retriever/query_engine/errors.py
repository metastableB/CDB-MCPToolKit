from __future__ import annotations


class RetrievalError(Exception):
    pass


class UnsafeCosmosPathError(RetrievalError):
    pass


class QueryCompilationError(RetrievalError):
    pass
