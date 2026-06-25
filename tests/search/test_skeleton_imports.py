"""Locks the skeleton contract: the package imports cleanly and the not-yet-built behavior is a stub.

When one of these is implemented, the matching assertion here should be deleted in the same change —
that is the intended signal that the skeleton is being filled.
"""

import importlib

import pytest

from src.api.schemas.search import SearchExecuteRequest
from src.search.contracts import QueryContext
from src.search.response_builder import build_execute_response

SKELETON_MODULES = [
    "src.search",
    "src.search.contracts",
    "src.search.opensearch_client",
    "src.search.pipeline",
    "src.search.response_builder",
    "src.search.retrievers",
    "src.search.retrievers.lexical_opensearch",
    "src.search.indexing",
    "src.search.indexing.documents",
    "src.search.indexing.normalization",
    "src.search.indexing.mapping",
    "src.search.indexing.eval_set",
    "src.search.indexing.builder",
    "src.search.indexing.cli",
]


@pytest.mark.parametrize("module_name", SKELETON_MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_response_builder_stub_raises_not_implemented():
    qc = QueryContext(raw_query="mercy")
    with pytest.raises(NotImplementedError):
        build_execute_response([], SearchExecuteRequest(query="mercy"), qc)
