"""Locks the package shape: every retrieval module imports cleanly.

All behavior is now implemented; this just guards against import-time breakage across the package.
"""

import importlib

import pytest

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
