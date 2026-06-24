"""Locks the skeleton contract: the package imports cleanly and every behavior stub is a *stub*.

When one of these is implemented, the matching assertion here should be deleted in the same change —
that is the intended signal that the skeleton is being filled.
"""

import importlib

import pytest

from src.api.schemas.search import SearchExecuteRequest
from src.search.contracts import QueryContext, SearchFilters
from src.search.indexing import builder, cli, documents, mapping, normalization
from src.search.pipeline import RetrievalPipeline
from src.search.response_builder import build_execute_response
from src.search.retrievers.lexical_opensearch import LexicalRetriever

SKELETON_MODULES = [
    "src.search",
    "src.search.contracts",
    "src.search.pipeline",
    "src.search.response_builder",
    "src.search.retrievers",
    "src.search.retrievers.lexical_opensearch",
    "src.search.indexing",
    "src.search.indexing.documents",
    "src.search.indexing.normalization",
    "src.search.indexing.mapping",
    "src.search.indexing.builder",
    "src.search.indexing.cli",
]


@pytest.mark.parametrize("module_name", SKELETON_MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_pipeline_stubs_raise_not_implemented():
    pipeline = RetrievalPipeline(retrievers=[LexicalRetriever()])
    qc = QueryContext(raw_query="mercy")
    with pytest.raises(NotImplementedError):
        pipeline.run(qc)
    with pytest.raises(NotImplementedError):
        pipeline._fuse([])


def test_retriever_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        LexicalRetriever().retrieve(QueryContext(raw_query="mercy"))


def test_response_builder_stub_raises_not_implemented():
    qc = QueryContext(raw_query="mercy")
    with pytest.raises(NotImplementedError):
        build_execute_response([], SearchExecuteRequest(query="mercy"), qc)


def test_search_filters_compile_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        SearchFilters.from_request_filters({})
    with pytest.raises(NotImplementedError):
        SearchFilters().to_opensearch_filter()


def test_indexing_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        documents.build_search_documents(None)
    with pytest.raises(NotImplementedError):
        normalization.normalize_text("الرحمن", "ar")
    with pytest.raises(NotImplementedError):
        mapping.build_index_settings()
    with pytest.raises(NotImplementedError):
        builder.build_index()
    with pytest.raises(NotImplementedError):
        builder.activate_index("v1")
    with pytest.raises(NotImplementedError):
        builder.index_status()
    with pytest.raises(NotImplementedError):
        cli.main([])
