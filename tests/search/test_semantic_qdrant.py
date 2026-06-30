"""Semantic Qdrant retriever: payload mapping, filter reuse, and one-embedding orchestration."""

from typing import Any

import pytest
from qdrant_client import models

from src.search.contracts import (
    RETRIEVER_QDRANT_DENSE,
    ContentType,
    QueryContext,
    SearchFilters,
)
from src.search.embeddings.query import embed_query_for_search
from src.search.retrievers.semantic_qdrant import (
    DEFAULT_CANDIDATE_POOL,
    SemanticQdrantRetriever,
    parse_semantic_candidates,
)
from src.search.vector.qdrant_store import QdrantError, VectorHit
from tests.support.embeddings import DeterministicEmbeddingProvider

_ALIAS = "qgraph-ayah-semantic-active"
_COLLECTION = "qgraph-ayah-semantic-20260630-001"
_VECTOR_NAME = "content"


class _FakeStore:
    def __init__(self, hits: list[VectorHit] | None = None):
        self._hits = hits or []
        self.queries: list[dict[str, Any]] = []

    def resolve_alias(self, alias: str) -> str:
        assert alias == _ALIAS
        return _COLLECTION

    def query(
        self,
        name: str,
        *,
        vector: Any,
        vector_name: str,
        query_filter: Any = None,
        limit: int = 50,
        with_payload: bool = True,
    ) -> list[VectorHit]:
        self.queries.append(
            {
                "name": name,
                "vector": vector,
                "vector_name": vector_name,
                "query_filter": query_filter,
                "limit": limit,
            }
        )
        return self._hits


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "document_id": "ayah:2:255:ar",
        "canonical_content_id": "ayah:2:255",
        "content_type": ContentType.QURAN_AYAH.value,
        "text": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ",
        "surah_number": 2,
        "ayah_number": 255,
        "ayah_global_number": 262,
        "language_code": "ar",
        "source_id": "quran-uthmani",
        "source_name": "Uthmani",
    }
    payload.update(overrides)
    return payload


def _retriever(store: _FakeStore) -> SemanticQdrantRetriever:
    return SemanticQdrantRetriever(store, _ALIAS, _VECTOR_NAME)


def _context(**overrides: Any) -> QueryContext:
    values: dict[str, Any] = {"raw_query": "mercy", "top_k": 5, "query_embedding": [0.1, 0.2]}
    values.update(overrides)
    return QueryContext(**values)


def test_implements_retriever_name():
    assert _retriever(_FakeStore()).name == RETRIEVER_QDRANT_DENSE


def test_maps_payload_to_candidate():
    store = _FakeStore([VectorHit(point_id="p1", score=0.87, payload=_payload())])
    [candidate] = _retriever(store).retrieve(_context())
    assert candidate.document_id == "ayah:2:255:ar"
    assert candidate.canonical_content_id == "ayah:2:255"
    assert candidate.content_type is ContentType.QURAN_AYAH
    assert candidate.retriever == RETRIEVER_QDRANT_DENSE
    assert candidate.score == 0.87
    assert candidate.rank == 1
    assert candidate.metadata["surah_number"] == 2
    assert candidate.metadata["source_name"] == "Uthmani"
    assert candidate.debug == {"semantic_score": 0.87, "semantic_rank": 1}


def test_ranks_are_sequential():
    store = _FakeStore(
        [
            VectorHit(point_id="p1", score=0.9, payload=_payload(document_id="ayah:1:1:ar")),
            VectorHit(point_id="p2", score=0.8, payload=_payload(document_id="ayah:1:2:ar")),
        ]
    )
    candidates = _retriever(store).retrieve(_context())
    assert [c.rank for c in candidates] == [1, 2]


def test_missing_embedding_raises():
    with pytest.raises(QdrantError) as excinfo:
        _retriever(_FakeStore()).retrieve(_context(query_embedding=None))
    assert excinfo.value.reason == "semantic_query_embedding_missing"


def test_compiles_filters_and_pools_candidates():
    store = _FakeStore()
    filters = SearchFilters(content_types=[ContentType.QURAN_AYAH], surah_numbers=[2])
    _retriever(store).retrieve(_context(filters=filters, top_k=3))
    call = store.queries[0]
    assert call["name"] == _COLLECTION
    assert isinstance(call["query_filter"], models.Filter)
    # top_k (3) is below the pool, so the larger pool is fetched for fusion.
    assert call["limit"] == DEFAULT_CANDIDATE_POOL


def test_top_k_above_pool_wins():
    store = _FakeStore()
    SemanticQdrantRetriever(store, _ALIAS, _VECTOR_NAME, candidate_pool=10).retrieve(
        _context(top_k=25)
    )
    assert store.queries[0]["limit"] == 25


def test_unrecognized_content_type_raises():
    store = _FakeStore(
        [VectorHit(point_id="p1", score=0.5, payload=_payload(content_type="bogus"))]
    )
    with pytest.raises(QdrantError) as excinfo:
        _retriever(store).retrieve(_context())
    assert excinfo.value.reason == "unexpected_content_type"


def test_blank_document_id_is_skipped():
    store = _FakeStore([VectorHit(point_id="p1", score=0.5, payload=_payload(document_id=""))])
    assert parse_semantic_candidates(store._hits) == []


def test_one_embedding_reused_across_scopes():
    provider = DeterministicEmbeddingProvider(dimensions=8)
    base = QueryContext(raw_query="رحمت", detected_language="fa", top_k=5)
    embedding = embed_query_for_search(provider, base)

    store = _FakeStore()
    ayah_ctx = base.model_copy(update={"query_embedding": embedding})
    translation_ctx = base.model_copy(update={"query_embedding": embedding})
    retriever = _retriever(store)
    retriever.retrieve(ayah_ctx)
    retriever.retrieve(translation_ctx)

    # Both scopes queried with the identical, single embedding object.
    assert store.queries[0]["vector"] is embedding
    assert store.queries[1]["vector"] is embedding
    assert len(embedding) == 8
