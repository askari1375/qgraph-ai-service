"""Hybrid (lexical + semantic) retrieval path: fusion provenance and no silent fallback."""

from typing import Any

import pytest

from src.api.schemas.search import SearchExecuteRequest
from src.config import Settings
from src.search.indexing.documents import DOCUMENT_SCHEMA_VERSION
from src.search.indexing.mapping import ANALYSIS_PROFILE_VERSION
from src.search.indexing.normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
)
from src.search.vector.qdrant_store import QdrantError, VectorHit
from src.services.search_service import SearchRetrievalError, build_search_execute_response
from tests.support.embeddings import DeterministicEmbeddingProvider

_COLLECTION = "qgraph-ayah-semantic-20260630-001"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "opensearch_url": "http://opensearch:9200",
        "qdrant_url": "http://qdrant:6333",
        "search_retrieval_policy": "hybrid_v1",
    }
    values.update(overrides)
    return Settings(**values)


def _profile() -> dict[str, Any]:
    return {
        "index_id": "qgraph-ayah-lexical-20260625-001",
        "corpus_snapshot_id": "snapshot-001",
        "corpus_snapshot_hash": "sha256:abc123",
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "analysis_profile_version": ANALYSIS_PROFILE_VERSION,
    }


class _Resp:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self) -> Any:
        return self._payload


class _FakeOpenSearch:
    """Returns the Arabic ayah for the ayah scope; aggregations for the size:0 query."""

    def __init__(self, ayah_hits):
        self._ayah_hits = ayah_hits

    def get(self, path: str) -> _Resp:
        return _Resp(
            200,
            {
                "qgraph-ayah-lexical-20260625-001": {
                    "mappings": {"_meta": {"qgraph_index_profile": _profile()}}
                }
            },
        )

    def post(self, path: str, *, json_payload=None, content=None, headers=None) -> _Resp:
        body = json_payload or {}
        if "aggs" in body:
            return _Resp(200, {"hits": {"hits": []}, "aggregations": {}})
        return _Resp(200, {"hits": {"hits": self._ayah_hits}})

    def put(self, path: str, *, json_payload) -> _Resp:  # pragma: no cover
        return _Resp(200, {})

    def delete(self, path: str) -> _Resp:  # pragma: no cover
        return _Resp(200, {})


class _FakeStore:
    def __init__(self, hits=None, *, fail: QdrantError | None = None):
        self._hits = hits or []
        self._fail = fail

    def resolve_alias(self, alias: str) -> str:
        return _COLLECTION

    def query(self, name: str, **kwargs: Any) -> list[VectorHit]:
        if self._fail is not None:
            raise self._fail
        return self._hits


def _arabic_hit(score: float) -> dict[str, Any]:
    return {
        "_id": "ayah:1:1:ar",
        "_score": score,
        "_source": {
            "id": "ayah:1:1:ar",
            "canonical_content_id": "ayah:1:1",
            "content_ar": "بسم الله الرحمن الرحيم",
            "metadata": {
                "content_type": "quran_ayah",
                "surah_number": 1,
                "ayah_number": 1,
                "ayah_global_number": 1,
                "language_code": "ar",
                "source_name": "Quran Arabic",
            },
        },
        "highlight": {"content_ar": ["بسم الله <mark>الرحمن</mark> الرحيم"]},
    }


def _semantic_hit(document_id: str, score: float) -> VectorHit:
    return VectorHit(
        point_id="p1",
        score=score,
        payload={
            "document_id": document_id,
            "canonical_content_id": "ayah:1:1",
            "content_type": "quran_ayah",
            "text": "بسم الله الرحمن الرحيم",
            "surah_number": 1,
            "ayah_number": 1,
            "ayah_global_number": 1,
            "language_code": "ar",
            "source_name": "Quran Arabic",
        },
    )


def _request() -> SearchExecuteRequest:
    # Translations off so only the Arabic scope runs (semantic hits are Arabic ayat).
    return SearchExecuteRequest(query="رحمت", filters={"include_translations": False})


def test_hybrid_fuses_and_records_provenance():
    response = build_search_execute_response(
        _request(),
        settings=_settings(),
        adapter=_FakeOpenSearch([_arabic_hit(9.0)]),
        store=_FakeStore([_semantic_hit("ayah:1:1:ar", 0.88)]),
        provider=DeterministicEmbeddingProvider(dimensions=8),
    )

    assert response.metadata["backend"] == "hybrid_rrf_v1"
    assert response.metadata["retrieval_policy"] == "hybrid_v1"
    assert response.metadata["semantic_collection"] == _COLLECTION
    assert response.metadata["embedding_provider"] == "deterministic-test"
    assert response.metadata["fusion_profile"]["profile_id"] == "qgraph_rrf"

    item = response.blocks[-1].items[0]
    assert item.match_metadata["document_id"] == "ayah:1:1:ar"
    # The same ayah came from both backends, so per-item provenance carries both ranks + the fused score.
    assert item.provenance["lexical_rank"] == 1
    assert item.provenance["semantic_rank"] == 1
    assert item.provenance["fused_score"] > 0
    assert item.provenance["fused_rank"] == 1


def test_hybrid_with_no_semantic_hits_still_returns_lexical():
    response = build_search_execute_response(
        _request(),
        settings=_settings(),
        adapter=_FakeOpenSearch([_arabic_hit(9.0)]),
        store=_FakeStore([]),
        provider=DeterministicEmbeddingProvider(dimensions=8),
    )
    item = response.blocks[-1].items[0]
    assert item.match_metadata["document_id"] == "ayah:1:1:ar"
    assert item.provenance["lexical_rank"] == 1
    assert "semantic_rank" not in item.provenance


def test_hybrid_does_not_silently_fall_back_when_qdrant_fails():
    with pytest.raises(SearchRetrievalError) as excinfo:
        build_search_execute_response(
            _request(),
            settings=_settings(),
            adapter=_FakeOpenSearch([_arabic_hit(9.0)]),
            store=_FakeStore(fail=QdrantError("down", reason="qdrant_unavailable")),
            provider=DeterministicEmbeddingProvider(dimensions=8),
        )
    assert excinfo.value.reason == "qdrant_unavailable"


def test_hybrid_does_not_silently_fall_back_when_provider_fails():
    class _BrokenProvider(DeterministicEmbeddingProvider):
        def embed_query(self, text: str) -> list[float]:
            from src.search.embeddings.contracts import EmbeddingError

            raise EmbeddingError("provider down", reason="embedding_provider_unavailable")

    with pytest.raises(SearchRetrievalError) as excinfo:
        build_search_execute_response(
            _request(),
            settings=_settings(),
            adapter=_FakeOpenSearch([_arabic_hit(9.0)]),
            store=_FakeStore([]),
            provider=_BrokenProvider(dimensions=8),
        )
    assert excinfo.value.reason == "embedding_provider_unavailable"


def test_lexical_policy_ignores_semantic_backends():
    # In lexical_v1 the provider/store are never used, even if a broken one is passed.
    class _BoomProvider(DeterministicEmbeddingProvider):
        def embed_query(self, text: str) -> list[float]:
            raise AssertionError("provider must not be called in lexical_v1")

    response = build_search_execute_response(
        SearchExecuteRequest(query="رحمت", filters={"include_translations": False}),
        settings=_settings(search_retrieval_policy="lexical_v1"),
        adapter=_FakeOpenSearch([_arabic_hit(9.0)]),
        store=_FakeStore(fail=QdrantError("down", reason="qdrant_unavailable")),
        provider=_BoomProvider(dimensions=8),
    )
    assert response.metadata["backend"] == "open_search"
    assert response.metadata["retrieval_policy"] == "lexical_v1"
    assert response.blocks[-1].items[0].provenance["lexical_score"] == 9.0
