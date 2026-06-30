"""Hybrid (lexical + semantic) search readiness: the extra Qdrant/profile/corpus checks."""

from pathlib import Path
from typing import Any

from src.config import Settings
from src.search.embeddings.input import (
    EMBEDDING_INPUT_PROFILE_ID,
    EMBEDDING_INPUT_PROFILE_VERSION,
)
from src.search.indexing.documents import DOCUMENT_SCHEMA_VERSION
from src.search.indexing.mapping import ANALYSIS_PROFILE_VERSION
from src.search.indexing.normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
)
from src.search.vector.profile import (
    CHUNKING_PROFILE_ID,
    CHUNKING_PROFILE_VERSION,
    DISTANCE_METRIC,
    QDRANT_BACKEND_NAME,
    SEMANTIC_INDEX_PROFILE_SCHEMA_VERSION,
    VECTOR_NAME,
    SemanticIndexProfile,
    write_semantic_profile,
)
from src.search.vector.qdrant_store import CollectionConfig, QdrantError
from src.services.search_service import check_search_readiness
from tests.support.embeddings import DeterministicEmbeddingProvider

_COLLECTION = "qgraph-ayah-semantic-20260630-001"
_SNAPSHOT_ID = "snapshot-001"
_SNAPSHOT_HASH = "sha256:abc123"
_PROVIDER = "openai"
_MODEL = "text-embedding-3-large"
_DIMS = 3072


def _settings(profiles_dir: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "opensearch_url": "http://opensearch:9200",
        "qdrant_url": "http://qdrant:6333",
        "search_retrieval_policy": "hybrid_v1",
        "embedding_provider": _PROVIDER,
        "embedding_model": _MODEL,
        "embedding_dimensions": _DIMS,
        "semantic_index_profiles_dir": profiles_dir,
    }
    values.update(overrides)
    return Settings(**values)


def _lexical_profile(**overrides: Any) -> dict[str, Any]:
    profile = {
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "analysis_profile_version": ANALYSIS_PROFILE_VERSION,
        "corpus_snapshot_id": _SNAPSHOT_ID,
        "corpus_snapshot_hash": _SNAPSHOT_HASH,
    }
    profile.update(overrides)
    return profile


def _write_semantic_profile(directory: Path, **overrides: Any) -> None:
    data: dict[str, Any] = {
        "collection_name": _COLLECTION,
        "schema_version": SEMANTIC_INDEX_PROFILE_SCHEMA_VERSION,
        "backend": QDRANT_BACKEND_NAME,
        "corpus_snapshot_id": _SNAPSHOT_ID,
        "corpus_snapshot_hash": _SNAPSHOT_HASH,
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "embedding_input_profile_id": EMBEDDING_INPUT_PROFILE_ID,
        "embedding_input_profile_version": EMBEDDING_INPUT_PROFILE_VERSION,
        "chunking_profile_id": CHUNKING_PROFILE_ID,
        "chunking_profile_version": CHUNKING_PROFILE_VERSION,
        "embedding_provider": _PROVIDER,
        "embedding_model": _MODEL,
        "embedding_dimensions": _DIMS,
        "vector_name": VECTOR_NAME,
        "distance_metric": DISTANCE_METRIC,
        "created_at": "2026-06-30T00:00:00+00:00",
        "document_count": 10,
        "vector_count": 10,
        "included_languages": ["ar", "en", "fa"],
        "source_ids": ["quran-uthmani"],
        "content_types": ["quran_ayah", "translation"],
    }
    data.update(overrides)
    write_semantic_profile(SemanticIndexProfile(**data), directory=directory)


class _Resp:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self) -> Any:
        return self._payload


class _FakeOpenSearch:
    def __init__(self, profile: dict[str, Any]):
        self._profile = profile

    def get(self, path: str) -> _Resp:
        if path.startswith("/_alias/"):
            return _Resp(200, {"idx-001": {}})
        return _Resp(
            200, {"idx-001": {"mappings": {"_meta": {"qgraph_index_profile": self._profile}}}}
        )

    def post(self, path: str, *, json_payload=None, content=None, headers=None) -> _Resp:
        return _Resp(200, {"hits": {"hits": [{"_id": "ayah:1:1:ar"}]}})

    def put(self, path: str, *, json_payload) -> _Resp:  # pragma: no cover
        return _Resp(200, {})

    def delete(self, path: str) -> _Resp:  # pragma: no cover
        return _Resp(200, {})


class _FakeStore:
    def __init__(self, *, count: int = 10, config: CollectionConfig | None = None, fail=None):
        self._count = count
        self._config = config or CollectionConfig(
            vector_name=VECTOR_NAME, dimensions=_DIMS, distance=DISTANCE_METRIC
        )
        self._fail = fail

    def resolve_alias(self, alias: str) -> str:
        if self._fail is not None:
            raise self._fail
        return _COLLECTION

    def count_points(self, name: str) -> int:
        return self._count

    def collection_config(self, name: str) -> CollectionConfig:
        return self._config


def _names(readiness) -> dict[str, bool]:
    return {check.name: check.ok for check in readiness.checks}


def test_hybrid_ready_when_all_checks_pass(tmp_path):
    _write_semantic_profile(tmp_path)
    readiness = check_search_readiness(
        _settings(tmp_path),
        adapter=_FakeOpenSearch(_lexical_profile()),
        store=_FakeStore(),
        provider=DeterministicEmbeddingProvider(),
    )
    assert readiness.ready is True
    assert readiness.active_collection == _COLLECTION
    names = _names(readiness)
    assert names["semantic_alias_single_target"] is True
    assert names["semantic_collection_non_empty"] is True
    assert names["semantic_profile_compatible"] is True
    assert names["semantic_collection_config_match"] is True
    assert names["hybrid_corpus_compatible"] is True


def test_hybrid_not_ready_when_corpus_differs(tmp_path):
    _write_semantic_profile(tmp_path, corpus_snapshot_hash="sha256:DIFFERENT")
    readiness = check_search_readiness(
        _settings(tmp_path),
        adapter=_FakeOpenSearch(_lexical_profile()),
        store=_FakeStore(),
        provider=DeterministicEmbeddingProvider(),
    )
    assert readiness.ready is False
    assert _names(readiness)["hybrid_corpus_compatible"] is False


def test_hybrid_not_ready_when_collection_empty(tmp_path):
    _write_semantic_profile(tmp_path)
    readiness = check_search_readiness(
        _settings(tmp_path),
        adapter=_FakeOpenSearch(_lexical_profile()),
        store=_FakeStore(count=0),
        provider=DeterministicEmbeddingProvider(),
    )
    assert readiness.ready is False
    assert _names(readiness)["semantic_collection_non_empty"] is False


def test_hybrid_not_ready_when_qdrant_unreachable(tmp_path):
    readiness = check_search_readiness(
        _settings(tmp_path),
        adapter=_FakeOpenSearch(_lexical_profile()),
        store=_FakeStore(fail=QdrantError("down", reason="qdrant_unavailable")),
        provider=DeterministicEmbeddingProvider(),
    )
    assert readiness.ready is False
    assert _names(readiness)["qdrant_reachable"] is False


def test_hybrid_not_ready_when_provider_unconfigured(tmp_path):
    readiness = check_search_readiness(
        _settings(tmp_path),
        adapter=_FakeOpenSearch(_lexical_profile()),
        store=_FakeStore(),
        provider=None,
    )
    assert readiness.ready is False
    assert _names(readiness)["embedding_provider_configured"] is False


def test_hybrid_not_ready_on_dimension_drift(tmp_path):
    _write_semantic_profile(tmp_path)
    readiness = check_search_readiness(
        _settings(tmp_path),
        adapter=_FakeOpenSearch(_lexical_profile()),
        store=_FakeStore(
            config=CollectionConfig(
                vector_name=VECTOR_NAME, dimensions=1536, distance=DISTANCE_METRIC
            )
        ),
        provider=DeterministicEmbeddingProvider(),
    )
    assert readiness.ready is False
    assert _names(readiness)["semantic_collection_config_match"] is False
