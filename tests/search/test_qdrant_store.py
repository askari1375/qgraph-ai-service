"""Qdrant adapter exercised against an in-memory client (no network, no container)."""

import pytest
from qdrant_client import QdrantClient

from src.search.contracts import ContentType, SearchFilters
from src.search.vector.mapping import PAYLOAD_INDEX_FIELDS, build_point_id, compile_qdrant_filter
from src.search.vector.qdrant_store import (
    QdrantClientStore,
    QdrantError,
    VectorHit,
    VectorPoint,
    build_qdrant_store,
)

# Local Qdrant warns that payload indexes are a no-op; irrelevant to these adapter-contract tests.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def _store() -> QdrantClientStore:
    return QdrantClientStore(QdrantClient(":memory:"))


def _seed(store: QdrantClientStore, name: str = "c1") -> None:
    store.create_collection(name, vector_name="content", dimensions=4, distance="cosine")
    store.create_payload_indexes(name, PAYLOAD_INDEX_FIELDS)
    store.upsert_points(
        name,
        vector_name="content",
        points=[
            VectorPoint(
                point_id=build_point_id("p-ar"),
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"language_code": "ar", "content_type": "quran_ayah"},
            ),
            VectorPoint(
                point_id=build_point_id("p-en"),
                vector=[0.9, 0.8, 0.7, 0.6],
                payload={"language_code": "en", "content_type": "translation"},
            ),
        ],
    )


def test_create_and_config_round_trip():
    store = _store()
    store.create_collection("c1", vector_name="content", dimensions=4, distance="cosine")
    config = store.collection_config("c1")
    assert (config.vector_name, config.dimensions, config.distance) == ("content", 4, "cosine")
    assert store.collection_exists("c1") is True
    assert store.collection_exists("missing") is False


def test_upsert_count_and_filtered_query():
    store = _store()
    _seed(store)
    assert store.count_points("c1") == 2
    filters = SearchFilters(content_types=[ContentType.QURAN_AYAH], languages=["ar"])
    hits = store.query(
        "c1",
        vector=[0.1, 0.2, 0.3, 0.4],
        vector_name="content",
        query_filter=compile_qdrant_filter(filters),
        limit=10,
    )
    assert all(isinstance(hit, VectorHit) for hit in hits)
    assert len(hits) == 1
    assert hits[0].payload["language_code"] == "ar"


def test_alias_resolve_and_swap():
    store = _store()
    store.create_collection("c1", vector_name="content", dimensions=4, distance="cosine")
    store.create_collection("c2", vector_name="content", dimensions=4, distance="cosine")
    store.swap_alias("active", "c1")
    assert store.resolve_alias("active") == "c1"
    store.swap_alias("active", "c2")  # repoint = rollback/activation, atomic
    assert store.resolve_alias("active") == "c2"


def test_resolve_missing_alias_raises():
    store = _store()
    with pytest.raises(QdrantError) as excinfo:
        store.resolve_alias("ghost")
    assert excinfo.value.reason == "semantic_alias_invalid"


def test_list_collections():
    store = _store()
    assert store.list_collections() == []
    store.create_collection("c1", vector_name="content", dimensions=4, distance="cosine")
    store.create_collection("c2", vector_name="content", dimensions=4, distance="cosine")
    assert sorted(store.list_collections()) == ["c1", "c2"]


def test_delete_collection():
    store = _store()
    store.create_collection("c1", vector_name="content", dimensions=4, distance="cosine")
    store.delete_collection("c1")
    assert store.collection_exists("c1") is False


def test_build_qdrant_store_requires_url():
    with pytest.raises(QdrantError) as excinfo:
        build_qdrant_store(url="")
    assert excinfo.value.reason == "qdrant_not_configured"
