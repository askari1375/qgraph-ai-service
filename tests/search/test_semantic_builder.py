"""Semantic builder exercised against in-memory Qdrant + a deterministic provider (no network/API)."""

import pytest
from qdrant_client import QdrantClient

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.config import Settings
from src.search.vector.profile import read_semantic_profile
from src.search.vector.qdrant_store import QdrantClientStore, QdrantError
from src.search.vector_indexing import builder
from tests.support.embeddings import DeterministicEmbeddingProvider

# Local Qdrant warns that payload indexes are a no-op; irrelevant to these build-contract tests.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def _store() -> QdrantClientStore:
    return QdrantClientStore(QdrantClient(":memory:"))


def _provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(dimensions=8)


def _settings(tmp_path) -> Settings:
    return Settings(
        qdrant_collection_alias="qgraph-ayah-semantic-active",
        qdrant_collection_prefix="qgraph-ayah-semantic",
        qdrant_vector_name="content",
        semantic_index_profiles_dir=tmp_path,
    )


def _ayah(surah: int, ayah: int, global_number: int) -> dict:
    # The curated semantic corpus embeds Arabic + the canonical English (Arberry) and Persian (Moezzi)
    # translations only, so the fixture carries exactly those translation sources.
    return {
        "surah_number": surah,
        "ayah_number": ayah,
        "ayah_global_number": global_number,
        "text_ar": "بسم الله الرحمن الرحيم",
        "translations": [
            {
                "language_code": "en",
                "source_id": "en.arberry",
                "source_name": "Arberry",
                "text": "In the Name of God, the Merciful, the Compassionate",
            },
            {
                "language_code": "fa",
                "source_id": "fa.moezzi",
                "source_name": "Moezzi",
                "text": "به نام خداوند بخشنده مهربان",
            },
        ],
    }


def _snapshot(*, ayahs=None, surahs=None) -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "qgraph-corpus-snapshot-v1",
            "corpus_snapshot_id": "snapshot-001",
            "corpus_snapshot_hash": "sha256:abc123",
            "produced_at": "2026-06-30T10:00:00Z",
            "filters": {},
            "counts": {},
            "translation_sources": [],
            "surahs": surahs
            if surahs is not None
            else [{"number": 1, "arabic_name": "الفاتحة", "transliteration": "Al-Fatihah"}],
            "ayahs": ayahs if ayahs is not None else [_ayah(1, 1, 1), _ayah(1, 2, 2)],
        }
    )


def _patch_corpus(monkeypatch, snapshot: QuranCorpusSnapshot) -> None:
    class _FakeCorpusClient:
        def fetch_quran_snapshot(self, **_kwargs) -> QuranCorpusSnapshot:
            return snapshot

    monkeypatch.setattr(builder, "build_django_corpus_client", lambda settings: _FakeCorpusClient())


def test_build_creates_collection_and_writes_profile(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())
    settings, store = _settings(tmp_path), _store()

    report = builder.build_semantic_collection(settings=settings, store=store, provider=_provider())

    assert report["ok"] is True
    assert report["activated"] is False
    assert report["collection"].startswith("qgraph-ayah-semantic-")
    # 2 ayat x (arabic + arberry + moezzi); surah-name docs are excluded by the corpus policy.
    assert report["document_count"] == 6
    assert report["embedding_provider"] == "deterministic-test"
    assert store.count_points(report["collection"]) == 6
    assert store.collection_config(report["collection"]).dimensions == 8
    # Sidecar written; build alone never activates.
    assert read_semantic_profile(report["collection"], directory=tmp_path).vector_count == 6
    with pytest.raises(QdrantError):
        store.resolve_alias(settings.qdrant_collection_alias)


def test_build_with_activate_swaps_alias(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())
    settings, store = _settings(tmp_path), _store()

    report = builder.build_semantic_collection(
        settings=settings, store=store, provider=_provider(), activate=True
    )

    assert report["activated"] is True
    assert store.resolve_alias(settings.qdrant_collection_alias) == report["collection"]


def test_count_mismatch_blocks_activation(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())

    class _MiscountStore(QdrantClientStore):
        def count_points(self, name: str) -> int:
            return super().count_points(name) + 1  # never matches the expected document count

    settings, store = _settings(tmp_path), _MiscountStore(QdrantClient(":memory:"))
    report = builder.build_semantic_collection(
        settings=settings, store=store, provider=_provider(), activate=True
    )

    assert report["ok"] is False
    assert report["activated"] is False
    assert "count_mismatch" in report["validation"]["hard_failures"]


def test_dry_run_writes_nothing_and_skips_provider(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())

    class _ExplodingProvider(DeterministicEmbeddingProvider):
        def embed_documents(self, texts):
            raise AssertionError("dry-run must not call the embedding provider")

    settings, store = _settings(tmp_path), _store()
    report = builder.build_semantic_collection(
        settings=settings, store=store, provider=_ExplodingProvider(dimensions=8), dry_run=True
    )

    assert report["dry_run"] is True
    # Surah-name docs are dropped; only the curated Arabic + Arberry + Moezzi ayah docs remain.
    assert report["language_counts"] == {"ar": 2, "en": 2, "fa": 2}
    assert report["embedding_dimensions"] == 8
    assert store.list_collections() == []  # nothing created


def test_build_activate_then_status(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())
    settings, store = _settings(tmp_path), _store()
    report = builder.build_semantic_collection(settings=settings, store=store, provider=_provider())

    activation = builder.activate_semantic_collection(
        report["collection"], settings=settings, store=store
    )
    assert activation["active_collection"] == report["collection"]

    status = builder.semantic_status(settings=settings, store=store)
    assert status["active_collection"] == report["collection"]
    assert status["compatible"] is True
    assert status["point_count"] == 6


def test_activate_other_collection_is_rollback(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())
    settings, store = _settings(tmp_path), _store()
    first = builder.build_semantic_collection(settings=settings, store=store, provider=_provider())
    second = builder.build_semantic_collection(settings=settings, store=store, provider=_provider())
    assert first["collection"] != second["collection"]

    builder.activate_semantic_collection(first["collection"], settings=settings, store=store)
    builder.activate_semantic_collection(second["collection"], settings=settings, store=store)
    assert store.resolve_alias(settings.qdrant_collection_alias) == second["collection"]

    # Repointing back to the earlier collection is rollback — no restart, just an alias swap.
    builder.activate_semantic_collection(first["collection"], settings=settings, store=store)
    assert store.resolve_alias(settings.qdrant_collection_alias) == first["collection"]


def test_activate_delete_old_removes_previous_and_sidecar(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())
    settings, store = _settings(tmp_path), _store()
    first = builder.build_semantic_collection(settings=settings, store=store, provider=_provider())
    second = builder.build_semantic_collection(settings=settings, store=store, provider=_provider())

    builder.activate_semantic_collection(first["collection"], settings=settings, store=store)
    builder.activate_semantic_collection(
        second["collection"], settings=settings, store=store, delete_old=True
    )

    assert store.collection_exists(first["collection"]) is False
    with pytest.raises(QdrantError) as excinfo:
        read_semantic_profile(first["collection"], directory=tmp_path)
    assert excinfo.value.reason == "semantic_profile_missing"


def test_empty_corpus_raises(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot(ayahs=[], surahs=[]))
    settings, store = _settings(tmp_path), _store()
    with pytest.raises(QdrantError) as excinfo:
        builder.build_semantic_collection(settings=settings, store=store, provider=_provider())
    assert excinfo.value.reason == "empty_corpus"


def test_build_without_provider_raises_not_configured(monkeypatch, tmp_path):
    _patch_corpus(monkeypatch, _snapshot())
    settings, store = _settings(tmp_path), _store()
    # No injected provider and none configured yet -> the factory seam fails loudly (no fake fallback).
    from src.search.embeddings.contracts import EmbeddingError

    with pytest.raises(EmbeddingError) as excinfo:
        builder.build_semantic_collection(settings=settings, store=store)
    assert excinfo.value.reason == "embedding_provider_not_configured"


def test_status_without_alias_reports_none(tmp_path):
    settings, store = _settings(tmp_path), _store()
    status = builder.semantic_status(settings=settings, store=store)
    assert status["active_collection"] is None
    assert status["compatible"] is None
