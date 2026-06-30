"""Semantic eval runner: per-case reporting, the CONFIRMED hard gate, and the build-time gate."""

import pytest
from qdrant_client import QdrantClient

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.config import Settings
from src.search.contracts import ContentType
from src.search.eval.runner import evaluate_semantic_collection
from src.search.eval.semantic_eval_set import SEMANTIC_GOLDEN_QUERIES
from src.search.indexing.eval_set import ExpectationStatus, GoldenQuery
from src.search.vector.qdrant_store import QdrantClientStore
from src.search.vector_indexing import builder
from tests.support.embeddings import DeterministicEmbeddingProvider

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def _ayah(surah: int, ayah: int, global_number: int) -> dict:
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
                "text": "x",
            },
            {"language_code": "fa", "source_id": "fa.moezzi", "source_name": "Moezzi", "text": "ی"},
        ],
    }


def _snapshot() -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "qgraph-corpus-snapshot-v1",
            "corpus_snapshot_id": "snapshot-001",
            "corpus_snapshot_hash": "sha256:abc123",
            "produced_at": "2026-06-30T10:00:00Z",
            "filters": {},
            "counts": {},
            "translation_sources": [],
            "surahs": [{"number": 1, "arabic_name": "الفاتحة", "transliteration": "Al-Fatihah"}],
            "ayahs": [_ayah(1, 1, 1), _ayah(1, 2, 2)],
        }
    )


def _build(monkeypatch, tmp_path):
    class _FakeCorpusClient:
        def fetch_quran_snapshot(self, **_kwargs):
            return _snapshot()

    monkeypatch.setattr(builder, "build_django_corpus_client", lambda settings: _FakeCorpusClient())
    store = QdrantClientStore(QdrantClient(":memory:"))
    provider = DeterministicEmbeddingProvider(dimensions=8)
    settings = Settings(
        qdrant_collection_alias="qgraph-ayah-semantic-active",
        qdrant_collection_prefix="qgraph-ayah-semantic",
        qdrant_vector_name="content",
        semantic_index_profiles_dir=tmp_path,
    )
    report = builder.build_semantic_collection(settings=settings, store=store, provider=provider)
    return store, provider, report


def test_evaluate_runs_every_case_and_passes_while_all_pending(monkeypatch, tmp_path):
    store, provider, _ = _build(monkeypatch, tmp_path)
    collection = store.list_collections()[0]

    report = evaluate_semantic_collection(
        collection, store=store, provider=provider, vector_name="content"
    )

    assert report["ok"] is True
    assert report["hard_failures"] == []
    assert report["case_count"] == len(SEMANTIC_GOLDEN_QUERIES)
    assert report["eval_set_version"]
    assert all("top_canonical_ids" in case for case in report["cases"])


def test_confirmed_missing_must_include_is_a_hard_failure(monkeypatch, tmp_path):
    store, provider, _ = _build(monkeypatch, tmp_path)
    collection = store.list_collections()[0]
    case = GoldenQuery(
        id="confirmed-miss",
        query="رحمت",
        language="fa",
        scope=(ContentType.QURAN_AYAH,),
        expected_content_types=(ContentType.QURAN_AYAH,),
        expected_language="ar",
        must_include_canonical_ids=("ayah:99:99",),  # not in the 2-ayah fixture
        status=ExpectationStatus.CONFIRMED,
        guards="hard gate",
    )

    report = evaluate_semantic_collection(
        collection, store=store, provider=provider, vector_name="content", cases=(case,)
    )

    assert report["ok"] is False
    assert report["hard_failures"] == ["confirmed-miss"]
    assert report["cases"][0]["missing_canonical_ids"] == ["ayah:99:99"]


def test_build_eval_gate_blocks_activation(monkeypatch, tmp_path):
    # Force the eval to report a hard failure; a build that requested activation must not activate.
    def _hard_fail(collection, **_kwargs):
        return {"collection": collection, "ok": False, "hard_failures": ["x"], "soft_misses": []}

    monkeypatch.setattr(builder, "evaluate_semantic_collection", _hard_fail)

    class _FakeCorpusClient:
        def fetch_quran_snapshot(self, **_kwargs):
            return _snapshot()

    monkeypatch.setattr(builder, "build_django_corpus_client", lambda settings: _FakeCorpusClient())
    store = QdrantClientStore(QdrantClient(":memory:"))
    settings = Settings(
        qdrant_collection_alias="qgraph-ayah-semantic-active",
        qdrant_collection_prefix="qgraph-ayah-semantic",
        qdrant_vector_name="content",
        semantic_index_profiles_dir=tmp_path,
    )

    report = builder.build_semantic_collection(
        settings=settings,
        store=store,
        provider=DeterministicEmbeddingProvider(dimensions=8),
        activate=True,
    )

    assert report["ok"] is False
    assert report["activated"] is False
    assert report["evaluation"]["ok"] is False
