"""Semantic index profile: strictness, immutability, sidecar round-trip, compatibility checks."""

import pytest
from pydantic import ValidationError

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.search.embeddings.contracts import EmbeddingProviderProfile
from src.search.indexing.documents import build_search_documents
from src.search.vector.profile import (
    SemanticIndexProfile,
    build_semantic_profile,
    collection_config_mismatches,
    delete_semantic_profile,
    expected_code_compatibility,
    profile_compatibility_mismatches,
    read_semantic_profile,
    write_semantic_profile,
)
from src.search.vector.qdrant_store import CollectionConfig, QdrantError


def _profile(**overrides) -> SemanticIndexProfile:
    base = dict(
        collection_name="qgraph-ayah-semantic-20260630-001",
        schema_version="qgraph_semantic_index_profile.v2",
        backend="qdrant",
        corpus_snapshot_id="snap-1",
        corpus_snapshot_hash="hash-1",
        document_schema_version="qgraph_search_document.v2",
        normalization_profile_id="qgraph_search_normalization",
        normalization_profile_version="2026-06-22.v1",
        embedding_input_profile_id="qgraph_embedding_input",
        embedding_input_profile_version="2026-06-30.v1",
        chunking_profile_id="qgraph_search_document_unit",
        chunking_profile_version="v1",
        representation_policy_id="qgraph_semantic_representation",
        representation_policy_version="single_document_v1",
        semantic_corpus_policy_id="qgraph_quran_semantic_corpus",
        semantic_corpus_policy_version="ar_arberry_moezzi_v1",
        embedding_provider="cohere",
        embedding_model="embed-v4",
        embedding_dimensions=1024,
        distinguishes_input_modes=True,
        vectors_normalized=True,
        vector_name="content",
        distance_metric="cosine",
        created_at="2026-06-30T00:00:00+00:00",
        document_count=18708,
        vector_count=18708,
        document_id_checksum="sha256:abc",
        default_scope={"arabic_ayahs": True, "surah_names": False},
        included_languages=["ar", "en", "fa"],
        source_ids=["quran-arabic", "en.arberry", "fa.moezzi"],
        content_types=["quran_ayah", "translation"],
    )
    base.update(overrides)
    return SemanticIndexProfile(**base)


def _expected() -> dict:
    return {
        "document_schema_version": "qgraph_search_document.v2",
        "normalization_profile_version": "2026-06-22.v1",
        "embedding_input_profile_version": "2026-06-30.v1",
        "chunking_profile_version": "v1",
        "representation_policy_id": "qgraph_semantic_representation",
        "representation_policy_version": "single_document_v1",
        "semantic_corpus_policy_id": "qgraph_quran_semantic_corpus",
        "semantic_corpus_policy_version": "ar_arberry_moezzi_v1",
        "embedding_provider": "cohere",
        "embedding_model": "embed-v4",
        "embedding_dimensions": 1024,
        "vector_name": "content",
        "distance_metric": "cosine",
    }


def test_profile_rejects_extra_and_missing_fields():
    with pytest.raises(ValidationError):
        _profile(unexpected="x")
    with pytest.raises(ValidationError):
        SemanticIndexProfile(collection_name="c")  # missing the rest


def test_profile_is_frozen():
    profile = _profile()
    with pytest.raises(ValidationError):
        profile.embedding_dimensions = 512


def test_sidecar_round_trip(tmp_path):
    profile = _profile()
    path = write_semantic_profile(profile, directory=tmp_path)
    assert path.exists()
    assert read_semantic_profile(profile.collection_name, directory=tmp_path) == profile


def test_read_missing_profile_raises(tmp_path):
    with pytest.raises(QdrantError) as excinfo:
        read_semantic_profile("nope", directory=tmp_path)
    assert excinfo.value.reason == "semantic_profile_missing"


def test_read_corrupt_profile_raises(tmp_path):
    (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(QdrantError) as excinfo:
        read_semantic_profile("broken", directory=tmp_path)
    assert excinfo.value.reason == "semantic_profile_invalid"


def test_compatibility_passes_on_match():
    assert profile_compatibility_mismatches(_profile(), expected=_expected()) == {}


def test_compatibility_flags_changed_fields():
    mismatches = profile_compatibility_mismatches(
        _profile(embedding_dimensions=512, embedding_model="embed-v3"),
        expected=_expected(),
    )
    assert set(mismatches) == {"embedding_dimensions", "embedding_model"}
    assert mismatches["embedding_dimensions"] == {"expected": 1024, "actual": 512}


def test_collection_config_mismatch_detected():
    profile = _profile()
    good = CollectionConfig(vector_name="content", dimensions=1024, distance="cosine")
    assert collection_config_mismatches(good, profile) == {}
    bad = CollectionConfig(vector_name="content", dimensions=512, distance="dot")
    mismatches = collection_config_mismatches(bad, profile)
    assert set(mismatches) == {"dimensions", "distance"}


def _snapshot() -> QuranCorpusSnapshot:
    return QuranCorpusSnapshot.model_validate(
        {
            "schema_version": "qgraph-corpus-snapshot-v1",
            "corpus_snapshot_id": "snap-9",
            "corpus_snapshot_hash": "sha256:def",
            "produced_at": "2026-06-30T00:00:00Z",
            "filters": {},
            "counts": {},
            "translation_sources": [],
            "surahs": [{"number": 1, "arabic_name": "الفاتحة", "transliteration": "Al-Fatihah"}],
            "ayahs": [
                {
                    "surah_number": 1,
                    "ayah_number": 1,
                    "ayah_global_number": 1,
                    "text_ar": "بسم الله الرحمن الرحيم",
                    "translations": [
                        {
                            "language_code": "en",
                            "source_id": "en-sahih",
                            "source_name": "Sahih International",
                            "text": "In the name of Allah",
                        }
                    ],
                }
            ],
        }
    )


def test_build_semantic_profile_assembles_identity_and_counts():
    snapshot = _snapshot()
    documents = build_search_documents(snapshot)
    provider_profile = EmbeddingProviderProfile(
        provider="cohere", model="embed-v4", dimensions=1024, distinguishes_input_modes=True
    )
    profile = build_semantic_profile(
        collection_name="qgraph-ayah-semantic-20260630-001",
        snapshot=snapshot,
        documents=documents,
        provider_profile=provider_profile,
        vector_name="content",
    )
    assert profile.collection_name == "qgraph-ayah-semantic-20260630-001"
    assert profile.corpus_snapshot_id == "snap-9"
    assert profile.embedding_provider == "cohere"
    assert profile.embedding_model == "embed-v4"
    assert profile.embedding_dimensions == 1024
    assert profile.document_count == profile.vector_count == len(documents)
    assert profile.included_languages == ["ar", "en"]
    # Built profile passes the code-constant compatibility gate it was built from.
    assert profile_compatibility_mismatches(profile, expected=expected_code_compatibility()) == {}


def test_build_semantic_profile_rejects_empty_documents():
    with pytest.raises(ValueError):
        build_semantic_profile(
            collection_name="c",
            snapshot=_snapshot(),
            documents=[],
            provider_profile=EmbeddingProviderProfile(provider="p", model="m", dimensions=8),
            vector_name="content",
        )


def test_delete_semantic_profile_is_idempotent(tmp_path):
    profile = _profile()
    write_semantic_profile(profile, directory=tmp_path)
    delete_semantic_profile(profile.collection_name, directory=tmp_path)
    delete_semantic_profile(profile.collection_name, directory=tmp_path)  # missing_ok
    with pytest.raises(QdrantError):
        read_semantic_profile(profile.collection_name, directory=tmp_path)
