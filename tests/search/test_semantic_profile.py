"""Semantic index profile: strictness, immutability, sidecar round-trip, compatibility checks."""

import pytest
from pydantic import ValidationError

from src.search.vector.profile import (
    SemanticIndexProfile,
    collection_config_mismatches,
    profile_compatibility_mismatches,
    read_semantic_profile,
    write_semantic_profile,
)
from src.search.vector.qdrant_store import CollectionConfig, QdrantError


def _profile(**overrides) -> SemanticIndexProfile:
    base = dict(
        collection_name="qgraph-ayah-semantic-20260630-001",
        schema_version="qgraph_semantic_index_profile.v1",
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
        embedding_provider="cohere",
        embedding_model="embed-v4",
        embedding_dimensions=1024,
        vector_name="content",
        distance_metric="cosine",
        created_at="2026-06-30T00:00:00+00:00",
        document_count=187307,
        vector_count=187307,
        included_languages=["ar", "en", "fa"],
        source_ids=["quran-uthmani"],
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
