"""Strict semantic-index profile, its JSON sidecar store, and compatibility checks.

Qdrant has no OpenSearch-style ``_meta`` mapping, so each immutable physical collection gets one JSON
profile sidecar; the Qdrant alias stays the source of truth for which collection serves. This mirrors
the lexical ``build_index_profile`` provenance and ``compatibility_mismatches`` gate.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.search.vector.qdrant_store import CollectionConfig, QdrantError

SEMANTIC_INDEX_PROFILE_SCHEMA_VERSION = "qgraph_semantic_index_profile.v1"
QDRANT_BACKEND_NAME = "qdrant"
# The semantic side reuses the existing per-(text, language, source) search documents as the embedded
# unit — no new Quran chunking. Recorded as an explicit, immutable profile.
CHUNKING_PROFILE_ID = "qgraph_search_document_unit"
CHUNKING_PROFILE_VERSION = "v1"
VECTOR_NAME = "content"
DISTANCE_METRIC = "cosine"


class SemanticIndexProfile(BaseModel):
    """Immutable identity of one physical Qdrant collection.

    Frozen: a built profile is a fact about an existing collection. Changing any embedding/identity
    field means building and activating a new collection, never editing this in place.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection_name: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    corpus_snapshot_id: str = Field(min_length=1)
    corpus_snapshot_hash: str = Field(min_length=1)
    document_schema_version: str = Field(min_length=1)
    normalization_profile_id: str = Field(min_length=1)
    normalization_profile_version: str = Field(min_length=1)
    embedding_input_profile_id: str = Field(min_length=1)
    embedding_input_profile_version: str = Field(min_length=1)
    chunking_profile_id: str = Field(min_length=1)
    chunking_profile_version: str = Field(min_length=1)
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int = Field(gt=0)
    vector_name: str = Field(min_length=1)
    distance_metric: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    document_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    included_languages: list[str]
    source_ids: list[str]
    content_types: list[str]


#: Fields that must match the running code / runtime expectation for a collection to be servable.
#: ``corpus_snapshot_id``/``corpus_snapshot_hash`` are provenance only here — the lexical↔semantic
#: corpus equality gate lives in the hybrid path, not in single-backend compatibility.
_COMPATIBILITY_FIELDS = (
    "document_schema_version",
    "normalization_profile_version",
    "embedding_input_profile_version",
    "chunking_profile_version",
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "vector_name",
    "distance_metric",
)


def profile_compatibility_mismatches(
    profile: SemanticIndexProfile, *, expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return ``{field: {expected, actual}}`` for each compatibility field that disagrees."""
    actual = profile.model_dump()
    return {
        field: {"expected": expected[field], "actual": actual.get(field)}
        for field in _COMPATIBILITY_FIELDS
        if field in expected and actual.get(field) != expected[field]
    }


def collection_config_mismatches(
    config: CollectionConfig, profile: SemanticIndexProfile
) -> dict[str, dict[str, Any]]:
    """Compare the live Qdrant collection config against the profile.

    This catches the catastrophic dimension/metric/vector-name drift straight from the backend, in
    addition to the sidecar-identity checks above.
    """
    expected = {
        "vector_name": profile.vector_name,
        "dimensions": profile.embedding_dimensions,
        "distance": profile.distance_metric,
    }
    actual = {
        "vector_name": config.vector_name,
        "dimensions": config.dimensions,
        "distance": config.distance,
    }
    return {
        field: {"expected": value, "actual": actual[field]}
        for field, value in expected.items()
        if actual[field] != value
    }


def profile_path(collection_name: str, *, directory: Path) -> Path:
    return directory / f"{collection_name}.json"


def write_semantic_profile(profile: SemanticIndexProfile, *, directory: Path) -> Path:
    """Atomically write the immutable profile sidecar; a reader never sees a partial file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = profile_path(profile.collection_name, directory=directory)
    payload = json.dumps(profile.model_dump(), ensure_ascii=False, indent=2, sort_keys=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".profile-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_semantic_profile(collection_name: str, *, directory: Path) -> SemanticIndexProfile:
    path = profile_path(collection_name, directory=directory)
    if not path.exists():
        raise QdrantError(
            f"semantic profile not found for {collection_name}",
            reason="semantic_profile_missing",
            detail={"path": str(path)},
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SemanticIndexProfile.model_validate(data)
    except (ValueError, ValidationError) as exc:
        raise QdrantError(
            f"semantic profile is invalid for {collection_name}",
            reason="semantic_profile_invalid",
            detail={"path": str(path)},
        ) from exc
