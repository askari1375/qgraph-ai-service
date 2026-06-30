"""Project↔Qdrant mapping: deterministic point IDs, payloads, and filter compilation.

Keeps the Qdrant filter DSL out of ``contracts.py``: ``SearchFilters`` stays SDK-free and the compiler
lives here as a free function, parallel to ``SearchFilters.to_opensearch_filter``.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models

from src.search.contracts import SearchFilters
from src.search.indexing.documents import SearchIndexDocument
from src.search.vector.corpus_policy import is_canonical_translation

#: Fixed namespace so point IDs are stable across rebuilds. Qdrant point IDs must be int or UUID, so
#: the string ``document.id`` (e.g. ``"ayah:2:255:ar"``) is hashed into a deterministic UUIDv5.
QDRANT_POINT_NAMESPACE = uuid.UUID("d3f4a1b2-9c8e-5f7a-b6d2-0123456789ab")

#: Payload fields that back ``SearchFilters`` (plus the scope flag), with the Qdrant payload-index kind.
PAYLOAD_INDEX_FIELDS: dict[str, str] = {
    "content_type": "keyword",
    "language_code": "keyword",
    "source_id": "keyword",
    "surah_number": "integer",
    "ayah_global_number": "integer",
    "is_canonical_translation": "bool",
}


def build_point_id(document_id: str) -> str:
    """Deterministic Qdrant point id derived from the project ``document_id``."""
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, document_id))


def build_point_payload(document: SearchIndexDocument) -> dict[str, Any]:
    """Flat payload carrying enough to build a ``RetrievalCandidate`` without a second backend call."""
    metadata = document.metadata
    return {
        "document_id": document.id,
        "canonical_content_id": document.canonical_content_id,
        "content_type": metadata.content_type.value,
        "text": document.content,
        "surah_number": metadata.surah_number,
        "ayah_number": metadata.ayah_number,
        "ayah_global_number": metadata.ayah_global_number,
        "language_code": metadata.language_code,
        "source_id": metadata.source_id,
        "source_name": metadata.source_name,
        # Scope flag: the curated canonical translation per language (Arabic ayahs are False). Recorded
        # so a future collection holding more sources can default-filter to the canonical scope.
        "is_canonical_translation": is_canonical_translation(metadata.source_id),
    }


def compile_qdrant_filter(filters: SearchFilters) -> models.Filter | None:
    """Compile ``SearchFilters`` to a Qdrant ``Filter``; mirrors ``to_opensearch_filter``.

    Returns ``None`` when no condition applies, so the caller queries unfiltered.
    """
    must: list[models.FieldCondition] = []
    if filters.content_types:
        must.append(
            models.FieldCondition(
                key="content_type",
                match=models.MatchAny(any=[ct.value for ct in filters.content_types]),
            )
        )
    if filters.languages:
        must.append(
            models.FieldCondition(
                key="language_code", match=models.MatchAny(any=list(filters.languages))
            )
        )
    if filters.source_ids:
        must.append(
            models.FieldCondition(
                key="source_id", match=models.MatchAny(any=list(filters.source_ids))
            )
        )
    if filters.surah_numbers:
        must.append(
            models.FieldCondition(
                key="surah_number", match=models.MatchAny(any=list(filters.surah_numbers))
            )
        )
    if filters.ayah_global_min is not None or filters.ayah_global_max is not None:
        must.append(
            models.FieldCondition(
                key="ayah_global_number",
                range=models.Range(gte=filters.ayah_global_min, lte=filters.ayah_global_max),
            )
        )
    if not must:
        return None
    return models.Filter(must=must)
