"""Orchestrate a semantic collection build and its alias-swap activation.

The semantic mirror of ``src/search/indexing/builder.py``: a corpus snapshot becomes a *new physical
Qdrant collection*, validated, and activated by repointing the serving alias.

Build chain:
    pull snapshot (corpus_client) -> build_search_documents -> build_semantic_profile
      -> create "<prefix>-<date>-<NNN>" + payload indexes -> embed + upsert in batches
      -> validate (count + live config) -> write the immutable sidecar profile
      -> (optionally) swap the alias to the new collection

``build`` never auto-activates unless asked and validation passes, so a build that fails validation
never becomes the served collection. A real build resolves the production embedding provider through
``build_embedding_provider``; tests inject a deterministic one, which takes precedence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from typing import Any

from src.config import Settings, get_settings
from src.search.embeddings.contracts import EmbeddingProvider, validate_embedding_vectors
from src.search.embeddings.factory import build_embedding_provider
from src.search.embeddings.input import prepare_embedding_input
from src.search.indexing.documents import SearchIndexDocument, build_search_documents
from src.search.vector.mapping import (
    PAYLOAD_INDEX_FIELDS,
    build_point_id,
    build_point_payload,
)
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
from src.search.vector.qdrant_store import (
    QdrantError,
    QdrantStore,
    VectorPoint,
    build_qdrant_store,
)
from src.services.corpus_client import build_django_corpus_client


def build_semantic_collection(
    *,
    settings: Settings | None = None,
    store: QdrantStore | None = None,
    provider: EmbeddingProvider | None = None,
    activate: bool = False,
    dry_run: bool = False,
    languages: Sequence[str] | None = None,
    surahs: Sequence[int] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Build a new physical semantic collection from the current corpus snapshot."""
    settings = settings or get_settings()
    snapshot = build_django_corpus_client(settings).fetch_quran_snapshot(
        translation_languages=languages, surah_numbers=surahs
    )
    documents = build_search_documents(snapshot)
    if not documents:
        raise QdrantError("Corpus snapshot produced no documents", reason="empty_corpus")
    # Deterministic document-id order so points and any future checksum are reproducible.
    documents.sort(key=lambda doc: doc.id)

    store = store or _store(settings)
    collection = _next_collection_name(store, settings.qdrant_collection_prefix)
    # A real build needs the production provider; an injected one (tests) wins. ``--dry-run`` plans
    # without one. Resolution raises ``embedding_provider_not_configured`` until a provider is wired.
    if provider is None and not dry_run:
        provider = build_embedding_provider(settings)

    report: dict[str, Any] = {
        "collection": collection,
        "alias": settings.qdrant_collection_alias,
        "document_count": len(documents),
        "corpus_snapshot_id": snapshot.corpus_snapshot_id,
        "language_counts": _counts(documents, lambda doc: doc.metadata.language_code),
        "content_type_counts": _counts(documents, lambda doc: doc.metadata.content_type.value),
    }
    if provider is not None:
        report["embedding_provider"] = provider.profile.provider
        report["embedding_model"] = provider.profile.model
        report["embedding_dimensions"] = provider.profile.dimensions
    if dry_run:
        report["dry_run"] = True
        return report

    profile = build_semantic_profile(
        collection_name=collection,
        snapshot=snapshot,
        documents=documents,
        provider_profile=provider.profile,
        vector_name=settings.qdrant_vector_name,
    )
    store.create_collection(
        collection,
        vector_name=profile.vector_name,
        dimensions=profile.embedding_dimensions,
        distance=profile.distance_metric,
    )
    store.create_payload_indexes(collection, PAYLOAD_INDEX_FIELDS)
    _embed_and_upsert(
        store,
        collection,
        documents,
        provider,
        profile,
        batch_size or settings.embedding_document_batch_size,
    )

    validation = _validate_collection(store, collection, profile, len(documents))
    report["validation"] = validation
    report["ok"] = not validation["hard_failures"]
    report["activated"] = False
    if report["ok"]:
        write_semantic_profile(profile, directory=settings.semantic_index_profiles_dir)
        if activate:
            store.swap_alias(settings.qdrant_collection_alias, collection)
            report["activated"] = True
    return report


def activate_semantic_collection(
    collection: str,
    *,
    settings: Settings | None = None,
    store: QdrantStore | None = None,
    delete_old: bool = False,
) -> dict[str, Any]:
    """Point the serving alias at an already-built collection (atomic repoint = activation/rollback)."""
    settings = settings or get_settings()
    store = store or _store(settings)
    alias = settings.qdrant_collection_alias
    profile = read_semantic_profile(collection, directory=settings.semantic_index_profiles_dir)
    mismatches = collection_config_mismatches(store.collection_config(collection), profile)
    if mismatches:
        raise QdrantError(
            f"collection {collection} config does not match its profile",
            reason="semantic_collection_config_mismatch",
            detail={"mismatches": mismatches},
        )
    previous = _previous_targets(store, alias, collection)
    store.swap_alias(alias, collection)
    if delete_old:
        for old in previous:
            store.delete_collection(old)
            delete_semantic_profile(old, directory=settings.semantic_index_profiles_dir)
    return {
        "alias": alias,
        "active_collection": collection,
        "previous_collections": previous,
        "deleted_old": delete_old,
    }


def semantic_status(
    *,
    settings: Settings | None = None,
    store: QdrantStore | None = None,
) -> dict[str, Any]:
    """Report the alias target, its sidecar profile, and the compatibility result."""
    settings = settings or get_settings()
    store = store or _store(settings)
    alias = settings.qdrant_collection_alias
    status: dict[str, Any] = {"alias": alias}
    try:
        collection = store.resolve_alias(alias)
    except QdrantError as exc:
        if exc.reason == "semantic_alias_invalid":
            status["active_collection"] = None
            status["compatible"] = None
            return status
        raise
    profile = read_semantic_profile(collection, directory=settings.semantic_index_profiles_dir)
    code_mismatches = profile_compatibility_mismatches(
        profile, expected=expected_code_compatibility()
    )
    config_mismatches = collection_config_mismatches(store.collection_config(collection), profile)
    status["active_collection"] = collection
    status["profile"] = profile.model_dump()
    status["point_count"] = store.count_points(collection)
    status["compatible"] = not code_mismatches and not config_mismatches
    if code_mismatches:
        status["mismatches"] = code_mismatches
    if config_mismatches:
        status["config_mismatches"] = config_mismatches
    return status


def _store(settings: Settings) -> QdrantStore:
    return build_qdrant_store(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )


def _next_collection_name(store: QdrantStore, prefix: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = f"{prefix}-{date}"
    suffixes = []
    for name in store.list_collections():
        if not name.startswith(f"{base}-"):
            continue
        tail = name[len(base) + 1 :]
        if tail.isdigit():
            suffixes.append(int(tail))
    sequence = (max(suffixes) + 1) if suffixes else 1
    return f"{base}-{sequence:03d}"


def _embed_and_upsert(
    store: QdrantStore,
    collection: str,
    documents: list[SearchIndexDocument],
    provider: EmbeddingProvider,
    profile: SemanticIndexProfile,
    batch_size: int,
) -> None:
    for batch in _batches(documents, batch_size):
        texts = [prepare_embedding_input(doc.content, doc.metadata.language_code) for doc in batch]
        vectors = provider.embed_documents(texts)
        validate_embedding_vectors(
            vectors, expected_count=len(batch), dimensions=profile.embedding_dimensions
        )
        points = [
            VectorPoint(
                point_id=build_point_id(doc.id),
                vector=vector,
                payload=build_point_payload(doc),
            )
            for doc, vector in zip(batch, vectors)
        ]
        store.upsert_points(collection, vector_name=profile.vector_name, points=points)


def _validate_collection(
    store: QdrantStore,
    collection: str,
    profile: SemanticIndexProfile,
    expected_count: int,
) -> dict[str, Any]:
    """Structural validation before activation: exact point count and live-config match.

    Semantic *quality* evaluation (a reviewed multilingual eval set) is a later phase; here a hard
    failure means the collection is structurally wrong (missing/extra points, or a backend
    dimension/metric/vector-name that disagrees with its own profile) and must not be activated.
    """
    hard_failures: list[str] = []
    actual_count = store.count_points(collection)
    if actual_count != expected_count:
        hard_failures.append("count_mismatch")
    config_mismatches = collection_config_mismatches(store.collection_config(collection), profile)
    if config_mismatches:
        hard_failures.append("collection_config_mismatch")
    return {
        "expected_count": expected_count,
        "actual_count": actual_count,
        "config_mismatches": config_mismatches,
        "hard_failures": hard_failures,
    }


def _previous_targets(store: QdrantStore, alias: str, collection: str) -> list[str]:
    try:
        current = store.resolve_alias(alias)
    except QdrantError as exc:
        if exc.reason == "semantic_alias_invalid":
            return []
        raise
    return [current] if current != collection else []


def _batches(items: list[SearchIndexDocument], size: int) -> Iterator[list[SearchIndexDocument]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _counts(
    documents: list[SearchIndexDocument], key: Callable[[SearchIndexDocument], str]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        bucket = key(document)
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts
