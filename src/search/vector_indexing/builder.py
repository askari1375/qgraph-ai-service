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
from math import ceil
from typing import Any

from src.config import Settings, get_settings
from src.search.embeddings.contracts import (
    EmbeddingError,
    EmbeddingProvider,
    validate_embedding_vectors,
)
from src.search.embeddings.factory import build_embedding_provider
from src.search.embeddings.input import prepare_embedding_input
from src.search.embeddings.tokenization import TokenCounter
from src.search.eval.runner import evaluate_semantic_collection
from src.search.indexing.documents import SearchIndexDocument, build_search_documents
from src.search.vector.corpus_policy import select_semantic_documents
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
    expected_runtime_compatibility,
    read_semantic_profile,
    semantic_artifact_mismatches,
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
    # The lexical index keeps every translation; the semantic collection embeds only the curated
    # corpus policy (Arabic + the canonical English/Persian translations). Carve that subset out before
    # any paid embedding, failing loudly if it is incomplete.
    documents = select_semantic_documents(documents)
    # Deterministic document-id order so points and the profile checksum are reproducible.
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
    # Provider facts come from the resolved provider (real build) or the configured settings (a
    # ``--dry-run`` constructs none). Both drive the preflight token/cost estimate below.
    provider_name = provider.profile.provider if provider else settings.embedding_provider
    model = provider.profile.model if provider else settings.embedding_model
    dimensions = provider.profile.dimensions if provider else settings.embedding_dimensions
    if provider is not None or model:
        report["embedding_provider"] = provider_name
        report["embedding_model"] = model
        report["embedding_dimensions"] = dimensions

    # Measure the prepared input before any paid call: per-source counts, token total/cost, request
    # count, vector-store footprint, and a hard per-input token ceiling that fails with the offending
    # document id rather than letting the provider reject or truncate mid-build.
    report["preflight"] = build_embedding_preflight(
        documents,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size or settings.embedding_document_batch_size,
        max_input_tokens=settings.embedding_max_input_tokens,
        usd_per_million_tokens=settings.embedding_usd_per_million_input_tokens,
    )
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
        # Quality gate: run the semantic eval against the physical collection before activation. A
        # CONFIRMED case missing a must-include id blocks activation; PENDING cases are reported only.
        evaluation = evaluate_semantic_collection(
            collection, store=store, provider=provider, vector_name=profile.vector_name
        )
        report["evaluation"] = evaluation
        report["ok"] = evaluation["ok"]
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
    # The same active-artifact validator readiness/execute use, against the configured runtime provider
    # — so ``status`` cannot report compatible while a different model/dimension would actually serve.
    # The lexical↔semantic corpus gate stays in readiness/execute, where the OpenSearch profile is read.
    mismatches = semantic_artifact_mismatches(
        profile=profile,
        collection_config=store.collection_config(collection),
        runtime_expected=expected_runtime_compatibility(
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        ),
    )
    status["active_collection"] = collection
    status["profile"] = profile.model_dump()
    status["point_count"] = store.count_points(collection)
    status["compatible"] = not any(mismatches.values())
    failures = {category: detail for category, detail in mismatches.items() if detail}
    if failures:
        status["mismatches"] = failures
    return status


def evaluate_collection(
    collection: str,
    *,
    settings: Settings | None = None,
    store: QdrantStore | None = None,
    provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Run the semantic eval against an existing physical collection on demand.

    Paid: embeds one query per eval case. Lets the owner re-run the cross-lingual top-K report after a
    build/activate (or after promoting PENDING→CONFIRMED) without rebuilding the collection.
    """
    settings = settings or get_settings()
    store = store or _store(settings)
    provider = provider or build_embedding_provider(settings)
    if not store.collection_exists(collection):
        raise QdrantError(
            f"collection {collection} does not exist",
            reason="semantic_collection_missing",
            detail={"collection": collection},
        )
    return evaluate_semantic_collection(
        collection, store=store, provider=provider, vector_name=settings.qdrant_vector_name
    )


def build_embedding_preflight(
    documents: list[SearchIndexDocument],
    *,
    model: str,
    dimensions: int,
    batch_size: int,
    max_input_tokens: int,
    usd_per_million_tokens: float,
) -> dict[str, Any]:
    """Measure prepared embedding input before paying; enforce the per-input token ceiling.

    Tokenizes each document exactly as it will be embedded and raises ``embedding_input_too_long`` with
    the offending document id if any input exceeds ``max_input_tokens`` — so an over-limit document
    fails the whole build up front instead of after partial paid embedding. Returns the planning
    summary (per-source counts, token total, request/footprint estimates) the dry-run reports.
    """
    counter = TokenCounter(model=model)
    source_counts: dict[str, int] = {}
    total_tokens = 0
    max_tokens = 0
    max_document_id: str | None = None
    for doc in documents:
        text = prepare_embedding_input(doc.content, doc.metadata.language_code)
        tokens = counter.count(text)
        if tokens > max_input_tokens:
            raise EmbeddingError(
                f"document {doc.id} has {tokens} tokens, exceeds max {max_input_tokens}",
                reason="embedding_input_too_long",
                detail={
                    "document_id": doc.id,
                    "tokens": tokens,
                    "max_input_tokens": max_input_tokens,
                },
            )
        total_tokens += tokens
        if tokens > max_tokens:
            max_tokens, max_document_id = tokens, doc.id
        source = doc.metadata.source_id
        source_counts[source] = source_counts.get(source, 0) + 1

    report: dict[str, Any] = {
        "document_count": len(documents),
        "source_counts": source_counts,
        "token_estimate_method": counter.method,
        "total_input_tokens": total_tokens,
        "max_document_tokens": max_tokens,
        "max_document_id": max_document_id,
        "max_input_tokens": max_input_tokens,
        "request_count": ceil(len(documents) / batch_size) if batch_size > 0 else 0,
        "estimated_vector_bytes": len(documents) * dimensions * 4,
    }
    if usd_per_million_tokens > 0:
        report["estimated_usd"] = round(total_tokens / 1_000_000 * usd_per_million_tokens, 6)
    return report


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
