from typing import Any

from src.api.schemas.search import (
    SearchExecuteRequest,
    SearchExecuteResponse,
    SearchReadinessCheck,
    SearchReadinessResponse,
)
from src.config import Settings, get_settings
from src.search.contracts import QueryContext, Retriever, SearchFilters
from src.search.embeddings.contracts import EmbeddingError, EmbeddingProvider
from src.search.embeddings.factory import build_embedding_provider
from src.search.embeddings.query import embed_query_for_search
from src.search.fusion import fusion_profile
from src.search.opensearch_client import (
    OpenSearchAdapter,
    OpenSearchError,
    OpenSearchHTTPAdapter,
    build_opensearch_adapter,
    get_alias_targets,
    read_index_profile,
    search,
)
from src.search.pipeline import RetrievalPipeline
from src.search.response_builder import build_execute_response
from src.search.retrievers.lexical_opensearch import (
    LexicalRetriever,
    aggregate_surah_distribution,
    build_search_body,
    compatibility_mismatches,
)
from src.search.retrievers.semantic_qdrant import SemanticQdrantRetriever
from src.search.vector.profile import (
    collection_config_mismatches,
    expected_runtime_compatibility,
    hybrid_corpus_mismatches,
    profile_compatibility_mismatches,
    read_semantic_profile,
)
from src.search.vector.qdrant_store import QdrantError, QdrantStore, build_qdrant_store

# A token guaranteed to appear across the Quran corpus, used to prove the served query path returns
# hits (not just that the cluster is up) in the readiness probe.
READINESS_SMOKE_QUERY = "الله"

#: Retrieval policies (config ``QGRAPH_AI_SEARCH_RETRIEVAL_POLICY``). Both run real backends.
LEXICAL_POLICY = "lexical_v1"
HYBRID_POLICY = "hybrid_v1"


class SearchRetrievalError(Exception):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.status_code = status_code
        self.detail = detail or {}


def build_search_execute_response(
    request: SearchExecuteRequest,
    settings: Settings | None = None,
    adapter: OpenSearchAdapter | None = None,
    store: QdrantStore | None = None,
    provider: EmbeddingProvider | None = None,
) -> SearchExecuteResponse:
    """Run the configured retrieval policy against the serving alias(es) and render the response.

    ``lexical_v1`` runs OpenSearch only; ``hybrid_v1`` also runs Qdrant and fuses by weighted RRF.
    Both are real backends — a missing/misconfigured backend surfaces as a ``SearchRetrievalError``
    (never fake results, never a silent fall back to lexical-only). Tests inject the fake backends.
    """
    cfg = settings if settings is not None else get_settings()
    filters = SearchFilters.from_request_filters(request.filters)
    top_k = _resolve_top_k(request.output_preferences)
    policy = cfg.search_retrieval_policy
    alias = cfg.opensearch_alias
    # Surah-name documents are indexed but intentionally not served by this execute path in V1: the
    # default scope is Arabic ayat + translations. Arabic verses and translations are retrieved in
    # separate scoped queries so each result block is independently populated and ranked, and the
    # distribution chart reflects the verses alone. Single-content-type scopes don't need collapse.
    ayah_context = QueryContext(
        raw_query=request.query, filters=filters.quran_ayah_scope(), top_k=top_k, collapse=False
    )
    translation_context = (
        QueryContext(
            raw_query=request.query,
            filters=filters.translation_scope(),
            top_k=top_k,
            collapse=False,
        )
        if filters.include_translations
        else None
    )
    try:
        active_adapter = adapter if adapter is not None else build_search_adapter(cfg)
        retrievers: list[Retriever] = [LexicalRetriever(active_adapter, alias)]
        semantic_provenance: dict[str, Any] = {}
        if policy == HYBRID_POLICY:
            active_store = store if store is not None else _build_qdrant_store(cfg)
            retrievers.append(
                SemanticQdrantRetriever(
                    active_store, cfg.qdrant_collection_alias, cfg.qdrant_vector_name
                )
            )
            active_provider = provider if provider is not None else build_embedding_provider(cfg)
            # One query embedding, reused across the ayah and translation scopes.
            embedding = embed_query_for_search(active_provider, ayah_context)
            ayah_context = ayah_context.model_copy(update={"query_embedding": embedding})
            if translation_context is not None:
                translation_context = translation_context.model_copy(
                    update={"query_embedding": embedding}
                )
            semantic_provenance = _semantic_provenance(
                active_provider, active_store.resolve_alias(cfg.qdrant_collection_alias)
            )
        pipeline = RetrievalPipeline(retrievers)
        ayah_candidates = pipeline.run(ayah_context)
        translation_candidates = (
            pipeline.run(translation_context) if translation_context is not None else []
        )
        distribution = aggregate_surah_distribution(active_adapter, alias, ayah_context)
        profile = read_index_profile(active_adapter, alias)
    except OpenSearchError as exc:
        raise SearchRetrievalError(
            exc.message, reason=exc.reason, status_code=exc.status_code, detail=exc.detail
        ) from exc
    except (EmbeddingError, QdrantError) as exc:
        # No silent fallback: a hybrid query whose semantic side fails is a service error.
        raise SearchRetrievalError(
            exc.message,
            reason=exc.reason,
            status_code=getattr(exc, "status_code", None),
            detail=exc.detail,
        ) from exc

    provenance = {**_profile_provenance(profile), **semantic_provenance, "retrieval_policy": policy}
    return build_execute_response(
        request,
        ayah_candidates=ayah_candidates,
        translation_candidates=translation_candidates,
        surah_distribution=distribution,
        provenance=provenance,
        render_schema_version=cfg.render_schema_version,
        confidence_scale_k=cfg.search_confidence_scale_k,
        retrieval_policy=policy,
    )


def _build_qdrant_store(settings: Settings) -> QdrantStore:
    return build_qdrant_store(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )


def _semantic_provenance(provider: EmbeddingProvider, collection: str) -> dict[str, Any]:
    """Runtime fusion/semantic provenance — distinct from the immutable collection profile."""
    return {
        "semantic_collection": collection,
        "embedding_provider": provider.profile.provider,
        "embedding_model": provider.profile.model,
        "embedding_dimensions": provider.profile.dimensions,
        "fusion_profile": fusion_profile(),
    }


def build_search_adapter(settings: Settings) -> OpenSearchHTTPAdapter:
    """Build the OpenSearch adapter for the configured cluster (one app-scoped client per process)."""
    return build_opensearch_adapter(
        url=settings.opensearch_url,
        timeout_seconds=settings.opensearch_timeout_seconds,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        verify=settings.opensearch_ca_cert_path or settings.opensearch_verify_certs,
    )


def check_search_readiness(
    settings: Settings | None = None,
    adapter: OpenSearchAdapter | None = None,
    store: QdrantStore | None = None,
    provider: EmbeddingProvider | None = None,
) -> SearchReadinessResponse:
    """Probe whether the configured retrieval policy can actually serve queries.

    For ``lexical_v1``: the serving alias resolves to one index, its build profile is compatible with
    the running code, and a known smoke query returns a hit. For ``hybrid_v1``: additionally the
    semantic alias resolves to one non-empty collection whose profile matches the runtime provider and
    its own live Qdrant config, and the lexical/semantic corpora agree — so a deploy or monitor cannot
    report search healthy while ``/v1/search/execute`` would return a service error. No paid embedding
    call is made on the probe.
    """
    cfg = settings if settings is not None else get_settings()
    alias = cfg.opensearch_alias
    checks: list[SearchReadinessCheck] = []
    active_collection: str | None = None

    if not cfg.opensearch_url:
        checks.append(
            SearchReadinessCheck(
                name="opensearch_configured",
                ok=False,
                detail={"reason": "opensearch_not_configured"},
            )
        )
        return SearchReadinessResponse(ready=False, alias=alias, active_index=None, checks=checks)

    active_adapter = adapter if adapter is not None else build_search_adapter(cfg)
    try:
        targets = get_alias_targets(active_adapter, alias)
        if len(targets) != 1:
            checks.append(
                SearchReadinessCheck(
                    name="alias_single_target",
                    ok=False,
                    detail={"alias": alias, "active_indices": targets},
                )
            )
            return SearchReadinessResponse(
                ready=False, alias=alias, active_index=None, checks=checks
            )
        active_index = targets[0]
        checks.append(
            SearchReadinessCheck(
                name="alias_single_target", ok=True, detail={"index": active_index}
            )
        )

        lexical_profile = read_index_profile(active_adapter, alias)
        mismatches = compatibility_mismatches(lexical_profile)
        checks.append(
            SearchReadinessCheck(
                name="index_profile_compatible",
                ok=not mismatches,
                detail={"mismatches": mismatches} if mismatches else {},
            )
        )
        if mismatches:
            return SearchReadinessResponse(
                ready=False, alias=alias, active_index=active_index, checks=checks
            )

        smoke = search(
            active_adapter,
            alias,
            build_search_body(QueryContext(raw_query=READINESS_SMOKE_QUERY, top_k=1)),
        )
        hit_count = len(smoke.get("hits", {}).get("hits", []))
        checks.append(
            SearchReadinessCheck(
                name="smoke_query",
                ok=hit_count > 0,
                detail={"query": READINESS_SMOKE_QUERY, "hit_count": hit_count},
            )
        )

        if cfg.search_retrieval_policy == HYBRID_POLICY:
            semantic_checks, active_collection = _hybrid_readiness_checks(
                cfg, store, provider, lexical_profile
            )
            checks.extend(semantic_checks)
    except OpenSearchError as exc:
        checks.append(
            SearchReadinessCheck(
                name="opensearch_reachable",
                ok=False,
                detail={"reason": exc.reason, "message": exc.message},
            )
        )
        return SearchReadinessResponse(ready=False, alias=alias, active_index=None, checks=checks)

    return SearchReadinessResponse(
        ready=all(check.ok for check in checks),
        alias=alias,
        active_index=active_index,
        active_collection=active_collection,
        checks=checks,
    )


def _hybrid_readiness_checks(
    settings: Settings,
    store: QdrantStore | None,
    provider: EmbeddingProvider | None,
    lexical_profile: dict[str, Any],
) -> tuple[list[SearchReadinessCheck], str | None]:
    """Semantic-side readiness for ``hybrid_v1`` — every failure mode is its own loud check."""
    checks: list[SearchReadinessCheck] = []
    if provider is None:
        checks.append(
            SearchReadinessCheck(
                name="embedding_provider_configured",
                ok=False,
                detail={"reason": "embedding_provider_not_configured"},
            )
        )
        return checks, None
    if store is None:
        checks.append(
            SearchReadinessCheck(
                name="qdrant_configured", ok=False, detail={"reason": "qdrant_not_configured"}
            )
        )
        return checks, None

    try:
        collection = store.resolve_alias(settings.qdrant_collection_alias)
        checks.append(
            SearchReadinessCheck(
                name="semantic_alias_single_target", ok=True, detail={"collection": collection}
            )
        )
        profile = read_semantic_profile(collection, directory=settings.semantic_index_profiles_dir)
        point_count = store.count_points(collection)
        checks.append(
            SearchReadinessCheck(
                name="semantic_collection_non_empty",
                ok=point_count > 0,
                detail={"point_count": point_count},
            )
        )
        code_mismatches = profile_compatibility_mismatches(
            profile,
            expected=expected_runtime_compatibility(
                embedding_provider=settings.embedding_provider,
                embedding_model=settings.embedding_model,
                embedding_dimensions=settings.embedding_dimensions,
            ),
        )
        checks.append(
            SearchReadinessCheck(
                name="semantic_profile_compatible",
                ok=not code_mismatches,
                detail={"mismatches": code_mismatches} if code_mismatches else {},
            )
        )
        config_mismatches = collection_config_mismatches(
            store.collection_config(collection), profile
        )
        checks.append(
            SearchReadinessCheck(
                name="semantic_collection_config_match",
                ok=not config_mismatches,
                detail={"mismatches": config_mismatches} if config_mismatches else {},
            )
        )
        corpus_mismatches = hybrid_corpus_mismatches(lexical_profile, profile)
        checks.append(
            SearchReadinessCheck(
                name="hybrid_corpus_compatible",
                ok=not corpus_mismatches,
                detail={"mismatches": corpus_mismatches} if corpus_mismatches else {},
            )
        )
        return checks, collection
    except QdrantError as exc:
        checks.append(
            SearchReadinessCheck(
                name="qdrant_reachable",
                ok=False,
                detail={"reason": exc.reason, "message": exc.message},
            )
        )
        return checks, None


def _profile_provenance(profile: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "corpus_snapshot_id",
        "corpus_snapshot_hash",
        "document_schema_version",
        "normalization_profile_id",
        "normalization_profile_version",
        "analysis_profile_version",
        "index_id",
    )
    return {key: profile.get(key) for key in keys}


def _resolve_top_k(output_preferences: dict[str, Any]) -> int:
    raw_top_k = output_preferences.get("top_k")
    if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
        return 10
    return max(1, min(raw_top_k, 25))
