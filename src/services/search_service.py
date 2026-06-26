from typing import Any

from src.api.schemas.search import (
    SearchExecuteRequest,
    SearchExecuteResponse,
    SearchReadinessCheck,
    SearchReadinessResponse,
)
from src.config import Settings, get_settings
from src.search.contracts import QueryContext, SearchFilters
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

# A token guaranteed to appear across the Quran corpus, used to prove the served query path returns
# hits (not just that the cluster is up) in the readiness probe.
READINESS_SMOKE_QUERY = "الله"


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
) -> SearchExecuteResponse:
    """Run lexical retrieval against the serving alias and render the response.

    OpenSearch is the only backend: a missing/misconfigured cluster surfaces as a
    ``SearchRetrievalError`` (never as fake results). Tests inject a fake ``adapter``.
    """
    cfg = settings if settings is not None else get_settings()
    filters = SearchFilters.from_request_filters(request.filters)
    top_k = _resolve_top_k(request.output_preferences)
    # Surah-name documents are indexed but intentionally not served by this execute path in V1: the
    # default scope is Arabic ayat + translations. Surah-name search is index-only for now.
    # Arabic verses and translations are retrieved in separate scoped queries so each result block is
    # independently populated and ranked, and the distribution chart reflects the verses alone (stable
    # regardless of the translation control). Single-content-type scopes don't need collapse.
    ayah_context = QueryContext(
        raw_query=request.query, filters=filters.quran_ayah_scope(), top_k=top_k, collapse=False
    )
    alias = cfg.opensearch_alias
    try:
        active_adapter = adapter if adapter is not None else build_search_adapter(cfg)
        pipeline = RetrievalPipeline([LexicalRetriever(active_adapter, alias)])
        ayah_candidates = pipeline.run(ayah_context)
        translation_candidates = (
            pipeline.run(
                QueryContext(
                    raw_query=request.query,
                    filters=filters.translation_scope(),
                    top_k=top_k,
                    collapse=False,
                )
            )
            if filters.include_translations
            else []
        )
        distribution = aggregate_surah_distribution(active_adapter, alias, ayah_context)
        profile = read_index_profile(active_adapter, alias)
    except OpenSearchError as exc:
        raise SearchRetrievalError(
            exc.message,
            reason=exc.reason,
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc

    return build_execute_response(
        request,
        ayah_candidates=ayah_candidates,
        translation_candidates=translation_candidates,
        surah_distribution=distribution,
        provenance=_profile_provenance(profile),
        render_schema_version=cfg.render_schema_version,
        confidence_scale_k=cfg.search_confidence_scale_k,
    )


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
) -> SearchReadinessResponse:
    """Probe whether the search path can actually serve queries.

    Confirms the serving alias resolves to exactly one index, the index build profile is compatible
    with the running code, and a known smoke query returns at least one hit — so a deploy or monitor
    cannot report search healthy while ``/v1/search/execute`` would return a service error.
    """
    cfg = settings if settings is not None else get_settings()
    alias = cfg.opensearch_alias
    checks: list[SearchReadinessCheck] = []

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

        mismatches = compatibility_mismatches(read_index_profile(active_adapter, alias))
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
        checks=checks,
    )


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
