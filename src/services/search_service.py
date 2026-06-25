from typing import Any

from src.api.schemas.search import SearchExecuteRequest, SearchExecuteResponse
from src.config import Settings, get_settings
from src.search.contracts import QueryContext, SearchFilters
from src.search.opensearch_client import (
    OpenSearchAdapter,
    OpenSearchError,
    build_opensearch_adapter,
    read_index_profile,
)
from src.search.pipeline import RetrievalPipeline
from src.search.response_builder import build_execute_response
from src.search.retrievers.lexical_opensearch import (
    LexicalRetriever,
    aggregate_surah_distribution,
)


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
    query_context = QueryContext(
        raw_query=request.query,
        filters=SearchFilters.from_request_filters(request.filters),
        top_k=_resolve_top_k(request.output_preferences),
        collapse=True,
    )
    alias = cfg.opensearch_alias
    try:
        active_adapter = adapter if adapter is not None else _build_adapter(cfg)
        candidates = RetrievalPipeline([LexicalRetriever(active_adapter, alias)]).run(query_context)
        distribution = aggregate_surah_distribution(active_adapter, alias, query_context)
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
        candidates,
        surah_distribution=distribution,
        provenance=_profile_provenance(profile),
        render_schema_version=cfg.render_schema_version,
        confidence_scale_k=cfg.search_confidence_scale_k,
    )


def _build_adapter(settings: Settings) -> OpenSearchAdapter:
    return build_opensearch_adapter(
        url=settings.opensearch_url,
        timeout_seconds=settings.opensearch_timeout_seconds,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        verify=settings.opensearch_ca_cert_path or settings.opensearch_verify_certs,
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
