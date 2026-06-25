from hashlib import sha256
from math import exp
from typing import Any

from src.api.schemas.search import (
    SearchExecuteRequest,
    SearchExecuteResponse,
    SearchResponseBlock,
    SearchResultItem,
)
from src.config import Settings, get_settings
from src.search.contracts import ContentType, QueryContext, RetrievalCandidate, SearchFilters
from src.search.opensearch_client import (
    OpenSearchAdapter,
    OpenSearchError,
    build_opensearch_adapter,
    read_index_profile,
)
from src.search.pipeline import RetrievalPipeline
from src.search.retrievers.lexical_opensearch import LexicalRetriever

DEFAULT_SURAH_DISTRIBUTION = (1, 2, 7)
MOCK_SURAH_VALUES = {
    1: 3,
    2: 17,
    7: 5,
}
RETRIEVAL_BACKEND_MODE = "opensearch"
MOCK_BACKEND_MODE = "mock"
OPEN_SEARCH_BACKEND_NAME = "open_search"


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
    cfg = settings if settings is not None else get_settings()
    mode = cfg.search_lexical_backend_mode.casefold()
    if mode == RETRIEVAL_BACKEND_MODE:
        return _build_retrieval_response(request, settings=cfg, adapter=adapter)
    if mode != MOCK_BACKEND_MODE:
        raise SearchRetrievalError(
            "Unsupported search lexical backend mode",
            reason="unsupported_backend_mode",
            detail={"mode": cfg.search_lexical_backend_mode},
        )

    return _build_mock_search_execute_response(request, settings=cfg)


# --------------------------------------------------------------------------------------------------
# OpenSearch retrieval path
# --------------------------------------------------------------------------------------------------


def _build_retrieval_response(
    request: SearchExecuteRequest,
    *,
    settings: Settings,
    adapter: OpenSearchAdapter | None,
) -> SearchExecuteResponse:
    top_k = _resolve_top_k(request.output_preferences)
    query_context = QueryContext(
        raw_query=request.query,
        filters=SearchFilters.from_request_filters(request.filters),
        top_k=top_k,
        collapse=True,
    )
    alias = settings.opensearch_alias
    try:
        active_adapter = adapter if adapter is not None else _build_adapter(settings)
        candidates = RetrievalPipeline([LexicalRetriever(active_adapter, alias)]).run(query_context)
        profile = read_index_profile(active_adapter, alias)
    except OpenSearchError as exc:
        raise SearchRetrievalError(
            exc.message,
            reason=exc.reason,
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc

    items = _build_retrieval_items(candidates, profile)
    confidence = _confidence_from_candidates(candidates, scale_k=settings.search_confidence_scale_k)
    provenance = _profile_provenance(profile)
    return SearchExecuteResponse(
        title=f"Search results for {request.query}",
        overall_confidence=confidence,
        render_schema_version=settings.render_schema_version,
        metadata={"mock": False, "backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
        blocks=[
            SearchResponseBlock(
                order=0,
                block_type="results",
                title="Lexical matches",
                payload={
                    "query": request.query,
                    "result_count": len(items),
                    "top_k": top_k,
                },
                explanation="OpenSearch BM25 lexical retrieval over Quran corpus documents.",
                confidence=confidence,
                provenance={"backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
                warning_text="" if items else "No lexical matches were returned.",
                items=items,
            )
        ],
    )


def _build_adapter(settings: Settings) -> OpenSearchAdapter:
    return build_opensearch_adapter(
        url=settings.opensearch_url,
        timeout_seconds=settings.opensearch_timeout_seconds,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        verify=settings.opensearch_ca_cert_path or settings.opensearch_verify_certs,
    )


def _build_retrieval_items(
    candidates: list[RetrievalCandidate], profile: dict[str, Any]
) -> list[SearchResultItem]:
    max_score = max((candidate.score for candidate in candidates), default=0.0)
    snapshot_provenance = _profile_provenance(profile)
    items: list[SearchResultItem] = []
    for candidate in candidates:
        # Min-max normalize for relative bar heights; overall confidence carries absolute strength.
        score = 0.0 if max_score <= 0 else min(candidate.score / max_score, 1.0)
        metadata = candidate.metadata
        items.append(
            SearchResultItem(
                rank=candidate.rank,
                result_type=_result_type(candidate.content_type),
                score=score,
                title=_build_result_title(metadata),
                snippet_text=_snippet(candidate.text),
                highlighted_text=candidate.highlighted_text or _snippet(candidate.text),
                match_metadata={
                    "document_id": candidate.document_id,
                    "canonical_content_id": candidate.canonical_content_id,
                    "content_type": candidate.content_type.value,
                    "surah_number": metadata.get("surah_number"),
                    "ayah_number": metadata.get("ayah_number"),
                    "ayah_global_number": metadata.get("ayah_global_number"),
                    "language_code": metadata.get("language_code"),
                    "source_id": metadata.get("source_id"),
                    "source_name": metadata.get("source_name"),
                },
                explanation="Ranked by OpenSearch lexical score.",
                provenance={
                    "backend": OPEN_SEARCH_BACKEND_NAME,
                    "document_id": candidate.document_id,
                    "lexical_score": candidate.score,
                    **snapshot_provenance,
                },
            )
        )
    return items


def _result_type(content_type: ContentType) -> str:
    return "surah" if content_type is ContentType.SURAH_NAME else "ayah"


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


def _confidence_from_candidates(candidates: list[RetrievalCandidate], *, scale_k: float) -> float:
    """Map the strongest absolute lexical score to a bounded 0..1 confidence.

    Per-item ``score`` is min-max normalized for relative bar heights, so this derives overall
    confidence from the absolute top score instead.
    """
    top_absolute_score = max((candidate.score for candidate in candidates), default=0.0)
    if top_absolute_score <= 0.0 or scale_k <= 0.0:
        return 0.0
    return 1.0 - exp(-top_absolute_score / scale_k)


def _resolve_top_k(output_preferences: dict[str, Any]) -> int:
    raw_top_k = output_preferences.get("top_k")
    if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
        return 10
    return max(1, min(raw_top_k, 25))


def _build_result_title(metadata: dict[str, Any]) -> str:
    surah_number = metadata.get("surah_number")
    ayah_number = metadata.get("ayah_number")
    if surah_number and ayah_number:
        return f"Surah {surah_number}, Ayah {ayah_number}"
    if surah_number:
        return f"Surah {surah_number}"
    return "Quran corpus match"


def _snippet(text: str, max_length: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 1].rstrip()}..."


# --------------------------------------------------------------------------------------------------
# Mock path (kept until the runtime mock is removed)
# --------------------------------------------------------------------------------------------------


def _coerce_surah_ids(filters: dict) -> list[int]:
    raw_surahs = filters.get("surahs")
    if raw_surahs is None:
        raw_surahs = filters.get("surah_ids")
    if not isinstance(raw_surahs, list):
        return []

    surah_ids: list[int] = []
    seen: set[int] = set()
    for value in raw_surahs:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value < 1 or value > 114 or value in seen:
            continue
        surah_ids.append(value)
        seen.add(value)
    return surah_ids


def _mock_surah_value(surah_id: int) -> int:
    if surah_id in MOCK_SURAH_VALUES:
        return MOCK_SURAH_VALUES[surah_id]
    return ((surah_id * 7) % 19) + 1


def _build_surah_distribution_values(filters: dict) -> list[dict[str, int]]:
    surah_ids = _coerce_surah_ids(filters)
    if not surah_ids:
        surah_ids = list(DEFAULT_SURAH_DISTRIBUTION)
    return [
        {
            "surah": surah_id,
            "value": _mock_surah_value(surah_id),
        }
        for surah_id in surah_ids
    ]


def _stable_bucket(value: str) -> int:
    digest = sha256(value.encode("utf-8")).digest()
    return digest[0]


def _should_include_surah_distribution(request: SearchExecuteRequest) -> bool:
    if _coerce_surah_ids(request.filters):
        return True
    return _stable_bucket(request.query) % 2 == 0


def _build_markdown_content() -> str:
    return (
        "## Why this theme matters\n\n"
        "The roots **r-h-m** and **gh-f-r** are common in discussions of mercy and "
        "forgiveness.\n\n"
        "### Mock observations\n\n"
        "- Mercy language appears near openings, prayers, and prophetic narratives.\n"
        "- The mock counts are synthetic, but the shape is useful for frontend testing.\n"
        "- Prefer the chart block for numeric distribution.\n"
        "- [x] supported by the mock\n\n"
        "Use `r-h-m` as the mock root marker.\n\n"
        "| Surah | Mock mentions | Note |\n"
        "| --- | ---: | --- |\n"
        "| Al-Fatihah | 3 | Opening formula |\n"
        "| Al-Baqarah | 17 | Dense legal and narrative material |\n"
        "| Maryam | 14 | Strong mercy motif |\n\n"
        "> This block is synthetic and exists to test Markdown rendering.\n\n"
        "```json\n"
        "{\n"
        '  "root": "r-h-m",\n'
        '  "mock": true,\n'
        '  "renderer": "markdown"\n'
        "}\n"
        "```\n\n"
        "See also [Quran Corpus](https://corpus.quran.com)."
    )


def _build_markdown_block(order: int) -> SearchResponseBlock:
    return SearchResponseBlock(
        order=order,
        block_type="markdown",
        title="Structured explanation",
        payload={
            "headline": "Mock markdown overview",
            "content": _build_markdown_content(),
        },
        explanation="Synthetic markdown block for renderer testing.",
        confidence=0.8,
        provenance={"backend": "mock"},
        warning_text="",
        items=[],
    )


def _build_mock_search_execute_response(
    request: SearchExecuteRequest,
    settings: Settings,
) -> SearchExecuteResponse:
    cfg = settings

    blocks = [
        SearchResponseBlock(
            order=0,
            block_type="text",
            title="Mercy across the Qur'an",
            payload={
                "headline": "Mock thematic overview",
                "details": (
                    "This mock response summarizes the requested theme in plain text so "
                    "Django can persist typed v1 blocks and the frontend can render them."
                    "\n\n"
                    "The content is synthetic and intended only for end-to-end wiring tests."
                ),
            },
            explanation="Synthetic prose block for end-to-end wiring tests.",
            confidence=0.88,
            provenance={"backend": "mock"},
            warning_text="",
            items=[],
        ),
    ]
    blocks.append(_build_markdown_block(len(blocks)))

    if _should_include_surah_distribution(request):
        surah_values = _build_surah_distribution_values(request.filters)
        blocks.append(
            SearchResponseBlock(
                order=len(blocks),
                block_type="surah_distribution",
                title="Where this theme appears",
                payload={
                    "values": surah_values,
                    "y_label": "Mock mentions",
                    "max_value": max(value["value"] for value in surah_values),
                },
                explanation="Synthetic counts for end-to-end wiring tests.",
                confidence=0.74,
                provenance={"backend": "mock"},
                warning_text="",
                items=[],
            )
        )

    return SearchExecuteResponse(
        title=f"Search results for {request.query}",
        overall_confidence=0.82,
        render_schema_version=cfg.render_schema_version,
        metadata={"mock": True},
        blocks=blocks,
    )
