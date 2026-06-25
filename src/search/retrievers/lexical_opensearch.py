"""OpenSearch lexical retriever — adapts the BM25 backend to the ``Retriever`` contract.

``LexicalRetriever`` is the bridge between the generic pipeline and the OpenSearch-specific backend:
it owns the query-building and the hit -> ``RetrievalCandidate`` mapping, so the OpenSearch types
never leak past this module. It searches **all** language ``content_*`` fields regardless of the
query's detected language (a soft hint); the primary normalize-don't-stem fields carry the boost and
the ``.stemmed``/``.exact`` sub-fields add lower-boost recall. ``canonical_content_id`` collapse is a
caller-controlled query-time flag.

``build_search_body`` is shared with the index build's golden-set validation so the query shape that
is validated is the same one served.
"""

from __future__ import annotations

from typing import Any

from src.search.contracts import (
    RETRIEVER_OPENSEARCH_LEXICAL,
    ContentType,
    QueryContext,
    RetrievalCandidate,
)
from src.search.indexing.documents import DOCUMENT_SCHEMA_VERSION
from src.search.indexing.mapping import ANALYSIS_PROFILE_VERSION
from src.search.indexing.normalization import NORMALIZATION_PROFILE_VERSION
from src.search.opensearch_client import (
    OpenSearchAdapter,
    OpenSearchError,
    read_index_profile,
    search,
)

# Primary normalize-don't-stem fields carry the boost; the stemmed/exact sub-fields add recall.
_PRIMARY_FIELDS = ["content_ar^3", "content_fa^2", "content_en^2", "content_general"]
_RECALL_FIELDS = ["content_ar.stemmed", "content_fa.stemmed", "content_en.exact"]
_RECALL_BOOST = 0.3
_CONTENT_FIELDS = ("content_ar", "content_fa", "content_en", "content_general")


def build_search_body(query_context: QueryContext) -> dict[str, Any]:
    """Build the OpenSearch ``_search`` body for a query context."""
    should = [
        {
            "multi_match": {
                "query": query_context.raw_query,
                "fields": _PRIMARY_FIELDS,
                "type": "best_fields",
            }
        },
        {
            "multi_match": {
                "query": query_context.raw_query,
                "fields": _RECALL_FIELDS,
                "type": "best_fields",
                "boost": _RECALL_BOOST,
            }
        },
    ]
    bool_query: dict[str, Any] = {"should": should, "minimum_should_match": 1}
    filter_clauses = query_context.filters.to_opensearch_filter()
    if filter_clauses:
        bool_query["filter"] = filter_clauses

    body: dict[str, Any] = {
        "size": query_context.top_k,
        "query": {"bool": bool_query},
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {field: {} for field in _CONTENT_FIELDS},
        },
    }
    if query_context.collapse:
        body["collapse"] = {"field": "canonical_content_id"}
    return body


class LexicalRetriever:
    """A ``Retriever`` backed by OpenSearch BM25 lexical search."""

    name = RETRIEVER_OPENSEARCH_LEXICAL

    def __init__(self, adapter: OpenSearchAdapter, target: str):
        self.adapter = adapter
        self.target = target

    def retrieve(self, query_context: QueryContext) -> list[RetrievalCandidate]:
        ensure_compatible_index(self.adapter, self.target)
        payload = search(self.adapter, self.target, build_search_body(query_context))
        return parse_candidates(payload)


def compatibility_mismatches(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the build-profile fields that disagree with the running code (empty == compatible).

    Compares only the code-constant compatibility versions; snapshot id/hash are provenance, not a
    compatibility gate.
    """
    expected = {
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "analysis_profile_version": ANALYSIS_PROFILE_VERSION,
    }
    return {
        field: {"expected": value, "actual": profile.get(field)}
        for field, value in expected.items()
        if profile.get(field) != value
    }


def ensure_compatible_index(adapter: OpenSearchAdapter, target: str) -> None:
    """Refuse to serve from an index whose build profile disagrees with the running code."""
    mismatches = compatibility_mismatches(read_index_profile(adapter, target))
    if mismatches:
        raise OpenSearchError(
            "OpenSearch index profile is incompatible with the running code",
            reason="index_profile_mismatch",
            detail={"mismatches": mismatches},
        )


def parse_candidates(payload: dict[str, Any]) -> list[RetrievalCandidate]:
    raw_hits = payload.get("hits", {}).get("hits", [])
    if not isinstance(raw_hits, list):
        raise OpenSearchError(
            "OpenSearch search returned malformed hits", reason="search_response_malformed"
        )

    candidates: list[RetrievalCandidate] = []
    for rank, raw_hit in enumerate(raw_hits, start=1):
        if not isinstance(raw_hit, dict):
            continue
        source = raw_hit.get("_source")
        if not isinstance(source, dict):
            source = {}
        document_id = str(raw_hit.get("_id") or source.get("id") or "").strip()
        if not document_id:
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        canonical = str(source.get("canonical_content_id") or "").strip() or document_id
        text = _extract_text(source)
        highlighted, matched_fields = _extract_highlight(raw_hit)
        candidates.append(
            RetrievalCandidate(
                document_id=document_id,
                canonical_content_id=canonical,
                content_type=_resolve_content_type(metadata.get("content_type")),
                retriever=RETRIEVER_OPENSEARCH_LEXICAL,
                score=float(raw_hit.get("_score") or 0.0),
                rank=rank,
                text=text,
                highlighted_text=highlighted or text,
                metadata=metadata,
                matched_fields=matched_fields,
            )
        )
    return candidates


def _resolve_content_type(value: Any) -> ContentType:
    try:
        return ContentType(value)
    except ValueError as exc:
        raise OpenSearchError(
            "OpenSearch hit has an unrecognized content_type",
            reason="unexpected_content_type",
            detail={"content_type": value},
        ) from exc


def _extract_text(source: dict[str, Any]) -> str:
    for field in _CONTENT_FIELDS:
        value = source.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_highlight(raw_hit: dict[str, Any]) -> tuple[str, list[str]]:
    highlight = raw_hit.get("highlight")
    if not isinstance(highlight, dict):
        return "", []
    matched_fields = [field for field in _CONTENT_FIELDS if highlight.get(field)]
    for field in _CONTENT_FIELDS:
        values = highlight.get(field)
        if isinstance(values, list) and values:
            return str(values[0]), matched_fields
    return "", matched_fields
