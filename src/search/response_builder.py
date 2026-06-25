"""Render ``RetrievalCandidate``s into Django's ``SearchExecuteResponse`` blocks/items.

This is the *only* module in the retrieval domain that knows Django's block contract
(``src/api/schemas/search.py``) — everything upstream speaks ``RetrievalCandidate``. Keeping the
contract knowledge here means the pipeline and retrievers never couple to the presentation envelope.

It emits a typed ``ayah_results`` block (one item per matched document, carrying the data a verse card
renders — reference, text, highlight, source, score) and, when there are matches, a
``surah_distribution`` block with the real per-surah match counts. Highlight ``<mark>`` tags are kept:
the typed frontend renderer renders them (unlike the markdown bridge, which had to strip them).
"""

from __future__ import annotations

from math import exp
from typing import Any

from src.api.schemas.search import (
    SearchExecuteRequest,
    SearchExecuteResponse,
    SearchResponseBlock,
    SearchResultItem,
)
from src.search.contracts import ContentType, RetrievalCandidate

OPEN_SEARCH_BACKEND_NAME = "open_search"

_NO_MATCHES_WARNING = "No lexical matches were returned."


def build_execute_response(
    request: SearchExecuteRequest,
    candidates: list[RetrievalCandidate],
    *,
    surah_distribution: list[dict[str, int]],
    provenance: dict[str, Any],
    render_schema_version: str,
    confidence_scale_k: float,
) -> SearchExecuteResponse:
    """Build the Django-facing response: a typed ayah_results block (+ a distribution chart)."""
    confidence = _confidence(candidates, scale_k=confidence_scale_k)
    items = _build_items(candidates, provenance)
    blocks = [
        SearchResponseBlock(
            order=0,
            block_type="ayah_results",
            title="Lexical matches",
            payload={"query": request.query, "result_count": len(items)},
            explanation="OpenSearch BM25 lexical retrieval over Quran corpus documents.",
            confidence=confidence,
            provenance={"backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
            warning_text="" if items else _NO_MATCHES_WARNING,
            items=items,
        )
    ]
    if surah_distribution:
        blocks.append(_surah_distribution_block(surah_distribution))

    return SearchExecuteResponse(
        title=f"Search results for {request.query}",
        overall_confidence=confidence,
        render_schema_version=render_schema_version,
        metadata={"backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
        blocks=blocks,
    )


def _build_items(
    candidates: list[RetrievalCandidate], provenance: dict[str, Any]
) -> list[SearchResultItem]:
    max_score = max((candidate.score for candidate in candidates), default=0.0)
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
                title=_reference(candidate),
                snippet_text=_snippet(candidate.text),
                highlighted_text=candidate.highlighted_text or _snippet(candidate.text),
                match_metadata={
                    "document_id": candidate.document_id,
                    "canonical_content_id": candidate.canonical_content_id,
                    "content_type": candidate.content_type.value,
                    "text": candidate.text,
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
                    **provenance,
                },
            )
        )
    return items


def _surah_distribution_block(distribution: list[dict[str, int]]) -> SearchResponseBlock:
    return SearchResponseBlock(
        order=1,
        block_type="surah_distribution",
        title="Where this appears",
        payload={
            "values": distribution,
            "y_label": "Matches",
            "max_value": max(entry["value"] for entry in distribution),
        },
        explanation="Match counts per surah for this query.",
        confidence=0.0,
        provenance={"backend": OPEN_SEARCH_BACKEND_NAME},
        warning_text="",
        items=[],
    )


def _result_type(content_type: ContentType) -> str:
    return "surah" if content_type is ContentType.SURAH_NAME else "ayah"


def _reference(candidate: RetrievalCandidate) -> str:
    metadata = candidate.metadata
    surah = metadata.get("surah_number")
    ayah = metadata.get("ayah_number")
    if candidate.content_type is ContentType.SURAH_NAME and surah:
        return f"Surah {surah}"
    if surah and ayah:
        return f"Surah {surah}, Ayah {ayah}"
    return "Quran corpus match"


def _confidence(candidates: list[RetrievalCandidate], *, scale_k: float) -> float:
    """Map the strongest absolute lexical score to a bounded 0..1 confidence."""
    top_absolute_score = max((candidate.score for candidate in candidates), default=0.0)
    if top_absolute_score <= 0.0 or scale_k <= 0.0:
        return 0.0
    return 1.0 - exp(-top_absolute_score / scale_k)


def _snippet(text: str, max_length: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 1].rstrip()}…"
