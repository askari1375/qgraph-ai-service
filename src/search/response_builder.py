"""Render ``RetrievalCandidate``s into Django's ``SearchExecuteResponse`` blocks/items.

This is the *only* module in the retrieval domain that knows Django's block contract
(``src/api/schemas/search.py``) — everything upstream speaks ``RetrievalCandidate``. Keeping the
contract knowledge here means the pipeline and retrievers never couple to the presentation envelope.

Current presentation: the ranked candidates are rendered as a supported **markdown** block (the
frontend renders ``markdown`` today). A typed verse block — with per-item bookmark/feedback and true
highlighting — is the scheduled next step; the markdown bridge intentionally drops per-item
interactivity and the ``<mark>`` highlights (markdown renders no raw HTML).
"""

from __future__ import annotations

import re
from math import exp
from typing import Any

from src.api.schemas.search import (
    SearchExecuteRequest,
    SearchExecuteResponse,
    SearchResponseBlock,
)
from src.search.contracts import ContentType, RetrievalCandidate

OPEN_SEARCH_BACKEND_NAME = "open_search"

_MARK_TAG_RE = re.compile(r"</?mark>")
_NO_MATCHES_WARNING = "No lexical matches were returned."


def build_execute_response(
    request: SearchExecuteRequest,
    candidates: list[RetrievalCandidate],
    *,
    provenance: dict[str, Any],
    render_schema_version: str,
    confidence_scale_k: float,
) -> SearchExecuteResponse:
    """Build the Django-facing response from ranked candidates as a markdown block."""
    confidence = _confidence(candidates, scale_k=confidence_scale_k)
    block = SearchResponseBlock(
        order=0,
        block_type="markdown",
        title="Lexical matches",
        payload={
            "headline": f"{len(candidates)} result(s) for “{request.query}”",
            "content": _render_markdown(request.query, candidates),
        },
        explanation="OpenSearch BM25 lexical retrieval over Quran corpus documents.",
        confidence=confidence,
        provenance={"backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
        warning_text="" if candidates else _NO_MATCHES_WARNING,
        items=[],
    )
    return SearchExecuteResponse(
        title=f"Search results for {request.query}",
        overall_confidence=confidence,
        render_schema_version=render_schema_version,
        metadata={"backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
        blocks=[block],
    )


def _render_markdown(query: str, candidates: list[RetrievalCandidate]) -> str:
    if not candidates:
        return f"No lexical matches were found for **{query}**."
    lines = [
        "| # | Reference | Match | Source |",
        "| ---: | --- | --- | --- |",
    ]
    for candidate in candidates:
        reference = _reference(candidate)
        match = _markdown_cell(_snippet(_strip_marks(candidate.highlighted_text or candidate.text)))
        source = _markdown_cell(str(candidate.metadata.get("source_name") or ""))
        lines.append(f"| {candidate.rank} | {reference} | {match} | {source} |")
    return "\n".join(lines)


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


def _strip_marks(text: str) -> str:
    return _MARK_TAG_RE.sub("", text)


def _markdown_cell(text: str) -> str:
    # Collapse whitespace and escape the pipe so a cell never breaks the table.
    return " ".join(text.split()).replace("|", "\\|")


def _snippet(text: str, max_length: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 1].rstrip()}…"
