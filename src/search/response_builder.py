"""Render ``RetrievalCandidate``s into Django's ``SearchExecuteResponse`` blocks/items.

This is the *only* module in the retrieval domain that knows Django's block contract
(``src/api/schemas/search.py``) — everything upstream speaks ``RetrievalCandidate``. Keeping the
contract knowledge here means the pipeline and retrievers never couple to the presentation envelope.

It owns the grouping and ordering of the result blocks: a ``surah_distribution`` chart first (when
there are matches), then the Arabic Quran verses, then one block per translation language. Arabic
verses and translations are different kinds of content, so they live in separate blocks rather than a
single mixed list. Every result block shares ``block_type="ayah_results"`` (the same verse-card
renderer handles them); the block ``title`` and ``payload.language_code`` distinguish them. Highlight
``<mark>`` tags are kept: the typed frontend renderer renders them.
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

#: Display order for translation-language blocks; languages outside this list follow, sorted by code.
_LANGUAGE_ORDER = ("en", "fa")
_LANGUAGE_LABELS = {"en": "English", "fa": "Persian"}


def build_execute_response(
    request: SearchExecuteRequest,
    *,
    ayah_candidates: list[RetrievalCandidate],
    translation_candidates: list[RetrievalCandidate],
    surah_distribution: list[dict[str, int]],
    provenance: dict[str, Any],
    render_schema_version: str,
    confidence_scale_k: float,
) -> SearchExecuteResponse:
    """Build the Django-facing response: chart, then Arabic verses, then per-language translations."""
    overall_confidence = _confidence(
        [*ayah_candidates, *translation_candidates], scale_k=confidence_scale_k
    )
    blocks: list[SearchResponseBlock] = []
    order = _OrderCounter()

    if surah_distribution:
        blocks.append(_surah_distribution_block(surah_distribution, order=order.next()))

    blocks.append(
        _result_block(
            request,
            ayah_candidates,
            order=order.next(),
            title="Quran",
            language_code="ar",
            provenance=provenance,
            confidence_scale_k=confidence_scale_k,
            warn_when_empty=True,
        )
    )

    for language_code, group in _group_by_language(translation_candidates):
        blocks.append(
            _result_block(
                request,
                group,
                order=order.next(),
                title=f"{_language_label(language_code)} translations",
                language_code=language_code,
                provenance=provenance,
                confidence_scale_k=confidence_scale_k,
                warn_when_empty=False,
            )
        )

    return SearchExecuteResponse(
        title=f"Search results for {request.query}",
        overall_confidence=overall_confidence,
        render_schema_version=render_schema_version,
        metadata={"backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
        blocks=blocks,
    )


class _OrderCounter:
    def __init__(self) -> None:
        self._value = -1

    def next(self) -> int:
        self._value += 1
        return self._value


def _result_block(
    request: SearchExecuteRequest,
    candidates: list[RetrievalCandidate],
    *,
    order: int,
    title: str,
    language_code: str,
    provenance: dict[str, Any],
    confidence_scale_k: float,
    warn_when_empty: bool,
) -> SearchResponseBlock:
    items = _build_items(candidates, provenance)
    return SearchResponseBlock(
        order=order,
        block_type="ayah_results",
        title=title,
        payload={
            "query": request.query,
            "result_count": len(items),
            "language_code": language_code,
        },
        explanation="OpenSearch BM25 lexical retrieval over Quran corpus documents.",
        confidence=_confidence(candidates, scale_k=confidence_scale_k),
        provenance={"backend": OPEN_SEARCH_BACKEND_NAME, **provenance},
        warning_text=_NO_MATCHES_WARNING if (warn_when_empty and not items) else "",
        items=items,
    )


def _build_items(
    candidates: list[RetrievalCandidate], provenance: dict[str, Any]
) -> list[SearchResultItem]:
    max_score = max((candidate.score for candidate in candidates), default=0.0)
    items: list[SearchResultItem] = []
    # Re-rank 1..n within the block so each block reads as its own ranked list.
    for rank, candidate in enumerate(candidates, start=1):
        # Min-max normalize for relative bar heights; overall confidence carries absolute strength.
        score = 0.0 if max_score <= 0 else min(candidate.score / max_score, 1.0)
        metadata = candidate.metadata
        items.append(
            SearchResultItem(
                rank=rank,
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


def _group_by_language(
    candidates: list[RetrievalCandidate],
) -> list[tuple[str, list[RetrievalCandidate]]]:
    """Partition translation candidates by language, in a stable display order.

    Each candidate keeps its retrieval order within its language, so the per-language blocks stay
    ranked. Languages in :data:`_LANGUAGE_ORDER` come first; the rest follow sorted by code.
    """
    groups: dict[str, list[RetrievalCandidate]] = {}
    for candidate in candidates:
        language_code = str(candidate.metadata.get("language_code") or "").strip().casefold()
        if not language_code:
            continue
        groups.setdefault(language_code, []).append(candidate)
    return [(code, groups[code]) for code in _ordered_languages(groups)]


def _ordered_languages(groups: dict[str, list[RetrievalCandidate]]) -> list[str]:
    preferred = [code for code in _LANGUAGE_ORDER if code in groups]
    rest = sorted(code for code in groups if code not in _LANGUAGE_ORDER)
    return [*preferred, *rest]


def _language_label(language_code: str) -> str:
    return _LANGUAGE_LABELS.get(language_code, language_code.upper())


def _surah_distribution_block(
    distribution: list[dict[str, int]], *, order: int
) -> SearchResponseBlock:
    return SearchResponseBlock(
        order=order,
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
