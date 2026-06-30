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
from src.search.contracts import (
    RETRIEVER_OPENSEARCH_LEXICAL,
    RETRIEVER_QDRANT_DENSE,
    ContentType,
    RetrievalCandidate,
)

OPEN_SEARCH_BACKEND_NAME = "open_search"
HYBRID_BACKEND_NAME = "hybrid_rrf_v1"

# Retrieval-policy strings (the canonical constants live in services.search_service; duplicated here as
# bare literals to keep the response builder free of a service-layer import).
_LEXICAL_POLICY = "lexical_v1"
_HYBRID_POLICY = "hybrid_v1"
#: Conservative hybrid confidence heuristic id — RRF scores are not calibrated probabilities.
HYBRID_CONFIDENCE_POLICY = "qgraph_hybrid_confidence.v1"

_NO_MATCHES_WARNING = "No lexical matches were returned."

_BLOCK_EXPLANATION = {
    _LEXICAL_POLICY: "OpenSearch BM25 lexical retrieval over Quran corpus documents.",
    _HYBRID_POLICY: (
        "Hybrid retrieval: OpenSearch BM25 lexical + Qdrant dense semantic, fused by weighted RRF."
    ),
}
_ITEM_EXPLANATION = {
    _LEXICAL_POLICY: "Ranked by OpenSearch lexical score.",
    _HYBRID_POLICY: "Ranked by weighted reciprocal rank fusion of lexical and semantic retrieval.",
}

#: Display order for translation-language blocks; languages outside this list follow, sorted by code.
_LANGUAGE_ORDER = ("en", "fa")
_LANGUAGE_LABELS = {"en": "English", "fa": "Persian"}


def _backend_name(retrieval_policy: str) -> str:
    return HYBRID_BACKEND_NAME if retrieval_policy == _HYBRID_POLICY else OPEN_SEARCH_BACKEND_NAME


def build_execute_response(
    request: SearchExecuteRequest,
    *,
    ayah_candidates: list[RetrievalCandidate],
    translation_candidates: list[RetrievalCandidate],
    surah_distribution: list[dict[str, int]],
    provenance: dict[str, Any],
    render_schema_version: str,
    confidence_scale_k: float,
    retrieval_policy: str = _LEXICAL_POLICY,
) -> SearchExecuteResponse:
    """Build the Django-facing response: chart, then Arabic verses, then per-language translations.

    ``retrieval_policy`` selects backend labelling, explanations, and the confidence heuristic; the
    candidate shape is identical for both policies, so blocks/items render the same way.
    """
    backend_name = _backend_name(retrieval_policy)
    overall_confidence = _confidence(
        [*ayah_candidates, *translation_candidates],
        policy=retrieval_policy,
        scale_k=confidence_scale_k,
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
            retrieval_policy=retrieval_policy,
            backend_name=backend_name,
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
                retrieval_policy=retrieval_policy,
                backend_name=backend_name,
                warn_when_empty=False,
            )
        )

    return SearchExecuteResponse(
        title=f"Search results for {request.query}",
        overall_confidence=overall_confidence,
        render_schema_version=render_schema_version,
        metadata={"backend": backend_name, **provenance},
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
    retrieval_policy: str,
    backend_name: str,
    warn_when_empty: bool,
) -> SearchResponseBlock:
    items = _build_items(candidates, provenance, retrieval_policy, backend_name)
    return SearchResponseBlock(
        order=order,
        block_type="ayah_results",
        title=title,
        payload={
            "query": request.query,
            "result_count": len(items),
            "language_code": language_code,
        },
        explanation=_BLOCK_EXPLANATION.get(retrieval_policy, _BLOCK_EXPLANATION[_LEXICAL_POLICY]),
        confidence=_confidence(candidates, policy=retrieval_policy, scale_k=confidence_scale_k),
        provenance={"backend": backend_name, **provenance},
        warning_text=_NO_MATCHES_WARNING if (warn_when_empty and not items) else "",
        items=items,
    )


def _build_items(
    candidates: list[RetrievalCandidate],
    provenance: dict[str, Any],
    retrieval_policy: str,
    backend_name: str,
) -> list[SearchResultItem]:
    max_score = max((candidate.score for candidate in candidates), default=0.0)
    explanation = _ITEM_EXPLANATION.get(retrieval_policy, _ITEM_EXPLANATION[_LEXICAL_POLICY])
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
                explanation=explanation,
                provenance=_item_provenance(candidate, provenance, backend_name),
            )
        )
    return items


def _item_provenance(
    candidate: RetrievalCandidate, provenance: dict[str, Any], backend_name: str
) -> dict[str, Any]:
    """Per-item provenance: per-backend ranks/scores and the fused score when a candidate was fused.

    A lexical-only candidate (no fusion ``debug``) keeps the flat ``lexical_score`` shape; a fused
    candidate exposes each backend's rank/score plus the fused score/rank for debugging and eval.
    """
    item_prov: dict[str, Any] = {"backend": backend_name, "document_id": candidate.document_id}
    per_retriever = candidate.debug.get("per_retriever") if candidate.debug else None
    if per_retriever:
        lexical = per_retriever.get(RETRIEVER_OPENSEARCH_LEXICAL)
        semantic = per_retriever.get(RETRIEVER_QDRANT_DENSE)
        if lexical is not None:
            item_prov["lexical_rank"] = lexical["rank"]
            item_prov["lexical_score"] = lexical["score"]
        if semantic is not None:
            item_prov["semantic_rank"] = semantic["rank"]
            item_prov["semantic_similarity"] = semantic["score"]
        item_prov["fused_score"] = candidate.debug.get("fusion", {}).get("fused_score")
        item_prov["fused_rank"] = candidate.rank
    else:
        item_prov["lexical_score"] = candidate.score
    item_prov.update(provenance)
    return item_prov


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


def _confidence(candidates: list[RetrievalCandidate], *, policy: str, scale_k: float) -> float:
    """Ranker-aware confidence: the BM25 heuristic for lexical, a conservative one for hybrid RRF."""
    if policy == _HYBRID_POLICY:
        return _hybrid_confidence(candidates)
    return _lexical_confidence(candidates, scale_k=scale_k)


def _lexical_confidence(candidates: list[RetrievalCandidate], *, scale_k: float) -> float:
    """Map the strongest absolute BM25 score to a bounded 0..1 confidence."""
    top_absolute_score = max((candidate.score for candidate in candidates), default=0.0)
    if top_absolute_score <= 0.0 or scale_k <= 0.0:
        return 0.0
    return 1.0 - exp(-top_absolute_score / scale_k)


def _hybrid_confidence(candidates: list[RetrievalCandidate]) -> float:
    """Conservative confidence from cross-backend agreement — never a calibrated RRF probability.

    RRF scores are not comparable to a probability, so confidence is driven by *agreement*: the share
    of results both retrievers surfaced. Bounded to a deliberately narrow band so the UI never reads a
    fused score as certainty. Versioned as :data:`HYBRID_CONFIDENCE_POLICY`.
    """
    if not candidates:
        return 0.0
    agreed = sum(1 for c in candidates if c.debug and len(c.debug.get("per_retriever", {})) >= 2)
    overlap = agreed / len(candidates)
    return round(0.35 + 0.4 * overlap, 4)


def _snippet(text: str, max_length: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 1].rstrip()}…"
