"""Versioned weighted Reciprocal Rank Fusion for combining retriever candidate lists.

BM25 and cosine scores are not comparable, so fusion is rank-based, not score-based: each retriever
contributes ``weight / (rrf_k + rank)`` for a document, summed across retrievers. The same document
found by both backends therefore outranks one found by a single backend, without ever normalizing and
averaging raw scores.

Everything that defines the ranking — ``rrf_k``, the per-retriever weights, the candidate-pool size —
lives here as one versioned profile, recorded in response provenance (not in the immutable semantic
collection profile: fusion can change without rebuilding vectors).
"""

from __future__ import annotations

from typing import Any

from src.search.contracts import (
    RETRIEVER_OPENSEARCH_LEXICAL,
    RETRIEVER_QDRANT_DENSE,
    RetrievalCandidate,
)

FUSION_PROFILE_ID = "qgraph_rrf"
FUSION_PROFILE_VERSION = "v1"
RRF_K = 60
LEXICAL_WEIGHT = 1.0
SEMANTIC_WEIGHT = 1.0
#: Bounded over-fetch from each retriever so fusion has enough overlap to work with before truncation.
CANDIDATE_POOL = 50

DEFAULT_WEIGHTS: dict[str, float] = {
    RETRIEVER_OPENSEARCH_LEXICAL: LEXICAL_WEIGHT,
    RETRIEVER_QDRANT_DENSE: SEMANTIC_WEIGHT,
}


def fusion_profile() -> dict[str, Any]:
    """The fusion parameters, for response provenance."""
    return {
        "profile_id": FUSION_PROFILE_ID,
        "profile_version": FUSION_PROFILE_VERSION,
        "rrf_k": RRF_K,
        "weights": dict(DEFAULT_WEIGHTS),
    }


def reciprocal_rank_fusion(
    results_by_retriever: list[list[RetrievalCandidate]],
    *,
    weights: dict[str, float],
    rrf_k: int,
    collapse: bool,
    top_k: int,
) -> list[RetrievalCandidate]:
    """Fuse per-retriever ranked lists into one deterministic ranking.

    Merges occurrences of the same ``document_id`` across retrievers (keeping each backend's raw
    rank/score in ``debug``), optionally collapses by ``canonical_content_id``, sorts deterministically,
    and truncates to ``top_k``.
    """
    merged: dict[str, _Merged] = {}
    for results in results_by_retriever:
        for candidate in results:
            entry = merged.get(candidate.document_id)
            if entry is None:
                entry = _Merged()
                merged[candidate.document_id] = entry
            weight = weights.get(candidate.retriever, 1.0)
            entry.add(candidate, weight / (rrf_k + candidate.rank))

    fused = [entry.to_candidate() for entry in merged.values()]
    if collapse:
        fused = _collapse(fused)
    fused.sort(key=_ordering_key)
    return [
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(fused[:top_k], start=1)
    ]


class _Merged:
    """Accumulates one document's contributions from every retriever that returned it."""

    def __init__(self) -> None:
        self.representative: RetrievalCandidate | None = None
        self.fused_score = 0.0
        self.per_retriever: dict[str, dict[str, Any]] = {}

    def add(self, candidate: RetrievalCandidate, contribution: float) -> None:
        self.fused_score += contribution
        self.per_retriever[candidate.retriever] = {"rank": candidate.rank, "score": candidate.score}
        # Representative = the variant with the best individual rank (ties broken by retriever name) so
        # the displayed text/highlight is deterministic.
        current = self.representative
        if (
            current is None
            or candidate.rank < current.rank
            or (candidate.rank == current.rank and candidate.retriever < current.retriever)
        ):
            self.representative = candidate

    def best_rank(self) -> int:
        return min(info["rank"] for info in self.per_retriever.values())

    def to_candidate(self) -> RetrievalCandidate:
        assert self.representative is not None
        debug = {
            "per_retriever": self.per_retriever,
            "fusion": {
                "fused_score": self.fused_score,
                "best_rank": self.best_rank(),
                "profile_id": FUSION_PROFILE_ID,
                "profile_version": FUSION_PROFILE_VERSION,
            },
        }
        return self.representative.model_copy(update={"score": self.fused_score, "debug": debug})


def _ordering_key(candidate: RetrievalCandidate) -> tuple[float, int, str]:
    # Strongest fused score first; ties by best individual rank, then document_id for total determinism.
    return (-candidate.score, candidate.debug["fusion"]["best_rank"], candidate.document_id)


def _collapse(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Keep the strongest fused representative per ``canonical_content_id``."""
    best: dict[str, RetrievalCandidate] = {}
    for candidate in candidates:
        key = candidate.canonical_content_id
        current = best.get(key)
        if current is None or _ordering_key(candidate) < _ordering_key(current):
            best[key] = candidate
    return list(best.values())
