"""Weighted RRF: merge, weighting, deterministic ordering, collapse, and truncation."""

from src.search.contracts import (
    RETRIEVER_OPENSEARCH_LEXICAL,
    RETRIEVER_QDRANT_DENSE,
    ContentType,
    RetrievalCandidate,
)
from src.search.fusion import (
    DEFAULT_WEIGHTS,
    RRF_K,
    fusion_profile,
    reciprocal_rank_fusion,
)


def _candidate(document_id: str, rank: int, retriever: str, *, canonical: str | None = None):
    return RetrievalCandidate(
        document_id=document_id,
        canonical_content_id=canonical or document_id,
        content_type=ContentType.QURAN_AYAH,
        retriever=retriever,
        score=1.0,
        rank=rank,
    )


def _fuse(results, *, collapse=False, top_k=10):
    return reciprocal_rank_fusion(
        results, weights=DEFAULT_WEIGHTS, rrf_k=RRF_K, collapse=collapse, top_k=top_k
    )


def test_document_in_both_retrievers_outranks_single_backend():
    shared = "ayah:1:1:ar"
    solo = "ayah:2:255:ar"
    lexical = [_candidate(shared, 1, RETRIEVER_OPENSEARCH_LEXICAL)]
    semantic = [
        _candidate(shared, 1, RETRIEVER_QDRANT_DENSE),
        _candidate(solo, 2, RETRIEVER_QDRANT_DENSE),
    ]
    fused = _fuse([lexical, semantic])
    assert [c.document_id for c in fused] == [shared, solo]
    top = fused[0]
    assert set(top.debug["per_retriever"]) == {RETRIEVER_OPENSEARCH_LEXICAL, RETRIEVER_QDRANT_DENSE}
    assert top.debug["per_retriever"][RETRIEVER_OPENSEARCH_LEXICAL]["rank"] == 1
    assert top.score == 2 / (RRF_K + 1)


def test_tie_breaks_are_deterministic():
    # Two docs, each found by one backend at rank 1: equal fused score, equal best rank -> by id.
    a = [_candidate("ayah:1:1:ar", 1, RETRIEVER_OPENSEARCH_LEXICAL)]
    b = [_candidate("ayah:9:9:ar", 1, RETRIEVER_QDRANT_DENSE)]
    forward = [c.document_id for c in _fuse([a, b])]
    reversed_inputs = [c.document_id for c in _fuse([b, a])]
    assert forward == reversed_inputs == ["ayah:1:1:ar", "ayah:9:9:ar"]


def test_ranks_are_resequenced_and_truncated_to_top_k():
    lexical = [_candidate(f"ayah:1:{i}:ar", i, RETRIEVER_OPENSEARCH_LEXICAL) for i in range(1, 6)]
    fused = _fuse([lexical, []], top_k=2)
    assert [c.rank for c in fused] == [1, 2]
    assert len(fused) == 2


def test_collapse_keeps_one_per_canonical_id():
    arabic = _candidate("ayah:1:1:ar", 1, RETRIEVER_OPENSEARCH_LEXICAL, canonical="ayah:1:1")
    translation = _candidate(
        "ayah:1:1:translation:en", 2, RETRIEVER_QDRANT_DENSE, canonical="ayah:1:1"
    )
    collapsed = _fuse([[arabic], [translation]], collapse=True)
    assert len(collapsed) == 1
    assert collapsed[0].canonical_content_id == "ayah:1:1"


def test_no_collapse_keeps_variants():
    arabic = _candidate("ayah:1:1:ar", 1, RETRIEVER_OPENSEARCH_LEXICAL, canonical="ayah:1:1")
    translation = _candidate(
        "ayah:1:1:translation:en", 2, RETRIEVER_QDRANT_DENSE, canonical="ayah:1:1"
    )
    assert len(_fuse([[arabic], [translation]], collapse=False)) == 2


def test_fusion_profile_exposes_parameters():
    profile = fusion_profile()
    assert profile["profile_id"] == "qgraph_rrf"
    assert profile["rrf_k"] == RRF_K
    assert profile["weights"][RETRIEVER_QDRANT_DENSE] == 1.0
