"""Offline checks on the cross-lingual semantic eval set + lexical no-regression under hybrid fusion.

Real semantic-quality validation needs a collection built with the production provider (the paid
Phase-4 gate); here we assert only what holds offline: the set is well-formed, the cross-lingual intent
is wired, every case is still PENDING (unverified), and a strong lexical hit is not dropped when its
list is fused with semantic results.
"""

from src.search.contracts import (
    RETRIEVER_OPENSEARCH_LEXICAL,
    RETRIEVER_QDRANT_DENSE,
    ContentType,
    RetrievalCandidate,
)
from src.search.eval.semantic_eval_set import (
    SEMANTIC_EVAL_SET_VERSION,
    SEMANTIC_GOLDEN_QUERIES,
    cross_lingual_cases,
)
from src.search.fusion import DEFAULT_WEIGHTS, RRF_K, reciprocal_rank_fusion
from src.search.indexing.eval_set import GOLDEN_QUERIES, ExpectationStatus

_CANONICAL_PREFIXES = ("ayah:", "surah:")


def test_versioned_and_non_empty():
    assert SEMANTIC_EVAL_SET_VERSION
    assert SEMANTIC_GOLDEN_QUERIES


def test_case_ids_are_unique():
    ids = [case.id for case in SEMANTIC_GOLDEN_QUERIES]
    assert len(ids) == len(set(ids))


def test_all_cases_pending_until_real_build_confirms():
    # Expectations are reviewed judgments confirmed only after the paid Phase-4 build.
    assert all(case.status is ExpectationStatus.PENDING for case in SEMANTIC_GOLDEN_QUERIES)


def test_cross_lingual_cases_present_and_target_arabic():
    cases = cross_lingual_cases()
    assert cases
    # The core target: a non-Arabic query whose correct answer is an Arabic ayah.
    assert any(c.language in {"fa", "en"} and c.expected_language == "ar" for c in cases)
    for case in cases:
        assert case.language != case.expected_language


def test_canonical_ids_well_formed():
    for case in SEMANTIC_GOLDEN_QUERIES:
        for canonical_id in case.must_include_canonical_ids:
            assert canonical_id.startswith(_CANONICAL_PREFIXES)


def test_confirmed_lexical_hit_survives_hybrid_fusion():
    # No-regression: a lexical-confirmed ayah at rank 1 must still appear after fusing with semantic
    # noise, so adding the semantic retriever cannot silently bury a strong lexical result.
    confirmed = [
        case.must_include_canonical_ids[0]
        for case in GOLDEN_QUERIES
        if case.is_hard and case.must_include_canonical_ids[0].startswith("ayah:")
    ]
    assert confirmed  # guard: the lexical golden set still has confirmed ayah cases

    for canonical_id in confirmed:
        document_id = f"{canonical_id}:ar"
        lexical = [
            RetrievalCandidate(
                document_id=document_id,
                canonical_content_id=canonical_id,
                content_type=ContentType.QURAN_AYAH,
                retriever=RETRIEVER_OPENSEARCH_LEXICAL,
                score=9.0,
                rank=1,
            )
        ]
        semantic = [
            RetrievalCandidate(
                document_id=f"ayah:9:{i}:ar",
                canonical_content_id=f"ayah:9:{i}",
                content_type=ContentType.QURAN_AYAH,
                retriever=RETRIEVER_QDRANT_DENSE,
                score=0.8,
                rank=i,
            )
            for i in range(1, 6)
        ]
        fused = reciprocal_rank_fusion(
            [lexical, semantic], weights=DEFAULT_WEIGHTS, rrf_k=RRF_K, collapse=False, top_k=10
        )
        assert document_id in {candidate.document_id for candidate in fused}
