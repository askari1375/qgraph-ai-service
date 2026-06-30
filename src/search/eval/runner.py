"""Executable semantic eval: run the golden cross-lingual cases against a physical collection.

The eval set (:mod:`src.search.eval.semantic_eval_set`) is the harness; this is the engine that turns it
into a repeatable command instead of a notebook. Each case is embedded once with the production provider
and queried against the *physical* collection (by name, not the serving alias) so a freshly built
collection can be judged **before** it is activated. Results separate **hard failures** — a CONFIRMED
case missing a must-include canonical id, which blocks activation — from **soft misses** — a PENDING
case whose candidate ids are not yet present, which is reported for the owner's editorial review and
never fails a build. Promoting PENDING→CONFIRMED stays the owner's judgment after seeing real top-K.
"""

from __future__ import annotations

from typing import Any

from src.search.contracts import QueryContext, SearchFilters
from src.search.embeddings.contracts import EmbeddingProvider
from src.search.embeddings.query import embed_query_for_search
from src.search.eval.semantic_eval_set import (
    SEMANTIC_EVAL_SET_VERSION,
    SEMANTIC_GOLDEN_QUERIES,
)
from src.search.indexing.eval_set import GoldenQuery
from src.search.retrievers.semantic_qdrant import parse_semantic_candidates
from src.search.vector.mapping import compile_qdrant_filter
from src.search.vector.qdrant_store import QdrantStore


def evaluate_semantic_collection(
    collection: str,
    *,
    store: QdrantStore,
    provider: EmbeddingProvider,
    vector_name: str,
    cases: tuple[GoldenQuery, ...] = SEMANTIC_GOLDEN_QUERIES,
) -> dict[str, Any]:
    """Run every eval case against ``collection`` and return a structured report.

    ``ok`` is ``True`` when no CONFIRMED case is missing a must-include id. Soft misses (PENDING) are
    listed but never affect ``ok``, so this is safe to gate a build on from the first all-PENDING run.
    """
    case_reports = [
        _evaluate_case(case, collection, store=store, provider=provider, vector_name=vector_name)
        for case in cases
    ]
    hard_failures = [report["id"] for report in case_reports if report["hard_failure"]]
    soft_misses = [report["id"] for report in case_reports if report["soft_miss"]]
    return {
        "collection": collection,
        "eval_set_version": SEMANTIC_EVAL_SET_VERSION,
        "case_count": len(case_reports),
        "hard_failures": hard_failures,
        "soft_misses": soft_misses,
        "ok": not hard_failures,
        "cases": case_reports,
    }


def _evaluate_case(
    case: GoldenQuery,
    collection: str,
    *,
    store: QdrantStore,
    provider: EmbeddingProvider,
    vector_name: str,
) -> dict[str, Any]:
    filters = SearchFilters(content_types=list(case.scope))
    context = QueryContext(
        raw_query=case.query,
        detected_language=case.language,
        filters=filters,
        top_k=case.top_k,
    )
    embedding = embed_query_for_search(provider, context)
    hits = store.query(
        collection,
        vector=embedding,
        vector_name=vector_name,
        query_filter=compile_qdrant_filter(filters),
        limit=case.top_k,
    )
    candidates = parse_semantic_candidates(hits)[: case.top_k]
    found_ids = {candidate.canonical_content_id for candidate in candidates}
    missing = [cid for cid in case.must_include_canonical_ids if cid not in found_ids]
    return {
        "id": case.id,
        "status": case.status.value,
        "query": case.query,
        "language": case.language,
        "expected_language": case.expected_language,
        "hit_count": len(candidates),
        "top_canonical_ids": [candidate.canonical_content_id for candidate in candidates],
        "must_include_canonical_ids": list(case.must_include_canonical_ids),
        "missing_canonical_ids": missing,
        # CONFIRMED + missing => blocks activation; PENDING + missing => reported only.
        "hard_failure": case.is_hard and bool(missing),
        "soft_miss": not case.is_hard and bool(missing),
    }
