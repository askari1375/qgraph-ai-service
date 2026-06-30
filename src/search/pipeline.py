"""Retrieval pipeline: query_context -> [retrievers] -> (fuse) -> ranked candidates.

The pipeline is the single place that runs retrievers and fuses their results, so the rest of the
app never knows how many retrievers exist or which one answered.

Data flow:
    QueryContext
      -> for each retriever: retriever.retrieve(query_context)
      -> _fuse(results)            # pass-through for one retriever; weighted RRF for several
      -> ranked list[RetrievalCandidate]
"""

from __future__ import annotations

from src.search.contracts import QueryContext, RetrievalCandidate, Retriever
from src.search.fusion import CANDIDATE_POOL, DEFAULT_WEIGHTS, RRF_K, reciprocal_rank_fusion


class RetrievalPipeline:
    """Runs the configured retrievers for a query and fuses their candidates.

    With one retriever (lexical-only) this is a pass-through. With several, each retriever over-fetches
    a bounded candidate pool and the lists are combined by weighted Reciprocal Rank Fusion, then
    truncated to the caller's ``top_k``.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        weights: dict[str, float] | None = None,
        rrf_k: int = RRF_K,
        candidate_pool: int = CANDIDATE_POOL,
    ):
        self.retrievers = retrievers
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def run(self, query_context: QueryContext) -> list[RetrievalCandidate]:
        """Retrieve from every retriever, fuse, and return ranked candidates."""
        if len(self.retrievers) <= 1:
            results = [retriever.retrieve(query_context) for retriever in self.retrievers]
            return results[0] if results else []
        # Fusion mode: over-fetch a pool from each retriever so fusion has overlap to work with, then
        # truncate the fused ranking back to the caller's top_k.
        fetch_context = query_context.model_copy(
            update={"top_k": max(query_context.top_k, self.candidate_pool)}
        )
        results = [retriever.retrieve(fetch_context) for retriever in self.retrievers]
        return reciprocal_rank_fusion(
            results,
            weights=self.weights,
            rrf_k=self.rrf_k,
            collapse=query_context.collapse,
            top_k=query_context.top_k,
        )
