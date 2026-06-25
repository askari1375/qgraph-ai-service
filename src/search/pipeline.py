"""Retrieval pipeline: query_context -> [retrievers] -> (fuse) -> ranked candidates.

The pipeline is the single place that runs retrievers and fuses their results, so the rest of the
app never knows how many retrievers exist or which one answered.

Intended data flow once filled in:
    QueryContext
      -> for each retriever: retriever.retrieve(query_context)
      -> _fuse(results)            # pass-through today; RRF lands here at the 2nd retriever
      -> ranked list[RetrievalCandidate]
"""

from __future__ import annotations

from src.search.contracts import QueryContext, RetrievalCandidate, Retriever


class RetrievalPipeline:
    """Runs the configured retrievers for a query and fuses their candidates.

    There is exactly one retriever today (the lexical one); the structure exists so adding a second
    backend is "append a retriever + implement fusion", not a rewrite.
    """

    def __init__(self, retrievers: list[Retriever]):
        self.retrievers = retrievers

    def run(self, query_context: QueryContext) -> list[RetrievalCandidate]:
        """Retrieve from every retriever, fuse, and return ranked candidates."""
        results = [retriever.retrieve(query_context) for retriever in self.retrievers]
        return self._fuse(results)

    def _fuse(
        self, results_by_retriever: list[list[RetrievalCandidate]]
    ) -> list[RetrievalCandidate]:
        """Combine per-retriever candidate lists into one ranking.

        With a single retriever this is a pass-through (its list is returned as-is). This is the seam
        where Reciprocal Rank Fusion lands the day a second retriever exists; dedup-on-fusion can
        also key on ``canonical_content_id`` here. Until then, multiple lists are simply concatenated.
        """
        if not results_by_retriever:
            return []
        if len(results_by_retriever) == 1:
            return results_by_retriever[0]
        merged: list[RetrievalCandidate] = []
        for results in results_by_retriever:
            merged.extend(results)
        return merged
