"""Qdrant dense semantic retriever — adapts the vector backend to the ``Retriever`` contract.

``SemanticQdrantRetriever`` is the bridge between the generic pipeline and Qdrant: it resolves the
serving alias, compiles the typed ``SearchFilters`` to a Qdrant payload filter, queries by the
already-computed query embedding, and maps each point's payload to a ``RetrievalCandidate`` — so Qdrant
SDK types never leak past this module (mirroring how ``LexicalRetriever`` contains OpenSearch).

The query vector is **not** computed here: it is computed once per search (see
``embeddings.query.embed_query_for_search``) and handed in via ``QueryContext.query_embedding``, so the
ayah and translation scopes reuse one embedding instead of paying for it twice. A missing embedding is
an orchestration bug, surfaced loudly rather than silently embedding here.
"""

from __future__ import annotations

from typing import Any

from src.search.contracts import (
    RETRIEVER_QDRANT_DENSE,
    ContentType,
    QueryContext,
    RetrievalCandidate,
)
from src.search.vector.mapping import compile_qdrant_filter
from src.search.vector.qdrant_store import QdrantError, QdrantStore, VectorHit

#: Default per-scope candidate pool fetched for fusion — a bounded over-fetch beyond the caller's
#: ``top_k`` so later rank fusion has enough overlap to work with.
DEFAULT_CANDIDATE_POOL = 50

_METADATA_FIELDS = (
    "content_type",
    "surah_number",
    "ayah_number",
    "ayah_global_number",
    "language_code",
    "source_id",
    "source_name",
)


class SemanticQdrantRetriever:
    """A ``Retriever`` backed by Qdrant dense-vector similarity search."""

    name = RETRIEVER_QDRANT_DENSE

    def __init__(
        self,
        store: QdrantStore,
        alias: str,
        vector_name: str,
        *,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ):
        self.store = store
        self.alias = alias
        self.vector_name = vector_name
        self.candidate_pool = candidate_pool

    def retrieve(self, query_context: QueryContext) -> list[RetrievalCandidate]:
        if query_context.query_embedding is None:
            raise QdrantError(
                "semantic retrieval requires a precomputed query embedding",
                reason="semantic_query_embedding_missing",
            )
        collection = self.store.resolve_alias(self.alias)
        hits = self.store.query(
            collection,
            vector=query_context.query_embedding,
            vector_name=self.vector_name,
            query_filter=compile_qdrant_filter(query_context.filters),
            limit=max(query_context.top_k, self.candidate_pool),
        )
        return parse_semantic_candidates(hits)


def parse_semantic_candidates(hits: list[VectorHit]) -> list[RetrievalCandidate]:
    """Map Qdrant points to the generic candidate shape using only the point payload.

    The build stamps each point with enough payload (``build_point_payload``) to render a result without
    a second backend call, so no OpenSearch lookup is needed here.
    """
    candidates: list[RetrievalCandidate] = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        document_id = str(payload.get("document_id") or "").strip()
        if not document_id:
            continue
        canonical = str(payload.get("canonical_content_id") or "").strip() or document_id
        candidates.append(
            RetrievalCandidate(
                document_id=document_id,
                canonical_content_id=canonical,
                content_type=_resolve_content_type(payload.get("content_type")),
                retriever=RETRIEVER_QDRANT_DENSE,
                score=hit.score,
                rank=rank,
                text=str(payload.get("text") or ""),
                metadata={field: payload.get(field) for field in _METADATA_FIELDS},
                debug={"semantic_score": hit.score, "semantic_rank": rank},
            )
        )
    return candidates


def _resolve_content_type(value: Any) -> ContentType:
    try:
        return ContentType(value)
    except ValueError as exc:
        raise QdrantError(
            "Qdrant point has an unrecognized content_type",
            reason="unexpected_content_type",
            detail={"content_type": value},
        ) from exc
