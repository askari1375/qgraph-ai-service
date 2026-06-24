"""OpenSearch lexical retriever — adapts the BM25 backend to the ``Retriever`` contract.

``LexicalRetriever`` is the bridge between the generic pipeline and the OpenSearch-specific backend:
it owns the query-building and the ``LexicalSearchHit`` -> ``RetrievalCandidate`` mapping, so the
OpenSearch types never leak past this module.

When implemented it wraps ``src.services.opensearch_lexical.OpenSearchLexicalBackend`` and maps each
``LexicalSearchHit`` into a ``RetrievalCandidate``, stamping ``retriever =
RETRIEVER_OPENSEARCH_LEXICAL`` and deriving ``canonical_content_id``/``content_type`` from the hit
metadata. It builds the query over all language ``content_*`` fields (the detected language only
nudges boosts), compiles ``SearchFilters`` to OpenSearch clauses, supports caller-controlled
collapse, and queries the serving alias rather than a pinned index name.
"""

from __future__ import annotations

from src.search.contracts import (
    RETRIEVER_OPENSEARCH_LEXICAL,
    QueryContext,
    RetrievalCandidate,
)


class LexicalRetriever:
    """A ``Retriever`` backed by OpenSearch BM25 lexical search."""

    name = RETRIEVER_OPENSEARCH_LEXICAL

    def retrieve(self, query_context: QueryContext) -> list[RetrievalCandidate]:
        """Query OpenSearch and map hits into candidates.

        Not implemented yet: build the request from ``query_context`` (all ``content_*`` fields;
        detected language only nudges boosts), run it against the index alias, and map each
        ``LexicalSearchHit`` -> ``RetrievalCandidate``.
        """
        raise NotImplementedError("LexicalRetriever.retrieve is not implemented yet")
