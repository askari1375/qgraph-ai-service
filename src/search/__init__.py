"""Retrieval domain package.

`src/search/` is the home for QGraph's retrieval pipeline. The guiding idea is that OpenSearch is
*one retriever behind a contract*, never "the search system" — so Qdrant (vector) and Neo4j (graph)
can be added later as additional retrievers returning the same shape, without retrofitting the call
sites.

Layout:
- `contracts` — the types the whole pipeline speaks (`RetrievalCandidate`, `SearchFilters`,
  `QueryContext`, `Retriever`) plus the canonical cross-backend `content_type` vocabulary.
- `pipeline` — `RetrievalPipeline`: query_context -> [retrievers] -> (fuse) -> candidates.
- `response_builder` — candidates -> Django's `SearchExecuteResponse` blocks/items.
- `retrievers/` — concrete `Retriever` implementations (`LexicalRetriever` now).
- `indexing/` — the offline build/activate pipeline for the OpenSearch index.

`contracts` is fully implemented; the pipeline/builder/retriever/indexing modules are currently
skeletons that spell out the data flow in their docstrings and raise ``NotImplementedError`` until
their behavior is wired in.
"""
