# Hybrid retrieval operations

How the AI service serves semantic + lexical search, switches the retrieval policy, and stays safe at
the boundaries. The AI service owns both backends; Django sees only the typed result blocks.

## The two policies

`QGRAPH_AI_SEARCH_RETRIEVAL_POLICY` selects the retrieval path:

- **`lexical_v1`** (default) — OpenSearch BM25 only. The original path; unchanged.
- **`hybrid_v1`** — OpenSearch lexical **and** Qdrant dense semantic, fused by weighted Reciprocal
  Rank Fusion. Opt-in; switch to it only after a semantic collection and embedding provider are built,
  activated, and pass readiness.

Both are real backends. `hybrid_v1` **never silently falls back** to lexical-only: if Qdrant or the
embedding provider is unavailable, `/v1/search/execute` returns a `503` service error rather than
quietly degrading. An operator may deliberately run `lexical_v1`; that is a product/runtime choice, not
error recovery.

## Query flow (`hybrid_v1`)

1. Parse the request into typed `SearchFilters`; build the ayah and translation scope contexts.
2. Compute **one** query embedding (`embed_query_for_search`) and reuse it across both scopes.
3. For each scope, run the lexical and semantic retrievers over a bounded candidate pool.
4. Fuse by weighted RRF (`fusion.py`, profile `qgraph_rrf` v1, `rrf_k=60`): merge the same
   `document_id` across backends, keep each backend's raw rank/score, optionally collapse by
   `canonical_content_id`, sort deterministically, truncate to `top_k`.
5. The `surah_distribution` chart stays lexical-only (a stable verse-count aggregation).
6. Render the existing typed blocks; provenance explains lexical, semantic, and fused participation.

## Confidence

The BM25 `1 - exp(-score/k)` heuristic does not apply to RRF scores (they are not calibrated
probabilities). Under `hybrid_v1`, confidence is a deliberately conservative band driven by
cross-backend **agreement** (the share of results both retrievers surfaced), versioned
`qgraph_hybrid_confidence.v1`. Raw and fused scores stay in per-item provenance for debugging and eval.

## Building & activating a semantic collection

The semantic collection is built and activated with the vector indexing CLI (see
[embeddings.md](embeddings.md) for the provider):

```bash
python -m src.search.vector_indexing.cli build            # build + validate (does not activate)
python -m src.search.vector_indexing.cli build --activate  # activate if validation passes
python -m src.search.vector_indexing.cli activate <collection>   # atomic alias swap
python -m src.search.vector_indexing.cli status            # active collection + compatibility
```

Activation is an atomic alias repoint (`qgraph-ayah-semantic-active`), and rollback is repointing the
alias at the previous collection — no application restart or env edit. Collections are immutable; each
carries an immutable JSON profile sidecar under `QGRAPH_AI_SEMANTIC_INDEX_PROFILES_DIR`.

> **Snapshot export/import** (build locally with the production key, ship the Qdrant snapshot + sidecar,
> restore in production) is a planned follow-on tied to the production embedding build — it is not yet
> implemented. Until then, a production collection is built where its provider key lives.

## Readiness

`GET /v1/search/readiness` proves the **configured** policy can serve (it does not make a paid embedding
call):

- `lexical_v1` — alias resolves to one index, build profile compatible, smoke query returns a hit.
- `hybrid_v1` — all of the above **plus**: the embedding provider is configured, Qdrant is reachable,
  the semantic alias resolves to exactly one non-empty collection, its sidecar profile matches the
  runtime provider/model/dimensions, the live Qdrant collection config (vector size/distance/name)
  matches the profile, and the active lexical and semantic indexes describe the **same corpus**
  (`hybrid_corpus_compatible`). Any failure returns `503` with the failing check named.

`GET /health` stays lightweight and never touches a backend.

## Failure reasons

Operational failures map to `503`; the stable `reason` identifies the cause, e.g.
`qdrant_unavailable`, `embedding_provider_unavailable`, `embedding_provider_not_configured`,
`semantic_alias_invalid`, `semantic_profile_missing`, `semantic_collection_config_mismatch`,
`hybrid_corpus_mismatch`.

## Evaluation

The cross-lingual semantic eval set (`src/search/eval/semantic_eval_set.py`) measures the design's core
bet: a query in one language surfacing the relevant Arabic ayah from one shared vector space. Its cases
start `PENDING` and are promoted to `CONFIRMED` only after a collection is built with the production
provider and the expected ayat are human-reviewed — see the activation gate in [embeddings.md](embeddings.md).
