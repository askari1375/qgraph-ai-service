# AI Backend Overview

## Purpose

`qgraph-ai-service` is the bootstrap AI backend for QGraph.
Its current job is to provide stable AI-service contracts, project-native
prepared artifacts, and the foundation for OpenSearch-backed Quran retrieval
while keeping Django-facing response shapes stable.

## Current Scope

- FastAPI request-response service
- Search planning endpoint: `POST /v1/search/plan`
- Search execution endpoint: `POST /v1/search/execute` (synchronous)
- Search readiness endpoint: `GET /v1/search/readiness`
- Search async job endpoints (`POST /v1/search/jobs`, `GET /v1/search/jobs/{job_id}`,
  `GET /v1/search/jobs/{job_id}/result`): present as a seam but **not implemented** — they
  return `501` (async is reserved for a future LLM/RAG path)
- Django corpus snapshot pull client for
  `GET /api/internal/ai/corpus-snapshots/quran`
- Versioned Arabic, Persian, and English normalization profiles
- Ayah and translation search document builder with stable document IDs
- OpenSearch lexical/BM25 backend, served through an alias and configured via
  `QGRAPH_AI_OPENSEARCH_URL` + `QGRAPH_AI_OPENSEARCH_ALIAS` (no mock mode)
- Segmentation generation endpoint: `POST /v1/segmentation/generate`
- Segmentation artifact manifest endpoint:
  `GET /v1/segmentation/artifacts/{artifact_id}/manifest`
- Segmentation artifact per-surah endpoint:
  `GET /v1/segmentation/artifacts/{artifact_id}/surahs/{surah_number}`
- Health endpoint: `GET /health`

Prepared segmentation artifacts are file-backed by default under
`data/segmentation_artifacts`, or the path set by
`QGRAPH_AI_SEGMENTATION_ARTIFACTS_DIR`. The public transfer payload uses
surah-local `start_ayah_number` and `end_ayah_number` fields.

Search execution is OpenSearch-only and synchronous. `/v1/search/execute` queries
the serving alias and returns the v1 response envelope with result items carrying
corpus snapshot, normalization profile, analysis profile, and lexical score
provenance read from the index profile. A missing/misconfigured cluster, an empty
alias, or a stale index profile returns a clear service error — the service never
falls back to fake results.

## Project Structure (Bootstrap)

```text
src/
  api/          # HTTP routes and request/response schemas
  services/     # request orchestration, normalization, corpus clients, retrieval
  stores/       # placeholder data-access modules (vector/graph)
  workflows/    # placeholder workflow modules
```

## Non-Goals In This Phase

- OpenAI embeddings
- Qdrant semantic retrieval
- Neo4j graph retrieval
- online LLM answer generation
- async job orchestration inside this service
- model quality optimization
- complex architecture or abstractions
