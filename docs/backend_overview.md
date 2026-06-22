# AI Backend Overview

## Purpose

`qgraph-ai-service` is the bootstrap AI backend for QGraph.
Its current job is to provide stable AI-service contracts, project-native
prepared artifacts, and the foundation for OpenSearch-backed Quran retrieval
while keeping Django-facing response shapes stable.

## Current Scope

- FastAPI request-response service
- Search planning endpoint: `POST /v1/search/plan`
- Search execution endpoint: `POST /v1/search/execute`
- Search async job create endpoint: `POST /v1/search/jobs`
- Search async job status endpoint: `GET /v1/search/jobs/{job_id}`
- Search async job result endpoint: `GET /v1/search/jobs/{job_id}/result`
- Django corpus snapshot pull client for
  `GET /api/internal/ai/corpus-snapshots/quran`
- Versioned Arabic, Persian, and English normalization profiles
- Ayah and translation search document builder with stable document IDs
- OpenSearch lexical/BM25 backend, enabled only when
  `QGRAPH_AI_SEARCH_LEXICAL_BACKEND_MODE=opensearch`
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

Search execution defaults to mock mode for local development and existing
Django integration tests. In `opensearch` mode, `/v1/search/execute` queries the
configured OpenSearch index and returns the same v1 response envelope with
result items carrying corpus snapshot, normalization profile, ranker profile,
and lexical score provenance. Missing or stale retrieval indexes return a clear
service error instead of silently falling back to mock results.

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
