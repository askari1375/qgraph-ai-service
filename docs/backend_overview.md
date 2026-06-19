# AI Backend Overview

## Purpose

`qgraph-ai-service` is the bootstrap AI backend for QGraph.
Its current job is to provide schema-correct responses and project-native
prepared artifacts to Django while keeping endpoint contracts stable.

## Current Scope

- FastAPI request-response service
- Search planning endpoint: `POST /v1/search/plan`
- Search execution endpoint: `POST /v1/search/execute`
- Search async job create endpoint: `POST /v1/search/jobs`
- Search async job status endpoint: `GET /v1/search/jobs/{job_id}`
- Search async job result endpoint: `GET /v1/search/jobs/{job_id}/result`
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

## Project Structure (Bootstrap)

```text
src/
  api/          # HTTP routes and request/response schemas
  services/     # request-level orchestration with dummy logic
  stores/       # placeholder data-access modules (vector/graph)
  workflows/    # placeholder workflow modules
```

## Non-Goals In This Phase

- real AI retrieval or generation
- async job orchestration inside this service
- model quality optimization
- complex architecture or abstractions
