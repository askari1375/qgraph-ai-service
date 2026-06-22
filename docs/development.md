# Development and Run Guide

For production deployment, see [Deployment Guide](deployment.md).

## Prerequisites

- Python `3.12+`
- `uv` installed
- Docker + Docker Compose (for containerized local run)

## Install Dependencies

```bash
uv sync --dev
```

## Run the API Locally (Native)

Local settings are read from `.env` when that file exists. Environment variables
set in the shell still take precedence.

```bash
uv run uvicorn src.main:app --reload --port 8001
```

Service will be available at `http://127.0.0.1:8001`.

## Continuous Integration

GitHub Actions runs `.github/workflows/ci.yml` on pushes and pull requests. CI uses
Python `3.12`, uv `0.11.15`, and the checked-in `uv.lock`.

The workflow mirrors these local checks:

```bash
uv sync --locked --dev
uv run ruff format --check .
uv run ruff check .
uv run pytest
DOCKER_BUILDKIT=1 docker build --tag qgraph-ai-service:ci .
```

No type-check step runs yet because the project does not currently configure a
type checker such as mypy, pyright, or ty. The Docker step only builds the image;
it does not publish or deploy anything.

## Run the API with Docker Compose (Recommended for Local Integration)

Build and start:

```bash
docker compose up --build
```

After the first build:

```bash
docker compose up
```

Service is exposed on `http://127.0.0.1:8001`.

The local Compose file also binds the host port to `127.0.0.1`, so it is not
published on all host interfaces during development.

Live-reload behavior:

- Compose bind-mounts the repository into the container (`.:/app`)
- Uvicorn runs with `--reload --reload-dir /app/src`
- Changes under `src/` are picked up without rebuilding the image
- Rebuild is only needed when dependencies change (`pyproject.toml` / `uv.lock`)

## Run Local OpenSearch For Retrieval Debugging

OpenSearch is available in the local Compose file behind the optional `search`
profile. Start it when you want to debug live lexical retrieval:

```bash
docker compose --profile search up -d opensearch
```

Wait for it to become healthy, then check it from the host:

```bash
curl http://127.0.0.1:9200
```

Host-run scripts should use:

```env
QGRAPH_AI_OPENSEARCH_URL=http://127.0.0.1:9200
```

The AI container uses Docker DNS instead:

```env
QGRAPH_AI_OPENSEARCH_URL=http://opensearch:9200
```

The local Compose file sets that container value automatically. OpenSearch is
not started by plain `docker compose up`; use the `search` profile so normal API
development does not always start a heavier search service.

## Django Connectivity Notes

- If Django runs on your host machine: use `http://127.0.0.1:8001`
- If Django runs in Docker on the same Compose network: use `http://ai-backend:8001`

## Search Retrieval Foundation

Search execution defaults to mock mode:

```env
QGRAPH_AI_SEARCH_LEXICAL_BACKEND_MODE=mock
```

Use OpenSearch retrieval only after a Quran corpus snapshot has been pulled from
Django, converted into search documents, indexed, and declared active:

```env
QGRAPH_AI_SEARCH_LEXICAL_BACKEND_MODE=opensearch
QGRAPH_AI_DJANGO_INTERNAL_BASE_URL=http://web:8000
QGRAPH_AI_DJANGO_INTERNAL_TOKEN=<shared-internal-token>
QGRAPH_AI_OPENSEARCH_URL=http://opensearch:9200
QGRAPH_AI_OPENSEARCH_INDEX_NAME=qgraph-ayah-lexical-v1
QGRAPH_AI_SEARCH_ACTIVE_CORPUS_SNAPSHOT_ID=<snapshot-id>
QGRAPH_AI_SEARCH_ACTIVE_CORPUS_SNAPSHOT_HASH=<snapshot-hash>
```

The Django corpus snapshot export expected by the AI service is:

```text
GET /api/internal/ai/corpus-snapshots/quran
```

Optional query parameters:

```text
translation_languages=en,fa
surah_numbers=1,2
```

Required header:

```text
X-QGraph-Internal-Token: <shared-internal-token>
```

The service-side indexing path is deliberately narrow:

```text
DjangoCorpusClient -> build_search_documents -> OpenSearchLexicalBackend.index_documents
```

OpenSearch indexing uses chunked `_bulk` requests rather than one request for
the entire corpus. The default batch limits are 1,000 documents and 8 MiB per
request. The local full-corpus indexing script exposes
`--bulk-batch-document-count` and `--bulk-batch-max-bytes` for debugging or
tuning.

Documents use stable IDs:

```text
ayah:{surah_number}:{ayah_number}:ar
ayah:{surah_number}:{ayah_number}:translation:{source_id}
```

Every document carries the corpus snapshot id/hash, document schema version,
normalization profile id/version, surah/ayah metadata, language code, and source
id. Arabic, Persian, and English normalization is versioned so an index can be
rejected when it was built with stale text processing.

Tests use fake adapters and do not require a running OpenSearch server. A real
OpenSearch node is needed only for manual retrieval smoke checks or production
retrieval mode. If `opensearch` mode is enabled and the URL, index, or active
profile is missing or stale, `/v1/search/execute` returns a service error rather
than mock results.

## Prepared Segmentation Artifacts

The artifact endpoints read reviewed JSON files from
`data/segmentation_artifacts` by default. Override this with
`QGRAPH_AI_SEGMENTATION_ARTIFACTS_DIR` when artifacts live somewhere else.

Expected local layout:

```text
data/segmentation_artifacts/
  {artifact_id}/
    manifest.json
    surahs/
      {surah_number}.json
```

The per-surah payload uses `start_ayah_number` and `end_ayah_number`, which are
ayah numbers within the surah.

## Quick Smoke Checks

```bash
curl http://127.0.0.1:8001/health
```

```bash
curl -X POST http://127.0.0.1:8001/v1/search/plan \
  -H "Content-Type: application/json" \
  -d '{"query":"verses about patience","filters":{},"output_preferences":{}}'
```

```bash
curl -X POST http://127.0.0.1:8001/v1/search/execute \
  -H "Content-Type: application/json" \
  -d '{"query":"verses about patience","filters":{},"output_preferences":{},"context":{}}'
```

```bash
curl -X POST http://127.0.0.1:8001/v1/search/jobs \
  -H "Content-Type: application/json" \
  -d '{"query":"verses about patience","filters":{},"output_preferences":{},"context":{},"idempotency_key":"search-exec-1","client_ref":{"query_id":1,"execution_id":1}}'
```

```bash
curl http://127.0.0.1:8001/v1/search/jobs/<job_id>
```

```bash
curl http://127.0.0.1:8001/v1/search/jobs/<job_id>/result
```

```bash
curl -X POST http://127.0.0.1:8001/v1/segmentation/generate \
  -H "Content-Type: application/json" \
  -d '{"surah_id":2,"ayahs":[],"options":{},"context":{}}'
```

```bash
curl http://127.0.0.1:8001/v1/segmentation/artifacts/<artifact_id>/manifest
```

```bash
curl http://127.0.0.1:8001/v1/segmentation/artifacts/<artifact_id>/surahs/1
```
