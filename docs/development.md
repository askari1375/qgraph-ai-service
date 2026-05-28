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

```bash
uv run uvicorn src.main:app --reload --port 8001
```

Service will be available at `http://127.0.0.1:8001`.

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

Live-reload behavior:

- Compose bind-mounts the repository into the container (`.:/app`)
- Uvicorn runs with `--reload --reload-dir /app/src`
- Changes under `src/` are picked up without rebuilding the image
- Rebuild is only needed when dependencies change (`pyproject.toml` / `uv.lock`)

## Django Connectivity Notes

- If Django runs on your host machine: use `http://127.0.0.1:8001`
- If Django runs in Docker on the same Compose network: use `http://ai-backend:8001`

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
