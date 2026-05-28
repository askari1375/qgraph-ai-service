# Production Deployment

This guide describes the portable Docker setup for `qgraph-ai-service`. It is not specific to Lightsail, EC2, or any one VPS provider.

## What This Adds

- A production Docker image based on `python:3.12-slim-bookworm`
- Pinned `uv` install: `0.11.15`
- Production dependencies from `uv.lock`
- Non-root runtime user
- Container port `8001`
- Dockerfile and Compose healthchecks for `GET /health`
- Production Compose without bind mounts or reload

The existing `docker-compose.yml` remains the local development workflow with bind mounts and Uvicorn reload.

## Environment File

Create a host-specific env file from the committed example:

```bash
cp .env.example .env.prod
```

Review `.env.prod` before deploying. Do not commit it.

Current app settings use the `QGRAPH_AI_` prefix and have safe defaults, but production should still set these values explicitly:

| Variable | Purpose |
| --- | --- |
| `QGRAPH_AI_ENVIRONMENT` | Runtime environment label, usually `production` |
| `QGRAPH_AI_LOG_LEVEL` | App log level setting reserved for service logging |
| `QGRAPH_AI_SERVICE_NAME` | FastAPI service title and health response name |
| `QGRAPH_AI_SERVICE_VERSION` | FastAPI version and health response version |
| `QGRAPH_AI_RENDER_SCHEMA_VERSION` | Search response render schema version |
| `QGRAPH_AI_SEARCH_BACKEND_NAME` | Search planning backend metadata |
| `QGRAPH_AI_SEARCH_BACKEND_VERSION` | Search planning backend metadata |
| `QGRAPH_AI_SEGMENTATION_MODEL_NAME` | Segmentation response model metadata |
| `QGRAPH_AI_SEGMENTATION_MODEL_VERSION` | Segmentation response model metadata |
| `QGRAPH_AI_BIND_ADDRESS` | Compose host bind address, default `127.0.0.1` |
| `QGRAPH_AI_HOST_PORT` | Compose host port, default `8001` |

The current bootstrap service does not read LLM provider keys, CORS settings, timeout settings, or Django callback URLs. Add those only when the code supports them.

## Build The Image

```bash
docker build -t qgraph-ai-service:prod .
```

The Dockerfile copies `pyproject.toml` and `uv.lock` before source code so dependency layers stay cached when only application code changes.

## Run With Production Compose

Validate the Compose file:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml config
```

Build and start:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

Check health:

```bash
curl http://127.0.0.1:8001/health
```

View logs:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f ai-backend
```

Stop:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml down
```

If you change `QGRAPH_AI_BIND_ADDRESS` or `QGRAPH_AI_HOST_PORT`, adjust the healthcheck URL you run from the host.

## Exposure Model

The production Compose file publishes `127.0.0.1:8001` by default. This is intended for a same-host reverse proxy or another trusted local integration to publish HTTPS publicly.

The Uvicorn command uses `--proxy-headers --forwarded-allow-ips="*"` so FastAPI receives forwarded client and scheme headers correctly behind a reverse proxy. The wildcard is appropriate only when the container is reachable solely from a trusted proxy or Docker network. If the container is exposed directly to untrusted clients, replace it with specific trusted proxy IPs or networks.

This change does not add authentication, TLS, rate limiting, Caddy, Nginx, Traefik, or cloud firewall rules. Add those at the perimeter when the service becomes publicly reachable.

## Healthcheck

The lightweight endpoint is:

```text
GET /health
```

It does not call LLM providers, Django, databases, vector stores, or other external services. Docker healthchecks use Python standard library:

```bash
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')"
```

## Scaling

Run one Uvicorn worker for now. The production command relies on Uvicorn's default of one worker.

Search jobs currently live in process memory in `src/services/search_jobs.py`. Increasing Uvicorn workers, running multiple containers, or scaling replicas would split that state across processes. Before scaling horizontally or increasing workers, move search job state to Redis, a database, or another durable shared store.

## Local vs Production

| Workflow | File | Behavior |
| --- | --- | --- |
| Development | `docker-compose.yml` | Bind-mounts the repo and runs Uvicorn with reload |
| Production | `docker-compose.prod.yml` | Builds an immutable image, no bind mounts, no reload, restart policy enabled |

Instant code updates through bind mounts are a development feature. Production changes should be deployed by rebuilding the image and restarting the service.

## VPS Commands

From the repository directory on the server:

```bash
cp .env.example .env.prod
```

Edit `.env.prod`, then run:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml config
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
curl http://127.0.0.1:8001/health
```

To deploy later code changes:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
docker compose --env-file .env.prod -f docker-compose.prod.yml ps
```
