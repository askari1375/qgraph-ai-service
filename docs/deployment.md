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
- Host port pinned to `127.0.0.1` for Django-to-FastAPI calls on the same VPS

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
| `QGRAPH_AI_HOST_PORT` | Compose host port, default `8001` |

The current bootstrap service does not read LLM provider keys, CORS settings, timeout settings, or Django callback URLs. Add those only when the code supports them.

`QGRAPH_AI_BIND_ADDRESS` is intentionally not supported by the production Compose file. The host bind address is pinned to `127.0.0.1` to avoid accidentally publishing the AI service on `0.0.0.0`.

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

Production services use `restart: unless-stopped`. Docker restarts them after a VPS/EC2 reboot or transient container crash; if you manually stop a container, it stays stopped until you start it again.

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

If you change `QGRAPH_AI_HOST_PORT`, adjust the healthcheck URL you run from the host.

## Exposure Model

Production traffic should flow:

```text
Frontend -> Django backend -> FastAPI AI service
```

Django remains responsible for authentication, authorization, subscription checks, rate limits, and deciding which AI behavior/model should be used.

The production Compose file publishes only `127.0.0.1:${QGRAPH_AI_HOST_PORT:-8001}:8001` on the VPS. The Uvicorn process still listens on `0.0.0.0` inside the container so Docker can route traffic to it, but Docker exposes that port only on host loopback.

Do not configure public DNS or a public reverse proxy route such as `ai.qgraph.org` to this service. If `ai.qgraph.org` currently points at the VPS or proxies to port `8001`, remove that route or make it return a closed/default response. Public HTTPS should terminate at Django, not at this FastAPI container.

For a same-host production deployment, set Django's AI backend URL to the private loopback URL:

```env
AI_BACKEND_URL=http://127.0.0.1:8001
SEARCH_AI_BACKEND_URL=http://127.0.0.1:8001
```

If Django and this service run as containers on the same Docker network, use the internal service URL instead, for example `http://ai-backend:8001`, and avoid publishing the AI service to the public internet.

This change does not add authentication, TLS, rate limiting, Caddy, Nginx, Traefik, or cloud firewall rules to the AI service. Those controls belong on the public Django perimeter for this architecture.

An optional internal shared-secret header can be added later as defense-in-depth between Django and FastAPI, but it must not be the primary protection. The primary protection is that the FastAPI port is not publicly reachable.

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
