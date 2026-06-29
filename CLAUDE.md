# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

QGraph AI Service is a FastAPI backend that receives requests from a separate Django backend, processes them (currently with dummy/placeholder logic), and returns structured responses. It is in **bootstrap phase** — the focus is on correct API contracts and wiring, not real AI logic. Django is the source of truth for data, auth, and execution lifecycle.

## Commands

```bash
# Install dependencies
uv sync --dev

# Run locally with hot-reload
uv run uvicorn src.main:app --reload --port 8001

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/api/test_search_execute.py

# Run a single test
uv run pytest tests/api/test_search_execute.py::test_search_execute_unique_block_orders -v

# Lint (auto-fix) and format
uv run ruff check --fix . && uv run ruff format .
```

Pre-commit hooks run `ruff check --fix` and `ruff format` automatically on commit.

## Architecture

**Entry point:** `src/main.py:create_app()` creates the FastAPI app, registers routers, and installs global exception handlers.

**Layer flow:** API routes → Services → Workflows → Stores

- `src/api/` — Thin route handlers. Schemas live in `src/api/schemas/` as Pydantic models with validators enforcing Django contract constraints (unique `blocks[].order`, unique `items[].rank` per block, sorted non-overlapping segments).
- `src/services/` — Request-level orchestration. Each service builds a complete response by calling into workflows.
- `src/workflows/` — Compose calls to stores. Currently minimal pass-through logic.
- `src/stores/` — Data access placeholders (vector store, graph store). Return hardcoded dummy results.

**Config:** `src/config.py` uses `pydantic-settings` with env prefix `QGRAPH_AI_`. Settings are cached via `@lru_cache`.

**Endpoints:**
- `GET /health` — Service liveness and version
- `POST /v1/search/plan` — Planning decision (random sync/async)
- `POST /v1/search/execute` — Search with structured blocks/items response
- `POST /v1/segmentation/generate` — Segmentation with provenance metadata

## Key constraints

- Response schemas must match Django's persistence expectations exactly (see `docs/bootstrap/django_backend_context.md` for contracts).
- All AI logic is placeholder/dummy. Do not implement real retrieval, model calls, or async job orchestration.
- The API layer must stay thin — orchestration belongs in services, not route handlers.
- Stores and workflows are first-class root modules, not sub-components of the API.

## Assistant scratch space (`.ai/`)

`.ai/` is local-only and gitignored — personal drafts, proposals, and phased implementation plans
(e.g. "Phase A/B", decision ids like "D3") live there and are not authoritative documentation. Never
reference `.ai/` files, paths, or contents from anything that ships with the repo: source code and
comments, docstrings, `NotImplementedError`/exception messages, `docs/`, READMEs, or commit/PR
messages. Carry the *substance* of a decision into the code on its own terms; leave the planning
vocabulary behind.

## Testing approach

Tests verify endpoint wiring and schema shape, not AI quality. Key patterns:
- `tests/conftest.py` provides `client` fixture (TestClient) and sample payloads
- API tests check response structure, field presence, and uniqueness constraints
- Service tests check allowed output values (e.g., planning modes)

## Ruff config

Line length: 100. Configured in `pyproject.toml`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
