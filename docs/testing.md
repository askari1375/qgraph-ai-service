# Testing Guide

## Scope

Bootstrap tests focus on:

- endpoint wiring
- schema shape correctness
- structural constraints (`blocks[].order`, `items[].rank`)
- corpus snapshot client payload validation
- Arabic, Persian, and English normalization behavior
- OpenSearch request construction through fake adapters
- retrieval-mode errors for missing or stale lexical indexes

Tests do not evaluate real AI quality and do not require live Django or
OpenSearch services in this phase.

## Run Tests

```bash
uv run pytest
```

Or with the existing virtual environment:

```bash
.venv/bin/pytest
```

## Current Test Layout

```text
tests/
  conftest.py
  api/
    test_health.py
    test_search_plan.py
    test_search_execute.py
    test_segmentation_artifacts.py
    test_segmentation_generate.py
  services/
    test_corpus_client.py
    test_opensearch_lexical.py
    test_planning.py
    test_search_documents.py
    test_search_normalization.py
    test_search_service_retrieval.py
```
