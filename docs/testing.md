# Testing Guide

## Scope

Bootstrap tests focus on:

- endpoint wiring
- schema shape correctness
- structural constraints (`blocks[].order`, `items[].rank`)

Tests do not evaluate real AI quality in this phase.

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
    test_planning.py
```
