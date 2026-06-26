"""Async search jobs are intentionally not implemented.

The AI service serves synchronous, retrieval-only search through
``POST /v1/search/execute``. Durable async job orchestration is reserved for a
future LLM/RAG answer-generation path; until that exists the job endpoints fail
loudly with a clear error instead of simulating queued/running progress over
in-memory state. The routes are kept as a seam so the contract is documented and
re-implementing real async later is additive.
"""

from __future__ import annotations

ASYNC_SEARCH_NOT_IMPLEMENTED_REASON = "async_search_not_implemented"
ASYNC_SEARCH_NOT_IMPLEMENTED_MESSAGE = (
    "Async search jobs are not implemented; use POST /v1/search/execute for synchronous search."
)
