"""Render ``RetrievalCandidate``s into Django's ``SearchExecuteResponse`` blocks/items.

This is the *only* module in the retrieval domain that knows Django's block contract
(``src/api/schemas/search.py``) — everything upstream speaks ``RetrievalCandidate``. Keeping the
contract knowledge here means the pipeline and retrievers never couple to the presentation envelope,
and the envelope can change without touching retrieval.

The first implementation bridges real results through the supported **markdown** block so the
current UI renders end-to-end; a typed verse block follows later. Collapse/dedup by
``canonical_content_id`` is applied here when ``QueryContext.collapse`` is set.
"""

from __future__ import annotations

from src.api.schemas.search import SearchExecuteRequest, SearchExecuteResponse
from src.search.contracts import QueryContext, RetrievalCandidate


def build_execute_response(
    candidates: list[RetrievalCandidate],
    request: SearchExecuteRequest,
    query_context: QueryContext,
) -> SearchExecuteResponse:
    """Build the Django-facing response from ranked candidates.

    Not implemented yet: optionally collapse/dedup by ``canonical_content_id`` (when
    ``query_context.collapse``), then emit a markdown block (highlight ``<mark>`` tags stripped — the
    markdown renderer disables raw HTML; true highlighting waits for the typed verse block).
    """
    raise NotImplementedError("response_builder.build_execute_response is not implemented yet")
