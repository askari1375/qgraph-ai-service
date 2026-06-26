from src.api.schemas.search import (
    RequesterContext,
    SearchMode,
    SearchPlanRequest,
    SearchPlanResponse,
)
from src.config import Settings, get_settings

POLICY_LABEL = "retrieval_sync_v1"


def choose_planning_mode(requester: RequesterContext) -> SearchMode:
    """Choose the response mode for the current retrieval-only search policy.

    V1 policy: both guests and authenticated users run synchronous lexical
    retrieval. Async is reserved for future LLM/RAG answer generation, which is
    the point where requester tier will actually gate behavior. Until then the
    planner is deterministic and never selects async.
    """
    return "sync"


def build_planning_response(
    request: SearchPlanRequest,
    settings: Settings | None = None,
) -> SearchPlanResponse:
    cfg = settings if settings is not None else get_settings()
    requester = RequesterContext.from_context(request.context)
    mode = choose_planning_mode(requester)
    return SearchPlanResponse(
        mode=mode,
        policy_label=POLICY_LABEL,
        policy_snapshot={"requester": requester.model_dump()},
        routing_metadata={"mode": mode, "retrieval": "lexical_opensearch"},
        backend_name=cfg.service_name,
        backend_version=cfg.service_version,
    )
