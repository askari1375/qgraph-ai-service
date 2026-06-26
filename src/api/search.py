from fastapi import APIRouter, HTTPException, status

from src.api.schemas.search import (
    SearchExecuteRequest,
    SearchExecuteResponse,
    SearchJobCreateRequest,
    SearchPlanRequest,
    SearchPlanResponse,
)
from src.services.search_jobs import (
    ASYNC_SEARCH_NOT_IMPLEMENTED_MESSAGE,
    ASYNC_SEARCH_NOT_IMPLEMENTED_REASON,
)
from src.services.planning import build_planning_response
from src.services.search_service import SearchRetrievalError, build_search_execute_response

router = APIRouter(prefix="/v1/search", tags=["search"])


@router.post("/plan", response_model=SearchPlanResponse)
def search_plan(payload: SearchPlanRequest) -> SearchPlanResponse:
    return build_planning_response(payload)


@router.post("/execute", response_model=SearchExecuteResponse)
def search_execute(payload: SearchExecuteRequest) -> SearchExecuteResponse:
    try:
        return build_search_execute_response(payload)
    except SearchRetrievalError as exc:
        raise _search_retrieval_http_error(exc) from exc


# Async search jobs are not implemented (the AI service serves synchronous retrieval-only
# search). The routes are kept as a seam but fail loudly rather than simulating progress.
@router.post("/jobs")
def search_job_create(payload: SearchJobCreateRequest) -> None:
    raise _async_not_implemented_http_error()


@router.get("/jobs/{job_id}")
def search_job_status(job_id: str) -> None:
    raise _async_not_implemented_http_error()


@router.get("/jobs/{job_id}/result")
def search_job_result(job_id: str) -> None:
    raise _async_not_implemented_http_error()


def _async_not_implemented_http_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "message": ASYNC_SEARCH_NOT_IMPLEMENTED_MESSAGE,
            "reason": ASYNC_SEARCH_NOT_IMPLEMENTED_REASON,
        },
    )


def _search_retrieval_http_error(exc: SearchRetrievalError) -> HTTPException:
    detail = {
        "message": exc.message,
        "reason": exc.reason,
    }
    if exc.status_code is not None:
        detail["backend_status_code"] = exc.status_code
    if exc.detail:
        detail["detail"] = exc.detail
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
