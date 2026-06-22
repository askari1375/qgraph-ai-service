from fastapi import APIRouter, HTTPException, status

from src.api.schemas.search import (
    SearchExecuteRequest,
    SearchExecuteResponse,
    SearchJobCreateRequest,
    SearchJobCreateResponse,
    SearchJobStatusResponse,
    SearchPlanRequest,
    SearchPlanResponse,
)
from src.services.search_jobs import (
    create_search_job,
    get_search_job_result,
    get_search_job_status,
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


@router.post(
    "/jobs",
    response_model=SearchJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def search_job_create(payload: SearchJobCreateRequest) -> SearchJobCreateResponse:
    try:
        return create_search_job(payload)
    except SearchRetrievalError as exc:
        raise _search_retrieval_http_error(exc) from exc


@router.get("/jobs/{job_id}", response_model=SearchJobStatusResponse)
def search_job_status(job_id: str) -> SearchJobStatusResponse:
    job_status = get_search_job_status(job_id)
    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Search job not found", "job_id": job_id},
        )
    return job_status


@router.get("/jobs/{job_id}/result", response_model=SearchExecuteResponse)
def search_job_result(job_id: str) -> SearchExecuteResponse:
    lookup = get_search_job_result(job_id)
    if lookup.status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Search job not found", "job_id": job_id},
        )
    if lookup.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Search job result not ready",
                "job_id": job_id,
                "status": lookup.status.status,
            },
        )
    return lookup.result


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
