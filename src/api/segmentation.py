from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas.segmentation import (
    SegmentationArtifactManifest,
    SegmentationArtifactSurahPayload,
    SegmentationGenerateRequest,
    SegmentationGenerateResponse,
)
from src.services.segmentation_service import build_segmentation_generate_response
from src.services.segmentation_artifacts import (
    FileSegmentationArtifactLoader,
    SegmentationArtifactNotFoundError,
    SegmentationArtifactValidationError,
    build_segmentation_artifact_loader,
)

router = APIRouter(prefix="/v1/segmentation", tags=["segmentation"])


def get_segmentation_artifact_loader() -> FileSegmentationArtifactLoader:
    return build_segmentation_artifact_loader()


@router.post("/generate", response_model=SegmentationGenerateResponse)
def segmentation_generate(
    payload: SegmentationGenerateRequest,
) -> SegmentationGenerateResponse:
    return build_segmentation_generate_response(payload)


@router.get(
    "/artifacts/{artifact_id}/manifest",
    response_model=SegmentationArtifactManifest,
)
def segmentation_artifact_manifest(
    artifact_id: str,
    loader: Annotated[
        FileSegmentationArtifactLoader,
        Depends(get_segmentation_artifact_loader),
    ],
) -> SegmentationArtifactManifest:
    try:
        return loader.load_manifest(artifact_id)
    except SegmentationArtifactNotFoundError as exc:
        raise _artifact_not_found_http_error(exc) from exc
    except SegmentationArtifactValidationError as exc:
        raise _artifact_validation_http_error(exc) from exc


@router.get(
    "/artifacts/{artifact_id}/surahs/{surah_number}",
    response_model=SegmentationArtifactSurahPayload,
)
def segmentation_artifact_surah(
    artifact_id: str,
    surah_number: int,
    loader: Annotated[
        FileSegmentationArtifactLoader,
        Depends(get_segmentation_artifact_loader),
    ],
) -> SegmentationArtifactSurahPayload:
    try:
        return loader.load_surah(artifact_id, surah_number)
    except SegmentationArtifactNotFoundError as exc:
        raise _artifact_not_found_http_error(exc) from exc
    except SegmentationArtifactValidationError as exc:
        raise _artifact_validation_http_error(exc) from exc


def _artifact_not_found_http_error(exc: SegmentationArtifactNotFoundError) -> HTTPException:
    detail = {
        "message": exc.message,
        "artifact_id": exc.artifact_id,
    }
    if exc.surah_number is not None:
        detail["surah_number"] = exc.surah_number
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _artifact_validation_http_error(exc: SegmentationArtifactValidationError) -> HTTPException:
    detail = {
        "message": exc.message,
        "artifact_id": exc.artifact_id,
        "source": exc.source,
        "errors": exc.errors,
    }
    if exc.surah_number is not None:
        detail["surah_number"] = exc.surah_number
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)
