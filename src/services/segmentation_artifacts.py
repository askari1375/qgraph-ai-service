import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.api.schemas.segmentation import (
    SegmentationArtifactManifest,
    SegmentationArtifactSurahPayload,
)
from src.config import Settings, get_settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class SegmentationArtifactNotFoundError(Exception):
    def __init__(self, message: str, artifact_id: str, surah_number: int | None = None):
        super().__init__(message)
        self.message = message
        self.artifact_id = artifact_id
        self.surah_number = surah_number


class SegmentationArtifactValidationError(Exception):
    def __init__(
        self,
        message: str,
        artifact_id: str,
        source: str,
        errors: list[dict[str, Any]],
        surah_number: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.artifact_id = artifact_id
        self.source = source
        self.errors = errors
        self.surah_number = surah_number


class FileSegmentationArtifactLoader:
    def __init__(self, artifacts_dir: Path | str):
        self.artifacts_dir = Path(artifacts_dir)

    def load_manifest(self, artifact_id: str) -> SegmentationArtifactManifest:
        artifact_dir = self._artifact_dir(artifact_id)
        path = artifact_dir / "manifest.json"
        if not path.is_file():
            raise SegmentationArtifactNotFoundError(
                "Segmentation artifact not found",
                artifact_id=artifact_id,
            )

        manifest = self._load_model(
            path=path,
            model_type=SegmentationArtifactManifest,
            artifact_id=artifact_id,
            source="manifest",
        )
        if manifest.artifact_id != artifact_id:
            raise SegmentationArtifactValidationError(
                "Stored segmentation artifact is invalid",
                artifact_id=artifact_id,
                source="manifest",
                errors=[
                    {
                        "loc": ["artifact_id"],
                        "msg": "manifest artifact_id must match requested artifact_id",
                    }
                ],
            )
        return manifest

    def load_surah(self, artifact_id: str, surah_number: int) -> SegmentationArtifactSurahPayload:
        if not 1 <= surah_number <= 114:
            raise SegmentationArtifactNotFoundError(
                "Segmentation artifact surah not found",
                artifact_id=artifact_id,
                surah_number=surah_number,
            )

        artifact_dir = self._artifact_dir(artifact_id)
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SegmentationArtifactNotFoundError(
                "Segmentation artifact not found",
                artifact_id=artifact_id,
            )

        path = artifact_dir / "surahs" / f"{surah_number}.json"
        if not path.is_file():
            raise SegmentationArtifactNotFoundError(
                "Segmentation artifact surah not found",
                artifact_id=artifact_id,
                surah_number=surah_number,
            )

        payload = self._load_model(
            path=path,
            model_type=SegmentationArtifactSurahPayload,
            artifact_id=artifact_id,
            source="surah",
            surah_number=surah_number,
        )
        errors = []
        if payload.artifact_id != artifact_id:
            errors.append(
                {
                    "loc": ["artifact_id"],
                    "msg": "surah payload artifact_id must match requested artifact_id",
                }
            )
        if payload.surah_number != surah_number:
            errors.append(
                {
                    "loc": ["surah_number"],
                    "msg": "surah payload surah_number must match requested surah_number",
                }
            )
        if errors:
            raise SegmentationArtifactValidationError(
                "Stored segmentation artifact is invalid",
                artifact_id=artifact_id,
                source="surah",
                errors=errors,
                surah_number=surah_number,
            )
        return payload

    def _artifact_dir(self, artifact_id: str) -> Path:
        if not artifact_id or artifact_id in {".", ".."} or Path(artifact_id).name != artifact_id:
            raise SegmentationArtifactNotFoundError(
                "Segmentation artifact not found",
                artifact_id=artifact_id,
            )
        return self.artifacts_dir / artifact_id

    def _load_model(
        self,
        path: Path,
        model_type: type[SchemaT],
        artifact_id: str,
        source: str,
        surah_number: int | None = None,
    ) -> SchemaT:
        data = self._load_json(
            path=path,
            artifact_id=artifact_id,
            source=source,
            surah_number=surah_number,
        )
        try:
            return model_type.model_validate(data)
        except ValidationError as exc:
            raise SegmentationArtifactValidationError(
                "Stored segmentation artifact is invalid",
                artifact_id=artifact_id,
                source=source,
                errors=exc.errors(
                    include_context=False,
                    include_input=False,
                    include_url=False,
                ),
                surah_number=surah_number,
            ) from exc

    def _load_json(
        self,
        path: Path,
        artifact_id: str,
        source: str,
        surah_number: int | None = None,
    ) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            raise SegmentationArtifactValidationError(
                "Stored segmentation artifact JSON is invalid",
                artifact_id=artifact_id,
                source=source,
                errors=[
                    {
                        "loc": [],
                        "msg": exc.msg,
                        "line": exc.lineno,
                        "column": exc.colno,
                    }
                ],
                surah_number=surah_number,
            ) from exc


def build_segmentation_artifact_loader(
    settings: Settings | None = None,
) -> FileSegmentationArtifactLoader:
    cfg = settings if settings is not None else get_settings()
    return FileSegmentationArtifactLoader(cfg.segmentation_artifacts_dir)
