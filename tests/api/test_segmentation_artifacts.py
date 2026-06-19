import json
from pathlib import Path
from typing import Any

import pytest

from src.api.segmentation import get_segmentation_artifact_loader
from src.services.segmentation_artifacts import FileSegmentationArtifactLoader

ARTIFACT_ID = "quran-v1-segmentation"


@pytest.fixture
def segmentation_artifact_dir(tmp_path, app):
    artifacts_dir = tmp_path / "segmentation_artifacts"
    write_artifact(
        artifacts_dir=artifacts_dir,
        artifact_id=ARTIFACT_ID,
        manifest=build_manifest(),
        surahs={1: build_surah_payload()},
    )

    app.dependency_overrides[get_segmentation_artifact_loader] = lambda: (
        FileSegmentationArtifactLoader(artifacts_dir)
    )
    yield artifacts_dir
    app.dependency_overrides.pop(get_segmentation_artifact_loader, None)


def test_segmentation_artifact_manifest_endpoint_returns_required_fields(
    client,
    segmentation_artifact_dir,
):
    response = client.get(f"/v1/segmentation/artifacts/{ARTIFACT_ID}/manifest")
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {
        "artifact_id",
        "schema_version",
        "title",
        "description",
        "model_name",
        "model_version",
        "params",
        "produced_at",
        "surahs",
    }
    assert payload["artifact_id"] == ARTIFACT_ID
    assert payload["schema_version"] == "segmentation-artifact-v1"
    assert payload["surahs"] == [{"surah_number": 1, "segment_count": 1}]


def test_segmentation_artifact_surah_endpoint_returns_project_native_shape(
    client,
    segmentation_artifact_dir,
):
    response = client.get(f"/v1/segmentation/artifacts/{ARTIFACT_ID}/surahs/1")
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {
        "artifact_id",
        "external_id",
        "surah_number",
        "model_name",
        "model_version",
        "params",
        "produced_at",
        "segments",
    }
    assert payload["artifact_id"] == ARTIFACT_ID
    assert payload["surah_number"] == 1

    segment = payload["segments"][0]
    assert {"start_ayah_number", "end_ayah_number", "title", "summary", "tags"} <= set(segment)
    assert "start_ayah" not in segment
    assert "end_ayah" not in segment
    assert segment["start_ayah_number"] == 1
    assert segment["end_ayah_number"] == 7
    assert segment["tags"] == [
        {
            "name": "guidance",
            "color": "#22c55e",
            "description": "Guidance-related theme.",
        }
    ]


def test_segmentation_artifact_manifest_endpoint_returns_404_for_unknown_artifact(
    client,
    segmentation_artifact_dir,
):
    response = client.get("/v1/segmentation/artifacts/missing-artifact/manifest")
    assert response.status_code == 404

    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["detail"]["message"] == "Segmentation artifact not found"
    assert payload["detail"]["artifact_id"] == "missing-artifact"


def test_segmentation_artifact_surah_endpoint_returns_404_for_unknown_surah(
    client,
    segmentation_artifact_dir,
):
    response = client.get(f"/v1/segmentation/artifacts/{ARTIFACT_ID}/surahs/2")
    assert response.status_code == 404

    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["detail"]["message"] == "Segmentation artifact surah not found"
    assert payload["detail"]["artifact_id"] == ARTIFACT_ID
    assert payload["detail"]["surah_number"] == 2


def test_segmentation_artifact_endpoint_rejects_invalid_artifact_data_clearly(
    client,
    segmentation_artifact_dir,
):
    artifact_id = "invalid-artifact"
    invalid_payload = build_surah_payload(artifact_id=artifact_id)
    invalid_payload["segments"][0]["tags"][0]["color"] = "green"
    write_artifact(
        artifacts_dir=segmentation_artifact_dir,
        artifact_id=artifact_id,
        manifest=build_manifest(artifact_id=artifact_id),
        surahs={1: invalid_payload},
    )

    response = client.get(f"/v1/segmentation/artifacts/{artifact_id}/surahs/1")
    assert response.status_code == 500

    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["detail"]["message"] == "Stored segmentation artifact is invalid"
    assert payload["detail"]["artifact_id"] == artifact_id
    assert payload["detail"]["source"] == "surah"
    assert payload["detail"]["surah_number"] == 1
    assert payload["detail"]["errors"]
    assert "tag color must be a valid hex color" in payload["detail"]["errors"][0]["msg"]


def write_artifact(
    artifacts_dir: Path,
    artifact_id: str,
    manifest: dict[str, Any],
    surahs: dict[int, dict[str, Any]],
) -> None:
    artifact_dir = artifacts_dir / artifact_id
    surahs_dir = artifact_dir / "surahs"
    surahs_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for surah_number, payload in surahs.items():
        (surahs_dir / f"{surah_number}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def build_manifest(artifact_id: str = ARTIFACT_ID) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "schema_version": "segmentation-artifact-v1",
        "title": "Quran V1 Segmentation",
        "description": "Prepared full-Quran segmentation artifact.",
        "model_name": "segmentation-pipeline",
        "model_version": "2026-06-19",
        "params": {},
        "produced_at": "2026-06-19T00:00:00Z",
        "surahs": [{"surah_number": 1, "segment_count": 1}],
    }


def build_surah_payload(artifact_id: str = ARTIFACT_ID) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "external_id": f"{artifact_id}:s001",
        "surah_number": 1,
        "model_name": "segmentation-pipeline",
        "model_version": "2026-06-19",
        "params": {"artifact_id": artifact_id},
        "produced_at": "2026-06-19T00:00:00Z",
        "segments": [
            {
                "start_ayah_number": 1,
                "end_ayah_number": 7,
                "title": "Opening invocation and guidance",
                "summary": "Surah al-Fatihah as praise, mercy, worship, and guidance.",
                "tags": [
                    {
                        "name": "guidance",
                        "color": "#22c55e",
                        "description": "Guidance-related theme.",
                    }
                ],
            }
        ],
    }
