from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QGRAPH_AI_", extra="ignore")

    environment: str = "development"
    log_level: str = "info"

    service_name: str = "qgraph-ai-service"
    service_version: str = "0.1.0"
    render_schema_version: str = "v1"

    search_backend_name: str = "qgraph-ai-search"
    search_backend_version: str = "2026-04-01"

    segmentation_model_name: str = "segmentation-pipeline"
    segmentation_model_version: str = "2026-04-01"
    segmentation_artifacts_dir: Path = Path("data/segmentation_artifacts")


@lru_cache
def get_settings() -> Settings:
    return Settings()
