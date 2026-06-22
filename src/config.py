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
    search_lexical_backend_mode: str = "mock"
    search_corpus_snapshot_cache_dir: Path = Path("data/corpus_snapshots")
    search_active_corpus_snapshot_id: str = ""
    search_active_corpus_snapshot_hash: str = ""
    search_ranker_profile_id: str = "lexical_bm25_v1"

    django_internal_base_url: str = ""
    django_internal_token: str = ""
    django_internal_timeout_seconds: float = 10.0

    opensearch_url: str = ""
    opensearch_index_name: str = "qgraph-ayah-lexical-v1"
    opensearch_timeout_seconds: float = 10.0

    segmentation_model_name: str = "segmentation-pipeline"
    segmentation_model_version: str = "2026-04-01"
    segmentation_artifacts_dir: Path = Path("data/segmentation_artifacts")


@lru_cache
def get_settings() -> Settings:
    return Settings()
