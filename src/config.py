from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalPolicy = Literal["lexical_v1", "hybrid_v1"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QGRAPH_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "info"

    service_name: str = "qgraph-ai-service"
    service_version: str = "0.1.0"
    render_schema_version: str = "v1"

    search_backend_name: str = "qgraph-ai-search"
    search_backend_version: str = "2026-04-01"
    # Retrieval policy: "lexical_v1" (OpenSearch only) or "hybrid_v1" (OpenSearch + Qdrant + RRF).
    # A constrained Literal so an unknown/typo'd value fails fast at startup instead of silently
    # running lexical. Defaults to lexical_v1; switch to hybrid_v1 only after a semantic collection and
    # the embedding provider are built, activated, and pass readiness. Both are real backends — hybrid
    # never silently degrades to lexical when Qdrant or the provider is unavailable.
    search_retrieval_policy: RetrievalPolicy = "lexical_v1"
    search_corpus_snapshot_cache_dir: Path = Path("data/corpus_snapshots")
    # Heuristic scale for mapping an absolute BM25 score to a 0..1 confidence via
    # 1 - exp(-score / k). Larger k => more conservative confidence. Tune against
    # observed lexical score distributions.
    search_confidence_scale_k: float = 10.0

    django_internal_base_url: str = ""
    django_internal_token: str = ""
    # The only caller is the offline index builder pulling the full corpus snapshot,
    # which Django assembles in one response — generous by design, not a request-path timeout.
    django_internal_timeout_seconds: float = 120.0

    opensearch_url: str = ""
    # The serving alias the app queries; activation repoints it at a new physical index version.
    opensearch_alias: str = "qgraph-ayah-lexical-active"
    # Physical index versions are named "<prefix>-<YYYYMMDD>-<NNN>".
    opensearch_index_prefix: str = "qgraph-ayah-lexical"
    opensearch_timeout_seconds: float = 10.0
    # Basic-auth credentials for a security-enabled OpenSearch. Empty username =>
    # no auth (plain dev cluster with the security plugin disabled).
    opensearch_username: str = ""
    opensearch_password: str = ""
    # TLS verification for https OpenSearch URLs. Verify by default; set a CA
    # bundle path to verify self-signed/internal certs, or disable verification
    # for an internal-only cluster using demo self-signed certs.
    opensearch_verify_certs: bool = True
    opensearch_ca_cert_path: str = ""

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_timeout_seconds: float = 10.0
    # The serving alias the app queries; activation repoints it at a new physical collection version.
    qdrant_collection_alias: str = "qgraph-ayah-semantic-active"
    # Physical collection versions are named "<prefix>-<YYYYMMDD>-<NNN>".
    qdrant_collection_prefix: str = "qgraph-ayah-semantic"
    # The single named dense vector per collection.
    qdrant_vector_name: str = "content"
    # Immutable per-collection profile sidecars (Qdrant has no OpenSearch-style _meta). Must be
    # persistent and shared with any future replicas.
    semantic_index_profiles_dir: Path = Path("data/semantic_index_profiles")
    # Documents embedded per provider call during a semantic build (Cohere caps a batch at 96).
    embedding_document_batch_size: int = 96
    # Production embedding provider resolved by build_embedding_provider. Empty => no provider is
    # configured (the safe default), so a real semantic build fails loudly rather than guessing.
    # Production value: "openai".
    embedding_provider: str = ""
    # Provider model id and its native output dimension. Both are immutable per collection and stamped
    # into the semantic profile; changing either means building a new collection, never an in-place
    # migration. Production: "text-embedding-3-large" / 3072 (no dimension reduction in V1).
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_api_key: str = ""
    embedding_timeout_seconds: float = 30.0
    embedding_max_retries: int = 2
    # The provider's hard per-input token ceiling; the build preflight fails (with the offending
    # document id) before any paid call rather than letting the provider truncate or reject mid-build.
    # Production: text-embedding-3-large accepts 8192 tokens per input.
    embedding_max_input_tokens: int = 8192
    # Provider input price for the dry-run cost estimate only; 0 => the estimate is omitted. Production:
    # 0.13 (text-embedding-3-large is $0.13 per 1M input tokens). Never used for billing, only planning.
    embedding_usd_per_million_input_tokens: float = 0.0

    segmentation_model_name: str = "segmentation-pipeline"
    segmentation_model_version: str = "2026-04-01"
    segmentation_artifacts_dir: Path = Path("data/segmentation_artifacts")


@lru_cache
def get_settings() -> Settings:
    return Settings()
