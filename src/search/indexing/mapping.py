"""OpenSearch index settings, analyzers, and mappings — the "index strategy".

The primary ``content_ar``/``content_fa`` fields use custom **normalize-don't-stem** analyzers
(normalize alef/yeh/teh-marbuta, strip harakat/tatweel and the superscript/dagger alif U+0670, fold
ZWNJ for Persian) so load-bearing Quranic particles (لا/ما/إن) survive while the bare modern spelling
(الرحمن) matches the Quranic orthography (الرحمٰن). The built-in ``arabic``/``persian`` stemmed analyzers are kept
only on the lower-value ``.stemmed`` sub-fields for opt-in recall. English keeps the built-in stemmed
``english`` analyzer as primary (better translation recall) with an ``.exact`` precision sub-field —
an intentional, documented asymmetry.

The index ``_meta`` carries the build profile, including ``analysis_profile_version`` (the
index-strategy version): bumping it is the signal to rebuild + re-validate + swap the alias.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.search.indexing.documents import DOCUMENT_SCHEMA_VERSION, SearchIndexDocument
from src.search.indexing.normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
)

ANALYSIS_PROFILE_VERSION = "2026-06-25.v1"
LEXICAL_INDEX_PROFILE_SCHEMA_VERSION = "qgraph_lexical_index_profile.v2"
OPEN_SEARCH_BACKEND_NAME = "open_search"

# ZWNJ -> space, so Persian compound words split on the half-space the same way the Python normalizer
# folds them.
_ZWNJ_TO_SPACE_MAPPING = "‌=> "

# Strip the superscript/dagger alif (U+0670). It is a combining mark like the harakat the token
# filters already drop, but ``arabic_normalization`` leaves it in place; removing it pre-tokenization
# folds the Quranic spelling (الرحمٰن) onto the bare modern one (الرحمن), matching the Python
# normalizer. The ``.stemmed`` recall sub-fields still fold it onto a full alif, so the full-alif
# spelling (الرحمان) keeps matching too.
_DAGGER_ALIF_TO_EMPTY = "ٰ=>"


def build_index_settings(profile_meta: dict[str, Any]) -> dict[str, Any]:
    """Return the full OpenSearch ``{settings, mappings}`` body for a new index version.

    ``profile_meta`` is embedded under ``mappings._meta.qgraph_index_profile`` and is the only place
    build-level provenance lives (per-document metadata stays minimal).
    """
    return {
        "settings": {
            "index": {"number_of_shards": 1, "number_of_replicas": 0},
            "analysis": {
                "char_filter": {
                    "zwnj_to_space": {"type": "mapping", "mappings": [_ZWNJ_TO_SPACE_MAPPING]},
                    "dagger_alif_strip": {
                        "type": "mapping",
                        "mappings": [_DAGGER_ALIF_TO_EMPTY],
                    },
                },
                "analyzer": {
                    "arabic_normalized": {
                        "type": "custom",
                        "char_filter": ["dagger_alif_strip"],
                        "tokenizer": "standard",
                        "filter": ["lowercase", "decimal_digit", "arabic_normalization"],
                    },
                    "persian_normalized": {
                        "type": "custom",
                        "char_filter": ["dagger_alif_strip", "zwnj_to_space"],
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "decimal_digit",
                            "arabic_normalization",
                            "persian_normalization",
                        ],
                    },
                    "english_exact": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                },
            },
        },
        "mappings": {
            "_meta": {"qgraph_index_profile": profile_meta},
            "dynamic": "strict",
            "properties": {
                "id": {"type": "keyword"},
                "canonical_content_id": {"type": "keyword"},
                "content_ar": {
                    "type": "text",
                    "analyzer": "arabic_normalized",
                    "fields": {
                        "stemmed": {"type": "text", "analyzer": "arabic"},
                        "keyword": {"type": "keyword"},
                    },
                },
                "content_fa": {
                    "type": "text",
                    "analyzer": "persian_normalized",
                    "fields": {
                        "stemmed": {"type": "text", "analyzer": "persian"},
                        "keyword": {"type": "keyword"},
                    },
                },
                "content_en": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {
                        "exact": {"type": "text", "analyzer": "english_exact"},
                        "keyword": {"type": "keyword"},
                    },
                },
                "content_general": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "metadata": {
                    "properties": {
                        "surah_number": {"type": "integer"},
                        "ayah_number": {"type": "integer"},
                        "ayah_global_number": {"type": "integer"},
                        "language_code": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "source_name": {"type": "keyword"},
                        "content_type": {"type": "keyword"},
                    }
                },
            },
        },
    }


def build_index_profile(
    *,
    index_name: str,
    snapshot: QuranCorpusSnapshot,
    documents: list[SearchIndexDocument],
) -> dict[str, Any]:
    """Assemble the build-level ``_meta`` profile for an index version.

    Carries the code-constant compatibility versions (document schema / normalization / analysis)
    that a running service checks against, plus the snapshot provenance and build summary.
    """
    if not documents:
        raise ValueError("documents must not be empty")
    return {
        "index_id": index_name,
        "schema_version": LEXICAL_INDEX_PROFILE_SCHEMA_VERSION,
        "backend": OPEN_SEARCH_BACKEND_NAME,
        "corpus_snapshot_id": snapshot.corpus_snapshot_id,
        "corpus_snapshot_hash": snapshot.corpus_snapshot_hash,
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_profile_version": NORMALIZATION_PROFILE_VERSION,
        "analysis_profile_version": ANALYSIS_PROFILE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "included_languages": sorted({doc.metadata.language_code for doc in documents}),
        "source_ids": sorted({doc.metadata.source_id for doc in documents}),
        "content_types": sorted({doc.metadata.content_type.value for doc in documents}),
    }
