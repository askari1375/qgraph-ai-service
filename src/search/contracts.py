"""Retrieval contracts and the canonical cross-backend content vocabulary.

This module is the *stable* core of the retrieval domain. Everything here is a type contract or a
pure constant/helper — there is no behavior to defer, so (unlike the rest of the retrieval skeleton)
it is fully implemented. Later code and future backends (Qdrant, Neo4j) plug into these exact shapes.

Four contracts (the "expensive to retrofit" set):
- ``RetrievalCandidate`` — the generic, retriever-tagged hit the whole pipeline speaks.
- ``SearchFilters`` — a typed filter object that compiles down to each backend's filter dialect.
- ``QueryContext`` — the per-request bundle handed to every retriever.
- ``Retriever`` — the ``retrieve(query_context) -> [RetrievalCandidate]`` protocol.

Plus the canonical ``content_type`` vocabulary and ID-pattern helpers, defined once here and reused
verbatim by OpenSearch now and Qdrant/Neo4j later, so a filter or a fused result means the same thing
in every backend.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------------------
# Canonical content vocabulary
# --------------------------------------------------------------------------------------------------


class ContentType(str, Enum):
    """The scope/filter axis shared across OpenSearch, Qdrant, and Neo4j.

    Only the members in :data:`CONTENT_TYPES_NOW` are produced and queried today; the rest are
    reserved so the naming is decided once and not re-litigated per backend.
    """

    # Built now.
    QURAN_AYAH = "quran_ayah"
    TRANSLATION = "translation"
    SURAH_NAME = "surah_name"
    # Reserved for future content (not produced or queried yet).
    TAFSIR_CHUNK = "tafsir_chunk"
    BOOK_CHUNK = "book_chunk"
    ARTICLE_CHUNK = "article_chunk"


#: Content types that actually exist in the index today.
CONTENT_TYPES_NOW: frozenset[ContentType] = frozenset(
    {ContentType.QURAN_AYAH, ContentType.TRANSLATION, ContentType.SURAH_NAME}
)

#: The default result scope for general search. Surah-name documents live in the same index but are
#: deliberately excluded from the default ayah/translation result list; they are surfaced only under
#: an explicit surah-name scope or a dedicated navigation block.
DEFAULT_RESULT_CONTENT_TYPES: tuple[ContentType, ...] = (
    ContentType.QURAN_AYAH,
    ContentType.TRANSLATION,
)

#: Stable ``retriever`` tag for the OpenSearch lexical backend. Future backends add their own
#: (e.g. ``"neo4j_graph"``).
RETRIEVER_OPENSEARCH_LEXICAL = "opensearch_lexical"
#: Stable ``retriever`` tag for the Qdrant dense semantic backend.
RETRIEVER_QDRANT_DENSE = "qdrant_dense"


# ``document_id`` builders — the unique searchable unit per content type.
def build_quran_ayah_document_id(surah: int, ayah: int) -> str:
    return f"ayah:{surah}:{ayah}:ar"


def build_translation_document_id(surah: int, ayah: int, source_id: str) -> str:
    return f"ayah:{surah}:{ayah}:translation:{source_id}"


def build_surah_name_document_id(surah: int, language_code: str) -> str:
    return f"surah:{surah}:name:{language_code}"


# ``canonical_content_id`` builders — the logical object a document belongs to. This is the
# collapse/dedup/graph-seed key: the Arabic ayah and all its translations share one canonical id, so
# collapsing on it groups a verse with its renderings.
def build_ayah_canonical_id(surah: int, ayah: int) -> str:
    return f"ayah:{surah}:{ayah}"


def build_surah_canonical_id(surah: int) -> str:
    return f"surah:{surah}"


# --------------------------------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------------------------------


class RetrievalCandidate(BaseModel):
    """A single retriever-tagged hit, in the shape the whole pipeline speaks.

    Every retriever maps its native hit into this: the OpenSearch backend maps its
    ``LexicalSearchHit`` here, and Qdrant/Neo4j return the *same* shape later, so fusion and the
    response builder never branch on which backend produced a result.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    canonical_content_id: str = Field(min_length=1)
    content_type: ContentType
    retriever: str = Field(min_length=1)
    score: float
    rank: int = Field(ge=1)
    text: str = ""
    highlighted_text: str = ""
    #: Per-hit source metadata (surah/ayah numbers, source id, language, ...).
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Which fields the query matched on; informs presentation and per-retriever debugging.
    matched_fields: list[str] = Field(default_factory=list)
    #: Free-form retriever diagnostics (raw scores, explain output), never part of the API contract.
    debug: dict[str, Any] = Field(default_factory=dict)


class SearchFilters(BaseModel):
    """Typed filters that compile down to each backend's filter dialect.

    This replaces threading the raw request ``filters`` dict straight into the query builder. Today
    it compiles to an OpenSearch ``bool/filter``; later it compiles to Qdrant payload filters and
    Neo4j ``WHERE`` clauses from the *same* typed object.

    ``content_types`` defaults to :data:`DEFAULT_RESULT_CONTENT_TYPES` (ayah + translation) so the
    surah-name documents stay out of general results unless a caller asks for them.

    ``include_translations`` and ``translation_languages`` carry the product-level intent of the
    translation control: whether translations are surfaced at all, and (optionally) which languages.
    They are decoupled from the generic ``languages`` filter — which language-restricts *every* hit —
    because the Arabic verses must always be kept regardless of the chosen translation languages.
    Search orchestration turns this intent into per-scope filters via :meth:`quran_ayah_scope` and
    :meth:`translation_scope`.
    """

    model_config = ConfigDict(extra="forbid")

    content_types: list[ContentType] = Field(
        default_factory=lambda: list(DEFAULT_RESULT_CONTENT_TYPES)
    )
    languages: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    surah_numbers: list[int] = Field(default_factory=list)
    ayah_global_min: int | None = None
    ayah_global_max: int | None = None
    include_translations: bool = True
    translation_languages: list[str] = Field(default_factory=list)

    @classmethod
    def from_request_filters(cls, raw: dict[str, Any]) -> SearchFilters:
        """Parse Django's raw request ``filters`` dict into typed filters.

        Tolerant: unknown keys are ignored and malformed values are dropped. ``content_types``
        defaults to the general-result scope (ayah + translation) when absent or all-invalid, so
        surah-name documents stay out of general results unless explicitly requested.
        ``include_translations`` defaults to ``True`` (translations shown unless explicitly disabled).
        """
        if not isinstance(raw, dict):
            raw = {}
        kwargs: dict[str, Any] = {
            "languages": _coerce_str_list(
                raw, "languages", fallback_key="language_codes", casefold=True
            ),
            "source_ids": _coerce_str_list(raw, "source_ids", casefold=False),
            "surah_numbers": _coerce_int_list(
                raw, "surahs", fallback_key="surah_ids", low=1, high=114
            ),
            "ayah_global_min": _coerce_optional_int(raw.get("ayah_global_min")),
            "ayah_global_max": _coerce_optional_int(raw.get("ayah_global_max")),
            "include_translations": _coerce_bool(raw.get("include_translations"), default=True),
            "translation_languages": _coerce_str_list(
                raw, "translation_languages", fallback_key="translation_language", casefold=True
            ),
        }
        content_types = _coerce_content_types(raw.get("content_types"))
        if content_types:
            kwargs["content_types"] = content_types
        return cls(**kwargs)

    def quran_ayah_scope(self) -> SearchFilters:
        """Narrow to Arabic Quran ayahs only — always all languages/sources of the verse.

        The Arabic verse list is independent of the translation control, so language/source/translation
        restrictions are cleared; verse-level restrictions (surah, ayah range) are preserved.
        """
        return self.model_copy(
            update={
                "content_types": [ContentType.QURAN_AYAH],
                "languages": [],
                "source_ids": [],
                "translation_languages": [],
            }
        )

    def translation_scope(self) -> SearchFilters:
        """Narrow to translations, optionally restricted to the chosen translation languages.

        Within this scope ``content_type`` is already ``translation``, so the chosen languages compile
        to a plain ``language_code`` filter that touches translations only.
        """
        return self.model_copy(
            update={
                "content_types": [ContentType.TRANSLATION],
                "languages": list(self.translation_languages),
                "translation_languages": [],
            }
        )

    def to_opensearch_filter(self) -> list[dict[str, Any]]:
        """Compile to a list of OpenSearch ``bool.filter`` clauses (``terms``/``range``)."""
        clauses: list[dict[str, Any]] = []
        if self.content_types:
            clauses.append(
                {"terms": {"metadata.content_type": [ct.value for ct in self.content_types]}}
            )
        if self.languages:
            clauses.append({"terms": {"metadata.language_code": self.languages}})
        if self.source_ids:
            clauses.append({"terms": {"metadata.source_id": self.source_ids}})
        if self.surah_numbers:
            clauses.append({"terms": {"metadata.surah_number": self.surah_numbers}})
        ayah_range: dict[str, int] = {}
        if self.ayah_global_min is not None:
            ayah_range["gte"] = self.ayah_global_min
        if self.ayah_global_max is not None:
            ayah_range["lte"] = self.ayah_global_max
        if ayah_range:
            clauses.append({"range": {"metadata.ayah_global_number": ayah_range}})
        return clauses


class QueryContext(BaseModel):
    """The per-request bundle handed to every retriever.

    ``detected_language`` is a **soft hint**, never a routing gate: AR/FA queries are often
    ambiguous, so the query builder searches *all* language ``content_*`` fields regardless of
    detection and uses the hint only to pick the query normalization and *nudge* boosts — never to
    exclude a language's field.

    ``collapse`` is a **caller-controlled query-time flag**: default-on for general search (group a
    verse with its translations via ``canonical_content_id``), off for modes like "all translations
    of this ayah". It is decoupled from the schema — the id is always indexed.

    ``query_embedding`` is a reserved seam (always ``None`` today) so that, once embeddings arrive,
    the vector is computed once and shared across retrievers instead of recomputed per one.
    """

    model_config = ConfigDict(extra="forbid")

    raw_query: str = Field(min_length=1)
    normalized_query: str = ""
    detected_language: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    user_tier: str = "anonymous"
    top_k: int = Field(default=10, ge=1)
    collapse: bool = True
    query_embedding: list[float] | None = None


@runtime_checkable
class Retriever(Protocol):
    """A source of ``RetrievalCandidate``s for a query.

    ``LexicalRetriever`` (OpenSearch) is the only implementation today; Qdrant/Neo4j retrievers
    implement the same protocol when they are wired.
    """

    #: Stable identifier stamped onto every candidate's ``retriever`` field (e.g.
    #: :data:`RETRIEVER_OPENSEARCH_LEXICAL`).
    name: str

    def retrieve(self, query_context: QueryContext) -> list[RetrievalCandidate]:
        """Return ranked candidates for ``query_context`` (rank 1 = best)."""
        ...


# --------------------------------------------------------------------------------------------------
# Request-filter coercion helpers (tolerant parsing of the untyped Django filters dict)
# --------------------------------------------------------------------------------------------------


def _coerce_content_types(raw: Any) -> list[ContentType]:
    if not isinstance(raw, list):
        return []
    values: list[ContentType] = []
    seen: set[ContentType] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        try:
            content_type = ContentType(item.strip())
        except ValueError:
            continue
        if content_type not in seen:
            values.append(content_type)
            seen.add(content_type)
    return values


def _coerce_int_list(
    raw: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    low: int,
    high: int,
) -> list[int]:
    raw_values = raw.get(key)
    if raw_values is None and fallback_key is not None:
        raw_values = raw.get(fallback_key)
    if not isinstance(raw_values, list):
        return []
    values: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value < low or value > high or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _coerce_str_list(
    raw: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    casefold: bool = True,
) -> list[str]:
    raw_values = raw.get(key)
    if raw_values is None and fallback_key is not None:
        raw_values = raw.get(fallback_key)
    if not isinstance(raw_values, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if isinstance(value, bool) or value is None:
            continue
        text = str(value).strip()
        if casefold:
            text = text.casefold()
        if not text or text in seen:
            continue
        values.append(text)
        seen.add(text)
    return values


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return default
