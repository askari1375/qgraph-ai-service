"""Cross-lingual semantic/hybrid evaluation set (the first-class target of the Qdrant work).

This set measures the decision the whole semantic design hinges on: does a query in one language surface
the relevant *Arabic* ayah (and its translations) from one shared multilingual vector space? It reuses
the lexical golden set's :class:`~src.search.indexing.eval_set.GoldenQuery` shape; a **cross-lingual**
case is simply one whose query ``language`` differs from its ``expected_language``.

Every case starts ``PENDING`` (soft): the expected ayah for a conceptual query is a reviewed judgment,
not an engineering guess, and the real expectations can only be confirmed once a collection is built
with the production embedding provider (the paid Phase-4 gate). After that build, human-validated cases
are promoted to ``CONFIRMED`` — mirroring how the lexical golden set was hardened. Until then this file
is the harness and the candidate hypotheses, not a passing quality bar.
"""

from __future__ import annotations

from src.search.contracts import DEFAULT_RESULT_CONTENT_TYPES, ContentType
from src.search.indexing.eval_set import ExpectationStatus, GoldenQuery

SEMANTIC_EVAL_SET_VERSION = "2026-06-30.v1"

_ARABIC_RESULT = (ContentType.QURAN_AYAH,)


SEMANTIC_GOLDEN_QUERIES: tuple[GoldenQuery, ...] = (
    GoldenQuery(
        id="fa-mercy-to-arabic",
        query="رحمت",
        language="fa",
        scope=_ARABIC_RESULT,
        expected_content_types=_ARABIC_RESULT,
        expected_language="ar",
        must_include_canonical_ids=("ayah:1:1", "ayah:1:3"),
        status=ExpectationStatus.PENDING,
        guards="cross-lingual: a Persian concept query surfaces the Arabic ayah (shared vector space)",
        notes="Candidate ids await validation against a real OpenAI-built collection (Phase-4 gate).",
    ),
    GoldenQuery(
        id="en-mercy-to-arabic",
        query="mercy",
        language="en",
        scope=_ARABIC_RESULT,
        expected_content_types=_ARABIC_RESULT,
        expected_language="ar",
        must_include_canonical_ids=("ayah:1:3",),
        status=ExpectationStatus.PENDING,
        guards="cross-lingual: an English concept query surfaces the Arabic ayah",
    ),
    GoldenQuery(
        id="en-patience-to-arabic",
        query="patience and steadfastness",
        language="en",
        scope=_ARABIC_RESULT,
        expected_content_types=_ARABIC_RESULT,
        expected_language="ar",
        must_include_canonical_ids=("ayah:2:153",),
        status=ExpectationStatus.PENDING,
        guards="cross-lingual paraphrase with little lexical overlap with the Arabic surface form",
    ),
    GoldenQuery(
        id="fa-charity-to-translation",
        query="انفاق در راه خدا",
        language="fa",
        scope=(ContentType.TRANSLATION,),
        expected_content_types=(ContentType.TRANSLATION,),
        expected_language="en",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="Persian conceptual query retrieving an English translation of the same ayah",
        notes="Must-include ids deferred: pick after inspecting real results.",
    ),
    GoldenQuery(
        id="en-mercy-to-arberry-translation",
        query="mercy",
        language="en",
        scope=(ContentType.TRANSLATION,),
        expected_content_types=(ContentType.TRANSLATION,),
        expected_language="en",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="recall over the curated English (Arberry) translation in the semantic corpus",
        notes="Confirms the single canonical English source is searchable on its own scope.",
    ),
    GoldenQuery(
        id="fa-patience-to-moezzi-translation",
        query="صبر و استقامت",
        language="fa",
        scope=(ContentType.TRANSLATION,),
        expected_content_types=(ContentType.TRANSLATION,),
        expected_language="fa",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="recall over the curated Persian (Moezzi) translation in the semantic corpus",
        notes="Confirms the single canonical Persian source is searchable on its own scope.",
    ),
    GoldenQuery(
        id="ar-orthographic-variant-semantic",
        query="الرحمان",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expected_content_types=_ARABIC_RESULT,
        expected_language="ar",
        must_include_canonical_ids=("ayah:1:1", "ayah:1:3"),
        status=ExpectationStatus.PENDING,
        guards="Arabic orthographic variant (full alif) still matches semantically",
    ),
    GoldenQuery(
        id="en-paraphrase-low-overlap",
        query="those who turn away from the reminder",
        language="en",
        scope=_ARABIC_RESULT,
        expected_content_types=_ARABIC_RESULT,
        expected_language="ar",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="paraphrase with little lexical overlap; semantic recall over BM25",
    ),
    GoldenQuery(
        id="en-ambiguous-zero-result",
        query="quarterly revenue projections",
        language="en",
        scope=_ARABIC_RESULT,
        expected_content_types=_ARABIC_RESULT,
        expected_language="ar",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="out-of-domain query should not surface a confident, irrelevant ayah",
        notes="Used to check the system does not over-claim on ambiguous/irrelevant input.",
    ),
)


def cross_lingual_cases() -> tuple[GoldenQuery, ...]:
    """Cases whose query language differs from the expected result language — the core target."""
    return tuple(
        case for case in SEMANTIC_GOLDEN_QUERIES if case.language != case.expected_language
    )
