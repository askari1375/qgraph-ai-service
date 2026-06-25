"""Golden-query evaluation set.

A small, versioned set of queries with conservative, must-include expectations — not strict ranking
assertions — so analyzer and mapping changes are judged against expected behavior instead of vibes.

It is consumed two ways:
- Offline tests assert the parts that need no live index (the set is well-formed, scopes are wired
  correctly, the analyzer-fix cases are present).
- The index build step re-runs it against the freshly built index before activating an alias, using
  the expectations below.

Each case carries a ``status``:
- ``CONFIRMED`` — the ``must_include_canonical_ids`` are known-correct; the build validation treats
  them as **hard** (a missing one fails the build).
- ``PENDING`` — the ids are candidates awaiting human validation; the build validation treats them as
  **soft** (reported, never fails). Structural expectations (scope, hit content type, language, and
  "returns at least one hit") are still checked. Promote a case to ``CONFIRMED`` once its ids are
  verified against real results.

The two cases marked ``proves_analyzer_fix`` (``لا إله``, ``الرحمن``) are the ones that specifically
prove the normalize-don't-stem analyzers — if they behave, the highest-value correctness change works.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.search.contracts import DEFAULT_RESULT_CONTENT_TYPES, ContentType

EVAL_SET_VERSION = "2026-06-24.v2"

_DEFAULT_TOP_K = 10
_SURAH_NAME_SCOPE = (ContentType.SURAH_NAME,)


class ExpectationStatus(str, Enum):
    CONFIRMED = "confirmed"  # hard: build validation fails if a must-include id is missing
    PENDING = "pending"  # soft: reported only, awaiting human validation


@dataclass(frozen=True)
class GoldenQuery:
    #: Stable case id (used to address a case when refining expectations).
    id: str
    query: str
    #: Language of the query (a soft hint — search still spans all content_* fields).
    language: str
    #: content_type filter the query is evaluated under.
    scope: tuple[ContentType, ...]
    #: content_type(s) the returned hits are expected to have.
    expected_content_types: tuple[ContentType, ...]
    #: Language the expected hits are stored under.
    expected_language: str
    #: canonical_content_id values expected to appear within the top ``top_k`` results.
    must_include_canonical_ids: tuple[str, ...]
    status: ExpectationStatus
    #: The analyzer/search behavior this case guards against regressing.
    guards: str
    top_k: int = _DEFAULT_TOP_K
    notes: str = ""
    proves_analyzer_fix: bool = False

    @property
    def is_hard(self) -> bool:
        """Whether the build validation must fail if an expected id is missing."""
        return self.status is ExpectationStatus.CONFIRMED and bool(self.must_include_canonical_ids)


GOLDEN_QUERIES: tuple[GoldenQuery, ...] = (
    GoldenQuery(
        id="ar-arrahman-diacritics",
        query="الرحمن",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expected_content_types=(ContentType.QURAN_AYAH,),
        expected_language="ar",
        must_include_canonical_ids=("ayah:1:1", "ayah:1:3"),
        status=ExpectationStatus.PENDING,
        guards="diacritic-insensitive normalization (الرحمٰن matches الرحمن)",
        notes="1:1 (basmala) and 1:3 contain الرحمن; confirm against the built corpus.",
        proves_analyzer_fix=True,
    ),
    GoldenQuery(
        id="ar-basmala-phrase",
        query="بسم الله الرحمن الرحيم",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expected_content_types=(ContentType.QURAN_AYAH,),
        expected_language="ar",
        must_include_canonical_ids=("ayah:1:1",),
        status=ExpectationStatus.CONFIRMED,
        guards="match_phrase on the normalized analyzer",
        notes="The basmala is exactly ayah 1:1.",
    ),
    GoldenQuery(
        id="ar-la-ilaha-particle",
        query="لا إله",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expected_content_types=(ContentType.QURAN_AYAH,),
        expected_language="ar",
        must_include_canonical_ids=("ayah:2:255", "ayah:2:163"),
        status=ExpectationStatus.PENDING,
        guards="negation particle لا preserved (the vanilla arabic analyzer would drop it)",
        notes="Candidates: Ayat al-Kursi 2:255 and 2:163; confirm against the built corpus.",
        proves_analyzer_fix=True,
    ),
    GoldenQuery(
        id="ar-musa-maqsura",
        query="موسى",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expected_content_types=(ContentType.QURAN_AYAH,),
        expected_language="ar",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="alef-maqsura / yeh folding (موسى vs موسي)",
        notes="Moses appears in many ayat; pick representative must-includes after seeing results.",
    ),
    GoldenQuery(
        id="ar-surah-name-fatiha",
        query="الفاتحة",
        language="ar",
        scope=_SURAH_NAME_SCOPE,
        expected_content_types=(ContentType.SURAH_NAME,),
        expected_language="ar",
        must_include_canonical_ids=("surah:1",),
        status=ExpectationStatus.CONFIRMED,
        guards="surah-name documents reachable under the surah-name scope",
    ),
    GoldenQuery(
        id="en-surah-name-baqara",
        query="Baqara",
        language="en",
        scope=_SURAH_NAME_SCOPE,
        expected_content_types=(ContentType.SURAH_NAME,),
        expected_language="en",
        must_include_canonical_ids=("surah:2",),
        status=ExpectationStatus.PENDING,
        guards="transliterated surah-name match",
        notes=(
            "Indexed transliteration is 'Al-Baqarah'; the partial 'Baqara' may not match under the "
            "plain english analyzer. This case is also the probe for whether the surah-name field "
            "needs an edge-ngram/fuzzy analyzer — if it fails, that is the signal to add one."
        ),
    ),
    GoldenQuery(
        id="en-mercy-recall",
        query="mercy",
        language="en",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expected_content_types=(ContentType.TRANSLATION,),
        expected_language="en",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="English translation recall",
        notes="Depends on which English translation sources are in the snapshot; 1:1/1:3 are likely.",
    ),
    GoldenQuery(
        id="en-patience-stemming",
        query="patience",
        language="en",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expected_content_types=(ContentType.TRANSLATION,),
        expected_language="en",
        must_include_canonical_ids=(),
        status=ExpectationStatus.PENDING,
        guards="English stemming (patience / patient)",
        notes="Translation-source dependent (sabr renderings).",
    ),
)
