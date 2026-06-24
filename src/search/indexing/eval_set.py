"""Golden-query evaluation set.

A small, versioned set of queries with their expected behavior, so analyzer or mapping changes are
judged against expectations instead of vibes. It is consumed two ways: offline tests assert the parts
that don't need a live index (scope, surah-name reachability, normalizer intent), and the index build
step re-runs it against the freshly built index before activating an alias.

``expected_canonical_ids`` is intentionally left to be refined against real data; the harness already
encodes each case's language, scope, and what it guards. ``لا إله`` and ``الرحمن`` are the two cases
that prove the normalize-don't-stem fix — if they pass, the highest-value correctness change works.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.search.contracts import DEFAULT_RESULT_CONTENT_TYPES, ContentType

EVAL_SET_VERSION = "2026-06-24.v1"

_SURAH_NAME_SCOPE = (ContentType.SURAH_NAME,)


@dataclass(frozen=True)
class GoldenQuery:
    query: str
    language: str
    #: The content_type scope the query is evaluated under.
    scope: tuple[ContentType, ...]
    #: Human description of the intended result; expected ids refined later.
    expects: str
    #: What regression this case guards against.
    guards: str
    #: To be filled with concrete canonical ids once refined against real data.
    expected_canonical_ids: tuple[str, ...] = ()
    #: True for the cases that specifically prove the normalize-don't-stem analyzers.
    proves_analyzer_fix: bool = False


GOLDEN_QUERIES: tuple[GoldenQuery, ...] = (
    GoldenQuery(
        query="الرحمن",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expects="ayat containing الرحمٰن with or without diacritics",
        guards="diacritic-insensitive normalization",
        proves_analyzer_fix=True,
    ),
    GoldenQuery(
        query="بسم الله الرحمن الرحيم",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expects="Al-Fatihah 1:1 as an exact phrase",
        guards="match_phrase on the normalized analyzer",
    ),
    GoldenQuery(
        query="لا إله",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expects="ayat keeping the negation particle لا",
        guards="the particle fix — the vanilla arabic analyzer would drop لا",
        proves_analyzer_fix=True,
    ),
    GoldenQuery(
        query="موسى",
        language="ar",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expects="ayat mentioning Moses",
        guards="alef-maqsura / yeh folding",
    ),
    GoldenQuery(
        query="الفاتحة",
        language="ar",
        scope=_SURAH_NAME_SCOPE,
        expects="the surah-name hit for surah 1",
        guards="surah-name documents + surah-name scope",
    ),
    GoldenQuery(
        query="Baqara",
        language="en",
        scope=_SURAH_NAME_SCOPE,
        expects="surah 2 by its transliterated name",
        guards="transliterated surah-name match",
    ),
    GoldenQuery(
        query="mercy",
        language="en",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expects="Al-Fatihah and mercy-themed translations",
        guards="English translation recall",
    ),
    GoldenQuery(
        query="patience",
        language="en",
        scope=DEFAULT_RESULT_CONTENT_TYPES,
        expects="sabr-related translations (patience / patient)",
        guards="English stemming",
    ),
)
