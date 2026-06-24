"""Offline checks on the golden eval set.

Live-index hit validation runs in the index build step; here we assert only what holds without
OpenSearch: the set is well-formed, the analyzer-fix cases are present, and surah-name vs general
scopes are wired as designed.
"""

from src.search.contracts import ContentType
from src.search.indexing.eval_set import EVAL_SET_VERSION, GOLDEN_QUERIES


def test_eval_set_is_versioned_and_non_empty():
    assert EVAL_SET_VERSION
    assert len(GOLDEN_QUERIES) == 8


def test_analyzer_fix_cases_present():
    proving = {q.query for q in GOLDEN_QUERIES if q.proves_analyzer_fix}
    assert proving == {"لا إله", "الرحمن"}


def test_surah_name_cases_use_surah_name_scope():
    surah_name_queries = {q.query for q in GOLDEN_QUERIES if q.scope == (ContentType.SURAH_NAME,)}
    assert surah_name_queries == {"الفاتحة", "Baqara"}


def test_general_scope_excludes_surah_name():
    for query in GOLDEN_QUERIES:
        if query.scope != (ContentType.SURAH_NAME,):
            assert ContentType.SURAH_NAME not in query.scope
            assert ContentType.QURAN_AYAH in query.scope
            assert ContentType.TRANSLATION in query.scope
