"""Offline checks on the golden eval set.

Live-index hit validation runs in the index build step; here we assert only what holds without
OpenSearch: the set is well-formed, the analyzer-fix cases are present, scopes are wired as designed,
and the confirmed/pending contract is internally consistent.
"""

from src.search.contracts import ContentType
from src.search.indexing.eval_set import (
    EVAL_SET_VERSION,
    GOLDEN_QUERIES,
    ExpectationStatus,
)

_CANONICAL_PREFIXES = ("ayah:", "surah:")


def test_eval_set_is_versioned_and_non_empty():
    assert EVAL_SET_VERSION
    assert len(GOLDEN_QUERIES) == 8


def test_case_ids_are_unique():
    ids = [case.id for case in GOLDEN_QUERIES]
    assert len(ids) == len(set(ids))


def test_analyzer_fix_cases_present():
    proving = {case.query for case in GOLDEN_QUERIES if case.proves_analyzer_fix}
    assert proving == {"لا إله", "الرحمن"}


def test_surah_name_cases_use_surah_name_scope():
    surah_name_queries = {
        case.query for case in GOLDEN_QUERIES if case.scope == (ContentType.SURAH_NAME,)
    }
    assert surah_name_queries == {"الفاتحة", "Baqara"}


def test_general_scope_excludes_surah_name():
    for case in GOLDEN_QUERIES:
        if case.scope != (ContentType.SURAH_NAME,):
            assert ContentType.SURAH_NAME not in case.scope
            assert ContentType.QURAN_AYAH in case.scope
            assert ContentType.TRANSLATION in case.scope


def test_every_case_is_well_formed():
    for case in GOLDEN_QUERIES:
        assert case.query and case.language and case.expected_language
        assert case.top_k > 0
        assert case.expected_content_types
        # Hits must be a content type the query's scope actually permits.
        assert set(case.expected_content_types).issubset(set(case.scope))
        for canonical_id in case.must_include_canonical_ids:
            assert canonical_id.startswith(_CANONICAL_PREFIXES)


def test_confirmed_cases_carry_hard_expectations_pending_may_be_empty():
    for case in GOLDEN_QUERIES:
        if case.status is ExpectationStatus.CONFIRMED:
            # A confirmed case must have something concrete to enforce.
            assert case.must_include_canonical_ids
            assert case.is_hard
        else:
            assert case.status is ExpectationStatus.PENDING
            assert not case.is_hard
