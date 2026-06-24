"""Guards the normalize-don't-stem intent of the canonical normalizer.

These assert the behavior the custom OpenSearch analyzers also rely on (the Python normalizer and the
analyzers must stay consistent in spirit): fold diacritics and letter variants, but never drop a
load-bearing particle. The two analyzer-fix eval cases (لا إله, الرحمن) are checked here at the
offline Python level.
"""

from src.search.indexing.normalization import (
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_PROFILE_VERSION,
    normalize_arabic,
    normalize_text,
)


def test_negation_particle_la_is_preserved():
    # The vanilla `arabic` analyzer would drop لا as a stopword; normalization must keep it.
    assert "لا" in normalize_arabic("لَا إِلَٰهَ").split()


def test_arrahman_diacritics_are_folded():
    # الرحمٰن with harakat and dagger-alef normalizes to a bare form that matches الرحمن.
    assert normalize_arabic("الرَّحْمَٰن") == normalize_arabic("الرحمن")


def test_alef_maqsura_folds_for_musa():
    assert normalize_arabic("مُوسَىٰ") == normalize_arabic("موسي")


def test_normalize_text_dispatches_by_language():
    assert normalize_text("In the Name!", "en") == "in the name"
    assert normalize_text("هُوَ", "ar") == "هو"


def test_profile_identity_is_stable():
    assert NORMALIZATION_PROFILE_ID == "qgraph_search_normalization"
    assert NORMALIZATION_PROFILE_VERSION == "2026-06-22.v1"
