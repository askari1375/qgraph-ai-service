from src.services.search_normalization import normalize_arabic, normalize_english, normalize_persian


def test_arabic_normalization_removes_diacritics_and_normalizes_alef_forms():
    assert normalize_arabic("إِنَّ ٱللّٰهَ الرَّحْمٰنُ") == "ان الله الرحمن"


def test_arabic_normalization_handles_tatweel_and_ya_variants():
    assert normalize_arabic("آمَنُــوا بِالْهُدَى") == "امنوا بالهدي"


def test_persian_normalization_maps_arabic_yeh_kaf_and_zwnj():
    assert normalize_persian("مي‌كند؛ كِتاب") == "می کند کتاب"


def test_english_normalization_lowercases_and_strips_punctuation():
    assert normalize_english("Mercy, MERCIFUL! Cafe\u0301?") == "mercy merciful café"
