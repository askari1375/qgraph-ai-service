"""OpenSearch index settings, analyzers, and mappings — the "index strategy".

This module owns the most important correctness property: using custom **normalize-don't-stem**
analyzers as the PRIMARY language fields instead of the vanilla built-in ``arabic``/``persian``
analyzers, which strip stopwords and stem and would destroy load-bearing Quranic particles
(لا/ما/إن). The built-in stemmed variants are demoted to a lower-boost ``.stemmed`` sub-field for
recall.

It also carries the index-strategy version: ``analysis_profile_version`` sits next to
``document_schema_version`` and ``normalization_profile_version`` in the index ``_meta`` so
"improve the indexing strategy" becomes bump-version -> rebuild -> validate -> swap-alias.

This replaces ``build_opensearch_index_config`` in ``src/services/opensearch_lexical.py``. There is
no indexed ``normalized_text`` field.
"""

from __future__ import annotations

from typing import Any


def build_index_settings() -> dict[str, Any]:
    """Return the full OpenSearch ``{settings, mappings}`` body for a new index version.

    Not implemented yet: custom ``arabic_normalized``/``persian_normalized``/``english_exact``
    analyzers, ``content_ar``/``content_fa`` with a ``.stemmed`` sub-field and ``.keyword``,
    ``canonical_content_id``/``content_type`` keyword fields, and the build profile (including
    ``analysis_profile_version``) under ``mappings._meta``.
    """
    raise NotImplementedError("indexing.mapping.build_index_settings is not implemented yet")
