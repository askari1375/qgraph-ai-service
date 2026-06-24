"""Build searchable documents from a Quran corpus snapshot.

This becomes the home of the document builder currently in ``src/services/search_documents.py``,
extended to: emit **surah-name** documents, stamp a ``canonical_content_id`` on every document, and
use the ``content_type`` vocabulary and ID helpers in ``src.search.contracts`` instead of locally
hardcoded patterns. The Arabic-vs-translation split (the narrow document model) stays.
"""

from __future__ import annotations

from src.api.schemas.corpus import QuranCorpusSnapshot
from src.services.search_documents import SearchIndexDocument


def build_search_documents(snapshot: QuranCorpusSnapshot) -> list[SearchIndexDocument]:
    """Turn a corpus snapshot into the documents to index.

    Not implemented yet: one Arabic document + one document per translation per ayah (as today),
    plus one surah-name document per surah per language, each carrying ``content_type`` and
    ``canonical_content_id`` from :mod:`src.search.contracts`.
    """
    raise NotImplementedError("indexing.documents.build_search_documents is not implemented yet")
