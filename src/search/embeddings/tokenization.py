"""Token counting for the build preflight — measure input size before paying to embed it.

The cost/token estimate and the hard per-input ceiling both need a token count for prepared text. For
the production OpenAI model ``tiktoken`` gives the exact tokenization; for any other model id (the
deterministic test provider, a future provider without a tiktoken mapping) there is no exact tokenizer
here, so the counter falls back to a coarse character-based estimate and records which method it used,
so a report never silently implies an exactness it does not have.
"""

from __future__ import annotations

import tiktoken

#: Coarse fallback when a model has no tiktoken mapping. ~4 chars/token is the usual English rule of
#: thumb; it is only an estimate, surfaced as ``method == "chars_per_token"`` so it is never mistaken
#: for an exact count.
_FALLBACK_CHARS_PER_TOKEN = 4


class TokenCounter:
    """Counts tokens for one model, using tiktoken when available and a char estimate otherwise."""

    def __init__(self, *, model: str):
        self.model = model
        try:
            self._encoding: tiktoken.Encoding | None = tiktoken.encoding_for_model(model)
            self.method = "tiktoken"
        except KeyError:
            self._encoding = None
            self.method = "chars_per_token"

    def count(self, text: str) -> int:
        """Token count for ``text`` — exact under tiktoken, a ceil char estimate under the fallback."""
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return -(-len(text) // _FALLBACK_CHARS_PER_TOKEN)
