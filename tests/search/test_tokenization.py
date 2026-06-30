"""Token counter: exact tiktoken counts for the production model, char fallback otherwise."""

from src.search.embeddings.tokenization import TokenCounter


def test_known_model_uses_tiktoken():
    counter = TokenCounter(model="text-embedding-3-large")
    assert counter.method == "tiktoken"
    assert counter.count("In the name of God") > 0


def test_unknown_model_falls_back_to_char_estimate():
    counter = TokenCounter(model="deterministic-test")
    assert counter.method == "chars_per_token"
    # 8 chars at ~4 chars/token, ceil-divided.
    assert counter.count("abcdefgh") == 2
    assert counter.count("abcde") == 2  # ceil(5 / 4)
    assert counter.count("") == 0
