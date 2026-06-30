# Embedding provider

How the AI service turns text into vectors for semantic search. The embedding provider is a small,
project-owned seam (`src/search/embeddings/`), deliberately not a third-party embedding framework:
the compatibility facts that protect a vector collection from silent model/dimension drift live in our
own types and are stamped onto each semantic index profile.

## Production provider

The first production provider is **OpenAI `text-embedding-3-large`**:

- **native dimension `3072`** — V1 uses the model's native size with **no reduction**. The API's
  `dimensions` truncation parameter is intentionally not sent; any future smaller-dimension build is a
  separate, derived collection, never an in-place change to stored vectors.
- **L2-normalized** output, so cosine and dot rank identically — the semantic profile pins **cosine**.
- **symmetric**: OpenAI does not distinguish query vs document inputs, so
  `distinguishes_input_modes=False` and callers never pass an input mode. (A future asymmetric
  provider such as Cohere would set this `True` and apply its own `input_type` internally.)
- max **8192 input tokens** per request. Quran ayah and translation documents are small; inputs are
  never silently truncated — an over-long input is surfaced as an error, not shortened.

Provider, model, and dimension are **immutable per collection**. Changing any of them means building
and activating a new Qdrant collection — never an in-place migration.

## Configuration

Settings use the `QGRAPH_AI_` prefix (`src/config.py`); see `.env.example`. The provider key is a
secret and belongs only in the deployment env file.

| Setting | Production value |
|---|---|
| `QGRAPH_AI_EMBEDDING_PROVIDER` | `openai` |
| `QGRAPH_AI_EMBEDDING_MODEL` | `text-embedding-3-large` |
| `QGRAPH_AI_EMBEDDING_DIMENSIONS` | `3072` |
| `QGRAPH_AI_EMBEDDING_API_KEY` | _(secret)_ |
| `QGRAPH_AI_EMBEDDING_TIMEOUT_SECONDS` | `30` |
| `QGRAPH_AI_EMBEDDING_MAX_RETRIES` | `2` |
| `QGRAPH_AI_EMBEDDING_DOCUMENT_BATCH_SIZE` | `96` |

An **empty** `EMBEDDING_PROVIDER` is the safe default: no provider is configured, so a real semantic
build fails loudly (`embedding_provider_not_configured`) rather than guessing.

## How it resolves

`build_embedding_provider(settings)` (`src/search/embeddings/factory.py`) is the single production
resolution point — one branch per provider, no registry. The semantic build CLI
(`python -m src.search.vector_indexing.cli build`) calls it to obtain the provider; the deterministic
test provider (`tests/support/embeddings.py`) is reachable **only** by direct dependency injection in
tests, never through configuration.

`OpenAIEmbeddingProvider` (`src/search/embeddings/openai_provider.py`) wraps the official `openai`
SDK. The SDK owns transport, timeout, and retry/backoff; the adapter owns the project contract:

- reassembles vectors in input order (sorted by the response `index`);
- validates response cardinality, dimension, and finiteness (`validate_embedding_vectors`) — a
  dimension that disagrees with the configured/profile value fails as `embedding_response_invalid`
  instead of being silently reduced;
- maps SDK failures (timeout, connection, rate limit, API status) to
  `embedding_provider_unavailable` with no silent fallback.

## Activation gate (not yet run)

Building a real collection with this provider and validating it against a reviewed **cross-lingual**
golden set (Persian/English query → correct Arabic ayah) is the gate that certifies semantic quality.
It needs a production key and a reviewed eval set, and is tracked as a follow-on to wiring the
provider here.
