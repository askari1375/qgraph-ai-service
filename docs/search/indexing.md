# Search indexing & activation runbook

How the OpenSearch lexical index is built, validated, and activated. The AI service owns this; Django
never touches OpenSearch.

## Model

- **Physical index** — an immutable build artifact named `qgraph-ayah-lexical-<YYYYMMDD>-<NNN>`.
- **Serving alias** — `qgraph-ayah-lexical-active` (config `QGRAPH_AI_OPENSEARCH_ALIAS`). The app
  always queries the alias, never a physical index name.
- **Activation is an alias swap.** Pointing the alias at a new physical index is the only "go live"
  step — there is no snapshot id/hash to copy into config. The swap is atomic (remove old + add new
  in one request).

Each physical index stores its build profile under `mappings._meta.qgraph_index_profile`: the corpus
snapshot id/hash (provenance) plus the code-compatibility versions (`document_schema_version`,
`normalization_profile_version`, `analysis_profile_version`).

## Text analysis

The primary `content_ar`/`content_fa` fields use custom normalize-don't-stem analyzers. As part of
normalization they strip the superscript/dagger alif (U+0670) before tokenizing, so the bare modern
spelling (`الرحمن`) matches the Quranic orthography (`الرحمٰن`) — mirroring the Python normalizer. The
lower-boost `.stemmed` recall sub-fields use the built-in `arabic`/`persian` analyzers, which fold the
dagger alif onto a full alif, so the full-alif spelling (`الرحمان`) keeps matching too. Changing the
analyzers requires bumping `ANALYSIS_PROFILE_VERSION` and rebuilding (see Drift protection below).

## Commands

```bash
# Build a new physical index from the current Django corpus snapshot. Validates against the golden
# query set; does NOT activate.
python -m src.search.indexing.cli build

#   --activate     activate immediately if validation passes
#   --dry-run      report the plan without writing
#   --languages    comma-separated translation languages (default: all)
#   --surahs       comma-separated surah numbers (default: all)

# Point the serving alias at a built index (atomic swap).
python -m src.search.indexing.cli activate qgraph-ayah-lexical-20260625-001
#   --delete-old   delete the previously-active indices after the swap

# Show the active index, its build profile, and code-compatibility.
python -m src.search.indexing.cli status
```

When the service runs in Docker, prefix with `docker compose ... exec ai-service`.

## Typical flow

1. `build` — pulls the snapshot, builds documents (ayah + translation + surah-name) and the index
   mapping (custom normalize-don't-stem analyzers), creates the physical index, bulk-loads it, and
   runs the golden set against it.
2. Review the printed report. `validation.hard_failures` must be empty; `soft_misses` are `pending`
   golden cases awaiting human validation (see `docs/search/golden-eval-set.md`).
3. `activate <index>` — swap the alias. The app serves the new index immediately.
4. `status` — confirm the active index and that `compatible` is `true`.

`build` never auto-activates unless `--activate` is given, and it refuses to activate a build that
fails hard golden-set expectations — so a bad index never becomes the served one.

## Drift protection

At query time the service reads the alias's `_meta` profile and refuses to serve if the
schema/normalization/analysis versions disagree with the running code (`index_profile_mismatch`).
This catches a deploy and an index that were built against different code. To resolve: rebuild and
activate a fresh index with the current code.

## Configuration

The service needs only:

- `QGRAPH_AI_OPENSEARCH_URL` and credentials (`QGRAPH_AI_OPENSEARCH_USERNAME` /
  `QGRAPH_AI_OPENSEARCH_PASSWORD`, TLS via `QGRAPH_AI_OPENSEARCH_VERIFY_CERTS` /
  `QGRAPH_AI_OPENSEARCH_CA_CERT_PATH`),
- `QGRAPH_AI_OPENSEARCH_ALIAS` (default `qgraph-ayah-lexical-active`),
- the Django snapshot source (`QGRAPH_AI_DJANGO_INTERNAL_BASE_URL` / `_TOKEN`).

There is no active-snapshot id/hash to set — the alias is the source of truth for what is live.
