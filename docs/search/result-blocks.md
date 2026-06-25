# Search result blocks

The low-token reference for what `POST /v1/search/execute` (and a ready search job's result) returns,
so you don't have to grep the frontend/backend to infer it.

## Envelope

`SearchExecuteResponse`:

| field | type | notes |
|---|---|---|
| `title` | string | |
| `overall_confidence` | float 0..1 | |
| `render_schema_version` | string | currently `"v1"` |
| `metadata` | object | includes `backend` + index provenance (snapshot id/hash, schema/normalization/analysis versions, `index_id`) |
| `blocks` | `SearchResponseBlock[]` | ordered |

`SearchResponseBlock`: `order` (int, **unique** across blocks), `block_type` (string), `title`,
`payload` (object), `explanation`, `confidence` (0..1), `provenance` (object), `warning_text`,
`items` (`SearchResultItem[]`, each `rank` **unique** within the block).

## Block types the frontend renders today

From `qgraph-frontend/src/features/search/blocks/block-registry.ts`. Any other `block_type` falls back
to `UnknownBlock` (degrades gracefully — not a real render).

- **`text`** → `payload = { headline?, details }` (`details`: string)
- **`markdown`** → `payload = { headline?, content }` (`content`: GFM string; raw HTML is not rendered)
- **`surah_distribution`** → `payload = { values: [{ surah: int, value: int }], y_label?, max_value? }`
- **`ayah_results`** → `payload = { query, result_count, language_code }`; the matches are the block's
  **`items[]`** (verse cards). Each `SearchResultItem`: `rank` (1-based, unique within the block),
  `result_type` (`ayah` | `surah`), `score` (0..1, min-max within the block for relative bars), `title`
  (reference, e.g. `Surah 1, Ayah 1`), `snippet_text`, `highlighted_text` (contains `<mark>…</mark>` —
  the typed renderer renders the highlight), and `match_metadata` (`document_id`,
  `canonical_content_id`, `content_type`, `text`, `surah_number`, `ayah_number`, `ayah_global_number`,
  `language_code`, `source_id`, `source_name`). Django assigns each persisted item an `id`;
  bookmark/feedback target it via `result_item_id`.

## What search returns now

The AI service groups the matches — Arabic verses and translations are different kinds of content, so
they are separate blocks rather than one mixed list. The block **order** is fixed by the service:

1. a **`surah_distribution`** block (when there are matches): per-surah match counts from a `terms`
   aggregation over the **Arabic verse** scope, so the chart is stable regardless of the translation
   control.
2. one **`ayah_results`** block for the Arabic Quran verses (`title = "Quran"`, `payload.language_code
   = "ar"`).
3. one **`ayah_results`** block **per translation language** present (`title` like
   `"English translations"`, `payload.language_code` the code), in a stable language order. Omitted
   entirely when translations are excluded or none matched.

All result blocks share `block_type = "ayah_results"` (the same verse-card renderer handles them);
`title` and `payload.language_code` distinguish them.

### Request filters

`filters` (forwarded verbatim by Django) understands:

- `include_translations` (bool, default `true`) — when `false`, only the Arabic verse block is built.
- `translation_languages` (list of language codes, e.g. `["en"]`; empty = all) — restricts which
  translation languages are retrieved/shown. Never restricts the Arabic verses.

Keep this file in sync when `block-registry.ts` changes.
