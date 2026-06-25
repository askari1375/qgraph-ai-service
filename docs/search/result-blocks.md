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
- **`ayah_results`** → `payload = { query, result_count }`; the matches are the block's **`items[]`**
  (verse cards). Each `SearchResultItem`: `rank`, `result_type` (`ayah` | `surah`), `score` (0..1,
  min-max for relative bars), `title` (reference, e.g. `Surah 1, Ayah 1`), `snippet_text`,
  `highlighted_text` (contains `<mark>…</mark>` — the typed renderer renders the highlight), and
  `match_metadata` (`document_id`, `canonical_content_id`, `content_type`, `text`, `surah_number`,
  `ayah_number`, `ayah_global_number`, `language_code`, `source_id`, `source_name`). Django assigns each
  persisted item an `id`; bookmark/feedback target it via `result_item_id`.

## What search returns now

- an **`ayah_results`** block: the ranked matches as typed `items[]` (verse cards).
- a **`surah_distribution`** block (when there are matches): real per-surah match counts from a `terms`
  aggregation on `metadata.surah_number`.

Keep this file in sync when `block-registry.ts` changes.
