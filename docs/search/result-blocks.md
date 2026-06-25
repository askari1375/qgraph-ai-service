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
- **`markdown`** → `payload = { headline?, content }` (`content`: GFM string — headings, tables, lists,
  code, blockquote, links). **Raw HTML is not rendered**, so highlight `<mark>` tags must be stripped
  before they go in `content`.
- **`surah_distribution`** → `payload = { values: [{ surah: int, value: int }], y_label?, max_value? }`

## What search returns now

A single **`markdown`** block: a table of the ranked matches (`# | Reference | Match | Source`).
`items` is empty. This is the renderable bridge for the current UI.

## Known gaps (next steps)

- **`items[]` is not rendered.** There is no `results`/verse renderer yet, so per-item bookmark/feedback
  (which the Django search app models) is not wired. A typed verse-results block + frontend renderer is
  the scheduled next step; it will carry `items[]` as proper verse cards (RTL Arabic, highlight ranges,
  click-through).
- **No distribution chart yet.** The real `surah_distribution` (a `terms` aggregation on
  `metadata.surah_number`) returns alongside the typed verse block.

Keep this file in sync when `block-registry.ts` changes.
