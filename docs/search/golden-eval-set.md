# Golden-query evaluation set

A small, versioned set of queries used to judge search-quality changes (analyzers, mappings, query
builder) against **expected behavior** instead of guesswork. The machine-readable source of truth is
[`src/search/indexing/eval_set.py`](../../src/search/indexing/eval_set.py); this document explains it
in prose. Keep the two in sync.

## How it is used

- **Offline tests** (`tests/search/test_eval_set.py`) assert only what needs no live index: the set
  is well-formed, scopes are wired correctly, and the analyzer-fix cases are present.
- **Index build validation** re-runs the set against the freshly built index *before* the alias is
  swapped, so a bad build never goes live.

## Conservative, soft-by-default expectations

Expectations are deliberately conservative: **"this canonical id should appear in the top K"**, not
strict ranking. Each case has a status:

- **`confirmed`** — the expected ids are known-correct. The build validation treats them as **hard**:
  a missing one fails the build.
- **`pending`** — the expected ids are candidates awaiting human validation. The build validation
  treats them as **soft**: missing ones are reported, never fail the build. Structural expectations
  (scope, hit content type, language, and "returns ≥ 1 hit") are still checked.

This lets the set ship now and be tightened later: run a real build, look at the results, then promote
`pending` cases to `confirmed` (or adjust their ids).

Identifiers: `canonical_content_id` is the logical object — `ayah:{surah}:{ayah}` for a verse (the
Arabic ayah and its translations share it) and `surah:{n}` for a surah name.

## The cases

| id | query | lang | scope | expects (must appear in top K) | status | guards |
|---|---|---|---|---|---|---|
| `ar-arrahman-diacritics` | `الرحمن` | ar | ayah + translation | `ayah:1:1`, `ayah:1:3` | **confirmed** | diacritic-insensitive normalization (الرحمٰن ↔ الرحمن) — **proves the analyzer fix** |
| `ar-basmala-phrase` | `بسم الله الرحمن الرحيم` | ar | ayah + translation | `ayah:1:1` | **confirmed** | `match_phrase` on the normalized analyzer |
| `ar-la-ilaha-particle` | `لا إله` | ar | ayah + translation | `ayah:2:163` | pending | negation particle **لا preserved** (the vanilla `arabic` analyzer would drop it) — **proves the analyzer fix** |
| `ar-musa-maqsura` | `موسى` | ar | ayah + translation | _(none yet)_ | pending | alef-maqsura / yeh folding (موسى ↔ موسي) |
| `ar-surah-name-fatiha` | `الفاتحة` | ar | surah-name only | `surah:1` | **confirmed** | surah-name docs reachable under the surah-name scope |
| `en-surah-name-baqara` | `Baqara` | en | surah-name only | `surah:2` | **confirmed** | transliterated surah-name match (Baqara ↔ Al-Baqarah) |
| `en-mercy-recall` | `mercy` | en | ayah + translation | _(none yet)_ | pending | English translation recall |
| `en-patience-stemming` | `patience` | en | ayah + translation | _(none yet)_ | pending | English stemming (patience / patient) |

`ar-la-ilaha-particle` and `ar-arrahman-diacritics` are the two that prove the
normalize-don't-stem analyzers. If they behave, the highest-value correctness change works.

## Remaining `pending` cases

These were validated against the full corpus and left `pending` because their exact membership is
ranking- or translation-dependent (so they should not hard-block a build):

- `ar-la-ilaha-particle` — `2:163` ranks in the top results; `2:255` (Ayat al-Kursi) also contains
  لا إله but its length normalizes it out of the top-k. The particle preservation itself is
  hard-guarded by the normalizer unit test.
- `ar-musa-maqsura`, `en-mercy-recall`, `en-patience-stemming` — return results, but their concrete
  `must_include_canonical_ids` are left empty for a human to pin from real output.

## What a human needs to do

Look at the actual results for each `pending` case and either fill in concrete
`must_include_canonical_ids` and promote it to `confirmed`, or leave it `pending` if its membership is
inherently fragile. Edit the cases in `src/search/indexing/eval_set.py` and bump `EVAL_SET_VERSION`
when the set changes.
