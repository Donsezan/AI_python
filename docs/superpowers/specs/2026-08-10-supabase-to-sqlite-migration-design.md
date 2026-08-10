# Migrate article storage from Supabase to local SQLite

**Date:** 2026-08-10
**Status:** Approved, ready for implementation planning

## Problem

The bot exceeded Supabase's free-tier **egress/bandwidth** cap (5 GB/month).

The cause is the read pattern, not the data volume. `DataService.fetch_recent_articles()`
(`data_service.py:71`) selects `title,embedding,url` for **every** row on **every** cycle.
Embeddings are stored as JSON arrays of ~3072 floats — roughly 50–60 KB per row as text.
At ~100 retained rows that is ~6 MB per fetch, and the scheduler runs 144 cycles per day:

```
6 MB x 144 cycles = ~860 MB/day  ->  ~26 GB/month
```

The dataset itself is trivial (10-day retention, ~100–300 rows). Moving the data onto the
machine that already runs the bot removes the metered resource entirely rather than tuning
it.

## Goals

1. Remove the Supabase dependency completely — no network calls for storage.
2. Preserve dedup history across the cutover, so no duplicate Telegram posts and no wasted
   Gemini quota on re-evaluating articles already handled.
3. Keep `main.py`'s per-article pipeline logic unchanged.
4. Make storage tests runnable without credentials or network.

## Non-goals

- Changing the dedup algorithm, similarity threshold, or scoring.
- Vector indexes / ANN search. Brute-force cosine over a few hundred rows in numpy is
  already how it works and is not a bottleneck.
- Automated backups (see Risks).
- Fixing the unrelated README drift documented in CLAUDE.md's "Known doc drift" section
  (`fetching_data.py`, the embedding-staggering claim). Only Supabase-related doc content
  is in scope.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Engine | SQLite via stdlib `sqlite3` | Zero new dependencies, no server process, no RAM cost against the systemd `MemoryMax=512M` cap. |
| Existing data | One-off migration script | Preserves dedup history; avoids a duplicate-post burst on cutover. |
| Supabase code | Removed entirely | The whole point is to stop using it; git history retains the old code. |
| Embedding format | float32 BLOB | ~12 KB vs ~60 KB per row; `np.frombuffer` decodes far faster than `json.loads`, and the whole table is decoded every cycle. |

### Why "start fresh" was rejected

`seen_cache.json` does **not** record successful posts — `main.py:122` writes the `posted`
status only when a Supabase save *fails*. Duplicate suppression for successfully posted
articles depends entirely on the database rows (`is_url_seen` + embedding similarity). An
empty database would therefore re-evaluate and re-post every article still on the homepage
from the last ~7 days.

## Design

### 1. Schema

A single `articles.db` file in the process working directory, alongside `seen_cache.json`.
Created and migrated on first connect — no manual DDL step, unlike the Supabase setup.

```sql
CREATE TABLE IF NOT EXISTS articles (
  id               TEXT PRIMARY KEY,
  title            TEXT NOT NULL,
  date             TEXT NOT NULL,   -- ISO-8601 string, unchanged from today
  url              TEXT,
  embedding        BLOB,            -- float32 little-endian; NULL when embedding failed
  title_translated TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS articles_url_idx  ON articles(url) WHERE url IS NOT NULL;
CREATE INDEX        IF NOT EXISTS articles_date_idx ON articles(date);
```

Columns map 1:1 to the current Supabase table, including `title_translated` (which the
README's DDL omits but the bot writes).

`date` stays an ISO-8601 **text** column. `cleanup_old_articles` compares it
lexicographically (`WHERE date < ?`), which is correct for a consistent ISO format and
matches today's PostgREST `lt.` behaviour exactly.

Connection pragmas: `journal_mode=WAL`, `synchronous=NORMAL`. Connections are short-lived
(one per operation, via a context manager) — this sidesteps `check_same_thread` entirely
and holds no long-running locks.

### 2. Embedding encoding

- Encode: `np.asarray(values, dtype=np.float32).tobytes()` — accepts both plain lists
  (returned by `_embed`) and numpy arrays (returned by `fetch_all`).
- Decode: `np.frombuffer(blob, dtype=np.float32)` — dimension-agnostic, inferred from
  length. Nothing in the codebase hardcodes the embedding dimension, and nothing should.
- `NULL` stays `NULL` and still triggers the existing Jaccard word-overlap fallback in
  `is_new_article_cached`.

**Accepted precision change:** float64 → float32 loses ~9 significant decimal digits. With a
cosine threshold of 0.85 this is immaterial. Migrated legacy rows are converted to float32
too, so all comparisons remain like-for-like.

### 3. Module boundaries

**New `sqlite_store.py` — storage only.** Owns the connection, schema bootstrap, BLOB
encode/decode, and raw CRUD:

- `fetch_all()` → list of row dicts with exactly `title`, `url`, `embedding` (mirroring
  today's `select=title,embedding,url`; other columns are write-only)
- `insert(...)` → bool
- `delete_older_than(cutoff_iso)` → deleted count

It knows nothing about embeddings-as-semantics, similarity, or thresholds.

**`data_service.py` keeps `DataService`** and everything it is actually about — `_embed`,
`_cosine`, `_jaccard`, and the dedup policy — delegating all persistence to `SqliteStore`.
`DataService` takes a `db_path` and constructs its own `SqliteStore` (keeping the `main.py`
call site a one-liner); tests point `db_path` at a tempdir file rather than injecting a
double. All six public methods keep **identical signatures and return shapes**:

| Method | Contract preserved |
|---|---|
| `fetch_recent_articles()` | list of dicts with `title`, `url`, `embedding` |
| `is_url_seen(href, rows)` | unchanged, pure |
| `is_new_article_cached(title, rows)` | `(bool, embedding)` |
| `save_article(...)` | `bool` |
| `cleanup_old_articles(max_age_days=10)` | logs count |
| `is_new_article(title)` | test-only convenience wrapper |

This matters because `main.py` appends in-cycle results directly into `known_articles`
(`main.py:119`, `main.py:262`) as plain dicts with list embeddings. Since `_cosine` calls
`np.array()` on both operands, mixed list/ndarray rows work without special-casing.

**Error-handling parity is required, not incidental.** DB failures must be caught and
logged, returning `[]` / `False` exactly as the HTTP path does. `main.py` depends on this:
a `False` from `save_article` is what triggers `record_terminal(href, "posted")`, the guard
against duplicate posts.

### 4. `main.py` changes

Only two edits:

- `main.py:47` — construct `DataService(db_path=..., DISTANCE_THRESHOLD=..., gemini_api_key=...)`.
- `main.py:37-39` — drop `SUPABASE_URL` / `SUPABASE_KEY` from the import-time env-var check.

New optional env var `DB_PATH` (default `articles.db`). No other pipeline logic changes.

### 5. Migration script

`scripts/migrate_supabase_to_sqlite.py`, run once during cutover:

- Reads `SUPABASE_URL` / `SUPABASE_KEY` from `.env` directly (the pattern the existing tests
  use), so it does not depend on `main.py`'s env validation.
- Pages the table via the PostgREST `Range` header, 1000 rows at a time, selecting
  `id,title,date,url,embedding,title_translated`.
- Converts each JSON embedding list to a float32 BLOB; `NULL`/missing stays `NULL`.
- Writes with `INSERT OR IGNORE` keyed on `id` — **idempotent and safe to re-run**.
- `--dry-run` reports row counts without writing.
- Prints a final `fetched / inserted / skipped` summary.

This is the last Supabase contact. It costs one full-table read (a few MB) against the
already-exceeded egress meter.

### 6. Tests

- **Delete** `tests/test_supabase_connection.py`.
- **New `tests/test_sqlite_store.py`** — real SQLite in a tempdir, no network, no
  credentials. Covers: CRUD round-trip; BLOB float32 fidelity; `NULL` embedding handling;
  URL unique-index behaviour; `delete_older_than` cutoff boundary (rows exactly at the
  cutoff are retained, older ones deleted); schema bootstrap on a fresh file and idempotent
  re-open.
- **Update `tests/test_similarity.py`** — replace the `_mock_supabase` request-patching
  context manager with a seeded tempdir store. Real-embedding tests stay gated on
  `GEMINI_API_KEY`.
- **New** unit test for the migration script's JSON→BLOB conversion helper (round-trip and
  `NULL` passthrough).

Net effect: storage tests need no credentials and hit no network.

### 7. Documentation

- **CLAUDE.md** — pipeline steps 5/8/9, the Key Configuration block, the import-time
  env-check note, the Testing section, and the `DataService` description.
- **README.md** — replace the Supabase prerequisite and DDL block with the SQLite file
  description; update the `.env` sample and the test list.
- **Redme-Oracle.md** — `.env` sample, plus the rollout/backup note below.
- **.gitignore** — add `*.db-wal` and `*.db-shm` (`*.db` is already covered).

### 8. VM rollout

```
sudo systemctl stop newsbot
cd ~/YOUR_REPO && git pull
python3 scripts/migrate_supabase_to_sqlite.py      # needs SUPABASE_* still in .env
python3 main.py --dry-run                          # verify
# remove SUPABASE_URL / SUPABASE_KEY from .env
sudo systemctl start newsbot
```

Order matters: the migration must run **before** `SUPABASE_*` is stripped from `.env`, and
`--dry-run` must pass before the service restarts (a dry run reads the DB but never writes
it).

## Risks

**`articles.db` becomes the only copy of the dedup history, with no managed backups.**
Losing it costs one round of duplicate posts plus the Gemini quota to re-evaluate ~7 days of
homepage articles — recoverable, not catastrophic. Automated backups are deliberately out of
scope; if added later, use `sqlite3 .backup` (or copy the `-wal`/`-shm` files too), never a
bare `cp` of a live WAL database.

**Migration script correctness is single-shot in practice.** Mitigated by idempotency
(`INSERT OR IGNORE` on `id`) and `--dry-run`, so a partial or repeated run is safe.

## Success criteria

1. `main.py --dry-run` completes with zero network calls to `*.supabase.co`.
2. `python3 -m unittest discover -s tests -p "test_*.py"` passes with no Supabase
   credentials present in `.env`.
3. Post-cutover, the first live cycle posts no article that was already posted before the
   migration.
4. `grep -ri supabase` over the repo matches only the migration script, its test, this spec,
   and the cutover instructions in Redme-Oracle.md — no runtime code path.
