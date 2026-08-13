# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A news aggregation bot that scrapes Málaga news articles, evaluates their relevance using AI, and posts curated content to a Telegram channel. The bot runs on a 10-minute scheduler.

## Commands

### Setup
```bash
source .venv/bin/activate       # Linux/VPS; on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
python3 main.py                 # Start the bot (runs indefinitely via scheduler)
python3 main.py --dry-run       # One cycle: fetch + evaluate + summarize without posting or saving, then exit
```

`main.py` validates `BOT_TOKEN`, `CHAT_ID`, `NEWS_URL` and `GEMINI_API_KEY` **at import time** and raises `EnvironmentError` if any is missing — so `main` cannot be imported without a complete `.env`. This is why no test imports it; test helpers parse `.env` themselves. Storage needs no credentials: `DB_PATH` defaults to `articles.db` and the file is created on first use.

Dry runs log per-article results (status, score breakdown, summary, original `title` and target-language `translated_title`) to a timestamped JSON file under `dry_run_logs/` via [dry_run_logger.py](dry_run_logger.py) — used for comparing score distributions before/after prompt changes. Dry runs do not write to the seen-cache or the database.

### Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py"   # Run all tests
python3 -m unittest tests.test_ai_services             # Run specific test module
python3 -m unittest tests.test_seen_cache.TestSeenCache.test_terminal_status_skips   # Single test method
```

### Lint
No linter is configured. Use `flake8` or `ruff` if needed.

## Architecture

The bot runs a `job()` function in [main.py](main.py) every 10 minutes. There is no scheduler library — `__main__` runs `job()` once immediately, then loops on a 60s `_shutdown.wait()` tick that fires `job()` when the next-run timestamp passes and runs the database cleanup on each date rollover. `SIGINT`/`SIGTERM` set the `_shutdown` event, which every wait and the per-article loop check, so the process drains rather than dying mid-article.

The per-article pipeline is ordered **cheapest-first** so free checks always run before quota-consuming API calls (free-tier Gemini quotas are requests-per-day bound, not token bound):

1. **Retry pending posts** — Telegram posts that failed earlier are re-sent from the in-memory `_pending_posts` queue without new LLM calls (the summary is already paid for); after 3 failed attempts the URL is marked `post_failed` in the seen-cache
2. **Fetch list** — each scraper in `fetch_services` (the `scrapers/` package) scrapes article links from its homepage; the combined `(title, href, scraper)` list is processed together. Link texts shorter than 20 characters are dropped (nav/section links), so a genuinely short headline will silently never be picked up. If the `(title, href)` pairs are byte-identical to the last cleanly-completed cycle (SHA-1 hash), the whole cycle is skipped for free
3. **Skip checks (free)** — per article: URL exact-match via `is_url_seen()` against the cached `known_articles` list, then [seen_cache.py](seen_cache.py) — a persistent JSON cache (`seen_cache.json`) of outcomes and transient-failure attempt counters. Two separate mechanisms make an entry skippable: a status in `TERMINAL_STATUSES` (`too_old`, `no_content`, `post_failed`, `failed`), **or** an attempt count that has reached `max_attempts` (3). `record_terminal()` uses the second mechanism — it writes the status *and* pins attempts to the max — which is why `duplicate` and `posted` skip correctly despite not appearing in `TERMINAL_STATUSES`. This prevents unsaveable articles from being re-processed every cycle
4. **Fetch article (free)** — `fetch_article()` returns `(soup, date_time)` or a status string: `"fetch_failed"` (HTTP/parse error, missing `<h1>` or timestamp) or `"too_old"` (older than 7 days). Failures are recorded in the seen-cache *before* any API call is spent
5. **Deduplicate (1 embed call)** — title cosine similarity via the `gemini-embedding-2` endpoint + `is_new_article_cached()` in [data_service.py](data_service.py) (threshold 0.85); falls back to Jaccard word-overlap on legacy rows without embeddings, and also whenever the embed call itself fails. The embedding is returned and reused on save
6. **Evaluate + summarize (1 generation call)** — a single combined LLM call scores 5 dimensions (`expat_impact`, `event_weight`, `politics`, `timeliness`, `practical_utility`), produces the emoji-rich summary, **and rewrites the scraped headline into the target language** (the `title` field — passed the original headline in the request; `EVALUATION_SCHEMA` in [ai/ai_prompts.py](ai/ai_prompts.py)). The final score is the mean of all five dimensions (a `politics: 0` lowers it). Articles scoring below 6 are **saved to the database** (to prevent re-evaluation next cycle) and skipped. The posted Telegram message uses the translated title so it matches the summary's language; the original scraped title remains the dedup/embedding key
7. **Post** — [telegram_service.py](telegram_service.py) sends media groups (up to 9 images) or text; on failure the composed message is queued in `_pending_posts` for retry
8. **Save** — original title, URL, embedding, and the translated title (`title_translated` column) saved to SQLite after a confirmed post and appended to the in-memory `known_articles` (so a same-story second URL in the same cycle is deduplicated); the embedding is still computed from the **original** title for dedup consistency with legacy rows. If the save fails, the URL is recorded as `posted` in the seen-cache to prevent a duplicate post
9. **Cleanup** — Daily job removes database entries older than 10 days; the seen-cache self-prunes entries not seen for 14 days

### Storage (`sqlite_store.py` + `data_service.py`)

Articles live in a local SQLite file (`DB_PATH`, default `articles.db`) next to `seen_cache.json`. There is no database server and no network call in the storage path — a hosted database was dropped because its free-tier egress cap was being blown by re-reading every embedding on all 144 cycles/day.

The two modules split along a policy/persistence line:

- [sqlite_store.py](sqlite_store.py) — **storage only.** Schema bootstrap on first connect (no manual DDL step), embedding BLOB encode/decode, and raw CRUD: `fetch_all()`, `insert()`, `delete_older_than()`. Knows nothing about similarity or thresholds. Connections are short-lived (one per operation, via a context manager), which sidesteps `check_same_thread` and holds no long-running locks; pragmas are `journal_mode=WAL` + `synchronous=NORMAL`
- [data_service.py](data_service.py) — **dedup policy.** `_embed`, `_cosine`, `_jaccard` and the new-vs-duplicate decision, delegating all persistence to `SqliteStore`. Constructs its own store from `db_path`; tests point it at a tempdir file rather than injecting a double

Embeddings are stored as **float32 little-endian BLOBs** (~12 KB/row vs ~60 KB as a JSON array), decoded with `np.frombuffer` — dimension-agnostic, inferred from byte length. Nothing in the codebase hardcodes the embedding dimension and nothing should. The float64→float32 precision loss is immaterial at a 0.85 cosine threshold. A `NULL` embedding stays `NULL` and still triggers the Jaccard fallback.

Two things `main.py` depends on and that must not regress:

- **Failures are logged and swallowed**, returning `[]` / `False` / `0` — never raised. A `False` from `save_article` is what triggers `record_terminal(href, "posted")`, the guard against duplicate posts. `insert()` therefore uses a plain `INSERT`, not `INSERT OR IGNORE`, so a duplicate id or url surfaces as `False` instead of silently reporting success
- **`fetch_all()` returns rows with numpy embeddings**, while `main.py` appends in-cycle results as plain dicts with list embeddings. `_cosine` calls `np.array()` on both operands, so the mixed types work without special-casing. (`np.frombuffer` returns a read-only array; `np.array()` copies, so nothing mutates it.)

`date` is an ISO-8601 **text** column and `delete_older_than` compares it lexicographically, which is chronological for a consistent ISO format.

### Multi-source scraping (`scrapers/`)

Multiple news sources are supported via a `BaseScraper` ABC with per-source subclasses:
- [base_scraper.py](scrapers/base_scraper.py) — shared pipeline: `fetch_latest_articles()` (uses `LINK_SELECTOR`), the `fetch_article()` skeleton (HTTP GET, `<h1>` check, `MAX_AGE_DAYS` `"too_old"` cutoff, `"fetch_failed"`/`(soup, datetime)` contract), and `parse_content()`. Subclasses override the abstract hooks `_extract_date()` / `_extract_images()` and optionally `_content_root()`
- [malagahoy_scraper.py](scrapers/malagahoy_scraper.py) — `MalagaHoyScraper`: Spanish-text dates from `<p class="timestamp-atom">`, `<source srcset>` images
- [diariosur_scraper.py](scrapers/diariosur_scraper.py) — `DiarioSurScraper`: ISO dates from `<meta article:published_time>` / `<time datetime>` (tz stripped to stay naive), body scoped to `<main>`, `og:image` + `<main>` images (author thumbnails filtered)

Add a source by writing a new `BaseScraper` subclass and appending it to `fetch_services` in [main.py](main.py). Dedup (`is_url_seen`, `seen_cache`, `is_new_article_cached`) is source-agnostic (URL/embedding keyed).

### AI Provider Abstraction (`ai/`)

Factory pattern with pluggable providers:
- [ai_service.py](ai/ai_service.py) — `AIService.get_service(provider)` factory
- [base_ai_service.py](ai/base_ai_service.py) — Abstract base with a single `evaluate_and_summarize()` method (one request per article); also defines the generic `RateLimitError` (carries `retry_after` seconds)
- [gemini_service.py](ai/gemini_service.py) — Google Gemini (configurable via `GEMINI_MODEL` env var, defaults to `gemini-2.5-flash-lite`). Uses the standard-JSON-Schema `responseJsonSchema` field for structured output; if the API rejects it (HTTP 400), automatically falls back process-wide to the legacy OpenAPI-subset `responseSchema` (sanitized via `_sanitize_schema`)
- [openai_service.py](ai/openai_service.py) — despite the name, this is **local LM Studio only**: the endpoint (`http://localhost:1234/v1/chat/completions`), the auth header (`Bearer lm-studio`) and the model (`microsoft/phi-4-reasoning-plus`) are all hardcoded, and `get_service()` passes it no key. Pointing it at the real OpenAI API means adding an API-key/base-URL path first
- [ai_prompts.py](ai/ai_prompts.py) — All prompt templates
- [ai_provider.py](ai/ai_provider.py) — `AIProvider` enum (`GEMINI`, `OPENAI`)

Switch providers by changing `AIProvider.GEMINI` / `AIProvider.OPENAI` in [main.py](main.py).

### Gemini Rate-Limit Handling

Free-tier quotas (`gemini-2.5-flash-lite`: 15 RPM / 1,000 RPD; `gemini-2.5-flash`: 10 RPM / 250 RPD) are easy to exhaust. [ai/gemini_service.py](ai/gemini_service.py) defends against this with:

- **Call staggering** — a class-level monotonic timestamp + lock enforces a minimum gap between every *generation* call (default 6.5s, configurable via `GEMINI_MIN_CALL_INTERVAL_SEC`). Keeps RPM safely under the cap.
- **`GeminiRateLimitError` on HTTP 429** — subclasses the generic `RateLimitError` from [ai/base_ai_service.py](ai/base_ai_service.py); carries the suggested delay parsed from either the `Retry-After` header or Google's `error.details[].retryDelay` JSON field.
- **`_with_retry()` in [main.py](main.py)** — when the raised exception has a `retry_after` attribute, it waits exactly that long instead of falling back to exponential backoff; also sets `_rate_limited` to abort the current job cycle early.
- **Thinking disabled** — `thinkingBudget: 0` is set on every request so Gemini 2.5 thought tokens don't starve `maxOutputTokens`.

**Embeddings are outside all of the above.** `DataService._embed()` in [data_service.py](data_service.py) posts directly to the `gemini-embedding-2` endpoint with its own private retry loop (3 attempts, blocking `time.sleep` at 20s/40s). It does not take the `GeminiService` stagger lock, does not raise `RateLimitError`, and therefore never trips the `_rate_limited` circuit breaker — an embedding 429 just exhausts its retries, logs a warning, and silently degrades that article's dedup to Jaccard. So the real per-article Gemini request count is 2 (one embed + one generation), and only one of them is rate-limit-aware. Keep this in mind when tuning `GEMINI_MIN_CALL_INTERVAL_SEC` against the RPM cap. (README.md claims embedding calls are staggered — they are not.)

### Key Configuration

All credentials live in `.env` (loaded via `python-dotenv`):
```
BOT_TOKEN                        # Telegram bot token
CHAT_ID                          # Target Telegram channel/chat
NEWS_URL                         # malagahoy.es source URL (malagahoy.es/malaga/)
DIARIOSUR_URL                    # Optional: diariosur.es source URL (default: diariosur.es/malaga/)
GEMINI_API_KEY                   # Google Generative AI key (used for both generation and embeddings)
DB_PATH                          # Optional: SQLite article store path (default: articles.db)
GEMINI_MODEL                     # Optional: Gemini model name (default: gemini-2.5-flash-lite)
GEMINI_MIN_CALL_INTERVAL_SEC     # Optional: min seconds between Gemini calls (default: 6.5)
LOG_LEVEL                        # Optional: logging level (default: INFO)
```

Constants in [main.py](main.py):
- `SIMILARITY_THRESHOLD = 0.85` — cosine similarity cutoff for duplicate detection
- `SCORE_THRESHOLD = 6` — minimum average score for posting
- `MAX_POST_ATTEMPTS = 3` — Telegram retry budget before a post is abandoned

### Response Parsing

[response_parser.py](response_parser.py) — `parse_evaluate_and_summarize()` strips `<think>` tags and markdown fences, parses the JSON, and returns `{'score', 'breakdown', 'summary', 'title'}` (`title` = the target-language headline, `""` when the model omitted it; `main.py` falls back to the original scraped headline) — or `None` on invalid JSON, so callers can distinguish "model failed" from "article scored low". Missing score dimensions default to `0` rather than erroring, which drags the mean down; a malformed response that still parses as JSON therefore looks like a low-scoring article. Schema enforcement happens provider-side (`responseJsonSchema` / `json_schema` response format).

## Dependencies

See [requirements.txt](requirements.txt) for runtime dependencies:
- `beautifulsoup4` + `requests` — Web scraping
- `numpy` — Cosine similarity computation
- `python-dotenv` — Environment variable loading

`requirements.txt` also pins `google-generativeai`, `openai` and `python-telegram-bot`, but **none of them are imported anywhere** — every Gemini, LM Studio and Telegram call is hand-rolled `requests`, and storage is stdlib `sqlite3`. Don't reach for the SDKs assuming they're already in use; match the existing raw-HTTP style instead.

Dev/test-only packages are in [requirements-dev.txt](requirements-dev.txt).

## Testing

Tests use `unittest`. Coverage is mixed mock/integration:

- [tests/test_ai_services.py](tests/test_ai_services.py) — `evaluate_and_summarize()` for both providers with mocked HTTP clients, including the `responseJsonSchema` → legacy fallback and 429/`retry_after` parsing
- [tests/test_response_parser.py](tests/test_response_parser.py) — combined-response parsing: scoring math (zeros count), fence/think-tag stripping, `None` on invalid JSON
- [tests/test_seen_cache.py](tests/test_seen_cache.py) — seen-cache skip/attempt/prune/persistence behavior (tempdir, no network)
- [tests/test_scrapers.py](tests/test_scrapers.py) — `DiarioSurScraper`/`MalagaHoyScraper` date, image, and content-scoping extraction against in-line HTML fixtures (no network)
- [tests/test_similarity.py](tests/test_similarity.py) — pure-numpy cosine math; `DataService` dedup against a seeded temp-directory database (the `_embed` network boundary stubbed, everything else real); real Gemini embedding calls skipped when `GEMINI_API_KEY` is absent
- [tests/test_sqlite_store.py](tests/test_sqlite_store.py) — `SqliteStore` CRUD, float32 BLOB fidelity, NULL embeddings, unique-index rejection, retention cutoff boundary, schema bootstrap (tempdir, no network)

Storage tests need no credentials and hit no network. Integration tests hit live APIs when keys are present in `.env` — be mindful of Gemini free-tier quotas when running the full suite.

Each test module prepends the repo root to `sys.path` itself, so tests run from the repo root without a package install. No test imports `main` (it would fail on the env-var check).

## Deployment

[Redme-Oracle.md](Redme-Oracle.md) documents the production target: an Oracle Cloud Ubuntu 24.04 VM running the bot as a `newsbot` systemd unit (`Restart=on-failure`, `MemoryMax=512M`, logs via `journalctl -u newsbot -f`). Code changes ship via `git pull` + `systemctl restart`.

## Known doc drift

[README.md](README.md) is partly stale and should not be trusted over the source: it lists a `fetching_data.py` that no longer exists (replaced by `scrapers/`) and claims embedding calls are staggered (they aren't — see above).
