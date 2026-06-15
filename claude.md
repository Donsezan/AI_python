# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A news aggregation bot that scrapes Málaga news articles, evaluates their relevance using AI, and posts curated content to a Telegram channel. The bot runs on a 10-minute scheduler.

## Commands

### Setup
```bash
source .venv/bin/activate       # Activate virtual environment
pip install -r requirements.txt
```

### Run
```bash
python3 main.py                 # Start the bot (runs indefinitely via scheduler)
python3 main.py --dry-run       # One cycle: fetch + evaluate + summarize without posting or saving, then exit
```

Dry runs log per-article results (status, score breakdown, summary, original `title` and target-language `translated_title`) to a timestamped JSON file under `dry_run_logs/` via [dry_run_logger.py](dry_run_logger.py) — used for comparing score distributions before/after prompt changes. Dry runs do not write to the seen-cache or Supabase.

### Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py"   # Run all tests
python3 -m unittest tests.test_ai_services              # Run specific test module
```

### Lint
No linter is configured. Use `flake8` or `ruff` if needed.

## Architecture

The bot runs a scheduled `job()` function in [main.py](main.py) every 10 minutes. The per-article pipeline is ordered **cheapest-first** so free checks always run before quota-consuming API calls (free-tier Gemini quotas are requests-per-day bound, not token bound):

1. **Retry pending posts** — Telegram posts that failed earlier are re-sent from the in-memory `_pending_posts` queue without new LLM calls (the summary is already paid for); after 3 failed attempts the URL is marked `post_failed` in the seen-cache
2. **Fetch list** — each scraper in `fetch_services` (the `scrapers/` package) scrapes article links from its homepage; the combined `(title, href, scraper)` list is processed together. If the `(title, href)` pairs are byte-identical to the last cleanly-completed cycle (SHA-1 hash), the whole cycle is skipped for free
3. **Skip checks (free)** — per article: URL exact-match via `is_url_seen()` against the cached `known_articles` list, then [seen_cache.py](seen_cache.py) — a persistent JSON cache (`seen_cache.json`) of terminal outcomes (`too_old`, `no_content`, `duplicate`, `post_failed`) and transient-failure attempt counters (terminal after 3 attempts). This prevents unsaveable articles from being re-processed every cycle
4. **Fetch article (free)** — `fetch_article()` returns `(soup, date_time)` or a status string: `"fetch_failed"` (HTTP/parse error, missing `<h1>` or timestamp) or `"too_old"` (older than 7 days). Failures are recorded in the seen-cache *before* any API call is spent
5. **Deduplicate (1 embed call)** — title cosine similarity via Gemini embeddings + `is_new_article_cached()` in [data_service.py](data_service.py) (threshold 0.85); falls back to Jaccard on legacy rows without embeddings. The embedding is returned and reused on save
6. **Evaluate + summarize (1 generation call)** — a single combined LLM call scores 5 dimensions (`expat_impact`, `event_weight`, `politics`, `timeliness`, `practical_utility`), produces the emoji-rich summary, **and rewrites the scraped headline into the target language** (the `title` field — passed the original headline in the request; `EVALUATION_SCHEMA` in [ai/ai_prompts.py](ai/ai_prompts.py)). The final score is the mean of all five dimensions (a `politics: 0` lowers it). Articles scoring below 6 are **saved to Supabase** (to prevent re-evaluation next cycle) and skipped. The posted Telegram message uses the translated title so it matches the summary's language; the original scraped title remains the dedup/embedding key
7. **Post** — [telegram_service.py](telegram_service.py) sends media groups (up to 9 images) or text; on failure the composed message is queued in `_pending_posts` for retry
8. **Save** — original title, URL, embedding, and the translated title (`title_translated` column) saved to Supabase after a confirmed post and appended to the in-memory `known_articles` (so a same-story second URL in the same cycle is deduplicated); the embedding is still computed from the **original** title for dedup consistency with legacy rows. If the save fails, the URL is recorded as `posted` in the seen-cache to prevent a duplicate post
9. **Cleanup** — Daily job removes Supabase entries older than 10 days; the seen-cache self-prunes entries not seen for 14 days

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
- [openai_service.py](ai/openai_service.py) — OpenAI or local LM Studio (`http://localhost:1234/v1`)
- [ai_prompts.py](ai/ai_prompts.py) — All prompt templates
- [ai_provider.py](ai/ai_provider.py) — `AIProvider` enum (`GEMINI`, `OPENAI`)

Switch providers by changing `AIProvider.GEMINI` / `AIProvider.OPENAI` in [main.py](main.py).

### Gemini Rate-Limit Handling

Free-tier quotas (`gemini-2.5-flash-lite`: 15 RPM / 1,000 RPD; `gemini-2.5-flash`: 10 RPM / 250 RPD) are easy to exhaust. [ai/gemini_service.py](ai/gemini_service.py) defends against this with:

- **Call staggering** — a class-level monotonic timestamp + lock enforces a minimum gap between *every* Gemini call (default 6.5s, configurable via `GEMINI_MIN_CALL_INTERVAL_SEC`). Keeps RPM safely under the cap.
- **`GeminiRateLimitError` on HTTP 429** — subclasses the generic `RateLimitError` from [ai/base_ai_service.py](ai/base_ai_service.py); carries the suggested delay parsed from either the `Retry-After` header or Google's `error.details[].retryDelay` JSON field.
- **`_with_retry()` in [main.py](main.py)** — when the raised exception has a `retry_after` attribute, it waits exactly that long instead of falling back to exponential backoff; also sets `_rate_limited` to abort the current job cycle early.
- **Thinking disabled** — `thinkingBudget: 0` is set on every request so Gemini 2.5 thought tokens don't starve `maxOutputTokens`.

### Key Configuration

All credentials live in `.env` (loaded via `python-dotenv`):
```
BOT_TOKEN                        # Telegram bot token
CHAT_ID                          # Target Telegram channel/chat
NEWS_URL                         # malagahoy.es source URL (malagahoy.es/malaga/)
DIARIOSUR_URL                    # Optional: diariosur.es source URL (default: diariosur.es/malaga/)
GEMINI_API_KEY                   # Google Generative AI key (used for both generation and embeddings)
SUPABASE_URL                     # Supabase project URL
SUPABASE_KEY                     # Supabase service role key
GEMINI_MODEL                     # Optional: Gemini model name (default: gemini-2.5-flash-lite)
GEMINI_MIN_CALL_INTERVAL_SEC     # Optional: min seconds between Gemini calls (default: 6.5)
LOG_LEVEL                        # Optional: logging level (default: INFO)
```

Constants in [main.py](main.py):
- `SIMILARITY_THRESHOLD = 0.85` — cosine similarity cutoff for duplicate detection
- `SCORE_THRESHOLD = 6` — minimum average score for posting
- `MAX_POST_ATTEMPTS = 3` — Telegram retry budget before a post is abandoned

### Response Parsing

[response_parser.py](response_parser.py) — `parse_evaluate_and_summarize()` strips `<think>` tags and markdown fences, parses the JSON, and returns `{'score', 'breakdown', 'summary'}` — or `None` on invalid JSON, so callers can distinguish "model failed" from "article scored low". Schema enforcement happens provider-side (`responseJsonSchema` / `json_schema` response format).

## Dependencies

See [requirements.txt](requirements.txt) for runtime dependencies:
- `beautifulsoup4` + `requests` — Web scraping
- `numpy` — Cosine similarity computation
- `python-dotenv` — Environment variable loading

Dev/test-only packages are in [requirements-dev.txt](requirements-dev.txt).

## Testing

Tests use `unittest`. Coverage is mixed mock/integration:

- [tests/test_ai_services.py](tests/test_ai_services.py) — `evaluate_and_summarize()` for both providers with mocked HTTP clients, including the `responseJsonSchema` → legacy fallback and 429/`retry_after` parsing
- [tests/test_response_parser.py](tests/test_response_parser.py) — combined-response parsing: scoring math (zeros count), fence/think-tag stripping, `None` on invalid JSON
- [tests/test_seen_cache.py](tests/test_seen_cache.py) — seen-cache skip/attempt/prune/persistence behavior (tempdir, no network)
- [tests/test_scrapers.py](tests/test_scrapers.py) — `DiarioSurScraper`/`MalagaHoyScraper` date, image, and content-scoping extraction against in-line HTML fixtures (no network)
- [tests/test_similarity.py](tests/test_similarity.py) — pure-numpy cosine math; real Gemini embedding calls (skipped when `GEMINI_API_KEY` is absent); `DataService` similarity with mocked Supabase
- [tests/test_supabase_connection.py](tests/test_supabase_connection.py) — **real Supabase integration** (CRUD round-trips); reads credentials from `.env` directly

Integration tests hit live APIs when keys are present in `.env` — be mindful of Gemini free-tier quotas when running the full suite.
