# Python News Bot

A news aggregation bot that scrapes local Málaga news, evaluates article relevance with AI, and posts curated summaries to a Telegram channel. Runs on a 10-minute schedule.

## How It Works

1. **Fetch** — Scrapes article links and content from the configured news source
2. **Deduplicate** — Checks the article URL against Supabase first (free, in-memory); then embeds the title with Gemini (`gemini-embedding-2`) and compares cosine similarity against stored embeddings; falls back to Jaccard on legacy rows
3. **Evaluate** — Gemini scores each article's relevance (0–10); articles below 6 are saved to Supabase (so they are not re-evaluated next cycle) and then skipped
4. **Summarize** — Gemini generates an emoji-rich, Telegram-ready summary
5. **Post** — Sends media groups (up to 9 images) or plain text to the Telegram channel
6. **Cleanup** — Daily job removes articles older than 10 days

All Gemini calls are staggered (default 6.5s minimum spacing) and respect Google's `Retry-After` / `RetryInfo.retryDelay` on HTTP 429 to stay within free-tier quotas.

## Setup

### Prerequisites

- Python 3.10+
- A Telegram bot token and target chat/channel ID
- A Google Gemini API key (used for article evaluation, summarization, **and** title embeddings)
- A [Supabase](https://supabase.com) project with an `articles` table:
  ```sql
  create table articles (
    id uuid primary key,
    title text,
    date text,
    embedding jsonb,
    url text
  );
  create unique index articles_url_idx on articles (url) where url is not null;
  ```

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
BOT_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id
NEWS_URL=https://www.malagahoy.es/malaga/
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_key

# Optional
GEMINI_MODEL=gemini-2.5-flash-lite          # default; switch to gemini-2.5-flash for higher quality
GEMINI_MIN_CALL_INTERVAL_SEC=6.5            # min seconds between Gemini calls (rate-limit guard)
LOG_LEVEL=INFO
```

## Usage

```bash
# Run the bot (loops every 10 minutes)
python main.py

# Dry run — fetch and evaluate without saving or posting
python main.py --dry-run
```

## AI Providers

Article evaluation and summarization use Gemini by default. Switch via `current_ai_provider` in [main.py](main.py):

| Provider | Model | Notes |
|---|---|---|
| `AIProvider.GEMINI` | `gemini-2.5-flash-lite` | Default; uses JSON schema validation. Free-tier: 15 RPM / 1,000 RPD |
| `AIProvider.OPENAI` | Any OpenAI-compatible | Also works with local LM Studio at `http://localhost:1234/v1` |

Deduplication embeddings use Gemini (`gemini-embedding-2`) and reuse the same `GEMINI_API_KEY`. The embedding computed during the dedup check is cached in-memory and reused on save, so each article costs at most one embed call.

### Rate-limit handling

Gemini 429s are handled in three places:
1. **Staggering** — every Gemini call (generation *and* embedding) is spaced by `GEMINI_MIN_CALL_INTERVAL_SEC` (default 6.5s).
2. **`Retry-After` parsing** — on HTTP 429, `GeminiRateLimitError` carries Google's suggested delay from the `Retry-After` header or `RetryInfo.retryDelay` body field.
3. **Job-level circuit breaker** — when a 429 fires, the current 10-minute job cycle aborts early and resumes on the next tick.

## Project Structure

```
├── main.py                  # Entry point, scheduler, job orchestration
├── fetching_data.py         # Web scraping (BeautifulSoup)
├── data_service.py          # Supabase deduplication (Gemini embeddings + cosine similarity)
├── telegram_service.py      # Telegram posting (media groups + text)
├── response_parser.py       # JSON + regex extraction from AI responses
├── requirements.txt
└── ai/
    ├── ai_service.py        # Factory: AIService.get_service(provider)
    ├── base_ai_service.py   # Abstract base (evaluate, summarize)
    ├── gemini_service.py    # Google Gemini implementation
    ├── openai_service.py    # OpenAI / LM Studio implementation
    ├── ai_prompts.py        # Prompt templates
    └── ai_provider.py       # AIProvider enum
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"        # All tests
python -m unittest tests.test_ai_services                   # AI service (evaluate + summarize)
python -m unittest tests.test_similarity                    # Cosine math + Cohere embedding + Supabase integration
python -m unittest tests.test_supabase_connection           # Live Supabase connection (requires credentials)
```

Unit tests mock all external API calls — no live credentials required for most tests.  
`test_similarity.py` runs real Gemini embedding calls when `GEMINI_API_KEY` is set; otherwise the API-dependent classes are skipped automatically.  
`test_supabase_connection.py` hits the live Supabase REST API and requires `SUPABASE_URL` and `SUPABASE_KEY`.

## Key Constants

| Constant | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity cutoff for deduplication |
| `DISTANCE_THRESHOLD` | `0.15` | `1 - SIMILARITY_THRESHOLD` |
| Scheduler interval | 10 min | How often `job()` runs |
| Cleanup age | 10 days | Max age of stored articles |
| AI retry delay | 20s base, exponential | Overridden by `Retry-After` when the server provides one (max 5 attempts) |
| Gemini call spacing | 6.5s | Min interval between Gemini calls (`GEMINI_MIN_CALL_INTERVAL_SEC`) |
| Embedding model | `gemini-embedding-2` | Google Gemini embedding model |
