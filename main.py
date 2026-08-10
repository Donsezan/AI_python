import os
import sys
import signal
import hashlib
import threading
import random
import logging
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

from scrapers import MalagaHoyScraper, DiarioSurScraper
from ai.ai_service import AIService
from telegram_service import TelegramService
from data_service import DataService
from seen_cache import SeenCache
from ai.ai_provider import AIProvider
from ai.base_ai_service import RateLimitError
from dry_run_logger import DryRunLogger

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0"}
SIMILARITY_THRESHOLD = 0.85
DISTANCE_THRESHOLD = 1 - SIMILARITY_THRESHOLD
SCORE_THRESHOLD = 6
MAX_POST_ATTEMPTS = 3
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
NEWS_URL = os.getenv('NEWS_URL')
DIARIOSUR_URL = os.getenv('DIARIOSUR_URL', 'https://www.diariosur.es/malaga/')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DB_PATH = os.getenv('DB_PATH', 'articles.db')
_missing = [k for k, v in {'BOT_TOKEN': BOT_TOKEN, 'CHAT_ID': CHAT_ID, 'NEWS_URL': NEWS_URL, 'GEMINI_API_KEY': GEMINI_API_KEY}.items() if not v]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(_missing)}")

# Toggle between AI providers: AIProvider.OPENAI or AIProvider.GEMINI
current_ai_provider = AIProvider.GEMINI

# Initialize services
data_service = DataService(db_path=DB_PATH, DISTANCE_THRESHOLD=DISTANCE_THRESHOLD, gemini_api_key=GEMINI_API_KEY)
fetch_services = [
    MalagaHoyScraper(NEWS_URL, HEADERS),
    DiarioSurScraper(DIARIOSUR_URL, HEADERS),
]
telegram_service = TelegramService(BOT_TOKEN, CHAT_ID)
ai_service = AIService.get_service(provider=current_ai_provider, gemini_api_key=GEMINI_API_KEY)
seen_cache = SeenCache()

_shutdown = threading.Event()
_rate_limited = False  # circuit breaker: True when Gemini returns 429
_last_articles_hash = None  # homepage fingerprint of the last cleanly-completed cycle
# Telegram-post payloads waiting for retry: href -> {message, images, title,
# date_time, embedding, translated_title, attempts}. The LLM work is already
# paid for, so a failed post must never trigger a re-evaluation.
_pending_posts = {}


def _with_retry(fn, retries=5, base_delay=20):
    global _rate_limited
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except RateLimitError as e:
            _rate_limited = True
            logger.warning(f"LLM rate-limited (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                sleep = (e.retry_after + 1) if e.retry_after else base_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep:.1f}s (provider-suggested)...")
                if _shutdown.wait(timeout=sleep):
                    return None
        except Exception as e:
            retry_after = getattr(e, 'retry_after', None)
            if retry_after is not None or '429' in str(e):
                _rate_limited = True
            logger.warning(f"LLM error (attempt {attempt}/{retries}): {e!r}")
            if attempt < retries:
                if retry_after is not None:
                    sleep = retry_after + random.uniform(0, 1)
                else:
                    sleep = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.info(f"Retrying in {sleep:.1f}s...")
                if _shutdown.wait(timeout=sleep):
                    return None
    return None


def _retry_pending_posts():
    """Re-send posts that failed on Telegram's side — no LLM calls involved."""
    for href in list(_pending_posts):
        if _shutdown.is_set():
            return
        pending = _pending_posts[href]
        attempt = pending["attempts"] + 1
        logger.info(f"Retrying Telegram post for '{pending['title']}' (attempt {attempt}/{MAX_POST_ATTEMPTS})")
        if telegram_service.post_to_telegram(pending["message"], pending["images"], href):
            _finalize_posted(pending["title"], pending["date_time"], href, pending["embedding"],
                             translated_title=pending.get("translated_title"))
            del _pending_posts[href]
        else:
            pending["attempts"] = attempt
            if attempt >= MAX_POST_ATTEMPTS:
                logger.error(f"Giving up on posting '{pending['title']}' after {MAX_POST_ATTEMPTS} attempts.")
                seen_cache.record_terminal(href, "post_failed")
                del _pending_posts[href]


def _finalize_posted(title, date_time, href, embedding, known_articles=None, translated_title=None):
    """Persist a successfully posted article; fall back to the seen-cache so a
    database hiccup can't cause a duplicate post next cycle."""
    if data_service.save_article(title, date_time, url=href, embedding=embedding, translated_title=translated_title):
        if known_articles is not None:
            known_articles.append({"title": title, "url": href, "embedding": embedding})
    else:
        logger.warning(f"Posted '{title}' but failed to save — recording in seen-cache to prevent a duplicate post.")
        seen_cache.record_terminal(href, "posted")


def job(dry_run=False):
    global _rate_limited, _last_articles_hash
    _rate_limited = False
    dry_run_log = DryRunLogger() if dry_run else None

    if not dry_run:
        _retry_pending_posts()

    logger.info("Fetching latest articles...")
    new_articles = []
    for scraper in fetch_services:
        for title, href in scraper.fetch_latest_articles():
            if href and len(title.strip()) >= 20:
                new_articles.append((title, href, scraper))
    logger.info(f"Found {len(new_articles)} candidate articles.")
    if not new_articles:
        if dry_run_log:
            dry_run_log.close()
        return

    # Free short-circuit: if the homepage shows exactly the same articles as the
    # last fully-completed cycle, there is nothing new to do.
    articles_hash = hashlib.sha1(
        repr([(t, h) for t, h, _ in new_articles]).encode("utf-8")
    ).hexdigest()
    if not dry_run and articles_hash == _last_articles_hash:
        logger.info("Homepage unchanged since last completed cycle — skipping.")
        return

    known_articles = data_service.fetch_recent_articles()
    completed = True
    try:
        for title, href, scraper in new_articles:
            if _shutdown.is_set():
                completed = False
                break
            if _rate_limited:
                logger.warning("Gemini rate-limited — aborting job cycle early, will retry next run.")
                completed = False
                break
            if data_service.is_url_seen(href, known_articles):
                logger.debug(f"Skipping already-seen URL: {href}")
                continue
            if seen_cache.should_skip(href):
                logger.debug(f"Skipping cached terminal URL: {href}")
                continue
            try:
                _process_article(title, href, scraper, known_articles, dry_run=dry_run, dry_run_log=dry_run_log)
            except Exception as e:
                logger.error(f"[job] Article '{title}' failed: {e!r}")
                if dry_run_log:
                    dry_run_log.record(title=title, url=href, status="exception", error=repr(e))
                elif not _rate_limited:
                    seen_cache.record_attempt(href, "exception")
            if _shutdown.wait(timeout=5):
                completed = False
                break
    finally:
        if dry_run_log:
            dry_run_log.close()
        seen_cache.flush_if_dirty()
    if completed and not _rate_limited and not dry_run:
        _last_articles_hash = articles_hash
    logger.info("Job finished.")


def _process_article(title, href, scraper, known_articles, dry_run=False, dry_run_log=None):
    """Pipeline ordered cheapest-first: free HTTP/date/content checks, then one
    embedding call for dedup, then a single combined evaluate+summarize call."""
    logger.info(f"Processing article: {title}")
    logger.info("Fetching article metadata...")
    meta = scraper.fetch_article(title, href)
    if isinstance(meta, str):
        if meta == "too_old":
            logger.info("Article is too old, skipping.")
            if not dry_run:
                seen_cache.record_terminal(href, "too_old")
        else:
            logger.warning("Failed to fetch article.")
            if not dry_run:
                seen_cache.record_attempt(href, meta)
        if dry_run_log:
            dry_run_log.record(title=title, url=href, status=meta)
        return
    soup, date_time = meta
    main_content, images = scraper.parse_content(soup)

    if not main_content:
        logger.warning("Article content is missing.")
        if not dry_run:
            seen_cache.record_terminal(href, "no_content")
        if dry_run_log:
            dry_run_log.record(title=title, url=href, status="no_content")
        return

    is_new, title_embedding = data_service.is_new_article_cached(title, known_articles)
    if not is_new:
        logger.info(f"Article '{title}' already processed, skipping.")
        if not dry_run:
            seen_cache.record_terminal(href, "duplicate")
        if dry_run_log:
            dry_run_log.record(title=title, url=href, status="duplicate_cached")
        return

    logger.info("Evaluating and summarizing article...")
    result = _with_retry(lambda: ai_service.evaluate_and_summarize(
        main_content, title, source_language=scraper.language, target_language='en'))
    if result is None or not result["summary"]:
        status = "evaluate_failed" if result is None else "summary_failed"
        logger.warning(f"LLM processing failed ({status}) for '{title}'. Skipping.")
        if not dry_run and not _rate_limited:
            seen_cache.record_attempt(href, status)
        if dry_run_log:
            dry_run_log.record(
                title=title, url=href, status=status,
                language=scraper.language, article_chars=len(main_content),
            )
        return
    article_score = result["score"]
    breakdown = result["breakdown"]
    summary = result["summary"]
    # Headline rewritten in the target language; fall back to the original
    # scraped headline if the model omitted it, so the post is never title-less.
    translated_title = result["title"] or title

    logger.info(f"Article score: {article_score:.1f}")
    if article_score < SCORE_THRESHOLD:
        logger.info(f"Article '{title}' scored {article_score:.1f}, below threshold. Skipping.")
        if dry_run_log:
            dry_run_log.record(
                title=title, url=href, status="below_threshold", score=article_score, breakdown=breakdown,
                translated_title=translated_title,
                language=scraper.language, article_chars=len(main_content),
            )
        if not dry_run:
            if data_service.save_article(title, date_time, url=href, embedding=title_embedding,
                                         translated_title=translated_title):
                known_articles.append({"title": title, "url": href, "embedding": title_embedding})
            else:
                seen_cache.record_attempt(href, "save_failed")
        return

    if dry_run:
        if dry_run_log:
            dry_run_log.record(
                title=title, url=href, status="evaluated_above_threshold",
                score=article_score, breakdown=breakdown, summary=summary,
                translated_title=translated_title,
                language=scraper.language, article_chars=len(main_content),
            )
        return

    logger.info("Posting to Telegram...")
    message = f"<b>{translated_title}</b>\n\n{summary}"
    if not telegram_service.post_to_telegram(message, images, href):
        logger.error(f"Failed to post article '{title}' to Telegram — queued for retry without re-evaluation.")
        _pending_posts[href] = {
            "message": message, "images": images, "title": title,
            "date_time": date_time, "embedding": title_embedding,
            "translated_title": translated_title, "attempts": 1,
        }
        return

    logger.info("Saving article...")
    _finalize_posted(title, date_time, href, title_embedding, known_articles, translated_title)


def _handle_signal(signum, _frame):
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown.set()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    dry_run = '--dry-run' in sys.argv
    job(dry_run=dry_run)
    if dry_run:
        sys.exit(0)

    next_job = datetime.now() + timedelta(minutes=10)
    last_cleanup_day = date.today()

    while not _shutdown.is_set():
        if _shutdown.wait(timeout=60):
            break
        now = datetime.now()
        logger.debug(f"Scheduler tick at {now.strftime('%Y-%m-%d %H:%M:%S')}")

        if now.date() > last_cleanup_day:
            try:
                data_service.cleanup_old_articles(max_age_days=10)
            except Exception as e:
                logger.error(f"[cleanup] {e!r}")
            last_cleanup_day = now.date()

        if now >= next_job:
            try:
                job(dry_run=False)
            except Exception as e:
                logger.error(f"[job] Top-level failure: {e!r}")
            next_job = datetime.now() + timedelta(minutes=10)

    logger.info("Shutdown complete.")
