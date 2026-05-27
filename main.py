import os
import sys
import signal
import threading
import random
import logging
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

from fetching_data import FetchingData
from ai.ai_service import AIService
from telegram_service import TelegramService
from data_service import DataService
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
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
NEWS_URL = os.getenv('NEWS_URL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

_missing = [k for k, v in {'BOT_TOKEN': BOT_TOKEN, 'CHAT_ID': CHAT_ID, 'NEWS_URL': NEWS_URL, 'GEMINI_API_KEY': GEMINI_API_KEY, 'SUPABASE_URL': SUPABASE_URL, 'SUPABASE_KEY': SUPABASE_KEY, 'COHERE_API_KEY': COHERE_API_KEY}.items() if not v]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(_missing)}")

# Toggle between AI providers: AIProvider.OPENAI or AIProvider.GEMINI
current_ai_provider = AIProvider.GEMINI

# Initialize services
data_service = DataService(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY, DISTANCE_THRESHOLD=DISTANCE_THRESHOLD, cohere_api_key=COHERE_API_KEY)
fetch_service = FetchingData(NEWS_URL, HEADERS)
telegram_service = TelegramService(BOT_TOKEN, CHAT_ID)
ai_service = AIService.get_service(provider=current_ai_provider, gemini_api_key=GEMINI_API_KEY)

_shutdown = threading.Event()
_rate_limited = False  # circuit breaker: True when Gemini returns 429


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
            logger.warning(f"LLM error (attempt {attempt}/{retries}): {e!r}")
            if attempt < retries:
                sleep = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.info(f"Retrying in {sleep:.1f}s...")
                if _shutdown.wait(timeout=sleep):
                    return None
    return None


def job(dry_run=False):
    global _rate_limited
    _rate_limited = False
    dry_run_log = DryRunLogger() if dry_run else None
    logger.info("Fetching latest articles...")
    new_articles = fetch_service.fetch_latest_articles()
    logger.info(f"Found {len(new_articles)} new articles.")
    known_articles = data_service.fetch_recent_articles()
    try:
        for title, href in new_articles:
            if _shutdown.is_set():
                break
            if _rate_limited:
                logger.warning("Gemini rate-limited — aborting job cycle early, will retry next run.")
                break
            if not href or len(title.strip()) < 20:
                logger.debug(f"Skipping invalid article entry: '{title}'")
                continue
            if data_service.is_url_seen(href, known_articles):
                logger.info(f"Skipping already-seen URL: {href}")
                continue
            try:
                _process_article(title, href, known_articles, dry_run=dry_run, dry_run_log=dry_run_log)
            except Exception as e:
                logger.error(f"[job] Article '{title}' failed: {e!r}")
                if dry_run_log:
                    dry_run_log.record(title=title, url=href, status="exception", error=repr(e))
            if _shutdown.wait(timeout=5):
                break
    finally:
        if dry_run_log:
            dry_run_log.close()
    logger.info("Job finished.")


def _process_article(title, href, known_articles, dry_run=False, dry_run_log=None):
    logger.info(f"Processing article: {title}")
    if not data_service.is_new_article_cached(title, known_articles):
        logger.info(f"Article '{title}' already processed, skipping.")
        if dry_run_log:
            dry_run_log.record(title=title, url=href, status="duplicate_cached")
        return

    logger.info("Fetching article metadata...")
    meta = fetch_service.fetch_article(title, href)
    if not meta:
        logger.warning("Failed to fetch article or article is too old.")
        if dry_run_log:
            dry_run_log.record(title=title, url=href, status="fetch_failed_or_too_old")
        return
    soup, date_time = meta
    main_content, images = fetch_service.parse_content(soup)

    if not main_content:
        logger.warning("Article content is missing.")
        if dry_run_log:
            dry_run_log.record(title=title, url=href, status="no_content")
        return

    logger.info("Evaluating main content...")
    eval_result = _with_retry(lambda: ai_service.evaluate_article(main_content, fetch_service.language))
    article_score = eval_result["score"] if eval_result else None
    breakdown = eval_result["breakdown"] if eval_result else None
    if not article_score:
        logger.warning(f"Failed to evaluate article '{title}'. Skipping.")
        if dry_run_log:
            dry_run_log.record(
                title=title, url=href, status="evaluate_failed",
                language=fetch_service.language, article_chars=len(main_content),
            )
        return

    logger.info(f"Article score: {article_score}")
    if article_score < 6:
        logger.info(f"Article '{title}' scored {article_score:.1f}, below threshold. Skipping.")
        if dry_run_log:
            dry_run_log.record(
                title=title, url=href, status="below_threshold", score=article_score, breakdown=breakdown,
                language=fetch_service.language, article_chars=len(main_content),
            )
        if not dry_run:
            data_service.save_article(title, date_time, url=href)
        return

    logger.info("Summarizing with emojis...")
    evaluated_content = _with_retry(lambda: ai_service.summarize_with_emojis(main_content, target_language='en', source_language=fetch_service.language))

    if not evaluated_content or not evaluated_content.strip():
        logger.warning(f"Failed to summarize article '{title}' with emojis. Skipping.")
        if dry_run_log:
            dry_run_log.record(
                title=title, url=href, status="summary_failed", score=article_score, breakdown=breakdown,
                language=fetch_service.language, article_chars=len(main_content),
            )
        return

    if dry_run:
        if dry_run_log:
            dry_run_log.record(
                title=title, url=href, status="evaluated_above_threshold",
                score=article_score, breakdown=breakdown, summary=evaluated_content,
                language=fetch_service.language, article_chars=len(main_content),
            )
        return

    logger.info("Posting to Telegram...")
    result_of_post = telegram_service.post_to_telegram(f"<b>{title}</b>\n\n{evaluated_content}", images, href)
    if not result_of_post:
        logger.error(f"Failed to post article '{title}' to Telegram, will retry next cycle.")
        return

    logger.info("Saving article...")
    if not data_service.save_article(title, date_time, url=href):
        logger.warning(f"Posted '{title}' but failed to save — may duplicate next run.")

    if _shutdown.wait(timeout=10):
        return


def _handle_signal(signum, _frame):
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown.set()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    dry_run = '--dry-run' in sys.argv
    job(dry_run=dry_run)
    # if dry_run:
    #     sys.exit(0)

    next_job = datetime.now() + timedelta(minutes=10)
    last_cleanup_day = date.today()

    while not _shutdown.is_set():
        if _shutdown.wait(timeout=60):
            break
        now = datetime.now()
        logger.info(f"Scheduler tick at {now.strftime('%Y-%m-%d %H:%M:%S')}")

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
