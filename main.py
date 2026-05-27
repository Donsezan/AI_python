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
_missing = [k for k, v in {'BOT_TOKEN': BOT_TOKEN, 'CHAT_ID': CHAT_ID, 'NEWS_URL': NEWS_URL, 'GEMINI_API_KEY': GEMINI_API_KEY, 'SUPABASE_URL': SUPABASE_URL, 'SUPABASE_KEY': SUPABASE_KEY}.items() if not v]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(_missing)}")

# Toggle between AI providers: AIProvider.OPENAI or AIProvider.GEMINI
current_ai_provider = AIProvider.GEMINI

# Initialize services
data_service = DataService(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY, DISTANCE_THRESHOLD=DISTANCE_THRESHOLD, gemini_api_key=GEMINI_API_KEY)
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


def job(dry_run=False):
    global _rate_limited
    _rate_limited = False
    logger.info("Fetching latest articles...")
    new_articles = fetch_service.fetch_latest_articles()
    logger.info(f"Found {len(new_articles)} new articles.")
    known_articles = data_service.fetch_recent_articles()
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
            _process_article(title, href, known_articles, dry_run=dry_run)
        except Exception as e:
            logger.error(f"[job] Article '{title}' failed: {e!r}")
        if _shutdown.wait(timeout=5):
            break
    logger.info("Job finished.")
 

def _process_article(title, href, known_articles, dry_run=False):
    logger.info(f"Processing article: {title}")
    is_new, title_embedding = data_service.is_new_article_cached(title, known_articles)
    if not is_new:
        logger.info(f"Article '{title}' already processed, skipping.")
        return

    logger.info("Fetching article metadata...")
    meta = fetch_service.fetch_article(title, href)
    if not meta:
        logger.warning("Failed to fetch article or article is too old.")
        return
    soup, date_time = meta
    main_content, images = fetch_service.parse_content(soup)

    if not main_content:
        logger.warning("Article content is missing.")
        return

    logger.info("Evaluating main content...")
    article_score = _with_retry(lambda: ai_service.evaluate_article(main_content))
    if not article_score:
        logger.warning(f"Failed to evaluate article '{title}'. Skipping.")
        return

    logger.info(f"Article score: {article_score}")
    if article_score < 6:
        logger.info(f"Article '{title}' scored {article_score:.1f}, below threshold. Skipping.")
        if not dry_run:
            data_service.save_article(title, date_time, url=href, embedding=title_embedding)
        return

    logger.info("Summarizing with emojis...")
    evaluated_content = _with_retry(lambda: ai_service.summarize_with_emojis(main_content, target_language='en'))

    if not evaluated_content or not evaluated_content.strip():
        logger.warning(f"Failed to summarize article '{title}' with emojis. Skipping.")
        return

    if dry_run:
        return

    logger.info("Posting to Telegram...")
    result_of_post = telegram_service.post_to_telegram(f"<b>{title}</b>\n\n{evaluated_content}", images, href)
    if not result_of_post:
        logger.error(f"Failed to post article '{title}' to Telegram, will retry next cycle.")
        return

    logger.info("Saving article...")
    if not data_service.save_article(title, date_time, url=href, embedding=title_embedding):
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
    if dry_run:
        sys.exit(0)

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
