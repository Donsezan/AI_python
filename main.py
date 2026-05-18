import os
import sys
import signal
import threading
import time
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


def _with_retry(fn, retries=3, base_delay=10):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt}/{retries}): {e!r}")
            if attempt < retries:
                sleep = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.info(f"Retrying in {sleep:.1f}s...")
                if _shutdown.wait(timeout=sleep):
                    return None
    return None


def job(dry_run=False):
    logger.info("Fetching latest articles...")
    new_articles = fetch_service.fetch_latest_articles()
    logger.info(f"Found {len(new_articles)} new articles.")
    known_articles = data_service.fetch_recent_articles()
    for title, href in new_articles:
        if _shutdown.is_set():
            break
        if data_service.is_url_seen(href, known_articles):
            logger.info(f"Skipping already-seen URL: {href}")
            continue
        try:
            _process_article(title, href, known_articles, dry_run=dry_run)
        except Exception as e:
            logger.error(f"[job] Article '{title}' failed: {e!r}")
    logger.info("Job finished.")


def _process_article(title, href, known_articles, dry_run=False):
    logger.info(f"Processing article: {title}")
    if not data_service.is_new_article_cached(title, known_articles):
        logger.info(f"Article '{title}' already processed, skipping.")
        return

    logger.info("Fetching and summarizing article...")
    result = fetch_service.fetch_and_summarize(title, href)
    if not result:
        logger.warning("Failed to fetch and summarize article.")
        return

    main_content, images, date_time = result
    if not main_content or not date_time:
        logger.warning("Article content or date/time is missing.")
        return

    logger.info("Evaluating article...")
    article_score = _with_retry(lambda: ai_service.evaluate_article(title))
    if not article_score:
        logger.warning(f"Failed to evaluate article '{title}'. Skipping.")
        return

    logger.info(f"Article score: {article_score}")
    if article_score < 6:
        logger.info(f"Article '{title}' scored {article_score:.1f}, below threshold. Skipping.")
        if not dry_run:
            data_service.save_article(title, date_time, url=href)
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
