import logging
import requests
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Common scraping pipeline shared by every news source.

    Subclasses supply only the source-specific pieces: date extraction, image
    extraction and (optionally) the content root. The list selector and the
    ``fetch_article`` skeleton (HTTP GET, ``<h1>`` check, age cutoff and the
    status-string contract) live here.
    """

    LINK_SELECTOR = "a[href*='/malaga/']"
    MAX_AGE_DAYS = 7

    def __init__(self, news_url, headers, language='es'):
        self.news_url = news_url
        self.headers = headers
        self.language = language

    def fetch_latest_articles(self):
        logger.info(f"Fetching latest articles from: {self.news_url}")
        try:
            resp = requests.get(self.news_url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            articles = []
            for link in soup.select(self.LINK_SELECTOR):
                href = link.get('href')
                title = link.get_text(strip=True)
                if href and title:
                    articles.append((title, href))
            articles.reverse()
            logger.info(f"Found {len(articles)} articles.")
            return articles
        except requests.RequestException as e:
            logger.error(f"Error fetching articles: {e}")
            return []

    def fetch_article(self, title, href):
        """Fetch an article page and validate its date.

        Returns ``(soup, date_time)`` on success, or a string status describing
        the failure: ``"fetch_failed"`` (HTTP/parse error or missing required
        nodes) or ``"too_old"`` (article older than ``MAX_AGE_DAYS``).
        """
        logger.info(f"Fetching article: {title}")
        try:
            resp = requests.get(href, headers=self.headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            h1 = soup.find('h1')
            if not h1:
                logger.warning(f"[fetch] No <h1> on {href}")
                return "fetch_failed"
            logger.info(f"Article title: {h1.get_text(strip=True)}")

            try:
                date_time = self._extract_date(soup)
            except (ValueError, IndexError) as e:
                logger.warning(f"[fetch] Date parse failed for {href}: {e}")
                return "fetch_failed"
            if date_time is None:
                logger.warning(f"[fetch] No date on {href}")
                return "fetch_failed"

            logger.info(f"Article date: {date_time}")
            if date_time < datetime.now() - timedelta(days=self.MAX_AGE_DAYS):
                logger.info(f"Article is older than {self.MAX_AGE_DAYS} days, skipping.")
                return "too_old"

            return soup, date_time

        except requests.RequestException as e:
            logger.error(f"[fetch] HTTP error for {href}: {e}")
            return "fetch_failed"
        except Exception as e:
            logger.error(f"[fetch] Unexpected error for {href}: {e!r}")
            return "fetch_failed"

    def parse_content(self, soup):
        """Extract text content and images from an already-fetched soup object."""
        root = self._content_root(soup)
        content = '\n'.join(p.get_text(strip=True) for p in root.find_all('p'))
        images = self._extract_images(soup)
        logger.info(f"Found {len(images)} images.")
        return content, images

    def _content_root(self, soup):
        """Element whose paragraphs make up the article body. Default: whole doc."""
        return soup

    @abstractmethod
    def _extract_date(self, soup):
        """Return a naive ``datetime`` for the article, or ``None`` if absent.

        May raise ``ValueError``/``IndexError`` on a malformed date.
        """

    @abstractmethod
    def _extract_images(self, soup):
        """Return a list of image URLs for the article."""
