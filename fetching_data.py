import re
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FetchingData:
    def __init__(self, news_url, headers):
        self.news_url = news_url
        self.headers = headers

    def fetch_latest_articles(self):
        logger.info(f"Fetching latest articles from: {self.news_url}")
        try:
            resp = requests.get(self.news_url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            articles = []
            for link in soup.select("a[href*='/malaga/']"):
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

    def _parse_spanish_date(self, date_text):
        month_mapping = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12',
        }
        parts = date_text.strip().split('\n')
        date_string = next(
            (p for p in parts if any(m in p.lower() for m in month_mapping)),
            None,
        )
        if date_string is None:
            raise ValueError(f"No date found in: {date_text!r}")
        for month_name, month_number in month_mapping.items():
            if month_name in date_string:
                date_string = date_string.replace(month_name, month_number)
                break
        date_string = date_string.replace(" ", "")
        return datetime.strptime(date_string, '%dde%m%Y-%H:%M')

    def _extract_images(self, soup):
        main_colleft = soup.find('main', id='content-body')
        source_images = []
        if main_colleft:
            source_images = [
                source['srcset'] for source in main_colleft.find_all('source')
                if not source.find_parent(class_='media-atom') and source.get('srcset')
            ]

        img_tag = soup.find('img')
        img_url = img_tag.get('src') if img_tag else None
        all_images = source_images + ([img_url] if img_url else [])

        max_resolution = 0
        for url in all_images:
            match = re.search(r'_(\d+)w_', url)
            if match:
                resolution = int(match.group(1))
                if resolution > max_resolution:
                    max_resolution = resolution

        unique_urls = set(all_images)
        return [url for url in unique_urls if url.endswith('.jpg') and f'_{max_resolution}w_' in url]

    def fetch_and_summarize(self, title, href):
        logger.info(f"Fetching and summarizing article: {title}")
        if title == "Málaga":
            return None
        try:
            resp = requests.get(href, headers=self.headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            h1 = soup.find('h1')
            if not h1:
                logger.warning(f"[fetch] No <h1> on {href}")
                return None
            title = h1.get_text(strip=True)
            logger.info(f"Article title: {title}")

            date_node = soup.find('p', class_='timestamp-atom')
            if not date_node:
                logger.warning(f"[fetch] No timestamp on {href}")
                return None

            try:
                date_time = self._parse_spanish_date(date_node.text)
            except (ValueError, IndexError) as e:
                logger.warning(f"[fetch] Date parse failed for {href}: {e}")
                return None

            logger.info(f"Article date: {date_time}")
            if date_time < datetime.now() - timedelta(days=7):
                logger.info("Article is older than 7 days, skipping.")
                return None

            content = '\n'.join(p.get_text(strip=True) for p in soup.find_all('p'))
            if not content.strip():
                logger.warning(f"[fetch] No content extracted from {href}")
                return None

            images = self._extract_images(soup)
            logger.info(f"Found {len(images)} images.")
            return content, images, date_time

        except requests.RequestException as e:
            logger.error(f"[fetch] HTTP error for {href}: {e}")
            return None
        except Exception as e:
            logger.error(f"[fetch] Unexpected error for {href}: {e!r}")
            return None
