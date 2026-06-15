import logging
from datetime import datetime

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class DiarioSurScraper(BaseScraper):
    """Scraper for diariosur.es — ISO timestamps and ``og:image``/``<main>`` images."""

    def _extract_date(self, soup):
        raw = None
        meta = soup.find('meta', attrs={'property': 'article:published_time'})
        if meta and meta.get('content'):
            raw = meta['content']
        else:
            time_node = soup.find('time', attrs={'datetime': True})
            if time_node:
                raw = time_node['datetime']
        if not raw:
            return None
        # diariosur stamps carry a timezone (e.g. +02:00 / Z); drop it so the
        # comparison against the naive datetime.now() in the base class holds.
        return datetime.fromisoformat(raw.replace('Z', '+00:00')).replace(tzinfo=None)

    def _content_root(self, soup):
        return soup.find('main') or soup

    def _extract_images(self, soup):
        images = []

        og = soup.find('meta', attrs={'property': 'og:image'})
        if og and og.get('content'):
            images.append(og['content'])

        main = soup.find('main')
        if main:
            for img in main.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                if 'ppllstatics.com/diariosur' not in src or '/multimedia/' not in src:
                    continue
                if '/comun/' in src or 'autor-' in src:
                    continue
                images.append(src)

        # Dedupe while preserving order (lead og:image stays first).
        seen = set()
        return [u for u in images if not (u in seen or seen.add(u))]
