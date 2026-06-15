import re
import logging
from datetime import datetime

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class MalagaHoyScraper(BaseScraper):
    """Scraper for malagahoy.es — Spanish-text timestamps and ``<source srcset>`` images."""

    _MONTHS = {
        'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
        'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
        'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12',
    }

    def _extract_date(self, soup):
        date_node = soup.find('p', class_='timestamp-atom')
        if not date_node:
            return None
        return self._parse_spanish_date(date_node.text)

    def _parse_spanish_date(self, date_text):
        parts = date_text.strip().split('\n')
        date_string = next(
            (p for p in parts if any(m in p.lower() for m in self._MONTHS)),
            None,
        )
        if date_string is None:
            raise ValueError(f"No date found in: {date_text!r}")
        for month_name, month_number in self._MONTHS.items():
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
