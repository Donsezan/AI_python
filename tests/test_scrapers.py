import os
import sys
import unittest
from datetime import datetime

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrapers import DiarioSurScraper, MalagaHoyScraper


DIARIOSUR_HTML = """
<html><head>
  <meta property="article:published_time" content="2026-06-15T14:51:07+02:00">
  <meta property="og:image" content="https://s2.ppllstatics.com/diariosur/www/multimedia/2026/06/15/lead-1200x840@Diario%20Sur.jpg">
</head><body>
  <header><p>nav boilerplate paragraph that should be ignored entirely</p></header>
  <main>
    <h1>Titular de prueba en Malaga</h1>
    <img src="https://s2.ppllstatics.com/diariosur/www/multimedia/2026/06/15/lead-1200x840@Diario%20Sur.jpg">
    <img src="https://s1.ppllstatics.com/comun/img/2014/autor/autor-433-foto-2.jpeg?width=90">
    <img src="https://s3.ppllstatics.com/diariosur/www/multimedia/2026/06/15/second-1200x840@Diario%20Sur.jpg">
    <p>Primer parrafo del cuerpo del articulo con contenido real.</p>
    <p>Segundo parrafo del cuerpo.</p>
  </main>
  <footer><p>footer paragraph boilerplate</p></footer>
</body></html>
"""


class TestDiarioSurScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = DiarioSurScraper("https://www.diariosur.es/malaga/", {"User-Agent": "t"})
        self.soup = BeautifulSoup(DIARIOSUR_HTML, "html.parser")

    def test_extract_date_strips_timezone(self):
        dt = self.scraper._extract_date(self.soup)
        self.assertEqual(dt, datetime(2026, 6, 15, 14, 51, 7))
        self.assertIsNone(dt.tzinfo)

    def test_extract_date_falls_back_to_time_tag(self):
        html = '<html><body><time datetime="2026-06-15T12:54:38Z">x</time></body></html>'
        dt = self.scraper._extract_date(BeautifulSoup(html, "html.parser"))
        self.assertEqual(dt, datetime(2026, 6, 15, 12, 54, 38))

    def test_extract_date_returns_none_when_absent(self):
        self.assertIsNone(self.scraper._extract_date(BeautifulSoup("<html></html>", "html.parser")))

    def test_content_root_scopes_to_main(self):
        content, _ = self.scraper.parse_content(self.soup)
        self.assertIn("Primer parrafo", content)
        self.assertNotIn("nav boilerplate", content)
        self.assertNotIn("footer paragraph", content)

    def test_extract_images_og_first_and_filters_author(self):
        _, images = self.scraper.parse_content(self.soup)
        self.assertEqual(images[0], "https://s2.ppllstatics.com/diariosur/www/multimedia/2026/06/15/lead-1200x840@Diario%20Sur.jpg")
        self.assertIn("https://s3.ppllstatics.com/diariosur/www/multimedia/2026/06/15/second-1200x840@Diario%20Sur.jpg", images)
        # og:image and the <main> lead img are the same URL — deduped, no author thumb.
        self.assertEqual(len(images), 2)
        self.assertFalse(any("autor-" in u for u in images))


class TestMalagaHoyScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = MalagaHoyScraper("https://www.malagahoy.es/malaga/", {"User-Agent": "t"})

    def test_parse_spanish_date(self):
        dt = self.scraper._parse_spanish_date("\n15 de diciembre 2025-14:30\n")
        self.assertEqual(dt, datetime(2025, 12, 15, 14, 30))

    def test_extract_date_missing_node_returns_none(self):
        self.assertIsNone(self.scraper._extract_date(BeautifulSoup("<html></html>", "html.parser")))


if __name__ == "__main__":
    unittest.main()
