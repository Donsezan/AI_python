import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seen_cache import SeenCache

URL = "https://example.com/article-1"


class TestSeenCache(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "seen_cache.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _cache(self, **kwargs):
        return SeenCache(path=self.path, **kwargs)

    def test_unknown_url_not_skipped(self):
        self.assertFalse(self._cache().should_skip(URL))

    def test_terminal_status_skips(self):
        cache = self._cache()
        cache.record_terminal(URL, "too_old")
        self.assertTrue(cache.should_skip(URL))

    def test_attempts_become_terminal_after_max(self):
        cache = self._cache(max_attempts=3)
        for expected in (1, 2):
            self.assertEqual(cache.record_attempt(URL, "fetch_failed"), expected)
            self.assertFalse(cache.should_skip(URL))
        self.assertEqual(cache.record_attempt(URL, "fetch_failed"), 3)
        self.assertTrue(cache.should_skip(URL))

    def test_persists_across_instances(self):
        self._cache().record_terminal(URL, "no_content")
        self.assertTrue(self._cache().should_skip(URL))

    def test_prunes_stale_entries_on_load(self):
        cache = self._cache()
        cache.record_terminal(URL, "too_old")
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data[URL]["last_seen"] = old
        self.path.write_text(json.dumps(data), encoding="utf-8")

        self.assertFalse(self._cache().should_skip(URL))

    def test_should_skip_refreshes_last_seen(self):
        cache = self._cache()
        cache.record_terminal(URL, "too_old")
        old = (datetime.now(timezone.utc) - timedelta(days=13)).isoformat()
        cache._entries[URL]["last_seen"] = old

        self.assertTrue(cache.should_skip(URL))
        self.assertGreater(cache._entries[URL]["last_seen"], old)
        cache.flush_if_dirty()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertGreater(data[URL]["last_seen"], old)

    def test_corrupt_file_starts_empty(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        self.assertFalse(self._cache().should_skip(URL))


if __name__ == '__main__':
    unittest.main()
