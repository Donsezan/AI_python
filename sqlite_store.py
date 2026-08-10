"""Local SQLite persistence for article records.

Storage only: connection handling, schema bootstrap, embedding BLOB
encode/decode and raw CRUD. It knows nothing about similarity, thresholds or
what an embedding means — that lives in `data_service.py`.

Connections are short-lived (one per operation), which sidesteps
`check_same_thread` and holds no long-running locks.
"""

import logging
import sqlite3
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
  id               TEXT PRIMARY KEY,
  title            TEXT NOT NULL,
  date             TEXT NOT NULL,
  url              TEXT,
  embedding        BLOB,
  title_translated TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS articles_url_idx  ON articles(url) WHERE url IS NOT NULL;
CREATE INDEX        IF NOT EXISTS articles_date_idx ON articles(date);
"""


def encode_embedding(values):
    """Pack an embedding into a float32 little-endian BLOB.

    Accepts plain lists (from `DataService._embed`) and numpy arrays alike.
    `None` passes through so a failed embedding stays NULL.
    """
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32).tobytes()


def decode_embedding(blob):
    """Unpack a float32 BLOB. Dimension is inferred from the byte length."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class SqliteStore:

    def __init__(self, db_path):
        self.db_path = db_path
        self._bootstrap()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _bootstrap(self):
        """Create the schema on first connect. Logged, never raised — a broken
        database must degrade to empty reads and failed writes, not crash the bot."""
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except Exception as e:
            logger.error(f"Error initialising database at '{self.db_path}': {e}")

    def fetch_all(self):
        """Every row as a dict of exactly title/url/embedding — the columns the
        dedup pass reads. Other columns are write-only."""
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT title, url, embedding FROM articles").fetchall()
            return [
                {"title": title, "url": url, "embedding": decode_embedding(embedding)}
                for title, url, embedding in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []

    def insert(self, article_id, title, date, url=None, embedding=None, title_translated=None):
        """Insert one row. Returns False on any failure — including a duplicate
        id or url — which is what tells `main.py` the save did not happen."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO articles (id, title, date, url, embedding, title_translated)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (article_id, title, date, url, encode_embedding(embedding), title_translated),
                )
            return True
        except Exception as e:
            logger.error(f"Error inserting article '{title}': {e}")
            return False

    def delete_older_than(self, cutoff_iso):
        """Delete rows dated strictly before `cutoff_iso`; returns the row count.

        `date` is ISO-8601 text, so the lexicographic comparison is chronological.
        """
        try:
            with self._connect() as conn:
                return conn.execute("DELETE FROM articles WHERE date < ?", (cutoff_iso,)).rowcount
        except Exception as e:
            logger.error(f"Error deleting old articles: {e}")
            return 0
