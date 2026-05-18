import uuid
import logging
from datetime import datetime, timedelta
import numpy as np
import requests

logger = logging.getLogger(__name__)

_EMBED_URL = "https://api.cohere.com/v1/embed"
_EMBED_MODEL = "embed-multilingual-v3.0"


class DataService:

    def __init__(self, supabase_url, supabase_key, DISTANCE_THRESHOLD, cohere_api_key):
        self.similarity_threshold = 1 - DISTANCE_THRESHOLD
        self.url = f"{supabase_url.rstrip('/')}/rest/v1/articles"
        self.headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        }
        self._cohere_api_key = cohere_api_key

    def _embed(self, text):
        resp = requests.post(
            _EMBED_URL,
            headers={
                "Authorization": f"Bearer {self._cohere_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "texts": [text],
                "model": _EMBED_MODEL,
                "input_type": "search_document",
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def _cosine(self, a, b):
        a, b = np.array(a), np.array(b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def _jaccard(self, a, b):
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def fetch_recent_articles(self):
        try:
            resp = requests.get(self.url, headers=self.headers, params={"select": "title,embedding,url"}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching recent articles: {e}")
            return []

    def is_url_seen(self, href, rows):
        return any(row.get("url") == href for row in rows)

    def is_new_article_cached(self, title, rows):
        try:
            embedding = self._embed(title)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to Jaccard: {e}")
            embedding = None

        for row in rows:
            stored_emb = row.get("embedding")
            if embedding is not None and stored_emb is not None:
                sim = self._cosine(embedding, stored_emb)
            else:
                sim = self._jaccard(title, row["title"])
            if sim >= self.similarity_threshold:
                return False
        return True

    def is_new_article(self, title):
        rows = self.fetch_recent_articles()
        return self.is_new_article_cached(title, rows)

    def save_article(self, title, date_time, url=None):
        try:
            embedding = self._embed(title)
        except Exception as e:
            logger.warning(f"Embedding failed, saving without embedding: {e}")
            embedding = None

        try:
            payload = {
                "id": str(uuid.uuid4()),
                "title": title,
                "date": date_time.isoformat(),
                "embedding": embedding,
            }
            if url:
                payload["url"] = url
            resp = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            logger.info(f"Article '{title}' saved to database.")
            return True
        except Exception as e:
            logger.error(f"Error saving article '{title}': {e}")
            return False

    def cleanup_old_articles(self, max_age_days=10):
        try:
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            resp = requests.delete(
                self.url,
                headers={**self.headers, "Prefer": "count=exact"},
                params={"date": f"lt.{cutoff}"},
                timeout=15,
            )
            resp.raise_for_status()
            count = resp.headers.get("Content-Range", "*/0").split("/")[-1]
            logger.info(f"Deleted {count} old articles from the database.")
        except Exception as e:
            logger.error(f"Error cleaning up old articles: {e}")
