"""
Similarity tests for DataService.

Classes:
  TestCosineMath            — pure numpy, no API calls
  TestGeminiEmbedding       — real Gemini API (skipped without GEMINI_API_KEY)
  TestDataServiceSimilarity — real embeddings against a seeded temp SQLite file
"""

import logging
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _stub_embed(svc, vector):
    """Pin the Gemini embedding call — the one unavoidable network boundary —
    so dedup mechanics can be tested without an API key."""
    return patch.object(svc, '_embed', return_value=vector)


@contextmanager
def _failing_embed(svc):
    """Simulate the embedding API being unreachable, muting the expected warning."""
    logging.disable(logging.CRITICAL)
    try:
        with patch.object(svc, '_embed', side_effect=Exception('embed unavailable')):
            yield
    finally:
        logging.disable(logging.NOTSET)


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))
    except FileNotFoundError:
        pass


_load_env()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

_DISTANCE_THRESHOLD = 0.15  # similarity_threshold = 0.85


class _TempDbTestCase(unittest.TestCase):
    """Gives each test a DataService backed by its own throwaway database."""

    def setUp(self):
        from data_service import DataService
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'articles.db')
        self.svc = DataService(
            db_path=self.db_path,
            DISTANCE_THRESHOLD=_DISTANCE_THRESHOLD,
            gemini_api_key=GEMINI_API_KEY or 'dummy',
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed(self, title, embedding, url=None):
        """Store an article exactly the way the bot does after a successful post."""
        self.assertTrue(self.svc.save_article(
            title, datetime.now(), url=url, embedding=embedding,
        ))


# ---------------------------------------------------------------------------
# 1. Pure math — no API, no network
# ---------------------------------------------------------------------------

class TestCosineMath(_TempDbTestCase):

    def test_identical_vectors_score_one(self):
        v = [1.0, 0.5, -0.3, 0.8]
        self.assertAlmostEqual(self.svc._cosine(v, v), 1.0, places=6)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(self.svc._cosine([1, 0], [0, 1]), 0.0, places=6)

    def test_opposite_vectors_score_minus_one(self):
        self.assertAlmostEqual(self.svc._cosine([1, 0], [-1, 0]), -1.0, places=6)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(self.svc._cosine([0, 0], [1, 0]), 0.0)

    def test_symmetry(self):
        a, b = [0.2, 0.8, -0.5], [0.9, 0.1, 0.3]
        self.assertAlmostEqual(self.svc._cosine(a, b), self.svc._cosine(b, a), places=10)


# ---------------------------------------------------------------------------
# 2. Storage-backed behaviour — no API, no network
# ---------------------------------------------------------------------------

class TestDataServiceStorage(_TempDbTestCase):
    """Dedup mechanics with hand-made vectors, so no Gemini key is needed."""

    def test_fetch_recent_articles_returns_title_url_embedding(self):
        self._seed('A stored headline', [1.0, 0.0], url='https://example.com/a')
        rows = self.svc.fetch_recent_articles()
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'title', 'url', 'embedding'})

    def test_fetch_recent_articles_is_empty_before_any_save(self):
        self.assertEqual(self.svc.fetch_recent_articles(), [])

    def test_is_url_seen_matches_a_stored_url(self):
        self._seed('A stored headline', [1.0, 0.0], url='https://example.com/a')
        self.assertTrue(self.svc.is_url_seen('https://example.com/a', self.svc.fetch_recent_articles()))

    def test_is_url_seen_rejects_an_unknown_url(self):
        self._seed('A stored headline', [1.0, 0.0], url='https://example.com/a')
        self.assertFalse(self.svc.is_url_seen('https://example.com/b', self.svc.fetch_recent_articles()))

    def test_identical_embedding_is_not_new(self):
        vector = [1.0, 0.0, 0.0]
        self._seed('A stored headline', vector)
        with _stub_embed(self.svc, vector):
            is_new, _ = self.svc.is_new_article_cached('Any headline', self.svc.fetch_recent_articles())
        self.assertFalse(is_new)

    def test_orthogonal_embedding_is_new(self):
        self._seed('A stored headline', [1.0, 0.0, 0.0])
        with _stub_embed(self.svc, [0.0, 1.0, 0.0]):
            is_new, _ = self.svc.is_new_article_cached('Any headline', self.svc.fetch_recent_articles())
        self.assertTrue(is_new)

    def test_is_new_article_cached_returns_the_embedding_it_used(self):
        vector = [0.6, 0.8]
        with _stub_embed(self.svc, vector):
            _, returned = self.svc.is_new_article_cached('Any headline', [])
        self.assertEqual(returned, vector)

    def test_row_without_embedding_falls_back_to_jaccard(self):
        title = 'Málaga beach wins blue flag award for cleanliness'
        self._seed(title, None)
        with _stub_embed(self.svc, [1.0, 0.0]):
            is_new, _ = self.svc.is_new_article_cached(title, self.svc.fetch_recent_articles())
        self.assertFalse(is_new)

    def test_failed_embedding_falls_back_to_jaccard(self):
        title = 'Málaga beach wins blue flag award for cleanliness'
        self._seed(title, [1.0, 0.0])
        with _failing_embed(self.svc):
            is_new, embedding = self.svc.is_new_article_cached(title, self.svc.fetch_recent_articles())
        self.assertFalse(is_new)
        self.assertIsNone(embedding)

    def test_saved_embedding_survives_the_float32_round_trip(self):
        vector = [0.6, 0.8]
        self._seed('A stored headline', vector)
        stored = self.svc.fetch_recent_articles()[0]['embedding']
        self.assertAlmostEqual(self.svc._cosine(vector, stored), 1.0, places=5)

    def test_save_article_rejects_a_duplicate_url(self):
        """A False return is what makes main.py record 'posted' in the seen-cache."""
        self._seed('First headline', [1.0, 0.0], url='https://example.com/a')
        logging.disable(logging.CRITICAL)
        try:
            self.assertFalse(self.svc.save_article(
                'Second headline', datetime.now(), url='https://example.com/a', embedding=[0.0, 1.0],
            ))
        finally:
            logging.disable(logging.NOTSET)

    def test_save_article_persists_the_translated_title(self):
        self.assertTrue(self.svc.save_article(
            'Titular original', datetime.now(), url='https://example.com/a',
            embedding=[1.0, 0.0], translated_title='Original headline',
        ))

    def test_cleanup_removes_articles_older_than_the_max_age(self):
        self.svc.save_article('Ancient headline', datetime(2020, 1, 1), embedding=[1.0, 0.0])
        self.svc.cleanup_old_articles(max_age_days=10)
        self.assertEqual(self.svc.fetch_recent_articles(), [])

    def test_cleanup_keeps_recent_articles(self):
        self.svc.save_article('Fresh headline', datetime.now(), embedding=[1.0, 0.0])
        self.svc.cleanup_old_articles(max_age_days=10)
        self.assertEqual(len(self.svc.fetch_recent_articles()), 1)

    def test_unreadable_database_defaults_to_new(self):
        """If storage is broken, treat the article as new rather than blocking the bot."""
        from data_service import DataService
        logging.disable(logging.CRITICAL)
        try:
            broken = DataService(
                db_path=os.path.join(self.tmpdir, 'nonexistent-dir', 'articles.db'),
                DISTANCE_THRESHOLD=_DISTANCE_THRESHOLD,
                gemini_api_key='dummy',
            )
            self.assertEqual(broken.fetch_recent_articles(), [])
            with _stub_embed(broken, [1.0, 0.0]):
                self.assertTrue(broken.is_new_article('Some headline'))
        finally:
            logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# 3. Real Gemini embedding API
# ---------------------------------------------------------------------------

@unittest.skipUnless(GEMINI_API_KEY, 'GEMINI_API_KEY must be set')
class TestGeminiEmbedding(_TempDbTestCase):

    def test_embed_returns_list_of_floats(self):
        emb = self.svc._embed('Heavy rain floods streets in Málaga')
        self.assertIsInstance(emb, list)
        self.assertTrue(all(isinstance(x, float) for x in emb))

    def test_embed_returns_nonempty_vector(self):
        emb = self.svc._embed('Local council approves new park in Málaga')
        self.assertGreater(len(emb), 0)

    def test_same_text_cosine_is_one(self):
        text = 'Málaga airport reaches record passenger numbers'
        a = self.svc._embed(text)
        b = self.svc._embed(text)
        self.assertAlmostEqual(self.svc._cosine(a, b), 1.0, places=4)

    def test_duplicate_headlines_score_above_threshold(self):
        """Near-identical news headlines should exceed the similarity threshold."""
        a = self.svc._embed('Torrential rain floods streets in Málaga city centre')
        b = self.svc._embed('Heavy rainfall causes flooding in central Málaga')
        sim = self.svc._cosine(a, b)
        print(f'\n  [duplicate pair] cosine = {sim:.4f}')
        self.assertGreaterEqual(sim, self.svc.similarity_threshold)

    def test_unrelated_headlines_score_below_threshold(self):
        """Completely different news topics should fall below the similarity threshold."""
        a = self.svc._embed('New tapas restaurant opens in Málaga old town')
        b = self.svc._embed('Real Madrid wins Champions League final in London')
        sim = self.svc._cosine(a, b)
        print(f'\n  [unrelated pair] cosine = {sim:.4f}')
        self.assertLess(sim, self.svc.similarity_threshold)

    def test_paraphrase_score_above_threshold(self):
        """Same event described in different words should be caught as a duplicate."""
        a = self.svc._embed('Málaga port expansion project approved by city council')
        b = self.svc._embed('City council gives green light to expand Málaga harbour')
        sim = self.svc._cosine(a, b)
        print(f'\n  [paraphrase pair]  cosine = {sim:.4f}')
        self.assertGreaterEqual(sim, self.svc.similarity_threshold)


# ---------------------------------------------------------------------------
# 4. DataService.is_new_article — real embeddings, real local storage
# ---------------------------------------------------------------------------

@unittest.skipUnless(GEMINI_API_KEY, 'GEMINI_API_KEY must be set')
class TestDataServiceSimilarity(_TempDbTestCase):

    def test_identical_title_is_not_new(self):
        title = 'Málaga airport reaches record passenger numbers this summer'
        self._seed(title, self.svc._embed(title))
        self.assertFalse(self.svc.is_new_article(title))

    def test_paraphrase_is_not_new(self):
        stored = 'Málaga port expansion project approved by city council'
        incoming = 'City council gives green light to expand Málaga harbour'
        stored_emb = self.svc._embed(stored)
        self._seed(stored, stored_emb)

        result = self.svc.is_new_article(incoming)
        sim = self.svc._cosine(self.svc._embed(incoming), stored_emb)
        print(f'\n  [paraphrase is_new] cosine = {sim:.4f}, is_new = {result}')
        self.assertFalse(result)

    def test_unrelated_article_is_new(self):
        stored = 'New tapas restaurant opens in Málaga old town'
        incoming = 'Real Madrid wins Champions League final in London'
        self._seed(stored, self.svc._embed(stored))
        self.assertTrue(self.svc.is_new_article(incoming))

    def test_empty_database_always_new(self):
        self.assertTrue(self.svc.is_new_article('Any headline'))

    def test_legacy_row_without_embedding_falls_back_to_jaccard(self):
        """Rows with no embedding stored fall back to Jaccard similarity."""
        title = 'Málaga beach wins blue flag award for cleanliness'
        self._seed(title, None)
        self.assertFalse(self.svc.is_new_article(title))


if __name__ == '__main__':
    unittest.main(verbosity=2)
