"""
Similarity tests for DataService.

Classes:
  TestCosineMath            — pure numpy, no API calls
  TestGeminiEmbedding       — real Gemini API (skipped without GEMINI_API_KEY)
  TestDataServiceSimilarity — real embeddings + mocked Supabase requests
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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

_DUMMY_URL = 'https://dummy.supabase.co'
_DUMMY_KEY = 'dummy-key'
_DISTANCE_THRESHOLD = 0.15  # similarity_threshold = 0.85


def _make_service():
    from data_service import DataService
    return DataService(
        supabase_url=_DUMMY_URL,
        supabase_key=_DUMMY_KEY,
        DISTANCE_THRESHOLD=_DISTANCE_THRESHOLD,
        gemini_api_key=GEMINI_API_KEY or 'dummy',
    )


# ---------------------------------------------------------------------------
# 1. Pure math — no API, no network
# ---------------------------------------------------------------------------

class TestCosineMath(unittest.TestCase):

    def setUp(self):
        self.svc = _make_service()

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
# 2. Real Gemini embedding API
# ---------------------------------------------------------------------------

@unittest.skipUnless(GEMINI_API_KEY, 'GEMINI_API_KEY must be set')
class TestGeminiEmbedding(unittest.TestCase):

    def setUp(self):
        self.svc = _make_service()

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
# 3. DataService.is_new_article — real embeddings, mocked Supabase
# ---------------------------------------------------------------------------

@unittest.skipUnless(GEMINI_API_KEY, 'GEMINI_API_KEY must be set')
class TestDataServiceSimilarity(unittest.TestCase):

    def setUp(self):
        self.svc = _make_service()

    def _mock_supabase(self, rows):
        """Return a context manager that patches requests.get with given rows."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = rows
        return patch('requests.get', return_value=mock_resp)

    def test_identical_title_is_not_new(self):
        title = 'Málaga airport reaches record passenger numbers this summer'
        embedding = self.svc._embed(title)
        rows = [{'title': title, 'embedding': embedding}]

        with self._mock_supabase(rows):
            self.assertFalse(self.svc.is_new_article(title))

    def test_paraphrase_is_not_new(self):
        stored = 'Málaga port expansion project approved by city council'
        incoming = 'City council gives green light to expand Málaga harbour'
        stored_emb = self.svc._embed(stored)
        rows = [{'title': stored, 'embedding': stored_emb}]

        with self._mock_supabase(rows):
            result = self.svc.is_new_article(incoming)
        sim = self.svc._cosine(self.svc._embed(incoming), stored_emb)
        print(f'\n  [paraphrase is_new] cosine = {sim:.4f}, is_new = {result}')
        self.assertFalse(result)

    def test_unrelated_article_is_new(self):
        stored = 'New tapas restaurant opens in Málaga old town'
        incoming = 'Real Madrid wins Champions League final in London'
        stored_emb = self.svc._embed(stored)
        rows = [{'title': stored, 'embedding': stored_emb}]

        with self._mock_supabase(rows):
            self.assertTrue(self.svc.is_new_article(incoming))

    def test_empty_database_always_new(self):
        with self._mock_supabase([]):
            self.assertTrue(self.svc.is_new_article('Any headline'))

    def test_legacy_row_without_embedding_falls_back_to_jaccard(self):
        """Rows with no embedding stored fall back to Jaccard similarity."""
        title = 'Málaga beach wins blue flag award for cleanliness'
        rows = [{'title': title, 'embedding': None}]

        with self._mock_supabase(rows):
            self.assertFalse(self.svc.is_new_article(title))

    def test_supabase_error_defaults_to_new(self):
        """If Supabase is unreachable, treat the article as new to avoid blocking."""
        with patch('requests.get', side_effect=Exception('network error')):
            self.assertTrue(self.svc.is_new_article('Some headline'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
