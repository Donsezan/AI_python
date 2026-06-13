import sys
import os
import uuid
import unittest
from datetime import datetime

import requests

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

SUPABASE_URL = os.getenv('SUPABASE_URL', '').rstrip('/')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
ARTICLES_ENDPOINT = f"{SUPABASE_URL}/rest/v1/articles"
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

REQUIRED_COLUMNS = {"id", "title", "date"}
_DUMMY_EMBEDDING = [0.1] * 1024   # arbitrary fixture for jsonb round-trip; dimension is not Gemini-specific


@unittest.skipUnless(SUPABASE_URL and SUPABASE_KEY, "SUPABASE_URL and SUPABASE_KEY must be set")
class TestSupabaseConnection(unittest.TestCase):

    def test_credentials_are_configured(self):
        self.assertTrue(SUPABASE_URL, "SUPABASE_URL is empty")
        self.assertTrue(SUPABASE_KEY, "SUPABASE_KEY is empty")
        self.assertTrue(SUPABASE_URL.startswith("https://"), "SUPABASE_URL must be an https URL")

    def test_rest_api_is_reachable(self):
        resp = requests.get(ARTICLES_ENDPOINT, headers=HEADERS, params={"limit": "0"}, timeout=10)
        self.assertEqual(resp.status_code, 200, f"Expected 200, got {resp.status_code}: {resp.text}")

    def test_response_is_json_array(self):
        resp = requests.get(ARTICLES_ENDPOINT, headers=HEADERS, params={"limit": "0"}, timeout=10)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, list, "Response body must be a JSON array")

    def test_authentication_is_required(self):
        """Unauthenticated request should be rejected (401 or 403)."""
        resp = requests.get(ARTICLES_ENDPOINT, timeout=10)
        self.assertIn(resp.status_code, (401, 403), f"Expected 401/403 without auth, got {resp.status_code}")


@unittest.skipUnless(SUPABASE_URL and SUPABASE_KEY, "SUPABASE_URL and SUPABASE_KEY must be set")
class TestSupabaseTableStructure(unittest.TestCase):

    def _openapi_schema(self):
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/",
            headers={"apikey": SUPABASE_KEY, "Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def test_required_columns_are_selectable(self):
        """PostgREST returns 400 if any column in `select` doesn't exist."""
        columns = ",".join(REQUIRED_COLUMNS)
        resp = requests.get(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            params={"select": columns, "limit": "0"},
            timeout=10,
        )
        self.assertEqual(
            resp.status_code, 200,
            f"Column select failed — a required column may be missing. {resp.status_code}: {resp.text}"
        )

    def test_id_column_exists(self):
        resp = requests.get(ARTICLES_ENDPOINT, headers=HEADERS, params={"select": "id", "limit": "0"}, timeout=10)
        self.assertEqual(resp.status_code, 200, f"'id' column missing: {resp.text}")

    def test_title_column_exists(self):
        resp = requests.get(ARTICLES_ENDPOINT, headers=HEADERS, params={"select": "title", "limit": "0"}, timeout=10)
        self.assertEqual(resp.status_code, 200, f"'title' column missing: {resp.text}")

    def test_date_column_exists(self):
        resp = requests.get(ARTICLES_ENDPOINT, headers=HEADERS, params={"select": "date", "limit": "0"}, timeout=10)
        self.assertEqual(resp.status_code, 200, f"'date' column missing: {resp.text}")

    def test_embedding_column_exists(self):
        """embedding (jsonb) column must exist — required for Gemini cosine deduplication."""
        resp = requests.get(ARTICLES_ENDPOINT, headers=HEADERS, params={"select": "embedding", "limit": "0"}, timeout=10)
        self.assertEqual(
            resp.status_code, 200,
            f"'embedding' column missing or inaccessible. Run: ALTER TABLE articles ADD COLUMN embedding jsonb; "
            f"Got {resp.status_code}: {resp.text}"
        )

    def test_no_unexpected_required_columns_block_insert(self):
        """
        Insert with only {id, title, date} must succeed (embedding is nullable).
        The test record is deleted immediately after.
        """
        test_id = str(uuid.uuid4())
        payload = {"id": test_id, "title": "__test_structure__", "date": datetime.now().isoformat()}

        insert_resp = requests.post(ARTICLES_ENDPOINT, headers=HEADERS, json=payload, timeout=10)
        self.assertIn(
            insert_resp.status_code, (200, 201),
            f"Insert with required columns failed: {insert_resp.status_code}: {insert_resp.text}"
        )

        requests.delete(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            params={"id": f"eq.{test_id}"},
            timeout=10,
        )

    def test_openapi_schema_lists_articles_table(self):
        """OpenAPI spec exposed by PostgREST must include an 'articles' definition."""
        try:
            schema = self._openapi_schema()
        except Exception as e:
            self.skipTest(f"OpenAPI endpoint not accessible: {e}")

        definitions = schema.get("definitions") or schema.get("components", {}).get("schemas", {})
        self.assertIn("articles", definitions, f"'articles' not found in OpenAPI definitions. Keys: {list(definitions)}")

    def test_openapi_column_types(self):
        """id/title/date types and presence of embedding column in OpenAPI spec."""
        try:
            schema = self._openapi_schema()
        except Exception as e:
            self.skipTest(f"OpenAPI endpoint not accessible: {e}")

        definitions = schema.get("definitions") or schema.get("components", {}).get("schemas", {})
        if "articles" not in definitions:
            self.skipTest("'articles' table not in OpenAPI spec")

        props = definitions["articles"].get("properties", {})

        self.assertIn("id", props, "'id' property missing from schema")
        id_prop = props["id"]
        self.assertIn(
            id_prop.get("format"), ("uuid", "text"),
            f"'id' column should have format 'uuid' or 'text', got: {id_prop}"
        )
        self.assertEqual(id_prop.get("type"), "string", f"'id' column should be type 'string', got: {id_prop}")

        self.assertIn("title", props, "'title' property missing from schema")
        self.assertEqual(props["title"].get("type"), "string", f"'title' should be type 'string', got: {props['title']}")

        self.assertIn("date", props, "'date' property missing from schema")
        self.assertEqual(
            props["date"].get("type"), "string",
            f"'date' column should be type 'string', got: {props['date']}"
        )

        self.assertIn("embedding", props, "'embedding' column missing from OpenAPI schema — has the column been added?")


@unittest.skipUnless(SUPABASE_URL and SUPABASE_KEY, "SUPABASE_URL and SUPABASE_KEY must be set")
class TestSupabaseCRUD(unittest.TestCase):
    """Round-trip tests to verify insert, read, and delete work end-to-end."""

    def setUp(self):
        self.test_id = str(uuid.uuid4())
        self.test_title = f"__integration_test_{self.test_id[:8]}__"
        self.test_date = datetime.now().isoformat()

    def tearDown(self):
        requests.delete(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            params={"id": f"eq.{self.test_id}"},
            timeout=10,
        )

    def test_insert_article(self):
        resp = requests.post(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            json={"id": self.test_id, "title": self.test_title, "date": self.test_date},
            timeout=10,
        )
        self.assertIn(resp.status_code, (200, 201), f"Insert failed: {resp.status_code}: {resp.text}")

    def test_read_inserted_article(self):
        requests.post(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            json={"id": self.test_id, "title": self.test_title, "date": self.test_date},
            timeout=10,
        )
        resp = requests.get(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            params={"id": f"eq.{self.test_id}", "select": "id,title,date"},
            timeout=10,
        )
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        self.assertEqual(len(rows), 1, f"Expected 1 row, got {len(rows)}")
        self.assertEqual(rows[0]["id"], self.test_id)
        self.assertEqual(rows[0]["title"], self.test_title)

    def test_insert_and_read_embedding(self):
        """Embedding vector must survive a Supabase round-trip unchanged."""
        resp = requests.post(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            json={"id": self.test_id, "title": self.test_title, "date": self.test_date, "embedding": _DUMMY_EMBEDDING},
            timeout=10,
        )
        self.assertIn(resp.status_code, (200, 201), f"Insert with embedding failed: {resp.status_code}: {resp.text}")

        read_resp = requests.get(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            params={"id": f"eq.{self.test_id}", "select": "embedding"},
            timeout=10,
        )
        self.assertEqual(read_resp.status_code, 200)
        rows = read_resp.json()
        self.assertEqual(len(rows), 1)
        stored = rows[0]["embedding"]
        self.assertIsInstance(stored, list, "Stored embedding should deserialise as a list")
        self.assertEqual(len(stored), 1024, f"Expected 1024 dimensions, got {len(stored)}")
        self.assertAlmostEqual(stored[0], _DUMMY_EMBEDDING[0], places=6)

    def test_delete_article(self):
        requests.post(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            json={"id": self.test_id, "title": self.test_title, "date": self.test_date},
            timeout=10,
        )
        del_resp = requests.delete(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            params={"id": f"eq.{self.test_id}"},
            timeout=10,
        )
        self.assertIn(del_resp.status_code, (200, 204), f"Delete failed: {del_resp.status_code}: {del_resp.text}")

        verify_resp = requests.get(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            params={"id": f"eq.{self.test_id}"},
            timeout=10,
        )
        self.assertEqual(verify_resp.json(), [], "Row should be deleted but still exists")

    def test_duplicate_id_is_rejected(self):
        payload = {"id": self.test_id, "title": self.test_title, "date": self.test_date}
        requests.post(ARTICLES_ENDPOINT, headers=HEADERS, json=payload, timeout=10)
        resp = requests.post(ARTICLES_ENDPOINT, headers=HEADERS, json=payload, timeout=10)
        self.assertNotIn(resp.status_code, (200, 201), "Duplicate UUID insert should be rejected by primary key constraint")


@unittest.skipUnless(SUPABASE_URL and SUPABASE_KEY and GEMINI_API_KEY,
                     "SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY must be set")
class TestDataServiceDeduplication(unittest.TestCase):
    """
    End-to-end deduplication tests using real Gemini embeddings and live Supabase.
    Each test cleans up after itself.
    """

    def setUp(self):
        from data_service import DataService
        self.svc = DataService(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY,
            DISTANCE_THRESHOLD=0.15,
            gemini_api_key=GEMINI_API_KEY,
        )
        self._inserted_ids = []

    def tearDown(self):
        for article_id in self._inserted_ids:
            requests.delete(
                ARTICLES_ENDPOINT,
                headers=HEADERS,
                params={"id": f"eq.{article_id}"},
                timeout=10,
            )

    def _save(self, title):
        """Save an article and track its id for cleanup."""
        import uuid as _uuid
        article_id = str(_uuid.uuid4())
        requests.post(
            ARTICLES_ENDPOINT,
            headers=HEADERS,
            json={
                "id": article_id,
                "title": title,
                "date": datetime.now().isoformat(),
                "embedding": self.svc._embed(title),
            },
            timeout=10,
        )
        self._inserted_ids.append(article_id)

    def test_new_article_is_detected_as_new(self):
        """A title not in the database should always be new."""
        unique = f"__unique_test_article_{uuid.uuid4().hex}__"
        self.assertTrue(self.svc.is_new_article(unique))

    def test_saved_article_is_not_new(self):
        """After saving an article, the exact same title should not be new."""
        title = "Málaga airport sets new passenger record this summer"
        self._save(title)
        self.assertFalse(self.svc.is_new_article(title))

    def test_paraphrase_is_not_new(self):
        """A paraphrase of a stored headline should be caught as a duplicate."""
        stored = "Málaga port expansion project approved by city council"
        incoming = "City council gives green light to expand Málaga harbour"
        self._save(stored)

        result = self.svc.is_new_article(incoming)
        sim = self.svc._cosine(self.svc._embed(incoming), self.svc._embed(stored))
        print(f"\n  [e2e paraphrase] cosine = {sim:.4f}, is_new = {result}")
        self.assertFalse(result)

    def test_unrelated_article_is_new(self):
        """A completely different headline should pass through as new."""
        stored = "New tapas restaurant opens in Málaga old town"
        incoming = f"Real Madrid wins Champions League final in London [{uuid.uuid4().hex[:6]}]"
        self._save(stored)

        result = self.svc.is_new_article(incoming)
        sim = self.svc._cosine(self.svc._embed(incoming), self.svc._embed(stored))
        print(f"\n  [e2e unrelated]  cosine = {sim:.4f}, is_new = {result}")
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
