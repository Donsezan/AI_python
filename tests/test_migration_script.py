"""
Unit tests for the one-off Supabase -> SQLite migration script.

Covers the row-conversion helper and the idempotent write path only — no
network, no credentials. The PostgREST paging itself is exercised by hand
during the cutover.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.migrate_supabase_to_sqlite import row_to_params, write_rows
from sqlite_store import SqliteStore, decode_embedding


class TestRowConversion(unittest.TestCase):

    def test_embedding_list_becomes_a_float32_blob(self):
        params = row_to_params({'id': 'a', 'title': 'T', 'date': '2026-08-10T09:00:00',
                                'embedding': [0.5, -0.25]})
        np.testing.assert_allclose(decode_embedding(params['embedding']), [0.5, -0.25], rtol=1e-6)

    def test_null_embedding_stays_null(self):
        params = row_to_params({'id': 'a', 'title': 'T', 'date': '2026-08-10T09:00:00',
                                'embedding': None})
        self.assertIsNone(params['embedding'])

    def test_missing_embedding_key_stays_null(self):
        params = row_to_params({'id': 'a', 'title': 'T', 'date': '2026-08-10T09:00:00'})
        self.assertIsNone(params['embedding'])

    def test_json_encoded_embedding_string_is_decoded(self):
        """PostgREST can hand back a jsonb column as a JSON string."""
        params = row_to_params({'id': 'a', 'title': 'T', 'date': '2026-08-10T09:00:00',
                                'embedding': '[0.5, -0.25]'})
        np.testing.assert_allclose(decode_embedding(params['embedding']), [0.5, -0.25], rtol=1e-6)

    def test_core_columns_are_carried_over(self):
        params = row_to_params({'id': 'a', 'title': 'Titular', 'date': '2026-08-10T09:00:00',
                                'url': 'https://example.com/a', 'title_translated': 'Headline'})
        self.assertEqual(
            (params['id'], params['title'], params['date'], params['url'], params['title_translated']),
            ('a', 'Titular', '2026-08-10T09:00:00', 'https://example.com/a', 'Headline'),
        )

    def test_missing_optional_columns_become_none(self):
        params = row_to_params({'id': 'a', 'title': 'T', 'date': '2026-08-10T09:00:00'})
        self.assertEqual((params['url'], params['title_translated']), (None, None))


class TestWriteRows(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'articles.db')
        SqliteStore(self.db_path)  # bootstrap the schema
        self.rows = [
            {'id': 'a', 'title': 'First', 'date': '2026-08-10T09:00:00',
             'url': 'https://example.com/a', 'embedding': [1.0, 0.0]},
            {'id': 'b', 'title': 'Second', 'date': '2026-08-10T10:00:00',
             'url': 'https://example.com/b', 'embedding': None},
        ]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_number_of_inserted_rows(self):
        self.assertEqual(write_rows(self.db_path, self.rows), 2)

    def test_rows_land_in_the_database(self):
        write_rows(self.db_path, self.rows)
        self.assertEqual(sorted(r['title'] for r in SqliteStore(self.db_path).fetch_all()),
                         ['First', 'Second'])

    def test_rerunning_inserts_nothing_new(self):
        """INSERT OR IGNORE on id makes the migration safe to re-run."""
        write_rows(self.db_path, self.rows)
        self.assertEqual(write_rows(self.db_path, self.rows), 0)

    def test_rerunning_does_not_duplicate_rows(self):
        write_rows(self.db_path, self.rows)
        write_rows(self.db_path, self.rows)
        self.assertEqual(len(SqliteStore(self.db_path).fetch_all()), 2)

    def test_resuming_a_partial_run_inserts_only_the_missing_rows(self):
        write_rows(self.db_path, self.rows[:1])
        self.assertEqual(write_rows(self.db_path, self.rows), 1)

    def test_embeddings_survive_the_migration(self):
        write_rows(self.db_path, self.rows)
        stored = {r['title']: r['embedding'] for r in SqliteStore(self.db_path).fetch_all()}
        np.testing.assert_allclose(stored['First'], [1.0, 0.0], rtol=1e-6)

    def test_null_embedding_survives_the_migration(self):
        write_rows(self.db_path, self.rows)
        stored = {r['title']: r['embedding'] for r in SqliteStore(self.db_path).fetch_all()}
        self.assertIsNone(stored['Second'])

    def test_dry_run_writes_nothing(self):
        self.assertEqual(write_rows(self.db_path, self.rows, dry_run=True), 0)
        self.assertEqual(SqliteStore(self.db_path).fetch_all(), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
