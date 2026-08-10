"""
Storage tests for SqliteStore.

Real SQLite in a temp directory — no network, no credentials.

Classes:
  TestEmbeddingCodec   — float32 BLOB encode/decode round-trip
  TestSchemaBootstrap  — schema creation on a fresh file, idempotent re-open
  TestInsertAndFetch   — CRUD round-trip, column projection, constraint handling
  TestDeleteOlderThan  — retention cutoff boundary
"""

import contextlib
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlite_store import SqliteStore, decode_embedding, encode_embedding


@contextlib.contextmanager
def _silenced():
    """Mute the store's expected error logging so failure-path tests stay quiet."""
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


class _TempDbTestCase(unittest.TestCase):
    """Gives each test its own throwaway database file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'articles.db')
        self.store = SqliteStore(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Embedding codec
# ---------------------------------------------------------------------------

class TestEmbeddingCodec(unittest.TestCase):

    def test_list_round_trips_through_blob(self):
        values = [0.5, -0.25, 0.125, 0.0]
        self.assertEqual(list(decode_embedding(encode_embedding(values))), values)

    def test_numpy_array_is_accepted(self):
        values = np.array([1.5, -2.5, 3.5], dtype=np.float64)
        decoded = decode_embedding(encode_embedding(values))
        np.testing.assert_allclose(decoded, values, rtol=1e-6)

    def test_decoded_dtype_is_float32(self):
        self.assertEqual(decode_embedding(encode_embedding([1.0, 2.0])).dtype, np.float32)

    def test_blob_is_four_bytes_per_value(self):
        self.assertEqual(len(encode_embedding([1.0] * 10)), 40)

    def test_dimension_is_inferred_from_blob_length(self):
        self.assertEqual(len(decode_embedding(encode_embedding([0.1] * 3072))), 3072)

    def test_none_encodes_to_none(self):
        self.assertIsNone(encode_embedding(None))

    def test_none_decodes_to_none(self):
        self.assertIsNone(decode_embedding(None))

    def test_float64_precision_survives_within_float32_tolerance(self):
        values = [0.123456789, 0.987654321, -0.555555555]
        np.testing.assert_allclose(decode_embedding(encode_embedding(values)), values, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. Schema bootstrap
# ---------------------------------------------------------------------------

class TestSchemaBootstrap(_TempDbTestCase):

    def test_database_file_is_created(self):
        self.assertTrue(os.path.exists(self.db_path))

    def test_articles_table_exists_with_expected_columns(self):
        with sqlite3.connect(self.db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
        self.assertEqual(cols, {'id', 'title', 'date', 'url', 'embedding', 'title_translated'})

    def test_reopening_an_existing_database_preserves_rows(self):
        self.store.insert('id-1', 'Persisted headline', '2026-08-10T09:00:00')
        reopened = SqliteStore(self.db_path)
        self.assertEqual([r['title'] for r in reopened.fetch_all()], ['Persisted headline'])

    def test_wal_journal_mode_is_enabled(self):
        with sqlite3.connect(self.db_path) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), 'wal')


# ---------------------------------------------------------------------------
# 3. Insert / fetch
# ---------------------------------------------------------------------------

class TestInsertAndFetch(_TempDbTestCase):

    def test_fetch_all_on_empty_database_returns_empty_list(self):
        self.assertEqual(self.store.fetch_all(), [])

    def test_inserted_row_is_returned_by_fetch_all(self):
        self.assertTrue(self.store.insert(
            'id-1', 'Málaga airport sets record', '2026-08-10T09:00:00',
            url='https://example.com/a', embedding=[0.1, 0.2],
            title_translated='Málaga airport sets record',
        ))
        rows = self.store.fetch_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['title'], 'Málaga airport sets record')
        self.assertEqual(rows[0]['url'], 'https://example.com/a')

    def test_fetch_all_projects_only_title_url_embedding(self):
        self.store.insert('id-1', 'A headline', '2026-08-10T09:00:00', url='https://example.com/a')
        self.assertEqual(set(self.store.fetch_all()[0]), {'title', 'url', 'embedding'})

    def test_embedding_survives_the_round_trip(self):
        embedding = [0.5, -0.25, 0.125]
        self.store.insert('id-1', 'A headline', '2026-08-10T09:00:00', embedding=embedding)
        np.testing.assert_allclose(self.store.fetch_all()[0]['embedding'], embedding, rtol=1e-6)

    def test_null_embedding_is_returned_as_none(self):
        self.store.insert('id-1', 'A headline', '2026-08-10T09:00:00', embedding=None)
        self.assertIsNone(self.store.fetch_all()[0]['embedding'])

    def test_missing_url_is_returned_as_none(self):
        self.store.insert('id-1', 'A headline', '2026-08-10T09:00:00')
        self.assertIsNone(self.store.fetch_all()[0]['url'])

    def test_title_translated_is_persisted(self):
        self.store.insert('id-1', 'Titular original', '2026-08-10T09:00:00',
                          title_translated='Original headline')
        with sqlite3.connect(self.db_path) as conn:
            stored = conn.execute("SELECT title_translated FROM articles WHERE id = 'id-1'").fetchone()[0]
        self.assertEqual(stored, 'Original headline')

    def test_duplicate_id_is_rejected(self):
        self.store.insert('id-1', 'First', '2026-08-10T09:00:00')
        with _silenced():
            self.assertFalse(self.store.insert('id-1', 'Second', '2026-08-10T10:00:00'))

    def test_duplicate_url_is_rejected(self):
        """The partial unique index on url must surface as a False return, not a crash."""
        self.store.insert('id-1', 'First', '2026-08-10T09:00:00', url='https://example.com/a')
        with _silenced():
            self.assertFalse(self.store.insert('id-2', 'Second', '2026-08-10T10:00:00', url='https://example.com/a'))

    def test_rejected_duplicate_does_not_add_a_row(self):
        self.store.insert('id-1', 'First', '2026-08-10T09:00:00', url='https://example.com/a')
        with _silenced():
            self.store.insert('id-2', 'Second', '2026-08-10T10:00:00', url='https://example.com/a')
        self.assertEqual(len(self.store.fetch_all()), 1)

    def test_multiple_null_urls_are_allowed(self):
        """The unique index is partial (WHERE url IS NOT NULL), so NULLs must not collide."""
        self.store.insert('id-1', 'First', '2026-08-10T09:00:00')
        self.assertTrue(self.store.insert('id-2', 'Second', '2026-08-10T10:00:00'))

    def test_insert_failure_returns_false_instead_of_raising(self):
        """main.py treats a False return as 'save failed' — it must never see an exception."""
        with _silenced():
            broken = SqliteStore(os.path.join(self.tmpdir, 'nonexistent-dir', 'articles.db'))
            self.assertFalse(broken.insert('id-1', 'A headline', '2026-08-10T09:00:00'))

    def test_fetch_failure_returns_empty_list_instead_of_raising(self):
        with _silenced():
            broken = SqliteStore(os.path.join(self.tmpdir, 'nonexistent-dir', 'articles.db'))
            self.assertEqual(broken.fetch_all(), [])


# ---------------------------------------------------------------------------
# 4. Retention cutoff
# ---------------------------------------------------------------------------

class TestDeleteOlderThan(_TempDbTestCase):

    def setUp(self):
        super().setUp()
        self.store.insert('old', 'Old article', '2026-08-01T12:00:00')
        self.store.insert('boundary', 'Boundary article', '2026-08-05T12:00:00')
        self.store.insert('recent', 'Recent article', '2026-08-09T12:00:00')

    def _remaining_titles(self):
        return sorted(r['title'] for r in self.store.fetch_all())

    def test_returns_number_of_deleted_rows(self):
        self.assertEqual(self.store.delete_older_than('2026-08-05T12:00:00'), 1)

    def test_row_exactly_at_the_cutoff_is_retained(self):
        self.store.delete_older_than('2026-08-05T12:00:00')
        self.assertIn('Boundary article', self._remaining_titles())

    def test_older_rows_are_deleted(self):
        self.store.delete_older_than('2026-08-05T12:00:00')
        self.assertNotIn('Old article', self._remaining_titles())

    def test_newer_rows_are_retained(self):
        self.store.delete_older_than('2026-08-05T12:00:00')
        self.assertIn('Recent article', self._remaining_titles())

    def test_cutoff_before_everything_deletes_nothing(self):
        self.assertEqual(self.store.delete_older_than('2026-01-01T00:00:00'), 0)

    def test_cutoff_after_everything_deletes_all(self):
        self.assertEqual(self.store.delete_older_than('2027-01-01T00:00:00'), 3)

    def test_delete_failure_returns_zero_instead_of_raising(self):
        with _silenced():
            broken = SqliteStore(os.path.join(self.tmpdir, 'nonexistent-dir', 'articles.db'))
            self.assertEqual(broken.delete_older_than('2026-08-05T12:00:00'), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
