"""One-off migration: copy the Supabase `articles` table into local SQLite.

Run once during the cutover, while SUPABASE_URL / SUPABASE_KEY are still in
`.env`. Preserves dedup history so the first live cycle after the switch does
not re-post articles that were already handled.

Idempotent: rows are written with INSERT OR IGNORE keyed on `id`, so a partial
or repeated run is safe.

    python3 scripts/migrate_supabase_to_sqlite.py [--dry-run] [--db-path articles.db]

Reads `.env` directly rather than importing `main`, whose import-time env check
no longer accepts Supabase credentials.
"""

import argparse
import json
import os
import sqlite3
import sys

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlite_store import SqliteStore, encode_embedding

_COLUMNS = "id,title,date,url,embedding,title_translated"
_PAGE_SIZE = 1000


def load_env():
    """Parse `.env` into os.environ without depending on python-dotenv."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    try:
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))
    except FileNotFoundError:
        pass


def row_to_params(row):
    """Convert one PostgREST row into SQLite insert parameters.

    The jsonb `embedding` arrives as a list (or occasionally as a JSON string);
    both become a float32 BLOB. Missing or NULL stays NULL, which keeps the
    Jaccard fallback working for legacy rows that never had an embedding.
    """
    embedding = row.get("embedding")
    if isinstance(embedding, str):
        embedding = json.loads(embedding)
    return {
        "id": row["id"],
        "title": row["title"],
        "date": row["date"],
        "url": row.get("url"),
        "embedding": encode_embedding(embedding),
        "title_translated": row.get("title_translated"),
    }


def fetch_rows(supabase_url, supabase_key):
    """Page the whole table via the PostgREST Range header."""
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/articles"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    rows = []
    offset = 0
    while True:
        resp = requests.get(
            endpoint,
            headers={**headers, "Range": f"{offset}-{offset + _PAGE_SIZE - 1}"},
            params={"select": _COLUMNS, "order": "id"},
            timeout=60,
        )
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        print(f"  fetched {len(rows)} rows...")
        if len(page) < _PAGE_SIZE:
            return rows
        offset += _PAGE_SIZE


def write_rows(db_path, rows, dry_run=False):
    """Insert rows, skipping ids already present. Returns the number inserted."""
    if dry_run:
        return 0
    inserted = 0
    conn = sqlite3.connect(db_path, timeout=15)
    try:
        for row in rows:
            params = row_to_params(row)
            cursor = conn.execute(
                "INSERT OR IGNORE INTO articles (id, title, date, url, embedding, title_translated)"
                " VALUES (:id, :title, :date, :url, :embedding, :title_translated)",
                params,
            )
            inserted += cursor.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='report row counts without writing to the database')
    parser.add_argument('--db-path', default=os.getenv('DB_PATH', 'articles.db'),
                        help='destination SQLite file (default: articles.db)')
    args = parser.parse_args()

    load_env()
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    if not supabase_url or not supabase_key:
        print("SUPABASE_URL and SUPABASE_KEY must be set in .env to run the migration.")
        return 1

    SqliteStore(args.db_path)  # bootstrap the schema before the bulk insert
    print(f"Reading articles from Supabase into '{args.db_path}'...")
    rows = fetch_rows(supabase_url, supabase_key)
    inserted = write_rows(args.db_path, rows, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[dry run] fetched {len(rows)} / inserted 0 / skipped {len(rows)} (nothing written)")
    else:
        print(f"\nfetched {len(rows)} / inserted {inserted} / skipped {len(rows) - inserted}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
