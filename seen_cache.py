import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class SeenCache:
    """Persistent URL → outcome cache that stops re-processing articles across cycles.

    Articles that reach a verdict the bot can't act on (too old, page broken,
    no content, repeatedly failing) are not saved to Supabase, so without this
    cache they would be re-fetched — and re-embedded / re-evaluated, burning
    free-tier quota — on every 10-minute cycle while they remain on the homepage.

    Terminal statuses skip the URL forever (until pruned). Transient statuses
    increment an attempt counter and become terminal after ``max_attempts``.
    Entries are pruned once not seen for ``max_age_days`` (longer than the
    homepage lifetime of any article).
    """

    TERMINAL_STATUSES = {"too_old", "no_content", "post_failed", "failed"}

    def __init__(self, path="seen_cache.json", max_attempts=3, max_age_days=14):
        self.path = Path(path)
        self.max_attempts = max_attempts
        self.max_age_days = max_age_days
        self._entries = {}
        self._dirty = False
        self._load()

    def _now(self):
        return datetime.now(timezone.utc)

    def _load(self):
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as f:
                self._entries = json.load(f)
        except (ValueError, OSError) as e:
            logger.warning(f"[seen-cache] Could not load {self.path}: {e}. Starting empty.")
            self._entries = {}
        if self._prune():
            self.flush()

    def _prune(self):
        cutoff = (self._now() - timedelta(days=self.max_age_days)).isoformat()
        stale = [url for url, e in self._entries.items() if e.get("last_seen", "") < cutoff]
        for url in stale:
            del self._entries[url]
        if stale:
            logger.info(f"[seen-cache] Pruned {len(stale)} stale entries.")
            self._dirty = True
        return bool(stale)

    def should_skip(self, url):
        """True when the URL has a terminal status or exhausted its attempts.

        Refreshes ``last_seen`` on hits so entries aren't pruned while the
        article is still listed on the homepage (refresh is persisted on the
        next ``flush()``).
        """
        entry = self._entries.get(url)
        if entry is None:
            return False
        skip = (
            entry["status"] in self.TERMINAL_STATUSES
            or entry.get("attempts", 0) >= self.max_attempts
        )
        if skip:
            entry["last_seen"] = self._now().isoformat()
            self._dirty = True
        return skip

    def record_terminal(self, url, status):
        """Mark a URL as permanently handled (skipped on every future cycle)."""
        self._entries[url] = {
            "status": status,
            "attempts": self.max_attempts,
            "last_seen": self._now().isoformat(),
        }
        self.flush()

    def record_attempt(self, url, status):
        """Count one failed attempt; the URL turns terminal after max_attempts.

        Returns the attempt count so callers can log progress.
        """
        entry = self._entries.get(url) or {"attempts": 0}
        entry["attempts"] = entry.get("attempts", 0) + 1
        entry["status"] = status
        entry["last_seen"] = self._now().isoformat()
        self._entries[url] = entry
        self.flush()
        return entry["attempts"]

    def flush(self):
        self._prune()
        tmp = self.path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=1)
            tmp.replace(self.path)
            self._dirty = False
        except OSError as e:
            logger.error(f"[seen-cache] Failed to write {self.path}: {e}")

    def flush_if_dirty(self):
        if self._dirty:
            self.flush()
