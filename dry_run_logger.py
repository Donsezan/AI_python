import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class DryRunLogger:
    """Collects evaluation results during a dry-run and persists them to a timestamped JSON file.

    Each dry-run produces one self-contained file under `output_dir/` for offline analysis
    (e.g. comparing score distributions before/after prompt changes).
    """

    def __init__(self, output_dir="dry_run_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"dry_run_{ts}.json"
        self._records = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"[dry-run] Logging results to {self.path}")

    def record(self, *, title, url, status, score=None, breakdown=None, summary=None,
               language=None, article_chars=None, error=None):
        """Append one article result and persist atomically.

        status values: 'evaluated_above_threshold', 'below_threshold',
        'evaluate_failed', 'summary_failed', 'fetch_failed', 'too_old',
        'no_content', 'duplicate_cached'.

        breakdown: per-dimension scores dict (expat_impact, event_weight,
        politics, timeliness, practical_utility) — critical for analyzing
        whether prompt changes affected specific dimensions.
        """
        self._records.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "url": url,
            "status": status,
            "score": score,
            "breakdown": breakdown,
            "summary": summary,
            "language": language,
            "article_chars": article_chars,
            "error": error,
        })
        self._flush()

    def _flush(self):
        payload = {
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self._records),
            "results": self._records,
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def close(self):
        self._flush()
        logger.info(f"[dry-run] Wrote {len(self._records)} records to {self.path}")
