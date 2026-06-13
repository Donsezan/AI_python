import re
import json
import logging

logger = logging.getLogger(__name__)

_EVALUATION_KEYS = ("expat_impact", "event_weight", "politics", "timeliness", "practical_utility")


def _strip_wrappers(response_text):
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    return re.sub(r'\s*```$', '', cleaned).strip()


def parse_evaluate_and_summarize(response_text):
    """Parses the combined LLM response into {'score': float, 'breakdown': dict, 'summary': str}.

    The score is the mean of all five dimensions — a 0 (e.g. pure party
    politics) lowers the average rather than being excluded. Returns None when
    the response is not valid JSON, so callers can distinguish "model failed"
    from "article legitimately scored low".
    """
    cleaned = _strip_wrappers(response_text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from response: {cleaned}")
        return None
    if not isinstance(obj, dict):
        logger.error(f"Expected JSON object, got {type(obj).__name__}: {cleaned}")
        return None

    breakdown = {key: obj.get(key, 0) for key in _EVALUATION_KEYS}
    score = sum(breakdown.values()) / len(_EVALUATION_KEYS)
    summary = (obj.get("summary") or "").strip()
    return {"score": score, "breakdown": breakdown, "summary": summary}
