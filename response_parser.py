import re
import json
import logging

logger = logging.getLogger(__name__)


def parse_summary_with_emojis(response_text):
    return re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()


def parse_summary_with_emojis_and_evaluate(response_text):
    cleaned_response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

    scores = {"expat_impact": 0, "malaga_relevance": 0, "feature_vs_politics": 0}
    scores_match = re.search(r"Scores:\s*E:(\d{1,2})\s*M:(\d{1,2})\s*P:(\d{1,2})", cleaned_response_text, re.IGNORECASE)

    summary_text = cleaned_response_text
    if scores_match:
        try:
            scores["expat_impact"] = int(scores_match.group(1))
            scores["malaga_relevance"] = int(scores_match.group(2))
            scores["feature_vs_politics"] = int(scores_match.group(3))
            summary_text = re.sub(r"Scores:\s*E:\d{1,2}\s*M:\d{1,2}\s*P:\d{1,2}", "", cleaned_response_text, flags=re.IGNORECASE).strip()
        except ValueError:
            logger.warning(f"Could not parse scores from AI response: {scores_match.groups()}")
    else:
        logger.warning(f"Scores pattern not found in AI response: '{cleaned_response_text}'")

    expat_impact = scores.get("expat_impact", 0)
    malaga_relevance = scores.get("malaga_relevance", 0)
    feature_vs_politics = scores.get("feature_vs_politics", 0)
    final_score = (expat_impact + malaga_relevance + feature_vs_politics) / len(scores) if scores else 0

    return summary_text, final_score


_EVALUATION_KEYS = ("expat_impact", "event_weight", "politics", "timeliness", "practical_utility")


def parse_evaluate_article(response_text):
    """Parses the raw LLM response into {'score': float, 'breakdown': dict|None}.

    Bot logic only needs `score`; `breakdown` is for dry-run analysis.
    On JSON decode failure returns {'score': 0, 'breakdown': None}.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'//.*', '', cleaned)
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from response: {cleaned}")
        return {"score": 0, "breakdown": None}

    breakdown = {key: obj.get(key, 0) for key in _EVALUATION_KEYS}
    non_zero = [v for v in breakdown.values() if v != 0]
    avg = sum(non_zero) / len(non_zero) if non_zero else 0
    return {"score": avg, "breakdown": breakdown}
