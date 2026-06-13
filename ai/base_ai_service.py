from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class RateLimitError(Exception):
    """Raised when an AI provider returns a rate-limit / quota error.

    ``retry_after`` is the seconds the provider asked us to wait, when known.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class BaseAIService(ABC):
    """
    Abstract base class for AI services. It defines the interface that all AI provider-specific
    services must implement.
    """

    @abstractmethod
    def evaluate_and_summarize(self, article_text, source_language='es', target_language='en') -> Optional[Dict[str, Any]]:
        """
        Scores and summarizes the article in a single LLM call (one request per
        article instead of two — requests, not tokens, are the scarce resource
        on free tiers).

        :param article_text: The text of the article to process.
        :param source_language: The language the article is written in (ISO 639-1 code).
        :param target_language: The target language for the summary.
        :return: {'score': float, 'breakdown': dict, 'summary': str} where
                 `score` is the mean of the five per-dimension integers
                 (expat_impact, event_weight, politics, timeliness,
                 practical_utility) and `summary` is the emoji-rich Telegram
                 text. None when the response failed to parse.
        """
        ...
