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
    def summarize_with_emojis(self, article_text, target_language='en', source_language='es') -> str:
        """
        Summarizes the given article text with emojis.

        :param article_text: The text of the article to summarize.
        :param target_language: The target language for the summary.
        :param source_language: The language the article is written in (ISO 639-1 code).
        :return: The summarized text with emojis.
        """
        ...

    @abstractmethod
    def evaluate_article(self, article_text, source_language='es') -> Dict[str, Any]:
        """
        Evaluates the article based on predefined criteria.

        :param article_text: The text of the article to evaluate.
        :param source_language: The language the article is written in (ISO 639-1 code).
        :return: {'score': float, 'breakdown': dict|None}. Bot logic uses `score`
                 (averaged); `breakdown` exposes per-dimension integers
                 (expat_impact, event_weight, politics, timeliness,
                 practical_utility) for dry-run analysis. `breakdown` is None
                 when the response failed to parse.
        """
        ...
