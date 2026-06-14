import unittest
from unittest.mock import patch, MagicMock

# Add project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.openai_service import OpenAIService
from ai.gemini_service import GeminiService, GeminiRateLimitError

_VALID_JSON = (
    '{"expat_impact": 8, "event_weight": 7, "politics": 6, "timeliness": 9, '
    '"practical_utility": 5, "title": "Test headline", "summary": "Test summary. \U0001F603"}'
)


class TestOpenAIService(unittest.TestCase):

    def setUp(self):
        self.service = OpenAIService()

    def _mock_response(self, text):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": text}}]}
        return mock_resp

    @patch('ai.openai_service.requests.post')
    def test_evaluate_and_summarize(self, mock_post):
        mock_post.return_value = self._mock_response(_VALID_JSON)

        result = self.service.evaluate_and_summarize("Test article", "Titulo de prueba")

        self.assertAlmostEqual(result["score"], 7.0)
        self.assertEqual(result["breakdown"]["expat_impact"], 8)
        self.assertEqual(result["breakdown"]["politics"], 6)
        self.assertEqual(result["summary"], "Test summary. \U0001F603")
        self.assertEqual(result["title"], "Test headline")
        mock_post.assert_called_once()
        messages = mock_post.call_args.kwargs['json']['messages']
        self.assertIn("Test article", messages[1]['content'])
        self.assertIn("Titulo de prueba", messages[1]['content'])

    @patch('ai.openai_service.requests.post')
    def test_evaluate_and_summarize_invalid_json(self, mock_post):
        mock_post.return_value = self._mock_response("not json at all")

        self.assertIsNone(self.service.evaluate_and_summarize("Test article", "Titulo"))


class TestGeminiService(unittest.TestCase):

    def setUp(self):
        self.service = GeminiService(api_key="test_key")
        GeminiService._last_call_at = 0.0
        GeminiService._use_legacy_schema = False

    def _mock_response(self, text, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.ok = status_code < 400
        mock_resp.text = text
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": text}]},
                "finishReason": "STOP",
            }]
        }
        return mock_resp

    @patch('ai.gemini_service.requests.post')
    def test_evaluate_and_summarize(self, mock_post):
        mock_post.return_value = self._mock_response(_VALID_JSON)

        result = self.service.evaluate_and_summarize("Test article", "Titulo de prueba")

        self.assertAlmostEqual(result["score"], 7.0)
        self.assertEqual(result["breakdown"]["expat_impact"], 8)
        self.assertEqual(result["summary"], "Test summary. \U0001F603")
        self.assertEqual(result["title"], "Test headline")
        mock_post.assert_called_once()
        body = mock_post.call_args.kwargs['json']
        self.assertIn("Test article", body['contents'][0]['parts'][0]['text'])
        self.assertIn("Titulo de prueba", body['contents'][0]['parts'][0]['text'])
        # Primary path uses the standard-JSON-Schema field, schema unmodified.
        config = body['generationConfig']
        self.assertIn('responseJsonSchema', config)
        self.assertIn('summary', config['responseJsonSchema']['properties'])
        self.assertIn('title', config['responseJsonSchema']['properties'])
        self.assertEqual(config['thinkingConfig']['thinkingBudget'], 0)

    @patch('ai.gemini_service.time.sleep')
    @patch('ai.gemini_service.requests.post')
    def test_falls_back_to_legacy_schema_on_400(self, mock_post, _mock_sleep):
        rejected = MagicMock()
        rejected.status_code = 400
        rejected.ok = False
        rejected.text = 'Unknown name "responseJsonSchema" in generation_config'
        mock_post.side_effect = [rejected, self._mock_response(_VALID_JSON)]

        result = self.service.evaluate_and_summarize("Test article", "Titulo")

        self.assertAlmostEqual(result["score"], 7.0)
        self.assertEqual(mock_post.call_count, 2)
        self.assertTrue(GeminiService._use_legacy_schema)
        retry_config = mock_post.call_args.kwargs['json']['generationConfig']
        self.assertNotIn('responseJsonSchema', retry_config)
        # Legacy schema is sanitized to the OpenAPI subset with uppercase types.
        self.assertEqual(retry_config['responseSchema']['type'], 'OBJECT')
        self.assertNotIn('additionalProperties', retry_config['responseSchema'])

    @patch('ai.gemini_service.requests.post')
    def test_429_raises_rate_limit_error_with_retry_after(self, mock_post):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.ok = False
        resp_429.headers = {"Retry-After": "17"}
        mock_post.return_value = resp_429

        with self.assertRaises(GeminiRateLimitError) as ctx:
            self.service.evaluate_and_summarize("Test article", "Titulo")
        self.assertEqual(ctx.exception.retry_after, 17.0)


if __name__ == '__main__':
    unittest.main()
