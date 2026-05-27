import unittest
from unittest.mock import patch, MagicMock

# Add project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.openai_service import OpenAIService
from ai.gemini_service import GeminiService
import response_parser

class TestOpenAIService(unittest.TestCase):

    def setUp(self):
        self.service = OpenAIService()

    def _mock_response(self, text):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": text}}]}
        return mock_resp

    @patch('ai.openai_service.requests.post')
    def test_summarize_with_emojis(self, mock_post):
        mock_post.return_value = self._mock_response("Test summary. 😃")

        summary = self.service.summarize_with_emojis("Test article")

        self.assertEqual(summary, "Test summary. 😃")
        mock_post.assert_called_once()
        messages = mock_post.call_args.kwargs['json']['messages']
        self.assertIn("Test article", messages[1]['content'])

    @patch('ai.openai_service.requests.post')
    def test_evaluate_article(self, mock_post):
        mock_post.return_value = self._mock_response(
            '{"expat_impact": 8, "event_weight": 7, "politics": 6, "timeliness": 9, "practical_utility": 5}'
        )

        result = self.service.evaluate_article("Test article")

        self.assertAlmostEqual(result["score"], 7.0)
        self.assertEqual(result["breakdown"]["expat_impact"], 8)
        self.assertEqual(result["breakdown"]["politics"], 6)
        mock_post.assert_called_once()

class TestGeminiService(unittest.TestCase):

    def setUp(self):
        self.service = GeminiService(api_key="test_key")

    def _mock_response(self, text):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": text}]},
                "finishReason": "STOP",
            }]
        }
        return mock_resp

    @patch('ai.gemini_service.requests.post')
    def test_summarize_with_emojis(self, mock_post):
        mock_post.return_value = self._mock_response("Test summary. 😃")

        summary = self.service.summarize_with_emojis("Test article")

        self.assertEqual(summary, "Test summary. 😃")
        mock_post.assert_called_once()
        sent_text = mock_post.call_args.kwargs['json']['contents'][0]['parts'][0]['text']
        self.assertIn("Test article", sent_text)

    @patch('ai.gemini_service.requests.post')
    def test_evaluate_article(self, mock_post):
        mock_post.return_value = self._mock_response(
            '{"expat_impact": 8, "event_weight": 7, "politics": 6, "timeliness": 9, "practical_utility": 5}'
        )

        result = self.service.evaluate_article("Test article")

        self.assertAlmostEqual(result["score"], 7.0)
        self.assertEqual(result["breakdown"]["expat_impact"], 8)
        self.assertEqual(result["breakdown"]["politics"], 6)
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
