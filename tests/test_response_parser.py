import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import response_parser


class TestParseEvaluateAndSummarize(unittest.TestCase):

    def test_valid_response(self):
        result = response_parser.parse_evaluate_and_summarize(
            '{"expat_impact": 8, "event_weight": 7, "politics": 6, "timeliness": 9, '
            '"practical_utility": 5, "summary": "Hello. \U0001F603"}'
        )
        self.assertAlmostEqual(result["score"], 7.0)
        self.assertEqual(result["breakdown"]["timeliness"], 9)
        self.assertEqual(result["summary"], "Hello. \U0001F603")

    def test_politics_zero_lowers_average(self):
        # A 0 must count against the score, not be excluded from the mean.
        result = response_parser.parse_evaluate_and_summarize(
            '{"expat_impact": 10, "event_weight": 10, "politics": 0, "timeliness": 10, '
            '"practical_utility": 10, "summary": "s"}'
        )
        self.assertAlmostEqual(result["score"], 8.0)

    def test_markdown_fences_and_think_tags_stripped(self):
        result = response_parser.parse_evaluate_and_summarize(
            '<think>internal musing</think>```json\n'
            '{"expat_impact": 5, "event_weight": 5, "politics": 5, "timeliness": 5, '
            '"practical_utility": 5, "summary": "s"}\n```'
        )
        self.assertAlmostEqual(result["score"], 5.0)

    def test_invalid_json_returns_none(self):
        self.assertIsNone(response_parser.parse_evaluate_and_summarize("garbage"))

    def test_non_object_json_returns_none(self):
        self.assertIsNone(response_parser.parse_evaluate_and_summarize("[1, 2, 3]"))

    def test_missing_keys_default_to_zero(self):
        result = response_parser.parse_evaluate_and_summarize('{"summary": "s"}')
        self.assertAlmostEqual(result["score"], 0.0)
        self.assertEqual(result["breakdown"]["expat_impact"], 0)


if __name__ == '__main__':
    unittest.main()
