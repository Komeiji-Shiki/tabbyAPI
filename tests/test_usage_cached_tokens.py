import unittest

from endpoints.OAI.types.common import PromptTokensDetails, UsageStats
from endpoints.OAI.utils.common_ import aggregate_usage_stats, get_usage_stats


class UsageCachedTokensTests(unittest.TestCase):
    def test_get_usage_stats_exposes_cached_tokens(self):
        generation = {
            "finish_reason": "stop",
            "prompt_tokens": 3353,
            "prompt_time": 0.72,
            "prompt_tokens_per_sec": 390.28,
            "gen_tokens": 55,
            "gen_time": 1.7,
            "gen_tokens_per_sec": 32.3,
            "total_time": 2.46,
            "cached_tokens": 3072,
        }

        usage = get_usage_stats(generation)

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.prompt_tokens, 3353)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 3072)
        self.assertEqual(usage.total_tokens, 3408)
        self.assertEqual(usage.completion_tokens, 55)

    def test_get_usage_stats_clamps_cached_tokens_to_prompt_tokens(self):
        generation = {
            "finish_reason": "stop",
            "prompt_tokens": 100,
            "gen_tokens": 10,
            "cached_tokens": 120,
        }

        usage = get_usage_stats(generation)

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 100)

    def test_aggregate_usage_stats_keeps_prompt_cached_tokens(self):
        first = UsageStats(
            prompt_tokens=1000,
            prompt_time=1.0,
            prompt_tokens_per_sec=100.0,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=768),
            completion_tokens=10,
            completion_time=0.5,
            completion_tokens_per_sec=20.0,
            total_tokens=1010,
            total_time=1.5,
        )
        second = UsageStats(
            prompt_tokens=1000,
            prompt_time=1.0,
            prompt_tokens_per_sec=100.0,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=768),
            completion_tokens=20,
            completion_time=0.75,
            completion_tokens_per_sec=26.666666666666668,
            total_tokens=1020,
            # Includes scheduling/requeue overhead beyond prompt + decode.
            total_time=4.0,
        )

        usage = aggregate_usage_stats([first, second])

        self.assertEqual(usage.prompt_tokens, 1000)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 768)
        self.assertEqual(usage.completion_tokens, 30)
        self.assertEqual(usage.total_tokens, 1030)
        self.assertEqual(usage.total_time, 4.0)


if __name__ == "__main__":
    unittest.main()
