"""CPU-only tests of incomplete-counter handling in campaign summaries."""
import unittest
from summarize_session import summarize


class SummaryTests(unittest.TestCase):
    def report(self, reads=0, fallback=False):
        return {'model': 'example', 'status': 'pass', 'device': 'tnca',
                'prompts': [{'id': 'fact', 'passed': True, 'turns': [{
                    'device_evidence': {'cpu_numerical_fallback': fallback},
                    'user_visible_timing': {'non_special_token_ids_per_second': 0.5},
                    'steps': [{'kind': 'decode', 'transfers_inside_step':
                               {'host_read_calls': reads, 'staging_download_bytes': 0}}]}]}],
                'warm_repeat': {'passed': True}, 'scripted_chat': {'passed': True}}

    def test_zero_and_nonzero_counters(self):
        clean = summarize(self.report(), 'input')
        self.assertTrue(clean['quality_no_cpu_numerical_fallback'])
        self.assertTrue(clean['quality_zero_decode_host_reads'])
        self.assertEqual(clean['fact_cold']['non_special_token_ids_per_second'], 0.5)
        bad = summarize(self.report(reads=1, fallback=True), 'input')
        self.assertFalse(bad['quality_no_cpu_numerical_fallback'])
        self.assertFalse(bad['quality_zero_decode_host_reads'])

    def test_missing_counters_are_unknown(self):
        unknown = summarize(self.report(reads=None, fallback=None), 'input')
        self.assertIsNone(unknown['quality_no_cpu_numerical_fallback'])
        self.assertIsNone(unknown['quality_zero_decode_host_reads'])
        self.assertIsNone(unknown['fact_warm']['prompt_to_return_seconds'])

    def test_no_prompts_are_not_an_implicit_pass(self):
        empty = summarize({'prompts': []}, 'empty')
        self.assertIsNone(empty['quality_no_cpu_numerical_fallback'])
        self.assertIsNone(empty['quality_zero_decode_host_reads'])


if __name__ == '__main__':
    unittest.main()
