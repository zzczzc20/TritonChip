"""Check that compile-confounded board results are not presented as warm A/B."""
import unittest

from bench_chat_board import comparison_scope


def row(*, available=True, misses=0, compiles=0):
    return {'timing': {'compiler': {'available': available,
                                    'fx_graph_cache_misses': misses,
                                    'dynamo_compile_calls': compiles}}}


class ComparisonScopeTest(unittest.TestCase):
    def test_in_process_warm_requires_observed_zero_compilation(self):
        clean = [row(), row()]
        self.assertIn('matched in-process warm', comparison_scope(
            clean, device_eos_probe=True, warm_cache=True))
        for bad in (row(misses=1, compiles=1), row(compiles=1), row(available=False), {}):
            self.assertIn('not a matched', comparison_scope(
                [bad, row()], device_eos_probe=True, warm_cache=True))

    def test_process_cache_hits_do_not_exclude_compile_time(self):
        self.assertIn('matched warm-cache', comparison_scope(
            [row(), row()], device_eos_probe=False, warm_cache=True))
        self.assertIn('timed compilation', comparison_scope(
            [row(compiles=1), row()], device_eos_probe=False, warm_cache=True))

    def test_cold_start_is_never_matched(self):
        self.assertIn('not a matched', comparison_scope(
            [row(), row()], device_eos_probe=False, warm_cache=False))


if __name__ == '__main__':
    unittest.main()
