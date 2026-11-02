"""Check that compile-confounded board results are not presented as warm A/B."""
import unittest
from pathlib import Path
from types import SimpleNamespace

from bench_chat_board import comparison_scope, matched_warm_wall_speedup, model_command


def row(*, available=True, misses=0, compiles=0):
    return {'timing': {'prompt_to_return_seconds': 10, 'compiler': {'available': available,
                                    'fx_graph_cache_misses': misses,
                                    'dynamo_compile_calls': compiles}}}


class ComparisonScopeTest(unittest.TestCase):
    def test_model_command_selects_the_requested_probe(self):
        args = SimpleNamespace(python=Path('/venv/python'), model='q35', layers=4,
                               prompt='question', new_tokens=4, hybrid_eos_continuation=True,
                               device_eos_probe=False, early_eos_probe=False, static_prefix_probe=False,
                               static_group_suffix=False, static_normal_chat=False)
        root, out = Path('/overlay'), Path('/results')
        command = model_command(args, root, out)
        self.assertEqual(command[2], '/overlay/tools/r41/chat_speed/bench_hybrid_continue_entry.py')
        self.assertEqual(command[-1], '/results')
        args.hybrid_eos_continuation, args.early_eos_probe = False, True
        self.assertEqual(model_command(args, root, out)[2], '/overlay/tools/r41/chat_speed/bench_eos_model_entry.py')
        args.early_eos_probe, args.static_prefix_probe = False, True
        self.assertEqual(model_command(args, root, out)[2], '/overlay/tools/r41/chat_speed/bench_static_continue_entry.py')
        args.static_group_suffix, args.static_normal_chat = True, True
        self.assertEqual(model_command(args, root, out)[-2:], ['--group-suffix', '--normal-chat-eos'])

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

    def test_speedup_is_withheld_for_timed_or_unobserved_compilation(self):
        baseline, optimized = row(), row()
        optimized['timing']['prompt_to_return_seconds'] = 4
        self.assertEqual(2.5, matched_warm_wall_speedup(
            [baseline, optimized], device_eos_probe=True, warm_cache=True))
        for bad in (row(misses=1, compiles=1), row(compiles=1), row(available=False), {}):
            self.assertIsNone(matched_warm_wall_speedup(
                [bad, optimized], device_eos_probe=True, warm_cache=True))
        optimized['timing']['prompt_to_return_seconds'] = 0
        self.assertIsNone(matched_warm_wall_speedup(
            [baseline, optimized], device_eos_probe=True, warm_cache=True))


if __name__ == '__main__':
    unittest.main()
