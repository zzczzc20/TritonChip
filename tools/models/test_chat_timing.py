#!/usr/bin/env python3
"""Observed prompt timing distinguishes returned IDs from executed decode work."""
import unittest
from chat_timing import compiler_delta, compiler_snapshot, finalize_user_visible_metrics, user_visible_metrics


class TimingTests(unittest.TestCase):
    def test_warmup_retrieval_and_discarded_decode_are_visible(self):
        report = {'generated_tokens': [42, 2], 'decode_warmup': {'seconds': 3.0},
                  'output_retrieval_seconds': 0.5,
                  'statistics': {'prefill_seconds': 2.0, 'decode_seconds': 4.0},
                  'steps': [{'kind': 'prefill', 'seconds': 2.0},
                            {'kind': 'decode', 'seconds': 1.0},
                            {'kind': 'decode', 'seconds': 3.0}]}
        result = user_visible_metrics(report, 12.0, special_ids=[2], reference_seconds=1.0)
        self.assertEqual(result['returned_token_ids'], 2)
        self.assertEqual(result['non_special_token_ids'], 1)
        self.assertEqual(result['returned_token_ids_per_second'], 2 / 12)
        self.assertEqual(result['non_special_token_ids_per_second'], 1 / 12)
        self.assertEqual(result['decode_step_seconds'], [1.0, 3.0])
        self.assertEqual(result['other_wall_seconds'], 1.5)
        self.assertFalse(result['accounted_span_exceeds_wall'])

    def test_missing_spans_and_overlaps_are_reported(self):
        result = user_visible_metrics({'generated_tokens': [3], 'statistics': {'prefill_seconds': 5}}, 4)
        self.assertTrue(result['accounted_span_exceeds_wall'])
        self.assertEqual(result['other_wall_seconds'], 0)
        self.assertIsNone(user_visible_metrics({}, 0)['returned_token_ids_per_second'])
        with self.assertRaises(ValueError):
            user_visible_metrics({}, -1)

    def test_retained_prefill_is_attributed_without_double_counting_warmup(self):
        report = {'generated_tokens': [1], 'decode_warmup': {'seconds': 5.0},
                  'statistics': {'prefill_seconds': 2.0, 'prefill_seconds_already_in_warmup': 1.5,
                                 'decode_seconds': 3.0}}
        timing = user_visible_metrics(report, 10.0)
        self.assertEqual(timing['other_wall_seconds'], 1.5)
        self.assertFalse(timing['accounted_span_exceeds_wall'])
        finalize_user_visible_metrics(timing, 11.0, 'through stdout')
        self.assertEqual(timing['other_wall_seconds'], 2.5)
        report['statistics']['prefill_seconds_already_in_warmup'] = 6.0
        with self.assertRaises(ValueError):
            user_visible_metrics(report, 10.0)

    def test_real_decode_inside_warmup_is_counted_once(self):
        report = {'generated_tokens': [1, 2], 'decode_warmup': {'seconds': 8.0},
                  'statistics': {'prefill_seconds': 3.0, 'prefill_seconds_already_in_warmup': 2.0,
                                 'decode_seconds': 4.0, 'decode_seconds_already_in_warmup': 3.0},
                  'steps': [{'kind': 'prefill', 'seconds': 1.0},
                            {'kind': 'decode', 'seconds': 1.0, 'decode_seconds_in_warmup': 3.0}]}
        timing = user_visible_metrics(report, 15.0)
        self.assertEqual(timing['decode_step_seconds'], [4.0])
        self.assertEqual(timing['other_wall_seconds'], 5.0)
        finalize_user_visible_metrics(timing, 16.0, 'through stdout')
        self.assertEqual(timing['other_wall_seconds'], 6.0)
        report['statistics']['decode_seconds_already_in_warmup'] = 7.0
        with self.assertRaises(ValueError):
            user_visible_metrics(report, 15.0)

    def test_compilation_subtotal_and_cache_hits_do_not_double_count_wall_time(self):
        metrics = {'_compile.compile_inner': [10.0]}
        counters = {'inductor': {'fxgraph_cache_hit': 2, 'fxgraph_cache_miss': 1}}
        before = compiler_snapshot(metrics, counters)
        metrics['_compile.compile_inner'].extend([0.25, 0.5])
        counters['inductor']['fxgraph_cache_hit'] += 1
        result = compiler_delta(before, metrics, counters)
        self.assertEqual(result['dynamo_compile_seconds'], 0.75)
        self.assertEqual(result['dynamo_compile_calls'], 2)
        self.assertEqual(result['fx_graph_cache_hits'], 1)
        self.assertEqual(result['fx_graph_cache_misses'], 0)
        self.assertIn('overlaps', result['scope'])
        metrics['_compile.compile_inner'].clear()
        self.assertFalse(compiler_delta(before, metrics, counters)['available'])

    def test_output_flush_extends_wall_and_recomputes_rates(self):
        timing = user_visible_metrics({'generated_tokens': [7, 2], 'statistics': {'prefill_seconds': 1}},
                                      2, special_ids=[2])
        finalize_user_visible_metrics(timing, 5, 'through stdout flush')
        self.assertEqual(timing['returned_token_ids_per_second'], 2 / 5)
        self.assertEqual(timing['non_special_token_ids_per_second'], 1 / 5)
        self.assertEqual(timing['other_wall_seconds'], 4)
        self.assertEqual(timing['scope'], 'through stdout flush')
        with self.assertRaises(ValueError):
            finalize_user_visible_metrics(timing, 4, 'invalid')


if __name__ == '__main__':
    unittest.main()
