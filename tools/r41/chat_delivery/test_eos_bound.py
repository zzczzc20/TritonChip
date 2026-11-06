#!/usr/bin/env python3
"""Check EOS bounds use discarded decode time rather than returned-token count."""
import unittest

import eos_bound


class EosBoundTests(unittest.TestCase):
    def test_only_post_eos_decode_is_subtracted(self):
        report = dict(device='tnca', model='test', prompts=[dict(id='fact', turns=[dict(tokens=[1, 2],
            user_visible_timing=dict(prompt_to_return_seconds=100),
            steps=[dict(kind='prefill', seconds=20), dict(kind='decode', seconds=10),
                   dict(kind='decode', seconds=11), dict(kind='decode', seconds=12)])])])
        row = eos_bound.bounds(report)['rows'][0]
        self.assertEqual(row['discarded_decode_steps'], 2)
        self.assertEqual(row['discarded_decode_seconds'], 23)
        self.assertEqual(row['ideal_wall_seconds'], 77)

    def test_bad_or_missing_timing_is_rejected(self):
        report = dict(device='tnca', prompts=[dict(id='fact', turns=[dict(tokens=[1],
            user_visible_timing=dict(prompt_to_return_seconds=1),
            steps=[dict(kind='prefill', seconds=0), dict(kind='decode', seconds=2)])])])
        with self.assertRaises(ValueError):
            eos_bound.bounds(report)

    def test_non_device_residual_uses_measured_clock(self):
        report = dict(device='tnca', prompts=[dict(id='fact', turns=[dict(tokens=[1, 2],
            user_visible_timing=dict(prompt_to_return_seconds=100), steps=[dict(kind='prefill', seconds=10),
                dict(kind='decode', seconds=20, transfers_inside_step=dict(device_cycles=100)),
                dict(kind='decode', seconds=20, transfers_inside_step=dict(device_cycles=100))])])])
        row = eos_bound.bounds(report, clock_hz=10)['rows'][0]
        self.assertEqual(row['non_device_residual_seconds'], 20)
        self.assertAlmostEqual(row['non_device_residual_fraction'], 0.5)
        self.assertAlmostEqual(row['residual_reduction_to_30_percent'], 20 - 60 / 7)
        report['device'] = 'cpu'
        with self.assertRaises(ValueError):
            eos_bound.bounds(report)


if __name__ == '__main__':
    unittest.main()
