#!/usr/bin/env python3
import unittest
from engine_overlap import parse, selected


class OverlapTests(unittest.TestCase):
    def test_intersection_uses_engine_service(self):
        trace = ('ENG 1 mtx-issue rctx1 op0\nENG 3 mtx-done\n'
                 'ENG 10 txf-issue rctx0 xfr\nENG 20 mtx-issue rctx1 op1\n'
                 'ENG 25 txf-done\nENG 30 mtx-done\n'
                 '[dev] seq=12 op=3 a0=0 a1=0 -> status=0 r0=0\n')
        self.assertEqual(parse(trace)[12]['overlap_cycles'], 5)
        self.assertEqual(parse(trace)[12]['mma_cycles'], 10)
        self.assertEqual(parse(trace)[12]['dma_cycles'], 15)

    def test_configuration_and_row_ops_cannot_claim_overlap(self):
        trace = ('ENG 2 txf-issue rctx0 cfg\nENG 3 mtx-issue rctx1 op2\n'
                 'ENG 4 txf-done\nENG 8 mtx-done\n'
                 '[dev] seq=3 op=9 a0=0 a1=0 -> status=0 r0=0\n')
        self.assertEqual(parse(trace)[3]['overlap_cycles'], 0)
        self.assertEqual(parse(trace)[3]['mma_cycles'], 0)

    def test_warmup_and_serial_launch_are_distinct(self):
        trace = ('ENG 2 txf-issue rctx0 xfr\nENG 3 mtx-issue rctx1 op1\n'
                 'ENG 5 txf-done\nENG 6 mtx-done\n'
                 '[dev] seq=1 op=3 -> status=0\n'
                 'ENG 1 txf-issue rctx0 xfr\nENG 5 txf-done\n'
                 'ENG 6 mtx-issue rctx1 op1\nENG 9 mtx-done\n'
                 '[dev] seq=2 op=3 -> status=0\n')
        self.assertEqual(selected(trace, [2])[2]['overlap_cycles'], 0)
        with self.assertRaises(ValueError):
            selected(trace, [3])

    def test_incomplete_faulted_or_overlapping_commands_are_refused(self):
        for trace in ('', 'ENG 1 mtx-issue rctx1 op1\n',
                      '[dev] seq=1 op=3 -> status=1\n',
                      'ENG 1 mtx-issue rctx1 op1\nENG 2 mtx-issue rctx1 op1\n',
                      'ENG 5 mtx-issue rctx1 op1\nENG 4 mtx-done\n'):
            with self.assertRaises(ValueError):
                parse(trace)


if __name__ == '__main__':
    unittest.main()
