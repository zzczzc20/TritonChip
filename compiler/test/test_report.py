#!/usr/bin/env python3
"""
sec 16.1: the per-kernel resource and bottleneck report.

The plan's reason for wanting it is specific -- 这比只输出"寄存器用了多少"更能
指导硬件修改 -- so the tests are about whether the report carries the facts a
hardware team could act on, not merely whether it formats.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))

from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.diagnostics import ResourceError  # noqa: E402
from tnca_cc.report import (  # noqa: E402
    KernelPlan, check_plan, format_report, try_plans)


def gemm_plan(contexts, acc):
    return KernelPlan(
        name='gemm_f16', active_contexts=contexts, rf_stages=2,
        shared_windows=4, private_vector_bytes=8192,
        accumulator_slots=acc, barriers=4, max_async_commands=6,
        layout_conversion_bytes=0,
        dma_commands_source=2, dma_commands_expanded=8,
        issue_cycles_by_unit={'matrix-unit': 4096, 'transfer-engine': 1024,
                              'vector-alu': 512},
        source_release_policy='command_completion')


class TestPlanFitting(unittest.TestCase):
    def setUp(self):
        self.t = TargetInfo.default()

    def test_fitting_plan_accepted(self):
        self.assertTrue(check_plan(gemm_plan(2, 8), self.t))

    def test_vector_registers_cannot_be_spilled(self):
        """There is no scratchpad to spill into: RF is the only level above
        NUCA, so an oversized tile is an error rather than a slowdown."""
        plan = gemm_plan(2, 8)
        plan.private_vector_bytes = 10 ** 7
        with self.assertRaises(ResourceError) as cm:
            check_plan(plan, self.t)
        self.assertIn('no scratchpad', cm.exception.because)


class TestFallbackOrder(unittest.TestCase):
    """sec 10.3 orders the fallbacks; sec 16.1 wants the rejections kept."""

    def setUp(self):
        self.t = TargetInfo.default()

    def test_first_fitting_plan_is_taken(self):
        chosen = try_plans([gemm_plan(4, 32), gemm_plan(2, 8)], self.t)
        self.assertEqual(chosen.active_contexts, 2)

    def test_rejected_plan_and_reason_are_recorded(self):
        """The exact output sec 16.1 shows:

            rejected 4-context plan:
                insufficient accumulator capacity

        The four-context plan is over on both contexts and accumulators, and
        the accumulator file is the one reported -- which is the more useful
        of the two, since it is the thing a hardware team would grow.
        """
        chosen = try_plans([gemm_plan(4, 32), gemm_plan(2, 8)], self.t)
        self.assertEqual(len(chosen.rejected), 1)
        r = chosen.rejected[0]
        self.assertIn('4-context plan', r.description)
        self.assertIn('accumulators', r.reason)

    def test_context_shortage_reported_when_it_is_the_only_problem(self):
        chosen = try_plans([gemm_plan(4, 8), gemm_plan(2, 8)], self.t)
        self.assertIn('Role Contexts', chosen.rejected[0].reason)

    def test_accumulator_shortage_is_named_as_such(self):
        """The sentence a hardware team can act on."""
        chosen = try_plans([gemm_plan(2, 32), gemm_plan(2, 8)], self.t)
        self.assertIn('accumulators', chosen.rejected[0].reason)

    def test_no_fitting_plan_raises(self):
        with self.assertRaises(ResourceError):
            try_plans([gemm_plan(4, 32)], self.t)


class TestReportContent(unittest.TestCase):
    def setUp(self):
        self.t = TargetInfo.default()
        self.plan = try_plans([gemm_plan(4, 32), gemm_plan(2, 8)], self.t)
        self.text = format_report(self.plan, self.t)

    def test_every_field_the_plan_names_is_present(self):
        for field in ('active contexts', 'RF stages', 'shared RF windows',
                      'private vector bytes', 'accumulator slots', 'barriers',
                      'max async commands', 'layout conversion bytes',
                      'DMA command expansion', 'predicted issue pressure',
                      'source-release policy', 'rejected'):
            with self.subTest(field=field):
                self.assertIn(field, self.text)

    def test_dma_expansion_shows_both_counts(self):
        """sec 8.2: the operation count is taken after expansion, so a report
        showing only the source count would not explain the barrier arming."""
        self.assertIn('2 -> 8', self.text)

    def test_issue_pressure_names_the_busiest_unit(self):
        """sec 10.4 accumulates per shared unit; the bottleneck is the max
        over units, not their sum."""
        unit, cycles = self.plan.predicted_issue_pressure
        self.assertEqual(unit, 'matrix-unit')
        self.assertEqual(cycles, 4096)
        self.assertIn('matrix-unit 4096', self.text)

    def test_conservative_source_release_is_flagged(self):
        """The target has one completion covering both ends, and a reader
        should not have to know that to read the number."""
        self.assertIn('conservative bound', self.text)

    def test_capacities_are_shown_beside_usage(self):
        """'8 of 16' is actionable; '8' is not."""
        self.assertIn('(of 16)', self.text)
        self.assertIn('(of 32)', self.text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
