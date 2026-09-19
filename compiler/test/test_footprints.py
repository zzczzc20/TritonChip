#!/usr/bin/env python3
"""
sec 13.4: implicit wide register reads must be in the compiler model.

Each test is a liveness mistake that the operand fields alone would not
prevent, and that the hardware does not detect for you.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))

from tnca_cc.diagnostics import Diagnostic  # noqa: E402
from tnca_cc.footprints import (  # noqa: E402
    OwnershipChecker, RegRange, dma_coordinate_footprint, dma_store_footprint,
    fragment_registers, mma_footprints, mma_source_footprint)


class TestFragmentExtent(unittest.TestCase):
    def test_16x16_fp16_fragment_is_eight_registers(self):
        """16*16*2 = 512 bytes over 64-byte registers."""
        self.assertEqual(fragment_registers(16, 16, 16), 8)

    def test_mma_reads_eight_registers_while_naming_one(self):
        f = mma_source_footprint(32)
        self.assertEqual((f.first, f.count), (32, 8))
        self.assertIn(39, f)
        self.assertNotIn(40, f)

    def test_k32_fragment_is_wider(self):
        self.assertEqual(fragment_registers(16, 32, 16), 16)

    def test_both_fragments_of_one_step(self):
        a, b = mma_footprints(32, 40)
        self.assertEqual(str(a), 'v32..v39')
        self.assertEqual(str(b), 'v40..v47')
        self.assertFalse(a.overlaps(b))


class TestCoordinateFootprint(unittest.TestCase):
    def test_rank_two_reads_two_consecutive_scalars(self):
        f = dma_coordinate_footprint(10, 2)
        self.assertEqual(str(f), 's10..s11')

    def test_base_s63_refused(self):
        """The hardware now raises ILLEGAL_INSTRUCTION; do not emit it."""
        with self.assertRaises(Diagnostic) as cm:
            dma_coordinate_footprint(63, 2)
        self.assertIn('wraps to s0', str(cm.exception))

    def test_highest_legal_base_for_rank_two(self):
        self.assertEqual(str(dma_coordinate_footprint(62, 2)), 's62..s63')

    def test_base_s0_refused(self):
        with self.assertRaises(Diagnostic):
            dma_coordinate_footprint(0, 2)

    def test_rank_one_may_use_s63(self):
        self.assertEqual(str(dma_coordinate_footprint(63, 1)), 's63')


class TestOwnership(unittest.TestCase):
    """The mistakes the operand fields would let through."""

    def setUp(self):
        self.ck = OwnershipChecker()

    def test_write_inside_a_live_fragment_refused(self):
        self.ck.hold(1, mma_footprints(32, 40), 'm.mma.async')
        with self.assertRaises(Diagnostic) as cm:
            self.ck.check_write('v', 33, by='v.movi')
        self.assertIn('v32..v39', str(cm.exception))

    def test_write_to_the_named_register_also_refused(self):
        self.ck.hold(1, mma_footprints(32, 40), 'm.mma.async')
        with self.assertRaises(Diagnostic):
            self.ck.check_write('v', 32)

    def test_write_outside_both_fragments_allowed(self):
        self.ck.hold(1, mma_footprints(32, 40), 'm.mma.async')
        self.assertTrue(self.ck.check_write('v', 48))
        self.assertTrue(self.ck.check_write('v', 31))

    def test_release_frees_the_range(self):
        self.ck.hold(1, mma_footprints(32, 40), 'm.mma.async')
        self.ck.release(1)
        self.assertTrue(self.ck.check_write('v', 33))

    def test_store_source_stays_live_after_issue(self):
        """sec 13.4's delayed read: the window is not free at issue."""
        self.ck.hold(7, [dma_store_footprint(16, 8)], 't.st.async')
        with self.assertRaises(Diagnostic):
            self.ck.check_write('v', 20)

    def test_coordinate_registers_are_held_too(self):
        self.ck.hold(3, [dma_coordinate_footprint(10, 2)], 't.ld.async')
        with self.assertRaises(Diagnostic) as cm:
            self.ck.check_write('s', 11, by='s.addi')
        self.assertIn('s10..s11', str(cm.exception))

    def test_two_commands_tracked_independently(self):
        self.ck.hold(1, [RegRange('v', 32, 8)], 'm.mma.async')
        self.ck.hold(2, [RegRange('v', 48, 8)], 't.ld.async')
        self.ck.release(1)
        self.assertTrue(self.ck.check_write('v', 33))
        with self.assertRaises(Diagnostic):
            self.ck.check_write('v', 50)

    def test_overlapping_write_range_refused(self):
        """A vector store writing a range that clips a live fragment."""
        self.ck.hold(1, [RegRange('v', 32, 8)], 'm.mma.async')
        with self.assertRaises(Diagnostic):
            self.ck.check_write_range(RegRange('v', 38, 4), by='t.ld.async')


if __name__ == '__main__':
    unittest.main(verbosity=2)
