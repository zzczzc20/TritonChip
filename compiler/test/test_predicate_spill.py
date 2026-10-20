"""Predicate pressure may use scalar masks without changing any lane bits."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test')]
from tnca_cc.codegen import CodeGen
from tnca_cc.diagnostics import ResourceError
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.predicate_spill import spill
from tnca_cc.target import TargetInfo
import rtl_harness as H


def case(count=32, mutable=False):
    kernel = make_kernel('predicate_pressure')
    b = Builder(kernel, kernel.role(0).region)
    lanes = b.arange()
    masks = [b.vcmp('lt', lanes, b.splat(b.const(i))) for i in range(1, count + 1)]
    if mutable:
        b._emit('assign', [], [masks[0], masks[1]])
    total = b.splat(b.const(0))
    for i, mask in enumerate(masks, 1):
        selected = b.vselect(mask, b.splat(b.const(i)), b.splat(b.const(0)))
        total = b.vbin('add', total, selected)
    b.store(b.const(0x4000), total)
    return kernel, masks


class PredicateSpillTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.load('tnca-sim-vrf-v5')

    def test_no_change_below_capacity(self):
        kernel, _ = case(count=4)
        original = list(kernel.role(0).region.ops)
        self.assertEqual(spill(kernel, self.target)['predicates'], 0)
        self.assertEqual(kernel.role(0).region.ops, original)

    def test_preserve_mutable_predicate_destinations(self):
        kernel, masks = case(mutable=True)
        plan = spill(kernel, self.target)
        self.assertGreater(plan['predicates'], 0)
        stored = {op.operands[0] for op in kernel.role(0).region.ops if op.kind == 'pred_to_scalar'}
        self.assertNotIn(masks[0], stored)

    def test_pressure_is_relieved_and_every_lane_is_correct_on_rtl(self):
        kernel, _ = case()
        old = CodeGen(copy.deepcopy(kernel), self.target, check_protocol=False)
        old.assign_shared()
        with self.assertRaises(ResourceError):
            old.run()
        plan = spill(kernel, self.target)
        self.assertGreater(plan['predicates'], 0)
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        self.assertIn('p.ballot', cg.b.text())
        self.assertIn('p.frombits', cg.b.text())
        if not H.available():
            self.skipTest(H.why_unavailable())
        program = H.Program('predicate_scalar_masks').kernel(cg).check_base(0x4000)
        for lane in range(16):
            program.check_word(0x4000, lane * 4, sum(i for i in range(1, 33) if lane < i))
        ok, stdout = H.run(program)
        self.assertTrue(ok, stdout)


if __name__ == '__main__':
    unittest.main()
