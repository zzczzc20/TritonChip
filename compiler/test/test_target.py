#!/usr/bin/env python3
"""
IR-independent unit tests for the target contract (compiler_overview.md 16.2,
"IR 单元测试": unsupported capability diagnostics among them).

These run with nothing installed, which is the point -- they guard the layer
every later lowering reads, before MLIR is in the picture.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))

from tnca_cc import ISA, CapabilityError, ResourceError, TargetInfo  # noqa: E402


class TestTargetContract(unittest.TestCase):
    """Every fact sec 15 C0 requires must be answerable."""

    def setUp(self):
        self.t = TargetInfo.default()

    def test_c0_required_fields_present(self):
        required = {
            'RCTX count': self.t.num_rctx,
            'VLEN': self.t.vlen,
            'logical V limit': self.t.vregs_per_rctx,
            'physical V limit': self.t.phys_vregs,
            'accumulator count': self.t.num_accumulators,
            'DMA rank': self.t.dma_max_rank,
            'DMA dtypes': self.t.dma['dtypes'],
            'MMA shape': self.t.mma_shape,
            'barrier count': self.t.num_barriers,
            'barrier count width': self.t.barrier_index_width,
            'source release': self.t.mma_source_release,
        }
        for name, value in required.items():
            with self.subTest(field=name):
                self.assertTrue(value, f"C0 requires {name}")
        for cap in ('sfu', 'atomics_device', 'atomics_system', 'fence'):
            with self.subTest(capability=cap):
                self.assertIsInstance(self.t.has(cap), bool)

    def test_physical_registers_exceed_one_rctx_allocation(self):
        """The slack is what makes a shared window possible at all.

        If physical == logical * rctx there is nothing left to alias, and the
        zero-copy role communication of sec 13.1 cannot exist. t13_share
        demonstrates the hardware side; this pins the target description.
        """
        self.assertGreater(self.t.phys_vregs, self.t.vregs_per_rctx)

    def test_no_addressable_scratchpad(self):
        """The architecture's defining constraint, not an incidental one."""
        self.assertFalse(self.t.has_addressable_scratchpad)

    def test_scalar_alu_has_no_floating_point(self):
        """sec 12: do not assume the Scalar ALU can do FP; it cannot."""
        self.assertFalse(self.t.has_scalar_fp)

    def test_gather_reach_is_bounded_and_known(self):
        lo, hi = self.t.gather_reach_elements()
        self.assertEqual((lo, hi), (-(1 << 31), (1 << 31) - 1))

    def test_gather_indices_are_element_scaled(self):
        """ISA sec 9.1 scales by access size; byte offsets are wrong."""
        self.assertEqual(self.t.gather_index_scaling, 'element')


class TestCapabilityGate(unittest.TestCase):
    """sec 12: a missing feature is a diagnostic, never a substitution."""

    def setUp(self):
        self.t = TargetInfo.default()

    def test_require_present_capability_returns(self):
        self.assertTrue(self.t.require('vector_fp32'))

    def test_require_absent_capability_raises(self):
        with self.assertRaises(CapabilityError):
            self.t.require('sfu')

    def test_atomics_refusal_offers_no_fallback(self):
        """A load-store pair is not an atomic, so no remedy may be suggested."""
        with self.assertRaises(CapabilityError) as cm:
            self.t.require('atomics_device')
        self.assertIn('no correct substitute', cm.exception.remedy)

    def test_sfu_refusal_demands_error_bound_reverification(self):
        """sec 12 warns against assuming an upstream polynomial meets the
        ISA sec 8.6 accuracy contract."""
        with self.assertRaises(CapabilityError) as cm:
            self.t.require('sfu')
        self.assertIn('error bound', cm.exception.remedy)

    def test_nan_padding_refused_not_approximated(self):
        """sec 12: TNCA's zero/clamp/fault are not equivalent to NaN padding."""
        from tnca_cc import CapabilityError as _CE
        for mode in ('zero', 'clamp', 'fault'):
            with self.subTest(mode=mode):
                self.assertEqual(self.t.dma_padding_mode(mode), mode)
        with self.assertRaises(_CE) as cm:
            self.t.dma_padding_mode('nan')
        self.assertIn('masked LSU loop', cm.exception.remedy)

    def test_unknown_capability_is_an_error_not_a_false(self):
        """Returning False for a typo would silently disable a real feature."""
        with self.assertRaises(KeyError):
            self.t.has('tensor_dmaa')


class TestResourceBudget(unittest.TestCase):
    """sec 10.1: these are whole-program resources."""

    def setUp(self):
        self.t = TargetInfo.default()

    def test_fitting_plan_accepted(self):
        self.assertTrue(self.t.check_budget(barriers=4, accumulators=2,
                                            windows=4, rctx=2))

    def test_overflow_names_the_resource(self):
        with self.assertRaises(ResourceError) as cm:
            self.t.check_budget(accumulators=self.t.num_accumulators + 1)
        self.assertIn('accumulators', cm.exception.message)

    def test_more_roles_than_contexts_refused(self):
        """sec 6.3 requires supporting a 2-RCTX profile; asking for four on a
        two-context target must fail loudly, not silently serialise."""
        with self.assertRaises(ResourceError):
            self.t.check_budget(rctx=self.t.num_rctx + 1)


class TestISASource(unittest.TestCase):
    """sec 11.3: one source of opcode numbers."""

    def test_opcodes_are_unique(self):
        seen = {}
        for ins in ISA:
            self.assertNotIn(ins.opcode, seen,
                             f"{ins.mnemonic} collides with {seen.get(ins.opcode)}")
            seen[ins.opcode] = ins.mnemonic

    def test_schema_matches_assembler_tables(self):
        """The generated schema must still describe the authoring tables."""
        sys.path.insert(0, os.path.join(ROOT, 'tools', 'asm'))
        import tnca_isa as authoring
        self.assertEqual(len(ISA), len(authoring.ISA))
        for mnem, (maj, func, _ops, _sfx) in authoring.ISA.items():
            with self.subTest(mnemonic=mnem):
                self.assertEqual(ISA[mnem].opcode,
                                 (authoring.MAJ[maj], func))

    def test_capability_gated_encodings_are_identified(self):
        """An encoding the target cannot run must be refused before emission."""
        t = TargetInfo.default()
        for mnem in ('v.exp2.approx', 'v.rcp.approx'):
            with self.subTest(mnemonic=mnem):
                cap = ISA.required_capability(mnem)
                self.assertEqual(cap, 'sfu')
                with self.assertRaises(CapabilityError):
                    t.require(cap, f"emit {mnem}")

    def test_plain_arithmetic_needs_no_capability(self):
        self.assertIsNone(ISA.required_capability('s.add'))

    def test_every_mnemonic_resolves_by_opcode(self):
        for ins in ISA:
            self.assertIs(ISA.by_opcode(*ins.opcode), ins)


if __name__ == '__main__':
    unittest.main(verbosity=2)
