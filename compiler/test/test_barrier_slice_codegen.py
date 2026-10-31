"""Post-allocation barrier IDs must fit the launch's resource slice."""
import unittest
import os

from tnca.adapters import r1
from tnca.errors import CompileError
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.target import TargetInfo


class BarrierSliceCodegenTests(unittest.TestCase):
    def build(self, count):
        kernel = make_kernel('barrier_slice')
        builder = Builder(kernel, kernel.role(0).region)
        for _ in range(count):
            builder.barrier()
        builder.exit()
        return kernel

    def test_allocated_barriers_respect_spmd_slice(self):
        target = TargetInfo.default()
        limit = target.num_barriers // 2
        prov = r1.Provenance()
        prov.record('rctx_spmd', 2)
        prov.record('rctx_slice', True)
        r1.emit(self.build(limit), target, prov=prov)
        with self.assertRaisesRegex(CompileError, 'outside slice'):
            r1.emit(self.build(limit + 1), target, prov=prov)

    def test_allocated_barriers_respect_paired_slice(self):
        target = TargetInfo.default()
        prov = r1.Provenance()
        prov.record('rctx_roles', 2)
        prov.record('rctx_roles_pairs', 2)
        with self.assertRaisesRegex(CompileError, 'outside slice'):
            r1.emit(self.build(target.num_barriers // 2 + 1), target, prov=prov)

    def test_three_barriers_choose_two_program_slices_on_eight_barrier_target(self):
        from tnca.aot import _rctx_spmd
        from tnca.dump import Dump
        from tnca_cc.ir import Builder
        target = TargetInfo.load('tnca-sim-vrf-v5-x8-g2c4')
        previous = os.environ.get('TNCA_RCTX_SPMD_SHARED')
        os.environ['TNCA_RCTX_SPMD_SHARED'] = '4'
        try:
            for count, expected in ((2, 4), (3, 2)):
                kernel = make_kernel('slice_fallback')
                builder = Builder(kernel, kernel.role(0).region)
                builder.program_id(0)
                for _ in range(count):
                    builder.barrier()
                builder.exit()
                prov = r1.Provenance()
                contexts = _rctx_spmd(kernel, target, prov, {'rctx_spmd': 1},
                                      Dump('slice_fallback'), 0, 0)
                self.assertEqual(contexts, expected)
                _, cg = r1.emit(kernel, target, prov=prov)
                self.assertEqual([b.reg for b in cg.kernel.barriers], list(range(count)))
        finally:
            if previous is None:
                os.environ.pop('TNCA_RCTX_SPMD_SHARED', None)
            else:
                os.environ['TNCA_RCTX_SPMD_SHARED'] = previous


if __name__ == '__main__':
    unittest.main()
