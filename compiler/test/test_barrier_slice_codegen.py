"""Post-allocation barrier IDs must fit the launch's resource slice."""
import unittest

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


if __name__ == '__main__':
    unittest.main()
