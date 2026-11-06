"""Wide loop accumulators stay in independent physical MMA fragments.

Uses real Triton SSA and the common importer/codegen. In particular this
checks loop-body traffic, so moving the same row copies outside a single
dot operation does not conceal an accumulator round trip on every K step.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test')]
import triton_bridge
from tnca_cc.frontend import TTIRImporter
from tnca_cc.target import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.diagnostics import ResourceError

if triton_bridge.available():
    import triton
    import triton.language as tl

    @triton.jit
    def fragment_loop(A, B, C, K: tl.constexpr, N: tl.constexpr,
                      OLD_USE: tl.constexpr):
        m = tl.arange(0, 16)
        n = tl.arange(0, N)
        k = tl.arange(0, 16)
        acc = tl.full((16, N), 0, tl.float32)
        for step in range(K // 16):
            a = tl.load(A + m[:, None] * K + (step * 16 + k[None, :]))
            b = tl.load(B + (step * 16 + k[:, None]) * N + n[None, :])
            product = tl.dot(a, b)
            if OLD_USE:
                # A second reader of the old carry forbids destructive
                # accumulation. The observable store precedes the update.
                tl.store(C + m[:, None] * N + n[None, :], acc)
            acc = acc + product
        tl.store(C + m[:, None] * N + n[None, :], acc)


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class FragmentResidencyTests(unittest.TestCase):
    def compile(self, columns, *, old_use=False, target=None):
        sig = {k: '*fp16' for k in ('A', 'B')}
        sig.update(C='*fp32', K='constexpr', N='constexpr', OLD_USE='constexpr')
        module, context = triton_bridge.make_ttir(
            fragment_loop, sig, dict(K=64, N=columns, OLD_USE=old_use))
        target = target or TargetInfo.load('tnca-sim-vrf-v5')
        imp = TTIRImporter(target, 'fragment_loop', index_opt=True,
                           scratch_addr=0x80000)
        kernel, _ = imp.run(module, 'fragment_loop',
                            dict(A=0x40000, B=0x44000, C=0x48000),
                            param_names=['A', 'B', 'C'])
        cg = CodeGen(kernel, target, check_protocol=False, window_reuse=True)
        cg.assign_shared()
        cg.run()
        return kernel.role(0).region.ops

    def loop_ops(self, ops):
        labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
        loops = [(labels[op.attrs['target']], i) for i, op in enumerate(ops)
                 if op.kind in ('branch', 'branch_nz') and
                 op.attrs.get('target') in labels and labels[op.attrs['target']] < i]
        self.assertEqual(len(loops), 1, 'exercise the actual K loop')
        start, end = loops[0]
        return ops[start:end + 1]

    def test_two_fragments_remain_resident_across_k(self):
        for columns in (16, 32):
            with self.subTest(columns=columns):
                ops = self.compile(columns)
                body = self.loop_ops(ops)
                mma = [op for op in body if op.kind == 'mma_async']
                self.assertEqual(len(mma), columns // 16)
                self.assertEqual(len({op.operands[0] for op in mma}), columns // 16)
                self.assertFalse([op for op in body if op.kind in
                                  ('acc_read', 'acc_write', 'acc_zero')],
                                 'the K loop must not round-trip or zero the carry')
                self.assertTrue(any(op.kind == 'acc_read' for op in ops),
                                'epilogue still reads the final result')

    def test_live_old_accumulator_is_not_overwritten(self):
        body = self.loop_ops(self.compile(32, old_use=True))
        self.assertTrue(any(op.kind == 'acc_read' for op in body))
        self.assertTrue(any(op.kind == 'acc_write' for op in body))

    def test_insufficient_accumulators_are_rejected_before_execution(self):
        target = TargetInfo.load('tnca-sim-vrf-v5')
        target._d['accumulators']['count'] = 1
        with self.assertRaises(ResourceError):
            self.compile(32, target=target)


if __name__ == '__main__':
    unittest.main()
