#!/usr/bin/env python3
"""The tiled element-wise lowering (option tiled_pointwise, pointwise_tiles.py).

A @triton.jit kernel through Triton's frontend, the importer with the option
set, the code generator and the core RTL bench, checked word by word against
the same arithmetic in Python. Covers both modes (1: tensor-engine tiles where
legal; 2: vector loads and stores only), several blocks, a masked tail and an
exact multiple, and the refusals that keep the default lowering.

Needs Triton importable and the RTL bench (skips otherwise, like
test_triton_e2e).
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

import rtl_harness as H  # noqa: E402
import triton_bridge  # noqa: E402
from fp32 import f2b  # noqa: E402
from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.codegen import CodeGen  # noqa: E402
from tnca_cc.frontend import TTIRImporter, Unsupported  # noqa: E402

SENTINEL = 0xdeadbeef
# Unaligned on purpose (4 bytes past an 8-byte boundary): the tensor engine's
# map base is rounded down and the coordinate moved up (ISA sec 9E.2).
X, Y, O = 0x10004, 0x11004, 0x12004
SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32', 'BLOCK': 'constexpr'}


def xv(i):
    return i * 0.5


def yv(i):
    return i * 0.25 + 1.0


@unittest.skipUnless(triton_bridge.available(), 'triton not importable')
class TestPointwiseTiles(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()

    def _import(self, fn, name, block, n, mode):
        mod, _ctx = triton_bridge.make_ttir(fn, SIG, {'BLOCK': block})
        imp = TTIRImporter(self.target, name, scratch_addr=0x3000, index_opt=True)
        imp.tiled_pointwise = mode
        kernel, _ = imp.run(mod, name, {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': n},
                            param_names=triton_bridge.param_names(fn))
        return imp, kernel

    def _cg(self, kernel):
        from tnca_cc.optimization import drop_dead_pure
        drop_dead_pure(kernel, flat_control=True)
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return cg

    def _run(self, fn, name, expect, block, n, mode):
        imp, kernel = self._import(fn, name, block, n, mode)
        self.assertIsNotNone(imp.tiled_pointwise_plan)
        cg = self._cg(kernel)
        text = cg.b.text()
        if mode == 1:
            self.assertIn('t.ld.async', text)
            self.assertIn('t.st.async', text)
        else:
            self.assertNotIn('t.ld.async', text)
        p = H.Program(f"tiles_{name}_{block}_{n}_{mode}")
        p.kernel(cg)
        p.check_base(O)
        for i in range(block):
            p.check_word(O, i * 4, f2b(expect(i)) if i < n else SENTINEL)
        p.words(X, H.pack32([f2b(xv(i)) for i in range(block)]), 'xs')
        p.words(Y, H.pack32([f2b(yv(i)) for i in range(block)]), 'ys')
        p.words(O, H.pack32([SENTINEL] * block), 'os')
        ok, out = H.run(p)
        self.assertTrue(ok, f"tiled {name} BLOCK={block} n={n} mode={mode} failed on the RTL:\n{out}")
        return imp

    def test_add_both_modes_tail_and_exact(self):
        if not H.available():
            self.skipTest(H.why_unavailable())
        import kernels_triton as K
        for mode in (1, 2):
            for block in (64, 256):
                for n in (40, block):
                    with self.subTest(mode=mode, block=block, n=n):
                        self._run(K.add_kernel, 'add_kernel', lambda i: xv(i) + yv(i), block, n, mode)

    def test_mul_add_mode1(self):
        if not H.available():
            self.skipTest(H.why_unavailable())
        import kernels_triton as K
        self._run(K.mul_add_kernel, 'mul_add_kernel', lambda i: xv(i) * yv(i) + xv(i), 128, 100, 1)

    def test_grouped_loads_unmasked(self):
        """Mode 2 issues the loads of two tiles before computing the first
        (lane groups); the result must be unchanged."""
        if not H.available():
            self.skipTest(H.why_unavailable())
        import kernels_triton as K
        imp = self._run(K.axpb_nomask_kernel, 'axpb_nomask_kernel', lambda i: xv(i) * 2.0 + yv(i), 256, 256, 2)
        self.assertGreater(imp.tiled_pointwise_plan['groups'], 1)

    def test_plan_shape(self):
        import kernels_triton as K
        imp, _ = self._import(K.add_kernel, 'add_kernel', 256, 40, 1)
        plan = imp.tiled_pointwise_plan
        self.assertEqual(plan['width'], 256)
        self.assertEqual(plan['accesses'], ['load', 'load', 'store'])
        self.assertTrue(plan['double_buffer'])
        imp, _ = self._import(K.add_kernel, 'add_kernel', 256, 40, 2)
        self.assertEqual(imp.tiled_pointwise_plan['accesses'], ['vload', 'vload', 'vstore'])

    def test_option_off_keeps_the_default_lowering(self):
        import kernels_triton as K
        imp, _ = self._import(K.add_kernel, 'add_kernel', 64, 40, 0)
        self.assertIsNone(imp.tiled_pointwise_plan)

    def test_narrow_block_is_not_tiled(self):
        import kernels_triton as K
        imp, _ = self._import(K.add_kernel, 'add_kernel', 16, 40, 1)
        self.assertIsNone(imp.tiled_pointwise_plan)

    def test_untileable_wide_program_is_refused(self):
        """A program wider than 128 lanes that cannot be tiled is refused, so
        Inductor keeps its 16-lane configuration (inductor.py)."""
        import kernels_triton as K
        if not hasattr(K, 'sum_kernel'):
            self.skipTest('no reduction kernel in kernels_triton')
        mod, _ctx = triton_bridge.make_ttir(K.sum_kernel, {'x_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32',
                                                           'BLOCK': 'constexpr'}, {'BLOCK': 256})
        imp = TTIRImporter(self.target, 'sum_kernel', scratch_addr=0x3000, index_opt=True)
        imp.tiled_pointwise = 1
        with self.assertRaises(Unsupported):
            imp.run(mod, 'sum_kernel', {'x_ptr': X, 'o_ptr': O, 'n': 40},
                    param_names=triton_bridge.param_names(K.sum_kernel))


if __name__ == '__main__':
    unittest.main()
