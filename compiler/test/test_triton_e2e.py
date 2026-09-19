#!/usr/bin/env python3
"""
End to end: a @triton.jit kernel, through Triton's own frontend, to the RTL.

This is what compiler_overview.md sec 15 C1 means by 第一个真实 Triton->RTL
闭环, and the bar it sets is 不修改输出汇编，不人工补 barrier，直接运行编译
产物. Triton builds the TTIR; tnca_cc.frontend walks it; tnca_cc.codegen
emits assembly; the existing tools/asm/tnca_asm.py assembles it; the existing
bench runs it. Nothing in between edits an instruction.

Needs Triton importable, which on this project means the virtualenv the
pinned build was installed into. Skips otherwise, so the rest of the suite
still runs.
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
from tnca_cc.frontend import TTIRImporter  # noqa: E402

SENTINEL = 0xdeadbeef
N = 40
X, Y, O = 0x4000, 0x4400, 0x4800

SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'BLOCK': 'constexpr'}


def xv(i):
    return i * 0.5


def yv(i):
    return i * 0.25 + 1.0


@unittest.skipUnless(triton_bridge.available(), 'triton not importable')
class TestTritonToRTL(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()
        if not H.available():
            self.skipTest(H.why_unavailable())

    def _compile(self, fn, name):
        mod, _ctx = triton_bridge.make_ttir(fn, SIG, {'BLOCK': self.target.vlen})
        imp = TTIRImporter(self.target, name)
        kernel, _ = imp.run(mod, name,
                            {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': N})
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return cg

    def _program(self, fn, name, expect):
        t = self.target
        padded = ((N + t.vlen - 1) // t.vlen) * t.vlen
        cg = self._compile(fn, name)

        p = H.Program(f"triton_{name}")
        p.kernel(cg)
        # One program id per run, because the bench launches a single context
        # rather than a grid: the kernel covers offs 0..VLEN-1.
        p.check_base(O)
        for i in range(t.vlen):
            p.check_word(O, i * 4, f2b(expect(i)) if i < N else SENTINEL)
        p.words(X, H.pack32([f2b(xv(i)) for i in range(padded)]), 'xs')
        p.words(Y, H.pack32([f2b(yv(i)) for i in range(padded)]), 'ys')
        p.words(O, H.pack32([SENTINEL] * padded), 'os')
        return p

    def test_add_kernel(self):
        import kernels_triton as K
        p = self._program(K.add_kernel, 'add_kernel', lambda i: xv(i) + yv(i))
        ok, out = H.run(p)
        self.assertTrue(ok, f"Triton kernel failed on the RTL:\n{out}")

    def test_mul_add_kernel(self):
        """a*b + a: two FP operations, to check the value graph is walked
        rather than pattern-matched on a single add."""
        import kernels_triton as K
        p = self._program(K.mul_add_kernel, 'mul_add_kernel',
                          lambda i: xv(i) * yv(i) + xv(i))
        ok, out = H.run(p)
        self.assertTrue(ok, f"Triton kernel failed on the RTL:\n{out}")

    def test_unsupported_block_is_refused(self):
        """sec 12: a block wider than one register is a diagnostic, not a
        silent lowering of the first register only."""
        import kernels_triton as K
        from tnca_cc.frontend import Unsupported
        mod, _ctx = triton_bridge.make_ttir(K.add_kernel, SIG, {'BLOCK': 64})
        imp = TTIRImporter(self.target, 'add_kernel')
        with self.assertRaises(Unsupported) as cm:
            imp.run(mod, 'add_kernel',
                    {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': N})
        self.assertIn('vector registers', str(cm.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
