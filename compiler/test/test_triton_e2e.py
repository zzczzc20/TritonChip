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

    def _compile(self, fn, name, block=None):
        block = self.target.vlen if block is None else block
        mod, _ctx = triton_bridge.make_ttir(fn, SIG, {'BLOCK': block})
        imp = TTIRImporter(self.target, name)
        kernel, _ = imp.run(mod, name,
                            {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': N},
                            param_names=triton_bridge.param_names(fn))
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return cg

    def _program(self, fn, name, expect, block=None):
        block = self.target.vlen if block is None else block
        padded = ((N + block - 1) // block) * block
        cg = self._compile(fn, name, block=block)

        p = H.Program(f"triton_{name}")
        p.kernel(cg)
        # One program id per run, because the bench launches a single context
        # rather than a grid: the kernel covers offs 0..BLOCK-1.
        p.check_base(O)
        for i in range(block):
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

    def test_context_specialised_add_kernel(self):
        """The closure this whole exercise is for: a @triton.jit kernel
        compiled to context-specialised ISA and run on the machine.

        Triton builds the TTIR; the partitioner classifies it and maps the
        partitions onto Role Contexts; split_roles turns the single region
        into one per role, recomputing the pure values each needs (sec 7.3)
        and handing the one tensor that cannot be recomputed across a shared
        RF window; the existing assembler assembles it. Nothing is edited in
        between, and no barrier is added by hand.
        """
        import kernels_triton as K
        from tnca_cc.partition import split_roles

        t = self.target
        mod, _ctx = triton_bridge.make_ttir(K.add_kernel, SIG,
                                            {'BLOCK': t.vlen})
        kernel, _ = TTIRImporter(t, 'add_kernel').run(
            mod, 'add_kernel', {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': N},
            param_names=triton_bridge.param_names(K.add_kernel))
        res = split_roles(kernel, t, name='add_cs')

        self.assertEqual(len(res.kernel.roles), 2)
        self.assertEqual(len(res.crossings), 1,
                         'the only value that cannot be recomputed is the sum')
        self.assertGreater(res.rematerialised, 0,
                           'the consumer must regenerate its own addresses')

        # The split has to be real. The epilogue role must not be doing the
        # loads, and the producer must not be doing the store -- otherwise
        # "two roles" is two copies of the same program.
        from tnca_cc.codegen import CodeGen
        texts = []
        for r in range(2):
            cg = CodeGen(res.kernel, t, check_protocol=False)
            cg.assign_shared()
            cg.run(r)
            texts.append(cg.b.text())
        self.assertIn('v.ld.w', texts[0])
        self.assertNotIn('v.ld.w', texts[1])
        self.assertIn('v.st.w', texts[1])
        self.assertNotIn('v.st.w', texts[0])
        # And the handoff goes through a register, not memory.
        self.assertIn('v.mov', texts[0])

        p, _peaks = H.multirole_program('triton_cs_add', res.kernel, t)
        padded = ((N + t.vlen - 1) // t.vlen) * t.vlen
        p.check_base(O)
        for i in range(t.vlen):
            p.check_word(O, i * 4,
                         f2b(xv(i) + yv(i)) if i < N else SENTINEL)
        p.words(X, H.pack32([f2b(xv(i)) for i in range(padded)]), 'xs')
        p.words(Y, H.pack32([f2b(yv(i)) for i in range(padded)]), 'ys')
        p.words(O, H.pack32([SENTINEL] * padded), 'os')

        ok, out = H.run(p, extra_args=('+allroles', '+aliasvrf'),
                        max_cycles=200000)
        self.assertTrue(ok, f"context-specialised Triton kernel failed:\n{out}")

    def test_context_specialised_mul_add_kernel(self):
        """A longer value graph: a*b + a splits the same way, with one
        tensor crossing and the address chain recomputed."""
        import kernels_triton as K
        from tnca_cc.partition import split_roles

        t = self.target
        mod, _ctx = triton_bridge.make_ttir(K.mul_add_kernel, SIG,
                                            {'BLOCK': t.vlen})
        kernel, _ = TTIRImporter(t, 'mul_add_kernel').run(
            mod, 'mul_add_kernel',
            {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': N},
            param_names=triton_bridge.param_names(K.mul_add_kernel))
        res = split_roles(kernel, t, name='muladd_cs')
        self.assertEqual(len(res.crossings), 1)

        p, _ = H.multirole_program('triton_cs_muladd', res.kernel, t)
        padded = ((N + t.vlen - 1) // t.vlen) * t.vlen
        p.check_base(O)
        for i in range(t.vlen):
            p.check_word(O, i * 4,
                         f2b(xv(i) * yv(i) + xv(i)) if i < N else SENTINEL)
        p.words(X, H.pack32([f2b(xv(i)) for i in range(padded)]), 'xs')
        p.words(Y, H.pack32([f2b(yv(i)) for i in range(padded)]), 'ys')
        p.words(O, H.pack32([SENTINEL] * padded), 'os')

        ok, out = H.run(p, extra_args=('+allroles', '+aliasvrf'),
                        max_cycles=200000)
        self.assertTrue(ok, f"context-specialised mul_add failed:\n{out}")

    def test_wide_block_add_kernel(self):
        """R40 register chunks cover the entire block and preserve the tail mask."""
        import kernels_triton as K
        for block in (32, 64, 128):
            with self.subTest(block=block):
                p = self._program(K.add_kernel, 'add_kernel',
                                  lambda i: xv(i) + yv(i), block=block)
                ok, out = H.run(p)
                self.assertTrue(ok, f"BLOCK={block} add failed on the RTL:\n{out}")

    def test_unsupported_block_is_refused(self):
        """Blocks above the supported 128-element limit must fail explicitly."""
        import kernels_triton as K
        from tnca_cc.frontend import Unsupported
        mod, _ctx = triton_bridge.make_ttir(K.add_kernel, SIG, {'BLOCK': 256})
        imp = TTIRImporter(self.target, 'add_kernel')
        with self.assertRaises(Unsupported) as cm:
            imp.run(mod, 'add_kernel',
                    {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': N},
                    param_names=triton_bridge.param_names(K.add_kernel))
        self.assertIn('register block width 256', str(cm.exception))
        self.assertIn('up to 128', str(cm.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
