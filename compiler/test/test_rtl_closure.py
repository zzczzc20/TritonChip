#!/usr/bin/env python3
"""
C1: compiled kernels run on the RTL (compiler_overview.md sec 15 C1).

The milestone's list: vector add, masked elementwise, integer reduction,
FP32 reduction, simple control flow, and a gather/scatter subset. The bar is
不修改输出汇编，不人工补 barrier，直接运行编译产物 -- so each test compiles
IR, appends only data and assertions, assembles with the existing
tools/asm/tnca_asm.py, and runs on the Icarus model of the core.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

import rtl_harness as H  # noqa: E402
from fp32 import f2b  # noqa: E402
from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.kernels import elementwise, gather_kernel, reduction  # noqa: E402

SENTINEL = 0xdeadbeef


class RTLCase(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()
        if not H.available():
            self.skipTest(H.why_unavailable())

    def assertRTL(self, program, **kw):
        ok, out = H.run(program, **kw)
        self.assertTrue(ok, f"compiled kernel failed on the RTL:\n{out}")


class TestVectorAdd(RTLCase):
    N = 40
    X, Y, O = 0x4000, 0x4400, 0x4800

    @staticmethod
    def xv(i):
        return i * 0.5

    @staticmethod
    def yv(i):
        return i * 0.25 + 1.0

    def _program(self, op='add'):
        t = self.target
        padded = ((self.N + t.vlen - 1) // t.vlen) * t.vlen
        k, _ = elementwise(t, name='add_kernel', op=op, n_elements=self.N,
                           addresses={'x': self.X, 'y': self.Y, 'out': self.O})
        cg = H.compile_kernel(k, t, check_protocol=False)

        p = H.Program(f"vecadd_{op}")
        p.kernel(cg)
        p.check_base(self.O)
        for i in range(self.N):
            a, b = self.xv(i), self.yv(i)
            want = a + b if op == 'add' else a * b
            p.check_word(self.O, i * 4, f2b(want))
        # The masked tail must not have been stored: a predicated store that
        # wrote it would still pass every arithmetic check above.
        for i in range(self.N, padded):
            p.check_word(self.O, i * 4, SENTINEL)

        p.words(self.X, H.pack32([f2b(self.xv(i)) for i in range(padded)]), 'xs')
        p.words(self.Y, H.pack32([f2b(self.yv(i)) for i in range(padded)]), 'ys')
        p.words(self.O, H.pack32([SENTINEL] * padded), 'os')
        return p

    def test_masked_vector_add(self):
        self.assertRTL(self._program('add'))

    def test_masked_vector_mul(self):
        self.assertRTL(self._program('mul'))

    def test_matches_hand_written_reference(self):
        """sec 11.1 keeps the comparison available so a compiler bug and an
        encoding bug can be told apart."""
        t = self.target
        k, _ = elementwise(t, name='add_kernel', n_elements=self.N,
                           addresses={'x': self.X, 'y': self.Y, 'out': self.O})
        cg = H.compile_kernel(k, t, check_protocol=False)

        ref_path = os.path.join(ROOT, 'tb', 'core', 'progs', 't16_vecadd.s')
        if not os.path.exists(ref_path):
            self.skipTest('reference not generated')

        def is_insn(line):
            line = line.split('//')[0].strip()
            return bool(line) and not line.endswith(':')

        def shape(line):
            line = line.split('//')[0].strip()
            line = re.sub(r'\b([svpabdw])\d+\b', r'\1?', line)
            return re.sub(r'\s+', ' ', line)

        emitted = [shape(l) for l in cg.b.text().splitlines() if is_insn(l)]
        with open(ref_path) as fh:
            ref_lines = fh.readlines()
        ref = [shape(l) for l in ref_lines
               if is_insn(l) and not l.strip().startswith(('.org', '.dword'))]
        end = next(i for i, l in enumerate(ref) if 'br.nz' in l)
        ref = ref[:end + 1]

        # Compared as a multiset, not a sequence. Two differences are real and
        # neither is a defect: the lowering interleaves each `v.movi` with the
        # load it primes where the hand-written version grouped them, and it
        # updates the loop counter as add-then-move where the hand-written
        # version incremented in place. Demanding an identical order would
        # make this test fail on any scheduling change while catching nothing;
        # what it is for is noticing a missing or invented instruction, and
        # the RTL run above is what proves the result.
        from collections import Counter
        missing = Counter(ref) - Counter(emitted)
        extra = Counter(emitted) - Counter(ref)
        # The counter update differs by construction: ref has one s.addi,
        # emitted has s.addi plus s.mov.
        extra.pop('s.mov s?, s?', None)
        self.assertFalse(missing, f"the lowering dropped: {dict(missing)}")
        self.assertFalse(extra, f"the lowering invented: {dict(extra)}")


class TestReduction(RTLCase):
    SRC, OUT = 0x4000, 0x4800

    def _program(self, op, dtype, n, values, expect):
        t = self.target
        k, _ = reduction(t, name=f"red_{op}_{dtype}", op=op, dtype=dtype,
                         n_elements=n,
                         addresses={'x': self.SRC, 'out': self.OUT})
        cg = H.compile_kernel(k, t, check_protocol=False)
        p = H.Program(f"reduce_{op}_{dtype}")
        p.kernel(cg)
        p.check_base(self.OUT)
        p.check_word(self.OUT, 0, expect)
        padded = list(values) + [0] * (t.vlen - len(values))
        p.words(self.SRC, H.pack32(padded), 'src')
        p.words(self.OUT, H.pack32([SENTINEL] * t.vlen), 'dst')
        return p

    def test_integer_sum_masked(self):
        """sec 15 C1: integer reduction. n < VLEN, so the tail is masked and
        ISA sec 8.7's identity rule is what makes it come out right."""
        n = 11
        vals = [i * 3 + 1 for i in range(n)]
        self.assertRTL(self._program('add', 'i32', n,
                                     [v & 0xffffffff for v in vals],
                                     sum(vals)))

    def test_integer_max_masked(self):
        n = 9
        vals = [(i * 7) % 23 for i in range(n)]
        self.assertRTL(self._program('max', 'i32', n, vals, max(vals)))

    def test_fp32_sum_masked(self):
        """sec 15 C1: FP32 reduction. The order is a fixed balanced tree
        (ISA sec 8.7), so the expected value is computed the same way rather
        than left to host summation order."""
        n = 13
        vals = [float(i) * 0.25 + 0.5 for i in range(n)]

        def tree(xs):
            xs = list(xs) + [0.0] * (self.target.vlen - len(xs))
            while len(xs) > 1:
                xs = [xs[i] + xs[i + 1] for i in range(0, len(xs), 2)]
            return xs[0]

        self.assertRTL(self._program('add', 'f32', n,
                                     [f2b(v) for v in vals], f2b(tree(vals))))


class TestGather(RTLCase):
    SRC, IDX, OUT = 0x4000, 0x4400, 0x4800

    def test_gather_by_element_index(self):
        """sec 15 C1's gather/scatter subset.

        ISA sec 9.1 scales the index by the access size, so these are element
        indices. A lowering that emitted byte offsets would pass lane 0 and
        fail every other lane.
        """
        t = self.target
        n = 12
        src = [0x11111100 | i for i in range(t.vlen)]
        idx = [(t.vlen - 1 - i) for i in range(n)] + [0] * (t.vlen - n)

        k, _ = gather_kernel(t, name='gather_kernel', n_elements=n, dtype='i32',
                             addresses={'x': self.SRC, 'i': self.IDX,
                                        'out': self.OUT})
        cg = H.compile_kernel(k, t, check_protocol=False)
        p = H.Program('gather')
        p.kernel(cg)
        p.check_base(self.OUT)
        for i in range(n):
            p.check_word(self.OUT, i * 4, src[idx[i]])
        for i in range(n, t.vlen):
            p.check_word(self.OUT, i * 4, SENTINEL)
        p.words(self.SRC, H.pack32(src), 'src')
        p.words(self.IDX, H.pack32(idx), 'idx')
        p.words(self.OUT, H.pack32([SENTINEL] * t.vlen), 'dst')
        self.assertRTL(p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
