"""`tl.dot` of FP8 tiles, on the RTL: K blocks, operand transfer, mixed encodings.

A tl.dot of two FP8 tiles is ISA sec 9C's FP8 MMA over the sec 9C.2 flat
row-major fragments. Checked here, each on the RTL against the exact
product (the inputs are small multiples of 1/2, so every partial sum is
exact and any K order gives the same bits; the per-K rounding order itself
is checked by tools/performance_harness/fp8_cases.py):

  * K = 32 and 64 in one dot is one K32/K64 MMA (register_blocks.
    _native_matrix). Splitting it as an elementwise operation paired A's
    second K block with B's first and was wrong.
  * An FP8 tile that only tl.dot reads takes the transfer engine
    (TTIRImporter._fp8_mma_operand), like an FP16 one; a column-major one
    is gathered straight into the MMA repack, without a round trip through
    scratch.
  * E4M3 x E5M2 is an FP16 MMA of the exactly converted operands for one
    16-deep K block, and refused with a diagnostic deeper than that.
"""
import itertools
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test')]
import rtl_harness as H
import triton_bridge
from tnca_cc import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.frontend import TTIRImporter, Unsupported

A, B, O, SCRATCH = 0x10000, 0x11000, 0x12000, 0x14000
SENTINEL = 0xdeadbeef

if triton_bridge.available():
    import triton
    import triton.language as tl

    @triton.jit
    def dot_rows(A, B, O, K: tl.constexpr):
        m = tl.arange(0, 16)
        k = tl.arange(0, K)
        n = tl.arange(0, 16)
        a = tl.load(A + m[:, None] * K + k[None, :])
        b = tl.load(B + k[:, None] * 16 + n[None, :])
        tl.store(O + m[:, None] * 16 + n[None, :], tl.dot(a, b))

    @triton.jit
    def dot_col_major_b(A, B, O, K: tl.constexpr):
        # B[k, n] read from a [16, K] weight, as a linear layer's is.
        m = tl.arange(0, 16)
        k = tl.arange(0, K)
        n = tl.arange(0, 16)
        a = tl.load(A + m[:, None] * K + k[None, :])
        b = tl.load(B + k[:, None] + n[None, :] * K)
        tl.store(O + m[:, None] * 16 + n[None, :], tl.dot(a, b))


# Values on the grid of multiples of 1/2 up to 3: exact in both encodings,
# and their products and sums exact in binary32.
VALUES = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0]
CODES = {'e4m3': {0.0: 0x00, 0.5: 0x30, 1.0: 0x38, 1.5: 0x3c, 2.0: 0x40, 3.0: 0x44},
         'e5m2': {0.0: 0x00, 0.5: 0x38, 1.0: 0x3c, 1.5: 0x3e, 2.0: 0x40, 3.0: 0x42}}


def code(v, fmt):
    return CODES[fmt][abs(v)] | (0x80 if v < 0 else 0)


def matrix(rows, cols, seed):
    return [[VALUES[(seed * 7 + r * 5 + c * 3 + r * c) % len(VALUES)] for c in range(cols)]
            for r in range(rows)]


def pack8(values):
    return [sum((v & 0xff) << (8 * j) for j, v in enumerate(values[k:k + 8]))
            for k in range(0, len(values), 8)]


def bits32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class FP8DotTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()

    def compile(self, fn, sig, K):
        module, _ctx = triton_bridge.make_ttir(fn, sig, {'K': K})
        imp = TTIRImporter(self.target, fn.__name__, scratch_addr=SCRATCH, index_opt=True)
        kernel, _ = imp.run(module, fn.__name__, dict(A=A, B=B, O=O),
                            param_names=triton_bridge.param_names(fn))
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return cg, cg.b.text()

    def run_rtl(self, name, cg, a, b, fa, fb, *, b_col_major=False):
        if not H.available():
            self.skipTest(H.why_unavailable())
        K = len(a[0])
        expect = [[sum(a[m][k] * b[k][n] for k in range(K)) for n in range(16)] for m in range(16)]
        p = H.Program(name).kernel(cg).check_base(O)
        for m, n in itertools.product(range(16), range(16)):
            p.check_word(O, (m * 16 + n) * 4, bits32(expect[m][n]))
        p.words(A, pack8([code(v, fa) for row in a for v in row]), 'as')
        bt = [[b[k][n] for k in range(K)] for n in range(16)] if b_col_major else b
        p.words(B, pack8([code(v, fb) for row in bt for v in row]), 'bs')
        p.words(O, H.pack32([SENTINEL] * 256), 'os')
        ok, out = H.run(p)
        self.assertTrue(ok, out)

    def test_deep_k_is_one_mma(self):
        for fmt, ty in (('e4m3', 'fp8e4nv'), ('e5m2', 'fp8e5')):
            for K in (32, 64):
                with self.subTest(fmt=fmt, K=K):
                    cg, asm = self.compile(dot_rows, dict(A=f'*{ty}', B=f'*{ty}', O='*fp32'), K)
                    self.assertEqual(asm.count(f'm.mma.async.k{K}.{fmt}'), 1)
                    self.run_rtl(f'fp8_dot_k{K}_{fmt}', cg, matrix(16, K, 1), matrix(K, 16, 2), fmt, fmt)

    def test_deep_k_split_as_elementwise_is_wrong_on_rtl(self):
        # The control: the dot treated as an elementwise operation over its
        # K chunks, as it was before FP8 was a native matrix operation.
        from unittest.mock import patch
        from tnca_cc import register_blocks
        if not H.available():
            self.skipTest(H.why_unavailable())
        upstream = register_blocks._native_matrix

        def f16_only(imp, op):
            found = upstream(imp, op)
            return None if found is not None and 'f8' in str(op.get_operand(0).get_type()) else found
        from tnca_cc.frontend import Unsupported
        with patch.object(register_blocks, '_native_matrix', f16_only):
            try:
                cg, asm = self.compile(dot_rows, dict(A='*fp8e4nv', B='*fp8e4nv', O='*fp32'), 32)
            except Unsupported:
                # Since the register-block lowering checks that a dot is one
                # matrix operation per column chunk, the elementwise form is
                # refused outright instead of compiling to a wrong result.
                return
        self.assertNotIn('m.mma.async.k32', asm)
        with self.assertRaises(AssertionError):
            self.run_rtl('fp8_dot_k32_split', cg, matrix(16, 32, 1), matrix(32, 16, 2), 'e4m3', 'e4m3')

    def test_operands_take_the_transfer_engine(self):
        cg, asm = self.compile(dot_rows, dict(A='*fp8e4nv', B='*fp8e4nv', O='*fp32'), 16)
        self.assertEqual(asm.count('t.ld.async.b'), 2)
        self.assertNotIn('v.ld.b', asm)
        self.run_rtl('fp8_dot_transfer', cg, matrix(16, 16, 3), matrix(16, 16, 4), 'e4m3', 'e4m3')

    def test_column_major_b_is_gathered_into_the_repack(self):
        cg, asm = self.compile(dot_col_major_b, dict(A='*fp8e4nv', B='*fp8e4nv', O='*fp32'), 16)
        # A by transfer; B gathered row by row and stored packed, once.
        self.assertEqual(asm.count('t.ld.async.b'), 1)
        self.assertEqual(asm.count('v.ld.gather.b'), 16)
        self.assertEqual(asm.count('v.st.b'), 16)
        self.run_rtl('fp8_dot_col_b', cg, matrix(16, 16, 5), matrix(16, 16, 6), 'e4m3', 'e4m3',
                     b_col_major=True)

    def test_mixed_encodings_are_an_exact_fp16_mma(self):
        cg, asm = self.compile(dot_rows, dict(A='*fp8e4nv', B='*fp8e5', O='*fp32'), 16)
        self.assertIn('m.mma.async.k16.f16', asm)
        self.assertNotIn('.e4m3 ', asm.replace('\n', ' \n'))
        self.run_rtl('fp8_dot_mixed', cg, matrix(16, 16, 7), matrix(16, 16, 8), 'e4m3', 'e5m2')

    def test_mixed_encodings_deeper_than_one_block_are_refused(self):
        with self.assertRaises(Unsupported) as ctx:
            self.compile(dot_rows, dict(A='*fp8e4nv', B='*fp8e5', O='*fp32'), 32)
        self.assertIn('E4M3 x E5M2', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
