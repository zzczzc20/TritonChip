#!/usr/bin/env python3
"""
C2: single-RCTX Tensor DMA + MMA (compiler_overview.md sec 15 C2).

The milestone's list: FP16 GEMM, BF16 GEMM, different strides, N/M/K tail,
microfragment layout, accumulator epilogue. And it says this stage is where
大 B tile 不等于紧凑 B fragment should surface -- so the planner is tested on
that directly, not only through the arithmetic.

Results are checked against tools/model/fp32.mma_ref, which accumulates in
increasing k order with one rounding per step, as ISA sec 9C.1 requires.
sec 16.2: numerical and bit-exact comparison are separate questions, and this
is the bit-exact one -- against TNCA's own contract, not another GPU's dot.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

import rtl_harness as H  # noqa: E402
from fp32 import (bf16_bits_to_f32_bits, f16_bits_to_f32_bits, f2b,  # noqa: E402
                  f32_to_f16_bits, fma_ref, mma_ref)
from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.diagnostics import Diagnostic  # noqa: E402
from tnca_cc.gemm import (build_gemm, check_descriptor_geometry,  # noqa: E402
                          plan_gemm)

SENTINEL = 0xdeadbeef
# Placed well above the code: 256 output checks is a few thousand
# instructions, and the assembler refuses an .org that moves backwards rather
# than silently overlaying data on the program.
A_BASE, B_BASE = 0x8000, 0xa000
DA_BASE, DB_BASE = 0xc000, 0xc100
OUT_BASE = 0xd000


def f32_to_bf16_bits(x):
    """Round-to-nearest-even truncation of binary32 to bfloat16."""
    bits = f2b(x)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7fff + lsb) >> 16) & 0xffff


def tensormap(base, e0, e1, s0, s1, b0, b1, rank=2, dtype=0, pad=0):
    """The 256-byte layout of ISA sec 9D.3, as 32 little-endian words."""
    w = [0] * 32
    w[0] = base
    w[1] = e0
    w[2] = e1
    w[6] = s0
    w[7] = s1
    w[11] = (b1 << 32) | b0
    w[13] = (pad << 48) | (dtype << 40) | (rank << 32)
    return w


def mma_ref_bf16(a_bits, b_bits, c, M, N, K):
    """The same contract as mma_ref, with BF16 inputs widened exactly.

    ISA sec 9C.1 after the correction: the product is formed at full width and
    rounded once, with the sum. Widening then using the fused reference is
    what that means.
    """
    out = list(c)
    for m in range(M):
        for n in range(N):
            acc = out[m * N + n]
            for k in range(K):
                av = bf16_bits_to_f32_bits(a_bits[m * K + k])
                bv = bf16_bits_to_f32_bits(b_bits[k * N + n])
                acc = fma_ref(av, bv, acc)
            out[m * N + n] = acc
    return out


class TestPlanner(unittest.TestCase):
    """sec 9: a large tile is not automatically a compact fragment."""

    def setUp(self):
        self.t = TargetInfo.default()

    def test_single_fragment_needs_one_transfer_per_operand(self):
        p = plan_gemm(self.t, m=16, n=16, k=16)
        self.assertEqual(p.k_steps, 1)
        self.assertEqual(p.dma_operations, 2)
        self.assertEqual(p.split_reason, '')

    def test_wide_k_is_split_into_compact_fragments(self):
        """A 16x64 A tile loaded whole would put 64*2 bytes between fragment
        rows where the MMA wants 16*2, and no change of base closes that."""
        p = plan_gemm(self.t, m=16, n=16, k=64)
        self.assertEqual(p.k_steps, 4)
        self.assertEqual(p.dma_operations, 8)
        self.assertIn('compact', p.split_reason)

    def test_the_split_changes_the_barrier_operation_count(self):
        """sec 9: the choice changes DMA command count and barrier operation
        count together, which is why one planner decides both."""
        for k, expect in ((16, 2), (32, 4), (64, 8)):
            with self.subTest(k=k):
                self.assertEqual(plan_gemm(self.t, m=16, n=16, k=k)
                                 .dma_operations, expect)

    def test_partial_k_is_padded_not_refused(self):
        """sec 15 C2's K tail: the short fragment is padded with zero by the
        descriptor, and zero products contribute exactly nothing."""
        p = plan_gemm(self.t, m=16, n=16, k=20)
        self.assertEqual(p.k_steps, 2)
        self.assertTrue(p.k_tail)
        self.assertEqual(p.k_padded, 32)

    def test_unaligned_row_stride_refused(self):
        """13 FP16 elements is a 26-byte row: neither a whole number of
        8-byte units nor a multiple of one. The engine now faults on this;
        the compiler should never emit it."""
        with self.assertRaises(Diagnostic) as cm:
            check_descriptor_geometry(self.t, base=0x8000,
                                      row_stride_bytes=26, elem_bytes=2,
                                      block_cols=16, what='B')
        self.assertIn('multiple of 8', str(cm.exception))

    def test_unaligned_base_refused(self):
        with self.assertRaises(Diagnostic):
            check_descriptor_geometry(self.t, base=0x8004,
                                      row_stride_bytes=32, elem_bytes=2,
                                      block_cols=16, what='A')

    def test_output_tile_larger_than_one_accumulator_refused(self):
        with self.assertRaises(Diagnostic):
            plan_gemm(self.t, m=32, n=16, k=16)

    def test_unknown_dtype_refused(self):
        with self.assertRaises(Diagnostic):
            plan_gemm(self.t, m=16, n=16, k=16, dtype='f8')


class TestGemmOnRTL(unittest.TestCase):
    def setUp(self):
        self.t = TargetInfo.default()
        if not H.available():
            self.skipTest(H.why_unavailable())

    def _run_gemm(self, *, k, dtype, lda, ldb, seed, m=16, n=16):
        """Build, compile and run one MxNxK GEMM."""
        t = self.t
        M, N = m, n
        rng = random.Random(seed)
        to_bits = f32_to_f16_bits if dtype == 'f16' else f32_to_bf16_bits
        widen = (f16_bits_to_f32_bits if dtype == 'f16'
                 else bf16_bits_to_f32_bits)

        # A is M x lda, B is k x ldb: the tile is a window onto a wider
        # matrix, so the strides are not the tile's own.
        a_store = [to_bits(rng.uniform(-4, 4)) for _ in range(M * lda)]
        b_store = [to_bits(rng.uniform(-4, 4)) for _ in range(k * ldb)]
        a_tile = [a_store[r * lda + kk] for r in range(M) for kk in range(k)]
        b_tile = [b_store[kk * ldb + c] for kk in range(k) for c in range(N)]

        ref = (mma_ref(a_tile, b_tile, [0] * (M * N), M, N, k)
               if dtype == 'f16'
               else mma_ref_bf16(a_tile, b_tile, [0] * (M * N), M, N, k))

        plan = plan_gemm(t, m=M, n=N, k=k, dtype=dtype)
        # The compiler refuses a geometry the engine cannot execute rather
        # than emitting it and getting shifted data back (ISA sec 9E.2).
        check_descriptor_geometry(t, base=A_BASE, row_stride_bytes=lda * 2,
                                  elem_bytes=2, block_cols=plan.frag_k,
                                  what='A')
        check_descriptor_geometry(t, base=B_BASE, row_stride_bytes=ldb * 2,
                                  elem_bytes=2, block_cols=plan.frag_n,
                                  what='B')
        kernel, _ = build_gemm(t, plan, name=f"gemm_{dtype}_{k}",
                               a_desc_addr=DA_BASE, b_desc_addr=DB_BASE,
                               out_addr=OUT_BASE)
        cg = H.compile_kernel(kernel, t, check_protocol=False)

        p = H.Program(f"gemm_{dtype}_{M}x{N}x{k}_lda{lda}_ldb{ldb}_s{seed}")
        p.kernel(cg)
        p.check_base(OUT_BASE)
        for i in range(M * N):
            p.check_word(OUT_BASE, i * 4, ref[i])
        # Past the real tile the output must be untouched: an N tail whose
        # masked store leaked would still pass every check above.
        for i in range(M * N, M * N + N):
            p.check_word(OUT_BASE, i * 4, SENTINEL)

        dt = 0 if dtype == 'f16' else 1
        p.words(A_BASE, H.pack16(a_store), 'a')
        p.words(B_BASE, H.pack16(b_store), 'b')
        # The descriptor extents are the REAL tile, not the fragment. What
        # lies past them is filled with zero by the transfer engine
        # (ISA sec 9E.1 step 7), which is what makes an M/N/K tail come out
        # exactly right rather than approximately.
        p.words(DA_BASE, tensormap(A_BASE, M, k, lda * 2, 2,
                                   plan.frag_m, plan.frag_k, dtype=dt),
                'desc_a')
        p.words(DB_BASE, tensormap(B_BASE, k, N, ldb * 2, 2,
                                   plan.frag_k, plan.frag_n, dtype=dt),
                'desc_b')
        p.words(OUT_BASE, H.pack32([SENTINEL] * (M * N + N + 4)), 'out')

        ok, out = H.run(p)
        self.assertTrue(ok, f"GEMM {dtype} K={k} lda={lda} failed:\n{out}")

    def test_fp16_gemm_k16(self):
        self._run_gemm(k=16, dtype='f16', lda=16, ldb=16, seed=1)

    def test_fp16_gemm_k32_split_fragments(self):
        """Two k steps: the split of sec 9 option 1, on the machine."""
        self._run_gemm(k=32, dtype='f16', lda=32, ldb=16, seed=2)

    def test_fp16_gemm_wider_stride(self):
        """sec 15 C2's 不同 stride: the tile is a window onto a wider matrix,
        so the row stride is not the tile width."""
        self._run_gemm(k=16, dtype='f16', lda=48, ldb=40, seed=3)

    def test_bf16_gemm(self):
        """sec 15 C2's BF16 GEMM. The reference widens exactly and rounds
        once, per ISA sec 9C.1 as corrected."""
        self._run_gemm(k=16, dtype='bf16', lda=16, ldb=16, seed=4)

    def test_bf16_gemm_k32(self):
        self._run_gemm(k=32, dtype='bf16', lda=32, ldb=16, seed=5)

    def test_k_tail(self):
        """sec 15 C2's K tail. K=20 is not a multiple of the fragment K, so
        the second fragment is half padding. The padding is zero, and
        RNE_f32(exact(0)*exact(b) + C) is exactly C, so the padded steps
        contribute nothing rather than something small."""
        self._run_gemm(k=20, dtype='f16', lda=20, ldb=16, seed=6)

    def test_m_tail(self):
        """Fewer rows than the fragment: the epilogue stores only the real
        ones."""
        self._run_gemm(k=16, dtype='f16', lda=16, ldb=16, seed=7, m=11)

    def test_n_tail(self):
        """Fewer columns than the fragment: the epilogue store is masked, and
        the sentinel past the tile proves nothing leaked."""
        self._run_gemm(k=16, dtype='f16', lda=16, ldb=16, seed=8, n=13)

    def test_all_three_tails(self):
        self._run_gemm(k=20, dtype='f16', lda=20, ldb=16, seed=9, m=11, n=13)

    def test_bf16_all_tails(self):
        self._run_gemm(k=20, dtype='bf16', lda=20, ldb=16, seed=10, m=11,
                       n=13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
