"""A [R, 1] -> [R, C] broadcast copies lane 0 unless the rows are proven replicated.

A loaded [R, 1] holds each row's value in lane 0 only, and so does anything
computed lane by lane from it: those keep the per-row lane-0 broadcast. A
column computed from row indices (an affine index, a comparison of one, a
conversion of one) holds the row's value in every lane already; its rows are
the broadcast and are renamed as R40 did, with no extract/splat per row.
Both kinds run on the RTL and every output lane is checked.
"""
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
from tnca_cc.frontend import TTIRImporter

X, B, O, SCRATCH = 0x10000, 0x11000, 0x12000, 0x14000
SENTINEL = 0xdeadbeef

if triton_bridge.available():
    import triton
    import triton.language as tl

    @triton.jit
    def replicated_column(X, B, O):
        r = tl.arange(0, 16)
        c = tl.arange(0, 16)
        # Computed from row indices: every lane of row r holds 3r + 1.
        col = (r[:, None] * 3 + 1).to(tl.float32)
        keep = r[:, None] < 11
        x = tl.load(X + r[:, None] * 16 + c[None, :])
        tl.store(O + r[:, None] * 16 + c[None, :], tl.where(keep, x + col, -x))

    @triton.jit
    def loaded_column(X, B, O):
        r = tl.arange(0, 16)
        c = tl.arange(0, 16)
        # Loaded: row r's value is in lane 0 only, before and after the
        # elementwise operations, including one with a replicated column.
        b = tl.load(B + r[:, None]) * 2.0 + (r[:, None] * 3 + 1).to(tl.float32)
        x = tl.load(X + r[:, None] * 16 + c[None, :])
        tl.store(O + r[:, None] * 16 + c[None, :], x + b)


def f2b(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def xv(r, c):
    return r * 16 + c * 0.25


def bv(r):
    return r * 0.5 - 2.0


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class ColumnBroadcastTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()

    def compile(self, fn, index_opt):
        sig = dict(X='*fp32', B='*fp32', O='*fp32')
        module, _ctx = triton_bridge.make_ttir(fn, sig, {})
        imp = TTIRImporter(self.target, fn.__name__, scratch_addr=SCRATCH, index_opt=index_opt)
        kernel, _ = imp.run(module, fn.__name__, dict(X=X, B=B, O=O),
                            param_names=triton_bridge.param_names(fn))
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return imp, cg

    def run_rtl(self, name, cg, expect, *, must_pass=True):
        if not H.available():
            self.skipTest(H.why_unavailable())
        p = H.Program(name).kernel(cg).check_base(O)
        for r in range(16):
            for c in range(16):
                p.check_word(O, (r * 16 + c) * 4, f2b(expect(r, c)))
        p.words(X, H.pack32([f2b(xv(r, c)) for r in range(16) for c in range(16)]), 'xs')
        p.words(B, H.pack32([f2b(bv(r)) for r in range(16)]), 'bs')
        p.words(O, H.pack32([SENTINEL] * 256), 'os')
        ok, out = H.run(p)
        if must_pass:
            self.assertTrue(ok, out)
        return ok, out

    def test_replicated_rows_are_renamed(self):
        for index_opt in (False, True):
            with self.subTest(index_opt=index_opt):
                imp, cg = self.compile(replicated_column, index_opt)
                # The float column and the row mask: both proven, no lane-0 copy.
                self.assertEqual(imp._column_broadcasts, {'lane0': 0, 'replicated': 2})
                self.run_rtl(f'replicated_column_{int(index_opt)}', cg,
                             lambda r, c: xv(r, c) + (3 * r + 1) if r < 11 else -xv(r, c))

    def test_loaded_column_keeps_lane0_broadcast(self):
        for index_opt in (False, True):
            with self.subTest(index_opt=index_opt):
                imp, cg = self.compile(loaded_column, index_opt)
                self.assertEqual(imp._column_broadcasts, {'lane0': 1, 'replicated': 0})
                self.run_rtl(f'loaded_column_{int(index_opt)}', cg,
                             lambda r, c: xv(r, c) + (bv(r) * 2.0 + (3 * r + 1)))

    def test_loaded_column_without_the_copy_is_wrong_on_rtl(self):
        # The control: treat the loaded column as replicated and rename its
        # rows. Lanes 1..15 then miss the row's value on the actual RTL.
        from unittest.mock import patch
        with patch.object(TTIRImporter, '_rows_replicated', lambda self, value: True):
            imp, cg = self.compile(loaded_column, True)
        self.assertEqual(imp._column_broadcasts, {'lane0': 0, 'replicated': 1})
        ok, out = self.run_rtl('loaded_column_unsafe', cg,
                               lambda r, c: xv(r, c) + (bv(r) * 2.0 + (3 * r + 1)),
                               must_pass=False)
        self.assertFalse(ok, 'the lane-0 copy must be what makes the loaded column correct')


if __name__ == '__main__':
    unittest.main()
