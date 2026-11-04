"""Signed FP8 narrowing through the emitted ISA, including register pressure."""
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler/python'), str(ROOT / 'compiler/test')]
import rtl_harness as H
from test_fp8_convert import F32_IN, torch_e4m3_from_f32, torch_e5m2_from_f32, words
from tnca_cc import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.diagnostics import Diagnostic, ResourceError
from tnca_cc.ir import Builder, make_kernel


class FP8SignedCodegenTests(unittest.TestCase):
    def test_signed_narrowing_rejects_other_format_pairs(self):
        k = make_kernel('fp8_signed_invalid')
        b = Builder(k, k.role(0).region)
        x = b.load(b.const(0x10000), dtype='f32', size='w')
        before = len(k.role(0).region.ops)
        for src, dst in (('f16', 'f8E4M3FN'), ('f32', 'f16')):
            with self.assertRaises(Diagnostic):
                b.vcvt(src, dst, x, preserve_fp8_nan_sign=True)
        self.assertEqual(len(k.role(0).region.ops), before)

    def run_conversion(self, fmt, reference, pressure):
        if not H.available():
            self.skipTest(H.why_unavailable())
        k = make_kernel('fp8_signed')
        b = Builder(k, k.role(0).region)
        x = b.load(b.const(0x10000), dtype='f32', size='w')
        y = b.vcvt('f32', fmt, x, preserve_fp8_nan_sign=True)
        b.store(b.const(0x12000), y, size='b')
        b.exit()
        cg = CodeGen(k, TargetInfo.default(), check_protocol=False)
        cg.assign_shared()
        if pressure:
            with mock.patch.object(cg.alloc, 'predicate', side_effect=ResourceError('full')):
                cg.run()
        else:
            cg.run()
        asm = cg.b.text()
        self.assertIn('a NaN keeps its sign', asm)
        self.assertIn('v.lshri' if pressure else 'v.cmpi.lt', asm)
        expected = words([reference(x) for x in F32_IN], 1)
        p = H.Program('fp8_signed').kernel(cg).check_base(0x12000)
        for i, word in enumerate(expected):
            p.check_word(0x12000, 4 * i, word)
        p.words(0x10000, H.pack32(F32_IN), 'xs')
        p.words(0x12000, H.pack32([0xdeadbeef] * 4), 'os')
        ok, output = H.run(p)
        self.assertTrue(ok, output)

    def test_finite_nan_and_zero_bits_with_and_without_predicate(self):
        for fmt, ref in (('f8E4M3FN', torch_e4m3_from_f32),
                         ('f8E5M2', torch_e5m2_from_f32)):
            for pressure in (False, True):
                with self.subTest(fmt=fmt, pressure=pressure):
                    self.run_conversion(fmt, ref, pressure)

    def test_source_remains_available_after_conversion(self):
        if not H.available():
            self.skipTest(H.why_unavailable())
        k = make_kernel('fp8_signed_live_source')
        b = Builder(k, k.role(0).region)
        x = b.load(b.const(0x10000), dtype='f32', size='w')
        y = b.vcvt('f32', 'f8E4M3FN', x, preserve_fp8_nan_sign=True)
        b.store(b.const(0x12000), y, size='b')
        b.store(b.const(0x13000), x, size='w')
        b.exit()
        cg = CodeGen(k, TargetInfo.default(), check_protocol=False)
        cg.assign_shared()
        cg.run()
        p = H.Program('fp8_signed_live_source').kernel(cg).check_base(0x12000)
        for i, word in enumerate(words([torch_e4m3_from_f32(x) for x in F32_IN], 1)):
            p.check_word(0x12000, 4 * i, word)
        for i, word in enumerate(F32_IN):
            p.check_word(0x13000, 4 * i, word)
        p.words(0x10000, H.pack32(F32_IN), 'xs')
        p.words(0x12000, H.pack32([0xdeadbeef] * 4), 'os')
        p.words(0x13000, H.pack32([0xdeadbeef] * len(F32_IN)), 'live')
        ok, output = H.run(p)
        self.assertTrue(ok, output)


if __name__ == '__main__':
    unittest.main()
