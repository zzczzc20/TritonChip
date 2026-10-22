"""Conversions with an FP8 end carry PyTorch's NaN encodings, on the RTL.

ISA sec 8.4 converts every NUMBER the way PyTorch's CPU conversion does
(round to nearest even, subnormals kept, E4M3 overflow and infinity
saturating to +-448, E5M2 overflow to +-inf); its NaN result is positive
and canonical. The importer rewrites the NaN lanes of a conversion with an
FP8 end (TTIRImporter._fp8_nan_fix): to FP8, a NaN keeps its sign; from
FP8, a NaN keeps sign and payload in binary32 and FP16 and is 0x7FC0 in
BF16 -- looking through the binary32 widening Inductor puts between a
16-bit load and the FP8 conversion, and between an FP8 value and a 16-bit
store. Each kernel converts sixteen chosen inputs; the RTL's output words
are checked against `torch_*` below, a restatement of torch's c10
conversion (c10/util/Float8_e4m3fn.h, Float8_e5m2.h; checked for every
input against torch itself by tools/performance_harness/
fp8_numeric_exhaustive.py). The assembly is checked for the rule, and for
its absence where nothing can read the NaN and on conversions with no FP8
end -- those keep exactly the code they had.
"""
import math
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

X, O, SCRATCH = 0x10000, 0x12000, 0x14000
SENTINEL = 0xdeadbeef

if triton_bridge.available():
    import triton
    import triton.language as tl

    @triton.jit
    def to_e4m3(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tl.load(X + i).to(tl.float8e4nv))

    @triton.jit
    def to_e5m2(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tl.load(X + i).to(tl.float8e5))

    @triton.jit
    def widened_to_e4m3(X, O):
        # Inductor's form for a 16-bit tensor: the load widened, then cast.
        i = tl.arange(0, 16)
        tl.store(O + i, tl.load(X + i).to(tl.float32).to(tl.float8e4nv))

    @triton.jit
    def to_f32(X, O):
        # Also Inductor's form for an FP8 -> BF16/FP16 cast: widen, then
        # the store narrows to the pointer's element type.
        i = tl.arange(0, 16)
        tl.store(O + i, tl.load(X + i).to(tl.float32))

    @triton.jit
    def to_f32_scaled(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tl.load(X + i).to(tl.float32) * 2.0)

    @triton.jit
    def tile_to_f32(X, O):
        r = tl.arange(0, 16)
        c = tl.arange(0, 16)
        x = tl.load(X + r[:, None] * 16 + c[None, :])
        tl.store(O + r[:, None] * 16 + c[None, :], x.to(tl.float32))

    @triton.jit
    def f32_to_f16(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tl.load(X + i).to(tl.float16))


# ---- torch's conversions (c10), restated ------------------------------------
def f32(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xffffffff))[0]


def bits32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def torch_e4m3_from_f32(b):
    """c10::detail::fp8e4m3fn_from_fp32_value (torch 2.14)."""
    sign = b & 0x80000000
    f = b ^ sign
    if f >= 1087 << 20:
        r = 0x7f if f > 0x7f800000 else 0x7e
    elif f < 121 << 23:
        r = (bits32(f32(f) + f32(141 << 23)) - (141 << 23)) & 0xff
    else:
        f = (f + ((7 - 127) << 23) + 0x7ffff + ((f >> 20) & 1)) & 0xffffffff
        r = (f >> 20) & 0xff
        r = 0x7e if r == 0x7f else r
    return r | (sign >> 24)


def torch_e5m2_from_f32(b):
    """c10::detail::fp8e5m2_from_fp32_value."""
    sign = b & 0x80000000
    f = b ^ sign
    if f >= 143 << 23:
        r = 0x7f if f > 255 << 23 else 0x7c
    elif f < 113 << 23:
        r = (bits32(f32(f) + f32(134 << 23)) - (134 << 23)) & 0xff
    else:
        f = (f + ((15 - 127) << 23) + 0xfffff + ((f >> 21) & 1)) & 0xffffffff
        r = (f >> 21) & 0xff
    return r | (sign >> 24)


def f16_to_f32_bits(h):
    """IEEE binary16 -> binary32, NaN payload kept and quieted (what the
    host's conversion does, and what torch's CPU path returns)."""
    s, e, m = h >> 15, (h >> 10) & 31, h & 1023
    if e == 31:
        return (s << 31) | 0x7f800000 | ((m << 13) | (0x400000 if m else 0))
    return bits32((-1) ** s * (m * 2.0 ** -24 if e == 0 else (1 + m / 1024) * 2.0 ** (e - 15)))


FORMATS = {'e4m3': (4, 3), 'e5m2': (5, 2)}


def fp8_value(code, fmt):
    E, M = FORMATS[fmt]
    s, e, m = code >> 7, (code >> M) & ((1 << E) - 1), code & ((1 << M) - 1)
    bias = (1 << (E - 1)) - 1
    if fmt == 'e4m3' and e == 15 and m == 7:
        return None
    if fmt == 'e5m2' and e == 31:
        return None if m else (-math.inf if s else math.inf)
    v = m / (1 << M) * 2.0 ** (1 - bias) if e == 0 else (1 + m / (1 << M)) * 2.0 ** (e - bias)
    return -v if s else v


def torch_fp8_to(code, fmt, dst):
    """torch CPU: FP8 -> float (c10), then -> dst (c10 Half / BFloat16)."""
    E, M = FORMATS[fmt]
    v = fp8_value(code, fmt)
    s = code >> 7
    if v is None:                                     # NaN: payload kept, quieted
        m = code & ((1 << M) - 1)
        if dst == 'f32':
            return (s << 31) | 0x7fc00000 | (m << (23 - M))
        if dst == 'f16':
            return (s << 15) | 0x7e00 | (m << (10 - M))
        return 0x7fc0                                 # c10 BFloat16: one NaN
    b = bits32(v)
    if dst == 'f32':
        return b
    if dst == 'bf16':
        return b >> 16                                # every FP8 value is a BF16 value
    return struct.unpack('<H', struct.pack('<e', v))[0]


# ---- inputs -----------------------------------------------------------------
F32_IN = [0x00000000, 0x80000000, 0x3f800000, 0xbf880000,        # +-0, 1, -1.0625 (tie)
          0x43e00000, 0x43e80000, 0x43e80001, 0x43f00000,        # 448, 464 (tie), 464+, 480
          0x47700000, 0x7f800000, 0xff800000, 0x7fc00000,        # 61440 (tie), +-inf, +NaN
          0xffc00000, 0xff800001, 0x3a800000, 0x3ac00000]        # -NaN, -sNaN, 2^-10, 1.5*2^-10
F16_IN = [0x0000, 0x8000, 0x3c00, 0xbc40, 0x5f00, 0x5f40, 0x7bff, 0x7c00,
          0xfc00, 0x7e00, 0xfe00, 0xfc01, 0x0001, 0x1400, 0x1600, 0x8400]
E4M3_IN = [0x00, 0x80, 0x01, 0x07, 0x08, 0x38, 0x7e, 0xfe,
           0x7f, 0xff, 0x3f, 0x40, 0x77, 0xc5, 0x81, 0x79]
E5M2_IN = [0x00, 0x80, 0x01, 0x03, 0x04, 0x3c, 0x7b, 0x7c,
           0xfc, 0x7d, 0x7e, 0x7f, 0xfd, 0xfe, 0xff, 0x83]


def pack8(values):
    out = []
    for k in range(0, len(values), 8):
        out.append(sum((v & 0xff) << (8 * j) for j, v in enumerate(values[k:k + 8])))
    return out


def words(values, width):
    """The little-endian 32-bit words a run of `width`-byte values makes."""
    raw = b''.join(v.to_bytes(width, 'little') for v in values)
    return [int.from_bytes(raw[k:k + 4], 'little') for k in range(0, len(raw), 4)]


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class FP8ConversionTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()

    def compile(self, fn, sig):
        module, _ctx = triton_bridge.make_ttir(fn, sig, {})
        imp = TTIRImporter(self.target, fn.__name__, scratch_addr=SCRATCH, index_opt=True)
        kernel, _ = imp.run(module, fn.__name__, dict(X=X, O=O),
                            param_names=triton_bridge.param_names(fn))
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return cg, cg.b.text()

    def run_rtl(self, name, cg, data, expect_words):
        if not H.available():
            self.skipTest(H.why_unavailable())
        p = H.Program(name).kernel(cg).check_base(O)
        for k, w in enumerate(expect_words):
            p.check_word(O, 4 * k, w)
        p.words(X, data, 'xs')
        p.words(O, H.pack32([SENTINEL] * max(len(expect_words), 2)), 'os')
        ok, out = H.run(p)
        self.assertTrue(ok, out)

    def test_narrowing_keeps_the_nan_sign(self):
        for fn, fmt, ref in ((to_e4m3, 'fp8e4nv', torch_e4m3_from_f32),
                             (to_e5m2, 'fp8e5', torch_e5m2_from_f32)):
            with self.subTest(fmt=fmt):
                cg, asm = self.compile(fn, dict(X='*fp32', O=f'*{fmt}'))
                self.assertIn('a NaN keeps its sign', asm)
                expect = [ref(b) for b in F32_IN]
                self.assertEqual(expect[11:14], [0x7f, 0xff, 0xff])      # +NaN, -NaN, -sNaN
                self.run_rtl(f'fp8_narrow_{fmt}', cg, H.pack32(F32_IN), words(expect, 1))

    def test_narrowing_looks_through_the_widened_load(self):
        cg, asm = self.compile(widened_to_e4m3, dict(X='*fp16', O='*fp8e4nv'))
        # The sign comes from the 16-bit value: the binary32 widening of an
        # FP16 NaN is the canonical, positive NaN.
        self.assertIn('f16 -> f8E4M3FN: a NaN keeps its sign', asm)
        expect = [torch_e4m3_from_f32(f16_to_f32_bits(h)) for h in F16_IN]
        self.assertEqual(expect[10:12], [0xff, 0xff])
        self.run_rtl('fp8_narrow_f16', cg, H.pack16(F16_IN), words(expect, 1))

    def test_widening_keeps_sign_and_payload(self):
        for fmt, codes, name in (('fp8e4nv', E4M3_IN, 'e4m3'), ('fp8e5', E5M2_IN, 'e5m2')):
            for dst, width in (('fp32', 4), ('fp16', 2), ('bf16', 2)):
                with self.subTest(fmt=fmt, dst=dst):
                    cg, asm = self.compile(to_f32, dict(X=f'*{fmt}', O=f'*{dst}'))
                    self.assertIn('vf.cmp.o', asm)
                    self.assertIn('v.sel', asm)
                    expect = [torch_fp8_to(c, name, {'fp32': 'f32', 'fp16': 'f16', 'bf16': 'bf16'}[dst])
                              for c in codes]
                    self.run_rtl(f'fp8_widen_{name}_{dst}', cg, pack8(codes), words(expect, width))

    def test_tile_widening(self):
        codes = [E4M3_IN[(r * 5 + c) % 16] for r in range(16) for c in range(16)]
        cg, asm = self.compile(tile_to_f32, dict(X='*fp8e4nv', O='*fp32'))
        self.assertIn('v.sel', asm)
        self.run_rtl('fp8_widen_tile', cg, pack8(codes), [torch_fp8_to(c, 'e4m3', 'f32') for c in codes])

    def test_the_instruction_alone_drops_the_sign_on_rtl(self):
        # The control: without the rule the kernel is the bare `vf.cvt`, and
        # on the actual RTL a negative NaN comes out as the positive 0x7F.
        from unittest.mock import patch
        if not H.available():
            self.skipTest(H.why_unavailable())
        with patch.object(TTIRImporter, '_fp8_nan_rule', lambda *args: None):
            cg, asm = self.compile(to_e4m3, dict(X='*fp32', O='*fp8e4nv'))
        self.assertNotIn('a NaN keeps its sign', asm)
        bare = [torch_e4m3_from_f32(b) for b in F32_IN]
        bare[12] = bare[13] = 0x7f
        p = H.Program('fp8_narrow_bare').kernel(cg).check_base(O)
        for k, w in enumerate(words(bare, 1)):
            p.check_word(O, 4 * k, w)
        p.words(X, H.pack32(F32_IN), 'xs')
        p.words(O, H.pack32([SENTINEL] * 4), 'os')
        ok, out = H.run(p)
        self.assertTrue(ok, out)

    def test_no_rule_where_no_nan_can_be_read(self):
        # An FP8 value widened only to be multiplied: the product's NaN is
        # the arithmetic's canonical one whatever the input's was.
        cg, asm = self.compile(to_f32_scaled, dict(X='*fp8e4nv', O='*fp32'))
        self.assertNotIn('vf.cmp.o', asm)
        self.assertNotIn('v.sel', asm)

    def test_no_rule_without_an_fp8_end(self):
        cg, asm = self.compile(f32_to_f16, dict(X='*fp32', O='*fp16'))
        body = []
        for line in asm.splitlines():
            tokens = line.split('//')[0].split()
            if tokens and not tokens[0].startswith(('.', ';')) and not tokens[0].endswith(':'):
                body.append(tokens[1] if tokens[0].startswith('@') and len(tokens) > 1 else tokens[0])
        self.assertEqual(sum(op.startswith('vf.cvt') for op in body), 1)
        for op in ('v.or', 'v.sel', 'vf.cmp.o', 'v.lshr'):
            self.assertNotIn(op, body)


if __name__ == '__main__':
    unittest.main()
