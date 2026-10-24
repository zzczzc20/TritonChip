"""log1p, sin, cos, erf and int64 -> binary32 conversion, on the RTL.

tnca_cc.mathlib.log1p/sin/cos/erf are built from integer and fma instructions
only (no SFU), and tnca_cc.wide.to_f32 converts a 64-bit integer held as two
32-bit lanes. Each kernel here maps sixteen chosen inputs; the RTL's output
words are checked bit for bit against the same functions executed on numpy
by tools/performance_harness/transcendental_numeric.py (Emu: the builder
calls executed instead of emitted), and that expectation is itself checked
against float64 (within one binary32 ulp) or, for the conversion, against
numpy's int64 -> float32 rounding. The inputs cover the special values,
tiny and subnormal arguments, the boundaries of each reduction, arguments
near multiples of pi/2 and the largest binary32 magnitudes.
"""
import math
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test'), str(ROOT/'python'),
                str(ROOT/'tools/performance_harness')]
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
    from tnca.triton_backend import libdevice as tnca_libdevice

    @triton.jit
    def k_sin(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tl.sin(tl.load(X + i)))

    @triton.jit
    def k_cos(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tl.cos(tl.load(X + i)))

    @triton.jit
    def k_sin_cos_tile(X, O):
        # A 2-D tile (Inductor's form for a broadcast RoPE table): both
        # functions of the same values, written side by side.
        r = tl.arange(0, 4)
        c = tl.arange(0, 4)
        x = tl.load(X + r[:, None] * 4 + c[None, :])
        tl.store(O + r[:, None] * 8 + c[None, :], tl.cos(x))
        tl.store(O + r[:, None] * 8 + 4 + c[None, :], tl.sin(x))

    @triton.jit
    def k_log1p(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tnca_libdevice.log1p(tl.load(X + i)))

    @triton.jit
    def k_erf(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tnca_libdevice.erf(tl.load(X + i)))

    @triton.jit
    def k_i64_divmod(X, O):
        # Truncating int64 division and remainder by constants (a power of
        # two, others: a reciprocal multiply) and by a runtime divisor.
        i = tl.arange(0, 16)
        x = tl.load(X + i)
        y = tl.load(X + 16 + i)
        tl.store(O + i, x // 7)
        tl.store(O + 16 + i, x % 640)
        tl.store(O + 32 + i, x // 32)
        tl.store(O + 48 + i, x % -3)
        tl.store(O + 64 + i, x // y)
        tl.store(O + 80 + i, x % y)

    @triton.jit
    def k_i64_to_f32(X, O):
        i = tl.arange(0, 16)
        tl.store(O + i, tl.load(X + i).to(tl.float32))


def f32(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xffffffff))[0]


def bits32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


# Sixteen-lane input sets (binary32 patterns).
TRIG_SETS = {
    'special': [0x00000000, 0x80000000, 0x0d000000, 0x00000001, 0x3f000000, 0xbf490fdb,   # 0, -0, tiny, subnormal, 0.5, -pi/4
                0x3fc90fdb, 0x40490fdb, 0x40c90fdb, 0x47ce4780, 0x49742400, 0x5d8f2f15,   # pi/2, pi, 2pi, 105615, 1e6, near-multiple
                0x7f7fffff, 0xff7fffff, 0x7f800000, 0x7fc00000],                          # max, -max, inf, nan
    'large': [0x4b800001, 0x4c000003, 0x4e800005, 0x50000007, 0x52800009, 0x5500000b,
              0x5780000d, 0x5a00000f, 0x5c800011, 0x5f000013, 0x61800015, 0x64000017,
              0x66800019, 0x6900001b, 0x7080001d, 0x7c00001f],
    'mid': [0x3f800000, 0xbf800000, 0x40000000, 0x40400000, 0x40800000, 0xc0a00000,
            0x41200000, 0x42c80000, 0x43000000, 0x447a0000, 0x461c4000, 0x47c35000,
            0x3f490fdb, 0x3f490fdc, 0x3c8efa35, 0x48800000],
}
LOG1P_SETS = {
    'special': [0x00000000, 0x80000000, 0x00000001, 0x2f800000, 0xb2d6bf95, 0x3727c5ac,   # 0, -0, subnormal, 2^-32, -2.5e-8, 1e-5
                0xbe95f61a, 0x3ed413cd, 0xbf000000, 0x3f400000, 0x3f800000, 0x41a00000,   # -0.293, 0.414, -0.5, 0.75, 1, 20
                0x4ce73d64, 0x7f7fffff, 0xbf800000, 0xc0000000],                          # 1.2e8, max, -1, -2
    'edges': [0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0xbf7fffff, 0x3f3504f3,
              0xbe95f619, 0xbe95f61b, 0x3ed413cc, 0x3ed413ce, 0x4b800000, 0x4c000001,
              0x33800000, 0xb3800000, 0x3e800000, 0xbe800000],
}
ERF_SETS = {
    # 0, -0, subnormal, tiny, the split at 1 and its neighbours, the range
    # of 1 - exp(Q), where erf rounds to 1, beyond it, inf, NaN
    'special': [0x00000000, 0x80000000, 0x000116c2, 0x1e3ce508, 0xbe800000, 0x3f000000, 0x3f7ffffe, 0x3f800000, 0xbf800000, 0x3fc00000, 0x40000000, 0xc0200000, 0x40400000, 0x4079999a, 0x407b0000, 0xc1200000],
    'edges': [0x7f800000, 0xff800000, 0x7fc00000, 0x3f7fffff, 0x3f800001, 0x3f6c6425, 0x3f9090f1,
              0x40750000, 0x407affff, 0x407b0000, 0xc07b0001, 0x3e800000, 0x80000001, 0x7f7fffff,
              0x3f3504f3, 0xbf3504f3],
}
I64_SET = [0, 1, -1, (1 << 24) + 1, (1 << 24) + 3, -((1 << 25) + 1), (1 << 53) + 1,
           -(1 << 63), (1 << 63) - 1, 123456789012345, -987654321098765, (1 << 32) - 1,
           1 << 32, (1 << 40) + (1 << 16), (1 << 62) + (1 << 38), -((1 << 31) + 1)]


DIV_X = [0, 1, -1, 6, 7, -7, 1 << 40, -(1 << 40) - 5, (1 << 63) - 1, -(1 << 63), 123456789012345,
         -987654321098765, (1 << 62) + 3, 639, -641, 31]
DIV_Y = [1, -1, 3, 7, -7, 1 << 33, 5, -9, 1, -1, 1 << 62, (1 << 63) - 1, -(1 << 63), 2, 640, 32]


def c_div(a, b):
    """C (truncating) division and remainder of Python ints, wrapped to int64."""
    q = abs(a) // abs(b)
    q = q if (a < 0) == (b < 0) else -q
    r = a - b * q
    wrap = lambda v: ((v + (1 << 63)) % (1 << 64)) - (1 << 63)
    return wrap(q), wrap(r)


def emulate(fn_name, inputs):
    import numpy as np
    from transcendental_numeric import evaluate
    return [int(v) for v in evaluate(fn_name, np.array(inputs, dtype=np.uint32))]


def within_one_ulp(fn, x_bits, y_bits):
    x, y = f32(x_bits), f32(y_bits)
    ref = fn(x)
    if math.isnan(ref):
        return math.isnan(y)
    if math.isinf(ref):
        return y == ref
    a = abs(ref)
    e = max(math.frexp(a)[1] - 1, -126) if a else -126
    return abs(y - ref) <= 2.0 ** (e - 23)


def safe(fn):
    def g(x):
        try:
            return fn(x)
        except (ValueError, OverflowError):
            if fn is math.log1p:
                return -math.inf if x == -1 else math.nan
            return math.nan
    return g


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class StdMathTests(unittest.TestCase):
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
        ok, out = H.run(p, max_cycles=400000)
        self.assertTrue(ok, out)

    def test_sin_cos(self):
        for name, kern, ref in (('sin', k_sin, math.sin), ('cos', k_cos, math.cos)):
            cg, asm = self.compile(kern, dict(X='*fp32', O='*fp32'))
            self.assertIn('v.mulh.u', asm)
            self.assertIn('v.clz', asm)
            self.assertNotIn('approx', asm)              # no SFU
            for label, xs in TRIG_SETS.items():
                with self.subTest(fn=name, inputs=label):
                    expect = emulate(name, xs)
                    for x, y in zip(xs, expect):
                        self.assertTrue(within_one_ulp(safe(ref), x, y), (name, hex(x), hex(y)))
                    self.run_rtl(f'{name}_{label}', cg, H.pack32(xs), expect)

    def test_sin_cos_tile(self):
        cg, _ = self.compile(k_sin_cos_tile, dict(X='*fp32', O='*fp32'))
        xs = TRIG_SETS['special']
        c, s = emulate('cos', xs), emulate('sin', xs)
        expect = []
        for r in range(4):
            expect += c[4 * r:4 * r + 4] + s[4 * r:4 * r + 4]
        self.run_rtl('sin_cos_tile', cg, H.pack32(xs), expect)

    def test_log1p(self):
        cg, asm = self.compile(k_log1p, dict(X='*fp32', O='*fp32'))
        self.assertNotIn('approx', asm)
        for label, xs in LOG1P_SETS.items():
            with self.subTest(inputs=label):
                expect = emulate('log1p', xs)
                for x, y in zip(xs, expect):
                    self.assertTrue(within_one_ulp(safe(math.log1p), x, y), (hex(x), hex(y)))
                self.run_rtl(f'log1p_{label}', cg, H.pack32(xs), expect)

    def test_erf(self):
        cg, asm = self.compile(k_erf, dict(X='*fp32', O='*fp32'))
        self.assertNotIn('approx', asm)                  # no SFU
        for label, xs in ERF_SETS.items():
            with self.subTest(inputs=label):
                expect = emulate('erf', xs)
                for x, y in zip(xs, expect):
                    self.assertTrue(within_one_ulp(safe(math.erf), x, y), (hex(x), hex(y)))
                self.run_rtl(f'erf_{label}', cg, H.pack32(xs), expect)

    def test_int64_divmod(self):
        cg, asm = self.compile(k_i64_divmod, dict(X='*i64', O='*i64'))
        self.assertIn('s.mulh.u', asm)
        expect = []
        for k, d in ((0, 7), (1, 640), (0, 32), (1, -3)):
            expect += [c_div(x, d)[k] for x in DIV_X]
        expect += [c_div(x, y)[0] for x, y in zip(DIV_X, DIV_Y)]
        expect += [c_div(x, y)[1] for x, y in zip(DIV_X, DIV_Y)]
        words = []
        for v in expect:
            v &= 0xFFFFFFFFFFFFFFFF
            words += [v & 0xFFFFFFFF, v >> 32]
        data = []
        for v in DIV_X + DIV_Y:
            v &= 0xFFFFFFFFFFFFFFFF
            data += [v & 0xFFFFFFFF, v >> 32]
        self.run_rtl('i64_divmod', cg, H.pack32(data), words)

    def test_int64_to_f32(self):
        import numpy as np
        cg, asm = self.compile(k_i64_to_f32, dict(X='*i64', O='*fp32'))
        self.assertIn('v.clz', asm)
        vals = np.array(I64_SET, dtype=np.int64)
        expect = [int(w) for w in vals.astype(np.float32).view(np.uint32)]
        raw = [v & 0xFFFFFFFFFFFFFFFF for v in I64_SET]
        data = []
        for v in raw:
            data += [v & 0xFFFFFFFF, v >> 32]
        self.run_rtl('i64_to_f32', cg, H.pack32(data), expect)


if __name__ == '__main__':
    unittest.main(verbosity=2)
