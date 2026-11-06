"""Addresses past 2**31 elements and pointers formed after the prologue (R41 S2-04 D).

Inductor indexes a kernel in 64 bits exactly when a numel or a storage
exceeds 2**31 - 1 elements (`tl.program_id(0).to(tl.int64)`): a uint8 or FP8
storage over 2 GiB, a bf16 one over 4 GiB. Such a kernel's offsets do not
fit the 32-bit gather index, so the importer must not narrow them:

  - `xoffset + arange` plus a constant (a slice past 2 GiB) joins the scalar
    address and stays one contiguous vector access, with no narrowing trap;
  - per-lane 64-bit offsets (a gather through int64 indices) form each
    lane's address in 64-bit scalar registers;
  - a 2-D tile of 64-bit offsets is rebased on its first element;
  - `//` and `%` by a constant are exact on 64-bit values.

A pointer broadcast of an [R, 1] tile of offsets gives every lane its row's
offset (it kept it in lane 0, so columns past 0 read row 0's address).

And a GEMM operand pointer `B + k` (Inductor's mm template on a weight slice)
builds its hoisted TensorMap from the argument plus k at the kernel's entry;
it used to name the in-loop pointer there and fail to assemble (`sNone`).

The large-offset forms are checked in the emitted code here; the same code
runs on the RTL with offsets that fit its memory, and on the SystemC model
with real offsets past 2 and 4 GiB (tools/performance_harness/addr64_case.py).
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
from tnca_cc.wide import NARROW_TRAP_CODE

X, Y, O, P, SCRATCH = 0x10000, 0x12000, 0x14000, 0x16000, 0x18000

if triton_bridge.available():
    import triton
    import triton.language as tl

    @triton.jit
    def sliced_operand_matmul(A, B, O):
        # The template's `B = B + offset` for a weight that is a slice of a
        # larger storage, then K steps through it.
        B = B + 64
        rm = tl.arange(0, 16)
        rn = tl.arange(0, 16)
        rk = tl.arange(0, 16)
        acc = tl.zeros((16, 16), dtype=tl.float32)
        for k in range(0, 2):
            a = tl.load(A + rm[:, None] * 32 + (k * 16 + rk)[None, :])
            b = tl.load(B + (k * 16 + rk)[:, None] * 16 + rn[None, :])
            acc += tl.dot(a, b)
        tl.store(O + rm[:, None] * 16 + rn[None, :], acc)

    @triton.jit
    def index64_slice(X, O, K: tl.constexpr):
        x = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
        tl.store(O + x, tl.load(X + (x + K)) * 2.0)

    @triton.jit
    def index64_gather(X, I, O):
        x = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
        i = tl.load(I + x)
        tl.store(O + x, tl.load(X + i * 3 + 1, mask=x < 13, other=-1.0), mask=x < 15)

    @triton.jit
    def index64_tile(X, O):
        r = tl.program_id(0).to(tl.int64) * 8 + tl.arange(0, 8)[:, None].to(tl.int64)
        c = tl.arange(0, 8)[None, :].to(tl.int64)
        tl.store(O + c * 8 + r, tl.load(X + r * 8 + c))

    @triton.jit
    def gathered_rows(X, I, O):
        # A 32-bit kernel gathering rows by int64 ids: the [8, 1] pointer
        # carries 64-bit (narrowed) offsets and is broadcast across the row.
        r = tl.arange(0, 8)
        rows = X + tl.load(I + r)[:, None] * 8
        tl.store(O + r[:, None] * 8 + tl.arange(0, 8)[None, :], tl.load(rows + tl.arange(0, 8)[None, :]))

    @triton.jit
    def index64_divmod(I, Q, R, D: tl.constexpr):
        x = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
        n = tl.load(I + x)
        tl.store(Q + x, n // D)
        tl.store(R + x, n % D)

    @triton.jit
    def scalar64_divmod(Q, R, N, D: tl.constexpr):
        x = tl.arange(0, 1)
        tl.store(Q + x, tl.full((1,), N // D, tl.int64))
        tl.store(R + x, tl.full((1,), N % D, tl.int64))

    @triton.jit
    def small_descriptor_box(X, O, ROW, STRIDE: tl.constexpr):
        src = tl.make_tensor_descriptor(X, shape=[1 << 30, 4], strides=[STRIDE, 1],
                                        block_shape=[2, 4])
        dst = tl.make_tensor_descriptor(O, shape=[1 << 30, 4], strides=[STRIDE, 1],
                                        block_shape=[2, 4])
        dst.store([5, 0], src.load([ROW, 2]) + 1.0)

    @triton.jit
    def scalar64_runtime_divmod(Q, R, N, D):
        x = tl.arange(0, 1)
        tl.store(Q + x, tl.full((1,), N // D, tl.int64))
        tl.store(R + x, tl.full((1,), N % D, tl.int64))


def f2b(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def h2b(v):
    return struct.unpack('<H', struct.pack('<e', v))[0]


def trunc_divmod(n, d):
    q = abs(n) // abs(d) * (1 if (n < 0) == (d < 0) else -1)
    return q, n - q * d


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class Addr64Tests(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()

    def compile(self, fn, sig, args, constexprs=None, index_opt=True):
        module, _ctx = triton_bridge.make_ttir(fn, sig, constexprs or {})
        imp = TTIRImporter(self.target, fn.__name__, scratch_addr=SCRATCH, index_opt=index_opt)
        kernel, _ = imp.run(module, fn.__name__, args, param_names=triton_bridge.param_names(fn))
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return imp, cg, cg.b.text()

    def run_rtl(self, name, cg, data, checks):
        if not H.available():
            self.skipTest(H.why_unavailable())
        p = H.Program(name).kernel(cg)
        for base, off, word in checks:
            p.check_base(base).check_word(base, off, word)
        for addr, words, label in data:
            p.words(addr, words, label)
        ok, out = H.run(p)
        self.assertTrue(ok, out[-3000:])

    def test_sliced_gemm_operand_hoists_from_its_argument(self):
        _imp, cg, text = self.compile(sliced_operand_matmul, dict(A='*fp16', B='*fp16', O='*fp32'),
                                      dict(A=X, B=Y, O=O))
        self.assertNotIn('sNone', text)
        # The hoisted map's base: B's address plus the slice's 128 bytes.
        self.assertIn("the pointer's constant offset (128 bytes)", text)
        a = [((r * 3 + c) % 7 - 3) * 0.5 for r in range(16) for c in range(32)]
        bmat = [((k * 5 + n) % 9 - 4) * 0.25 for k in range(32) for n in range(16)]
        bbuf = [9.0] * 64 + bmat
        pack = lambda xs: H.pack16([h2b(x) for x in xs])  # noqa: E731
        checks = []
        for r in range(16):
            for n in range(16):
                ref = sum(a[r * 32 + k] * bmat[k * 16 + n] for k in range(32))
                checks.append((O, (r * 16 + n) * 4, f2b(ref)))
        self.run_rtl('addr64_sliced_gemm', cg, [(X, pack(a), 'as'), (Y, pack(bbuf), 'bs'),
                                                (O, H.pack32([0] * 256), 'os')], checks)

    def test_index64_slice_past_2g_is_one_contiguous_access(self):
        K = 3 << 30                       # 12 GiB of f32 past X: no 32-bit index holds it
        _imp, _cg, text = self.compile(index64_slice, dict(X='*fp32', O='*fp32'), dict(X=X, O=O),
                                       constexprs={'K': K})
        self.assertNotIn(f'trap         #{NARROW_TRAP_CODE}', text)
        self.assertNotIn('gather', text)
        # xoffset + K in 64-bit scalar registers, scaled to bytes there, and
        # one contiguous load at that address.
        self.assertIn(f'.li       s', text)
        self.assertIn(f'#{K}\n', text)
        self.assertRegex(text, r's\.shli\s+s\d+, s\d+, #2\s+// element offset -> bytes')
        self.assertRegex(text, r'v\.ld\.w\s+v\d+, \[s\d+, #0\]')
        # The same lowering with an offset the RTL's memory holds.
        _imp, cg, _ = self.compile(index64_slice, dict(X='*fp32', O='*fp32'), dict(X=X, O=O),
                                   constexprs={'K': 40})
        xs = [i * 0.75 - 9 for i in range(64)]
        self.run_rtl('addr64_slice', cg, [(X, H.pack32([f2b(v) for v in xs]), 'xs'),
                                          (O, H.pack32([0] * 16), 'os')],
                     [(O, 4 * i, f2b(xs[40 + i] * 2.0)) for i in range(16)])

    def test_index64_gather_forms_each_lane_address_in_64_bits(self):
        _imp, cg, text = self.compile(index64_gather, dict(X='*fp32', I='*i64', O='*fp32'),
                                      dict(X=X, I=Y, O=O))
        self.assertNotIn(f'trap         #{NARROW_TRAP_CODE}', text)
        self.assertEqual(text.count('a 64-bit address'), 16)
        idx = [5, 0, 17, 3, 9, 1, 20, 2, 7, 11, 4, 6, 8, 10, 12, 13]
        xs = [i * 1.5 - 7 for i in range(64)]
        words = []
        for i in idx:
            words += [i, 0]
        expect = [xs[3 * i + 1] if lane < 13 else -1.0 for lane, i in enumerate(idx[:15])]
        self.run_rtl('addr64_gather', cg, [(X, H.pack32([f2b(v) for v in xs]), 'xs'),
                                           (Y, H.pack32(words), 'is'),
                                           (O, H.pack32([f2b(123.0)] * 16), 'os')],
                     [(O, 4 * k, f2b(v)) for k, v in enumerate(expect)] + [(O, 60, f2b(123.0))])

    def test_index64_tile_is_rebased_on_its_first_element(self):
        _imp, cg, text = self.compile(index64_tile, dict(X='*fp32', O='*fp32'), dict(X=X, O=O))
        self.assertIn('a 64-bit tile origin', text)
        xs = [r * 10 + c + 0.5 for r in range(8) for c in range(8)]
        self.run_rtl('addr64_tile', cg, [(X, H.pack32([f2b(v) for v in xs]), 'xs'),
                                         (O, H.pack32([0] * 64), 'os')],
                     [(O, 4 * (c * 8 + r), f2b(xs[r * 8 + c])) for r in range(8) for c in range(8)])

    def test_affine64_tile_with_i64_elements(self):
        _, cg, _ = self.compile(index64_tile, dict(X='*i64', O='*i64'), dict(X=X, O=O))
        xs = [((-1) ** r) * ((1 << 40) + r * 10 + c) for r in range(8) for c in range(8)]
        words = [part for n in xs for part in (n & 0xffffffff, (n >> 32) & 0xffffffff)]
        checks = [(O, 8 * (c * 8 + r) + off, (xs[r * 8 + c] >> (off * 8)) & 0xffffffff)
                  for r in range(8) for c in range(8) for off in (0, 4)]
        self.run_rtl('addr64_affine_i64', cg,
            [(X, H.pack32(words), 'xs'), (O, H.pack32([0] * 128), 'os')], checks)

    def test_row_pointer_broadcast_gives_every_lane_its_row(self):
        # Broadcasting an [R, 1] pointer whose offsets are a tile (not an
        # affine index) used to keep the offset in lane 0 only: columns 1-7
        # of every row read row 0 of the table.
        _imp, cg, _text = self.compile(gathered_rows, dict(X='*fp32', I='*i64', O='*fp32'),
                                       dict(X=X, I=Y, O=O))
        ids = [3, 0, 7, 5, 1, 6, 2, 4]
        xs = [r * 10 + c + 0.25 for r in range(8) for c in range(8)]
        words = []
        for i in ids:
            words += [i, 0]
        self.run_rtl('addr64_row_broadcast', cg, [(X, H.pack32([f2b(v) for v in xs]), 'xs'),
                                                  (Y, H.pack32(words), 'is'),
                                                  (O, H.pack32([0] * 64), 'os')],
                     [(O, 4 * (r * 8 + c), f2b(xs[ids[r] * 8 + c])) for r in range(8) for c in range(8)])

    def test_index64_division_by_constants_is_exact(self):
        ns = [0, 1, 2047, 2048, 151935, 151936, (1 << 40) + 12345, (1 << 62) + 7, -1, -2048,
              -151937, -(1 << 40) - 3, (1 << 63) - 1, -(1 << 63) + 1, 98765432101, 3]
        words = []
        for n in ns:
            words += [n & 0xFFFFFFFF, (n >> 32) & 0xFFFFFFFF]
        for d in (2048, 151936, 7):
            with self.subTest(divisor=d):
                _imp, cg, text = self.compile(index64_divmod, dict(I='*i64', Q='*i64', R='*i64'),
                                              dict(I=X, Q=Y, R=O), constexprs={'D': d})
                if d & (d - 1):
                    self.assertIn('s.mulh.u', text)
                checks = []
                for k, n in enumerate(ns):
                    q, r = trunc_divmod(n, d)
                    for base, v in ((Y, q), (O, r)):
                        checks += [(base, 8 * k, v & 0xFFFFFFFF), (base, 8 * k + 4, (v >> 32) & 0xFFFFFFFF)]
                self.run_rtl(f'addr64_divmod_{d}', cg, [(X, H.pack32(words), 'ns'),
                                                        (Y, H.pack32([0] * 32), 'qs'),
                                                        (O, H.pack32([0] * 32), 'rs')], checks)

    def test_scalar64_grid_division(self):
        for n in ((1 << 40) + 12345, -(1 << 40) - 12345):
            for d in (16, 7):
                with self.subTest(n=n, divisor=d):
                    _, cg, _ = self.compile(scalar64_divmod,
                        dict(Q='*i64', R='*i64', N='i64'), dict(Q=Y, R=O, N=n),
                        constexprs={'D': d})
                    q, r = trunc_divmod(n, d)
                    checks = [(base, off, (v >> (off * 8)) & 0xffffffff)
                              for base, v in ((Y, q), (O, r)) for off in (0, 4)]
                    self.run_rtl(f'addr64_scalar_divmod_{n}_{d}', cg,
                        [(Y, H.pack32([0, 0]), 'qs'), (O, H.pack32([0, 0]), 'rs')], checks)

    def test_small_descriptor_box_keeps_absolute_offsets_in_64_bits(self):
        # Eight lanes can still name rows farther than 2**31 elements apart.
        sig = dict(X='*fp32', O='*fp32', ROW='i32')
        _, _, text = self.compile(small_descriptor_box, sig,
                                  dict(X=X, O=O, ROW=3), constexprs={'STRIDE': (1 << 31) + 8})
        self.assertNotIn('v.mul', text)
        xs = [float(i) for i in range(48)]
        for row in (3, -1):
            _, cg, _ = self.compile(small_descriptor_box, sig,
                                   dict(X=X, O=O, ROW=row), constexprs={'STRIDE': 8})
            checks = [(O, 4 * ((5 + r) * 8 + c), f2b(xs[(row + r) * 8 + 2 + c] + 1
                       if c < 2 and row + r >= 0 else 1.0)) for r in range(2) for c in range(4)]
            self.run_rtl(f'addr64_small_descriptor_{row}', cg,
                [(X, H.pack32([f2b(v) for v in xs]), 'xs'),
                 (O, H.pack32([0] * 64), 'os')], checks)

    def test_scalar64_runtime_grid_division(self):
        for unsigned, n, d in ((False, (1 << 40) + 12345, 7),
                               (False, -(1 << 40) - 12345, -7),
                               (True, (1 << 64) - 1, (1 << 63) + 9)):
            typ = 'u64' if unsigned else 'i64'
            _, cg, _ = self.compile(scalar64_runtime_divmod,
                dict(Q='*i64', R='*i64', N=typ, D=typ), dict(Q=Y, R=O, N=n, D=d))
            q, r = divmod(n, d) if unsigned else trunc_divmod(n, d)
            checks = [(base, off, (v >> (off * 8)) & 0xffffffff)
                      for base, v in ((Y, q), (O, r)) for off in (0, 4)]
            self.run_rtl(f'addr64_runtime_divmod_{typ}_{n}', cg,
                [(Y, H.pack32([0, 0]), 'qs'), (O, H.pack32([0, 0]), 'rs')], checks)


if __name__ == '__main__':
    unittest.main()
