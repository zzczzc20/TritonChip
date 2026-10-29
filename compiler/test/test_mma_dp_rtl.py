"""`tl.dot` lowered onto m.mma.dp.async, run on the RTL core with the shared MAC lane.

The compiler's own output for `tl.dot` of E4M3, E5M2, FP16 and BF16 operands (target
`tnca-sim-dp`: MATRIX_DP) runs on `tb_tnca_core` elaborated with MAC_DP=1 (sim/build/
tb_core_dp14.vvp, built by `make core-mmadp`), and every output word is compared with the
exact oracle tools/model/fdp.py (Fractions, groups of 4 for E4M3 x E4M3 and 2 otherwise,
RO_48). Operands are random encodings over several binades, so the one rounding per group
differs from the one rounding per product. Domain: RTL simulation of one core.
"""
import os
from pathlib import Path
import random
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test'), str(ROOT/'tools/model')]
import rtl_harness as H
import triton_bridge
import fdp
from tnca_cc import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.frontend import TTIRImporter

A, B, O, SCRATCH = 0x10000, 0x11000, 0x12000, 0x14000
SENTINEL = 0xdeadbeef
FORMATS = (('e4m3', 'fp8e4nv', 8), ('e5m2', 'fp8e5', 8), ('f16', 'fp16', 16), ('bf16', 'bf16', 16))

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


def element(fmt, rng):
    """A finite encoding over a few binades around 1."""
    bias, ebits, fbits = {'e4m3': (7, 4, 3), 'e5m2': (15, 5, 2), 'f16': (15, 5, 10), 'bf16': (127, 8, 7)}[fmt]
    spread = 5 if fmt in ('e4m3', 'e5m2') else 9
    e = max(1, min((1 << ebits) - 2, bias + rng.randint(-spread, spread)))
    return (rng.getrandbits(1) << (ebits + fbits)) | (e << fbits) | rng.getrandbits(fbits)


def pack(values, width):
    per = 64 // width
    words = []
    for i in range(0, len(values), per):
        w = 0
        for j, v in enumerate(values[i:i + per]):
            w |= (v & ((1 << width) - 1)) << (width * j)
        words.append(w)
    return words


def expected(fmt, a, b, K):
    g = 4 if fmt == 'e4m3' else 2
    out = []
    for m in range(16):
        for n in range(16):
            acc = 0
            for k0 in range(0, K, g):
                s = fdp.group_sum(fmt, fmt, [a[m][k] for k in range(k0, k0 + g)], [b[k][n] for k in range(k0, k0 + g)])
                if s[0] == 'fin':
                    s = ('fin', fdp.ro(s[1]), s[2])
                acc = fdp.add_round(s, acc)
            out.append(acc)
    return out


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class DotOnRTL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not H.bench_available('dp'):
            raise unittest.SkipTest(f"{H.bench_path('dp')} not built; run `make core-mmadp`")
        cls.target = TargetInfo.load('tnca-sim-dp')

    def compile(self, ty, K):
        sig = dict(A=f'*{ty}', B=f'*{ty}', O='*fp32')
        module, _ctx = triton_bridge.make_ttir(dot_rows, sig, {'K': K})
        imp = TTIRImporter(self.target, 'dot_rows', scratch_addr=SCRATCH, index_opt=True)
        kernel, _ = imp.run(module, 'dot_rows', dict(A=A, B=B, O=O), param_names=triton_bridge.param_names(dot_rows))
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        return cg

    def test_every_format_and_k(self):
        ran, limited = [], []
        for fmt, ty, width in FORMATS:
            for K in (16, 32, 64):
                for seed in (1, 2):
                    with self.subTest(fmt=fmt, K=K, seed=seed):
                        rng = random.Random(1000 * seed + K)
                        a = [[element(fmt, rng) for _ in range(K)] for _ in range(16)]
                        b = [[element(fmt, rng) for _ in range(16)] for _ in range(K)]
                        try:
                            cg = self.compile(ty, K)
                        except Exception as e:                       # noqa: BLE001
                            # a fragment pair the target cannot hold is the baseline's limit too
                            from tnca_cc.diagnostics import ResourceError
                            self.assertIsInstance(e, ResourceError)
                            limited.append((fmt, K))
                            continue
                        self.assertIn(f'm.mma.dp.async.k{K}.{fmt}', cg.b.text())
                        want = expected(fmt, a, b, K)
                        p = H.Program(f'mmadp_{fmt}_k{K}_s{seed}').kernel(cg).check_base(O)
                        for i, w in enumerate(want):
                            p.check_word(O, 4 * i, w)
                        p.words(A, pack([v for r in a for v in r], width), 'as')
                        p.words(B, pack([v for r in b for v in r], width), 'bs')
                        p.words(O, H.pack32([SENTINEL] * 256), 'os')
                        ok, out = H.run(p, bench='dp', max_cycles=400000)
                        self.assertTrue(ok, out[-600:])
                        ran.append((fmt, K, seed))
        print(f"\nran on the RTL core: {len(ran)} programs {sorted(set((f, k) for f, k, _ in ran))}; "
              f"not compilable on this target (resource limit): {sorted(set(limited))}")
        self.assertTrue(ran)


if __name__ == '__main__':
    unittest.main()
