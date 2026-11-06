"""`tl.dot` lowering onto m.mma.dp.async (ISA sec 9C.4), gated by CAPABILITY0.MATRIX_DP.

The choice of opcode depends on the capability and nothing else: every matrix
command a dot produces becomes m.mma.dp.async, in the operand format(s) the dot
has, with the same K step, windows, footprints and barriers. A target without
the capability is lowered exactly as before. Checked here at the assembly
level (the arithmetic is checked on the SystemC/TLM model by
tools/performance_harness/fp8_cases.py and fp8_kernels_cases.py, against the
exact oracle tools/model/fdp.py).
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test'), str(ROOT/'tools/asm'),
                str(ROOT/'tools/isa')]
import triton_bridge
from tnca_cc import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.footprints import mma_dp_group
from tnca_cc.frontend import TTIRImporter

A, B, O, SCRATCH = 0x10000, 0x11000, 0x12000, 0x14000
BASE, DP = 'tnca-sim-vrf-v5-x8-g2c4', 'tnca-sim-vrf-v5-x8-g2c4-dp'

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
    def dot_acc(A, B, O, K: tl.constexpr):
        # acc += tl.dot(a, b) over a K loop: the form Inductor's templates use.
        m = tl.arange(0, 16)
        k = tl.arange(0, 16)
        n = tl.arange(0, 16)
        acc = tl.zeros((16, 16), dtype=tl.float32)
        for k0 in range(0, K, 16):
            a = tl.load(A + m[:, None] * K + k0 + k[None, :])
            b = tl.load(B + (k0 + k[:, None]) * 16 + n[None, :])
            acc += tl.dot(a, b)
        tl.store(O + m[:, None] * 16 + n[None, :], acc)


def compile_text(target, fn, sig, K):
    module, _ctx = triton_bridge.make_ttir(fn, sig, {'K': K})
    imp = TTIRImporter(target, fn.__name__, scratch_addr=SCRATCH, index_opt=True)
    kernel, _ = imp.run(module, fn.__name__, dict(A=A, B=B, O=O),
                        param_names=triton_bridge.param_names(fn))
    cg = CodeGen(kernel, target, check_protocol=False)
    cg.assign_shared()
    cg.run()
    return cg.b.text()


def both(base, dp, fn, sig, K):
    """(plain, packed) assembly. A shape the baseline cannot compile (a resource
    limit, e.g. a K64 FP16 fragment pair) must fail identically with the
    capability, never differently: then both are None."""
    try:
        plain = compile_text(base, fn, sig, K)
    except Exception as e:                         # noqa: BLE001
        try:
            compile_text(dp, fn, sig, K)
        except Exception as e2:                    # noqa: BLE001
            assert type(e) is type(e2) and str(e) == str(e2), (e, e2)
            return None, None
        raise AssertionError('compiled with MATRIX_DP but not without')
    return plain, compile_text(dp, fn, sig, K)


FORMATS = (('e4m3', 'fp8e4nv'), ('e5m2', 'fp8e5'), ('f16', 'fp16'), ('bf16', 'bf16'))


class GroupSize(unittest.TestCase):
    def test_group_is_a_function_of_the_two_formats(self):
        for fa in ('f16', 'bf16', 'e5m2', 'e4m3'):
            for fb in ('f16', 'bf16', 'e5m2', 'e4m3'):
                self.assertEqual(mma_dp_group(fa, fb), 4 if (fa, fb) == ('e4m3', 'e4m3') else 2, (fa, fb))


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class DotLowering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = TargetInfo.load(BASE)
        cls.dp = TargetInfo.load(DP)

    def test_capability_is_only_on_the_dp_target(self):
        self.assertFalse(self.base.has('matrix_dp'))
        self.assertTrue(self.dp.has('matrix_dp'))
        self.assertFalse(TargetInfo.default().has('matrix_dp'))

    def test_every_format_and_k_step(self):
        for fmt, ty in FORMATS:
            for K in (16, 32, 64):
                with self.subTest(fmt=fmt, K=K):
                    sig = dict(A=f'*{ty}', B=f'*{ty}', O='*fp32')
                    plain, packed = both(self.base, self.dp, dot_rows, sig, K)
                    if plain is None:
                        continue
                    self.assertNotIn('m.mma.dp', plain)
                    self.assertEqual(packed.count(f'm.mma.dp.async.k{K}.{fmt}'), 1, packed)
                    self.assertNotIn('m.mma.async', packed)
                    # nothing but the mnemonic differs: windows, footprints,
                    # barriers, scheduling are those of m.mma.async
                    self.assertEqual(packed.replace('m.mma.dp.async', 'm.mma.async'), plain)

    def test_k_loop_accumulates_in_place(self):
        for fmt, ty in FORMATS:
            with self.subTest(fmt=fmt):
                sig = dict(A=f'*{ty}', B=f'*{ty}', O='*fp32')
                plain, packed = both(self.base, self.dp, dot_acc, sig, 64)
                self.assertIsNotNone(plain)
                self.assertGreaterEqual(packed.count('m.mma.dp.async'), 1)
                self.assertEqual(packed.replace('m.mma.dp.async', 'm.mma.async'), plain)

    def test_mixed_fp8_pair(self):
        # E4M3 x E5M2: the dot is an FP16 matrix command of the exactly
        # converted operands (a group of two either way), and the lowering
        # onto m.mma.dp.async changes only the mnemonic.
        sig = dict(A='*fp8e4nv', B='*fp8e5', O='*fp32')
        plain, packed = both(self.base, self.dp, dot_rows, sig, 16)
        self.assertIsNotNone(plain)
        self.assertEqual(packed.replace('m.mma.dp.async', 'm.mma.async'), plain)

    def test_emitted_text_assembles(self):
        import tnca_asm
        for fmt, ty in FORMATS:
            text = compile_text(self.dp, dot_rows, dict(A=f'*{ty}', B=f'*{ty}', O='*fp32'), 16)
            asm = tnca_asm.Assembler()
            asm.assemble(text)


if __name__ == '__main__':
    unittest.main()
