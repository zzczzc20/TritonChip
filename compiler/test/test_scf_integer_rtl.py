"""Integer conditional branches run only their selected arm on the RTL core."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler/python'), str(ROOT / 'compiler/test')]
import rtl_harness as H
import triton_bridge
from tnca_cc import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.frontend import TTIRImporter

if triton_bridge.available():
    import triton
    import triton.language as tl

    @triton.jit
    def branch_remainder(O, flag, divisor):
        lane = tl.arange(0, 16)
        if flag > 0:
            value = (lane - 8) % divisor
        else:
            value = lane + 37
        tl.store(O + lane, value)

    @triton.jit
    def branch_remainder_else(O, flag, divisor):
        lane = tl.arange(0, 16)
        if flag > 0:
            value = lane + 37
        else:
            value = (lane - 8) % divisor
        tl.store(O + lane, value)

    @triton.jit
    def branch_scalar_divide(O, flag, numerator, divisor):
        if flag > 0:
            value = numerator // divisor
            other = numerator + 3
        else:
            value = numerator + 11
            other = numerator + 7
        tl.store(O, value)
        tl.store(O + 1, other)


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class IntegerIfRTL(unittest.TestCase):
    def test_selected_arm(self):
        if not H.bench_available('dp'):
            self.skipTest('run make core-mmadp first')
        target = TargetInfo.load('tnca-sim-dp')
        out = 0x12000
        cases = ((branch_remainder, 'branch_remainder', 1, 5, True),
                 (branch_remainder, 'branch_remainder', 0, 0, False),
                 (branch_remainder, 'branch_remainder', 0, 5, False),
                 (branch_remainder_else, 'branch_remainder_else', 1, 0, False),
                 (branch_remainder_else, 'branch_remainder_else', 0, 5, True))
        for fn, name, flag, divisor, divides in cases:
            with self.subTest(name=name, flag=flag, divisor=divisor):
                module, ctx = triton_bridge.make_ttir(
                    fn, {'O': '*i32', 'flag': 'i32', 'divisor': 'i32'}, {})
                importer = TTIRImporter(target, name, index_opt=True)
                kernel, _ = importer.run(module, name,
                                         {'O': out, 'flag': flag, 'divisor': divisor},
                                         param_names=triton_bridge.param_names(fn))
                cg = CodeGen(kernel, target, check_protocol=False)
                cg.assign_shared()
                cg.run()
                p = H.Program(f'{name}_{flag}_{divisor}').kernel(cg).check_base(out)
                for lane in range(16):
                    # arith.remsi truncates toward zero.
                    n = lane - 8
                    expected = (n - int(n / divisor) * divisor) if divides else lane + 37
                    p.check_word(out, 4 * lane, expected & 0xffffffff)
                p.words(out, H.pack32([0xdeadbeef] * 16), 'out')
                # A non-selected restoring division takes >600 cycles; this
                # limit checks it was skipped, including the zero divisor.
                ok, log = H.run(p, bench='dp', max_cycles=400000 if divides else 600)
                self.assertTrue(ok, log[-1200:])

    def test_scalar_and_multiple_results(self):
        if not H.bench_available('dp'):
            self.skipTest('run make core-mmadp first')
        target, out = TargetInfo.load('tnca-sim-dp'), 0x12000
        signature = {'O': '*i32', 'flag': 'i32', 'numerator': 'i32', 'divisor': 'i32'}
        for flag, divisor in ((1, 5), (0, 0)):
            with self.subTest(flag=flag, divisor=divisor):
                module, ctx = triton_bridge.make_ttir(branch_scalar_divide, signature, {})
                imp = TTIRImporter(target, 'branch_scalar_divide', index_opt=True)
                kernel, _ = imp.run(module, 'branch_scalar_divide',
                                    {'O': out, 'flag': flag, 'numerator': 37, 'divisor': divisor},
                                    param_names=triton_bridge.param_names(branch_scalar_divide))
                cg = CodeGen(kernel, target, check_protocol=False)
                cg.assign_shared()
                cg.run()
                p = H.Program(f'if_scalar_{flag}').kernel(cg).check_base(out)
                p.check_word(out, 0, 7 if flag else 48)
                p.check_word(out, 4, 40 if flag else 44)
                p.words(out, H.pack32([0xdeadbeef, 0xdeadbeef]), 'out')
                ok, log = H.run(p, bench='dp', max_cycles=400000 if flag else 600)
                self.assertTrue(ok, log[-1200:])


if __name__ == '__main__':
    unittest.main()
