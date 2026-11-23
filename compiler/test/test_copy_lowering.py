"""R42-copy lowerings, checked on an eager builder (no Triton, no RTL).

`Eager` runs the importer's vector operations on Python lists, one list of VLEN
lanes a register, so a method of the importer that only talks to `self.b` can
be called on a stand-in object and its result compared with the definition:

  * the register transpose (TTIRImporter._transpose_registers),
  * the log-step inclusive scan (TTIRImporter._scan_steps),
  * the 64-bit value pair of `invariant + sext(lane)` (wide._form_pair).
"""
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler/python')]
from tnca_cc import wide
from tnca_cc.frontend import TTIRImporter

VLEN = 16
M32 = 0xFFFFFFFF


class V(list):
    """A vector register: VLEN lanes, remembering a dtype for the builder's signature."""
    dtype = 'i32'


class Eager:
    """The Builder methods the lowerings above use, evaluated at once."""
    def arange(self):
        return V(range(VLEN))

    def splat(self, s, dtype=None):
        return V([s & M32 if isinstance(s, int) else s] * VLEN)

    def vmovi(self, k, dtype=None):
        return V([k & M32] * VLEN)

    def const(self, k):
        return k

    def sbini(self, op, a, k):
        return {'add': a + k, 'lshr': (a & ((1 << 64) - 1)) >> k, 'mul': a * k}[op]

    def vbin(self, op, a, b, dtype=None, fmm=None):
        f = {'add': lambda x, y: (x + y) & M32, 'and': lambda x, y: x & y,
             'ashr': lambda x, y: ((x - (1 << 32) if x >> 31 else x) >> y) & M32}[op]
        return V([f(x, y) for x, y in zip(a, b)])

    def vcmp(self, cc, a, b):
        f = {'ne': lambda x, y: x != y, 'ltu': lambda x, y: x < y, 'ge': lambda x, y: x >= y}[cc]
        return [f(x, y) for x, y in zip(a, b)]            # a predicate: a list of bools

    def vselect(self, p, a, b):
        return V([x if m else y for m, x, y in zip(p, a, b)])

    def vshuffle(self, kind, v, amount, dtype=None):
        if kind == 'xor':
            return V([v[i ^ (amount & 15)] for i in range(VLEN)])
        if kind == 'up':
            return V([v[i - amount] if i >= amount else v[i] for i in range(VLEN)])
        raise AssertionError(kind)


def stand_in():
    ns = types.SimpleNamespace(b=Eager())
    ns._const_scalar = lambda k: k
    return ns


class RegisterTranspose(unittest.TestCase):
    def test_transpose(self):
        vs = [V(i * VLEN + j for j in range(VLEN)) for i in range(VLEN)]
        out = TTIRImporter._transpose_registers(stand_in(), [V(v) for v in vs])
        for i in range(VLEN):
            for j in range(VLEN):
                self.assertEqual(out[i][j], vs[j][i], (i, j))


class LogStepScan(unittest.TestCase):
    def test_inclusive_sum(self):
        ns = stand_in()
        x = V([(7 * i + 3) % 11 for i in range(VLEN)])
        ns._one_row = lambda value: (x, VLEN)
        ns._eval_region = lambda op, args: [V([a + b for a, b in zip(args[0], args[1])])]
        ns._vec_of = lambda value, like: value
        out = {}
        ns._v = out
        ns._tiles = {}
        result = types.SimpleNamespace(get_type=lambda: 'tensor<16xi32>', id=lambda: 'r')
        op = types.SimpleNamespace(get_num_operands=lambda: 1, get_num_results=lambda: 1, get_operand=lambda i: None,
                                   get_result=lambda i: result)
        TTIRImporter._scan_steps(ns, op)
        want, run = [], 0
        for v in x:
            run += v
            want.append(run)
        self.assertEqual(list(out['r']), want)

    def test_order_of_a_non_commutative_combiner(self):
        # first-operand-wins: the scan of a sequence is its first element everywhere
        ns = stand_in()
        x = V(range(100, 100 + VLEN))
        ns._one_row = lambda value: (x, VLEN)
        ns._eval_region = lambda op, args: [V(list(args[0]))]      # combine(a, b) = a
        ns._vec_of = lambda value, like: value
        ns._v, ns._tiles = {}, {}
        result = types.SimpleNamespace(get_type=lambda: 'tensor<16xi32>', id=lambda: 'r')
        op = types.SimpleNamespace(get_num_operands=lambda: 1, get_num_results=lambda: 1, get_operand=lambda i: None,
                                   get_result=lambda i: result)
        TTIRImporter._scan_steps(ns, op)
        self.assertEqual(list(ns._v['r']), [100] * VLEN)


class WidePair(unittest.TestCase):
    def check(self, inv, lane):
        ns = stand_in()
        ns._const_scalar = lambda k: k
        lane_value = V(lane)
        orig = wide._vector
        wide._vector = lambda imp, v, typ: lane_value
        try:
            lo, hi = wide._form_pair(ns, (inv, object() if lane is not None else None))
        finally:
            wide._vector = orig
        for i in range(VLEN):
            sext = (lane[i] - (1 << 32) if lane[i] >> 31 else lane[i]) if lane is not None else 0
            want = (inv + sext) & ((1 << 64) - 1)
            self.assertEqual(lo[i] | (hi[i] << 32), want, (inv, lane, i))

    def test_pairs(self):
        lanes = [0, 1, 5, M32, 0x80000000, 0x7FFFFFFF, 123456, M32 - 1] * 2
        for inv in (0, 1, 64, 0xFFFFFFFF, 0x1_0000_0000, 0x1_FFFF_FFFF, -5 & ((1 << 64) - 1), 12345678901):
            self.check(inv, lanes)

    def test_uniform(self):
        ns = stand_in()
        lo, hi = wide._form_pair(ns, (0x1234_5678_9ABC_DEF0, None))
        self.assertEqual(lo[3], 0x9ABCDEF0)
        self.assertEqual(hi[3], 0x12345678)


if __name__ == '__main__':
    unittest.main()
