"""Column reductions keep independent output registers for wide chunks."""
import unittest
from tnca_cc.frontend import TTIRImporter, Unsupported
from tnca_cc.register_blocks import _reduce
from tnca_cc.target import TargetInfo
from tnca_cc.tiles import Tile


class Value:
    def __init__(self, key, typ):
        self.key, self.typ = key, typ

    def id(self):
        return self.key

    def get_type(self):
        return self.typ


class Reduction:
    def __init__(self, axis=0):
        self.axis = axis

    def get_num_operands(self):
        return 1

    def get_name(self):
        return 'tt.reduce'

    def get_num_results(self):
        return 1

    def get_operand(self, index):
        return Value('input', 'tensor<3x32xf32>')

    def get_result(self, index):
        return Value('output', 'tensor<32xf32>')

    def get_int_attr(self, name):
        return self.axis


class WideColumnReductionTests(unittest.TestCase):
    def test_each_column_chunk_combines_only_its_own_rows(self):
        for kind in ('add', 'max_prop', 'min_prop'):
            imp = TTIRImporter(TargetInfo.default())
            imp._combinator = lambda op: kind
            imp._dispatch = imp._op_tt_reduce
            rows = []
            for chunk in range(2):
                regs = tuple(imp.b.splat(imp.b.const(chunk * 10 + row), dtype='f32')
                             for row in range(3))
                rows.append(regs)
                imp._tiles[('register-chunk', 'input', chunk)] = Tile('rows', regs, 3, 16, 'f32')
            start = len(imp.kernel.role(0).region.ops)
            self.assertTrue(_reduce(imp, Reduction(), 32))
            self.assertNotIn('output', imp._v)
            self.assertNotEqual(imp._v[('register-chunk', 'output', 0)],
                                imp._v[('register-chunk', 'output', 1)])
            ops = imp.kernel.role(0).region.ops[start:]
            self.assertEqual(len(ops), 4)  # two row combinations per chunk
            for chunk in range(2):
                first, second = ops[chunk * 2:chunk * 2 + 2]
                self.assertEqual(first.operands, list(rows[chunk][:2]))
                self.assertEqual(second.operands, [first.results[0], rows[chunk][2]])
                if kind.endswith('_prop'):
                    self.assertEqual(first.attrs['fmm'], 'prop')
            self.assertFalse(imp._in_register_chunk)

    def test_invalid_axis_is_refused(self):
        imp = TTIRImporter(TargetInfo.default())
        with self.assertRaises(Unsupported):
            _reduce(imp, Reduction(axis=2), 32)


if __name__ == '__main__':
    unittest.main()
