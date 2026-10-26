"""Resource retile preserves immutable chunks and one ordered accumulator."""
import unittest
from tnca_cc.frontend import TTIRImporter, Unsupported
from tnca_cc.tiles import Tile
from tnca_cc.target import TargetInfo
from tnca_cc.codegen import CodeGen


class TestMMARetile(unittest.TestCase):
    def inputs(self, reuse=True):
        imp = TTIRImporter(TargetInfo.default(), scratch_addr=0x3000)
        imp._window_reuse = reuse
        imp._register_mma_fragment_cache = {}
        vectors = [imp.b.vmovi(bits) for bits in (0x3c00, 0x4000, 0x4200, 0x4400)]
        parts = tuple(Tile('rows', (v,) * 16, 16, 16, 'f16') for v in vectors)
        return imp, Tile('column-chunks', parts, 16, 64, 'f16'), \
            Tile('row-chunks', parts, 64, 16, 'f16')

    def test_k64_uses_two_ordered_k32_commands(self):
        imp, a, b = self.inputs()
        acc = imp.b.accumulator()
        imp.b.acc_zero(acc)
        pairs = []
        for left, right in imp._mma_fragments(a, b):
            pairs.append((left.cols, right.rows))
            imp._mma_into(acc, left, right)
        self.assertEqual(pairs, [(32, 32), (32, 32)])
        self.assertEqual(imp._mma_retiles, {(64, 32)})
        commands = [op for op in imp.kernel.role(0).region.ops if op.kind == 'mma_async']
        self.assertEqual([op.attrs['k'] for op in commands], ['k32', 'k32'])
        self.assertTrue(all(op.operands[0] is acc for op in commands))
        cg = CodeGen(imp.kernel, imp.target, check_protocol=False, window_reuse=True)
        cg.assign_shared()
        cg.run()
        self.assertEqual(cg.window_registers, 32)

    def test_same_left_parent_reuses_both_slices(self):
        imp, a, b = self.inputs()
        first = list(imp._mma_fragments(a, b))
        other = Tile('row-chunks', b.handle, b.rows, b.cols, b.dtype)
        second = list(imp._mma_fragments(a, other))
        self.assertTrue(all(x[0] is y[0] for x, y in zip(first, second)))
        self.assertTrue(all(x[1] is not y[1] for x, y in zip(first, second)))

    def test_packed_source_is_refused_instead_of_reread(self):
        imp, a, b = self.inputs()
        packed = Tile('window', imp.b.window(32), 16, 64, 'f16')
        with self.assertRaisesRegex(Unsupported, 'immutable row/column chunks'):
            list(imp._mma_fragments(packed, b))

    def test_disabled_keeps_the_original_k64_fragment(self):
        imp, a, b = self.inputs(reuse=False)
        pairs = list(imp._mma_fragments(a, b))
        self.assertEqual([(x.cols, y.rows) for x, y in pairs], [(64, 64)])
        self.assertFalse(getattr(imp, '_mma_retiles', None))


if __name__ == '__main__':
    unittest.main()
