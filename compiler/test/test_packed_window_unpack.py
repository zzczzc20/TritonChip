"""Packed DMA snapshots can be read with the existing lane ISA."""
import unittest
from tnca_cc.frontend import TTIRImporter, Unsupported
from tnca_cc.tiles import Tile
from tnca_cc.target import TargetInfo
from tnca_cc.codegen import CodeGen


class PackedUnpackTests(unittest.TestCase):
    def test_even_and_odd_rows_read_both_halves_of_each_lane(self):
        for dtype in ('f16', 'bf16'):
            imp = TTIRImporter(TargetInfo.default())
            imp._packed_window_unpack = True
            tile = Tile('window', imp.b.window(2), 4, 16, dtype)
            get = imp._tile_row_getter(tile)
            for row in range(4):
                start = len(imp.kernel.role(0).region.ops)
                value = get(row)
                ops = imp.kernel.role(0).region.ops[start:]
                aliases = [op for op in ops if op.kind == 'window_reg']
                self.assertEqual([op.attrs['offset'] for op in aliases], [row // 2])
                reads = [op.attrs['lane'] for op in ops if op.kind == 'vextract']
                self.assertEqual(reads, list(range((row % 2) * 8, (row % 2 + 1) * 8)))
                self.assertEqual([op.attrs['lane'] for op in ops if op.kind == 'vinsert'], list(range(16)))
                self.assertFalse(any(op.kind in ('load', 'gather') for op in ops))
                imp.b.store(imp.b.const(0x7000 + row * 32), value, size='h')
            cg = CodeGen(imp.kernel, imp.target, check_protocol=False, window_reuse=True)
            cg.assign_shared()
            cg.run()
            text = cg.b.text()
            self.assertIn('s.lshri', text)
            self.assertEqual(imp._packed_unpack_rows, 4)

    def test_default_refuses_a_packed_window_without_external_descriptor(self):
        imp = TTIRImporter(TargetInfo.default())
        tile = Tile('window', imp.b.window(2), 4, 16, 'f16')
        with self.assertRaises(Unsupported):
            imp._tile_row_getter(tile)


if __name__ == '__main__':
    unittest.main()
