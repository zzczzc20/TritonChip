"""Chunk projections preserve memory-effect order and DMA eligibility."""
import unittest

from tnca_cc.frontend import TTIRImporter
from tnca_cc.ir import DESC
from tnca_cc.register_blocks import _OpView
from tnca_cc.target import TargetInfo
from tnca_cc.tiles import Affine2D


class Value:
    def __init__(self, name, typ='tensor<16x32xf16>'):
        self.name, self.typ = name, typ

    def id(self):
        return self.name

    def get_type(self):
        return self.typ


class Op:
    def __init__(self, name, result, block=None, operands=()):
        self.name, self.result = name, result
        self.block = block or Value('block')
        self.operands = operands

    def get_name(self):
        return self.name

    def get_result(self, index):
        return self.result

    def get_operand(self, index):
        return self.operands[index]

    def get_num_operands(self):
        return len(self.operands)

    def get_block(self):
        return self.block


class ChunkEffectTests(unittest.TestCase):
    def importer(self):
        return TTIRImporter(TargetInfo.load('tnca-sim-vrf-v5'), index_opt=True)

    def deferral(self, intervening=(), *, extra_user=False, other_block=False):
        imp = self.importer()
        source = Op('arith.addf', Value('source'))
        user = Op('arith.truncf', Value('user'),
                  block=Value('other') if other_block else None)
        effects = [Op(name, Value('effect')) for name in intervening]
        imp._all_ops = [source, *effects, user]
        imp._op_pos = {id(op): i for i, op in enumerate(imp._all_ops)}
        imp._value_users = {'source': [(user, 0)] * (2 if extra_user else 1)}
        return imp._deferrable(_OpView(source, 1, 16))

    def test_pure_chunk_epilogue_can_be_deferred(self):
        self.assertTrue(self.deferral(('arith.addi',)))

    def test_chunk_load_or_expression_cannot_cross_effects(self):
        for effect in ('ttg.barrier', 'tt.descriptor_reduce', 'tt.store',
                       'tt.atomic_rmw', 'tt.dot', 'scf.if'):
            with self.subTest(effect=effect):
                self.assertFalse(self.deferral((effect,)))

    def test_multiple_users_and_different_blocks_prevent_deferral(self):
        self.assertFalse(self.deferral(extra_user=True))
        self.assertFalse(self.deferral(other_block=True))

    def transpose(self, *, extra_user=False, masked=False):
        imp = self.importer()
        source = Op('tt.load', Value('source'), operands=(Value('ptr'),)
                    + ((Value('mask'),) if masked else ()))
        dot = Op('tt.dot', Value('dot'))
        imp._value_users = {'source': [(dot, 1)]}
        if extra_user:
            imp._value_users['source'].append((Op('arith.addf', Value('add')), 0))
        desc = imp.b._value(DESC, hint='d')
        imp._hoisted_tile_descriptor = lambda *args: (desc, [imp.b.const(0), imp.b.const(0)])
        result = imp._transposed_tile_load(
            _OpView(source, 1, 16), 'B', 'f16',
            Affine2D(row_stride=1, col_stride=64), (16, 16))
        return result, imp.kernel.role(0).region.ops

    def test_later_column_chunk_uses_transposed_dma(self):
        tile, ops = self.transpose()
        self.assertIsNotNone(tile)
        transfers = [op for op in ops if op.kind == 'dma_load']
        self.assertEqual(len(transfers), 1)
        self.assertTrue(transfers[0].attrs['transpose'])
        self.assertTrue(any(op.kind == 'barrier_wait' for op in ops))

    def test_live_non_matrix_user_keeps_gather_path(self):
        tile, ops = self.transpose(extra_user=True)
        self.assertIsNone(tile)
        self.assertFalse(any(op.kind == 'dma_load' for op in ops))

    def test_masked_load_keeps_gather_path(self):
        tile, ops = self.transpose(masked=True)
        self.assertIsNone(tile)
        self.assertFalse(any(op.kind == 'dma_load' for op in ops))


if __name__ == '__main__':
    unittest.main()
