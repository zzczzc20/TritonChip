"""R42 fp8sw: address displacement folding, memory disambiguation in the scheduler, the shared reciprocal.

Each is a rewrite of the same program: the tests check the rewrite and that what it leaves assembles to the same
addresses (and, for the reciprocal, to the same operations in the same order).
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test')]
from tnca_cc import mathlib, schedule
from tnca_cc.codegen import CodeGen
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.optimization import fold_address_offsets
from tnca_cc.target import TargetInfo


def scratch_kernel():
    """A spilled tile: row r stored at scratch+64r, read back, with the scratch base tagged as the importer does."""
    kernel = make_kernel('spill')
    b = Builder(kernel, kernel.role(0).region)
    base = b.sload(b.special('arg_base'), offset=88, size='d')
    b.region.ops[-1].attrs['region'] = 'scratch'
    tensor = b.sload(b.special('arg_base'), offset=0, size='d')
    return kernel, b, base, tensor


class FoldAddressOffsets(unittest.TestCase):
    def test_a_constant_sum_becomes_the_displacement(self):
        kernel, b, base, _ = scratch_kernel()
        v = b.vmovi(7)
        for r in range(4):
            b.store(b.sbini('add', base, 64 * r), v, size='w')
        x = b.load(b.sbini('add', b.sbini('add', base, 128), 8), dtype='i32', size='w')
        b.store(base, x, size='w')
        b.exit()
        out = fold_address_offsets(kernel)
        ops = kernel.role(0).region.ops
        stores = [o for o in ops if o.kind == 'store']
        self.assertEqual([o.attrs.get('offset', 0) for o in stores], [0, 64, 128, 192, 0])
        self.assertTrue(all(o.operands[0] is base for o in stores))
        self.assertEqual(next(o for o in ops if o.kind == 'load').attrs['offset'], 136)
        self.assertEqual(out['folded'], 5)
        self.assertEqual(sum(1 for o in ops if o.kind == 'sbini'), 0)       # every sum is gone

    def test_a_sum_that_something_else_reads_stays(self):
        kernel, b, base, _ = scratch_kernel()
        p = b.sbini('add', base, 64)
        b.store(p, b.vmovi(1), size='w')
        b.store(b.sbin('add', p, p), b.vmovi(2), size='w')     # names the sum as a value
        b.exit()
        fold_address_offsets(kernel)
        ops = kernel.role(0).region.ops
        self.assertEqual(sum(1 for o in ops if o.kind == 'sbini'), 1)
        self.assertEqual(ops[[o.kind for o in ops].index('store')].attrs['offset'], 64)

    def test_a_pointer_written_in_place_is_not_folded(self):
        kernel, b, base, _ = scratch_kernel()
        ptr = b.sbini('add', base, 0)
        b.label('L')
        b.store(b.sbini('add', ptr, 16), b.vmovi(1), size='w')
        b._emit('assign', [], [ptr, b.sbini('add', ptr, 64)])
        b.branch_nz(b.const(1), 'L')
        b.exit()
        out = fold_address_offsets(kernel)
        self.assertEqual(out['folded'], 0)

    def test_a_displacement_beyond_16_bits_is_not_folded(self):
        kernel, b, base, _ = scratch_kernel()
        b.store(b.sbini('add', base, 40000), b.vmovi(1), size='w')
        b.exit()
        self.assertEqual(fold_address_offsets(kernel)['folded'], 0)

    def test_the_assembly_names_the_same_address(self):
        target = TargetInfo.load('tnca-sim-vrf-v5')
        text = []
        for fold in (False, True):
            kernel, b, base, _ = scratch_kernel()
            for r in range(3):
                b.store(b.sbini('add', base, 64 * r + 8), b.vmovi(r), size='w')
            b.exit()
            if fold:
                fold_address_offsets(kernel)
            cg = CodeGen(kernel, target, check_protocol=False)
            cg.assign_shared()
            cg.run()
            text.append(cg.b.text())
        self.assertEqual(len(re.findall(r'\bs\.addi\b', text[0])), 3)
        self.assertEqual(len(re.findall(r'\bs\.addi\b', text[1])), 0)
        self.assertEqual(re.findall(r'v\.st\.w\s+\[s\d+, #(\d+)\]', text[1]), ['8', '72', '136'])


class MemoryDisambiguation(unittest.TestCase):
    def locations(self):
        kernel, b, base, tensor = scratch_kernel()
        b.store(b.sbini('add', base, 0), b.vmovi(1), size='w')            # scratch [0, 64)
        b.store(b.sbini('add', base, 64), b.vmovi(2), size='w')           # scratch [64, 128)
        b.load(b.sbini('add', base, 32), dtype='i32', size='w')           # scratch [32, 96): meets both
        b.load(b.sbini('add', base, 200), dtype='i32', size='w')          # scratch [200, 264): meets none
        b.load(tensor, dtype='f32', size='w')                             # a tensor
        b.exit()
        ops = kernel.role(0).region.ops
        locs = schedule.memory_locations(ops)
        mem = [o for o in ops if id(o) in locs and o.kind in ('store', 'load')]
        return [locs[id(o)] for o in mem]

    def test_ranges_and_regions(self):
        s0, s1, l_mid, l_far, l_tensor = self.locations()
        self.assertEqual([s0[0], s1[0], l_mid[0], l_far[0], l_tensor[0]], ['scratch'] * 4 + ['other'])
        self.assertFalse(schedule.may_alias(s0, s1))
        self.assertTrue(schedule.may_alias(s0, l_mid))
        self.assertTrue(schedule.may_alias(s1, l_mid))
        self.assertFalse(schedule.may_alias(s0, l_far))
        self.assertFalse(schedule.may_alias(s1, l_tensor))                # scratch is never a tensor
        self.assertTrue(schedule.may_alias(l_tensor, (('other'), None, None, None)))

    def test_a_load_passes_a_store_it_cannot_meet_and_not_one_it_can(self):
        model = schedule.Model('flipflop', {
            'latency': {'salu': 2, 'smul': 5, 'valu': 2, 'vfp': 4, 'sfu': 27, 'vcmp': 1, 'pred': 2, 'reduce': 1,
                        'reduce_b': 2, 'vextract': 1, 'vinsert': 2, 'acc_read': 2, 'alias': 0, 'other': 2},
            'issue': {'scalar': 1, 'vector_reads_vrf': 1, 'vector_no_vrf_read': 1}, 'lsu': {'base': 2, 'bytes_per_beat': 8}})
        kernel, b, base, tensor = scratch_kernel()
        store = b.vmovi(3)
        b.store(b.sbini('add', base, 0), store, size='w')
        far = b.load(b.sbini('add', base, 512), dtype='i32', size='w')           # another scratch row
        tens = b.load(tensor, dtype='f32', size='w')                              # a tensor
        same = b.load(b.sbini('add', base, 0), dtype='i32', size='w')            # the row just stored
        b.store(b.sbini('add', base, 1024), b.vbin('add', far, tens, dtype='f32'), size='w')
        b.store(b.sbini('add', base, 2048), same, size='w')
        b.exit()
        ops = kernel.role(0).region.ops
        locs = schedule.memory_locations(ops)
        units, _ = schedule._units(ops)
        legacy = schedule._graph(units, model, set())
        fine = schedule._graph(units, model, set(), locs)
        index = {id(n.op): n for n in fine}
        legacy_index = {id(n.op): n for n in legacy}
        first_store = next(o for o in ops if o.kind == 'store')
        for load, may in ((far.name, False), (tens.name, False), (same.name, True)):
            op = next(o for o in ops if o.kind == 'load' and o.results[0].name == load)
            self.assertIn(legacy_index[id(first_store)], legacy_index[id(op)].preds)         # before: always ordered
            self.assertEqual(index[id(first_store)] in index[id(op)].preds, may, load)       # now: only the one it meets


class SharedReciprocal(unittest.TestCase):
    def _ops(self, build):
        kernel = make_kernel('div')
        b = Builder(kernel, kernel.role(0).region)
        a, y = b.vmovi(0x40000000, dtype='f32'), b.vmovi(0x40400000, dtype='f32')
        k = lambda c: b.vmovi(int.from_bytes(__import__('struct').pack('<f', c), 'little'), dtype='f32')
        build(b, k, a, y)
        return [(o.kind, o.attrs.get('op')) for o in kernel.role(0).region.ops][2:]

    def test_the_same_operations_after_the_reciprocal(self):
        for ieee, div in ((False, mathlib.div), (True, mathlib.div_ieee)):
            whole = self._ops(lambda b, k, a, y: div(b, k, a, y, dtype='f32'))
            def parts(b, k, a, y):
                r = mathlib.rcp(b, k, y, dtype='f32')
                mathlib.div_with_rcp(b, a, y, r, ieee=ieee, dtype='f32')
            split = self._ops(parts)
            self.assertEqual(whole, split, f'ieee={ieee}')


if __name__ == '__main__':
    unittest.main()
