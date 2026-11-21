"""Loop optimisation on TNCA IR (tnca_cc/loop_opt.py, R42).

The loops are written the way the importer writes an Inductor reduction:

    L:  c = iv >= n;  branch c -> E
        lanes = splat(iv) + arange;  m = lanes < N
        x = load(base + (iv << 2), mask=m)
        acc' = select(m, acc + x * k, acc)       k rebuilt every iteration
        T = 0; assign(T, acc'); assign(acc, T)
        iv = iv + 16;  branch 1 -> L
    E:
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test')]
from tnca_cc import loop_opt
from tnca_cc.codegen import CodeGen
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.target import TargetInfo


def reduction(n=64, bound=None, fence=False, offset=0, step=16):
    """sum over n elements of x * (1 / 3.0-ish constant chain), 16 lanes a trip."""
    bound = n if bound is None else bound
    kernel = make_kernel('loop_case')
    b = Builder(kernel, kernel.role(0).region)
    base = b.const(0x4000)
    lanes0 = b.arange()
    zero = b.const(0)
    limit = b.const(n)
    stepv = b.const(step)
    iv = b.sbini('add', zero, 0)
    acc = b.vmovi(0, dtype='f32')
    b.label('for_1')
    done = b.scmp('ge', iv, limit)
    b.branch_nz(done, 'endfor_2')
    first = b.sbini('add', iv, offset) if offset else iv
    lanes = b.vbin('add', b.splat(first), lanes0)
    mask = b.vcmp('lt', lanes, b.vmovi(bound))
    addr = b.sbin('add', base, b.sbini('shl', iv, 2))
    x = b.load(addr, mask=mask, other=0, dtype='f32', size='w')
    k = b.vmovi(0x40400000, dtype='f32')                 # loop-invariant
    r = b.vun('rcp', k, dtype='f32')                     # loop-invariant, SFU
    y = b.vbin('mul', x, r, dtype='f32')
    total = b.vbin('add', acc, y, dtype='f32')
    kept = b.vselect(mask, total, acc)
    if fence:
        b.fence()
    tmp = b.vmovi(0, dtype='f32')
    b._emit('assign', [], [tmp, kept])
    b._emit('assign', [], [acc, tmp])
    nxt = b.sbin('add', iv, stepv)
    b._emit('assign', [], [iv, nxt])
    b.branch_nz(b.const(1), 'for_1')
    b.label('endfor_2')
    b.store(b.const(0x8000), b.reduce('add', acc, broadcast=True), size='w')
    b.exit()
    return kernel


def kinds(kernel, start, end):
    ops = kernel.role(0).region.ops
    names = [o.attrs.get('name') if o.kind == 'label' else None for o in ops]
    return [o.kind for o in ops[names.index(start):names.index(end)]]


class LoopOptTests(unittest.TestCase):
    def test_all_true_mask_is_folded_and_the_loop_cleaned(self):
        kernel = reduction(n=64)
        before = len(kinds(kernel, 'for_1', 'endfor_2'))
        out = loop_opt.optimize(kernel, 16)
        body = kinds(kernel, 'for_1', 'endfor_2')
        self.assertGreater(out['masks'], 0)
        self.assertEqual(out['rotated'], 1)
        self.assertGreater(out['hoisted'], 0)
        # label, shl, add, load, mul, add, assign, add(iv), assign(iv), cmp, branch
        self.assertLess(len(body), before - 8)
        self.assertNotIn('vcmp', body)
        self.assertNotIn('vselect', body)
        self.assertNotIn('vun', body)                    # the reciprocal left the loop
        self.assertEqual(body.count('branch_nz'), 1)
        load = next(o for o in kernel.role(0).region.ops if o.kind == 'load')
        self.assertIsNone(load.attrs.get('mask'))
        # a trip count known to be >= 1: no guard before the loop
        ops = kernel.role(0).region.ops
        at = [o.attrs.get('name') if o.kind == 'label' else None for o in ops].index('for_1')
        self.assertNotEqual(ops[at - 1].kind, 'branch_nz')

    def test_a_tail_mask_is_kept(self):
        for case in (dict(n=37), dict(n=64, bound=63), dict(n=64, offset=16)):
            kernel = reduction(**case)
            out = loop_opt.optimize(kernel, 16)
            self.assertEqual(out['masks'], 0, case)
            load = next(o for o in kernel.role(0).region.ops if o.kind == 'load')
            self.assertIsNotNone(load.attrs.get('mask'), case)
            self.assertIn('vselect', kinds(kernel, 'for_1', 'endfor_2'), case)

    def test_chunk_offsets_inside_the_bound_fold(self):
        kernel = reduction(n=64, bound=96, offset=16, step=16)
        self.assertGreater(loop_opt.optimize(kernel, 16)['masks'], 0)

    def test_a_value_written_in_the_loop_is_not_hoisted(self):
        kernel = reduction(n=64)
        loop_opt.optimize(kernel, 16)
        body = kernel.role(0).region.ops
        names = [o.attrs.get('name') if o.kind == 'label' else None for o in body]
        inside = body[names.index('for_1'):names.index('endfor_2')]
        # the accumulation and the address still run every iteration
        self.assertEqual(sum(1 for o in inside if o.kind == 'vbin'), 2)
        self.assertEqual(sum(1 for o in inside if o.kind == 'load'), 1)
        # the address is a pointer advanced by 16 elements of 4 bytes
        self.assertFalse(any(o.kind == 'sbini' and o.attrs.get('op') == 'shl' for o in inside))
        bumps = [o for o in inside if o.kind == 'sbini' and o.attrs.get('op') == 'add']
        self.assertEqual([o.attrs['imm'] for o in bumps], [64])
        load = next(o for o in inside if o.kind == 'load')
        self.assertIs(load.operands[0], bumps[0].operands[0])

    def test_accesses_a_constant_apart_share_one_pointer(self):
        kernel = make_kernel('two_chunks')
        b = Builder(kernel, kernel.role(0).region)
        base, out = b.const(0x4000), b.const(0x8000)
        zero, limit, step = b.const(0), b.const(128), b.const(32)
        iv = b.sbini('add', zero, 0)
        b.label('for_1')
        b.branch_nz(b.scmp('ge', iv, limit), 'endfor_2')
        second = b.sbini('add', iv, 16)
        lo = b.load(b.sbin('add', base, b.sbini('shl', iv, 2)), dtype='f32', size='w')
        hi = b.load(b.sbin('add', base, b.sbini('shl', second, 2)), dtype='f32', size='w')
        b.store(b.sbin('add', out, b.sbini('shl', iv, 2)), b.vbin('add', lo, hi, dtype='f32'), size='w')
        nxt = b.sbin('add', iv, step)
        b._emit('assign', [], [iv, nxt])
        b.branch_nz(b.const(1), 'for_1')
        b.label('endfor_2')
        b.exit()
        out_ = loop_opt.optimize(kernel, 16)
        self.assertEqual(out_['pointers'], 3)
        ops = kernel.role(0).region.ops
        loads = [o for o in ops if o.kind == 'load']
        self.assertIs(loads[0].operands[0], loads[1].operands[0])
        self.assertEqual(sorted(o.attrs.get('offset', 0) for o in loads), [0, 64])
        bumps = [o.attrs['imm'] for o in ops if o.kind == 'sbini' and (o.attrs.get('comment') or '').startswith('the next')]
        # both bases are constants 0x4000 apart: the store shares the pointer too
        self.assertEqual(bumps, [128])
        self.assertEqual(next(o for o in ops if o.kind == 'store').attrs['offset'], 0x4000)

    def test_a_loop_with_a_fence_keeps_its_code(self):
        kernel = reduction(n=64, fence=True)
        before = [o.kind for o in kernel.role(0).region.ops]
        out = loop_opt.optimize(kernel, 16)
        self.assertEqual((out['masks'], out['hoisted'], out['rotated']), (0, 0, 0))
        self.assertEqual([o.kind for o in kernel.role(0).region.ops], before)

    def test_the_accumulator_is_computed_in_place(self):
        target = TargetInfo.load('tnca-sim-vrf-v5')
        kernel = reduction(n=64)
        loop_opt.optimize(kernel, 16)
        kernel.coalesce_assigns = True
        cg = CodeGen(kernel, target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        text = cg.b.text()
        loop = text[text.index('for_1:'):text.index('endfor_2:')]
        self.assertEqual(len(re.findall(r'\b[vs]\.mov\s', loop)), 0)
        self.assertIn('vf.add', loop)

    def test_unoptimised_kernel_assembles_the_same_program_shape(self):
        # the pass is a rewrite: without it the loop still has its two moves
        target = TargetInfo.load('tnca-sim-vrf-v5')
        kernel = reduction(n=64)
        cg = CodeGen(kernel, target, check_protocol=False)
        cg.assign_shared()
        cg.run()
        text = cg.b.text()
        loop = text[text.index('for_1:'):text.index('endfor_2:')]
        self.assertEqual(len(re.findall(r'\bv\.mov\s', loop)), 2)

    def test_an_unused_tensor_map_is_not_built(self):
        kernel = make_kernel('map_case')
        b = Builder(kernel, kernel.role(0).region)
        scratch = b.sbini('add', b.const(0x9000), 0)
        b.sstore(scratch, b.const(0x4000), offset=0)
        b.sstore(scratch, b.const(4), offset=8)
        b.descriptor(scratch)
        used = b.sbini('add', b.const(0xA000), 0)
        b.sstore(used, b.const(0x4000), offset=0)
        d = b.descriptor(used)
        b.store(b.const(0x8000), b.vmovi(1), size='w')
        b._emit('comment', [], [d], text='a reader of the second map')
        b.exit()
        self.assertEqual(loop_opt.drop_unused_maps(kernel), 1)
        ops = kernel.role(0).region.ops
        self.assertEqual(sum(1 for o in ops if o.kind == 'desc_snapshot'), 1)
        self.assertEqual(sum(1 for o in ops if o.kind == 'sstore'), 1)
        self.assertEqual(len(kernel.descriptors), 1)


if __name__ == '__main__':
    unittest.main()
