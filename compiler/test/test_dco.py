#!/usr/bin/env python3
"""
The DCO pass (tnca_cc/dco.py): what the compiler tells the TMU.

Three kernels whose read counts are known by construction, compiled for the
four-program grid of the 2x2 system. Each test checks the plan -- nAcc, bypass
and the tile granule -- and that the registration stores are in the kernel
ahead of its first load. Running the kernels and measuring the policies is
tools/r38/dco_bench.py; this is the compiler half on its own, so it is quick.

The counts are the paper's rule (doc/DCO.pdf sec IV): nAcc is the number of
iterations, over the grid and every loop, that read the same tile.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))

import triton_bridge  # noqa: E402

GRID = (4, 1, 1)


def _compile(fn, sig, const, args, *, dco=True):
    from tnca_cc import TargetInfo
    from tnca_cc.frontend import TTIRImporter
    target = TargetInfo.load('tnca-sim-2x2')
    mod, _ctx = triton_bridge.make_ttir(fn, sig, const)
    imp = TTIRImporter(target, fn.__name__, scratch_addr=0x200000,
                       scratch_stride=0x4000)
    kernel, _ = imp.run(mod, fn.__name__, args,
                        param_names=triton_bridge.param_names(fn),
                        dco={'grid': GRID} if dco else None)
    return imp, kernel


def _by_name(plan):
    return {t.name: t for t in plan.tensors}


@unittest.skipUnless(triton_bridge.available(), 'triton not importable')
class TestDCOPlan(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, 'tools', 'r38'))
        import dco_bench
        self.B = dco_bench

    def _plan(self, w):
        addrs, _ = self.B.place(w)
        imp, kernel = _compile(w['fn'], w['sig'], w['const'], dict(addrs))
        self.assertIsNotNone(imp.dco_plan, imp.dco_note)
        return imp, kernel, _by_name(imp.dco_plan)

    def test_attention_reuses_kv_once_per_query_block(self):
        """Every Q block of a head reads all of that head's K and V; each Q
        tile is read exactly once, a whole number of lines, so it bypasses."""
        w = self.B.w_attn()
        c = w['const']
        _, _, t = self._plan(w)
        self.assertEqual(t['kt_ptr'].nacc, c['NQB'])
        self.assertEqual(t['v_ptr'].nacc, c['NQB'])
        self.assertEqual(t['kt_ptr'].tiles, c['H'] * c['NKB'])
        self.assertFalse(t['kt_ptr'].bypass)
        self.assertEqual(t['q_ptr'].nacc, 1)
        self.assertTrue(t['q_ptr'].bypass)
        # K and V tiles are 512 contiguous bytes: the granule is the tile.
        self.assertEqual(t['kt_ptr'].run, c['D'] * c['BN'] * 2)

    def test_stream_through_a_weight(self):
        """X is read once but its tiles are half a line wide: it must NOT be
        bypassed, or every line would be fetched twice. W is read by every
        row block."""
        w = self.B.w_xw()
        c = w['const']
        imp, _, t = self._plan(w)
        self.assertEqual(t['w_ptr'].nacc, c['NMB'])
        self.assertEqual(t['x_ptr'].nacc, 1)
        self.assertFalse(t['x_ptr'].bypass)
        self.assertEqual(t['w_ptr'].lsb, 9)           # W's 512-byte tile
        self.assertEqual(t['x_ptr'].lsb, 6)           # X's half-line rows: a line

    def test_gemm_split_program_id(self):
        """`mb = t // NNB, nb = t % NNB`: one loop over a flattened tile
        index, strided by the program id. The product-of-trip-counts rule
        reads it as one dimension; counting gets each operand right."""
        w = self.B.w_gemm()
        c = w['const']
        _, _, t = self._plan(w)
        self.assertEqual(t['a_ptr'].nacc, c['NNB'])
        self.assertEqual(t['b_ptr'].nacc, c['NMB'])
        # Each operand in its own granule: A's 32-byte rows are counted per
        # line, B's contiguous 512-byte tiles whole.
        self.assertEqual(t['a_ptr'].lsb, 6)
        self.assertEqual(t['b_ptr'].lsb, 9)

    def test_registration_precedes_every_load(self):
        """The stores go to the TMU window before anything is loaded, and a
        kernel compiled without DCO has none of them."""
        from tnca_cc import dco
        w = self.B.w_attn()
        addrs, _ = self.B.place(w)
        imp, kernel = _compile(w['fn'], w['sig'], w['const'], dict(addrs))
        ops = kernel.role(0).region.ops
        first_tmu = next(i for i, o in enumerate(ops)
                         if o.kind == 'const' and o.attrs.get('value') == dco.TMU_BASE)
        loads = [i for i, o in enumerate(ops)
                 if o.kind in ('load', 'gather', 'dma_load', 'sload')
                 and i > first_tmu]
        stores = [i for i, o in enumerate(ops) if o.kind == 'sstore']
        n = len(imp.dco_plan.tensors)
        self.assertGreaterEqual(len(stores), 1 + 3 * n)
        self.assertLess(max(stores[:1 + 3 * n]), min(loads))
        imp2, kernel2 = _compile(w['fn'], w['sig'], w['const'], dict(addrs), dco=False)
        self.assertIsNone(imp2.dco_plan)
        self.assertFalse(any(o.kind == 'const' and o.attrs.get('value') == dco.TMU_BASE
                             for o in kernel2.role(0).region.ops))

    def test_refuses_what_it_cannot_count(self):
        """A loop bound read from memory: the pass registers nothing and
        says why, and the kernel still compiles."""
        import triton
        import triton.language as tl

        @triton.jit
        def data_bound(n_ptr, x_ptr, o_ptr, BLOCK: tl.constexpr):
            n = tl.load(n_ptr)
            offs = tl.arange(0, BLOCK)
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for i in range(n):
                acc += tl.load(x_ptr + i * BLOCK + offs)
            tl.store(o_ptr + offs, acc)

        imp, _ = _compile(data_bound,
                          {'n_ptr': '*i32', 'x_ptr': '*fp32', 'o_ptr': '*fp32',
                           'BLOCK': 'constexpr'},
                          {'BLOCK': 16},
                          {'n_ptr': 0x10000, 'x_ptr': 0x11000, 'o_ptr': 0x12000})
        self.assertIsNone(imp.dco_plan)
        self.assertIn('no DCO registration', imp.dco_note)


if __name__ == '__main__':
    unittest.main(verbosity=2)
