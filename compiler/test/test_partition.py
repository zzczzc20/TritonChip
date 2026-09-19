#!/usr/bin/env python3
"""
C4: automatic Context Specialization (compiler_overview.md sec 15 C4).

sec 15 C4 lists what to build at this point -- graph construction, partition
propagation, loop cloning, producer/consumer analysis -- and says to start with
GEMM-shaped loops before widening the control flow.

The two constraints being tested are sec 7.1's and sec 7.6's:

  the graph must carry more than SSA def-use, or a partitioner will split an
  accumulator's updates across roles or move a write past an asynchronous
  command still reading the register;

  the physical RCTX count must not shape the graph -- 否则两角色和四角色版本
  会形成两套编译器 -- so partitioning is logical and merging happens last.

The end of the file compiles an automatically partitioned GEMM and runs it.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

import rtl_harness as H  # noqa: E402
from fp32 import f16_bits_to_f32_bits, f32_to_f16_bits, mma_ref  # noqa: E402
from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.diagnostics import Diagnostic  # noqa: E402
from tnca_cc.gemm import plan_gemm  # noqa: E402
from tnca_cc.kernels import elementwise  # noqa: E402
from tnca_cc.partition import (Communication, EdgeKind, OpClass,  # noqa: E402
                               SpecializationTargetInfo, build_graph,
                               map_to_rctx, partition_graph, specialize,
                               specialize_gemm, specialize_gemm_rolled)
from tnca_cc.gemm import build_gemm  # noqa: E402

# Placed above the code: 256 output checks is several thousand instructions,
# and the assembler refuses a backwards .org rather than overlaying data on
# the program. The image is 16384 doublewords, so 0x20000 is the ceiling.
A_BASE, B_BASE = 0x10000, 0x14000
DA_BASE, DB_BASE = 0x18000, 0x18100
OUT_BASE, FLAG = 0x19000, 0x18200
SENTINEL = 0xdeadbeef


def gemm_region(t, k=32):
    plan = plan_gemm(t, m=16, n=16, k=k)
    kernel, _ = build_gemm(t, plan, a_desc_addr=DA_BASE, b_desc_addr=DB_BASE,
                           out_addr=OUT_BASE)
    return kernel.role(0).region, plan


class TestGraph(unittest.TestCase):
    """sec 7.1: def-use alone is not the graph."""

    def setUp(self):
        self.t = TargetInfo.default()
        self.info = SpecializationTargetInfo(self.t)

    def test_accumulator_order_is_an_edge(self):
        """ISA sec 9C.1 fixes the accumulation order, so the updates cannot be
        reordered or split across roles. Without this edge nothing in the
        graph says so."""
        region, _ = gemm_region(self.t)
        g = build_graph(region, self.info)
        acc = [e for e in g.edges if e.kind is EdgeKind.ACCUMULATOR]
        self.assertTrue(acc)

    def test_loop_carried_edges_on_a_loop(self):
        k, _ = elementwise(self.t, name='ew', n_elements=40,
                           addresses={'x': 0, 'y': 0, 'out': 0})
        g = build_graph(k.role(0).region, self.info)
        lc = [e for e in g.edges if e.kind is EdgeKind.LOOP_CARRIED]
        self.assertTrue(lc, 'a loop produced no loop-carried edges')

    def test_memory_edges_order_stores(self):
        region, _ = gemm_region(self.t)
        g = build_graph(region, self.info)
        self.assertTrue([e for e in g.edges if e.kind is EdgeKind.MEMORY])

    def test_every_operation_is_classified(self):
        """sec 7.2 classifies every operation; an unclassified one would be
        placed arbitrarily, so the adapter raises instead."""
        region, _ = gemm_region(self.t)
        for op in region.ops:
            self.assertIsInstance(self.info.classify(op), OpClass)


class TestCommunication(unittest.TestCase):
    """sec 7.3's order, applied in order."""

    def setUp(self):
        self.t = TargetInfo.default()

    def test_nothing_falls_back_to_memory(self):
        """The order exists so a tensor reaches step 4 and becomes a shared
        window. Sending one through a global temporary would be a copy through
        memory in a machine built around not having one."""
        for k in (16, 32, 64):
            with self.subTest(k=k):
                region, _ = gemm_region(self.t, k)
                _g, plan = specialize(region, self.t)
                globals_ = [c for c in plan.crossings
                            if c[3] is Communication.GLOBAL]
                self.assertFalse(globals_, f"{globals_} needed memory")

    def test_a_tensor_crosses_as_a_shared_window(self):
        region, _ = gemm_region(self.t)
        _g, plan = specialize(region, self.t)
        windows = [c for c in plan.crossings
                   if c[3] is Communication.SHARED_WINDOW]
        self.assertTrue(windows)

    def test_a_descriptor_is_a_resource_not_a_value(self):
        """Barriers, descriptors and accumulators are per-core objects every
        role names by the same static index (sec 10.1). Treating one as a
        value to move would send it through a global temporary."""
        region, _ = gemm_region(self.t)
        _g, plan = specialize(region, self.t)
        descs = [c for c in plan.crossings if c[0].kind == 'desc']
        self.assertTrue(descs)
        for c in descs:
            self.assertIs(c[3], Communication.SHARED_RESOURCE)

    def test_a_predicate_is_recomputed(self):
        """The predicate file is per-RCTX private, and a v.cmp is one
        instruction: cheaper to recompute than to communicate."""
        k, _ = elementwise(self.t, name='ew', n_elements=40,
                           addresses={'x': 0, 'y': 0, 'out': 0})
        _g, plan = specialize(k.role(0).region, self.t)
        preds = [c for c in plan.crossings if c[0].kind == 'pred']
        for c in preds:
            self.assertIs(c[3], Communication.REMATERIALIZE)


class TestMapping(unittest.TestCase):
    """sec 7.6: the physical mapping happens last."""

    def setUp(self):
        self.t = TargetInfo.default()
        self.info = SpecializationTargetInfo(self.t)

    def _plan(self, rctx):
        region, _ = gemm_region(self.t)
        g = build_graph(region, self.info)
        plan = partition_graph(g, self.info)

        class Info(SpecializationTargetInfo):
            def budget(inner):
                b = dict(SpecializationTargetInfo.budget(inner))
                b['rctx'] = rctx
                return b

        map_to_rctx(plan, Info(self.t))
        return plan

    def test_the_logical_partition_does_not_depend_on_the_rctx_count(self):
        """One graph, several mappings. If the partition itself changed with
        the context count there would be two compilers, which is exactly what
        sec 7.6 forbids."""
        region, _ = gemm_region(self.t)
        g = build_graph(region, self.info)
        base = [p.label for p in partition_graph(g, self.info).partitions]
        for rctx in (2, 4):
            with self.subTest(rctx=rctx):
                self.assertEqual([p.label for p in self._plan(rctx).partitions],
                                 base)

    def test_two_contexts_merge_matrix_with_epilogue(self):
        plan = self._plan(2)
        self.assertEqual(len(plan.roles), 2)
        self.assertTrue(any('M' in v and 'E' in v for v in plan.roles.values()))

    def test_four_contexts_need_no_merge(self):
        """sec 15 C4's four-context check: with room, transfer, matrix and
        epilogue each get their own context and nothing is merged."""
        plan = self._plan(4)
        self.assertEqual(plan.merges, [])
        worker_sets = [set(v) - {'A', 'S'} for v in plan.roles.values()]
        self.assertIn({'P'}, worker_sets)
        self.assertIn({'M'}, worker_sets)
        self.assertIn({'E'}, worker_sets)

    def test_transfer_and_matrix_are_never_merged(self):
        """They are the two that overlap in time; merging them serialises the
        pipeline specialisation exists to create."""
        for rctx in (2, 3, 4):
            with self.subTest(rctx=rctx):
                for labels in self._plan(rctx).roles.values():
                    self.assertFalse('P' in labels and 'M' in labels)

    def test_one_context_is_refused_rather_than_serialised(self):
        with self.assertRaises(Diagnostic):
            self._plan(1)


class TestSpecializedGemmOnRTL(unittest.TestCase):
    """The end of C4: an automatically partitioned GEMM, on the machine."""

    def setUp(self):
        self.t = TargetInfo.default()
        if not H.available():
            self.skipTest(H.why_unavailable())

    def _run(self, k, *, stages=2, seed=1, bench='fast', rolled=False):
        t = self.t
        M = N = 16
        rng = random.Random(seed)
        a = [f32_to_f16_bits(rng.uniform(-4, 4)) for _ in range(M * k)]
        b = [f32_to_f16_bits(rng.uniform(-4, 4)) for _ in range(k * N)]
        ref = mma_ref(a, b, [0] * (M * N), M, N, k)

        gp = plan_gemm(t, m=M, n=N, k=k)
        build = specialize_gemm_rolled if rolled else specialize_gemm
        kernel, plan = build(
            t, gp, name=f"cs_gemm_k{k}{'r' if rolled else ''}",
            a_desc_addr=DA_BASE, b_desc_addr=DB_BASE, out_addr=OUT_BASE,
            stages=stages)
        self.assertEqual(len(kernel.roles), 2)

        # The seed and bench are part of the name: two tests that differ only
        # in those would otherwise write the same .s and .hex, and whichever
        # ran last would be the one left on disk to debug.
        p, _ = H.multirole_program(
            f"cs_gemm_k{k}_s{stages}{'r' if rolled else ''}_{bench}{seed}",
            kernel, t)
        p.check_base(OUT_BASE)
        for i in range(M * N):
            p.check_word(OUT_BASE, i * 4, ref[i])

        def tmap(base, e0, e1, s0, s1, b0, b1):
            w = [0] * 32
            w[0], w[1], w[2], w[6], w[7] = base, e0, e1, s0, s1
            w[11] = (b1 << 32) | b0
            w[13] = (2 << 32)
            return w

        p.words(A_BASE, H.pack16(a), 'a')
        p.words(B_BASE, H.pack16(b), 'b')
        p.words(DA_BASE, tmap(A_BASE, M, k, k * 2, 2, 16, 16), 'da')
        p.words(DB_BASE, tmap(B_BASE, k, N, N * 2, 2, 16, 16), 'db')
        p.words(FLAG, [0, 0], 'flag')
        p.words(OUT_BASE, H.pack32([SENTINEL] * (M * N + 8)), 'out')

        ok, out = H.run(p, extra_args=('+allroles', '+aliasvrf'),
                        max_cycles=600000, bench=bench)
        self.assertTrue(ok, f"specialised GEMM K={k} failed:\n{out}")

    def test_k32_two_stage(self):
        self._run(32)

    def test_k64_two_stage(self):
        """Four k steps over two slots: every slot is reused, so the free
        barrier is what keeps the producer from overwriting a fragment the
        matrix unit is still reading."""
        self._run(64)

    def test_k16_single_step(self):
        """One k step: the ring never wraps, and the drain is the whole
        program."""
        self._run(16)

    def test_k64_three_stage(self):
        self._run(64, stages=3, seed=5)

    def test_k64_under_delayed_completion(self):
        if not H.bench_available('slow'):
            self.skipTest('tb_core_slow.vvp not built')
        self._run(64, seed=7, bench='slow')

    def test_rolled_k64(self):
        """sec 15 C4's loop cloning. Each role gets its own copy of the k
        loop, the body covers one trip through the slot ring so every W and V
        number stays static (sec 9), and each role derives the generation from
        its own trip counter -- it cannot read the other's token, because the
        scalar file is per-RCTX private (sec 7.3)."""
        self._run(64, rolled=True, seed=11)

    def test_rolled_k128(self):
        """Twice the K, the same code. The unrolled form grows with K; the
        rolled one does not."""
        self._run(128, rolled=True, seed=12)

    def test_rolled_three_stage(self):
        self._run(96, stages=3, rolled=True, seed=13)

    def test_rolled_under_delayed_completion(self):
        if not H.bench_available('slow'):
            self.skipTest('tb_core_slow.vvp not built')
        self._run(64, rolled=True, seed=14, bench='slow')

    def test_rolled_k_not_divisible_refused(self):
        """A K that does not divide into whole trips is refused: the body
        covers one trip through the ring, and a partial trip would need a
        different W for the same code."""
        gp = plan_gemm(self.t, m=16, n=16, k=48)
        with self.assertRaises(Diagnostic):
            specialize_gemm_rolled(self.t, gp, a_desc_addr=DA_BASE,
                                   b_desc_addr=DB_BASE, out_addr=OUT_BASE,
                                   stages=4)

    def test_k64_out_of_order(self):
        if not H.bench_available('ooo'):
            self.skipTest('tb_core_ooo.vvp not built')
        self._run(64, seed=9, bench='ooo')


if __name__ == '__main__':
    unittest.main(verbosity=2)
