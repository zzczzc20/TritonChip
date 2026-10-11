#!/usr/bin/env python3
"""
C3: an explicit two-role pipeline (compiler_overview.md sec 15 C3).

sec 15 calls this 第一阶段最重要的里程碑 and lists what it must cover:
ready/free double buffering, 0/1/2/odd iteration counts, predicated-off async,
out-of-order returns, delayed completion, error propagation, and draining at
program exit.

The producer fills alternating RF windows; the consumer reads them where they
lie. Nothing goes through memory except the startup flag, which is the point
of sec 7.3's RF channel -- there is no scratchpad for a shared-memory Aref to
live in.

Run with +allroles +aliasvrf: the handoff only exists if the two contexts'
logical vector names map to the same physical registers, which is the
programmable per-RCTX base that sec 13.1 asked for and t13_share first
demonstrated.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

import rtl_harness as H  # noqa: E402
from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.diagnostics import Diagnostic  # noqa: E402
from tnca_cc.pipeline import build_pipeline, plan_pipeline  # noqa: E402

SRC = 0x8000
DESC = 0xc000
FLAG = 0xe000
RESULT = 0xe100
SENTINEL = 0xdeadbeef

# Wide enough that every iteration's tile is in bounds. At COLS=64 the fifth
# iteration started at column 64, which the descriptor pads with zero -- the
# transfer was right and the expectation was not.
ROWS, COLS = 16, 160         # ten 16-column tiles
TILE = 16


def tensormap(base, e0, e1, s0, s1, b0, b1, rank=2, dtype=0, pad=0):
    w = [0] * 32
    w[0] = base
    w[1] = e0
    w[2] = e1
    w[6] = s0
    w[7] = s1
    w[11] = (b1 << 32) | b0
    w[13] = (pad << 48) | (dtype << 40) | (rank << 32)
    return w


def half(r, c):
    """A distinct FP16 bit pattern per element, as a plain integer."""
    return (r * COLS + c) & 0x7fff


class TestPipelinePlan(unittest.TestCase):
    def setUp(self):
        self.t = TargetInfo.default()

    def test_ready_and_free_are_separate_barriers(self):
        """ISA sec 9A.1: a transfer completing is not the consumer having
        finished reading. A pipeline with one barrier per slot implements only
        load-ready, which sec 13.5 names as the most common way to get this
        wrong."""
        p = plan_pipeline(self.t, stages=2, iterations=4)
        self.assertEqual(p.barriers, 4)

    def test_generation_follows_the_iteration(self):
        p = plan_pipeline(self.t, stages=2, iterations=6)
        self.assertEqual([p.generation(i) for i in range(6)],
                         [1, 1, 2, 2, 3, 3])
        self.assertEqual([p.slot(i) for i in range(6)], [0, 1, 0, 1, 0, 1])

    def test_too_many_stages_refused(self):
        with self.assertRaises(Diagnostic):
            plan_pipeline(self.t, stages=9, iterations=4, window_registers=8)

    def test_more_roles_than_contexts_refused(self):
        with self.assertRaises(Exception):
            plan_pipeline(self.t, stages=2, iterations=4, consumers=4)


class TestPipelineOnRTL(unittest.TestCase):
    def setUp(self):
        self.t = TargetInfo.default()
        if not H.available():
            self.skipTest(H.why_unavailable())

    def _run(self, iterations, *, stages=2, bench='fast', predicated_off=False,
             seed=None):
        t = self.t
        plan = plan_pipeline(t, stages=stages, iterations=iterations)

        def coords(i):
            return (0, i * TILE)

        def consume(cb, i, window):
            # Read the window where it lies -- no copy, no memory (sec 7.3).
            v = cb.window_reg(window, 0)
            addr = cb.const(RESULT + i * 64)
            cb.store(addr, v)

        kernel, _ = build_pipeline(t, plan, name=f"pipe{iterations}",
                                   desc_addr=DESC,
                                   coords_for=coords, consume=consume)
        p, _peaks = H.multirole_program(f"pipe{iterations}", kernel, t)

        # Each tile's first window register holds row 0, columns [i*16, i*16+4)
        # packed two FP16 per 32-bit lane.
        p.check_base(RESULT)
        for i in range(iterations):
            for lane in range(4):
                c = i * TILE + lane * 2
                expect = (half(0, c + 1) << 16) | half(0, c)
                p.check_word(RESULT, i * 64 + lane * 4, expect)

        src = [half(r, c) for r in range(ROWS) for c in range(COLS)]
        p.words(SRC, H.pack16(src), 'src')
        p.words(DESC, tensormap(SRC, ROWS, COLS, COLS * 2, 2, TILE, TILE),
                'desc')
        p.words(FLAG, [0, 0], 'flag')
        p.words(RESULT, H.pack32([SENTINEL] * (iterations * 16 + 16)), 'res')

        # The memory model's LFSR drives both the back-pressure pattern and
        # which eligible response is retired first, so this is the seed
        # R1-08 REG-04 means by "传到真正控制 stimulus、stall、reorder 的 PRNG".
        args = ['+allroles', '+aliasvrf']
        if seed is not None:
            args.append(f"+memseed={seed}")
        ok, out = H.run(p, extra_args=tuple(args),
                        max_cycles=400000, bench=bench)
        self.assertTrue(ok, f"{iterations}-iteration pipeline on the "
                            f"{bench} memory failed:\n{out}")
        return out

    def test_zero_iterations(self):
        """sec 15 C3's 0 iterations: the program must still start, initialise
        and drain."""
        self._run(0)

    def test_one_iteration(self):
        """Fewer iterations than stages: the ring never fills."""
        self._run(1)

    def test_two_iterations(self):
        """Exactly the stage count: every slot used once, none reused."""
        self._run(2)

    def test_three_iterations_odd(self):
        """sec 15 C3's odd count: slot 0 is reused, slot 1 is not, so the
        drain is asymmetric."""
        self._run(3)

    def test_five_iterations(self):
        self._run(5)

    def test_six_iterations(self):
        self._run(6)

    def test_delayed_completion(self):
        """sec 15 C3's 延迟 completion. A seven-cycle memory with random
        back-pressure: the transfers finish long after they were issued, so a
        consumer that assumed issue order or a fixed latency breaks here and
        not on the fast model."""
        if not H.bench_available('slow'):
            self.skipTest('tb_core_slow.vvp not built; run `make core-slowmem`')
        self._run(6, bench='slow')

    def test_out_of_order_returns(self):
        """sec 15 C3's 乱序返回. Memory responses come back in a different
        order from the requests, which is the case a barrier that counted
        completions in issue order would get wrong."""
        if not H.bench_available('ooo'):
            self.skipTest('tb_core_ooo.vvp not built; run `make core-ooo`')
        self._run(6, bench='ooo')

    def test_predicated_off_async_still_completes(self):
        """sec 15 C3's predicated-off async.

        ISA sec 9A.3: a fully predicated-off asynchronous command still
        completes once, which is what keeps a static operation count correct
        under predication. Iteration 1's transfer is masked off, so its window
        keeps what it held -- the sentinel the producer never overwrote -- and
        the pipeline still advances rather than hanging on a completion that
        never came.
        """
        t = self.t
        plan = plan_pipeline(t, stages=2, iterations=4)

        def consume(cb, i, window):
            v = cb.window_reg(window, 0)
            addr = cb.const(RESULT + i * 64)
            cb.store(addr, v)

        # Four iterations over two slots, with the LAST one predicated off.
        # Slot 1 therefore already holds iteration 1's tile, so "the transfer
        # wrote nothing" is checkable precisely: iteration 3 must read back
        # iteration 1's data. Predicating the first use of a slot would only
        # have tested the reset value, which is not the property.
        kernel, _ = build_pipeline(t, plan, name='pipe_pred', desc_addr=DESC,
                                   coords_for=lambda i: (0, i * TILE),
                                   consume=consume, predicate_off={3})
        p, _ = H.multirole_program('pipe_pred', kernel, t)

        p.check_base(RESULT)
        for i in (0, 1, 2):
            for lane in range(4):
                c = i * TILE + lane * 2
                p.check_word(RESULT, i * 64 + lane * 4,
                             (half(0, c + 1) << 16) | half(0, c))
        for lane in range(4):
            c = 1 * TILE + lane * 2      # iteration 1's tile, still in place
            p.check_word(RESULT, 3 * 64 + lane * 4,
                         (half(0, c + 1) << 16) | half(0, c))

        src = [half(r, c) for r in range(ROWS) for c in range(COLS)]
        p.words(SRC, H.pack16(src), 'src')
        p.words(DESC, tensormap(SRC, ROWS, COLS, COLS * 2, 2, TILE, TILE),
                'desc')
        p.words(FLAG, [0, 0], 'flag')
        p.words(RESULT, H.pack32([SENTINEL] * 80), 'res')

        ok, out = H.run(p, extra_args=('+allroles', '+aliasvrf'),
                        max_cycles=400000)
        self.assertTrue(ok, f"predicated-off async broke the pipeline:\n{out}")

    def _strided_transfer(self, rank):
        t = self.t
        plan = plan_pipeline(t, stages=2, iterations=2)

        def consume(cb, i, window):
            v = cb.window_reg(window, 0)
            addr = cb.const(RESULT + i * 64)
            cb.store(addr, v)

        kernel, _ = build_pipeline(t, plan, name='pipe_err', desc_addr=DESC,
                                   coords_for=lambda i: (0, i * TILE),
                                   consume=consume)
        p, _ = H.multirole_program('pipe_err', kernel, t)

        src = [half(r, c) for r in range(ROWS) for c in range(COLS)]
        p.words(SRC, H.pack16(src), 'src')
        # The current engine accepts a 26-byte FP16 row stride. An invalid
        # rank is the negative twin; both use the same data and pipeline.
        p.words(DESC, tensormap(SRC, ROWS, 13, 13 * 2, 2, TILE, TILE,
                               rank=rank), 'desc')
        p.words(FLAG, [0, 0], 'flag')
        p.words(RESULT, H.pack32([SENTINEL] * 48), 'res')
        return p

    def test_unaligned_row_stride(self):
        """A legal 26-byte row stride preserves packed data and tail padding."""
        p = self._strided_transfer(rank=2)
        halves = list(range(13)) + [0] * 3 + list(range(13, 26)) + [0] * 3
        expected = [halves[2 * i] | (halves[2 * i + 1] << 16) for i in range(16)]
        expected += [0] * 16 + [SENTINEL] * 16
        p.check_base(RESULT)
        for i, value in enumerate(expected):
            p.check_word(RESULT, i * 4, value)
        ok, out = H.run(p, extra_args=('+allroles', '+aliasvrf'), max_cycles=400000)
        self.assertTrue(ok, f"strided transfer data or padding is wrong:\n{out}")

    def test_error_propagation(self):
        """Rank zero faults, wakes the consumer and propagates barrier error."""
        p = self._strided_transfer(rank=0)
        p.check_base(RESULT)
        for i in range(48):
            p.check_word(RESULT, i * 4, SENTINEL)

        ok, out = H.run(p, extra_args=('+allroles', '+aliasvrf',
                                       '+expectfault=04'),
                        max_cycles=400000)
        self.assertTrue(ok,
                        f"a faulting transfer did not propagate:\n{out}")
        self.assertNotIn('ran to completion', out)

    def test_two_consumers_on_four_contexts(self):
        """sec 15 C4 defers P + M0 + M1 + E to 四 RCTX 硬件 profile 可用后.
        The profile now exists (`make core4`), so this runs one producer and
        two consumers on it.

        The property being tested is sec 7.4's: 只有两个 consumer 都释放，
        producer 才能覆盖该 slot. The producer arms `free` for two arrivals and
        waits for both; a slot released after one would let it overwrite a
        window the other consumer is still reading.
        """
        self._two_consumers()

    def _two_consumers(self, bench='rctx4', seed=None):
        if not H.bench_available(bench):
            self.skipTest(f"{H.BENCHES[bench]} not built; run `make core4`")
        # A different machine, so a different target description -- generated
        # from the same RTL with the parameters the bench passes.
        t = TargetInfo.load('tnca-sim-4rctx')
        self.assertEqual(t.num_rctx, 4)
        plan = plan_pipeline(t, stages=2, iterations=4, consumers=2,
                             window_registers=4)

        def consume(cb, i, window):
            v = cb.window_reg(window, 0)
            addr = cb.const(RESULT + i * 64)
            cb.store(addr, v)

        kernel, _ = build_pipeline(t, plan, name='pipe4', desc_addr=DESC,
                                   coords_for=lambda i: (0, i * TILE),
                                   consume=consume)
        # Both consumers write the same result, so either order gives the
        # same memory; what the checks prove is that the ring advanced.
        p, _ = H.multirole_program('pipe4', kernel, t)
        p.check_base(RESULT)
        for i in range(4):
            for lane in range(4):
                c = i * TILE + lane * 2
                p.check_word(RESULT, i * 64 + lane * 4,
                             (half(0, c + 1) << 16) | half(0, c))

        src = [half(r, c) for r in range(ROWS) for c in range(COLS)]
        p.words(SRC, H.pack16(src), 'src')
        # A 4-register window holds half a 16x16 tile, so the block shape is
        # 8 rows rather than 16.
        p.words(DESC, tensormap(SRC, ROWS, COLS, COLS * 2, 2, 8, TILE),
                'desc')
        p.words(RESULT, H.pack32([SENTINEL] * 80), 'res')

        args = ['+allroles', '+aliasvrf']
        if seed is not None:
            args.append(f"+memseed={seed}")
        ok, out = H.run(p, extra_args=tuple(args),
                        max_cycles=400000, bench=bench)
        self.assertTrue(ok, f"four-context pipeline on {bench} failed:\n{out}")
        return out

    def test_three_stage_ring(self):
        """Three slots rather than two: the reuse distance changes, and with
        it which generation the producer waits on."""
        self._run(7, stages=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
