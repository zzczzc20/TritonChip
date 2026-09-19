#!/usr/bin/env python3
"""
Protocol model tests (compiler_overview.md sec 16.2, "协议模型测试").

The enumeration the plan asks for, verbatim: 1-3 stages, 1-2 consumers,
different completion orders, short loops, all-false, faults. Each case drives
the ownership/event model the way a generated pipeline would, and the model
refuses anything a correct lowering must never emit.
"""
import itertools
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))

from tnca_cc.protocol import (  # noqa: E402
    BufferState, ProtocolError, ProtocolModel, completion_orders)

FILL = 0      # barrier counting transfer completions
EMPTY = 1     # barrier counting consumer releases


def run_pipeline(*, stages, consumers, iters, order_fn,
                 cmds_per_stage=1, all_false=False):
    """Drive a depth-`stages` pipeline for `iters` iterations.

    This is the shape sec 5.3 and sec 7.4 describe: an RF channel with a fill
    barrier and an empty barrier, a producer role, and one or two consumer
    roles that jointly release each buffer.
    """
    consumer_names = [f"consumer{i}" for i in range(consumers)]
    m = ProtocolModel(depth=stages, consumers=consumer_names)

    for it in range(iters):
        buf = it % stages

        # Reclaim the buffer from the previous trip through the ring.
        if it >= stages:
            m.barrier_arm(EMPTY, arrivals=consumers)
            for name in consumer_names:
                m.consumer_release(buf, name, EMPTY)

        # ---- produce -------------------------------------------------------
        m.producer_acquire(buf, 'producer')
        # sec 8.3: an all-false stage still issues its commands and still
        # counts their completions. Skipping them desynchronises the barrier.
        m.barrier_arm(FILL, arrivals=1, ops=cmds_per_stage)
        cmds = [m.issue_async(buf, 'producer', FILL, kind='dma')
                for _ in range(cmds_per_stage)]
        for k in order_fn(len(cmds)):
            m.complete(cmds[k])
        m.publish(buf, 'producer', FILL)

        # ---- consume -------------------------------------------------------
        for name in consumer_names:
            m.consumer_acquire(buf, name)
            if all_false:
                pass  # the reads are predicated off; the release still happens

    # Drain the ring.
    live = min(iters, stages)
    for k in range(live):
        buf = (iters - live + k) % stages
        m.barrier_arm(EMPTY, arrivals=consumers)
        for name in consumer_names:
            m.consumer_release(buf, name, EMPTY)

    m.check_quiescent()
    return m


class TestEnumeration(unittest.TestCase):
    """The grid sec 16.2 names."""

    def test_stages_consumers_and_completion_orders(self):
        checked = 0
        for stages, consumers, iters, cmds in itertools.product(
                (1, 2, 3), (1, 2), (1, 2, 5), (1, 2, 3)):
            for order in completion_orders(cmds):
                with self.subTest(stages=stages, consumers=consumers,
                                  iters=iters, cmds=cmds, order=order):
                    run_pipeline(stages=stages, consumers=consumers,
                                 iters=iters, cmds_per_stage=cmds,
                                 order_fn=lambda n, o=order: o)
                    checked += 1
        # 3 stages x 2 consumers x 3 loop lengths x (1!+2!+3!) orders
        self.assertEqual(checked, 3 * 2 * 3 * (1 + 2 + 6))

    def test_all_false_still_completes(self):
        """sec 8.3: predication off must not change the completion count."""
        for stages, consumers in itertools.product((1, 2, 3), (1, 2)):
            with self.subTest(stages=stages, consumers=consumers):
                run_pipeline(stages=stages, consumers=consumers, iters=3,
                             order_fn=lambda n: range(n), all_false=True)

    def test_short_loops(self):
        """Fewer iterations than stages: the ring never fills."""
        for iters in (1, 2):
            with self.subTest(iters=iters):
                run_pipeline(stages=3, consumers=2, iters=iters,
                             order_fn=lambda n: range(n))


class TestFaults(unittest.TestCase):
    """Each case is something a wrong lowering would emit."""

    def test_read_before_publish(self):
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        with self.assertRaises(ProtocolError) as cm:
            m.consumer_acquire(0, 'c0')
        self.assertIn('never published', str(cm.exception))

    def test_publish_before_fill_completes(self):
        """sec 8.1: publish is not a mechanical translation of mb.arm."""
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        m.issue_async(0, 'producer', FILL)
        with self.assertRaises(ProtocolError) as cm:
            m.publish(0, 'producer', FILL)
        self.assertIn('still writing', str(cm.exception))

    def test_first_of_two_consumers_does_not_free_the_buffer(self):
        """sec 17's question: how two consumers jointly release."""
        m = ProtocolModel(depth=1, consumers=['c0', 'c1'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        cmd = m.issue_async(0, 'producer', FILL)
        m.complete(cmd)
        m.publish(0, 'producer', FILL)
        m.consumer_acquire(0, 'c0')
        m.consumer_acquire(0, 'c1')
        m.barrier_arm(EMPTY, arrivals=2)
        m.consumer_release(0, 'c0', EMPTY)
        self.assertEqual(m.buffers[0].state, BufferState.CONSUMING)
        with self.assertRaises(ProtocolError) as cm:
            m.producer_acquire(0, 'producer')
        self.assertIn('consuming', str(cm.exception))
        m.consumer_release(0, 'c1', EMPTY)
        self.assertEqual(m.buffers[0].state, BufferState.FREE)

    def test_refill_while_a_consumer_still_reads(self):
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        m.complete(m.issue_async(0, 'producer', FILL))
        m.publish(0, 'producer', FILL)
        m.consumer_acquire(0, 'c0')
        with self.assertRaises(ProtocolError):
            m.producer_acquire(0, 'producer')

    def test_operation_count_must_match_expansion(self):
        """sec 8.2: the count is taken after the command expands."""
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)   # armed for one...
        a = m.issue_async(0, 'producer', FILL)
        b = m.issue_async(0, 'producer', FILL)   # ...but expanded to two
        m.complete(a)
        with self.assertRaises(ProtocolError) as cm:
            m.complete(b)
        self.assertIn('after command expansion', str(cm.exception))

    def test_uncounted_completion_is_refused(self):
        """A command issued against an unarmed barrier loses its completion."""
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        with self.assertRaises(ProtocolError) as cm:
            m.issue_async(0, 'producer', FILL)
        self.assertIn('not armed', str(cm.exception))

    def test_wait_on_unarmed_barrier_would_sleep_forever(self):
        m = ProtocolModel(depth=1, consumers=['c0'])
        with self.assertRaises(ProtocolError) as cm:
            m.barrier_wait(FILL, 'c0')
        self.assertIn('sleep forever', str(cm.exception))

    def test_init_under_a_waiter_is_a_barrier_fault(self):
        """sec 13.3 / ISA sec 9A.6, the hardware rule t13 work introduced."""
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.barrier_arm(FILL, arrivals=1, ops=1)
        m.barrier_wait(FILL, 'c0')
        with self.assertRaises(ProtocolError) as cm:
            m.barrier_init(FILL)
        self.assertIn('BARRIER_FAULT', str(cm.exception))

    def test_unfinished_command_at_exit(self):
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        m.issue_async(0, 'producer', FILL)
        with self.assertRaises(ProtocolError) as cm:
            m.check_quiescent()
        self.assertIn('never completed', str(cm.exception))

    def test_source_release_refused_without_hardware_support(self):
        """sec 10.2 wants an earlier source-reusable event; v0.1 has none, so
        a scheduler must not assume one."""
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        cmd = m.issue_async(0, 'producer', FILL, kind='mma')
        with self.assertRaises(ProtocolError) as cm:
            m.source_release(cmd)
        self.assertIn('no separate source-release', str(cm.exception))

    def test_source_release_allowed_when_advertised(self):
        m = ProtocolModel(depth=1, consumers=['c0'],
                          separate_source_completion=True)
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        cmd = m.issue_async(0, 'producer', FILL, kind='mma')
        m.source_release(cmd)
        self.assertTrue(m.leases[cmd].src_done)
        self.assertFalse(m.leases[cmd].dest_done)


class TestBarrierCounting(unittest.TestCase):
    """sec 13.1: arrivals and operations are different counts."""

    def test_arrivals_alone_do_not_satisfy_an_operation_count(self):
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.barrier_arm(FILL, arrivals=1, ops=1)
        m.barrier_arrive(FILL, 'producer')
        self.assertFalse(m.barriers[FILL].satisfied)

    def test_operations_alone_do_not_satisfy_an_arrival_count(self):
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        m.complete(m.issue_async(0, 'producer', FILL))
        self.assertFalse(m.barriers[FILL].satisfied)

    def test_both_counts_release_the_waiter(self):
        m = ProtocolModel(depth=1, consumers=['c0'])
        m.producer_acquire(0, 'producer')
        m.barrier_arm(FILL, arrivals=1, ops=1)
        cmd = m.issue_async(0, 'producer', FILL)
        m.barrier_wait(FILL, 'c0')
        m.complete(cmd)
        # The operation count is met, the arrival count is not, so the waiter
        # must still be asleep: one counter could not tell these apart.
        self.assertEqual(m.barriers[FILL].waiters, ['c0'])
        m.barrier_arrive(FILL, 'producer')
        self.assertEqual(m.barriers[FILL].waiters, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
