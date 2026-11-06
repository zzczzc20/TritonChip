#!/usr/bin/env python3
"""CPU-only tests of the deadline queue's shared-card handoff."""
import datetime
import fcntl
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import deadline_baseline_queue as queue


class HandoffTests(unittest.TestCase):
    def test_idle_handoff_keeps_existing_lease(self):
        row = dict(uuid=queue.UUID, contexts=0, **{'outstanding execs': 0, 'total execs': 3, 'client pid': ''})
        auto = SimpleNamespace(probe=mock.Mock(return_value=row), idle=lambda x: x['contexts'] == 0,
                               clock_mhz=lambda _: 100)
        with tempfile.TemporaryFile() as lock, mock.patch.object(queue, 'record') as record:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(queue.wait_for_handoff(auto, Path('/unused'), lock, 'q35'), row)
            record.assert_not_called()

    def test_busy_handoff_releases_lock_and_waits_for_new_idle_interval(self):
        idle = dict(uuid=queue.UUID, contexts=0, **{'outstanding execs': 0, 'total execs': 8, 'client pid': ''})
        busy = dict(idle, contexts=1)
        tracker = SimpleNamespace(observe=mock.Mock(side_effect=[False, True]))
        auto = SimpleNamespace(probe=mock.Mock(side_effect=[busy, idle, idle, idle]),
                               idle=lambda x: x['contexts'] == 0,
                               clock_mhz=lambda _: 100, IdleTracker=lambda **_: tracker)
        stop = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=1)
        with tempfile.TemporaryFile() as lock, mock.patch.object(queue, 'record') as record, \
                mock.patch.object(queue.time, 'sleep'), mock.patch.object(queue, 'STOP', stop):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(queue.wait_for_handoff(auto, Path('/unused'), lock, 'q35'), idle)
            self.assertEqual(tracker.observe.call_count, 2)
            record.assert_called_once_with('waiting-card', model='q35', board=busy)

    def test_foreign_image_never_becomes_a_valid_handoff(self):
        foreign = dict(uuid='foreign', contexts=0, **{'outstanding execs': 0, 'total execs': 8, 'client pid': ''})
        tracker = SimpleNamespace(observe=mock.Mock(return_value=True))
        auto = SimpleNamespace(probe=mock.Mock(return_value=foreign), idle=lambda _: True,
                               clock_mhz=lambda _: 100, IdleTracker=lambda **_: tracker)
        with tempfile.TemporaryFile() as lock, mock.patch.object(queue, 'record'), \
                mock.patch.object(queue.time, 'sleep', side_effect=RuntimeError('stop test')), \
                mock.patch.object(queue, 'STOP', datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=1)):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'stop test'):
                queue.wait_for_handoff(auto, Path('/unused'), lock, 'q35')


if __name__ == '__main__':
    unittest.main()
