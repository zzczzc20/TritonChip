"""Idle tracking for interactive chat, using only fake board observations."""
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'run2_board'))
import auto
import interactive_board as chat


class FakeBoard:
    def __init__(self):
        self.row = dict(uuid=chat.IMAGE_UUID, contexts=0, **{'outstanding execs': 0,
                        'total execs': 1, 'client pid': ''})
        self.clock = 100

    def probe(self, _sysfs):
        return self.row.copy()

    def clock_mhz(self, _sysfs):
        return self.clock

    idle = staticmethod(auto.idle)


class InteractiveIdleTests(unittest.TestCase):
    @mock.patch.object(chat, 'board_idle', return_value=False)
    def test_stale_shared_monitor_uses_private_continuous_interval(self, _monitor):
        board, tracker = FakeBoard(), auto.IdleTracker()
        for now in range(0, 900, 5):
            self.assertFalse(chat.idle_ready(board, tracker, None, None, now))
        self.assertTrue(chat.idle_ready(board, tracker, None, None, 900))
        board.row['total execs'] += 1
        self.assertFalse(chat.idle_ready(board, tracker, None, None, 905))

    @mock.patch.object(chat, 'board_idle', return_value=True)
    def test_shared_interval_still_requires_image_clock_and_idle(self, _monitor):
        board, tracker = FakeBoard(), auto.IdleTracker()
        self.assertTrue(chat.idle_ready(board, tracker, None, None, 0))
        board.clock = 90
        self.assertFalse(chat.idle_ready(board, tracker, None, None, 5))
        board.clock = 100
        board.row['client pid'] = '123'
        self.assertFalse(chat.idle_ready(board, tracker, None, None, 10))
        board.row['client pid'] = ''
        board.row['uuid'] = 'foreign'
        self.assertFalse(chat.idle_ready(board, tracker, None, None, 15))

    @mock.patch.object(chat, 'board_idle', return_value=False)
    def test_sampling_gap_restarts_interval(self, _monitor):
        board, tracker = FakeBoard(), auto.IdleTracker()
        self.assertFalse(chat.idle_ready(board, tracker, None, None, 0))
        self.assertFalse(chat.idle_ready(board, tracker, None, None, 901))


if __name__ == '__main__':
    unittest.main()
