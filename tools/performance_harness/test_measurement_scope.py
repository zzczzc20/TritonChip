"""A wait timeout cannot become candidate cancellation or successful fallback."""
from collections import OrderedDict
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
from tnca.runtime import Session, Event, SUCCEEDED
from tnca.errors import ArgumentError, CapabilityError, DeviceLost, SessionBusy, Timeout


def session():
    s = Session.__new__(Session)
    s.closed, s.poisoned, s.epoch = False, False, 7
    s._measurement_owner = None
    s._lock = threading.RLock()
    s._events = OrderedDict()
    s._deadline, s._host_timeout = 2000000, 30
    s.dev = SimpleNamespace(deadline_cycles=2000000, host_timeout=30, bounded_launch_limits=True)
    return s


class MeasurementScopeTests(unittest.TestCase):
    def test_limits_restore_even_when_candidate_raises(self):
        s = session()
        with self.assertRaisesRegex(ValueError, 'candidate'):
            with s.measurement_scope(wall_seconds=.1, clock_hz=1000000):
                self.assertEqual(s.dev.deadline_cycles, 25000)
                self.assertEqual(s.dev.host_timeout, .05)
                raise ValueError('candidate failed')
        self.assertEqual((s.deadline_cycles, s.dev.deadline_cycles, s.dev.host_timeout),
                         (2000000, 2000000, 30))
        self.assertIsNone(s._measurement_owner)
        self.assertFalse(s.poisoned)

    def test_host_wait_expiry_is_followed_by_real_completion_drain(self):
        s = session()
        e = Event(1, s.epoch, 'delayed')
        with s.measurement_scope(wall_seconds=.2, clock_hz=1000000):
            s._events[1] = e
            with self.assertRaises(Timeout):
                s.wait(e, timeout=.001)
            self.assertFalse(e.terminal)
            timer = threading.Timer(.01, lambda: e._to(SUCCEEDED))
            timer.start()
        timer.join()
        self.assertTrue(e.ok)
        self.assertFalse(s.unfinished)
        self.assertFalse(s.poisoned)
        self.assertEqual(s.epoch, 7)

    def test_undrainable_work_poisoned_without_fabricating_a_verdict(self):
        s = session()
        e = Event(1, s.epoch, 'stuck')
        with self.assertRaises(DeviceLost):
            with s.measurement_scope(wall_seconds=.02, clock_hz=1000000):
                s._events[1] = e
        self.assertTrue(s.poisoned)
        self.assertFalse(e.terminal)
        self.assertEqual(s.epoch, 7)
        self.assertIsNone(s._measurement_owner)

    def test_busy_and_unsupported_sessions_are_refused(self):
        s = session()
        s._events[1] = Event(1, s.epoch, 'existing')
        with self.assertRaises(SessionBusy), s.measurement_scope(wall_seconds=1, clock_hz=1):
            self.fail('an existing task entered measurement scope')
        s._events.clear()
        s.dev.bounded_launch_limits = False
        with self.assertRaises(CapabilityError), s.measurement_scope(wall_seconds=1, clock_hz=1):
            self.fail('an unsupported backend entered measurement scope')


if __name__ == '__main__':
    unittest.main()
