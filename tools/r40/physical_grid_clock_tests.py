#!/usr/bin/env python3
"""Physical-grid Event clock propagation and unavailable-clock semantics."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from tnca.runtime import Session, SUCCEEDED
from tnca.transport import Reply


class GridClockTests(unittest.TestCase):
    def run_grid(self, stamp):
        reply = Reply(7, 0, 123, 1)
        if stamp is not None:
            reply.device_timing = stamp
        session = Session.__new__(Session)
        session.id = 91
        session.active_cores = 7
        session._lock = threading.RLock()
        session._counters = None
        session.dev = SimpleNamespace(launch_grid=lambda *a, **kw: reply,
                                     quiesce=lambda: Reply(8, 0, 0, 0))
        event = SimpleNamespace(_to=lambda state: None, device_timing=None)
        session._finish = lambda task, state: setattr(event, 'state', state)
        task = dict(event=event, module=SimpleNamespace(code_addr=0,
                    manifest={'execution': {'program_to_role': {'0': ['role0']}}}),
                    arg_addr=0x20000, grid=(17, 1, 1), arg_stride=256)
        session._run_physical_grid(task)
        return event

    def test_endpoints_retained_separately_from_execution_cycles(self):
        stamp = dict(clock='mmio-device-clock/1', epoch=4, start=100, end=310, sequence=7)
        event = self.run_grid(stamp)
        self.assertEqual(event.state, SUCCEEDED)
        self.assertEqual(event.device_cycles, (1 << 32) + 123)
        self.assertEqual(event.device_timing['cycles'], 210)
        self.assertEqual(event.device_timing['session_id'], 91)
        self.assertIsNone(event.program_cycles)
        self.assertNotIn('session_id', stamp)

    def test_old_backend_does_not_fabricate_clock(self):
        self.assertIsNone(self.run_grid(None).device_timing)

    def test_invalid_endpoints_refused(self):
        with self.assertRaises(ValueError):
            self.run_grid(dict(clock='mmio-device-clock/1', epoch=4, start=100, end=100))


if __name__ == '__main__':
    unittest.main()
