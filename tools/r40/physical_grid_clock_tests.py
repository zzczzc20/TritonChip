#!/usr/bin/env python3
"""Physical-grid Event clock propagation and unavailable-clock semantics."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from tnca.runtime import Session, SUCCEEDED, _active_core_mask
from tnca.transport import Reply


class GridClockTests(unittest.TestCase):
    def test_frozen_active_core_count_is_not_a_transport_mask(self):
        for count, mask in ((1, 1), (2, 3), (3, 7)):
            with patch.dict('os.environ', TNCA_ACTIVE_CORES=str(count)):
                self.assertEqual(_active_core_mask(True, None), mask)
                self.assertEqual(_active_core_mask(True, mask), mask)
                with self.assertRaisesRegex(ValueError, 'conflicts'):
                    _active_core_mask(True, 16)
                with self.assertRaisesRegex(ValueError, 'physical-grid'):
                    _active_core_mask(False, None)
        for value in ('0', '17', 'bad'):
            with patch.dict('os.environ', TNCA_ACTIVE_CORES=value):
                with self.assertRaisesRegex(ValueError, 'count'):
                    _active_core_mask(True, None)

    def test_active_mask_above_actual_physical_capacity_is_refused(self):
        session = Session.__new__(Session)
        session.physical_grid = True
        session._requested_active_cores = 15
        session.dev = SimpleNamespace(physical_grid_capability=lambda: (3, 16777216),
                                     stop=lambda: None)
        with self.assertRaisesRegex(Exception, 'exceeds the physical grid'):
            session._negotiate_physical_grid()

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
