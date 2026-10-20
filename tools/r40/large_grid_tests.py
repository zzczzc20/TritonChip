#!/usr/bin/env python3
"""Negotiated large grids preserve legacy packets and reject old images."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.transport import Transport, Reply, OP_GRID_CAP, OP_LAUNCH, OP_LAUNCH32
from tnca.runtime import Session
from tnca.errors import AdmissionError, DeviceLost


class GridTests(unittest.TestCase):
    def transport(self, reply):
        t = Transport.__new__(Transport)
        t._io_lock = threading.RLock()
        t.request = Mock(return_value=reply)
        return t

    def test_legacy_packet_is_unchanged_without_cap_query(self):
        t = self.transport(Reply(1, 0, 0, 0))
        t.launch(8, 16, 3, pid=(2, 3, 4), npid=(5, 6, 7))
        t.request.assert_called_once_with(OP_LAUNCH, 8, 16, 3,
                                         2 | (3 << 16), 4 | (5 << 16), 6 | (7 << 16), timeout=None)

    def test_all_six_wide_axes_survive_packing(self):
        t = self.transport(Reply(1, 0, 32, 1 << 24))
        pid = (65536, 65537, 65538)
        npid = (65539, 65540, 65541)
        t.launch(8, 16, 3, pid=pid, npid=npid)
        t.request.assert_any_call(OP_GRID_CAP)
        t.request.assert_called_with(OP_LAUNCH32, 8, 16, 3,
                                     pid[0] | (pid[1] << 32), pid[2] | (npid[0] << 32),
                                     npid[1] | (npid[2] << 32), timeout=None)
        self.assertEqual(t.large_grid_limit(), 1 << 24)
        self.assertEqual(sum(c.args[0] == OP_GRID_CAP for c in t.request.call_args_list), 1)

    def test_old_image_cannot_silently_truncate_a_launch(self):
        t = self.transport(Reply(1, 3, 0, 0))
        with self.assertRaises(ValueError):
            t.launch(8, 16, npid=(65537, 1, 1))
        t.request.assert_called_once_with(OP_GRID_CAP)

    def test_invalid_capability_or_axis_is_refused(self):
        t = self.transport(Reply(1, 0, 16, 1 << 24))
        with self.assertRaises(DeviceLost):
            t.large_grid_limit()
        t = self.transport(Reply(1, 0, 32, 1 << 24))
        for v in (-1, 1 << 32, 1.5):
            with self.assertRaises(ValueError):
                t.launch(8, 16, npid=(v, 1, 1))
        t.request.assert_not_called()

    def test_admission_requires_opt_in_and_a_matching_image(self):
        s = Session.__new__(Session)
        s.closed = True  # no device lifecycle for this admission-only fixture
        s._check_usable = lambda: None
        s._lock = threading.RLock()
        s._events, s.epoch = {}, 0
        s.dev = SimpleNamespace(large_grid_limit=Mock(return_value=1 << 24))
        s._resolve = Mock(side_effect=RuntimeError('reached argument validation'))
        m = SimpleNamespace(epoch=0, alive=True, name='grid',
                            manifest={'identity': {'kernel_name': 'grid'}})
        s.large_grid = False
        s.physical_grid = False
        with self.assertRaises(AdmissionError):
            s.submit(m, {}, grid=(6144, 1, 1))
        s.dev.large_grid_limit.assert_not_called()
        s.large_grid = True
        for grid in ((6144, 1, 1), (65537, 1, 1), (64, 64, 2)):
            with self.assertRaisesRegex(RuntimeError, 'reached argument validation'):
                s.submit(m, {}, grid=grid)
        s.dev.large_grid_limit.return_value = None
        with self.assertRaises(AdmissionError):
            s.submit(m, {}, grid=(6144, 1, 1))
        s.dev.large_grid_limit.return_value = 1 << 24
        with self.assertRaises(AdmissionError):
            s.submit(m, {}, grid=((1 << 24) + 1, 1, 1))
        s.poisoned = False
        s.dev.large_grid_limit.side_effect = DeviceLost('invalid capability reply')
        s._resolve.reset_mock()
        with self.assertRaises(DeviceLost):
            s.submit(m, {}, grid=(6144, 1, 1))
        self.assertTrue(s.poisoned)
        s._resolve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
