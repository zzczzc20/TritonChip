#!/usr/bin/env python3
"""Read consistency and unavailable-image checks for the MMIO clock."""
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.backend import BackendError
from tnca.backends.u50 import U50Backend
from tnca.errors import DeviceLost


class ClockTests(unittest.TestCase):
    def backend(self, read):
        backend = object.__new__(U50Backend)
        backend.link = type('Link', (), {'read_reg': staticmethod(read)})()
        backend._io_lock = threading.RLock()
        backend.epoch = 3
        return backend

    def test_old_image_is_unavailable(self):
        self.assertIsNone(self.backend(lambda offset: 0).launch_timestamps(1))

    def test_stable_endpoints_and_epoch(self):
        regs = {0xE0: 1, 0xE8: 91, 0xF0: 140, 0xF8: 7}
        got = self.backend(regs.__getitem__).launch_timestamps(7)
        self.assertEqual(got, dict(start=91, end=140, sequence=7, epoch=3,
                                  clock='mmio-device-clock/1'))

    def test_stale_or_in_progress_sequence_refused(self):
        for sequence in (0, 6):
            regs = {0xE0: 1, 0xE8: 91, 0xF0: 140, 0xF8: sequence}
            with self.assertRaises(BackendError):
                self.backend(regs.__getitem__).launch_timestamps(7)

    def test_torn_read_retried(self):
        seq = iter((6, 7, 7, 7))
        regs = {0xE0: 1, 0xE8: 91, 0xF0: 140}
        def read(offset):
            return next(seq) if offset == 0xF8 else regs[offset]
        self.assertEqual(self.backend(read).launch_timestamps(7)['sequence'], 7)

    def test_unstable_read_reports_device_loss(self):
        seq = iter(range(16))
        def read(offset):
            return next(seq) if offset == 0xF8 else 1
        with self.assertRaises(DeviceLost):
            self.backend(read).launch_timestamps(7)

    def test_unknown_version_refused(self):
        with self.assertRaises(BackendError):
            self.backend(lambda offset: 2).launch_timestamps(7)


if __name__ == '__main__':
    unittest.main()
