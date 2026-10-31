"""The shared-card pilot refuses changed or occupied hardware before execution."""
import contextlib
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import qwen35_stage2_pilot as pilot

ROOT = Path(__file__).resolve().parents[2]


class BoardPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.image = self.base/'candidate.xclbin'
        self.image.write_bytes(b'fixture')
        self.clock = self.base/'clock.json'
        self.clock.write_text(json.dumps(dict(image_sha256=pilot.IMAGE_SHA, clock_hz=100_000_000)))
        self.sysfs = self.base/'sysfs'
        user = self.sysfs/'0000:04:00.1'
        user.mkdir(parents=True)
        (user/'xclbinuuid').write_text(pilot.IMAGE_UUID)
        self.kds = user/'kdsstat'
        self.kds.write_text('contexts: 0\noutstanding execs: 0\ntotal execs: 0\nclient pid:\n')
        counters = self.sysfs/'0000:04:00.0'/'clock_freq_counter.fixture'
        counters.mkdir(parents=True)
        self.frequency = counters/'clock_counter_freqs'
        self.frequency.write_text('100\n500\n430\n0\n')
        self.env = dict(TNCA_BACKEND='fpga-u280-multicore', TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp',
                        TNCA_OPERATING_CLOCK_EVIDENCE=str(self.clock),
                        TNCA_BACKEND_ARGS=json.dumps(dict(xclbin=str(self.image),
                            image_profile='core-vrf-sync-v5-x8-g2c4-dp', core_count=3,
                            memory_ports=3, hbm_bytes=8*(1 << 30), hbm_dma=True, device_clock=True)))
        original_sha = pilot.sha
        self.patcher = mock.patch.object(pilot, 'sha', side_effect=lambda p: pilot.IMAGE_SHA if p == self.image else original_sha(p))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def check(self):
        with contextlib.ExitStack() as guard:
            return pilot.board_preflight(ROOT, self.env, guard, self.sysfs)

    def test_idle_correct_image_and_clock(self):
        result = self.check()
        self.assertEqual(result['uuid'], pilot.IMAGE_UUID)
        self.assertEqual(result['clock_hz'], 100_000_000)

    def test_busy_card_and_clock_changes_are_refused(self):
        self.kds.write_text('contexts: 1\noutstanding execs: 0\ntotal execs: 1\nclient pid: 123\n')
        with self.assertRaisesRegex(RuntimeError, 'card busy'):
            self.check()
        self.kds.write_text('contexts: 0\noutstanding execs: 0\ntotal execs: 0\nclient pid:\n')
        self.frequency.write_text('90\n500\n430\n0\n')
        with self.assertRaisesRegex(RuntimeError, 'clock changed'):
            self.check()

    def test_changed_identity_and_backend_are_refused(self):
        (self.sysfs/'0000:04:00.1'/'xclbinuuid').write_text('00000000-0000-0000-0000-000000000000')
        with self.assertRaisesRegex(RuntimeError, 'image/clock changed'):
            self.check()
        self.env['TNCA_PROFILE'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'backend and profile'):
            self.check()

    def test_active_queue_lock_is_refused(self):
        with (self.base/'candidate_queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.check()

    def test_changed_image_and_clock_evidence_are_refused(self):
        with mock.patch.object(pilot, 'sha', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'image hash'):
                self.check()
        self.clock.write_text(json.dumps(dict(image_sha256=pilot.IMAGE_SHA, clock_hz=90_000_000)))
        with self.assertRaisesRegex(ValueError, 'clock evidence'):
            self.check()


if __name__ == '__main__':
    unittest.main()
