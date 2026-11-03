"""The performance and FPGA tools coordinate without accessing a board."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from tritonchip_fpga_harness.board_lease import board_lease, lease_path, shared_r41_device


class BoardLeaseTests(unittest.TestCase):
    def test_r41_reuses_existing_chat_queue_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.assertEqual(lease_path('0000:04:00.1', home), home / 'r41run3/candidate_queue.lock')
            with board_lease('0000:04:00.1', home=home):
                with self.assertRaises(BlockingIOError):
                    with board_lease('0000:04:00.1', home=home, blocking=False):
                        pass
            with board_lease('0000:04:00.1', home=home, blocking=False):
                pass

    def test_r41_direct_measurement_is_scoped_to_shared_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.assertFalse(shared_r41_device('u280', None, home))
            (home / 'r41run3').mkdir()
            self.assertTrue(shared_r41_device('u280', None, home))
            self.assertFalse(shared_r41_device('u280', '0000:05:00.1', home))
            self.assertFalse(shared_r41_device('u50', None, home))

    def test_direct_shared_board_stage_is_refused_before_any_command(self):
        from tritonchip_fpga_harness.measurement import MeasurementError, board_preflight, measure_board
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'run'
            (run / 'artifacts').mkdir(parents=True)
            (run / 'artifacts/image.xclbin').write_bytes(b'image')
            profile = SimpleNamespace(data={'board': 'u280', 'expected_artifacts': ['image.xclbin']})
            manifest = {'board': {'program': False, 'conformance': False}}
            with mock.patch('tritonchip_fpga_harness.measurement.shared_r41_device', return_value=True):
                checks = board_preflight(root=root, profile=profile, manifest=manifest)
                self.assertEqual([(c['name'], c['ok']) for c in checks], [('shared_r41_queue', False)])
                with self.assertRaisesRegex(MeasurementError, 'board-queue'):
                    measure_board(root=root, run_dir=run, profile=profile, manifest=manifest, timeout_s=1)

    def test_invalid_device_cannot_escape_lock_directory(self):
        with self.assertRaises(ValueError):
            lease_path('../bad')
