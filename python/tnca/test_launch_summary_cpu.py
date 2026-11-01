"""The low-overhead chat counters must match the full launch log."""
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import triton  # Discover backend entry points before importing the driver directly.
from tnca.triton_backend import driver


class LaunchSummaryTests(unittest.TestCase):
    def launcher(self):
        return driver.TNCALauncher(SimpleNamespace(signature={}, constants={}),
                                   SimpleNamespace(name='kernel', tnca_profile='dp', tnca_params=[]))

    def test_aggregate_matches_full_log_and_preserves_transfer_counts(self):
        session = SimpleNamespace(backend_name='fake', wait=lambda event: None)
        event = SimpleNamespace(ok=True, device_cycles=123, program_cycles=[123],
                                device_timing=None, counters=None,
                                history=[('RUNNING', 1), ('SUCCEEDED', 2)])
        moved = dict(upload_bytes=16, download_bytes=8, resident_bytes=4096)
        launcher = self.launcher()
        with mock.patch.object(driver, '_load', return_value=(session, object())), \
             mock.patch.object(driver._Sessions, 'get', return_value=session), \
             mock.patch.object(driver, '_guards', return_value=(0, [0])), \
             mock.patch.object(launcher, '_launch_once',
                               return_value=(event, {}, moved, (1., 2., 3.))), \
             mock.patch.dict(os.environ, {'TNCA_SUBMIT': 'batched', 'TNCA_LAUNCH': 'run'}):
            with driver.launch_log() as full, driver.launch_summary() as compact:
                launcher(1, 1, 1, None, None, None, None, None, None)
        self.assertEqual(len(full), 1)
        self.assertEqual(compact.result, dict(launches=1,
            by_kernel={'kernel': {'launches': 1, 'device_cycles': 123}},
            staging_upload_bytes=full[0]['upload_bytes'],
            staging_download_bytes=full[0]['download_bytes'],
            resident_bytes=full[0]['resident_bytes'], skipped_launches=0, device_cycles=123))

    def test_skipped_launch_does_not_look_like_measured_device_work(self):
        launcher = self.launcher()
        with mock.patch.dict(os.environ, {'TNCA_LAUNCH': 'skip'}):
            with driver.launch_log() as full, driver.launch_summary() as compact:
                launcher(1, 1, 1, None, None, None, None, None, None)
        self.assertTrue(full[0]['skipped'])
        self.assertEqual(compact.result['skipped_launches'], 1)
        self.assertIsNone(compact.result['device_cycles'])

    def test_pending_or_failed_events_are_waited_and_reported(self):
        class Event:
            ok = False
            error = 'device failed'
            device_cycles = None
        event = Event()
        session = SimpleNamespace(wait=mock.Mock())
        with self.assertRaisesRegex(driver.TNCALaunchError, 'device failed'):
            with driver.launch_summary() as compact:
                compact.record('broken', session, event, {})
        session.wait.assert_called_once_with(event)


if __name__ == '__main__':
    unittest.main()
