"""Full Q41 continuation rejects unverified pilots and bounds its timing."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import q41_full_waiter as waiter


class AutoFixture:
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = Path(self.temp.name)
        self.plan = {'pilot_script_sha256': 'a'*64, 'point_sha256': 'b'*64}
        self.path = self.queue/'attempt-001'/'report.json'
        self.path.parent.mkdir()
        self.report = dict(status='pass', device='tnca', script_sha256='a'*64, point='XL41-stream',
            point_sha256='b'*64, board=dict(image_sha256=waiter.q41.IMAGE_SHA,
            uuid=waiter.q41.IMAGE_UUID, clock_hz=100_000_000), checks=dict(finite=True, mismatches=0,
            checked_elements=1073741888), device_ranges=dict(input_bytes=2147483776, output_bytes=2147483776))
        self.write_report()
        self.events = [dict(kind='pilot-end', attempt=1, returncode=0, status='pass',
                            report_sha256=AutoFixture.sha(self.path)), dict(kind='completed')]
        self.policy = dict(prefill_multiplier=2, decode_multiplier=2, fixed_seconds=1200, minimum_seconds=1800)

    def write_report(self):
        self.path.write_text(json.dumps(self.report))

    def test_verified_stream_is_accepted(self):
        self.assertTrue(waiter.verify_stream(self.queue, self.events, self.plan, AutoFixture))

    def test_incomplete_or_changed_stream_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'not completed'):
            waiter.verify_stream(self.queue, self.events[:-1], self.plan, AutoFixture)
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            waiter.verify_stream(self.queue, self.events, self.plan, AutoFixture)

    def test_wrong_device_and_incomplete_check_are_refused(self):
        for key, value in [('device', 'cpu'), ('checks', dict(finite=True, mismatches=0, checked_elements=2))]:
            old = self.report[key]
            self.report[key] = value
            self.write_report()
            self.events[0]['report_sha256'] = AutoFixture.sha(self.path)
            with self.assertRaisesRegex(ValueError, 'identity or checks'):
                waiter.verify_stream(self.queue, self.events, self.plan, AutoFixture)
            self.report[key] = old

    def test_budget_uses_observed_times_and_predeclared_rule(self):
        report = dict(steps=[dict(wall_seconds=100), dict(wall_seconds=20)])
        self.assertEqual(waiter.budget_for(17, report, self.policy), 2040)
        self.assertEqual(waiter.budget_for(5, report, self.policy), 1800)

    def test_invalid_or_missing_timing_is_refused(self):
        for rows in ([], [dict(wall_seconds=1)], [dict(wall_seconds=1), dict(wall_seconds=float('nan'))]):
            with self.assertRaises(ValueError):
                waiter.budget_for(17, dict(steps=rows), self.policy)


if __name__ == '__main__':
    unittest.main()
