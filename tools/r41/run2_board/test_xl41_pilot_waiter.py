"""The XL pilot must not run on incomplete or changed Q41 evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import xl41_pilot_waiter as waiter


class AutoFixture:
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Q41Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = Path(self.temp.name)
        self.plan = {'pilot_script_sha256': 'a'*64}
        self.report_path = self.queue/'attempt-001'/'report.json'
        self.report_path.parent.mkdir()
        self.report = dict(status='pass', device='tnca', script_sha256='a'*64, point='Q41-core',
                           steps_requested=2, board=dict(image_sha256=waiter.prior.IMAGE_SHA,
                           uuid=waiter.prior.IMAGE_UUID, clock_hz=100_000_000),
                           steps=[dict(passed=True), dict(passed=True)])
        self.write_report()
        self.events = [dict(kind='pilot-end', attempt=1, returncode=0, status='pass',
                            report_sha256=AutoFixture.sha(self.report_path)), dict(kind='completed')]

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def test_passed_q41_is_accepted(self):
        self.assertTrue(waiter.verify_q41(self.queue, self.events, self.plan, AutoFixture))

    def test_incomplete_queue_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'not completed'):
            waiter.verify_q41(self.queue, self.events[:-1], self.plan, AutoFixture)

    def test_changed_report_is_refused(self):
        self.report_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            waiter.verify_q41(self.queue, self.events, self.plan, AutoFixture)

    def test_cpu_result_or_failed_step_is_refused(self):
        self.report['device'] = 'cpu'
        self.write_report()
        self.events[0]['report_sha256'] = AutoFixture.sha(self.report_path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_q41(self.queue, self.events, self.plan, AutoFixture)
        self.report['device'] = 'tnca'
        self.report['steps'][1]['passed'] = False
        self.write_report()
        self.events[0]['report_sha256'] = AutoFixture.sha(self.report_path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_q41(self.queue, self.events, self.plan, AutoFixture)

    def test_wrong_image_is_refused(self):
        self.report['board']['uuid'] = '00000000-0000-0000-0000-000000000000'
        self.write_report()
        self.events[0]['report_sha256'] = AutoFixture.sha(self.report_path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_q41(self.queue, self.events, self.plan, AutoFixture)


if __name__ == '__main__':
    unittest.main()
