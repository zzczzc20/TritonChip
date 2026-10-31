"""The stream pilot refuses incomplete or changed XL GEMV evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import xl41_stream_waiter as waiter


class AutoFixture:
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class XLTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = Path(self.temp.name)
        self.plan = {'pilot_script_sha256': 'a'*64, 'point_sha256': 'b'*64}
        self.report_path = self.queue/'attempt-001'/'report.json'
        self.report_path.parent.mkdir()
        self.values = self.report_path.parent/'values.safetensors'
        self.values.write_bytes(b'fixture')
        self.report = dict(status='pass', device='tnca', script_sha256='a'*64, point='XL41-gemv',
                           point_sha256='b'*64, finite=True, mismatches=0,
                           output_values_sha256=AutoFixture.sha(self.values),
                           board=dict(image_sha256=waiter.prior.IMAGE_SHA,
                                      uuid=waiter.prior.IMAGE_UUID, clock_hz=100_000_000))
        self.write_report()
        self.events = [dict(kind='pilot-end', attempt=1, returncode=0, status='pass',
                            report_sha256=AutoFixture.sha(self.report_path)), dict(kind='completed')]

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def test_passed_gemv_is_accepted(self):
        self.assertTrue(waiter.verify_xl(self.queue, self.events, self.plan, AutoFixture))

    def test_incomplete_queue_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'not completed'):
            waiter.verify_xl(self.queue, self.events[:-1], self.plan, AutoFixture)

    def test_changed_report_or_values_are_refused(self):
        self.values.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_xl(self.queue, self.events, self.plan, AutoFixture)
        self.report_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            waiter.verify_xl(self.queue, self.events, self.plan, AutoFixture)

    def test_cpu_result_or_numeric_failure_is_refused(self):
        self.report['device'] = 'cpu'
        self.write_report()
        self.events[0]['report_sha256'] = AutoFixture.sha(self.report_path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_xl(self.queue, self.events, self.plan, AutoFixture)
        self.report['device'] = 'tnca'
        self.report['mismatches'] = 1
        self.write_report()
        self.events[0]['report_sha256'] = AutoFixture.sha(self.report_path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_xl(self.queue, self.events, self.plan, AutoFixture)

    def test_wrong_image_is_refused(self):
        self.report['board']['uuid'] = '00000000-0000-0000-0000-000000000000'
        self.write_report()
        self.events[0]['report_sha256'] = AutoFixture.sha(self.report_path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_xl(self.queue, self.events, self.plan, AutoFixture)


if __name__ == '__main__':
    unittest.main()
