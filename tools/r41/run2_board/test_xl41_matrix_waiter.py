"""XL matrix queue rejects incomplete or altered Q41 full-session evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import xl41_matrix_waiter as waiter


class AutoFixture:
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Q41FullTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = Path(self.temp.name)
        self.plan = dict(pilot_script_sha256='a'*64, stages=[dict(id='core', point='Q41-core', steps=2),
                         dict(id='tail', point='Q41-tail', steps=1), dict(id='independent-core', point='Q41-core', steps=2)])
        self.events = []
        for stage in self.plan['stages']:
            path = self.queue/f"{stage['id']}-attempt-001"/'report.json'
            path.parent.mkdir()
            path.write_text(json.dumps(dict(status='pass', device='tnca', script_sha256='a'*64,
                point=stage['point'], steps_requested=stage['steps'], steps=[dict(passed=True)]*stage['steps'],
                board=dict(image_sha256=waiter.common.IMAGE_SHA, uuid=waiter.common.IMAGE_UUID, clock_hz=100_000_000))))
            self.events.append(dict(kind='stage-end', stage=stage['id'], attempt=1, returncode=0,
                                    status='pass', report_sha256=AutoFixture.sha(path)))
        self.events.append(dict(kind='completed'))

    def test_verified_full_queue(self):
        self.assertTrue(waiter.verify_q41_full(self.queue, self.events, self.plan, AutoFixture))

    def test_incomplete_and_missing_stage_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'not completed'):
            waiter.verify_q41_full(self.queue, self.events[:-1], self.plan, AutoFixture)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            waiter.verify_q41_full(self.queue, self.events[:2]+self.events[-1:], self.plan, AutoFixture)

    def test_changed_report_is_refused(self):
        (self.queue/'core-attempt-001/report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            waiter.verify_q41_full(self.queue, self.events, self.plan, AutoFixture)

    def test_cpu_or_failed_step_is_refused(self):
        path = self.queue/'tail-attempt-001/report.json'
        report = json.loads(path.read_text())
        for key, value in [('device', 'cpu'), ('steps', [dict(passed=False)])]:
            old = report[key]
            report[key] = value
            path.write_text(json.dumps(report))
            self.events[1]['report_sha256'] = AutoFixture.sha(path)
            with self.assertRaisesRegex(ValueError, 'identity or checks'):
                waiter.verify_q41_full(self.queue, self.events, self.plan, AutoFixture)
            report[key] = old

    def test_wrong_image_is_refused(self):
        path = self.queue/'core-attempt-001/report.json'
        report = json.loads(path.read_text())
        report['board']['uuid'] = '00000000-0000-0000-0000-000000000000'
        path.write_text(json.dumps(report))
        self.events[0]['report_sha256'] = AutoFixture.sha(path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_q41_full(self.queue, self.events, self.plan, AutoFixture)


if __name__ == '__main__':
    unittest.main()
