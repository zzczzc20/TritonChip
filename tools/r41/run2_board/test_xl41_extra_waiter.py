"""Extra XL pilots require complete matrix evidence and preserve scoped checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import xl41_extra_waiter as waiter


class AutoFixture:
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExtraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = Path(self.temp.name)
        self.plan = dict(pilot_script_sha256='a'*64, stages=[dict(id='transpose', point='XL41-transpose', point_sha256='1'*64),
                         dict(id='gemm', point='XL41-gemm', point_sha256='2'*64),
                         dict(id='projection', point='XL41-projection', point_sha256='3'*64)])
        self.events = []
        for stage in self.plan['stages']:
            path = self.queue/f"{stage['id']}-attempt-001"/'report.json'
            path.parent.mkdir()
            path.write_text(json.dumps(dict(status='pass', device='tnca', script_sha256='a'*64,
                point=stage['point'], point_sha256=stage['point_sha256'], checks=dict(finite=True, mismatches=0,
                checked_elements=waiter.COUNTS[stage['id']]), board=dict(image_sha256=waiter.common.IMAGE_SHA,
                uuid=waiter.common.IMAGE_UUID, clock_hz=100_000_000))))
            self.events.append(dict(kind='stage-end', stage=stage['id'], attempt=1, returncode=0,
                                    status='pass', report_sha256=AutoFixture.sha(path)))
        self.events.append(dict(kind='completed'))

    def test_verified_matrix_queue(self):
        self.assertTrue(waiter.verify_matrix(self.queue, self.events, self.plan, AutoFixture))

    def test_incomplete_and_missing_stage_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'not completed'):
            waiter.verify_matrix(self.queue, self.events[:-1], self.plan, AutoFixture)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            waiter.verify_matrix(self.queue, self.events[:2]+self.events[-1:], self.plan, AutoFixture)

    def test_changed_or_incomplete_report_is_refused(self):
        path = self.queue/'gemm-attempt-001/report.json'
        report = json.loads(path.read_text())
        report['checks']['checked_elements'] = 1
        path.write_text(json.dumps(report))
        self.events[1]['report_sha256'] = AutoFixture.sha(path)
        with self.assertRaisesRegex(ValueError, 'identity or checks'):
            waiter.verify_matrix(self.queue, self.events, self.plan, AutoFixture)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            waiter.verify_matrix(self.queue, self.events, self.plan, AutoFixture)

    def test_gather_scope_must_remain_explicit(self):
        r = dict(formal_single_vector_unresolved=True, formula_missing_required_rows=[0,32767,65535],
                 runs=[dict(name='formula',count=8193,passed=True),dict(name='boundary',count=4,passed=True)])
        self.assertTrue(waiter.valid_extra('gather', r))
        r['formal_single_vector_unresolved'] = False
        self.assertFalse(waiter.valid_extra('gather', r))

    def test_attention_and_recurrence_checks(self):
        self.assertTrue(waiter.valid_extra('attention', dict(tail_zero=True,checks=dict(mismatches=0))))
        self.assertFalse(waiter.valid_extra('attention', dict(tail_zero=False,checks=dict(mismatches=0))))
        r = dict(formal_update_rule_unresolved=True, updates=[dict(passed=True,chunks=[32,32,1])]*4)
        self.assertTrue(waiter.valid_extra('recurrence', r))
        r['formal_update_rule_unresolved'] = False
        self.assertFalse(waiter.valid_extra('recurrence', r))


if __name__ == '__main__':
    unittest.main()
