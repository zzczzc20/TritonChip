"""Verify a completed campaign cannot be replaced by an incomplete one."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import q41_pilot_waiter as waiter


class AutoFixture:
    REQUIRED = ('identity', 'original113')

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class UpstreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'root'
        self.queue = self.base/'queue'
        self.events = []
        for stage in AutoFixture.REQUIRED:
            reused = stage == 'identity'
            target = (self.base/'prior'/stage if reused else self.queue/'results'/stage)
            experiment = stage+'-experiment'
            (target/experiment).mkdir(parents=True)
            report = target/experiment/'report.json'
            report.write_text('{"status":"completed-experiment"}')
            check = target/experiment/'check.json'
            check.write_text('{"status":"complete"}')
            manifest = self.root/'manifests'/f'{stage}.json'
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text('{"id":"'+experiment+'"}')
            reg = dict(id=stage, image_identity=dict(sha256=waiter.IMAGE_SHA), experiments=[dict(
                id=experiment, manifest=str(manifest.relative_to(self.root)), sha256=AutoFixture.sha(manifest),
                checks=[dict(path='check.json')])])
            regpath = self.root/'spec/r41/run3/stages'/f'{stage}.json'
            regpath.parent.mkdir(parents=True, exist_ok=True)
            regpath.write_text(json.dumps(reg))
            summary = target/'stage.json'
            summary.write_text(json.dumps(dict(status='pass', experiments=[dict(id=experiment,
                manifest_sha256=AutoFixture.sha(manifest), report_sha256=AutoFixture.sha(report),
                checks=[dict(path=str(check), sha256=AutoFixture.sha(check))])])))
            if reused:
                self.events.append(dict(kind='stage-reused', stage=stage, prior=str(summary), sha256=AutoFixture.sha(summary)))
            else:
                self.events.append(dict(kind='stage-end', stage=stage, returncode=0))
        self.events.append(dict(kind='completed'))

    def test_verified_campaign_is_accepted(self):
        self.assertTrue(waiter.verify_upstream(self.root, self.queue, self.events, AutoFixture))

    def test_incomplete_or_wrong_stage_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'not completed'):
            waiter.verify_upstream(self.root, self.queue, self.events[:-1], AutoFixture)
        events = self.events[:1]+[self.events[-1]]
        with self.assertRaisesRegex(ValueError, 'inventory'):
            waiter.verify_upstream(self.root, self.queue, events, AutoFixture)

    def test_changed_summary_or_record_is_refused(self):
        summary = Path(self.events[0]['prior'])
        summary.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'summary changed'):
            waiter.verify_upstream(self.root, self.queue, self.events, AutoFixture)

    def test_changed_formal_check_is_refused(self):
        check = self.queue/'results/original113/original113-experiment/check.json'
        check.write_text('{"status":"changed"}')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            waiter.verify_upstream(self.root, self.queue, self.events, AutoFixture)

    def test_partial_event_line_is_not_accepted(self):
        path = self.base/'queue.jsonl'
        path.write_text('{"kind":"stage-start"}\n{"kind":"completed"')
        self.assertEqual(waiter.read_events(path), [dict(kind='stage-start')])


if __name__ == '__main__':
    unittest.main()
