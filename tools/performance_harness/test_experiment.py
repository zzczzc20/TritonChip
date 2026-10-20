"""Evidence reuse must not hide failed, changed or timed-out experiments."""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiment import run


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'source').write_text('frozen source\n')
        self.manifest = self.root/'experiment.json'

    def spec(self, command, *, timeout=3, outputs=()):
        self.manifest.write_text(json.dumps(dict(
            id='proof-test', allowed_hosts=[socket.gethostname()], hypothesis='test actual completion',
            competing_explanations=['missing or stale evidence'], expected_observations=['bounded command'],
            sources=['source'], steps=[dict(id='command', kind='compile', argv=command,
                                           timeout_seconds=timeout, outputs=list(outputs))])))

    def execute(self):
        return run(self.manifest, self.root, self.root/'results', python=Path(sys.executable))

    def test_exact_resume_and_changed_output_refusal(self):
        code = ("from pathlib import Path; p=Path('count'); "
                "p.write_text(str(int(p.read_text())+1) if p.exists() else '1')")
        self.spec([sys.executable, '-c', code], outputs=['count'])
        with patch.dict(os.environ, {'UNRELATED_SECRET': 'must-not-enter-evidence'}):
            first = self.execute()
        second = self.execute()
        self.assertEqual(first['status'], 'completed-experiment')
        self.assertTrue(second['steps'][0]['reused'])
        self.assertEqual((self.root/'count').read_text(), '1')
        self.assertNotIn('UNRELATED_SECRET', first['steps'][0]['identity']['environment'])
        (self.root/'count').write_text('forged')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.execute()

    def test_failure_is_preserved_without_retry(self):
        self.spec([sys.executable, '-c', 'raise SystemExit(7)'])
        first = self.execute()
        self.assertEqual(first['status'], 'stopped-on-failure')
        second = self.execute()
        self.assertEqual(second['status'], 'stopped-on-preserved-failure')
        self.assertEqual(first['steps'][0]['attempt'], second['steps'][0]['attempt'])

    def test_timeout_terminates_the_actual_worker(self):
        self.spec([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.1)
        report = self.execute()
        row = report['steps'][0]
        self.assertEqual(row['status'], 'timeout')
        self.assertLess(row['returncode'], 0)
        self.assertFalse(Path('/proc', str(row['pid'])).exists())

    def test_missing_success_output_is_an_error(self):
        self.spec([sys.executable, '-c', 'pass'], outputs=['not-produced'])
        self.assertEqual(self.execute()['steps'][0]['status'], 'error')


if __name__ == '__main__':
    unittest.main()
