import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import perf_recheck_waiter as waiter


class Auto:
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PredecessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.events = []
        for name in ('legacy', 'hbm8', 'q41'):
            report = self.root/name/('report.json' if name == 'q41' else 'stage.json')
            report.parent.mkdir()
            data = dict(status='pass')
            if name == 'q41':
                data.update(device='tnca', point='Q41-core', steps=[dict(passed=True) for _ in range(17)])
            report.write_text(json.dumps(data))
            self.events.append(dict(kind='stage-end', stage=name, returncode=0,
                                    status='pass', report_sha256=Auto.sha(report)))

    def write(self, events):
        (self.root/'queue.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))

    def test_running_predecessor_is_not_complete(self):
        self.write(self.events)
        self.assertFalse(waiter.verify_predecessor(self.root, Auto))

    def test_complete_requires_all_passing_hashed_reports(self):
        self.write(self.events+[dict(kind='completed')])
        self.assertTrue(waiter.verify_predecessor(self.root, Auto))
        (self.root/'hbm8/stage.json').write_text('{"status":"fail"}')
        with self.assertRaises(ValueError):
            waiter.verify_predecessor(self.root, Auto)

    def test_incomplete_or_failed_predecessor_refused(self):
        self.write(self.events[:-1]+[dict(kind='completed')])
        with self.assertRaises(ValueError):
            waiter.verify_predecessor(self.root, Auto)
        self.events[0]['returncode'] = 1
        self.write(self.events+[dict(kind='completed')])
        with self.assertRaises(ValueError):
            waiter.verify_predecessor(self.root, Auto)


if __name__ == '__main__':
    unittest.main()
