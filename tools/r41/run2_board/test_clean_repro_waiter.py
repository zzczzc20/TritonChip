import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import clean_repro_waiter as waiter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReproductionTests(unittest.TestCase):
    def test_predecessor_requires_each_hashed_passing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = []
            for name in ('gather', 'attention', 'recurrence'):
                path = root/(name+'-attempt-001')/'report.json'
                path.parent.mkdir()
                path.write_text('{"status":"pass"}')
                events.append(dict(kind='stage-end',stage=name,attempt=1,returncode=0,
                                   status='pass',report_sha256=sha(path)))
            def save():
                (root/'queue.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
            save()
            self.assertFalse(waiter.verify_extra(root, sha))
            events.append(dict(kind='completed'))
            save()
            self.assertTrue(waiter.verify_extra(root, sha))
            (root/'gather-attempt-001/report.json').write_text('{"status":"fail"}')
            with self.assertRaises(ValueError):
                waiter.verify_extra(root, sha)

    def test_budget_uses_verified_board_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'attempt-001/report.json'
            path.parent.mkdir()
            path.write_text(json.dumps(dict(status='pass',script_sha256='script',point='Q41-core',
                steps=[dict(passed=True,wall_seconds=100),dict(passed=True,wall_seconds=50)])))
            event = dict(kind='pilot-end',attempt=1,returncode=0,status='pass',report_sha256=sha(path))
            (root/'queue.jsonl').write_text(json.dumps(event)+'\n')
            policy = dict(minimum_seconds=1800,fixed_seconds=1200,prefill_multiplier=2,decode_multiplier=2)
            budget, digest = waiter.pilot_budget(root,sha,policy,'script')
            self.assertEqual(budget,3000)
            self.assertEqual(digest,sha(path))
            path.write_text('{}')
            with self.assertRaises(ValueError):
                waiter.pilot_budget(root,sha,policy,'script')


if __name__ == '__main__':
    unittest.main()
