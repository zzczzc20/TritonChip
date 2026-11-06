import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import grid_rerun_waiter as waiter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class UpstreamTests(unittest.TestCase):
    def test_requires_all_stages_and_a_hashed_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = []
            for name in waiter.ORDER:
                path = root/name/'report.json'
                path.parent.mkdir()
                path.write_text('{}')
                events.append(dict(kind='stage-end', stage=name, status='pass', returncode=0,
                                   report_sha256=sha(path)))
            clock = root/'run1_clock_verify.json'
            clock.write_text('{"status":"pass"}')
            events.append(dict(kind='run1-clock-verified', report_sha256=sha(clock)))
            def save():
                (root/'queue.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            save()
            self.assertFalse(waiter.upstream_done(root, sha))
            events.append(dict(kind='completed'))
            save()
            self.assertTrue(waiter.upstream_done(root, sha))
            clock.write_text('{"status":"fail"}')
            with self.assertRaises(ValueError):
                waiter.upstream_done(root, sha)
            clock.write_text('{"status":"pass"}')
            events[0]['returncode'] = 1
            save()
            with self.assertRaises(ValueError):
                waiter.upstream_done(root, sha)


if __name__ == '__main__':
    unittest.main()
