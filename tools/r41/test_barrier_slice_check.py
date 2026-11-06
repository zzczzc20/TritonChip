"""The slice census must not turn missing evidence into a passing verdict."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECK = Path(__file__).with_name('barrier_slice_check.py')


class SliceEvidenceTests(unittest.TestCase):
    def test_missing_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            entry = {'execution': {'program_to_role': {'0': ['role0']}},
                     'resources': {'barriers': 1}, 'barrier_immediates': [0]}
            def check(*args):
                return subprocess.run([sys.executable, str(CHECK), *map(str, args)],
                                      capture_output=True, text=True)

            manifest.write_text(json.dumps({'kernel': entry}))
            self.assertEqual(check('--manifests', manifest).returncode, 0)
            (root / 'compile_only.json').write_text(json.dumps({'status': 'incomplete'}))
            self.assertNotEqual(check('--manifests', manifest).returncode, 0)
            (root / 'compile_only.json').unlink()
            entry['execution'] = {}
            manifest.write_text(json.dumps({'kernel': entry}))
            self.assertNotEqual(check('--manifests', manifest).returncode, 0)
            self.assertNotEqual(check().returncode, 0)
            report = root / 'report.json'
            kernel = {'launch_structure': {'execution': {'program_to_role': {'0': ['role0']}},
                                            'resources': {'barriers': 0}}}
            report.write_text(json.dumps({'points': {'point': {'kernels': {'kernel': kernel}}}}))
            self.assertNotEqual(check('--audit-report', report, '--audit-runs', root).returncode, 0)


if __name__ == '__main__':
    unittest.main()
