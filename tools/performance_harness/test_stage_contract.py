"""The freeze check writes one JSON record bound to the checked software tree."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


class FreezeRecordTests(unittest.TestCase):
    def test_record_is_valid_json_with_tree_identity(self):
        from archbench.provenance import software_tree
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'nested/freeze-check.json'
            run = subprocess.run([sys.executable, str(HERE/'stage_contract.py'), '--stage', '1', '--no-registry',
                                  '--out', str(out)], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            record = json.loads(out.read_text())
            self.assertEqual(json.loads(run.stdout), record)
        self.assertEqual(record['schema'], 'tnca-r41-freeze-check/2')
        self.assertEqual(record['status'], 'frozen-inputs-verified')
        self.assertEqual(record['software_tree'], software_tree(REPO, refresh=True))
        self.assertEqual(record['command'][1:], [str(HERE/'stage_contract.py'), '--stage', '1', '--no-registry',
                                                 '--out', str(out)])
        self.assertIs(record['registry_checked'], False)
        # None in a copy without .git (the board host runs an rsync'd tree).
        self.assertTrue(record['git'] is None or 'commit' in record['git'], record['git'])
        self.assertEqual(record['delivery_acceptance'], 'not-measured')

    def test_stage_2_is_not_frozen(self):
        run = subprocess.run([sys.executable, str(HERE/'stage_contract.py'), '--stage', '2', '--no-registry'],
                             cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('Stage 2 inputs have not been frozen', run.stderr)


if __name__ == '__main__':
    unittest.main()
