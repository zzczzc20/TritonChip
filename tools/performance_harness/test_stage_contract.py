"""The freeze check writes one JSON record bound to the checked software tree."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import stage_contract  # noqa: E402

FROZEN_INPUTS = REPO/'evidence/r41/stage1/frozen_inputs/inputs_25094197f6e6.tgz'
FROZEN_INPUTS_SHA256 = '0233c2dc2b311633e5cc05b08f68469e2c166538b366db4acaa0e55bf156dcfe'


class FreezeRecordTests(unittest.TestCase):
    def test_record_is_valid_json_with_tree_identity(self):
        from archbench.provenance import software_tree
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'frozen'
            root.mkdir()
            self.assertEqual(hashlib.sha256(FROZEN_INPUTS.read_bytes()).hexdigest(), FROZEN_INPUTS_SHA256)
            with tarfile.open(FROZEN_INPUTS) as archive:
                archive.extractall(root)
            out = Path(directory)/'nested/freeze-check.json'
            args = [str(HERE/'stage_contract.py'), '--stage', '1', '--no-registry', '--out', str(out)]
            original_record = stage_contract.record
            def record_on_frozen_inputs(**kwargs):
                return original_record(root=root, **kwargs)
            stdout = io.StringIO()
            with mock.patch.object(stage_contract, 'record', side_effect=record_on_frozen_inputs), \
                 mock.patch.object(sys, 'argv', args), contextlib.redirect_stdout(stdout):
                stage_contract.main()
            record = json.loads(out.read_text())
            self.assertEqual(json.loads(stdout.getvalue()), record)
            self.assertEqual(record['software_tree'], software_tree(root, refresh=True))
            # A changed frozen input must never be accepted as the original.
            (root/'archbench/registry.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                stage_contract.verify(root, stage=1, registry_check=False)
        self.assertEqual(record['schema'], 'tnca-r41-freeze-check/2')
        self.assertEqual(record['status'], 'frozen-inputs-verified')
        self.assertEqual(record['command'][1:], [str(HERE/'stage_contract.py'), '--stage', '1', '--no-registry',
                                                 '--out', str(out)])
        self.assertIs(record['registry_checked'], False)
        # None in a copy without .git (the board host runs an rsync'd tree).
        self.assertTrue(record['git'] is None or 'commit' in record['git'], record['git'])
        self.assertEqual(record['delivery_acceptance'], 'not-measured')

    def test_live_stage_1_rejects_changed_locked_inputs(self):
        lock = json.loads((REPO/'spec/r41/stage1/lock.json').read_text())
        changed = [name for name, digest in lock['files'].items()
                   if hashlib.sha256((REPO/name).read_bytes()).hexdigest() != digest]
        run = subprocess.run([sys.executable, str(HERE/'stage_contract.py'), '--stage', '1', '--no-registry'],
                             cwd=REPO, capture_output=True, text=True)
        if changed:
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('frozen input changed:', run.stderr)
        else:
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_stage_2_prospective_freeze_preserves_historical_failure(self):
        run = subprocess.run([sys.executable, str(HERE/'stage_contract.py'), '--stage', '2', '--no-registry'],
                             cwd=REPO, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result['freeze_kind'], 'prospective')
        self.assertTrue(result['historical_failure_preserved'])
        self.assertEqual(result['delivery_acceptance'], 'not-measured')


if __name__ == '__main__':
    unittest.main()
