"""Unit tests for the R41 holdout runner."""
import json
from pathlib import Path
import tempfile
import unittest

from holdout_runner import verify_holdout_spec, run_holdout, digest_point

ROOT = Path(__file__).resolve().parents[2]


class HoldoutRunnerTests(unittest.TestCase):
    def test_official_spec_verifies(self):
        spec = verify_holdout_spec(ROOT / 'spec/r41/holdout.json')
        self.assertEqual(len(spec['points']), 4)
        ids = [p['id'] for p in spec['points']]
        self.assertEqual(ids, ['H41-mm', 'H41-linear', 'H41-softmax', 'H41-rmsnorm'])

    def test_corrupted_hash_refused(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            doc = json.loads((ROOT / 'spec/r41/holdout.json').read_text())
            doc['points'][0]['rtol'] = 0.5
            f.write(json.dumps(doc))
            f.flush()
            with self.assertRaises(ValueError):
                verify_holdout_spec(f.name)

    def test_missing_point_refused(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            doc = json.loads((ROOT / 'spec/r41/holdout.json').read_text())
            doc['points'].pop()
            f.write(json.dumps(doc))
            f.flush()
            with self.assertRaises(ValueError):
                verify_holdout_spec(f.name)

    def test_check_only_mode(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json') as out:
            report = run_holdout(ROOT / 'spec/r41/holdout.json', out_path=out.name, check_only=True)
            self.assertEqual(report['status'], 'verified-only')
            self.assertEqual(report['points'], 4)
            data = json.loads(Path(out.name).read_text())
            self.assertEqual(data['status'], 'verified-only')


if __name__ == '__main__':
    unittest.main()
