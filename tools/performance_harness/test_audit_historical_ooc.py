"""Historical OOC cannot qualify changed, missing or added RTL sources."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from audit_historical_ooc import audit

ROOT = Path(__file__).resolve().parents[2]


class HistoricalOocTests(unittest.TestCase):
    def test_actual_sources_and_mutation_sensitivity(self):
        reference = json.loads((ROOT/'spec/r41/stage1/historical_ooc_sources.json').read_text())
        self.assertEqual(audit(ROOT, reference)['status'], 'historical-source-equivalence-pass')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'rtl', root/'rtl')
            (root/'evidence').symlink_to(ROOT/'evidence', target_is_directory=True)
            target = root/'rtl/noc/tnca_idle_host_mux.v'
            saved = target.read_bytes()
            target.write_bytes(saved+b'\n// mutation probe\n')
            changed = audit(root, reference)
            self.assertEqual(changed['status'], 'refused-source-mismatch')
            self.assertTrue(any(r['reason'] == 'changed' for r in changed['mismatches']))
            target.unlink()
            self.assertTrue(any(r['reason'] == 'missing' for r in audit(root, reference)['mismatches']))
            target.write_bytes(saved)
            (root/'rtl/new_logic.v').write_text('module new_logic; endmodule\n')
            self.assertEqual(audit(root, reference)['extra_rtl_sources'], ['rtl/new_logic.v'])


if __name__ == '__main__':
    unittest.main()
