"""Prospective freeze integrity/refusal tests, independent of Torch."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import stage2_contract as contract
ROOT=Path(__file__).resolve().parents[2]
class FreezeTests(unittest.TestCase):
    def test_reviewed_prospective_lock_passes_without_claiming_delivery(self):
        result=contract.verify(ROOT,registry_check=False)
        self.assertEqual(result['status'],'frozen-inputs-verified')
        self.assertEqual(result['delivery_acceptance'],'not-measured')
        self.assertTrue(result['historical_failure_preserved'])
    def altered(self,name,change):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lock=json.loads((ROOT/'spec/r41/stage2/lock.json').read_text())
            for n in list(lock['files'])+['spec/r41/stage2/lock.json']:
                p=root/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
            path=root/name;x=json.loads(path.read_text());change(x);path.write_text(json.dumps(x))
            with self.assertRaises(ValueError):contract.verify(root,registry_check=False)
    def test_reduced_layer_count_refused(self):
        self.altered('spec/r41/stage2/model_lock.json',lambda x:x.update(layers=2))
    def test_relaxed_tolerance_refused(self):
        self.altered('spec/r41/stage2/model_lock.json',lambda x:x['numerics'].update(state_rel_l2=1))
    def test_truncated_xl_inventory_refused(self):
        self.altered('spec/r41/stage2/xl_core.json',lambda x:x['points'].pop())
    def test_model_predictions_cannot_supply_reference_cross_validation(self):
        self.altered('evidence/r41/acceptance/p0/cross-validation.json',lambda x:x.update(tnca_outputs_used=True))
if __name__=='__main__':unittest.main()
