"""Capability-only X8 target must retain the physical banked-VRF model."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from tnca_cc.schedule import Model
class PhysicalModel(unittest.TestCase):
    def test_x8_uses_same_banked_model_as_v5(self):
        a=Model.for_target(SimpleNamespace(name='tnca-sim-vrf-v5'))
        b=Model.for_target(SimpleNamespace(name='tnca-sim-vrf-v5-x8'))
        self.assertEqual(b.name,'banked2')
        self.assertEqual((a.lat,a.issue,a.lsu),(b.lat,b.issue,b.lsu))
    def test_flipflop_default_remains_scoped_to_unmapped_targets(self):
        self.assertEqual(Model.for_target(SimpleNamespace(name='tnca-sim-v0.1')).name,'flipflop')
if __name__=='__main__':unittest.main()
