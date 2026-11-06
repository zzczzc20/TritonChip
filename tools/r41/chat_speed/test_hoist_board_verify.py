#!/usr/bin/env python3
"""Exercise the board evidence validator on recorded session-shaped reports."""
import copy
import json
from pathlib import Path
import unittest
import verify_hoist_board as verifier


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        cls.recorded = json.loads((root / 'evidence/r41/chat/run3_priority/q35_nohost_v6_session.json').read_text())
        cls.recorded['fp8_route_in_use'] = {'kind': 'experimental-tnca-hoist'}
        cls.recorded['steps'][0]['transfers_inside_step']['by_kernel']['_precise_block_quant'] = {'launches': 1}

    def test_complete_evidence(self):
        self.assertEqual(verifier.check(copy.deepcopy(self.recorded)), ([], 1))

    def test_cpu_parameter_and_decode_read_are_rejected(self):
        row = copy.deepcopy(self.recorded)
        row['model_tensor_placements']['parameters']['cpu'] = 1
        row['steps'][1]['transfers_inside_step']['host_read_calls'] = 1
        errors, _ = verifier.check(row)
        self.assertTrue(any('parameters placement' in e for e in errors))
        self.assertTrue(any('decode host reads' in e for e in errors))

    def test_missing_wrapper_or_unexecuted_hoist_is_rejected(self):
        row = copy.deepcopy(self.recorded)
        row['persistent_session_check']['repeat']['compiled_wrapper_fallbacks'] = {}
        del row['steps'][0]['transfers_inside_step']['by_kernel']['_precise_block_quant']
        errors, _ = verifier.check(row)
        self.assertTrue(any('compiled wrapper evidence' in e for e in errors))
        self.assertTrue(any('did not run' in e for e in errors))


if __name__ == '__main__':
    unittest.main()
