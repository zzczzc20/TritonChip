#!/usr/bin/env python3
"""Replay actual placed reports and reject misleading physical diagnoses."""
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import resources as R

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'evidence/r40/v2/convergence'
SLR = EVIDENCE/'placement_followup/slr_util_placed_20261015.rpt'
FULL = EVIDENCE/'placement_followup/full_util_placed_20261015.rpt.gz'
LOG = EVIDENCE/'placement_followup/route_initial_20261015.log.gz'
MANIFEST = EVIDENCE/'candidate_manifest.json'


class ResourceEvidenceTests(unittest.TestCase):
    def test_slr_only_report_cannot_become_global_total(self):
        total, slr, crossings = R.parse_util(SLR.read_text())
        self.assertEqual(total, {})
        self.assertEqual(slr['SLR0']['lut'], 369666)
        self.assertEqual(sum(s['lut'] for s in slr.values()), 1090775)
        self.assertEqual(crossings['SLR1<->SLR0']['used'], 14423)

    def test_actual_full_report_and_optional_packing_resources(self):
        total, slr, _ = R.parse_util(gzip.open(FULL, 'rt').read())
        self.assertEqual(total['lut'], 1090775)
        self.assertEqual(total['bram'], 414)
        self.assertEqual(total['clb'], 152504)
        self.assertEqual(total['control_sets'], 14544)
        self.assertEqual(slr['SLR1']['control_sets'], 5966)
        self.assertEqual(total['lut'], sum(s['lut'] for s in slr.values()))

    def test_local_sll_pressure_survives_aggregate_headroom(self):
        result = R.placement_diagnosis(SLR.read_text(), gzip.open(LOG, 'rt').read())
        self.assertEqual(result['dense_slrs'], ['SLR0', 'SLR1'])
        self.assertEqual(result['design_state'], 'Fully Placed')
        self.assertFalse(result['global_total_available'])
        self.assertNotIn('control_sets', result['summed_reported_slrs'])
        cut = result['estimated_sll_by_column']['SLR1<->SLR2']
        self.assertLess(cut['aggregate_demand_ratio'], 1)
        self.assertGreater(cut['peak_demand_ratio'], 2)
        self.assertEqual(cut['overloaded_columns'], [12, 13, 14, 15])
        self.assertTrue(result['congestion_warning_present'])
        self.assertIsNone(result['predicted_speedup'])
        self.assertEqual(result['gate_status'], 'not-measured')
        partial = R.placement_diagnosis(SLR.read_text(), gzip.open(LOG, 'rt').read()
                                        + '\nNumber SLLs per Column: 1440\n')
        self.assertEqual(partial['estimated_sll_by_column'], {})

    def test_density_threshold_is_configurable_and_not_a_gate(self):
        result = R.placement_diagnosis(SLR.read_text(), pressure_threshold=.99)
        self.assertEqual(result['dense_slrs'], [])
        self.assertEqual(result['gate_status'], 'not-measured')
        self.assertEqual(result['estimated_sll_by_column'], {})
        for value in (0, 1.1, float('nan'), True):
            with self.assertRaises(ValueError):
                R.placement_diagnosis(SLR.read_text(), pressure_threshold=value)

    def test_column_order_and_missing_percentage_are_explicit(self):
        text = '''| Site Type | SLR2 % | SLR0 | SLR2 | SLR0 % |
| CLB | 95.00 | 20 | 30 | 40.00 |
| CLB LUTs | 60.00 | 50 | 70 | 10.00 |
'''
        result = R.placement_diagnosis(text)
        self.assertEqual(result['slr']['SLR0']['clb'], 20)
        self.assertEqual(result['dense_slrs'], ['SLR2'])
        result = R.placement_diagnosis(text.replace('95.00', '<95.00'))
        self.assertEqual(result['dense_slrs'], [])
        self.assertEqual(result['reported_slr_percentages']['SLR2']['CLB'], '<95.00')
        with self.assertRaises(ValueError):
            R.placement_diagnosis(text.replace('95.00', 'NaN'))

    def test_corrupt_or_incomplete_sll_capacity_refused(self):
        text = gzip.open(LOG, 'rt').read()
        for mutant in (text.replace('2940 (204%)', '2941 (204%)'),
                       text.replace('Number SLLs per Column: 1440', 'Number SLLs per Column: 1441'),
                       text.replace('Number SLLs per Column: 1440', 'column capacity omitted')):
            self.assertNotEqual(mutant, text)
            with self.assertRaises(ValueError):
                R.placement_diagnosis(SLR.read_text(), mutant)

    def test_conflicting_totals_refused(self):
        text = '''| Site Type | Used | Available |
| CLB LUTs | 100 | 200 |

| Site Type | Used | Available |
| CLB LUTs | 101 | 200 |
'''
        with self.assertRaises(ValueError):
            R.parse_util(text)

    def test_build_identity_and_raw_inputs_are_bound(self):
        result = R.placement_bundle(SLR, LOG, MANIFEST)
        self.assertEqual(result['build_identity']['hardware_source'], '09ebe06')
        self.assertIsNone(result['measured_clock_hz'])
        self.assertIn(str(LOG), result['input_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            for name, value in [('contract_sha256', '0'*64), ('clock_target_hz', True),
                                ('hardware_source', ''), ('implementation_root', '')]:
                identity = copy.deepcopy(result['build_identity'])
                identity[name] = value
                path.write_text(json.dumps(identity))
                with self.assertRaises(ValueError):
                    R.placement_bundle(SLR, LOG, path)
            synthesized = Path(directory)/'synthesized.rpt'
            synthesized.write_text(SLR.read_text().replace('Fully Placed', 'Synthesized'))
            with self.assertRaises(ValueError):
                R.placement_bundle(synthesized, LOG, MANIFEST)

    def test_cli_diagnostic_cannot_claim_image_timing_or_overwrite(self):
        command = [sys.executable, str(Path(R.__file__)), '--diagnose-placement',
                   '--util', str(SLR), '--build-manifest', str(MANIFEST)]
        rejected = subprocess.run(command+['--image', 'unqualified'], capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            completed = subprocess.run(command+['--out', str(path)], capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            original = path.read_bytes()
            repeated = subprocess.run(command+['--out', str(path)], capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
