"""The frequency audit must reject partial populations and wrong regions."""
import copy
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from frequency_preflight import ROOT, parse_tsv, group_rows, check_coverage, parse_slr, validate_slr


class FrequencyPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(ROOT/'evidence/r40/v3/final_image_candidate/census/census.tgz', 'r:gz') as archive:
            raw = archive.extractfile('residual_failing_endpoints.tsv').read()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'residual.tsv'
            path.write_bytes(raw)
            cls.rows = parse_tsv(path)
        cls.families, cls.units = group_rows(cls.rows)
        cls.frozen = json.loads((ROOT/'evidence/r40/v3/final_image_candidate/census/frontier.json').read_text())

    def test_complete_authoritative_population(self):
        check_coverage(self.families, self.units, self.rows, self.frozen)

    def test_missing_endpoint_or_family_is_not_complete(self):
        with self.assertRaises(ValueError):
            check_coverage(self.families, self.units, self.rows[:-1], self.frozen)
        with self.assertRaises(ValueError):
            check_coverage(self.families[:-1], self.units, self.rows, self.frozen)

    def test_changed_slack_population_is_refused(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['slack'] = 0
        with self.assertRaises(ValueError):
            check_coverage(self.families, self.units, rows, self.frozen)

    def test_regions_match_the_real_a3_report(self):
        report = Path(os.environ['R41_FREQUENCY_SLR_REPORT']).read_text()
        regions = parse_slr(report)
        reference = json.loads((ROOT/'evidence/r40/v3/final_implementation_plan/plan.json').read_text())['baseline']['utilization_routed']
        validate_slr(regions, reference)
        regions['CLB']['SLR0']['percent'] -= 1
        with self.assertRaises(ValueError):
            validate_slr(regions, reference)
        with self.assertRaises(ValueError):
            parse_slr(report.replace('Design State : Routed', 'Design State : Synthesized'))


if __name__ == '__main__':
    unittest.main()
