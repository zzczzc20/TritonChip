#!/usr/bin/env python3
"""Reject partial or falsely CUDA-labelled conformance using an actual full log."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import device_conformance_acceptance as audit
ROOT=Path(__file__).resolve().parents[2]


class ConformanceTests(unittest.TestCase):
    def setUp(self):
        self.raw=json.loads((ROOT/'evidence/r40/v2/convergence/attempt03/cuda_conformance_997bd0b_complete.json').read_text())

    def test_actual_complete_cuda_coexistence_log(self):
        checks=audit.complete_cases(self.raw['log'],'cuda-host')
        self.assertEqual(len(checks),16)
        self.assertTrue(all(c['ok'] for c in checks))
        with self.assertRaises(ValueError):audit.complete_cases(self.raw['log'],'no-cuda-host')

    def test_omission_duplicate_failure_and_skipped_cuda_cannot_pass(self):
        lines=self.raw['log'].splitlines()
        mutations=[ '\n'.join(l for l in lines if 'PASS  copies ' not in l),
                    self.raw['log']+'\n  PASS copies duplicate\n',
                    self.raw['log'].replace('PASS  copies','FAIL  copies'),
                    self.raw['log'].replace('2 rounds of CUDA and tnca graphs in one process, then a CPU-tensor graph','skipped: no CUDA device on this host')]
        for log in mutations:
            with self.assertRaises(ValueError):audit.complete_cases(log,'cuda-host')


if __name__=='__main__':unittest.main()
