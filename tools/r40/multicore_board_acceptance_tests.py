#!/usr/bin/env python3
"""Mutate actual physical checkpoint evidence; incomplete claims must fail."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('physical_audit',ROOT/'tools/r40/multicore_board_acceptance.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class PhysicalEvidenceTests(unittest.TestCase):
    def setUp(self):
        raw=json.loads((ROOT/'evidence/r40/v2/convergence/attempt03/cold_inventory_3c403a4.json').read_text())
        self.checkpoint=json.loads(raw['checkpoint/report.json'])
        identity=self.checkpoint['identity']
        self.arguments=dict(tree=identity['software_tree'],image=identity['image_sha256'],clock=dict(
            clock_hz=identity['clock_hz'],operating_clock_evidence_sha256=identity['operating_clock_evidence_sha256']))

    def test_actual_checkpoint_accepts_only_its_measured_identity(self):
        self.assertTrue(all(c['ok'] for c in audit.checkpoint_checks(self.checkpoint,**self.arguments)))
        for field in ('tree','image'):
            wrong=dict(self.arguments);wrong[field]='different'
            with self.assertRaises(ValueError):audit.checkpoint_checks(self.checkpoint,**wrong)

    def test_missing_physical_traffic_or_visibility_is_refused(self):
        changes={
            'duplicate core':lambda d:d['cores']['cores'][2].update(core=1),
            'missing outlet':lambda d:d['outlets'].pop(),
            'no outlet writes':lambda d:d['outlets'][0].update(write_lines=0),
            'no drain':lambda d:d['events'][0].update(history=[['SUCCEEDED',0]]),
            'missing ownership':lambda d:d.update(host_visibility_history=[]),
            'failed ownership':lambda d:[r.update(status=1) for r in d['host_visibility_history'] if r['op']==18],
            'shrunk input':lambda d:d.update(input_bytes=64<<20),
            'wrong numerical result':lambda d:d['correctness'].update(ok=False),
        }
        for name,change in changes.items():
            with self.subTest(name=name):
                bad=copy.deepcopy(self.checkpoint);change(bad)
                with self.assertRaises(ValueError):audit.checkpoint_checks(bad,**self.arguments)


if __name__=='__main__':unittest.main()
