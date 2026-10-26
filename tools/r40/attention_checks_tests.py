#!/usr/bin/env python3
"""Audit frozen native records and reject incomplete or staged inventories."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from attention_checks import ROOT, commands, inventory, validate_inventory


class AttentionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'inventory.jsonl'
        ids = {p['id'] for p in inventory()}
        with gzip.open(ROOT / 'evidence/r40/R40-01/common_clock_board77/results.jsonl.gz', 'rt') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        self.records = [r for r in records if r['point_id'] in ids]

    def write(self, records):
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in records))

    def test_real_frozen_inventory_and_required_commands(self):
        self.write(self.records)
        validate_inventory(self.path)
        jobs = commands(Path(self.temp.name), 'core-vrf-sync-v5')
        self.assertEqual(set(jobs), {'inventory', 'online', 'decode'})
        self.assertEqual(jobs['decode'][-3:], ['--steps', '64', '--bucket'])
        self.assertEqual({r['family'] for r in self.records},
                         {'attention_decode', 'attention_prefill', 'llama_decode', 'llama_prefill', 'gpt2_block'})

    def test_missing_duplicate_or_unexpected_point_refused(self):
        for records in (self.records[:-1], self.records + [self.records[0]],
                        self.records[:-1] + [self.records[0]]):
            self.write(records)
            with self.assertRaises(ValueError):validate_inventory(self.path)

    def test_modified_definition_staging_fallback_and_disabled_descriptors_refused(self):
        for field in ('hash', 'staging', 'fallback', 'descriptor', 'device'):
            records = copy.deepcopy(self.records)
            r = records[0]
            if field == 'hash':r['point_sha256'] = 'changed'
            elif field == 'staging':r['host']['upload_bytes'] = 8
            elif field == 'fallback':r['fallbacks'] = ['cpu']
            elif field == 'descriptor':r['platform']['inductor']['tensor_descriptors'] = False
            else:r['platform']['knobs']['TNCA_TORCH_DEVICE'] = 'cpu'
            self.write(records)
            with self.assertRaises(ValueError):validate_inventory(self.path)


if __name__ == '__main__':
    unittest.main()
