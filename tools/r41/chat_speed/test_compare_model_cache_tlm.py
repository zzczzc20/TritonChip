#!/usr/bin/env python3
"""The model TLM comparison must reject missing evidence and changed bits."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file
import compare_model_cache_tlm as gate


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.control, self.candidate = (Path(self.tmp.name) / name for name in ('control', 'candidate'))
        self.report = dict(status='pass', model='test', generated_tokens=[1, 2],
                           cpu_numerical_fallback=False, compiled_wrapper_roles_covered=True,
                           steady_decode_host_reads_detected=False,
                           device_target={'TNCA_BACKEND': 'systemc-tlm-grid',
                                          'TNCA_PROFILE': 'core-vrf-sync-v5-x8-g2c4-dp'},
                           model_tensor_placements={'parameters': {'tnca:0': 1}, 'buffers': {'tnca:0': 1}},
                           steps=[dict(kind=kind, transfers_inside_step={'host_read_calls': 0,
                                                                      'staging_download_bytes': 0})
                                  for kind in ('prefill', 'decode')])
        self.rows = [dict(scope_kind=kind, kernel='w8a8_block_dynamic_fp8_matmul_kernel', grid=[1, 1, 1],
                          artifact='kernel.tnca', device_cycles=1, code_load_count=0,
                          code_upload_bytes=0, evicted=0, counters={'core_x_instructions': 8},
                          args=[{'role': 'act', 'dtype': 'bfloat16', 'shape': [1, 4]},
                                {'role': 'weight', 'dtype': 'float8_e4m3fn', 'shape': [4, 4]}])
                     for kind in ('prefill', 'decode')]
        self.logits = {'step_0000': torch.tensor([1, 2], dtype=torch.bfloat16),
                       'step_0001': torch.tensor([3, 4], dtype=torch.bfloat16)}
        self.write(self.control)
        self.write(self.candidate)
        self.artifact = mock.patch('tnca.artifact.load', return_value=({}, {'text': b'kernel'}))
        self.artifact.start()
        self.addCleanup(self.artifact.stop)

    def write(self, directory, report=None, logits=None):
        (directory / 'profile').mkdir(parents=True, exist_ok=True)
        (directory / 'report.json').write_text(json.dumps(report if report is not None else self.report))
        (directory / 'profile/launches.json').write_text(json.dumps({'launches': self.rows}))
        save_file(logits if logits is not None else self.logits, str(directory / 'logits.safetensors'))

    def test_identical_complete_runs_pass(self):
        self.assertEqual(gate.compare(self.control, self.candidate)['status'], 'pass')

    def test_changed_bits_fail(self):
        changed = {key: value.clone() for key, value in self.logits.items()}
        bits = changed['step_0000'].view(torch.int16)
        bits[0] = int(bits[0]) ^ 1
        self.write(self.candidate, logits=changed)
        result = gate.compare(self.control, self.candidate)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['logits']['step_0000']['bit_mismatches'], 1)

    def test_missing_logits_fail(self):
        self.write(self.candidate, logits={'step_0000': self.logits['step_0000']})
        self.assertEqual(gate.compare(self.control, self.candidate)['status'], 'fail')

    def test_decode_read_counter_cannot_hide_behind_summary_flag(self):
        report = json.loads(json.dumps(self.report))
        report['steps'][1]['transfers_inside_step']['host_read_calls'] = 1
        self.write(self.candidate, report=report)
        result = gate.compare(self.control, self.candidate)
        self.assertIn('candidate: decode transfer counters are missing or nonzero', result['problems'])


if __name__ == '__main__':
    unittest.main()
